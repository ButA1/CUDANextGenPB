/*
 *  Barnes-Hut tree-code for NGPB electrostatic energy kernels.
 *  See include/barnes_hut.h for the design overview.
 *
 *  Pipeline (all on device, FP64):
 *    1. bounding box of atoms                (custom reduction w/ double atomics)
 *    2. 63-bit Morton codes + radix sort     (cub::DeviceRadixSort)
 *    3. LBVH binary radix tree build         (Karras 2012)
 *    4. bottom-up multipole summarize (M,D,T) (atomic-flag upward walk)
 *    5. per-target traversal w/ MAC          (potential or field evaluator)
 */

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <vector>
#include <cstdio>
#include <cstdint>
#include <cfloat>
#include <algorithm>

static void check_cuda(cudaError_t err, const char *msg, int line) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error at line %d (%s): %s\n",
            line, msg, cudaGetErrorString(err));
  }
}
#define CUDA_CHECK(call) check_cuda((call), #call, __LINE__)

// ====================================================================
//  Tree handle (device-resident). Combined node array of size 2N-1:
//    index 0 .. N-2     : internal nodes
//    index N-1 .. 2N-2  : leaf nodes (leaf k at index (N-1)+k -> sorted atom k)
//  "is leaf" test: idx >= N-1  (also correct for N==1, where everything is a leaf)
// ====================================================================
struct bh_tree {
  int     N;          // number of atoms (leaves)
  int     n_nodes;    // 2N-1
  double  theta;

  // sorted source atoms (Morton order)
  double *d_pos;      // 3N
  double *d_q;        // N

  // node geometry (AABB) and topology
  double *d_min;      // 3 * n_nodes
  double *d_max;      // 3 * n_nodes
  int    *d_left;     // n_nodes (valid for internal nodes only)
  int    *d_right;    // n_nodes
  int    *d_parent;   // n_nodes

  // node multipole moments (about AABB center)
  double *d_M;        // n_nodes
  double *d_D;        // 3 * n_nodes
  double *d_T;        // 6 * n_nodes : xx,yy,zz,xy,xz,yz
};

// ====================================================================
//  double atomic min/max (compute capability has no native FP64 min/max atomic)
// ====================================================================
__device__ __forceinline__ double atomicMinDouble(double *addr, double val) {
  unsigned long long *a = (unsigned long long *)addr;
  unsigned long long old = *a, assumed;
  do {
    assumed = old;
    if (__longlong_as_double(assumed) <= val) break;
    old = atomicCAS(a, assumed, __double_as_longlong(val));
  } while (assumed != old);
  return __longlong_as_double(old);
}
__device__ __forceinline__ double atomicMaxDouble(double *addr, double val) {
  unsigned long long *a = (unsigned long long *)addr;
  unsigned long long old = *a, assumed;
  do {
    assumed = old;
    if (__longlong_as_double(assumed) >= val) break;
    old = atomicCAS(a, assumed, __double_as_longlong(val));
  } while (assumed != old);
  return __longlong_as_double(old);
}

// ====================================================================
//  Kernel 1: bounding box of atoms (per-block shared reduction -> global atomics)
// ====================================================================
__global__ void bbox_kernel(int N, const double *__restrict__ pos,
                            double *__restrict__ gmin, double *__restrict__ gmax) {
  __shared__ double smin[3], smax[3];
  if (threadIdx.x < 3) { smin[threadIdx.x] =  DBL_MAX; smax[threadIdx.x] = -DBL_MAX; }
  __syncthreads();

  double lmin[3] = { DBL_MAX,  DBL_MAX,  DBL_MAX};
  double lmax[3] = {-DBL_MAX, -DBL_MAX, -DBL_MAX};
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += gridDim.x * blockDim.x) {
    #pragma unroll
    for (int d = 0; d < 3; ++d) {
      double v = pos[3*i + d];
      lmin[d] = fmin(lmin[d], v);
      lmax[d] = fmax(lmax[d], v);
    }
  }
  #pragma unroll
  for (int d = 0; d < 3; ++d) {
    atomicMinDouble(&smin[d], lmin[d]);
    atomicMaxDouble(&smax[d], lmax[d]);
  }
  __syncthreads();
  if (threadIdx.x < 3) {
    atomicMinDouble(&gmin[threadIdx.x], smin[threadIdx.x]);
    atomicMaxDouble(&gmax[threadIdx.x], smax[threadIdx.x]);
  }
}

// ====================================================================
//  Kernel 2: Morton codes (21 bits / axis -> 63-bit key)
// ====================================================================
__device__ __forceinline__ uint64_t expandBits21(uint64_t v) {
  v &= 0x1fffffULL;
  v = (v | v << 32) & 0x1f00000000ffffULL;
  v = (v | v << 16) & 0x1f0000ff0000ffULL;
  v = (v | v << 8)  & 0x100f00f00f00f00fULL;
  v = (v | v << 4)  & 0x10c30c30c30c30c3ULL;
  v = (v | v << 2)  & 0x1249249249249249ULL;
  return v;
}

__global__ void morton_kernel(int N, const double *__restrict__ pos,
                              double bx0, double by0, double bz0,
                              double inv_sx, double inv_sy, double inv_sz,
                              uint64_t *__restrict__ codes, int *__restrict__ idx) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  const double scale = 2097151.0; // 2^21 - 1
  double nx = (pos[3*i]   - bx0) * inv_sx;
  double ny = (pos[3*i+1] - by0) * inv_sy;
  double nz = (pos[3*i+2] - bz0) * inv_sz;
  uint64_t xx = expandBits21((uint64_t)fmin(fmax(nx * scale, 0.0), scale));
  uint64_t yy = expandBits21((uint64_t)fmin(fmax(ny * scale, 0.0), scale));
  uint64_t zz = expandBits21((uint64_t)fmin(fmax(nz * scale, 0.0), scale));
  codes[i] = xx | (yy << 1) | (zz << 2);
  idx[i]   = i;
}

// reorder atom positions/charges into Morton order, and init leaf nodes
__global__ void reorder_kernel(int N, const int *__restrict__ order,
                               const double *__restrict__ pos_in,
                               const double *__restrict__ q_in,
                               double *__restrict__ pos_out,
                               double *__restrict__ q_out,
                               double *__restrict__ nmin, double *__restrict__ nmax,
                               double *__restrict__ nM, double *__restrict__ nD,
                               double *__restrict__ nT, int *__restrict__ parent,
                               int n_nodes) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= N) return;
  int src = order[k];
  double x = pos_in[3*src], y = pos_in[3*src+1], z = pos_in[3*src+2];
  double q = q_in[src];
  pos_out[3*k] = x; pos_out[3*k+1] = y; pos_out[3*k+2] = z;
  q_out[k] = q;

  int leaf = (N - 1) + k;               // combined-array index of this leaf
  nmin[3*leaf] = x; nmin[3*leaf+1] = y; nmin[3*leaf+2] = z;
  nmax[3*leaf] = x; nmax[3*leaf+1] = y; nmax[3*leaf+2] = z;
  nM[leaf] = q;                          // leaf center == atom pos => D=T=0
  nD[3*leaf] = nD[3*leaf+1] = nD[3*leaf+2] = 0.0;
  for (int t = 0; t < 6; ++t) nT[6*leaf + t] = 0.0;

  if (k == 0) parent[0] = -1;            // root has no parent (N>1 case)
}

// ====================================================================
//  Kernel 3: LBVH binary radix tree build (Karras 2012)
// ====================================================================
// common-prefix length of codes[i], codes[j] with index tie-break for duplicates
__device__ __forceinline__ int delta(int i, int j, const uint64_t *codes, int n) {
  if (j < 0 || j >= n) return -1;
  uint64_t ci = codes[i], cj = codes[j];
  if (ci == cj) return 64 + __clzll((uint64_t)(i ^ j));
  return __clzll(ci ^ cj);
}

__global__ void build_internal_kernel(int n, const uint64_t *__restrict__ codes,
                                      int *__restrict__ left, int *__restrict__ right,
                                      int *__restrict__ parent) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n - 1) return;               // n-1 internal nodes

  // direction of the range owned by internal node i
  int dl = delta(i, i + 1, codes, n);
  int dr = delta(i, i - 1, codes, n);
  int d  = (dl - dr) >= 0 ? 1 : -1;

  // lower bound on prefix length for this range
  int dmin = delta(i, i - d, codes, n);

  // exponentially grow an upper bound on the range length
  int lmax = 2;
  while (delta(i, i + lmax * d, codes, n) > dmin) lmax <<= 1;

  // binary search for the exact range length
  int l = 0;
  for (int t = lmax >> 1; t >= 1; t >>= 1) {
    if (delta(i, i + (l + t) * d, codes, n) > dmin) l += t;
  }
  int j = i + l * d;                     // other end of the range
  int first = min(i, j), last = max(i, j);

  // find the split position within [first, last]
  int dnode = delta(first, last, codes, n);
  int split = first, step = last - first;
  do {
    step = (step + 1) >> 1;
    int ns = split + step;
    if (ns < last && delta(first, ns, codes, n) > dnode) split = ns;
  } while (step > 1);

  int lc = (split     == first) ? (n - 1) + split       : split;
  int rc = (split + 1 == last)  ? (n - 1) + (split + 1)  : split + 1;
  left[i]  = lc;
  right[i] = rc;
  parent[lc] = i;
  parent[rc] = i;
}

// ====================================================================
//  Kernel 4: bottom-up multipole summarize (M, D, T about AABB center)
// ====================================================================
__device__ __forceinline__ void m2m_accumulate(
    double Mc, const double *Dc, const double *Tc,
    double dx, double dy, double dz,          // child_center - parent_center
    double &M, double *D, double *T) {
  M += Mc;
  D[0] += Dc[0] + Mc * dx;
  D[1] += Dc[1] + Mc * dy;
  D[2] += Dc[2] + Mc * dz;
  T[0] += Tc[0] + 2.0 * Dc[0] * dx + Mc * dx * dx;          // xx
  T[1] += Tc[1] + 2.0 * Dc[1] * dy + Mc * dy * dy;          // yy
  T[2] += Tc[2] + 2.0 * Dc[2] * dz + Mc * dz * dz;          // zz
  T[3] += Tc[3] + Dc[0] * dy + Dc[1] * dx + Mc * dx * dy;   // xy
  T[4] += Tc[4] + Dc[0] * dz + Dc[2] * dx + Mc * dx * dz;   // xz
  T[5] += Tc[5] + Dc[1] * dz + Dc[2] * dy + Mc * dy * dz;   // yz
}

__global__ void summarize_kernel(int N,
                                 const int *__restrict__ left,
                                 const int *__restrict__ right,
                                 const int *__restrict__ parent,
                                 double *__restrict__ nmin, double *__restrict__ nmax,
                                 double *__restrict__ nM, double *__restrict__ nD,
                                 double *__restrict__ nT, int *__restrict__ flags) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= N) return;

  int node = parent[(N - 1) + k];        // start at this leaf's parent
  while (node != -1) {
    __threadfence();                     // publish child writes before claiming parent
    if (atomicAdd(&flags[node], 1) == 0)
      return;                            // first child to arrive: sibling will finish node

    // second child has arrived: both children's data are ready
    int lc = left[node], rc = right[node];

    // merge AABB
    double mnx = fmin(nmin[3*lc],   nmin[3*rc]);
    double mny = fmin(nmin[3*lc+1], nmin[3*rc+1]);
    double mnz = fmin(nmin[3*lc+2], nmin[3*rc+2]);
    double mxx = fmax(nmax[3*lc],   nmax[3*rc]);
    double mxy = fmax(nmax[3*lc+1], nmax[3*rc+1]);
    double mxz = fmax(nmax[3*lc+2], nmax[3*rc+2]);
    nmin[3*node] = mnx; nmin[3*node+1] = mny; nmin[3*node+2] = mnz;
    nmax[3*node] = mxx; nmax[3*node+1] = mxy; nmax[3*node+2] = mxz;

    double cx = 0.5 * (mnx + mxx), cy = 0.5 * (mny + mxy), cz = 0.5 * (mnz + mxz);

    double M = 0.0, D[3] = {0,0,0}, T[6] = {0,0,0,0,0,0};
    #pragma unroll
    for (int s = 0; s < 2; ++s) {
      int c = (s == 0) ? lc : rc;
      double ccx = 0.5 * (nmin[3*c]   + nmax[3*c]);
      double ccy = 0.5 * (nmin[3*c+1] + nmax[3*c+1]);
      double ccz = 0.5 * (nmin[3*c+2] + nmax[3*c+2]);
      m2m_accumulate(nM[c], &nD[3*c], &nT[6*c],
                     ccx - cx, ccy - cy, ccz - cz, M, D, T);
    }
    nM[node] = M;
    nD[3*node] = D[0]; nD[3*node+1] = D[1]; nD[3*node+2] = D[2];
    for (int t = 0; t < 6; ++t) nT[6*node + t] = T[t];

    node = parent[node];
  }
}

// ====================================================================
//  Kernel 5: per-target traversal with MAC. Templated on field vs potential.
// ====================================================================
// Accumulate the multipole potential of a node at offset R (= target - center).
__device__ __forceinline__ double mp_potential(
    double Rx, double Ry, double Rz, double dist2,
    double M, const double *D, const double *T) {
  double inv_r  = rsqrt(dist2);
  double inv_r2 = inv_r * inv_r;
  double inv_r3 = inv_r * inv_r2;
  double inv_r5 = inv_r3 * inv_r2;
  double DdotR  = D[0]*Rx + D[1]*Ry + D[2]*Rz;
  double RTR = Rx*Rx*T[0] + Ry*Ry*T[1] + Rz*Rz*T[2]
             + 2.0 * (Rx*Ry*T[3] + Rx*Rz*T[4] + Ry*Rz*T[5]);
  double trT = T[0] + T[1] + T[2];
  return M*inv_r + DdotR*inv_r3 + 0.5 * (3.0*RTR - dist2*trT) * inv_r5;
}

// Accumulate the multipole field g = -grad(phi) of a node at offset R.
__device__ __forceinline__ void mp_field(
    double Rx, double Ry, double Rz, double dist2,
    double M, const double *D, const double *T,
    double &gx, double &gy, double &gz) {
  double inv_r  = rsqrt(dist2);
  double inv_r2 = inv_r * inv_r;
  double inv_r3 = inv_r * inv_r2;
  double inv_r5 = inv_r3 * inv_r2;
  double inv_r7 = inv_r5 * inv_r2;

  double DdotR = D[0]*Rx + D[1]*Ry + D[2]*Rz;
  // T*R (symmetric matrix-vector)
  double TRx = T[0]*Rx + T[3]*Ry + T[4]*Rz;
  double TRy = T[3]*Rx + T[1]*Ry + T[5]*Rz;
  double TRz = T[4]*Rx + T[5]*Ry + T[2]*Rz;
  double A   = Rx*TRx + Ry*TRy + Rz*TRz;     // R.T.R
  double trT = T[0] + T[1] + T[2];

  // monopole + dipole-gradient + quadrupole-gradient
  double cR = M*inv_r3                       // monopole coeff on R
            + 3.0*DdotR*inv_r5               // dipole
            + 7.5*A*inv_r7                   // quadrupole (15/2)
            - 1.5*trT*inv_r5;                // quadrupole trace (3/2)
  gx += cR*Rx - D[0]*inv_r3 - 3.0*TRx*inv_r5;
  gy += cR*Ry - D[1]*inv_r3 - 3.0*TRy*inv_r5;
  gz += cR*Rz - D[2]*inv_r3 - 3.0*TRz*inv_r5;
}

template<bool FIELD>
__device__ void traverse(double tx, double ty, double tz, int self_atom,
                         const bh_tree t,
                         double &out_phi, double &gx, double &gy, double &gz) {
  out_phi = 0.0; gx = gy = gz = 0.0;
  const double theta2 = t.theta * t.theta;
  const int leaf0 = t.N - 1;

  // A binary radix tree over 63-bit Morton codes can approach depth 63; the
  // DFS stack grows by +1 per descended level, so 96 leaves comfortable margin
  // (incl. the duplicate-code index tie-break that can extend prefixes).
  int stack[96];
  int sp = 0;
  stack[sp++] = 0;                         // root (leaf if N==1, internal otherwise)

  while (sp > 0) {
    int node = stack[--sp];
    double cx = 0.5 * (t.d_min[3*node]   + t.d_max[3*node]);
    double cy = 0.5 * (t.d_min[3*node+1] + t.d_max[3*node+1]);
    double cz = 0.5 * (t.d_min[3*node+2] + t.d_max[3*node+2]);
    double Rx = tx - cx, Ry = ty - cy, Rz = tz - cz;
    double dist2 = Rx*Rx + Ry*Ry + Rz*Rz;

    if (node >= leaf0) {                    // leaf: single source atom
      int a = node - leaf0;
      if (a == self_atom) continue;
      if (dist2 <= 0.0) continue;
      double q = t.d_q[a];
      double inv_r = rsqrt(dist2);
      if (FIELD) {
        double inv_r3 = inv_r * inv_r * inv_r;
        gx += q * Rx * inv_r3;
        gy += q * Ry * inv_r3;
        gz += q * Rz * inv_r3;
      } else {
        out_phi += q * inv_r;
      }
      continue;
    }

    // MAC: accept cell if (size/dist) < theta  <=>  size^2 < theta^2 * dist^2
    double ex = t.d_max[3*node]   - t.d_min[3*node];
    double ey = t.d_max[3*node+1] - t.d_min[3*node+1];
    double ez = t.d_max[3*node+2] - t.d_min[3*node+2];
    double size = fmax(ex, fmax(ey, ez));
    if (dist2 > 0.0 && size*size < theta2 * dist2) {
      if (FIELD)
        mp_field(Rx, Ry, Rz, dist2, t.d_M[node], &t.d_D[3*node], &t.d_T[6*node],
                 gx, gy, gz);
      else
        out_phi += mp_potential(Rx, Ry, Rz, dist2, t.d_M[node],
                                &t.d_D[3*node], &t.d_T[6*node]);
    } else {
      // open the node: visit both children
      stack[sp++] = t.d_left[node];
      stack[sp++] = t.d_right[node];
    }
  }
}

// E_coul: each source atom is also a target; exclude its own leaf, halve.
__global__ void coulomb_kernel(bh_tree t, double *__restrict__ partial) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= t.N) return;
  double phi, gx, gy, gz;
  traverse<false>(t.d_pos[3*i], t.d_pos[3*i+1], t.d_pos[3*i+2], i, t,
                  phi, gx, gy, gz);
  partial[i] = 0.5 * t.d_q[i] * phi;
}

// E_pol: potential at flux points, weighted by flux.
__global__ void potential_kernel(bh_tree t, int num_pts,
                                 const double *__restrict__ V,
                                 const double *__restrict__ flux,
                                 double *__restrict__ partial) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= num_pts) return;
  double phi, gx, gy, gz;
  traverse<false>(V[3*p], V[3*p+1], V[3*p+2], -1, t, phi, gx, gy, gz);
  partial[p] = flux[p] * phi;
}

// E_ion: field at triangle vertices dotted with normal, weighted by factor.
__global__ void field_kernel(bh_tree t, int num_tri_verts,
                             const double *__restrict__ vert,
                             const double *__restrict__ norms,
                             const double *__restrict__ phi_sup,
                             const double *__restrict__ area,
                             double inv_4pi, double *__restrict__ partial) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= num_tri_verts) return;
  double phi, gx, gy, gz;
  traverse<true>(vert[3*v], vert[3*v+1], vert[3*v+2], -1, t, phi, gx, gy, gz);
  double dot = gx*norms[3*v] + gy*norms[3*v+1] + gz*norms[3*v+2];
  double factor = phi_sup[v] * inv_4pi * area[v / 3] / 3.0;
  partial[v] = dot * factor;
}

// ====================================================================
//  host-side reduction (matches energy_cuda.cu summation order for A/B parity)
// ====================================================================
static double reduce_sum_host(const double *d_arr, int n) {
  std::vector<double> h(n);
  CUDA_CHECK(cudaMemcpy(h.data(), d_arr, n * sizeof(double), cudaMemcpyDeviceToHost));
  double s = 0.0;
  for (int i = 0; i < n; ++i) s += h[i];
  return s;
}

// ========================== public C API ==========================
extern "C" {

void bh_build_atom_tree(int num_atoms,
                        const double *d_atoms,
                        const double *d_charges,
                        double theta,
                        bh_tree **out) {
  bh_tree *t = new bh_tree();
  t->N = num_atoms;
  t->n_nodes = (num_atoms > 0) ? (2 * num_atoms - 1) : 0;
  t->theta = theta;
  *out = t;
  if (num_atoms <= 0) {
    t->d_pos = t->d_q = t->d_min = t->d_max = t->d_M = t->d_D = t->d_T = nullptr;
    t->d_left = t->d_right = t->d_parent = nullptr;
    return;
  }

  const int N = num_atoms, nn = t->n_nodes;
  CUDA_CHECK(cudaMalloc(&t->d_pos, 3 * N * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&t->d_q,       N * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&t->d_min, 3 * nn * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&t->d_max, 3 * nn * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&t->d_M,       nn * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&t->d_D,   3 * nn * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&t->d_T,   6 * nn * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&t->d_left,    nn * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&t->d_right,   nn * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&t->d_parent,  nn * sizeof(int)));

  const int tpb = 256;

  // 1. bounding box
  double *d_gmin, *d_gmax;
  CUDA_CHECK(cudaMalloc(&d_gmin, 3 * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_gmax, 3 * sizeof(double)));
  { double init_min[3] = { DBL_MAX,  DBL_MAX,  DBL_MAX};
    double init_max[3] = {-DBL_MAX, -DBL_MAX, -DBL_MAX};
    CUDA_CHECK(cudaMemcpy(d_gmin, init_min, 3*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gmax, init_max, 3*sizeof(double), cudaMemcpyHostToDevice)); }
  int bb_blocks = std::min(1024, (N + tpb - 1) / tpb);
  bbox_kernel<<<bb_blocks, tpb>>>(N, d_atoms, d_gmin, d_gmax);
  CUDA_CHECK(cudaGetLastError());

  double h_min[3], h_max[3];
  CUDA_CHECK(cudaMemcpy(h_min, d_gmin, 3*sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_max, d_gmax, 3*sizeof(double), cudaMemcpyDeviceToHost));
  cudaFree(d_gmin); cudaFree(d_gmax);

  // guard against degenerate (zero-extent) axes
  double sx = h_max[0] - h_min[0]; if (sx <= 0) sx = 1.0;
  double sy = h_max[1] - h_min[1]; if (sy <= 0) sy = 1.0;
  double sz = h_max[2] - h_min[2]; if (sz <= 0) sz = 1.0;

  // 2. Morton codes + radix sort
  uint64_t *d_codes, *d_codes_sorted;
  int *d_idx, *d_idx_sorted;
  CUDA_CHECK(cudaMalloc(&d_codes,        N * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&d_codes_sorted, N * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&d_idx,          N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_idx_sorted,   N * sizeof(int)));

  int mt_blocks = (N + tpb - 1) / tpb;
  morton_kernel<<<mt_blocks, tpb>>>(N, d_atoms, h_min[0], h_min[1], h_min[2],
                                    1.0/sx, 1.0/sy, 1.0/sz, d_codes, d_idx);
  CUDA_CHECK(cudaGetLastError());

  void *d_tmp = nullptr; size_t tmp_bytes = 0;
  cub::DeviceRadixSort::SortPairs(d_tmp, tmp_bytes, d_codes, d_codes_sorted,
                                  d_idx, d_idx_sorted, N);
  CUDA_CHECK(cudaMalloc(&d_tmp, tmp_bytes));
  cub::DeviceRadixSort::SortPairs(d_tmp, tmp_bytes, d_codes, d_codes_sorted,
                                  d_idx, d_idx_sorted, N);
  cudaFree(d_tmp);
  cudaFree(d_codes);
  cudaFree(d_idx);

  // reorder atoms into Morton order + initialize leaf nodes
  reorder_kernel<<<mt_blocks, tpb>>>(N, d_idx_sorted, d_atoms, d_charges,
                                     t->d_pos, t->d_q, t->d_min, t->d_max,
                                     t->d_M, t->d_D, t->d_T, t->d_parent, nn);
  CUDA_CHECK(cudaGetLastError());
  cudaFree(d_idx_sorted);

  if (N > 1) {
    // 3. build internal nodes
    int in_blocks = (N - 1 + tpb - 1) / tpb;
    build_internal_kernel<<<in_blocks, tpb>>>(N, d_codes_sorted,
                                              t->d_left, t->d_right, t->d_parent);
    CUDA_CHECK(cudaGetLastError());

    // 4. bottom-up summarize (zero the per-node arrival flags first)
    int *d_flags;
    CUDA_CHECK(cudaMalloc(&d_flags, nn * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_flags, 0, nn * sizeof(int)));
    summarize_kernel<<<mt_blocks, tpb>>>(N, t->d_left, t->d_right, t->d_parent,
                                         t->d_min, t->d_max, t->d_M, t->d_D, t->d_T,
                                         d_flags);
    CUDA_CHECK(cudaGetLastError());
    cudaFree(d_flags);
  }
  cudaFree(d_codes_sorted);
  CUDA_CHECK(cudaDeviceSynchronize());
}

void bh_free_tree(bh_tree *t) {
  if (!t) return;
  cudaFree(t->d_pos);  cudaFree(t->d_q);
  cudaFree(t->d_min);  cudaFree(t->d_max);
  cudaFree(t->d_M);    cudaFree(t->d_D);   cudaFree(t->d_T);
  cudaFree(t->d_left); cudaFree(t->d_right); cudaFree(t->d_parent);
  delete t;
}

double bh_coulombic_energy(bh_tree *t) {
  if (!t || t->N < 2) return 0.0;
  double *d_partial;
  CUDA_CHECK(cudaMalloc(&d_partial, t->N * sizeof(double)));
  int tpb = 256, blocks = (t->N + tpb - 1) / tpb;
  coulomb_kernel<<<blocks, tpb>>>(*t, d_partial);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  double r = reduce_sum_host(d_partial, t->N);
  cudaFree(d_partial);
  return r;
}

double bh_polarization_energy(bh_tree *t, int num_pts,
                              const double *h_V, const double *h_flux) {
  if (!t || t->N == 0 || num_pts == 0) return 0.0;
  double *d_V, *d_flux, *d_partial;
  CUDA_CHECK(cudaMalloc(&d_V,       num_pts * 3 * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_flux,    num_pts     * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_partial, num_pts     * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_V,    h_V,    num_pts*3*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_flux, h_flux, num_pts  *sizeof(double), cudaMemcpyHostToDevice));

  int tpb = 256, blocks = (num_pts + tpb - 1) / tpb;
  potential_kernel<<<blocks, tpb>>>(*t, num_pts, d_V, d_flux, d_partial);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  double r = reduce_sum_host(d_partial, num_pts);
  cudaFree(d_V); cudaFree(d_flux); cudaFree(d_partial);
  return r;
}

double bh_ionic_energy(bh_tree *t, int num_tri_verts,
                       const double *h_vert, const double *h_norms,
                       const double *h_phi_sup, const double *h_area,
                       double inv_4pi) {
  if (!t || t->N == 0 || num_tri_verts == 0) return 0.0;
  int num_tris = num_tri_verts / 3;
  double *d_vert, *d_norms, *d_phi, *d_area, *d_partial;
  CUDA_CHECK(cudaMalloc(&d_vert,    num_tri_verts * 3 * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_norms,   num_tri_verts * 3 * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_phi,     num_tri_verts     * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_area,    num_tris          * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_partial, num_tri_verts     * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_vert,  h_vert,    num_tri_verts*3*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_norms, h_norms,   num_tri_verts*3*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_phi,   h_phi_sup, num_tri_verts  *sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_area,  h_area,    num_tris       *sizeof(double), cudaMemcpyHostToDevice));

  int tpb = 256, blocks = (num_tri_verts + tpb - 1) / tpb;
  field_kernel<<<blocks, tpb>>>(*t, num_tri_verts, d_vert, d_norms, d_phi, d_area,
                                inv_4pi, d_partial);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  double r = reduce_sum_host(d_partial, num_tri_verts);
  cudaFree(d_vert); cudaFree(d_norms); cudaFree(d_phi); cudaFree(d_area); cudaFree(d_partial);
  return r;
}

} // extern "C"

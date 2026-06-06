/*
 *  Barnes-Hut Phase A: TABI-style high-order Cartesian Taylor treecode.
 *  See /home/leirex/.claude/plans/ok-enough-math-time-fancy-lighthouse.md
 *
 *  Differences from Phase 0:
 *    - Per-node moments are stored as a flat array m^k indexed by multi-index
 *      k=(kx,ky,kz) with |k| <= p. Count: COMP(p) = (p+1)(p+2)(p+3)/6.
 *    - The multipole evaluator uses Cartesian Taylor coefficients
 *          T^k(R) = d^k(1/r)
 *      computed via the recurrence (Li-Johnston-Krasny 2009, kappa=0 case):
 *          r^2 * T^(k+e_i) = -(2n+1) * R_i * T^k  -  n * T^(k-e_i),    n = |k|
 *    - M2M shift is the generic multinomial expansion:
 *          m_parent^k += sum_{l <= k} C(k,l) * delta^(k-l) * m_child^l.
 *    - Field evaluator (E_ion) reuses T^k through order p+1.
 *
 *  LBVH topology (one leaf per atom) and the bbox/Morton/build kernels are unchanged.
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
//  Compile-time constants for the multi-index machinery.
//  BH_MAX_P is the largest moment order supported; the field evaluator
//  needs Taylor coefficients up to order p+1, hence BH_MAX_P_FIELD.
// ====================================================================
#define BH_MAX_P        6
#define BH_MAX_P_FIELD  7
__host__ __device__ constexpr int comp_of_h(int p) { return (p + 1) * (p + 2) * (p + 3) / 6; }
#define BH_COMP_MAX        comp_of_h(BH_MAX_P)         // 84
#define BH_COMP_MAX_FIELD  comp_of_h(BH_MAX_P_FIELD)   // 120

__host__ __device__ __forceinline__ int comp_of(int p) {
  return (p + 1) * (p + 2) * (p + 3) / 6;
}

// ====================================================================
//  Tree handle (device-resident). Combined node array of size 2N-1:
//    index 0 .. N-2     : internal nodes
//    index N-1 .. 2N-2  : leaf nodes  (leaf k at index (N-1)+k)
//  "is leaf" test: idx >= N-1 (also correct for N==1).
// ====================================================================
struct bh_tree {
  int     N;          // number of atoms (leaves)
  int     n_nodes;    // 2N - 1
  int     p;          // multipole order in [1, BH_MAX_P]
  int     comp;       // moments per node = COMP(p)
  double  theta;

  // sorted source atoms (Morton order)
  double *d_pos;      // 3N
  double *d_q;        // N

  // node geometry (AABB) and topology
  double *d_min;      // 3 * n_nodes
  double *d_max;      // 3 * n_nodes
  int    *d_left;     // n_nodes
  int    *d_right;    // n_nodes
  int    *d_parent;   // n_nodes

  // per-node multi-index moments m^k, k indexed in c_mi_kx/ky/kz order
  double *d_moments;  // n_nodes * comp
};

// ====================================================================
//  Constant-memory tables (populated once on first build).
//
//    c_mi_kx/ky/kz[s]   -> Cartesian components of the multi-index at slot s
//    c_mi_lookup[idx]   -> slot for the multi-index (kx,ky,kz),  idx = kx<<6 | ky<<3 | kz
//    c_inv_fact[s]      -> 1/(kx! ky! kz!)  for slot s  (used in the evaluators)
//    c_int_fact[i]      -> i!  for i in 0..7  (used in M2M binomial coefficients)
//    c_comp_at_order[n] -> # slots with |k| < n  (i.e. first slot index with |k|=n)
//
//  Slots are enumerated in ascending |k|, then by lex ordering within each shell.
// ====================================================================
__constant__ char   c_mi_kx[BH_COMP_MAX_FIELD];
__constant__ char   c_mi_ky[BH_COMP_MAX_FIELD];
__constant__ char   c_mi_kz[BH_COMP_MAX_FIELD];
__constant__ short  c_mi_lookup[8 * 8 * 8];
__constant__ double c_inv_fact[BH_COMP_MAX_FIELD];
__constant__ double c_int_fact[8];
__constant__ int    c_comp_at_order[BH_MAX_P_FIELD + 2];

#define MI_LOOKUP(kx, ky, kz) (c_mi_lookup[((kx) << 6) | ((ky) << 3) | (kz)])

// Host-side one-shot initializer for the constant tables.
static void init_constant_tables() {
  static bool initialized = false;
  if (initialized) return;

  char   h_mi_kx[BH_COMP_MAX_FIELD];
  char   h_mi_ky[BH_COMP_MAX_FIELD];
  char   h_mi_kz[BH_COMP_MAX_FIELD];
  short  h_mi_lookup[8 * 8 * 8];
  double h_inv_fact[BH_COMP_MAX_FIELD];
  double h_int_fact[8];
  int    h_comp_at_order[BH_MAX_P_FIELD + 2];

  h_int_fact[0] = 1.0;
  for (int i = 1; i < 8; ++i) h_int_fact[i] = h_int_fact[i - 1] * i;

  for (int i = 0; i < 8 * 8 * 8; ++i) h_mi_lookup[i] = -1;

  int slot = 0;
  h_comp_at_order[0] = 0;
  for (int n = 0; n <= BH_MAX_P_FIELD; ++n) {
    for (int kx = 0; kx <= n; ++kx) {
      for (int ky = 0; ky <= n - kx; ++ky) {
        int kz = n - kx - ky;
        h_mi_kx[slot] = (char)kx;
        h_mi_ky[slot] = (char)ky;
        h_mi_kz[slot] = (char)kz;
        h_mi_lookup[(kx << 6) | (ky << 3) | kz] = (short)slot;
        h_inv_fact[slot] = 1.0 / (h_int_fact[kx] * h_int_fact[ky] * h_int_fact[kz]);
        ++slot;
      }
    }
    h_comp_at_order[n + 1] = slot;
  }
  // (slot now equals BH_COMP_MAX_FIELD)

  CUDA_CHECK(cudaMemcpyToSymbol(c_mi_kx,         h_mi_kx,         sizeof(h_mi_kx)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_mi_ky,         h_mi_ky,         sizeof(h_mi_ky)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_mi_kz,         h_mi_kz,         sizeof(h_mi_kz)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_mi_lookup,     h_mi_lookup,     sizeof(h_mi_lookup)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_inv_fact,      h_inv_fact,      sizeof(h_inv_fact)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_int_fact,      h_int_fact,      sizeof(h_int_fact)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_comp_at_order, h_comp_at_order, sizeof(h_comp_at_order)));

  initialized = true;
}

// ====================================================================
//  double atomic min/max (no native FP64 min/max atomic on sm_86)
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
//  Kernel 1: bounding box
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
//  Kernel 2: Morton codes
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
  const double scale = 2097151.0;
  double nx = (pos[3*i]   - bx0) * inv_sx;
  double ny = (pos[3*i+1] - by0) * inv_sy;
  double nz = (pos[3*i+2] - bz0) * inv_sz;
  uint64_t xx = expandBits21((uint64_t)fmin(fmax(nx * scale, 0.0), scale));
  uint64_t yy = expandBits21((uint64_t)fmin(fmax(ny * scale, 0.0), scale));
  uint64_t zz = expandBits21((uint64_t)fmin(fmax(nz * scale, 0.0), scale));
  codes[i] = xx | (yy << 1) | (zz << 2);
  idx[i]   = i;
}

// ====================================================================
//  Kernel 3: reorder atoms into Morton order + per-leaf moment init
// ====================================================================
// Leaf expansion center is the atom position itself, so m^0 = q and m^k = 0 for |k|>=1.
__global__ void reorder_kernel(int N, int comp,
                               const int *__restrict__ order,
                               const double *__restrict__ pos_in,
                               const double *__restrict__ q_in,
                               double *__restrict__ pos_out,
                               double *__restrict__ q_out,
                               double *__restrict__ nmin,
                               double *__restrict__ nmax,
                               double *__restrict__ moments,
                               int *__restrict__ parent) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  int src = order[i];
  double x = pos_in[3*src], y = pos_in[3*src+1], z = pos_in[3*src+2];
  double q = q_in[src];
  pos_out[3*i] = x; pos_out[3*i+1] = y; pos_out[3*i+2] = z;
  q_out[i] = q;

  int leaf = (N - 1) + i;
  nmin[3*leaf] = x; nmin[3*leaf+1] = y; nmin[3*leaf+2] = z;
  nmax[3*leaf] = x; nmax[3*leaf+1] = y; nmax[3*leaf+2] = z;

  double *m = moments + (size_t)leaf * comp;
  m[0] = q;
  for (int s = 1; s < comp; ++s) m[s] = 0.0;

  if (i == 0) parent[0] = -1;
}

// ====================================================================
//  Kernel 4: LBVH binary radix tree build (Karras 2012)
// ====================================================================
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
  if (i >= n - 1) return;

  int dl = delta(i, i + 1, codes, n);
  int dr = delta(i, i - 1, codes, n);
  int d  = (dl - dr) >= 0 ? 1 : -1;
  int dmin = delta(i, i - d, codes, n);

  int lmax = 2;
  while (delta(i, i + lmax * d, codes, n) > dmin) lmax <<= 1;

  int l = 0;
  for (int t = lmax >> 1; t >= 1; t >>= 1) {
    if (delta(i, i + (l + t) * d, codes, n) > dmin) l += t;
  }
  int j = i + l * d;
  int first = min(i, j), last = max(i, j);

  int dnode = delta(first, last, codes, n);
  int split = first, step = last - first;
  do {
    step = (step + 1) >> 1;
    int ns = split + step;
    if (ns < last && delta(first, ns, codes, n) > dnode) split = ns;
  } while (step > 1);

  int lc = (split     == first) ? (n - 1) + split       : split;
  int rc = (split + 1 == last)  ? (n - 1) + (split + 1) : split + 1;
  left[i]  = lc;
  right[i] = rc;
  parent[lc] = i;
  parent[rc] = i;
}

// ====================================================================
//  Kernel 5: bottom-up multipole summarize (M2M upward pass)
// ====================================================================
// Accumulate m_parent^k += sum_{l<=k} C(k,l) * delta^(k-l) * m_child^l
// where delta = (dx,dy,dz) = c_child - c_parent.
__device__ __forceinline__ void m2m_shift(int comp,
                                          const double *__restrict__ m_child,
                                          double dx, double dy, double dz,
                                          double *__restrict__ m_parent) {
  // Precompute powers of dx, dy, dz up to BH_MAX_P (highest exponent we can need
  // for any l <= k with |k| <= p, since k_i - l_i <= k_i <= p).
  double pwx[BH_MAX_P + 1], pwy[BH_MAX_P + 1], pwz[BH_MAX_P + 1];
  pwx[0] = pwy[0] = pwz[0] = 1.0;
  #pragma unroll
  for (int i = 1; i <= BH_MAX_P; ++i) {
    pwx[i] = pwx[i - 1] * dx;
    pwy[i] = pwy[i - 1] * dy;
    pwz[i] = pwz[i - 1] * dz;
  }
  for (int sk = 0; sk < comp; ++sk) {
    int kx = c_mi_kx[sk], ky = c_mi_ky[sk], kz = c_mi_kz[sk];
    int n_k = kx + ky + kz;
    // l <= k component-wise implies |l| <= |k|; only walk slots up to the |l|=|k| shell.
    int sl_end = c_comp_at_order[n_k + 1];
    double k_fact = c_int_fact[kx] * c_int_fact[ky] * c_int_fact[kz];
    double sum = 0.0;
    for (int sl = 0; sl < sl_end; ++sl) {
      int lx = c_mi_kx[sl], ly = c_mi_ky[sl], lz = c_mi_kz[sl];
      int ex = kx - lx, ey = ky - ly, ez = kz - lz;
      if (ex < 0 || ey < 0 || ez < 0) continue;
      double l_fact = c_int_fact[lx] * c_int_fact[ly] * c_int_fact[lz];
      double e_fact = c_int_fact[ex] * c_int_fact[ey] * c_int_fact[ez];
      double C_kl  = k_fact / (l_fact * e_fact);   // C(k,l)
      double dpow  = pwx[ex] * pwy[ey] * pwz[ez];  // delta^(k-l)
      sum += C_kl * dpow * m_child[sl];
    }
    m_parent[sk] += sum;
  }
}

__global__ void summarize_kernel(int N, int comp,
                                 const int *__restrict__ left,
                                 const int *__restrict__ right,
                                 const int *__restrict__ parent,
                                 double *__restrict__ nmin,
                                 double *__restrict__ nmax,
                                 double *__restrict__ moments,
                                 int *__restrict__ flags) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  int node = parent[(N - 1) + i];
  while (node != -1) {
    __threadfence();
    if (atomicAdd(&flags[node], 1) == 0) return;

    int lc = left[node], rc = right[node];

    double mnx = fmin(nmin[3*lc],   nmin[3*rc]);
    double mny = fmin(nmin[3*lc+1], nmin[3*rc+1]);
    double mnz = fmin(nmin[3*lc+2], nmin[3*rc+2]);
    double mxx = fmax(nmax[3*lc],   nmax[3*rc]);
    double mxy = fmax(nmax[3*lc+1], nmax[3*rc+1]);
    double mxz = fmax(nmax[3*lc+2], nmax[3*rc+2]);
    nmin[3*node] = mnx; nmin[3*node+1] = mny; nmin[3*node+2] = mnz;
    nmax[3*node] = mxx; nmax[3*node+1] = mxy; nmax[3*node+2] = mxz;

    double cx = 0.5 * (mnx + mxx);
    double cy = 0.5 * (mny + mxy);
    double cz = 0.5 * (mnz + mxz);

    double *mp = moments + (size_t)node * comp;
    for (int s = 0; s < comp; ++s) mp[s] = 0.0;

    #pragma unroll
    for (int s_child = 0; s_child < 2; ++s_child) {
      int c = (s_child == 0) ? lc : rc;
      double ccx = 0.5 * (nmin[3*c]   + nmax[3*c]);
      double ccy = 0.5 * (nmin[3*c+1] + nmax[3*c+1]);
      double ccz = 0.5 * (nmin[3*c+2] + nmax[3*c+2]);
      const double *mc = moments + (size_t)c * comp;
      m2m_shift(comp, mc, ccx - cx, ccy - cy, ccz - cz, mp);
    }

    node = parent[node];
  }
}

// ====================================================================
//  Tree traversal + multipole evaluation (device helpers, called by the
//  target-side kernels below).  Multi-index Cartesian Taylor.
// ====================================================================
//
// Fill T_buf[0..comp_of(max_order)-1] with Cartesian Taylor coefficients
//   T^k(R) = d^k(1/r),   r = sqrt(R.R)
// using r^2 * T^(k+e_i) = -(2n+1) R_i T^k - n T^(k-e_i),   n = |k|.
//
// We pick the increment axis per slot (prefer x if kx>=1, else y, else z); this
// gives a unique "previous-shell" predecessor for every multi-index of order n>=2.
__device__ __forceinline__ void compute_T(double Rx, double Ry, double Rz,
                                          double inv_r2, double inv_r,
                                          int max_order, double *T_buf) {
  T_buf[0] = inv_r;                                   // T^(0,0,0) = 1/r
  if (max_order < 1) return;

  double inv_r3 = inv_r * inv_r2;
  T_buf[MI_LOOKUP(1, 0, 0)] = -Rx * inv_r3;           // T^(1,0,0) = -Rx/r^3
  T_buf[MI_LOOKUP(0, 1, 0)] = -Ry * inv_r3;
  T_buf[MI_LOOKUP(0, 0, 1)] = -Rz * inv_r3;

  // Lindsay-Krasny recurrence for T^k = d^k(1/r), summed over ALL axes:
  //   r^2 |k| T^k = -(2|k|-1) sum_i k_i R_i T^{k-e_i}
  //                -(|k|-1)   sum_i k_i(k_i-1) T^{k-2e_i}
  // (A single-axis two-term form is only exact through order 2.)
  for (int n = 2; n <= max_order; ++n) {
    int slot_start = c_comp_at_order[n];
    int slot_end   = c_comp_at_order[n + 1];
    double c1    = -(2.0 * n - 1.0);   // -(2|k|-1)
    double c2    = -(double)(n - 1);   // -(|k|-1)
    double inv_n = 1.0 / (double)n;
    for (int s = slot_start; s < slot_end; ++s) {
      int kx = c_mi_kx[s], ky = c_mi_ky[s], kz = c_mi_kz[s];
      double s1 = 0.0;   // sum_i k_i R_i T^{k-e_i}
      double s2 = 0.0;   // sum_i k_i(k_i-1) T^{k-2e_i}
      if (kx >= 1) {
        s1 += kx * Rx * T_buf[MI_LOOKUP(kx - 1, ky, kz)];
        if (kx >= 2) s2 += kx * (kx - 1) * T_buf[MI_LOOKUP(kx - 2, ky, kz)];
      }
      if (ky >= 1) {
        s1 += ky * Ry * T_buf[MI_LOOKUP(kx, ky - 1, kz)];
        if (ky >= 2) s2 += ky * (ky - 1) * T_buf[MI_LOOKUP(kx, ky - 2, kz)];
      }
      if (kz >= 1) {
        s1 += kz * Rz * T_buf[MI_LOOKUP(kx, ky, kz - 1)];
        if (kz >= 2) s2 += kz * (kz - 1) * T_buf[MI_LOOKUP(kx, ky, kz - 2)];
      }
      T_buf[s] = (c1 * s1 + c2 * s2) * inv_r2 * inv_n;
    }
  }
}

// Potential evaluator: phi += sum_{|k|<=p} (-1)^|k| / k! * T^k(R) * m^k
__device__ __forceinline__ double mp_potential_eval(int comp,
                                                    const double *__restrict__ m,
                                                    const double *__restrict__ T_buf) {
  double phi = 0.0;
  for (int s = 0; s < comp; ++s) {
    int n = c_mi_kx[s] + c_mi_ky[s] + c_mi_kz[s];
    double sign = (n & 1) ? -1.0 : 1.0;
    phi += sign * c_inv_fact[s] * T_buf[s] * m[s];
  }
  return phi;
}

// Field evaluator: g_i += -sum_{|k|<=p} (-1)^|k| / k! * T^(k+e_i)(R) * m^k
__device__ __forceinline__ void mp_field_eval(int comp,
                                              const double *__restrict__ m,
                                              const double *__restrict__ T_buf,
                                              double &gx, double &gy, double &gz) {
  for (int sk = 0; sk < comp; ++sk) {
    int kx = c_mi_kx[sk], ky = c_mi_ky[sk], kz = c_mi_kz[sk];
    int n = kx + ky + kz;
    double sign = (n & 1) ? -1.0 : 1.0;
    double common = sign * c_inv_fact[sk] * m[sk];
    int s_qx = MI_LOOKUP(kx + 1, ky, kz);
    int s_qy = MI_LOOKUP(kx, ky + 1, kz);
    int s_qz = MI_LOOKUP(kx, ky, kz + 1);
    gx -= common * T_buf[s_qx];
    gy -= common * T_buf[s_qy];
    gz -= common * T_buf[s_qz];
  }
}

template<bool FIELD>
__device__ void traverse(double tx, double ty, double tz, int self_atom,
                         const bh_tree t,
                         double &out_phi, double &gx, double &gy, double &gz) {
  out_phi = 0.0; gx = gy = gz = 0.0;
  const double theta2 = t.theta * t.theta;
  const int leaf0     = t.N - 1;
  const int p         = t.p;
  const int comp      = t.comp;
  const int max_T_ord = FIELD ? (p + 1) : p;

  // Per-thread Taylor-coefficient buffer; sized for the worst case.
  // Compiler will spill to local memory beyond the register budget.
  double T_buf[BH_COMP_MAX_FIELD];

  // Binary radix tree over 63-bit Morton codes has depth ~< 63 plus duplicate
  // index tie-break; 96 leaves comfortable margin on the DFS stack.
  int stack[96];
  int sp = 0;
  stack[sp++] = 0;

  while (sp > 0) {
    int node = stack[--sp];
    double cx = 0.5 * (t.d_min[3*node]   + t.d_max[3*node]);
    double cy = 0.5 * (t.d_min[3*node+1] + t.d_max[3*node+1]);
    double cz = 0.5 * (t.d_min[3*node+2] + t.d_max[3*node+2]);
    double Rx = tx - cx, Ry = ty - cy, Rz = tz - cz;
    double dist2 = Rx*Rx + Ry*Ry + Rz*Rz;

    if (node >= leaf0) {                          // leaf: single source atom
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

    // MAC: accept cell if (size / dist) < theta  <=>  size^2 < theta^2 * dist^2
    double ex = t.d_max[3*node]   - t.d_min[3*node];
    double ey = t.d_max[3*node+1] - t.d_min[3*node+1];
    double ez = t.d_max[3*node+2] - t.d_min[3*node+2];
    double size = fmax(ex, fmax(ey, ez));
    if (dist2 > 0.0 && size*size < theta2 * dist2) {
      double inv_r2 = 1.0 / dist2;
      double inv_r  = sqrt(inv_r2);
      compute_T(Rx, Ry, Rz, inv_r2, inv_r, max_T_ord, T_buf);
      const double *m = t.d_moments + (size_t)node * comp;
      if (FIELD) {
        mp_field_eval(comp, m, T_buf, gx, gy, gz);
      } else {
        out_phi += mp_potential_eval(comp, m, T_buf);
      }
    } else {
      stack[sp++] = t.d_left[node];
      stack[sp++] = t.d_right[node];
    }
  }
}

// ====================================================================
//  Target-side evaluation kernels (launched from the public C API)
// ====================================================================
__global__ void coulomb_kernel(bh_tree t, double *__restrict__ partial) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= t.N) return;
  double phi, gx, gy, gz;
  traverse<false>(t.d_pos[3*i], t.d_pos[3*i+1], t.d_pos[3*i+2], i, t,
                  phi, gx, gy, gz);
  partial[i] = 0.5 * t.d_q[i] * phi;
}

__global__ void potential_kernel(bh_tree t, int num_pts,
                                 const double *__restrict__ V,
                                 const double *__restrict__ flux,
                                 double *__restrict__ partial) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_pts) return;
  double phi, gx, gy, gz;
  traverse<false>(V[3*i], V[3*i+1], V[3*i+2], -1, t, phi, gx, gy, gz);
  partial[i] = flux[i] * phi;
}

__global__ void field_kernel(bh_tree t, int num_tri_verts,
                             const double *__restrict__ vert,
                             const double *__restrict__ norms,
                             const double *__restrict__ phi_sup,
                             const double *__restrict__ area,
                             double inv_4pi, double *__restrict__ partial) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_tri_verts) return;
  double phi, gx, gy, gz;
  traverse<true>(vert[3*i], vert[3*i+1], vert[3*i+2], -1, t, phi, gx, gy, gz);
  double dot    = gx*norms[3*i] + gy*norms[3*i+1] + gz*norms[3*i+2];
  double factor = phi_sup[i] * inv_4pi * area[i / 3] / 3.0;
  partial[i] = dot * factor;
}

// ====================================================================
//  Host-side reduction (matches Phase 0 summation order for A/B parity)
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
                        int p,
                        bh_tree **out) {
  // Clamp p to the supported range with a stderr warning.
  if (p < 1) {
    fprintf(stderr, "[barnes_hut] bh_order=%d clamped to 1\n", p);
    p = 1;
  } else if (p > BH_MAX_P) {
    fprintf(stderr, "[barnes_hut] bh_order=%d clamped to %d (BH_MAX_P)\n", p, BH_MAX_P);
    p = BH_MAX_P;
  }

  init_constant_tables();

  bh_tree *t = new bh_tree();
  t->N       = num_atoms;
  t->n_nodes = (num_atoms > 0) ? (2 * num_atoms - 1) : 0;
  t->p       = p;
  t->comp    = comp_of(p);
  t->theta   = theta;
  *out = t;
  if (num_atoms <= 0) {
    t->d_pos = t->d_q = t->d_min = t->d_max = t->d_moments = nullptr;
    t->d_left = t->d_right = t->d_parent = nullptr;
    return;
  }

  const int N    = num_atoms;
  const int nn   = t->n_nodes;
  const int comp = t->comp;

  CUDA_CHECK(cudaMalloc(&t->d_pos,     3 * N  * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&t->d_q,           N  * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&t->d_min,     3 * nn * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&t->d_max,     3 * nn * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&t->d_left,        nn * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&t->d_right,       nn * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&t->d_parent,      nn * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&t->d_moments,
                        (size_t)nn * comp * sizeof(double)));

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

  // 3. Reorder atoms into Morton order + initialize leaf nodes (incl. moments)
  reorder_kernel<<<mt_blocks, tpb>>>(N, comp, d_idx_sorted, d_atoms, d_charges,
                                     t->d_pos, t->d_q,
                                     t->d_min, t->d_max,
                                     t->d_moments, t->d_parent);
  CUDA_CHECK(cudaGetLastError());
  cudaFree(d_idx_sorted);

  if (N > 1) {
    // 4. Build internal nodes (Karras)
    int in_blocks = (N - 1 + tpb - 1) / tpb;
    build_internal_kernel<<<in_blocks, tpb>>>(N, d_codes_sorted,
                                              t->d_left, t->d_right, t->d_parent);
    CUDA_CHECK(cudaGetLastError());

    // 5. Bottom-up multipole summarize
    int *d_flags;
    CUDA_CHECK(cudaMalloc(&d_flags, nn * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_flags, 0, nn * sizeof(int)));
    summarize_kernel<<<mt_blocks, tpb>>>(N, comp,
                                         t->d_left, t->d_right, t->d_parent,
                                         t->d_min, t->d_max,
                                         t->d_moments, d_flags);
    CUDA_CHECK(cudaGetLastError());
    cudaFree(d_flags);
  }
  cudaFree(d_codes_sorted);
  CUDA_CHECK(cudaDeviceSynchronize());
}

void bh_free_tree(bh_tree *t) {
  if (!t) return;
  cudaFree(t->d_pos);     cudaFree(t->d_q);
  cudaFree(t->d_min);     cudaFree(t->d_max);
  cudaFree(t->d_left);    cudaFree(t->d_right);   cudaFree(t->d_parent);
  cudaFree(t->d_moments);
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

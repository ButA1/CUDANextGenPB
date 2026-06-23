/*
 *  Barnes-Hut / FMM treecode (FMM Stage 2: complex spherical-harmonic moments).
 *
 *  Multipole representation (Cheng, Greengard & Rokhlin 1999, JCP 155:468):
 *    - Per-node moments are complex M_n^m (CGR'99 Eq. 4 normalization), stored
 *      for the m>=0 half only (0<=m<=n<=p); M_n^{-m}=conj(M_n^m). Layout: doubles
 *      [re,im] at slot s=sph_slot(n,m). Count comp_sph(p) = (p+1)(p+2) doubles.
 *    - P2M: leaf monopole M_0^0 = q (ρ=0). M2M: T_MM (Eq. 13). M2P potential:
 *      irregular solid harmonics S_n^m = Y_n^m/r^{n+1} contracted with the moments
 *      (Eq. 5). M2P field (E_ion): exact gradient of the potential expansion via
 *      forward-mode automatic differentiation (the `dual` scalar), so no separate
 *      solid-harmonic gradient recurrence is needed.
 *
 *  Traversal: cooperative breadth-first (Stage 1) for the DTT path, private-stack
 *  DFS for the per-target treecode path. LBVH topology and the bbox/Morton/build
 *  kernels are unchanged.
 */

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <vector>
#include <cstdio>
#include <cstdint>
#include <cfloat>
#include <cmath>
#include <algorithm>
#include <utility>

static void check_cuda(cudaError_t err, const char *msg, int line) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error at line %d (%s): %s\n",
            line, msg, cudaGetErrorString(err));
  }
}
#define CUDA_CHECK(call) check_cuda((call), #call, __LINE__)

// ====================================================================
//  FMM Stage 2: complex spherical-harmonic multipole moments (CGR'99,
//  J. Comput. Phys. 155:468). BH_MAX_P is the largest multipole order
//  supported. Moments M_n^m are stored for the m>=0 half only (0<=m<=n<=p);
//  the m<0 half is recovered by M_n^{-m} = conj(M_n^m) (Eq. 4 convention,
//  no (-1)^m). The field (gradient) is obtained by forward-mode automatic
//  differentiation of the potential expansion (see solid_harmonics / dual),
//  so it needs no expansion above order p.
// ====================================================================
#define BH_MAX_P        10
#define BH_FACT_MAX     (2 * BH_MAX_P + 2)   // highest factorial index needed: (n+m)! <= (2p)!

// # complex moments per node for orders 0..p, m in [0,n] (the stored m>=0 half).
__host__ __device__ constexpr int nmom_sph(int p) { return (p + 1) * (p + 2) / 2; }
// # doubles per node = 2 * complex count (interleaved re,im at slot s -> [2s],[2s+1]).
__host__ __device__ constexpr int comp_sph(int p) { return (p + 1) * (p + 2); }
#define BH_NMOM_MAX  nmom_sph(BH_MAX_P)   // 66

// runtime doubles-per-node for a given order (host alloc / bh_tree.comp)
__host__ __device__ __forceinline__ int comp_of(int p) { return comp_sph(p); }

// linear slot for (n,m) with 0 <= m <= n :  s = n(n+1)/2 + m
__host__ __device__ constexpr int sph_slot(int n, int m) { return n * (n + 1) / 2 + m; }

// ====================================================================
//  Arithmetic types for the spherical-harmonic path.
//  - dual: forward-mode AD scalar (value + d/dx,d/dy,d/dz). Used to obtain
//    the field as the exact gradient of the potential expansion, so no
//    hand-derived solid-harmonic gradient recurrence is required.
//  - cT<T>: complex number over scalar T (T = double for the potential,
//    T = dual for the field). cmplx is the plain double-complex alias.
// ====================================================================
struct dual {
  double v, dx, dy, dz;
  __host__ __device__ dual() : v(0.0), dx(0.0), dy(0.0), dz(0.0) {}
  __host__ __device__ dual(double a) : v(a), dx(0.0), dy(0.0), dz(0.0) {}
  __host__ __device__ dual(double a, double bx, double by, double bz)
      : v(a), dx(bx), dy(by), dz(bz) {}
};
__host__ __device__ __forceinline__ dual operator+(dual a, dual b) {
  return dual(a.v + b.v, a.dx + b.dx, a.dy + b.dy, a.dz + b.dz);
}
__host__ __device__ __forceinline__ dual operator-(dual a, dual b) {
  return dual(a.v - b.v, a.dx - b.dx, a.dy - b.dy, a.dz - b.dz);
}
__host__ __device__ __forceinline__ dual operator*(dual a, dual b) {
  return dual(a.v * b.v, a.dx * b.v + a.v * b.dx,
              a.dy * b.v + a.v * b.dy, a.dz * b.v + a.v * b.dz);
}
__host__ __device__ __forceinline__ dual operator*(double s, dual a) {
  return dual(s * a.v, s * a.dx, s * a.dy, s * a.dz);
}
__host__ __device__ __forceinline__ dual operator*(dual a, double s) { return s * a; }
__host__ __device__ __forceinline__ dual operator/(dual a, dual b) {
  double inv = 1.0 / b.v, v = a.v * inv;
  return dual(v, (a.dx - v * b.dx) * inv, (a.dy - v * b.dy) * inv, (a.dz - v * b.dz) * inv);
}
__host__ __device__ __forceinline__ dual operator/(dual a, double s) {
  double inv = 1.0 / s;
  return dual(a.v * inv, a.dx * inv, a.dy * inv, a.dz * inv);
}
__host__ __device__ __forceinline__ dual operator/(double a, dual b) {
  double inv = 1.0 / b.v, v = a * inv;
  return dual(v, -v * b.dx * inv, -v * b.dy * inv, -v * b.dz * inv);
}
// scalar sqrt, overloaded so the templated harmonic builder works for both T.
__host__ __device__ __forceinline__ double tsqrt(double a) { return sqrt(a); }
__host__ __device__ __forceinline__ dual   tsqrt(dual a) {
  double s = sqrt(a.v), h = (s > 0.0) ? 0.5 / s : 0.0;
  return dual(s, a.dx * h, a.dy * h, a.dz * h);
}

template<class T> struct cT {
  T re, im;
  __host__ __device__ cT() {}
  __host__ __device__ cT(T r, T i) : re(r), im(i) {}
};
template<class T> __host__ __device__ __forceinline__ cT<T> operator+(const cT<T>& a, const cT<T>& b) {
  return cT<T>(a.re + b.re, a.im + b.im);
}
template<class T> __host__ __device__ __forceinline__ cT<T> operator*(const cT<T>& a, const cT<T>& b) {
  return cT<T>(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}
template<class T> __host__ __device__ __forceinline__ cT<T> operator*(const T& s, const cT<T>& a) {
  return cT<T>(s * a.re, s * a.im);
}
typedef cT<double> cmplx;
__host__ __device__ __forceinline__ cmplx conjc(cmplx a) { return cmplx(a.re, -a.im); }

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
  int     leaf_size;  // max atoms per terminal cluster (P2P cutoff)
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

  // contiguous Morton-sorted atom range [d_first, d_last] covered by each node
  int    *d_first;    // n_nodes
  int    *d_last;     // n_nodes

  // per-node complex spherical-harmonic moments M_n^m, m>=0 half, interleaved
  // re,im at slot s=sph_slot(n,m): d_moments[node*comp + 2s], [..+2s+1].
  double *d_moments;  // n_nodes * comp   (comp = comp_sph(p) doubles)
};

// ====================================================================
//  Target tree (geometry only) for the DTT energy path. Same 2N-1 LBVH
//  layout as bh_tree but carries no charges or multipole moments -- targets
//  are only grouped into cells so one CUDA block can serve a whole leaf.
//  d_orig[sorted] -> original (caller) point index, so per-target weights
//  (flux / norms / phi_sup / area[i/3]) are read in the caller's order and
//  results scatter back to the original layout.
// ====================================================================
struct bh_target_tree {
  int     N;          // number of target points (leaves)
  int     n_nodes;    // 2N - 1
  int     leaf_size;  // max points per terminal cluster
  int     n_leaves;   // # frontier (block-assigned) leaf cells

  double *d_pos;      // 3N, Morton-sorted target positions
  int    *d_orig;     // N,  original index of each sorted point

  double *d_min;      // 3 * n_nodes
  double *d_max;      // 3 * n_nodes
  int    *d_left;     // n_nodes
  int    *d_right;    // n_nodes
  int    *d_parent;   // n_nodes
  int    *d_first;    // n_nodes
  int    *d_last;     // n_nodes

  int    *d_leaf_nodes; // n_leaves, node indices of the leaf-size frontier
};

// ====================================================================
//  Constant-memory tables (populated once on first build):
//    c_fact[k] = k!                      (k up to (2p)! ; doubles)
//    c_A[idx]  = A_n^m = (-1)^n / sqrt((n-m)!(n+m)!)   (CGR'99 Eq. 14)
//                idx = n*n + (m+n), m in [-n, n]
// ====================================================================
__constant__ double c_fact[BH_FACT_MAX + 1];
__constant__ double c_A[(BH_MAX_P + 1) * (BH_MAX_P + 1)];

__device__ __forceinline__ double fact_dev(int k) { return c_fact[k]; }
__device__ __forceinline__ double A_nm(int n, int m) { return c_A[n * n + (m + n)]; }

// Host-side one-shot initializer for the constant tables.
static void init_constant_tables() {
  static bool initialized = false;
  if (initialized) return;

  double h_fact[BH_FACT_MAX + 1];
  h_fact[0] = 1.0;
  for (int k = 1; k <= BH_FACT_MAX; ++k) h_fact[k] = h_fact[k - 1] * (double)k;

  double h_A[(BH_MAX_P + 1) * (BH_MAX_P + 1)];
  for (int n = 0; n <= BH_MAX_P; ++n) {
    double sgn = (n & 1) ? -1.0 : 1.0;
    for (int m = -n; m <= n; ++m)
      h_A[n * n + (m + n)] = sgn / sqrt(h_fact[n - m] * h_fact[n + m]);
  }

  CUDA_CHECK(cudaMemcpyToSymbol(c_fact, h_fact, sizeof(h_fact)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_A,    h_A,    sizeof(h_A)));
  initialized = true;
}

// scalar value extractor (for value-only branch guards inside the AD path)
__host__ __device__ __forceinline__ double val(double a) { return a; }
__host__ __device__ __forceinline__ double val(dual a)   { return a.v; }

// ====================================================================
//  Solid harmonics of a vector (x,y,z) in CGR'99 normalization (Eq. 4):
//    Y_n^m(θ,φ) = sqrt((n-|m|)!/(n+|m|)!) · P_n^{|m|}(cosθ) · e^{imφ}   (CS in P)
//  REGULAR=true  -> R_n^m =  r^n     · Y_n^m   (forming/translating moments)
//  REGULAR=false -> S_n^m =  Y_n^m / r^{n+1}   (irregular; evaluating Φ)
//  Filled for the m>=0 half: out[sph_slot(n,m)], 0<=m<=n<=maxn (m<0 = conj).
//  Templated on scalar T: T=double for the potential, T=dual for the field
//  (the gradient falls out by forward-mode AD, no derivative recurrence).
// ====================================================================
template<class T, bool REGULAR>
__device__ void solid_harmonics(int maxn, T x, T y, T z, cT<T>* out) {
  T r   = tsqrt(x * x + y * y + z * z);
  T rxy = tsqrt(x * x + y * y);
  bool ron  = val(r)   > 0.0;
  bool rxon = val(rxy) > 0.0;
  T ct   = ron  ? (z   / r)   : T(1.0);   // cosθ
  T st   = ron  ? (rxy / r)   : T(0.0);   // sinθ >= 0
  T cphi = rxon ? (x   / rxy) : T(1.0);   // on the z-axis P_n^{m>0}=0 kills e^{imφ}
  T sphi = rxon ? (y   / rxy) : T(0.0);

  // associated Legendre P_n^m(cosθ) (Condon–Shortley), m>=0
  T P[BH_NMOM_MAX];
  P[sph_slot(0, 0)] = T(1.0);
  for (int m = 1; m <= maxn; ++m)
    P[sph_slot(m, m)] = (-(2.0 * m - 1.0)) * (st * P[sph_slot(m - 1, m - 1)]);
  for (int m = 0; m < maxn; ++m)
    P[sph_slot(m + 1, m)] = (2.0 * m + 1.0) * (ct * P[sph_slot(m, m)]);
  for (int m = 0; m <= maxn; ++m)
    for (int n = m + 2; n <= maxn; ++n)
      P[sph_slot(n, m)] = ((2.0 * n - 1.0) * (ct * P[sph_slot(n - 1, m)])
                          - (n + m - 1.0) * P[sph_slot(n - 2, m)]) / (double)(n - m);

  // e^{i m φ}, m = 0..maxn
  cT<T> eim[BH_MAX_P + 1];
  eim[0] = cT<T>(T(1.0), T(0.0));
  cT<T> ep(cphi, sphi);
  for (int m = 1; m <= maxn; ++m) eim[m] = eim[m - 1] * ep;

  // radial scaling per degree: r^n (regular) or 1/r^{n+1} (irregular)
  T rad[BH_MAX_P + 1];
  if constexpr (REGULAR) {
    rad[0] = T(1.0);
    for (int n = 1; n <= maxn; ++n) rad[n] = rad[n - 1] * r;
  } else {
    T inv_r = ron ? (T(1.0) / r) : T(0.0);
    rad[0] = inv_r;
    for (int n = 1; n <= maxn; ++n) rad[n] = rad[n - 1] * inv_r;
  }

  for (int n = 0; n <= maxn; ++n)
    for (int m = 0; m <= n; ++m) {
      double c = sqrt(fact_dev(n - m) / fact_dev(n + m));   // sqrt((n-m)!/(n+m)!)
      T a = c * (P[sph_slot(n, m)] * rad[n]);
      out[sph_slot(n, m)] = a * eim[m];
    }
}

// Fetch M_n^m / R_n^m for any m in [-n,n] from the stored m>=0 half (conj for m<0).
__device__ __forceinline__ cmplx get_M(const double* M, int n, int m) {
  if (m >= 0) { int s = sph_slot(n, m);  return cmplx(M[2 * s], M[2 * s + 1]); }
  int s = sph_slot(n, -m);               return cmplx(M[2 * s], -M[2 * s + 1]);
}
__device__ __forceinline__ cmplx get_R(const cmplx* R, int n, int m) {
  if (m >= 0) return R[sph_slot(n, m)];
  cmplx r = R[sph_slot(n, -m)];          return cmplx(r.re, -r.im);
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
                               int *__restrict__ parent,
                               int *__restrict__ first,
                               int *__restrict__ last) {
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

  // leaf covers the single atom i (in Morton-sorted order)
  first[leaf] = i;
  last[leaf]  = i;

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
                                      int *__restrict__ parent,
                                      int *__restrict__ first_out,
                                      int *__restrict__ last_out) {
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

  // this internal node covers the contiguous leaf/atom range [first, last]
  first_out[i] = first;
  last_out[i]  = last;
}

// ====================================================================
//  Kernel 5: bottom-up multipole summarize (M2M upward pass)
// ====================================================================
// recover p from the doubles-per-node count: comp = comp_sph(p) = (p+1)(p+2).
__device__ __forceinline__ int p_from_comp(int comp) {
  int p = 0; while (comp_sph(p) < comp) ++p; return p;
}
__device__ __forceinline__ int iabs(int a) { return a < 0 ? -a : a; }

// Multipole-to-multipole translation T_MM (CGR'99 Thm 2.3, Eq. 13), shifting a
// child expansion {O} about the child center to the parent center and adding it
// into {M}. (ρ,α,β) are the spherical coords of the shift (child-parent), encoded
// here as the regular solid harmonics R_n^m = ρ^n Y_n^m of the shift vector:
//   M_j^k += Σ_{n=0}^{j} Σ_{m=-n}^{n} O_{j-n}^{k-m} · i^{|k|-|m|-|k-m|}
//                · A_n^m A_{j-n}^{k-m} / A_j^k · R_n^{-m}.
// The phase i^{|k|-|m|-|k-m|} has an even exponent, hence the real (-1)^(e/2).
__device__ void m2m_shift(int comp,
                          const double *__restrict__ O,
                          double dx, double dy, double dz,
                          double *__restrict__ M) {
  int p = p_from_comp(comp);
  cmplx R[BH_NMOM_MAX];
  solid_harmonics<double, true>(p, dx, dy, dz, R);   // R_n^m of shift, n<=p

  for (int j = 0; j <= p; ++j)
    for (int k = 0; k <= j; ++k) {                   // store the m>=0 half only
      cmplx acc(0.0, 0.0);
      for (int n = 0; n <= j; ++n)
        for (int m = -n; m <= n; ++m) {
          int jn = j - n, km = k - m;
          if (km < -jn || km > jn) continue;         // O_{jn}^{km} needs |km| <= jn
          int e = k - iabs(m) - iabs(km);            // |k|=k (k>=0); e is even
          double ipow = ((e / 2) & 1) ? -1.0 : 1.0;
          double coef = ipow * A_nm(n, m) * A_nm(jn, km) / A_nm(j, k);
          acc = acc + (coef * (get_M(O, jn, km) * get_R(R, n, -m)));
        }
      int s = sph_slot(j, k);
      M[2 * s]     += acc.re;
      M[2 * s + 1] += acc.im;
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
//  Target-tree build kernels (geometry only; clones of the atom-tree
//  reorder/summarize stripped of charges and multipole moments).
// ====================================================================

// Reorder target points into Morton order, init single-point leaves, and keep
// the sort permutation so results can be scattered back to caller order.
__global__ void reorder_geom_kernel(int N,
                                    const int *__restrict__ order,
                                    const double *__restrict__ pos_in,
                                    double *__restrict__ pos_out,
                                    int *__restrict__ orig,
                                    double *__restrict__ nmin,
                                    double *__restrict__ nmax,
                                    int *__restrict__ parent,
                                    int *__restrict__ first,
                                    int *__restrict__ last) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  int src = order[i];
  double x = pos_in[3*src], y = pos_in[3*src+1], z = pos_in[3*src+2];
  pos_out[3*i] = x; pos_out[3*i+1] = y; pos_out[3*i+2] = z;
  orig[i] = src;

  int leaf = (N - 1) + i;
  nmin[3*leaf] = x; nmin[3*leaf+1] = y; nmin[3*leaf+2] = z;
  nmax[3*leaf] = x; nmax[3*leaf+1] = y; nmax[3*leaf+2] = z;
  first[leaf] = i;
  last[leaf]  = i;

  if (i == 0) parent[0] = -1;
}

// Bottom-up AABB propagation only (no M2M). Same flag handshake as
// summarize_kernel: the second child to arrive proceeds into the parent.
__global__ void summarize_aabb_kernel(int N,
                                      const int *__restrict__ left,
                                      const int *__restrict__ right,
                                      const int *__restrict__ parent,
                                      double *__restrict__ nmin,
                                      double *__restrict__ nmax,
                                      int *__restrict__ flags) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  int node = parent[(N - 1) + i];
  while (node != -1) {
    __threadfence();
    if (atomicAdd(&flags[node], 1) == 0) return;

    int lc = left[node], rc = right[node];
    nmin[3*node]   = fmin(nmin[3*lc],   nmin[3*rc]);
    nmin[3*node+1] = fmin(nmin[3*lc+1], nmin[3*rc+1]);
    nmin[3*node+2] = fmin(nmin[3*lc+2], nmin[3*rc+2]);
    nmax[3*node]   = fmax(nmax[3*lc],   nmax[3*rc]);
    nmax[3*node+1] = fmax(nmax[3*lc+1], nmax[3*rc+1]);
    nmax[3*node+2] = fmax(nmax[3*lc+2], nmax[3*rc+2]);

    node = parent[node];
  }
}

// Mark the leaf-size frontier: topmost nodes whose covered point count is
// <= leaf_size (the root counts as having an infinite-size parent). These
// nodes partition all target points and become the per-block leaf cells.
__global__ void mark_frontier_kernel(int n_nodes, int N, int leaf_size,
                                     const int *__restrict__ first,
                                     const int *__restrict__ last,
                                     const int *__restrict__ parent,
                                     int *__restrict__ flag) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= n_nodes) return;
  int cnt = last[v] - first[v] + 1;
  int par = parent[v];
  int pcnt = (par >= 0) ? (last[par] - first[par] + 1) : (N + 1);
  flag[v] = (cnt <= leaf_size && (par < 0 || pcnt > leaf_size)) ? 1 : 0;
}

// Fill out[i] = i (node-id source for the frontier compaction).
__global__ void iota_kernel(int n, int *__restrict__ out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = i;
}

// ====================================================================
//  Multipole evaluation (M2P) in the complex spherical-harmonic basis.
//  Both evaluators contract the stored moments {M_n^m, m>=0} against the
//  irregular solid harmonics S_n^m of the target-relative vector, using
//  M_n^{-m}=conj(M_n^m), S_n^{-m}=conj(S_n^m) to fold the m<0 half:
//     Σ_{m=-n}^{n} M_n^m S_n^m = M_n^0 S_n^0 + 2·Re Σ_{m=1}^{n} M_n^m S_n^m.
// ====================================================================

// Potential Φ (CGR'99 Thm 2.1, Eq. 5). S = irregular solid harmonics (double).
__device__ double mp_potential_eval_sph(int p, const double *__restrict__ M,
                                                        const cmplx *__restrict__ S) {
  double phi = 0.0;
  for (int n = 0; n <= p; ++n) {
    int s0 = sph_slot(n, 0);
    phi += M[2 * s0] * S[s0].re;                                   // m = 0 (real)
    for (int m = 1; m <= n; ++m) {
      int s = sph_slot(n, m);
      phi += 2.0 * (M[2 * s] * S[s].re - M[2 * s + 1] * S[s].im);  // 2 Re(M_n^m S_n^m)
    }
  }
  return phi;
}

// Field E = -∇Φ. S is built in dual arithmetic (S_n^m carries d/dx,d/dy,d/dz),
// so contracting it with the moments yields Φ together with its exact gradient
// (forward-mode AD); the field is the negated gradient. No expansion above p.
__device__ void mp_field_eval_sph(int p, const double *__restrict__ M,
                                                  const cT<dual> *__restrict__ S,
                                                  double &gx, double &gy, double &gz) {
  dual phi;
  for (int n = 0; n <= p; ++n) {
    int s0 = sph_slot(n, 0);
    phi = phi + M[2 * s0] * S[s0].re;
    for (int m = 1; m <= n; ++m) {
      int s = sph_slot(n, m);
      dual termre = M[2 * s] * S[s].re - M[2 * s + 1] * S[s].im;   // Re(M_n^m S_n^m)
      phi = phi + 2.0 * termre;
    }
  }
  gx -= phi.dx; gy -= phi.dy; gz -= phi.dz;
}

template<bool FIELD, int P>
__device__ void traverse(double tx, double ty, double tz, int self_atom,
                         const bh_tree t,
                         double &out_phi, double &gx, double &gy, double &gz) {
  out_phi = 0.0; gx = gy = gz = 0.0;
  const double theta2 = t.theta * t.theta;
  const int leaf_size = t.leaf_size;

  constexpr int NM       = nmom_sph(P);   // # complex moments per node (m>=0 half)
  constexpr int COMP_DBL = comp_sph(P);   // # doubles per node
  // The per-thread solid-harmonic buffer is declared inside the far-field branch
  // (cmplx for the potential, dual-complex for the field).

  // Binary radix tree over 63-bit Morton codes has depth ~< 63 plus duplicate
  // index tie-break; 96 leaves comfortable margin on the DFS stack.
  int stack[96];
  int sp = 0;
  stack[sp++] = 0;

  while (sp > 0) {
    int node = stack[--sp];
    // load each AABB bound once; reused for both the center and the extent
    double mnx = t.d_min[3*node],   mxx = t.d_max[3*node];
    double mny = t.d_min[3*node+1], mxy = t.d_max[3*node+1];
    double mnz = t.d_min[3*node+2], mxz = t.d_max[3*node+2];
    double cx = 0.5 * (mnx + mxx);
    double cy = 0.5 * (mny + mxy);
    double cz = 0.5 * (mnz + mxz);
    double Rx = tx - cx, Ry = ty - cy, Rz = tz - cz;
    double dist2 = Rx*Rx + Ry*Ry + Rz*Rz;

    // MAC: accept cell if (size / dist) < theta  <=>  size^2 < theta^2 * dist^2
    double ex = mxx - mnx, ey = mxy - mny, ez = mxz - mnz;
    double size = fmax(ex, fmax(ey, ez));

    if (dist2 > 0.0 && size*size < theta2 * dist2) {
      // far enough: particle-cluster multipole (M2P) interaction.
      const double *m = t.d_moments + (size_t)node * COMP_DBL;
      if constexpr (FIELD) {
        cT<dual> S[NM];
        solid_harmonics<dual, false>(P, dual(Rx, 1.0, 0.0, 0.0),
                                        dual(Ry, 0.0, 1.0, 0.0),
                                        dual(Rz, 0.0, 0.0, 1.0), S);
        mp_field_eval_sph(P, m, S, gx, gy, gz);
      } else {
        cmplx S[NM];
        solid_harmonics<double, false>(P, Rx, Ry, Rz, S);
        out_phi += mp_potential_eval_sph(P, m, S);
      }
      continue;
    }

    int cnt = t.d_last[node] - t.d_first[node] + 1;
    if (cnt <= leaf_size) {
      // terminal cluster: resolve its atoms by direct particle-particle sums
      int a0 = t.d_first[node], a1 = t.d_last[node];
      for (int a = a0; a <= a1; ++a) {
        if (a == self_atom) continue;
        double rx = tx - t.d_pos[3*a], ry = ty - t.d_pos[3*a+1], rz = tz - t.d_pos[3*a+2];
        double d2 = rx*rx + ry*ry + rz*rz;
        if (d2 <= 0.0) continue;
        double q = t.d_q[a];
        double inv_r = rsqrt(d2);
        if constexpr (FIELD) {
          // hoist q/r^3 so each component update is a single FMA
          double qir3 = q * inv_r * inv_r * inv_r;
          gx += qir3 * rx;
          gy += qir3 * ry;
          gz += qir3 * rz;
        } else {
          out_phi += q * inv_r;
        }
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
template<int P>
__global__ void coulomb_kernel(bh_tree t, double *__restrict__ partial) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= t.N) return;
  double phi, gx, gy, gz;
  traverse<false, P>(t.d_pos[3*i], t.d_pos[3*i+1], t.d_pos[3*i+2], i, t,
                     phi, gx, gy, gz);
  partial[i] = 0.5 * t.d_q[i] * phi;
}

template<int P>
__global__ void potential_kernel(bh_tree t, int num_pts,
                                 const double *__restrict__ V,
                                 const double *__restrict__ flux,
                                 double *__restrict__ partial) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_pts) return;
  double phi, gx, gy, gz;
  traverse<false, P>(V[3*i], V[3*i+1], V[3*i+2], -1, t, phi, gx, gy, gz);
  partial[i] = flux[i] * phi;
}

template<int P>
__global__ void field_kernel(bh_tree t, int num_tri_verts,
                             const double *__restrict__ vert,
                             const double *__restrict__ norms,
                             const double *__restrict__ phi_sup,
                             const double *__restrict__ area,
                             double inv_4pi, double *__restrict__ partial) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_tri_verts) return;
  double phi, gx, gy, gz;
  traverse<true, P>(vert[3*i], vert[3*i+1], vert[3*i+2], -1, t, phi, gx, gy, gz);
  double dot    = gx*norms[3*i] + gy*norms[3*i+1] + gz*norms[3*i+2];
  double factor = phi_sup[i] * inv_4pi * area[i / 3] / 3.0;
  partial[i] = dot * factor;
}

// ====================================================================
//  DTT evaluation kernel (FMM Stage 1): one CUDA block per target leaf cell.
//  The block descends the source (atom) tree ONCE, cooperatively and
//  breadth-first (Bonsai; Bedorf, Gaburov & Portegies Zwart 2012, sec. 2.2):
//  each step the DTT_BDIM threads test DTT_BDIM source nodes in parallel against
//  a two-sided MAC (source cell vs the whole target leaf), and a block prefix-sum
//  (cub::BlockScan) compacts the survivors into per-block lists -- admissible
//  source cells -> M2P list, terminal source clusters -> P2P list, internal nodes
//  -> the next BFS level. The lists hold node INDICES; the expensive FP64 payload
//  then runs over the shared list with one target point per thread, reading
//  moments / particles from global. This removes the per-thread redundant descent
//  of the old private-stack DFS and materializes exactly the admissible-cell list
//  that M2L will consume (Stage 3). Far field is still M2P (no M2L yet). Targets
//  != sources, so no self-exclusion. Weights and the final scatter use the
//  original (caller) point order via tgt.d_orig.
// ====================================================================

// Cooperative block width. MUST equal the launch blockDim -- cub::BlockScan needs
// it at compile time. Target leaves hold <= 256 points, so 256 threads always
// cover a leaf and give the widest cooperative traversal.
constexpr int DTT_BDIM      = 256;
constexpr int DTT_FRONT_CAP = 2048;  // BFS frontier slots per level (shared); see overflow guard
constexpr int DTT_LIST_CAP  = 512;   // M2P/P2P shared buffer; drained when within DTT_BDIM of full

// M2P drain: each active thread evaluates the admissible-cell list into its own
// target point. Verbatim multipole math from the old kernel; moments from global.
template<bool FIELD, int P>
__device__ void dtt_drain_m2p(const bh_tree &src,
                                             const int *__restrict__ list, int n, bool active,
                                             double px, double py, double pz,
                                             double &phi, double &gx, double &gy, double &gz) {
  if (!active) return;
  constexpr int NM       = nmom_sph(P);
  constexpr int COMP_DBL = comp_sph(P);
  for (int j = 0; j < n; ++j) {
    int B = list[j];
    double Bcx = 0.5*(src.d_min[3*B]   + src.d_max[3*B]);
    double Bcy = 0.5*(src.d_min[3*B+1] + src.d_max[3*B+1]);
    double Bcz = 0.5*(src.d_min[3*B+2] + src.d_max[3*B+2]);
    double Rxp = px - Bcx, Ryp = py - Bcy, Rzp = pz - Bcz;
    const double *m = src.d_moments + (size_t)B * COMP_DBL;
    if constexpr (FIELD) {
      cT<dual> S[NM];
      solid_harmonics<dual, false>(P, dual(Rxp, 1.0, 0.0, 0.0),
                                      dual(Ryp, 0.0, 1.0, 0.0),
                                      dual(Rzp, 0.0, 0.0, 1.0), S);
      mp_field_eval_sph(P, m, S, gx, gy, gz);
    } else {
      cmplx S[NM];
      solid_harmonics<double, false>(P, Rxp, Ryp, Rzp, S);
      phi += mp_potential_eval_sph(P, m, S);
    }
  }
}

// P2P drain: each active thread direct-sums the terminal-leaf list into its point.
template<bool FIELD>
__device__ void dtt_drain_p2p(const bh_tree &src,
                                             const int *__restrict__ list, int n, bool active,
                                             double px, double py, double pz,
                                             double &phi, double &gx, double &gy, double &gz) {
  if (!active) return;
  for (int j = 0; j < n; ++j) {
    int B = list[j];
    int b0 = src.d_first[B], b1 = src.d_last[B];
    for (int b = b0; b <= b1; ++b) {
      double rx = px - src.d_pos[3*b], ry = py - src.d_pos[3*b+1], rz = pz - src.d_pos[3*b+2];
      double dd = rx*rx + ry*ry + rz*rz;
      if (dd <= 0.0) continue;
      double q  = src.d_q[b];
      double ir = rsqrt(dd);
      if constexpr (FIELD) {
        double qir3 = q * ir * ir * ir;
        gx += qir3 * rx; gy += qir3 * ry; gz += qir3 * rz;
      } else {
        phi += q * ir;
      }
    }
  }
}
// ====================================================================
template<bool FIELD, int P>
__global__ void dtt_eval_kernel(bh_tree src, bh_target_tree tgt,
                                double theta,
                                const double *__restrict__ flux,    // pol   (FIELD=false)
                                const double *__restrict__ norms,   // ionic (FIELD=true)
                                const double *__restrict__ phi_sup,
                                const double *__restrict__ area,
                                double inv_4pi,
                                double *__restrict__ partial) {
  // ---- this block's target leaf cell ----
  int A   = tgt.d_leaf_nodes[blockIdx.x];
  int a0  = tgt.d_first[A];
  int cnt = tgt.d_last[A] - a0 + 1;

  // target cell center + radius (half max-extent), same metric as the source side
  double Tmnx = tgt.d_min[3*A],   Tmxx = tgt.d_max[3*A];
  double Tmny = tgt.d_min[3*A+1], Tmxy = tgt.d_max[3*A+1];
  double Tmnz = tgt.d_min[3*A+2], Tmxz = tgt.d_max[3*A+2];
  double Tcx = 0.5*(Tmnx+Tmxx), Tcy = 0.5*(Tmny+Tmxy), Tcz = 0.5*(Tmnz+Tmxz);
  // enclosing-sphere radius = half the AABB diagonal (NOT half the max-extent:
  // that under-bounds anisotropic cells and lets corner targets fall inside the
  // expansion's inaccurate region).
  double Tex = Tmxx-Tmnx, Tey = Tmxy-Tmny, Tez = Tmxz-Tmnz;
  double rT  = 0.5 * sqrt(Tex*Tex + Tey*Tey + Tez*Tez);
  double theta2 = theta * theta;

  // ---- this thread's target point (one per thread; cnt <= leaf_size <= blockDim) ----
  bool active = (threadIdx.x < cnt);
  int  my = active ? (a0 + threadIdx.x) : -1;
  double px = 0.0, py = 0.0, pz = 0.0;
  if (active) { px = tgt.d_pos[3*my]; py = tgt.d_pos[3*my+1]; pz = tgt.d_pos[3*my+2]; }
  double phi = 0.0, gx = 0.0, gy = 0.0, gz = 0.0;

  // ---- cooperative breadth-first descent of the source tree (Bonsai sec. 2.2) ----
  // The whole block walks the source tree ONCE. Each step, DTT_BDIM threads each
  // take one source node from the current BFS frontier, test the (block-uniform)
  // two-sided MAC + leaf test, and a block prefix-sum compacts the outcomes into
  // shared lists: admissible -> s_m2p, terminal leaf -> s_p2p, internal -> s_next
  // (its two children). The lists are drained (evaluated) when nearly full and at
  // the end. Control is shared by the block; only the drain payload is per-point.
  __shared__ int s_bufA[DTT_FRONT_CAP];
  __shared__ int s_bufB[DTT_FRONT_CAP];
  __shared__ int s_m2p[DTT_LIST_CAP];
  __shared__ int s_p2p[DTT_LIST_CAP];
  __shared__ int n_curr, n_next, n_m2p, n_p2p;
  __shared__ typename cub::BlockScan<int, DTT_BDIM>::TempStorage scan_tmp;

  int *s_curr = s_bufA;
  int *s_next = s_bufB;

  if (threadIdx.x == 0) { s_curr[0] = 0; n_curr = 1; n_next = 0; n_m2p = 0; n_p2p = 0; }
  __syncthreads();

  while (n_curr > 0) {
    int nc = n_curr;
    for (int base = 0; base < nc; base += DTT_BDIM) {
      int idx = base + threadIdx.x;
      int B = (idx < nc) ? s_curr[idx] : -1;

      // classify this source node against the target cell (block-uniform per node)
      int admit = 0, leaf = 0, open = 0, Lc = -1, Rc = -1;
      if (B >= 0) {
        double Bmnx = src.d_min[3*B],   Bmxx = src.d_max[3*B];
        double Bmny = src.d_min[3*B+1], Bmxy = src.d_max[3*B+1];
        double Bmnz = src.d_min[3*B+2], Bmxz = src.d_max[3*B+2];
        double Bcx = 0.5*(Bmnx+Bmxx), Bcy = 0.5*(Bmny+Bmxy), Bcz = 0.5*(Bmnz+Bmxz);
        double Bex = Bmxx-Bmnx, Bey = Bmxy-Bmny, Bez = Bmxz-Bmnz;
        double rB  = 0.5 * sqrt(Bex*Bex + Bey*Bey + Bez*Bez);   // half-diagonal (see rT)
        double Rx = Tcx - Bcx, Ry = Tcy - Bcy, Rz = Tcz - Bcz;
        double R2 = Rx*Rx + Ry*Ry + Rz*Rz;
        double sumr = rT + rB;
        // two-sided MAC: (rT + rB)/R < theta  <=>  (rT+rB)^2 < theta^2 * R^2
        if (R2 > 0.0 && sumr*sumr < theta2 * R2) {
          admit = 1;
        } else {
          int  bcnt  = src.d_last[B] - src.d_first[B] + 1;
          bool bleaf = (B >= src.N - 1) || (bcnt <= src.leaf_size);
          if (bleaf) { leaf = 1; }
          else       { open = 1; Lc = src.d_left[B]; Rc = src.d_right[B]; }
        }
      }

      // compact the three outcome streams via block prefix-sums; cub requires a
      // __syncthreads between reuses of the same TempStorage.
      int p_a, t_a, p_l, t_l, p_o, t_o;
      cub::BlockScan<int, DTT_BDIM>(scan_tmp).ExclusiveSum(admit, p_a, t_a);
      __syncthreads();
      if (admit) s_m2p[n_m2p + p_a] = B;
      cub::BlockScan<int, DTT_BDIM>(scan_tmp).ExclusiveSum(leaf, p_l, t_l);
      __syncthreads();
      if (leaf) s_p2p[n_p2p + p_l] = B;
      cub::BlockScan<int, DTT_BDIM>(scan_tmp).ExclusiveSum(open, p_o, t_o);
      __syncthreads();
      if (open) {
        int w = n_next + 2*p_o;
        if (w + 1 < DTT_FRONT_CAP) { s_next[w] = Lc; s_next[w + 1] = Rc; }
      }

      if (threadIdx.x == 0) {
        n_m2p += t_a;
        n_p2p += t_l;
        int nn = n_next + 2*t_o;
        if (nn > DTT_FRONT_CAP) {
          printf("[dtt] BFS frontier overflow (%d > %d) at block %d: Stage-1 shared-frontier "
                 "limit -- this block's result is incomplete. Raise DTT_FRONT_CAP or move to a "
                 "global-spill frontier.\n", nn, DTT_FRONT_CAP, blockIdx.x);
          nn = DTT_FRONT_CAP;
        }
        n_next = nn;
      }
      __syncthreads();

      // drain the interaction lists before they can overflow (Bonsai: evaluate
      // once a list exceeds the block width), then reset.
      if (n_m2p > DTT_LIST_CAP - DTT_BDIM) {
        dtt_drain_m2p<FIELD, P>(src, s_m2p, n_m2p, active, px, py, pz, phi, gx, gy, gz);
        __syncthreads();
        if (threadIdx.x == 0) n_m2p = 0;
        __syncthreads();
      }
      if (n_p2p > DTT_LIST_CAP - DTT_BDIM) {
        dtt_drain_p2p<FIELD>(src, s_p2p, n_p2p, active, px, py, pz, phi, gx, gy, gz);
        __syncthreads();
        if (threadIdx.x == 0) n_p2p = 0;
        __syncthreads();
      }
    }

    // advance one BFS level: swap the current/next frontier buffers (the pointer
    // swap is identical on every thread, so the block stays consistent).
    int *tmp = s_curr; s_curr = s_next; s_next = tmp;
    if (threadIdx.x == 0) { n_curr = n_next; n_next = 0; }
    __syncthreads();
  }

  // ---- drain the tails ----
  dtt_drain_m2p<FIELD, P>(src, s_m2p, n_m2p, active, px, py, pz, phi, gx, gy, gz);
  __syncthreads();
  dtt_drain_p2p<FIELD>(src, s_p2p, n_p2p, active, px, py, pz, phi, gx, gy, gz);
  __syncthreads();

  // ---- apply the term weight and scatter back to original point order ----
  if (active) {
    int orig = tgt.d_orig[my];
    if constexpr (FIELD) {
      double dot    = gx*norms[3*orig] + gy*norms[3*orig+1] + gz*norms[3*orig+2];
      double factor = phi_sup[orig] * inv_4pi * area[orig / 3] / 3.0;
      partial[orig] = dot * factor;
    } else {
      partial[orig] = flux[orig] * phi;
    }
  }
}

// Dispatch a P-templated kernel on the runtime multipole order t->p (clamped to
// [1, BH_MAX_P] at build time, so the default arm covers BH_MAX_P).
#define BH_LAUNCH_BY_ORDER(KERNEL, GRID, BLOCK, ...)                      \
  do {                                                                    \
    switch (t->p) {                                                       \
      case 1: KERNEL<1><<<(GRID), (BLOCK)>>>(__VA_ARGS__); break;         \
      case 2: KERNEL<2><<<(GRID), (BLOCK)>>>(__VA_ARGS__); break;         \
      case 3: KERNEL<3><<<(GRID), (BLOCK)>>>(__VA_ARGS__); break;         \
      case 4: KERNEL<4><<<(GRID), (BLOCK)>>>(__VA_ARGS__); break;         \
      case 5: KERNEL<5><<<(GRID), (BLOCK)>>>(__VA_ARGS__); break;         \
      case 6: KERNEL<6><<<(GRID), (BLOCK)>>>(__VA_ARGS__); break;         \
      case 7: KERNEL<7><<<(GRID), (BLOCK)>>>(__VA_ARGS__); break;         \
      case 8: KERNEL<8><<<(GRID), (BLOCK)>>>(__VA_ARGS__); break;         \
      case 9: KERNEL<9><<<(GRID), (BLOCK)>>>(__VA_ARGS__); break;         \
      default: KERNEL<10><<<(GRID), (BLOCK)>>>(__VA_ARGS__); break;       \
    }                                                                     \
  } while (0)

// Dispatch the DTT kernel on (compile-time FIELD, runtime src->p). One block per
// target leaf; BLOCK must be >= the target leaf_size so every leaf point maps to
// a distinct thread.
#define BH_LAUNCH_DTT(FIELD, GRID, BLOCK, ...)                                     \
  do {                                                                             \
    switch (src->p) {                                                              \
      case 1: dtt_eval_kernel<FIELD,1><<<(GRID),(BLOCK)>>>(__VA_ARGS__); break;    \
      case 2: dtt_eval_kernel<FIELD,2><<<(GRID),(BLOCK)>>>(__VA_ARGS__); break;    \
      case 3: dtt_eval_kernel<FIELD,3><<<(GRID),(BLOCK)>>>(__VA_ARGS__); break;    \
      case 4: dtt_eval_kernel<FIELD,4><<<(GRID),(BLOCK)>>>(__VA_ARGS__); break;    \
      case 5: dtt_eval_kernel<FIELD,5><<<(GRID),(BLOCK)>>>(__VA_ARGS__); break;    \
      case 6: dtt_eval_kernel<FIELD,6><<<(GRID),(BLOCK)>>>(__VA_ARGS__); break;    \
      case 7: dtt_eval_kernel<FIELD,7><<<(GRID),(BLOCK)>>>(__VA_ARGS__); break;    \
      case 8: dtt_eval_kernel<FIELD,8><<<(GRID),(BLOCK)>>>(__VA_ARGS__); break;    \
      case 9: dtt_eval_kernel<FIELD,9><<<(GRID),(BLOCK)>>>(__VA_ARGS__); break;    \
      default: dtt_eval_kernel<FIELD,10><<<(GRID),(BLOCK)>>>(__VA_ARGS__); break;  \
    }                                                                              \
  } while (0)

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

// ====================================================================
//  Target tree build / free (geometry-only; reuses the atom-tree pipeline).
// ====================================================================
static void build_target_tree(int num_pts, const double *d_pts, int leaf_size,
                              bh_target_tree *tt) {
  tt->N         = num_pts;
  tt->n_nodes   = (num_pts > 0) ? (2 * num_pts - 1) : 0;
  tt->leaf_size = (leaf_size < 1) ? 1 : leaf_size;
  tt->n_leaves  = 0;
  tt->d_pos = tt->d_min = tt->d_max = nullptr;
  tt->d_orig = tt->d_left = tt->d_right = tt->d_parent = nullptr;
  tt->d_first = tt->d_last = tt->d_leaf_nodes = nullptr;
  if (num_pts <= 0) return;

  const int N  = num_pts;
  const int nn = tt->n_nodes;

  CUDA_CHECK(cudaMalloc(&tt->d_pos,    3 * N  * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&tt->d_orig,       N  * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&tt->d_min,    3 * nn * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&tt->d_max,    3 * nn * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&tt->d_left,       nn * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&tt->d_right,      nn * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&tt->d_parent,     nn * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&tt->d_first,      nn * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&tt->d_last,       nn * sizeof(int)));

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
  bbox_kernel<<<bb_blocks, tpb>>>(N, d_pts, d_gmin, d_gmax);
  CUDA_CHECK(cudaGetLastError());

  double h_min[3], h_max[3];
  CUDA_CHECK(cudaMemcpy(h_min, d_gmin, 3*sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_max, d_gmax, 3*sizeof(double), cudaMemcpyDeviceToHost));
  cudaFree(d_gmin); cudaFree(d_gmax);

  double sx = h_max[0]-h_min[0]; if (sx <= 0) sx = 1.0;
  double sy = h_max[1]-h_min[1]; if (sy <= 0) sy = 1.0;
  double sz = h_max[2]-h_min[2]; if (sz <= 0) sz = 1.0;

  // 2. Morton codes + radix sort (keep the permutation)
  uint64_t *d_codes, *d_codes_sorted;
  int *d_idx, *d_idx_sorted;
  CUDA_CHECK(cudaMalloc(&d_codes,        N * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&d_codes_sorted, N * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&d_idx,          N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_idx_sorted,   N * sizeof(int)));

  int mt_blocks = (N + tpb - 1) / tpb;
  morton_kernel<<<mt_blocks, tpb>>>(N, d_pts, h_min[0], h_min[1], h_min[2],
                                    1.0/sx, 1.0/sy, 1.0/sz, d_codes, d_idx);
  CUDA_CHECK(cudaGetLastError());

  void *d_tmp = nullptr; size_t tmp_bytes = 0;
  cub::DeviceRadixSort::SortPairs(d_tmp, tmp_bytes, d_codes, d_codes_sorted,
                                  d_idx, d_idx_sorted, N);
  CUDA_CHECK(cudaMalloc(&d_tmp, tmp_bytes));
  cub::DeviceRadixSort::SortPairs(d_tmp, tmp_bytes, d_codes, d_codes_sorted,
                                  d_idx, d_idx_sorted, N);
  cudaFree(d_tmp); cudaFree(d_codes); cudaFree(d_idx);

  // 3. reorder into Morton order (geometry only) + leaf init + permutation
  reorder_geom_kernel<<<mt_blocks, tpb>>>(N, d_idx_sorted, d_pts,
                                          tt->d_pos, tt->d_orig,
                                          tt->d_min, tt->d_max,
                                          tt->d_parent, tt->d_first, tt->d_last);
  CUDA_CHECK(cudaGetLastError());
  cudaFree(d_idx_sorted);

  if (N > 1) {
    // 4. internal nodes + bottom-up AABB
    int in_blocks = (N - 1 + tpb - 1) / tpb;
    build_internal_kernel<<<in_blocks, tpb>>>(N, d_codes_sorted,
                                              tt->d_left, tt->d_right, tt->d_parent,
                                              tt->d_first, tt->d_last);
    CUDA_CHECK(cudaGetLastError());

    int *d_flags;
    CUDA_CHECK(cudaMalloc(&d_flags, nn * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_flags, 0, nn * sizeof(int)));
    summarize_aabb_kernel<<<mt_blocks, tpb>>>(N, tt->d_left, tt->d_right, tt->d_parent,
                                              tt->d_min, tt->d_max, d_flags);
    CUDA_CHECK(cudaGetLastError());
    cudaFree(d_flags);
  }
  cudaFree(d_codes_sorted);

  // 5. mark + compact the leaf-size frontier -> d_leaf_nodes
  int *d_frontier;
  CUDA_CHECK(cudaMalloc(&d_frontier, nn * sizeof(int)));
  int fr_blocks = (nn + tpb - 1) / tpb;
  mark_frontier_kernel<<<fr_blocks, tpb>>>(nn, N, tt->leaf_size,
                                           tt->d_first, tt->d_last, tt->d_parent,
                                           d_frontier);
  CUDA_CHECK(cudaGetLastError());

  int *d_ids;
  CUDA_CHECK(cudaMalloc(&d_ids, nn * sizeof(int)));
  iota_kernel<<<fr_blocks, tpb>>>(nn, d_ids);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMalloc(&tt->d_leaf_nodes, nn * sizeof(int)));
  int *d_num;
  CUDA_CHECK(cudaMalloc(&d_num, sizeof(int)));
  void *d_sel_tmp = nullptr; size_t sel_bytes = 0;
  cub::DeviceSelect::Flagged(d_sel_tmp, sel_bytes, d_ids, d_frontier,
                             tt->d_leaf_nodes, d_num, nn);
  CUDA_CHECK(cudaMalloc(&d_sel_tmp, sel_bytes));
  cub::DeviceSelect::Flagged(d_sel_tmp, sel_bytes, d_ids, d_frontier,
                             tt->d_leaf_nodes, d_num, nn);
  CUDA_CHECK(cudaMemcpy(&tt->n_leaves, d_num, sizeof(int), cudaMemcpyDeviceToHost));
  cudaFree(d_sel_tmp); cudaFree(d_num); cudaFree(d_frontier); cudaFree(d_ids);

  CUDA_CHECK(cudaDeviceSynchronize());
}

static void free_target_tree(bh_target_tree *tt) {
  if (!tt) return;
  cudaFree(tt->d_pos);    cudaFree(tt->d_orig);
  cudaFree(tt->d_min);    cudaFree(tt->d_max);
  cudaFree(tt->d_left);   cudaFree(tt->d_right);  cudaFree(tt->d_parent);
  cudaFree(tt->d_first);  cudaFree(tt->d_last);
  cudaFree(tt->d_leaf_nodes);
}

// ========================== public C API ==========================
extern "C" {

void bh_build_atom_tree(int num_atoms,
                        const double *d_atoms,
                        const double *d_charges,
                        double theta,
                        int p,
                        int leaf_size,
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
  t->leaf_size = (leaf_size < 1) ? 1 : leaf_size;
  t->theta   = theta;
  *out = t;
  if (num_atoms <= 0) {
    t->d_pos = t->d_q = t->d_min = t->d_max = t->d_moments = nullptr;
    t->d_left = t->d_right = t->d_parent = nullptr;
    t->d_first = t->d_last = nullptr;
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
  CUDA_CHECK(cudaMalloc(&t->d_first,       nn * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&t->d_last,        nn * sizeof(int)));
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
                                     t->d_moments, t->d_parent,
                                     t->d_first, t->d_last);
  CUDA_CHECK(cudaGetLastError());
  cudaFree(d_idx_sorted);

  if (N > 1) {
    // 4. Build internal nodes (Karras)
    int in_blocks = (N - 1 + tpb - 1) / tpb;
    build_internal_kernel<<<in_blocks, tpb>>>(N, d_codes_sorted,
                                              t->d_left, t->d_right, t->d_parent,
                                              t->d_first, t->d_last);
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
  cudaFree(t->d_first);   cudaFree(t->d_last);
  cudaFree(t->d_moments);
  delete t;
}

double bh_coulombic_energy(bh_tree *t) {
  if (!t || t->N < 2) return 0.0;
  double *d_partial;
  CUDA_CHECK(cudaMalloc(&d_partial, t->N * sizeof(double)));
  int tpb = 256, blocks = (t->N + tpb - 1) / tpb;
  BH_LAUNCH_BY_ORDER(coulomb_kernel, blocks, tpb, *t, d_partial);
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
  BH_LAUNCH_BY_ORDER(potential_kernel, blocks, tpb, *t, num_pts, d_V, d_flux, d_partial);
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
  BH_LAUNCH_BY_ORDER(field_kernel, blocks, tpb, *t, num_tri_verts, d_vert, d_norms, d_phi,
                     d_area, inv_4pi, d_partial);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  double r = reduce_sum_host(d_partial, num_tri_verts);
  cudaFree(d_vert); cudaFree(d_norms); cudaFree(d_phi); cudaFree(d_area); cudaFree(d_partial);
  return r;
}

// ---------------------- DTT (dual-tree-traversal) path ----------------------

double bh_polarization_energy_dtt(bh_tree *src, int num_pts,
                                  const double *h_V, const double *h_flux) {
  if (!src || src->N == 0 || num_pts == 0) return 0.0;

  double *d_V, *d_flux, *d_partial;
  CUDA_CHECK(cudaMalloc(&d_V,       num_pts * 3 * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_flux,    num_pts     * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_partial, num_pts     * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_V,    h_V,    num_pts*3*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_flux, h_flux, num_pts  *sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_partial, 0, num_pts * sizeof(double)));  // any uncovered target -> 0

  // Target leaf cells are capped at 256 pts so a single block can serve a leaf.
  int tleaf = src->leaf_size; if (tleaf > 256) tleaf = 256; if (tleaf < 1) tleaf = 1;
  bh_target_tree tt;
  build_target_tree(num_pts, d_V, tleaf, &tt);

  int block = DTT_BDIM;   // fixed cooperative width (must match cub::BlockScan<DTT_BDIM>)
  int grid  = tt.n_leaves;
  if (grid > 0) {
    BH_LAUNCH_DTT(false, grid, block, *src, tt, src->theta,
                  d_flux, nullptr, nullptr, nullptr, 0.0, d_partial);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  double r = reduce_sum_host(d_partial, num_pts);
  free_target_tree(&tt);
  cudaFree(d_V); cudaFree(d_flux); cudaFree(d_partial);
  return r;
}

double bh_ionic_energy_dtt(bh_tree *src, int num_tri_verts,
                           const double *h_vert, const double *h_norms,
                           const double *h_phi_sup, const double *h_area,
                           double inv_4pi) {
  if (!src || src->N == 0 || num_tri_verts == 0) return 0.0;
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
  CUDA_CHECK(cudaMemset(d_partial, 0, num_tri_verts * sizeof(double)));  // any uncovered target -> 0

  int tleaf = src->leaf_size; if (tleaf > 256) tleaf = 256; if (tleaf < 1) tleaf = 1;
  bh_target_tree tt;
  build_target_tree(num_tri_verts, d_vert, tleaf, &tt);

  int block = DTT_BDIM;   // fixed cooperative width (must match cub::BlockScan<DTT_BDIM>)
  int grid  = tt.n_leaves;
  if (grid > 0) {
    BH_LAUNCH_DTT(true, grid, block, *src, tt, src->theta,
                  nullptr, d_norms, d_phi, d_area, inv_4pi, d_partial);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  double r = reduce_sum_host(d_partial, num_tri_verts);
  free_target_tree(&tt);
  cudaFree(d_vert); cudaFree(d_norms); cudaFree(d_phi); cudaFree(d_area); cudaFree(d_partial);
  return r;
}

} // extern "C"

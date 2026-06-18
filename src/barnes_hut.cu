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
#include <utility>

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
//  Compile-time multi-index machinery (constexpr twins of the runtime
//  __constant__ tables).  Used only by the templated traversal/evaluator
//  hot path: because every index folds to a literal after unrolling, the
//  per-thread Taylor buffer can be scalar-replaced into registers instead
//  of spilling to local memory.  Enumeration order matches
//  init_constant_tables(): ascending |k|, then lex (kx, ky) within a shell.
// ====================================================================
// # multi-indices with |k| < n  (== comp_of_h(n-1), with comp_of_h(-1)=0)
__host__ __device__ constexpr int mi_base(int n) { return n * (n + 1) * (n + 2) / 6; }
// smallest n with comp_of_h(n) > s  ==> the order |k| of slot s
__host__ __device__ constexpr int mi_order(int s) {
  int n = 0;
  while (comp_of_h(n) <= s) ++n;
  return n;
}
__host__ __device__ constexpr int mi_kx_of(int s) {
  int n = mi_order(s), w = s - mi_base(n), kx = 0;
  while (w >= (n - kx + 1)) { w -= (n - kx + 1); ++kx; }
  return kx;
}
__host__ __device__ constexpr int mi_ky_of(int s) {
  int n = mi_order(s), w = s - mi_base(n), kx = 0;
  while (w >= (n - kx + 1)) { w -= (n - kx + 1); ++kx; }
  return w;                                   // leftover within the shell == ky
}
__host__ __device__ constexpr int mi_kz_of(int s) {
  return mi_order(s) - mi_kx_of(s) - mi_ky_of(s);
}
// slot index of multi-index (kx, ky, kz)
__host__ __device__ constexpr int mi_slot(int kx, int ky, int kz) {
  int n = kx + ky + kz;
  return mi_base(n) + kx * (n + 1) - (kx * (kx - 1)) / 2 + ky;
}
__host__ __device__ constexpr double mi_cfact(int i) {
  double f = 1.0;
  for (int j = 2; j <= i; ++j) f *= (double)j;
  return f;
}
// 1 / (kx! ky! kz!) for slot s
__host__ __device__ constexpr double mi_inv_fact(int s) {
  return 1.0 / (mi_cfact(mi_kx_of(s)) * mi_cfact(mi_ky_of(s)) * mi_cfact(mi_kz_of(s)));
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

  // per-node multi-index moments m^k, k indexed in c_mi_kx/ky/kz order
  double *d_moments;  // n_nodes * comp
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
//  Tree traversal + multipole evaluation (device helpers, called by the
//  target-side kernels below).  Multi-index Cartesian Taylor.
// ====================================================================
//
// Fill T[0 .. comp_of_h(MAXO)-1] with Cartesian Taylor coefficients
//   T^k(R) = d^k(1/r),   r = sqrt(R.R)
// via the all-axis Lindsay-Krasny recurrence
//   r^2 |k| T^k = -(2|k|-1) sum_i k_i R_i T^{k-e_i} - (|k|-1) sum_i k_i(k_i-1) T^{k-2e_i}.
// Slot S and all predecessor slots are compile-time constants (constexpr mi_*),
// so T[] can be scalar-replaced into registers instead of spilling.
template<int S>
__device__ __forceinline__ void fill_T_slot(double Rx, double Ry, double Rz,
                                            double inv_r, double inv_r2, double inv_r3,
                                            double *T) {
  constexpr int kx = mi_kx_of(S), ky = mi_ky_of(S), kz = mi_kz_of(S);
  constexpr int n  = kx + ky + kz;
  if constexpr (n == 0) {
    T[S] = inv_r;                                       // T^(0,0,0) = 1/r
  } else if constexpr (n == 1) {
    T[S] = (kx ? -Rx : (ky ? -Ry : -Rz)) * inv_r3;      // T^(e_i) = -R_i / r^3
  } else {
    // fold the 1/|k| normalization into the constants at compile time, so the
    // per-slot result is just (c1*s1 + c2*s2)*inv_r2 -- one fewer FP64 multiply.
    constexpr double inv_n = 1.0 / (double)n;
    constexpr double c1    = -(2.0 * n - 1.0) * inv_n;  // -(2|k|-1)/|k|
    constexpr double c2    = -(double)(n - 1) * inv_n;  // -(|k|-1)/|k|
    double s1 = 0.0, s2 = 0.0;
    if constexpr (kx >= 1) {
      s1 += kx * Rx * T[mi_slot(kx - 1, ky, kz)];
      if constexpr (kx >= 2) s2 += kx * (kx - 1) * T[mi_slot(kx - 2, ky, kz)];
    }
    if constexpr (ky >= 1) {
      s1 += ky * Ry * T[mi_slot(kx, ky - 1, kz)];
      if constexpr (ky >= 2) s2 += ky * (ky - 1) * T[mi_slot(kx, ky - 2, kz)];
    }
    if constexpr (kz >= 1) {
      s1 += kz * Rz * T[mi_slot(kx, ky, kz - 1)];
      if constexpr (kz >= 2) s2 += kz * (kz - 1) * T[mi_slot(kx, ky, kz - 2)];
    }
    T[S] = (c1 * s1 + c2 * s2) * inv_r2;
  }
}

template<int... S>
__device__ __forceinline__ void compute_T_impl(double Rx, double Ry, double Rz,
                                               double inv_r, double inv_r2, double inv_r3,
                                               double *T, std::integer_sequence<int, S...>) {
  // comma fold over ascending slot S: the comma operator sequences left-to-right,
  // so every predecessor T^{k-e_i} is written before it is read.
  (fill_T_slot<S>(Rx, Ry, Rz, inv_r, inv_r2, inv_r3, T), ...);
}

template<int MAXO>
__device__ __forceinline__ void compute_T_t(double Rx, double Ry, double Rz,
                                            double inv_r2, double inv_r, double *T) {
  double inv_r3 = inv_r * inv_r2;
  compute_T_impl(Rx, Ry, Rz, inv_r, inv_r2, inv_r3, T,
                 std::make_integer_sequence<int, comp_of_h(MAXO)>{});
}

// Potential: phi += sum_{|k|<=p} (-1)^|k| / k! * T^k * m^k
template<int S>
__device__ __forceinline__ double pot_term(const double *m, const double *T) {
  constexpr int n = mi_kx_of(S) + mi_ky_of(S) + mi_kz_of(S);
  constexpr double sign  = (n & 1) ? -1.0 : 1.0;
  constexpr double inv_f = mi_inv_fact(S);
  return sign * inv_f * T[S] * m[S];
}
template<int... S>
__device__ __forceinline__ double mp_potential_eval_impl(const double *m, const double *T,
                                                         std::integer_sequence<int, S...>) {
  return (... + pot_term<S>(m, T));            // left fold: matches s=0..comp-1 order
}
template<int COMP>
__device__ __forceinline__ double mp_potential_eval_t(const double *m, const double *T) {
  return mp_potential_eval_impl(m, T, std::make_integer_sequence<int, COMP>{});
}

// Field: g_i += -sum_{|k|<=p} (-1)^|k| / k! * T^(k+e_i) * m^k
template<int S>
__device__ __forceinline__ void field_term(const double *m, const double *T,
                                           double &gx, double &gy, double &gz) {
  constexpr int kx = mi_kx_of(S), ky = mi_ky_of(S), kz = mi_kz_of(S);
  constexpr int n  = kx + ky + kz;
  constexpr double sign  = (n & 1) ? -1.0 : 1.0;
  constexpr double inv_f = mi_inv_fact(S);
  double common = sign * inv_f * m[S];
  gx -= common * T[mi_slot(kx + 1, ky, kz)];
  gy -= common * T[mi_slot(kx, ky + 1, kz)];
  gz -= common * T[mi_slot(kx, ky, kz + 1)];
}
template<int... S>
__device__ __forceinline__ void mp_field_eval_impl(const double *m, const double *T,
                                                   double &gx, double &gy, double &gz,
                                                   std::integer_sequence<int, S...>) {
  (field_term<S>(m, T, gx, gy, gz), ...);
}
template<int COMP>
__device__ __forceinline__ void mp_field_eval_t(const double *m, const double *T,
                                               double &gx, double &gy, double &gz) {
  mp_field_eval_impl(m, T, gx, gy, gz, std::make_integer_sequence<int, COMP>{});
}

template<bool FIELD, int P>
__device__ void traverse(double tx, double ty, double tz, int self_atom,
                         const bh_tree t,
                         double &out_phi, double &gx, double &gy, double &gz) {
  out_phi = 0.0; gx = gy = gz = 0.0;
  const double theta2 = t.theta * t.theta;
  const int leaf_size = t.leaf_size;

  constexpr int MAX_T = FIELD ? (P + 1) : P;   // Taylor order needed
  constexpr int COMP  = comp_of_h(P);          // moments per node

  // Per-thread Taylor buffer; compile-time sized so it can be register-resident
  // (fully for low P; at P=6 the order-7 buffer still exceeds the register
  // budget and partially spills).
  double T_buf[comp_of_h(MAX_T)];

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
      // far enough: particle-cluster multipole interaction.
      // rsqrt(dist2) is ~17 FP64 ops vs ~45 for 1.0/dist2 then sqrt().
      double inv_r  = rsqrt(dist2);
      double inv_r2 = inv_r * inv_r;
      compute_T_t<MAX_T>(Rx, Ry, Rz, inv_r2, inv_r, T_buf);
      const double *m = t.d_moments + (size_t)node * COMP;
      if constexpr (FIELD) {
        mp_field_eval_t<COMP>(m, T_buf, gx, gy, gz);
      } else {
        out_phi += mp_potential_eval_t<COMP>(m, T_buf);
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
//  DTT evaluation kernel: one CUDA block per target leaf cell. Every thread
//  descends the source (atom) tree with its own private stack (the descent is
//  control-flow identical across the block), testing a two-sided
//  MAC once per source node (source cell vs the whole target leaf). Admissible
//  source cells are resolved by M2P into each target point in the leaf;
//  terminal source clusters by direct P2P. Far field is still M2P (no M2L yet).
//  Targets != sources, so no self-exclusion. Weights and the final scatter use
//  the original (caller) point order via tgt.d_orig.
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
  constexpr int MAX_T = FIELD ? (P + 1) : P;   // Taylor order needed
  constexpr int COMP  = comp_of_h(P);          // moments per source node

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

  // ---- per-thread DFS over the source tree ----
  // The descent is control-flow identical on every thread in the block: which
  // node is popped, the MAC/leaf tests, and the push/pop all depend only on the
  // target cell + source node, never on threadIdx. Only the M2P/P2P payload is
  // per-point. So instead of one thread owning the stack and broadcasting the
  // popped node through shared memory (which needs a barrier each side), every
  // thread keeps its own stack and walks the same descent independently: no
  // shared state, no __syncthreads. The duplicated control is a few int/FP ops
  // per node, far cheaper than the two block barriers it replaces.
  // STACK_CAP matches the old shared cap (binary DFS, push-2/pop-1 => depth-bounded).
  constexpr int STACK_CAP = 96;
  int stack[STACK_CAP];               // dynamic index => local memory, L1-cached
  int sp = 0;
  stack[sp++] = 0;                    // root

  for (;;) {
    if (sp == 0) break;               // stack empty: uniform across the block
    int B = stack[--sp];

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
      if (active) {
        double Rxp = px - Bcx, Ryp = py - Bcy, Rzp = pz - Bcz;
        double d2  = Rxp*Rxp + Ryp*Ryp + Rzp*Rzp;
        double inv_r  = rsqrt(d2);
        double inv_r2 = inv_r * inv_r;
        double T_buf[comp_of_h(MAX_T)];
        compute_T_t<MAX_T>(Rxp, Ryp, Rzp, inv_r2, inv_r, T_buf);
        const double *m = src.d_moments + (size_t)B * COMP;
        if constexpr (FIELD) mp_field_eval_t<COMP>(m, T_buf, gx, gy, gz);
        else                 phi += mp_potential_eval_t<COMP>(m, T_buf);
      }
      continue;
    }

    int  bcnt  = src.d_last[B] - src.d_first[B] + 1;
    bool bleaf = (B >= src.N - 1) || (bcnt <= src.leaf_size);
    if (bleaf) {
      if (active) {
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
      continue;
    }

    stack[sp++] = src.d_left[B];
    stack[sp++] = src.d_right[B];
  }

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
      default: KERNEL<6><<<(GRID), (BLOCK)>>>(__VA_ARGS__); break;        \
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
      default: dtt_eval_kernel<FIELD,6><<<(GRID),(BLOCK)>>>(__VA_ARGS__); break;   \
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

// Round the target block size up to a power of two >= leaf_size (cap 256, floor 32),
// so each leaf point owns a distinct thread within the block.
static int dtt_block_for(int leaf_size) {
  int b = 32;
  while (b < leaf_size) b <<= 1;
  if (b > 256) b = 256;
  return b;
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

  int block = dtt_block_for(tt.leaf_size);
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

  int block = dtt_block_for(tt.leaf_size);
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

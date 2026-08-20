/*
 *  FMM with complex spherical-harmonic moments (CGR'99 operator chain).
 *
 *  Multipole representation (Cheng, Greengard & Rokhlin 1999, JCP 155:468):
 *    - Per-node moments are complex M_n^m (CGR'99 Eq. 4 normalization), stored
 *      for the m>=0 half only (0<=m<=n<=p); M_n^{-m}=conj(M_n^m). Layout: doubles
 *      [re,im] at slot s=sph_slot(n,m). Count comp_sph(p) = (p+1)(p+2) doubles.
 *    - P2M: leaf monopole M_0^0 = q (ρ=0). Upward pass M2M: T_MM (Eq. 13).
 *      Downward pass M2L: T_ML (Eq. 17), L2L: T_LL (Eq. 21). Targets evaluate
 *      their leaf's local expansion by L2P, plus a direct near field (P2P).
 *      The potential uses the irregular/regular solid harmonics contracted with
 *      the moments (Eq. 5/8); the field (E_ion) is the exact gradient of the same
 *      expansion via forward-mode automatic differentiation (the `dual` scalar),
 *      so no separate solid-harmonic gradient recurrence is needed.
 *
 *  Traversal: a single dual-tree pair traversal over the source (atom) tree and
 *  an internally built target tree splits each cell pair into near (P2P) and far
 *  (M2L) work. LBVH topology and the bbox/Morton/build kernels are unchanged.
 */

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cfloat>
#include <cmath>
#include <algorithm>
#include <utility>

#include "fmm.h"   // public C API + (under __CUDACC__) the shared FMM types,
                   // defines, operators, CUDA_CHECK, and small device helpers

// ====================================================================
//  Constant-memory tables (populated once on first build). Kept here (not in
//  fmm.h) because nvcc whole-program mode -- no -rdc, to preserve cross-function
//  device inlining -- rejects an extern __constant__ decl + definition as a
//  redefinition, so the tables and their fact_dev/A_nm accessors stay TU-local:
//    c_fact[k] = k!                      (k up to (2p)! ; doubles)
//    c_A[idx]  = A_n^m = (-1)^n / sqrt((n-m)!(n+m)!)   (CGR'99 Eq. 14)
//                idx = n*n + (m+n), m in [-n, n]
//    c_Ainv[idx] = 1 / A_n^m = (-1)^n * sqrt((n-m)!(n+m)!)   (same layout)
// ====================================================================
__constant__ double c_fact[FMM_FACT_MAX + 1];
// A_n^m table to degree 2p: M2L (Eq.17) indexes A_{j+n}^{m-k} with degree j+n up to 2p.
__constant__ double c_A[(FMM_MAX_2P + 1) * (FMM_MAX_2P + 1)];
// Reciprocal of c_A. M2L (Eq.17) and L2L (Eq.21) divide by an A_n^m whose indices vary with the
// inner (n,m) loop -- i.e. one FP64 division per O(p^4) term, which is brutal at the 3080's 1:64
// FP64 rate. IEEE-754 pins a/b to the correctly-rounded result and a*(1/b) is a different value,
// so nvcc cannot make that substitution on its own (no -use_fast_math here); it has to be done in
// the source. Worth 1.5x on m2l_accumulate_kernel at p=10. Costs one extra (2p+1)^2 doubles of
// __constant__ (3.5 KB at p=10) and shifts results by ~1 ulp per coefficient -- ~1e-12 even if it
// accumulated linearly across the inner sum, far under the ~1e-10 energy tolerance.
__constant__ double c_Ainv[(FMM_MAX_2P + 1) * (FMM_MAX_2P + 1)];

__device__ __forceinline__ double fact_dev(int k) { return c_fact[k]; }
__device__ __forceinline__ double A_nm(int n, int m) { return c_A[n * n + (m + n)]; }
__device__ __forceinline__ double A_inv(int n, int m) { return c_Ainv[n * n + (m + n)]; }

// Host-side one-shot initializer for the constant tables.
static void init_constant_tables() {
  static bool initialized = false;
  if (initialized) return;

  double h_fact[FMM_FACT_MAX + 1];
  h_fact[0] = 1.0;
  for (int k = 1; k <= FMM_FACT_MAX; ++k) h_fact[k] = h_fact[k - 1] * (double)k;

  double h_A   [(FMM_MAX_2P + 1) * (FMM_MAX_2P + 1)];
  double h_Ainv[(FMM_MAX_2P + 1) * (FMM_MAX_2P + 1)];
  for (int n = 0; n <= FMM_MAX_2P; ++n) {
    double sgn = (n & 1) ? -1.0 : 1.0;
    for (int m = -n; m <= n; ++m) {
      double r = sqrt(h_fact[n - m] * h_fact[n + m]);
      // 1/A_n^m is built straight from the factorials rather than as 1.0/h_A[..], so it carries
      // one rounding instead of two. (-1)^n is its own inverse. Largest entry is sqrt((4p)!):
      // 2.9e23 at p=10, 2.7e59 at p=20 -- nowhere near the 1.8e308 double max.
      h_A   [n * n + (m + n)] = sgn / r;
      h_Ainv[n * n + (m + n)] = sgn * r;
    }
  }

  CUDA_CHECK(cudaMemcpyToSymbol(c_fact, h_fact, sizeof(h_fact)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_A,    h_A,    sizeof(h_A)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_Ainv, h_Ainv, sizeof(h_Ainv)));
  initialized = true;
}

// ====================================================================
//  Solid harmonics of a vector (x,y,z) in CGR'99 normalization (Eq. 4):
//    Y_n^m(θ,φ) = sqrt((n-|m|)!/(n+|m|)!) · P_n^{|m|}(cosθ) · e^{imφ}   (CS in P)
//  REGULAR=true  -> R_n^m =  r^n     · Y_n^m   (forming/translating moments)
//  REGULAR=false -> S_n^m =  Y_n^m / r^{n+1}   (irregular; evaluating Φ)
//  Filled for the m>=0 half: out[sph_slot(n,m)], 0<=m<=n<=maxn (m<0 = conj).
//  Templated on scalar T: T=double for the potential, T=dual for the field
//  (the gradient falls out by forward-mode AD, no derivative recurrence).
// ====================================================================
// NMOMCAP / DEGCAP size the per-thread scratch; every caller passes explicit caps (the
// FMM_NMOM_MAX / FMM_MAX_P defaults are just a fallback). M2M and L2L instantiate at order p;
// M2L at order 2p, so only the M2L instantiation carries the larger buffer -- the order-p
// instantiations keep their small stack frames.
template<class T, bool REGULAR, int NMOMCAP = FMM_NMOM_MAX, int DEGCAP = FMM_MAX_P>
__device__ void solid_harmonics(int maxn, T x, T y, T z, cT<T>* out) {
  if constexpr (REGULAR) {
    // Regular solid harmonics R_n^m = r^n Y_n^m (CGR'99 Eq.4 normalization), m>=0 half,
    // built by a POLE-FREE Cartesian recurrence -- every step is polynomial in (x,y,z), so
    // the forward-mode AD (T=dual) stays well-conditioned at r=0. The spherical build
    // (below) divides by r and rxy; its 1/r,1/rxy intermediates are fine for the VALUE but
    // wreck the GRADIENT near the expansion center, which is exactly where the L2P field is
    // evaluated (target points inside their own leaf). This recurrence reproduces the very
    // same quantity the spherical build does (verified order-by-order):
    //   diagonal: R_m^m = -sqrt((2m-1)/(2m)) (x+iy) R_{m-1}^{m-1},   R_0^0 = 1
    //   vertical: R_n^m = (2n-1)/sqrt((n-m)(n+m)) · z · R_{n-1}^m
    //                   - sqrt((n+m-1)(n-m-1)/((n+m)(n-m))) · r^2 · R_{n-2}^m.
    T r2 = x * x + y * y + z * z;
    cT<T> w(x, y);                                   // x + i y
    out[sph_slot(0, 0)] = cT<T>(T(1.0), T(0.0));
    for (int m = 1; m <= maxn; ++m) {                // diagonal seeds R_m^m
      double c = -sqrt((2.0 * m - 1.0) / (2.0 * m));
      out[sph_slot(m, m)] = T(c) * (w * out[sph_slot(m - 1, m - 1)]);
    }
    for (int m = 0; m <= maxn; ++m)                  // climb n at fixed m
      for (int n = m + 1; n <= maxn; ++n) {
        double a = (2.0 * n - 1.0) / sqrt((double)(n - m) * (double)(n + m));
        cT<T> acc = T(a) * (z * out[sph_slot(n - 1, m)]);
        if (n - 2 >= m) {
          double b = sqrt(((double)(n + m - 1) * (double)(n - m - 1))
                        / ((double)(n + m) * (double)(n - m)));
          acc = acc + (T(-b) * (r2 * out[sph_slot(n - 2, m)]));
        }
        out[sph_slot(n, m)] = acc;
      }
    return;
  }

  // Irregular S_n^m = Y_n^m / r^{n+1} (CGR'99 Eq.4). Always evaluated far-field (M2P at a
  // MAC-separated source center; M2L of a cell-cell vector), so the spherical parametrization
  // is well-conditioned. Singular at r=0 by nature -- never evaluated near the origin.
  T r   = tsqrt(x * x + y * y + z * z);
  T rxy = tsqrt(x * x + y * y);
  bool ron  = val(r)   > 0.0;
  bool rxon = val(rxy) > 0.0;
  T ct   = ron  ? (z   / r)   : T(1.0);   // cosθ
  T st   = ron  ? (rxy / r)   : T(0.0);   // sinθ >= 0
  T cphi = rxon ? (x   / rxy) : T(1.0);   // on the z-axis P_n^{m>0}=0 kills e^{imφ}
  T sphi = rxon ? (y   / rxy) : T(0.0);

  // associated Legendre P_n^m(cosθ) (Condon–Shortley), m>=0
  T P[NMOMCAP];
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
  cT<T> eim[DEGCAP + 1];
  eim[0] = cT<T>(T(1.0), T(0.0));
  cT<T> ep(cphi, sphi);
  for (int m = 1; m <= maxn; ++m) eim[m] = eim[m - 1] * ep;

  // radial scaling per degree: 1/r^{n+1}
  T inv_r = ron ? (T(1.0) / r) : T(0.0);
  T rad[DEGCAP + 1];
  rad[0] = inv_r;
  for (int n = 1; n <= maxn; ++n) rad[n] = rad[n - 1] * inv_r;

  for (int n = 0; n <= maxn; ++n)
    for (int m = 0; m <= n; ++m) {
      double c = sqrt(fact_dev(n - m) / fact_dev(n + m));   // sqrt((n-m)!/(n+m)!)
      T a = c * (P[sph_slot(n, m)] * rad[n]);
      out[sph_slot(n, m)] = a * eim[m];
    }
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
//  Kernel 3: reorder atoms into Morton order + init leaf geometry
// ====================================================================
// No moments are written here: radix leaves are below the leaf-size frontier and carry
// no moment slot. The multipole expansion is seeded directly at the frontier boxes by
// src_p2m_kernel, from the atom range each box covers.
__global__ void reorder_kernel(int N,
                               const int *__restrict__ order,
                               const double *__restrict__ pos_in,
                               const double *__restrict__ q_in,
                               double *__restrict__ pos_out,
                               double *__restrict__ q_out,
                               double *__restrict__ nmin,
                               double *__restrict__ nmax,
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
__device__ __forceinline__ int iabs(int a) { return a < 0 ? -a : a; }

// Multipole-to-multipole translation T_MM (CGR'99 Thm 2.3, Eq. 13), shifting a
// child expansion {O} about the child center to the parent center and adding it
// into {M}. (ρ,α,β) are the spherical coords of the shift (child-parent), encoded
// here as the regular solid harmonics R_n^m = ρ^n Y_n^m of the shift vector:
//   M_j^k += Σ_{n=0}^{j} Σ_{m=-n}^{n} O_{j-n}^{k-m} · i^{|k|-|m|-|k-m|}
//                · A_n^m A_{j-n}^{k-m} / A_j^k · R_n^{-m}.
// The phase i^{|k|-|m|-|k-m|} has an even exponent, hence the real (-1)^(e/2).
// ATOMIC=true accumulates with atomicAdd so several threads can share one destination
// (used by the P2M, where a whole block cooperates on one box's expansion). The maths is
// identical either way -- only the store differs.
template<int P, bool ATOMIC>
__device__ void m2m_shift_impl(const double *__restrict__ O,
                               double dx, double dy, double dz,
                               double *M) {
  cmplx R[nmom_sph(P)];
  solid_harmonics<double, true, nmom_sph(P), P>(P, dx, dy, dz, R);   // R_n^m of shift, n<=P

  for (int j = 0; j <= P; ++j)
    for (int k = 0; k <= j; ++k) {                   // store the m>=0 half only
      cmplx acc(0.0, 0.0);
      for (int n = 0; n <= j; ++n)
        for (int m = -n; m <= n; ++m) {
          int jn = j - n, km = k - m;
          if (km < -jn || km > jn) continue;         // O_{jn}^{km} needs |km| <= jn
          int e = k - iabs(m) - iabs(km);            // |k|=k (k>=0); e is even
          double ipow = ((e / 2) & 1) ? -1.0 : 1.0;
          double coef = ipow * A_nm(n, m) * A_nm(jn, km);
          acc = acc + (coef * (get_M(O, jn, km) * get_R(R, n, -m)));
        }
      // 1/A_j^k does not depend on (n,m), so it scales the finished sum once instead of
      // dividing every term -- see the c_Ainv note at the top of this file.
      acc = A_inv(j, k) * acc;
      int s = sph_slot(j, k);
      if (ATOMIC) {
        atomicAdd(&M[2 * s],     acc.re);
        atomicAdd(&M[2 * s + 1], acc.im);
      } else {
        M[2 * s]     += acc.re;
        M[2 * s + 1] += acc.im;
      }
    }
}

template<int P>
__device__ __forceinline__ void m2m_shift(const double *__restrict__ O,
                                          double dx, double dy, double dz,
                                          double *__restrict__ M) {
  m2m_shift_impl<P, false>(O, dx, dy, dz, M);
}

// Multipole-to-local translation T_ML (CGR'99 Thm 2.4, Eq. 17). Converts a source
// multipole {O} about the source center into a local expansion {L} about the target
// center and adds it in. The shift vector t = (source center) - (target center) is
// supplied already encoded as the IRREGULAR solid harmonics S_l^q = Y_l^q(α,β)/ρ^{l+1}
// of t, built by the caller to order 2p (the index j+n reaches 2p):
//   L_j^k += Σ_{n=0}^{p} Σ_{m=-n}^{n} O_n^m · i^{|k-m|-|k|-|m|}
//                · A_n^m A_j^k / ((-1)^n A_{j+n}^{m-k}) · S_{j+n}^{m-k}.
// Phase exponent |k-m|-|k|-|m| is even -> real (-1)^(e/2). S indices stay in range:
// |m-k| <= |m|+|k| <= n+j = j+n, so no bounds guard is needed.
// One output coefficient L_j^k of M2L (the inner (n,m) sum of Eq. 17). Factored out so
// the M2L accumulate kernel (m2l_accumulate_kernel) can reuse the per-coefficient math.
__device__ __forceinline__ cmplx m2l_coeff(int p,
                                           const double *__restrict__ O,
                                           const cmplx  *__restrict__ S,
                                           int j, int k) {
  cmplx acc(0.0, 0.0);
  for (int n = 0; n <= p; ++n)
    for (int m = -n; m <= n; ++m) {
      int jn = j + n, mk = m - k;
      int e  = iabs(mk) - k - iabs(m);           // |k|=k (k>=0); e is even
      double ipow  = ((e / 2) & 1) ? -1.0 : 1.0;
      double sgn_n = (n & 1) ? -1.0 : 1.0;       // 1/(-1)^n = (-1)^n
      double coef  = ipow * sgn_n * A_nm(n, m) * A_inv(jn, mk);
      acc = acc + (coef * (get_M(O, n, m) * get_R(S, jn, mk)));
    }
  // A_j^k does not depend on (n,m) -> applied once here rather than per term.
  return A_nm(j, k) * acc;
}

// Local-to-local translation T_LL (CGR'99 Thm 2.5, Eq. 21). Shifts a parent local
// expansion {O} about the parent center to the child center and adds it into {L}.
// The shift t = (parent center) - (child center) is encoded as the REGULAR solid
// harmonics R_l^q = ρ^l Y_l^q of t (order p suffices, since n-j <= p):
//   L_j^k += Σ_{n=j}^{p} Σ_{m=-n}^{n} O_n^m · i^{|m|-|m-k|-|k|}
//                · A_{n-j}^{m-k} A_j^k / ((-1)^{n+j} A_n^m) · R_{n-j}^{m-k}.
// Phase exponent is even -> real (-1)^(e/2). R_{n-j}^{m-k} requires |m-k| <= n-j (guard).
template<int P>
__device__ void l2l_shift(const double *__restrict__ O,
                          double dx, double dy, double dz,   // t = parent center - child center
                          double *__restrict__ L) {
  cmplx R[nmom_sph(P)];
  solid_harmonics<double, true, nmom_sph(P), P>(P, dx, dy, dz, R);   // R_l^q of t, l<=P

  for (int j = 0; j <= P; ++j)
    for (int k = 0; k <= j; ++k) {                   // store the m>=0 half only
      cmplx acc(0.0, 0.0);
      for (int n = j; n <= P; ++n)
        for (int m = -n; m <= n; ++m) {
          int nj = n - j, mk = m - k;
          if (mk < -nj || mk > nj) continue;         // R_{nj}^{mk} needs |mk| <= nj
          int e  = iabs(m) - iabs(mk) - k;           // |k|=k (k>=0); e is even
          double ipow = ((e / 2) & 1) ? -1.0 : 1.0;
          double sgn  = ((n + j) & 1) ? -1.0 : 1.0;  // 1/(-1)^{n+j}
          double coef = ipow * sgn * A_nm(nj, mk) * A_inv(n, m);
          acc = acc + (coef * (get_M(O, n, m) * get_R(R, nj, mk)));
        }
      // A_j^k does not depend on (n,m) -> applied once here rather than per term.
      acc = A_nm(j, k) * acc;
      int s = sph_slot(j, k);
      L[2 * s]     += acc.re;
      L[2 * s + 1] += acc.im;
    }
}

// NOTE: the old fused summarize_kernel (AABB + dense per-node M2M) was removed when the
// source moments moved onto box slots. Its AABB half lives on as summarize_aabb_kernel
// below; its M2M half is now src_p2m_kernel + src_m2m_kernel, which write through
// d_box_slot. Do not reintroduce a node-indexed moment writer -- d_moments is only
// n_box*comp long, so raw-node-id indexing would run off the end.

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

// Fused L2P: build the regular solid harmonics R_n^m of (x,y,z) by the pole-free Cartesian
// recurrence (same one as solid_harmonics<T,true>) and contract them against the local
// expansion L on the fly, in the same m>=0-folded form (the M_n^0 term + 2*Re over m>0). Walked column-major
// (m outer, n inner) so the live set is O(P) -- a diagonal seed plus a 2-deep vertical window,
// held in NAMED scalars -- never the full O(P^2) array. That keeps everything in registers
// instead of spilling the harmonic buffer to local memory. T=double: potential (returns Phi).
// T=dual: forward-mode AD field (returned Phi carries d/dx,d/dy,d/dz). Compile-time P makes
// every sph_slot() a constant offset under the unrolled loops.
template<int P, class T>
__device__ __forceinline__ T l2p_contract(const double* __restrict__ L, T x, T y, T z) {
  T r2 = x*x + y*y + z*z;
  cT<T> w(x, y);                       // x + i y
  cT<T> diag(T(1.0), T(0.0));          // R_0^0
  T phi = T(0.0);
  #pragma unroll
  for (int m = 0; m <= P; ++m) {
    if (m > 0) {                        // advance diagonal seed R_m^m = c*(x+iy)*R_{m-1}^{m-1}
      double c = -sqrt((2.0*m - 1.0) / (2.0*m));
      diag = T(c) * (w * diag);
    }
    cT<T> g1, g2;                       // named scalars (NOT an indexed array) -> no re-spill
    #pragma unroll
    for (int n = m; n <= P; ++n) {
      cT<T> g;
      if (n == m) g = diag;
      else {                            // vertical climb R_n^m = a*z*R_{n-1}^m - b*r^2*R_{n-2}^m
        double a = (2.0*n - 1.0) / sqrt((double)(n-m)*(double)(n+m));
        g = T(a) * (z * g1);
        if (n - 2 >= m) {
          double b = sqrt(((double)(n+m-1)*(double)(n-m-1))
                        / ((double)(n+m)*(double)(n-m)));
          g = g + (T(-b) * (r2 * g2));
        }
      }
      int s = sph_slot(n, m);           // compile-time under unroll
      if (m == 0) phi = phi + L[2*s] * g.re;
      else        phi = phi + 2.0 * (L[2*s] * g.re - L[2*s+1] * g.im);
      g2 = g1; g1 = g;                  // roll the 2-deep window
    }
  }
  return phi;
}

// ====================================================================
//  FMM target-leaf evaluation. One CUDA block per target leaf cell. The
//  near/far split is decided once by the pair traversal (fmm_pair_kernel):
//  far interactions are folded into each leaf's local expansion (M2L + L2L,
//  stored in tgt.d_local) and the near source leaves are listed per target
//  leaf in the P2P CSR (tgt.d_p2p_*). The fused kernel below then evaluates,
//  per target point, the local expansion (L2P) plus the direct near field
//  (P2P). Targets != sources, so no self-exclusion. Weights and the final
//  scatter use the original (caller) point order via tgt.d_orig.
// ====================================================================

// P2P drain: each active thread direct-sums the terminal-leaf list into its point.
template<bool FIELD>
__device__ void fmm_drain_p2p(const fmm_tree &src,
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
// Fused near (P2P) + far (L2P) leaf kernel. One block per target leaf A. Each thread owns one
// target point. The near/far split was already decided by the single pair-traversal pass: the far
// field sits in tgt.d_local (M2L + L2L), and this leaf's near source leaves are tgt.d_p2p_val
// over the CSR segment [d_p2p_off[A], d_p2p_off[A+1]). No tree descent, no shared frontier, no
// MAC here -- so no frontier cap to overflow.
template<bool FIELD, int P>
__global__ void l2p_p2p_kernel(fmm_tree src, fmm_target_tree tgt,
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

  // target cell center (L2P expands about it)
  double Tcx = 0.5*(tgt.d_min[3*A]   + tgt.d_max[3*A]);
  double Tcy = 0.5*(tgt.d_min[3*A+1] + tgt.d_max[3*A+1]);
  double Tcz = 0.5*(tgt.d_min[3*A+2] + tgt.d_max[3*A+2]);

  // ---- this thread's target point (one per thread; cnt <= leaf_size <= blockDim) ----
  bool active = (threadIdx.x < cnt);
  int  my = active ? (a0 + threadIdx.x) : -1;
  double px = 0.0, py = 0.0, pz = 0.0;
  if (active) { px = tgt.d_pos[3*my]; py = tgt.d_pos[3*my+1]; pz = tgt.d_pos[3*my+2]; }
  double phi = 0.0, gx = 0.0, gy = 0.0, gz = 0.0;

  // ---- near field: direct P2P over this leaf's CSR list of near source leaves ----
  int p0 = tgt.d_p2p_off[A], p1 = tgt.d_p2p_off[A + 1];
  fmm_drain_p2p<FIELD>(src, tgt.d_p2p_val + p0, p1 - p0, active, px, py, pz, phi, gx, gy, gz);

  // ---- L2P: evaluate this leaf's complete far-field local expansion d_local[A] at each
  // target point (relative to the leaf center) and add it to the per-point P2P near field.
  // Regular solid harmonics R_j^k of (point - Tc); field via dual-AD R (order p). ----
  if (active) {
    const double *L = tgt.d_local + (size_t)tgt.d_box_slot[A] * tgt.comp;   // A = frontier box
    double Rxp = px - Tcx, Ryp = py - Tcy, Rzp = pz - Tcz;
    if constexpr (FIELD) {
      dual phi_l = l2p_contract<P, dual>(L, dual(Rxp, 1.0, 0.0, 0.0),
                                            dual(Ryp, 0.0, 1.0, 0.0),
                                            dual(Rzp, 0.0, 0.0, 1.0));
      gx -= phi_l.dx; gy -= phi_l.dy; gz -= phi_l.dz;   // E = -grad Phi (forward-mode AD)
    } else {
      phi += l2p_contract<P, double>(L, Rxp, Ryp, Rzp);
    }
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

// Dispatch the fused L2P+P2P leaf kernel for the runtime multipole order p. l2p_p2p_kernel<FIELD,P>
// is a distinct instantiation per order P (its harmonic buffers are sized at compile time), so we
// switch p -> P here. One block per target leaf; block must be >= the target leaf_size so every
// leaf point maps to a distinct thread.
template<bool FIELD>
static void launch_l2p_p2p(int p, int grid, int block,
                           const fmm_tree &src, const fmm_target_tree &tgt,
                           const double *flux, const double *norms,
                           const double *phi_sup, const double *area,
                           double inv_4pi, double *partial) {
  switch (p) {
    case 1: l2p_p2p_kernel<FIELD,1 ><<<grid, block>>>(src, tgt, flux, norms, phi_sup, area, inv_4pi, partial); break;
    case 2: l2p_p2p_kernel<FIELD,2 ><<<grid, block>>>(src, tgt, flux, norms, phi_sup, area, inv_4pi, partial); break;
    case 3: l2p_p2p_kernel<FIELD,3 ><<<grid, block>>>(src, tgt, flux, norms, phi_sup, area, inv_4pi, partial); break;
    case 4: l2p_p2p_kernel<FIELD,4 ><<<grid, block>>>(src, tgt, flux, norms, phi_sup, area, inv_4pi, partial); break;
    case 5: l2p_p2p_kernel<FIELD,5 ><<<grid, block>>>(src, tgt, flux, norms, phi_sup, area, inv_4pi, partial); break;
    case 6: l2p_p2p_kernel<FIELD,6 ><<<grid, block>>>(src, tgt, flux, norms, phi_sup, area, inv_4pi, partial); break;
    case 7: l2p_p2p_kernel<FIELD,7 ><<<grid, block>>>(src, tgt, flux, norms, phi_sup, area, inv_4pi, partial); break;
    case 8: l2p_p2p_kernel<FIELD,8 ><<<grid, block>>>(src, tgt, flux, norms, phi_sup, area, inv_4pi, partial); break;
    case 9: l2p_p2p_kernel<FIELD,9 ><<<grid, block>>>(src, tgt, flux, norms, phi_sup, area, inv_4pi, partial); break;
    default: l2p_p2p_kernel<FIELD,10><<<grid, block>>>(src, tgt, flux, norms, phi_sup, area, inv_4pi, partial); break;
  }
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

// ====================================================================
//  Target tree build / free (geometry-only; reuses the atom-tree pipeline).
// ====================================================================
// Per-node tree depth (root = 0) for the top-down L2L sweep: each node walks parent
// pointers to the root. O(depth) per node; cheap relative to the FP64 translations.
__global__ void node_depth_kernel(int n_nodes, const int *__restrict__ parent,
                                  int *__restrict__ depth) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= n_nodes) return;
  int d = 0, u = parent[v];
  while (u != -1) { ++d; u = parent[u]; }
  depth[v] = d;
}

// Flag the "box" nodes that carry a local expansion: the root and every node whose PARENT
// holds more than leaf_size points (i.e., the leaf-size frontier + all strict ancestors --
// exactly the target nodes the pair traversal can reach before stopping). flag[v] in {0,1}.
__global__ void mark_box_kernel(int n_nodes, int leaf_size,
                                const int *__restrict__ parent,
                                const int *__restrict__ first, const int *__restrict__ last,
                                int *__restrict__ flag) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= n_nodes) return;
  int u = parent[v];
  flag[v] = (u < 0) || ((last[u] - first[u] + 1) > leaf_size) ? 1 : 0;
}

// Turn the exclusive-prefix-sum of the box flags into a compact slot map and a masked depth
// (box depth, else -1) used to bound the L2L level loop.
__global__ void box_slot_kernel(int n_nodes, const int *__restrict__ flag,
                                const int *__restrict__ pref, const int *__restrict__ depth,
                                int *__restrict__ slot, int *__restrict__ box_depth) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= n_nodes) return;
  bool box = flag[v] != 0;
  slot[v]      = box ? pref[v] : -1;
  box_depth[v] = box ? depth[v] : -1;
}

// ====================================================================
//  Shared LBVH build stages 1-2, used by BOTH the source (fmm_build_atom_tree)
//  and target (build_target_tree) builders: scene AABB -> Morton codes ->
//  radix sort. Allocates and returns the Morton-sorted codes and the
//  original->sorted permutation; the caller owns and frees BOTH. All bbox/scale
//  temporaries are internal (h_min/scale are consumed only by morton_kernel).
// ====================================================================
static void lbvh_morton_sort(int N, const double *d_pts, int tpb,
                             uint64_t **d_codes_sorted_out, int **d_idx_sorted_out) {
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

  *d_codes_sorted_out = d_codes_sorted;
  *d_idx_sorted_out   = d_idx_sorted;
}

// ---- target-tree build stages (see build_target_tree for the top-level flow) ----

// Allocate the geometry + topology node arrays (uses tt->N, tt->n_nodes).
static void tt_alloc_nodes(fmm_target_tree *tt) {
  const int N  = tt->N;
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
}

// Stage 3: reorder into Morton order (geometry only) + leaf init + permutation.
static void tt_reorder(fmm_target_tree *tt, int tpb,
                       const int *d_idx_sorted, const double *d_pts) {
  int mt_blocks = (tt->N + tpb - 1) / tpb;
  reorder_geom_kernel<<<mt_blocks, tpb>>>(tt->N, d_idx_sorted, d_pts,
                                          tt->d_pos, tt->d_orig,
                                          tt->d_min, tt->d_max,
                                          tt->d_parent, tt->d_first, tt->d_last);
  CUDA_CHECK(cudaGetLastError());
}

// Stage 4: internal nodes (Karras) + bottom-up AABB. Caller guards on tt->N > 1.
static void tt_internal_aabb(fmm_target_tree *tt, int tpb,
                             const uint64_t *d_codes_sorted) {
  const int N  = tt->N;
  const int nn = tt->n_nodes;
  int in_blocks = (N - 1 + tpb - 1) / tpb; // with N leaves you always get N-1 internal nodes (full binary tree)
  int mt_blocks = (N + tpb - 1) / tpb;
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

// Stage 5: mark + compact the leaf-size frontier -> tt->d_leaf_nodes (+ tt->n_leaves).
static void tt_frontier(fmm_target_tree *tt, int tpb) {
  const int N  = tt->N;
  const int nn = tt->n_nodes;
  int fr_blocks = (nn + tpb - 1) / tpb;

  int *d_frontier;
  CUDA_CHECK(cudaMalloc(&d_frontier, nn * sizeof(int)));
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
}

// Stage 6: FMM downward-pass storage. Local expansions live ONLY on box nodes (frontier +
// ancestors), kept compact via d_box_slot, so memory is O(n_box*comp) not O(N*comp).
static void tt_downward_storage(fmm_target_tree *tt, int tpb) {
  const int nn = tt->n_nodes;
  int fr_blocks = (nn + tpb - 1) / tpb;

  CUDA_CHECK(cudaMalloc(&tt->d_depth,    nn * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&tt->d_box_slot, nn * sizeof(int)));
  node_depth_kernel<<<fr_blocks, tpb>>>(nn, tt->d_parent, tt->d_depth);
  CUDA_CHECK(cudaGetLastError());

  int *d_flag, *d_pref, *d_bdepth;
  CUDA_CHECK(cudaMalloc(&d_flag,   nn * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_pref,   nn * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_bdepth, nn * sizeof(int)));
  mark_box_kernel<<<fr_blocks, tpb>>>(nn, tt->leaf_size, tt->d_parent,
                                      tt->d_first, tt->d_last, d_flag);
  CUDA_CHECK(cudaGetLastError());
  { void *d_t = nullptr; size_t tb = 0;        // exclusive prefix sum of the box flags
    cub::DeviceScan::ExclusiveSum(d_t, tb, d_flag, d_pref, nn);
    CUDA_CHECK(cudaMalloc(&d_t, tb));
    cub::DeviceScan::ExclusiveSum(d_t, tb, d_flag, d_pref, nn);
    cudaFree(d_t); }
  box_slot_kernel<<<fr_blocks, tpb>>>(nn, d_flag, d_pref, tt->d_depth,
                                      tt->d_box_slot, d_bdepth);
  CUDA_CHECK(cudaGetLastError());
  { int last_pref, last_flag;                  // n_box = pref[nn-1] + flag[nn-1]
    CUDA_CHECK(cudaMemcpy(&last_pref, d_pref + (nn-1), sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&last_flag, d_flag + (nn-1), sizeof(int), cudaMemcpyDeviceToHost));
    tt->n_box = last_pref + last_flag; }
  { int *d_md; CUDA_CHECK(cudaMalloc(&d_md, sizeof(int)));   // max BOX depth bounds the L2L loop
    void *d_t = nullptr; size_t tb = 0;
    cub::DeviceReduce::Max(d_t, tb, d_bdepth, d_md, nn);
    CUDA_CHECK(cudaMalloc(&d_t, tb));
    cub::DeviceReduce::Max(d_t, tb, d_bdepth, d_md, nn);
    CUDA_CHECK(cudaMemcpy(&tt->max_depth, d_md, sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_t); cudaFree(d_md); }
  cudaFree(d_flag); cudaFree(d_pref); cudaFree(d_bdepth);

  CUDA_CHECK(cudaMalloc(&tt->d_local, (size_t)tt->n_box * tt->comp * sizeof(double)));
}

static void build_target_tree(int num_pts, const double *d_pts, int leaf_size, int p,
                              fmm_target_tree *tt) {
  tt->N         = num_pts;
  tt->n_nodes   = (num_pts > 0) ? (2 * num_pts - 1) : 0;
  tt->leaf_size = (leaf_size < 1) ? 1 : leaf_size;
  tt->n_leaves  = 0;
  tt->p         = p;
  tt->comp      = comp_sph(p);
  tt->max_depth = 0;
  tt->n_box     = 0;
  tt->n_p2p     = 0;
  tt->d_pos = tt->d_min = tt->d_max = tt->d_local = nullptr;
  tt->d_orig = tt->d_left = tt->d_right = tt->d_parent = nullptr;
  tt->d_first = tt->d_last = tt->d_leaf_nodes = tt->d_depth = tt->d_box_slot = nullptr;
  tt->d_p2p_off = tt->d_p2p_val = nullptr;
  if (num_pts <= 0) return;

  const int tpb = 256;
  tt_alloc_nodes(tt);

  uint64_t *d_codes_sorted; int *d_idx_sorted;
  lbvh_morton_sort(tt->N, d_pts, tpb, &d_codes_sorted, &d_idx_sorted);  // stages 1-2

  tt_reorder(tt, tpb, d_idx_sorted, d_pts);                            // stage 3
  cudaFree(d_idx_sorted);

  if (tt->N > 1) tt_internal_aabb(tt, tpb, d_codes_sorted);            // stage 4
  cudaFree(d_codes_sorted);

  tt_frontier(tt, tpb);                                                // stage 5
  tt_downward_storage(tt, tpb);                                        // stage 6

  CUDA_CHECK(cudaDeviceSynchronize());
}

static void free_target_tree(fmm_target_tree *tt) {
  if (!tt) return;
  cudaFree(tt->d_pos);    cudaFree(tt->d_orig);
  cudaFree(tt->d_min);    cudaFree(tt->d_max);
  cudaFree(tt->d_left);   cudaFree(tt->d_right);  cudaFree(tt->d_parent);
  cudaFree(tt->d_first);  cudaFree(tt->d_last);
  cudaFree(tt->d_leaf_nodes);
  cudaFree(tt->d_local);  cudaFree(tt->d_depth);  cudaFree(tt->d_box_slot);
  cudaFree(tt->d_p2p_off); cudaFree(tt->d_p2p_val);
}

// ====================================================================
//  FMM Stage 3.2: dual-tree traversal (pair-BFS) + downward pass
//  (M2L accumulate -> L2L sweep). Populates tgt.d_local with each node's
//  complete local expansion of the far field; the leaf kernel then does L2P.
// ====================================================================

// One BFS round over (target node A, source node B) pairs. Well-separated pairs emit an
// M2L entry; pairs whose two sides are both leaf-size cells are NEAR pairs emitted to the
// P2P worklist (later grouped by target leaf into a CSR for the fused L2P+P2P kernel);
// otherwise the larger cell is split and its two child pairs enqueued. The two-sided MAC is
// identical to the leaf kernel's. Geometry only -> not templated on the order. cap_* guard the
// worklists; the host retries with bigger buffers if the true counts (from the atomics) exceed them.
__global__ void fmm_pair_kernel(fmm_tree src, fmm_target_tree tgt, double theta,
                                const int2 *__restrict__ cur, int n_cur,
                                int2 *__restrict__ nxt, int *__restrict__ n_nxt, int cap_nxt,
                                int2 *__restrict__ m2l, int *__restrict__ n_m2l, int cap_m2l,
                                int2 *__restrict__ p2p, int *__restrict__ n_p2p, int cap_p2p) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_cur) return;
  int A = cur[i].x, B = cur[i].y;

  double Amnx=tgt.d_min[3*A],Amxx=tgt.d_max[3*A], Amny=tgt.d_min[3*A+1],Amxy=tgt.d_max[3*A+1],
         Amnz=tgt.d_min[3*A+2],Amxz=tgt.d_max[3*A+2];
  double Acx=0.5*(Amnx+Amxx),Acy=0.5*(Amny+Amxy),Acz=0.5*(Amnz+Amxz);
  double Aex=Amxx-Amnx,Aey=Amxy-Amny,Aez=Amxz-Amnz;
  double rA=0.5*sqrt(Aex*Aex+Aey*Aey+Aez*Aez);          // half-diagonal (matches leaf metric)

  double Bmnx=src.d_min[3*B],Bmxx=src.d_max[3*B], Bmny=src.d_min[3*B+1],Bmxy=src.d_max[3*B+1],
         Bmnz=src.d_min[3*B+2],Bmxz=src.d_max[3*B+2];
  double Bcx=0.5*(Bmnx+Bmxx),Bcy=0.5*(Bmny+Bmxy),Bcz=0.5*(Bmnz+Bmxz);
  double Bex=Bmxx-Bmnx,Bey=Bmxy-Bmny,Bez=Bmxz-Bmnz;
  double rB=0.5*sqrt(Bex*Bex+Bey*Bey+Bez*Bez);

  double Rx=Acx-Bcx,Ry=Acy-Bcy,Rz=Acz-Bcz; double R2=Rx*Rx+Ry*Ry+Rz*Rz;
  double sumr=rA+rB;
  if (R2 > 0.0 && sumr*sumr < theta*theta*R2) {         // well separated -> M2L
    int o = atomicAdd(n_m2l, 1);
    if (o < cap_m2l) m2l[o] = make_int2(A, B);
    return;
  }
  int acnt = tgt.d_last[A] - tgt.d_first[A] + 1;
  int bcnt = src.d_last[B] - src.d_first[B] + 1;
  bool A_leaf = (A >= tgt.N - 1) || (acnt <= tgt.leaf_size);
  bool B_leaf = (B >= src.N - 1) || (bcnt <= src.leaf_size);
  if (A_leaf && B_leaf) {                               // near pair -> P2P list (keyed by target leaf A)
    int o = atomicAdd(n_p2p, 1);
    if (o < cap_p2p) p2p[o] = make_int2(A, B);
    return;
  }
  bool split_A = A_leaf ? false : (B_leaf ? true : (rA >= rB));   // split the larger
  int o = atomicAdd(n_nxt, 2);
  if (o + 1 < cap_nxt) {
    if (split_A) { nxt[o] = make_int2(tgt.d_left[A], B); nxt[o+1] = make_int2(tgt.d_right[A], B); }
    else         { nxt[o] = make_int2(A, src.d_left[B]); nxt[o+1] = make_int2(A, src.d_right[B]); }
  }
}

// M2L accumulate: one thread per (A,B) M2L pair. Build the order-2p irregular harmonics
// S(Bc-Ac) and add each output local coefficient L_j^k into d_local[A] (atomic, since many
// pairs share a target node A). d_local must be zeroed first. (Correctness-first; the heavy
// per-thread S build + atomics are the obvious 3.2 optimization target.)
template<int P>
__global__ void m2l_accumulate_kernel(fmm_tree src, fmm_target_tree tgt,
                                      const int2 *__restrict__ m2l, int n_m2l) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_m2l) return;
  int A = m2l[i].x, B = m2l[i].y;
  constexpr int COMP_DBL = comp_sph(P);
  double Acx=0.5*(tgt.d_min[3*A]+tgt.d_max[3*A]), Acy=0.5*(tgt.d_min[3*A+1]+tgt.d_max[3*A+1]),
         Acz=0.5*(tgt.d_min[3*A+2]+tgt.d_max[3*A+2]);
  double Bcx=0.5*(src.d_min[3*B]+src.d_max[3*B]), Bcy=0.5*(src.d_min[3*B+1]+src.d_max[3*B+1]),
         Bcz=0.5*(src.d_min[3*B+2]+src.d_max[3*B+2]);
  cmplx S[nmom_sph(2 * P)];
  // t = (source center) - (target center)  (Eq.17 sign convention) wdym sign convention Eq 17?
  solid_harmonics<double, false, nmom_sph(2*P), 2*P>(2*P, Bcx-Acx, Bcy-Acy, Bcz-Acz, S);
  const double *O = src.d_moments + (size_t)src.d_box_slot[B] * COMP_DBL;   // B is always a box node
  double *L = tgt.d_local + (size_t)tgt.d_box_slot[A] * COMP_DBL;   // A is always a box node
  for (int j = 0; j <= P; ++j)
    for (int k = 0; k <= j; ++k) {
      cmplx acc = m2l_coeff(P, O, S, j, k);
      int s = sph_slot(j, k);
      atomicAdd(&L[2 * s],     acc.re);
      atomicAdd(&L[2 * s + 1], acc.im);
    }
}

// One level of the top-down L2L sweep: every node v at depth==lvl shifts its parent's
// (already complete) local expansion to v's center and adds it into d_local[v]. Run for
// lvl = 1..max_depth so each node sees a finalized parent. Distinct v -> no write races.
template<int P>
__global__ void l2l_level_kernel(fmm_target_tree tgt, int lvl) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= tgt.n_nodes || tgt.d_depth[v] != lvl) return;
  int sv = tgt.d_box_slot[v];
  if (sv < 0) return;                                   // only box nodes carry locals
  int u = tgt.d_parent[v];
  int su = tgt.d_box_slot[u];                           // parent of a non-root box is a box
  constexpr int COMP_DBL = comp_sph(P);
  double ucx=0.5*(tgt.d_min[3*u]+tgt.d_max[3*u]), ucy=0.5*(tgt.d_min[3*u+1]+tgt.d_max[3*u+1]),
         ucz=0.5*(tgt.d_min[3*u+2]+tgt.d_max[3*u+2]);
  double vcx=0.5*(tgt.d_min[3*v]+tgt.d_max[3*v]), vcy=0.5*(tgt.d_min[3*v+1]+tgt.d_max[3*v+1]),
         vcz=0.5*(tgt.d_min[3*v+2]+tgt.d_max[3*v+2]);
  // t = (parent center) - (child center)  (Eq.21 sign convention)
  l2l_shift<P>(tgt.d_local + (size_t)su * COMP_DBL, ucx-vcx, ucy-vcy, ucz-vcz,
               tgt.d_local + (size_t)sv * COMP_DBL);
}

// Level-synchronous dual-tree traversal (expands the whole (A,B) pair frontier per launch --
// NOT the Bonsai single-tree shared-memory BFS discovery, which we dropped; that walked the
// source tree per target group and needed a separate near-field pass. Here one sweep emits both
// far (M2L) and near (P2P) pairs). Returns the M2L pair count and the near P2P pair count, handing
// back both raw device pair arrays (caller frees). The P2P pairs are unsorted (A,B);
// fmm_build_p2p_csr groups them by target.
//
// Two distinct overflow regimes, handled separately (at small theta the interaction lists are
// large -- O(theta^-3) -- so naive doubling restarts the whole O(N) traversal many times):
//   * Frontier (per-level live split-pairs): can't know its peak ahead of time; if it overflows
//     the descent aborts early (counts are partial) -> grow x4 and redo. Small, rarely binds.
//   * Cumulative M2L / P2P lists: the atomic counters report the EXACT totals even past the cap,
//     so on overflow we resize straight to the true size (+ margin) and redo exactly ONCE -- no
//     doubling spiral.
static int fmm_pair_traverse(const fmm_tree &src, fmm_target_tree &tt, double theta,
                             int2 *&d_m2l_out, int2 *&d_p2p_out, int &n_p2p_out) {
  // Seed the cumulative caps from a generous interaction-list estimate so the common case is a
  // single pass; the exact-resize below is the safety net when this still underestimates.
  int cap_m2l = 1 << 18;
  while ((size_t)cap_m2l < (size_t)64 * tt.n_box && cap_m2l < (1 << 27)) cap_m2l <<= 1;
  int cap_p2p   = cap_m2l;
  int cap_front = cap_m2l;   // per-level frontier is <= the cumulative totals; same seed, grows x4 if it binds
  while (true) {
    int2 *d_cur, *d_nxt, *d_m2l, *d_p2p; int *d_cnt;   // d_cnt[0]=n_nxt, [1]=n_m2l, [2]=n_p2p
    CUDA_CHECK(cudaMalloc(&d_cur, (size_t)cap_front * sizeof(int2)));
    CUDA_CHECK(cudaMalloc(&d_nxt, (size_t)cap_front * sizeof(int2)));
    CUDA_CHECK(cudaMalloc(&d_m2l, (size_t)cap_m2l   * sizeof(int2)));
    CUDA_CHECK(cudaMalloc(&d_p2p, (size_t)cap_p2p   * sizeof(int2)));
    CUDA_CHECK(cudaMalloc(&d_cnt, 3 * sizeof(int)));
    int2 seed = make_int2(0, 0);
    CUDA_CHECK(cudaMemcpy(d_cur, &seed, sizeof(int2), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(&d_cnt[1], 0, 2 * sizeof(int)));   // zero n_m2l and n_p2p

    int n_cur = 1; bool front_overflow = false;   // n_cur = number of pairs in the current frontier
    while (n_cur > 0) {   // BFS level loop (loop until no more pairs)
      CUDA_CHECK(cudaMemset(&d_cnt[0], 0, sizeof(int)));
      int tpb = 128, g = (n_cur + tpb - 1) / tpb;
      fmm_pair_kernel<<<g, tpb>>>(src, tt, theta, d_cur, n_cur,
                                  d_nxt, &d_cnt[0], cap_front, d_m2l, &d_cnt[1], cap_m2l,
                                  d_p2p, &d_cnt[2], cap_p2p);
      CUDA_CHECK(cudaGetLastError());
      int n_nxt; 
      CUDA_CHECK(cudaMemcpy(&n_nxt, &d_cnt[0], sizeof(int), cudaMemcpyDeviceToHost));
      if (n_nxt > cap_front) { front_overflow = true; break; }
      int2 *tmp = d_cur; d_cur = d_nxt; d_nxt = tmp; n_cur = n_nxt;
    }
    int cnt[3]; CUDA_CHECK(cudaMemcpy(cnt, d_cnt, 3 * sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_cur); cudaFree(d_nxt); cudaFree(d_cnt);

    if (front_overflow) {                         // partial counts -> grow frontier and redo
      cudaFree(d_m2l); cudaFree(d_p2p);
      cap_front *= 4;
      continue;
    }
    int n_m2l = cnt[1], n_p2p = cnt[2];            // descent completed -> these are EXACT totals
    if (n_m2l > cap_m2l || n_p2p > cap_p2p) {
      cudaFree(d_m2l); cudaFree(d_p2p);
      if (n_m2l > cap_m2l) cap_m2l = n_m2l + (n_m2l >> 4) + 1024;   // resize to exact (+ ~6%)
      if (n_p2p > cap_p2p) cap_p2p = n_p2p + (n_p2p >> 4) + 1024;
      continue;
    }
    d_m2l_out = d_m2l;
    d_p2p_out = d_p2p;
    n_p2p_out = n_p2p;
    return n_m2l;
  }
}

// M2L accumulate + L2L downward sweep at compile-time order P.
template<int P>
static void fmm_translations(const fmm_tree &src, fmm_target_tree &tt,
                             const int2 *d_m2l, int n_m2l) {
  CUDA_CHECK(cudaMemset(tt.d_local, 0, (size_t)tt.n_box * tt.comp * sizeof(double)));
  if (n_m2l > 0) {
    int tpb = 128, g = (n_m2l + tpb - 1) / tpb;
    m2l_accumulate_kernel<P><<<g, tpb>>>(src, tt, d_m2l, n_m2l);
    CUDA_CHECK(cudaGetLastError());
  }
  int tpb = 256, g = (tt.n_nodes + tpb - 1) / tpb;
  for (int lvl = 1; lvl <= tt.max_depth; ++lvl) {
    l2l_level_kernel<P><<<g, tpb>>>(tt, lvl);
    CUDA_CHECK(cudaGetLastError());
  }
}

// Count near pairs per target node A (CSR row sizes). key = pairs[i].x = target leaf node id.
__global__ void p2p_count_kernel(const int2 *__restrict__ pairs, int n_pairs,
                                 int *__restrict__ cnt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_pairs) return;
  atomicAdd(&cnt[pairs[i].x], 1);
}

// Scatter each near pair's source-leaf B into its target node's CSR segment. Intra-segment
// order is non-deterministic (atomic fill) -- fine: P2P sum order already differs from the
// host basis (parity ~1e-10, not bit-exact).
__global__ void p2p_scatter_kernel(const int2 *__restrict__ pairs, int n_pairs,
                                   const int *__restrict__ off, int *__restrict__ fill,
                                   int *__restrict__ val) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_pairs) return;
  int A = pairs[i].x, B = pairs[i].y;
  val[off[A] + atomicAdd(&fill[A], 1)] = B;
}

// Group the raw near (A,B) pairs into a CSR keyed by target node id: counting sort over
// n_nodes buckets. Populates tt.{n_p2p, d_p2p_off[n_nodes+1], d_p2p_val[n_p2p]}.
static void fmm_build_p2p_csr(fmm_target_tree &tt, const int2 *d_p2p, int n_p2p) {
  int nn = tt.n_nodes;
  tt.n_p2p = n_p2p;
  CUDA_CHECK(cudaMalloc(&tt.d_p2p_off, (size_t)(nn + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&tt.d_p2p_val, (size_t)(n_p2p > 0 ? n_p2p : 1) * sizeof(int)));

  int *d_cnt, *d_fill;
  CUDA_CHECK(cudaMalloc(&d_cnt,  (size_t)nn * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_fill, (size_t)nn * sizeof(int)));
  CUDA_CHECK(cudaMemset(d_cnt,  0, (size_t)nn * sizeof(int)));
  CUDA_CHECK(cudaMemset(d_fill, 0, (size_t)nn * sizeof(int)));

  if (n_p2p > 0) {
    int tpb = 128, g = (n_p2p + tpb - 1) / tpb;
    p2p_count_kernel<<<g, tpb>>>(d_p2p, n_p2p, d_cnt);
    CUDA_CHECK(cudaGetLastError());
  }
  // exclusive scan of the bucket counts -> CSR offsets off[0..nn-1]; off[nn] = total.
  { void *d_t = nullptr; size_t tb = 0;
    cub::DeviceScan::ExclusiveSum(d_t, tb, d_cnt, tt.d_p2p_off, nn);
    CUDA_CHECK(cudaMalloc(&d_t, tb));
    cub::DeviceScan::ExclusiveSum(d_t, tb, d_cnt, tt.d_p2p_off, nn);
    cudaFree(d_t); }
  CUDA_CHECK(cudaMemcpy(tt.d_p2p_off + nn, &n_p2p, sizeof(int), cudaMemcpyHostToDevice));
  if (n_p2p > 0) {
    int tpb = 128, g = (n_p2p + tpb - 1) / tpb;
    p2p_scatter_kernel<<<g, tpb>>>(d_p2p, n_p2p, tt.d_p2p_off, d_fill, tt.d_p2p_val);
    CUDA_CHECK(cudaGetLastError());
  }
  cudaFree(d_cnt); cudaFree(d_fill);
}

// Build every target node's complete far-field local expansion into tt.d_local, and the
// near-field P2P CSR (tt.d_p2p_*). One pair-traversal pass feeds both.
static void fmm_build_local(const fmm_tree &src, fmm_target_tree &tt, double theta) {
  int2 *d_m2l = nullptr, *d_p2p = nullptr; int n_p2p = 0;
  int   n_m2l = fmm_pair_traverse(src, tt, theta, d_m2l, d_p2p, n_p2p);
  fmm_build_p2p_csr(tt, d_p2p, n_p2p);
  cudaFree(d_p2p);
  switch (tt.p) {
    case 1: fmm_translations<1>(src, tt, d_m2l, n_m2l); break;
    case 2: fmm_translations<2>(src, tt, d_m2l, n_m2l); break;
    case 3: fmm_translations<3>(src, tt, d_m2l, n_m2l); break;
    case 4: fmm_translations<4>(src, tt, d_m2l, n_m2l); break;
    case 5: fmm_translations<5>(src, tt, d_m2l, n_m2l); break;
    case 6: fmm_translations<6>(src, tt, d_m2l, n_m2l); break;
    case 7: fmm_translations<7>(src, tt, d_m2l, n_m2l); break;
    case 8: fmm_translations<8>(src, tt, d_m2l, n_m2l); break;
    case 9: fmm_translations<9>(src, tt, d_m2l, n_m2l); break;
    default: fmm_translations<10>(src, tt, d_m2l, n_m2l); break;
  }
  cudaFree(d_m2l);
}

// ====================================================================
//  P2M: seed each frontier box from the atoms it covers.
// ====================================================================
// One thread per frontier box, so the box's moment slot is owned exclusively -- no
// atomics, no races. Each box holds <= leaf_size atoms.
//
// The per-atom contribution is computed with m2m_shift from a stack monopole
// {q, 0, 0, ...} rather than a direct solid-harmonic P2M. That IS the same operation
// (an atom's expansion about its own position is a pure monopole) and it inherits the
// existing sign/conjugation conventions verbatim, which removes the one place this
// change could plausibly get the physics wrong. A specialised P2M -- the O sum collapses
// to the single (n,m)=(j,k) term -- would be faster and is worth doing later, but only
// once this is validated against the analytic Kirkwood reference.
// ONE BLOCK PER FRONTIER BOX, one thread per atom within it. Parallelism has to come
// from the atoms, not the boxes: at leaf_size=256 a 21k-atom system has only ~83 frontier
// boxes, so a thread-per-box launch runs ~83-wide and is effectively serial.
// Threads accumulate into a shared-memory expansion with atomicAdd, then write it out
// once -- so the O(comp) global traffic happens per box, not per atom.
template<int P>
__global__ void src_p2m_kernel(fmm_tree src) {
  constexpr int comp = comp_sph(P);
  __shared__ double Ms[comp];

  const int i = blockIdx.x;                // one block per frontier box
  if (i >= src.n_leaves) return;           // whole block returns together

  int A    = src.d_leaf_nodes[i];
  int slot = src.d_box_slot[A];            // a frontier node is always a box -> slot >= 0

  // expansion center = box AABB center (raw node id: only the moment array is slotted)
  double cx = 0.5 * (src.d_min[3*A]   + src.d_max[3*A]);
  double cy = 0.5 * (src.d_min[3*A+1] + src.d_max[3*A+1]);
  double cz = 0.5 * (src.d_min[3*A+2] + src.d_max[3*A+2]);

  for (int s = threadIdx.x; s < comp; s += blockDim.x) Ms[s] = 0.0;
  __syncthreads();

  double O[comp];                          // stack monopole; only slot (0,0) is nonzero
  for (int s = 1; s < comp; ++s) O[s] = 0.0;

  const int first = src.d_first[A], last = src.d_last[A];
  for (int a = first + threadIdx.x; a <= last; a += blockDim.x) {
    O[0] = src.d_q[a];                     // O[1] (imaginary part) stays 0
    m2m_shift_impl<P, true>(O,
                            src.d_pos[3*a]   - cx,
                            src.d_pos[3*a+1] - cy,
                            src.d_pos[3*a+2] - cz,
                            Ms);
  }
  __syncthreads();

  double *M = src.d_moments + (size_t)slot * comp;
  for (int s = threadIdx.x; s < comp; s += blockDim.x) M[s] = Ms[s];
}

// ====================================================================
//  M2M: upward sweep over box ancestors only.
// ====================================================================
// Same flag handshake as summarize_kernel, but seeded at the frontier boxes (already
// filled by P2M) instead of at radix leaves, and moments-only -- the AABBs were
// finalised earlier by summarize_aabb_kernel over the full tree.
//
// Every non-frontier box receives exactly two arrivals, one per child: a frontier child
// contributes its seeded thread, an interior child contributes its handshake winner.
// Both children of a non-frontier box ARE boxes -- count(v) > leaf_size makes each
// child's parent-count exceed leaf_size, which is mark_box_kernel's predicate -- so
// d_box_slot[lc] and d_box_slot[rc] are never -1.
//
// flags is indexed by BOX SLOT, so it is n_box ints (~700 KB) rather than n_nodes
// (~114 MB at 14M atoms).
template<int P>
__global__ void src_m2m_kernel(fmm_tree src, int *__restrict__ flags) {
  constexpr int comp = comp_sph(P);
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= src.n_leaves) return;

  int node = src.d_parent[src.d_leaf_nodes[i]];
  while (node != -1) {
    __threadfence();
    int sn = src.d_box_slot[node];               // ancestor of a box is a box
    if (atomicAdd(&flags[sn], 1) == 0) return;   // first child to arrive -> die

    int lc = src.d_left[node], rc = src.d_right[node];

    double cx = 0.5 * (src.d_min[3*node]   + src.d_max[3*node]);
    double cy = 0.5 * (src.d_min[3*node+1] + src.d_max[3*node+1]);
    double cz = 0.5 * (src.d_min[3*node+2] + src.d_max[3*node+2]);

    double *M = src.d_moments + (size_t)sn * comp;
    for (int s = 0; s < comp; ++s) M[s] = 0.0;

    #pragma unroll
    for (int s_child = 0; s_child < 2; ++s_child) {
      int c = (s_child == 0) ? lc : rc;
      double ccx = 0.5 * (src.d_min[3*c]   + src.d_max[3*c]);
      double ccy = 0.5 * (src.d_min[3*c+1] + src.d_max[3*c+1]);
      double ccz = 0.5 * (src.d_min[3*c+2] + src.d_max[3*c+2]);
      const double *mc = src.d_moments + (size_t)src.d_box_slot[c] * comp;
      m2m_shift<P>(mc, ccx - cx, ccy - cy, ccz - cz, M);
    }

    node = src.d_parent[node];
  }
}

// ---- source-tree build stages (see fmm_build_atom_tree for the top-level flow) ----

// Allocate geometry and topology. d_moments is NOT allocated here -- it is sized
// n_box*comp and so has to wait until the box set is known, exactly as tt_alloc_nodes
// defers d_local to tt_downward_storage.
static void src_alloc_nodes(fmm_tree *t) {
  const int N    = t->N;
  const int nn   = t->n_nodes;
  CUDA_CHECK(cudaMalloc(&t->d_pos,     3 * N  * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&t->d_q,           N  * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&t->d_min,     3 * nn * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&t->d_max,     3 * nn * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&t->d_left,        nn * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&t->d_right,       nn * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&t->d_parent,      nn * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&t->d_first,       nn * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&t->d_last,        nn * sizeof(int)));
}

// Stage 3: reorder atoms into Morton order + initialize leaf geometry.
static void src_reorder(fmm_tree *t, int tpb, const int *d_idx_sorted,
                        const double *d_atoms, const double *d_charges) {
  int mt_blocks = (t->N + tpb - 1) / tpb;
  reorder_kernel<<<mt_blocks, tpb>>>(t->N, d_idx_sorted, d_atoms, d_charges,
                                     t->d_pos, t->d_q,
                                     t->d_min, t->d_max,
                                     t->d_parent,
                                     t->d_first, t->d_last);
  CUDA_CHECK(cudaGetLastError());
}

// Box storage for the SOURCE tree: mark the leaf-size frontier and its ancestors, build
// the compact node->slot map, compact the frontier list, and allocate d_moments at
// n_box*comp. Mirrors tt_downward_storage + tt_frontier; duplicated rather than shared
// because those write target-typed fields and are currently correct.
//
// MUST use t->leaf_size UNCLAMPED. build_target_tree clamps its own leaf size to 256
// (see the tleaf locals in fmm_polarization_energy/fmm_ionic_energy); using that value
// here would make the box set disagree with fmm_pair_kernel's descent stop and index
// slot -1.
static void src_box_storage(fmm_tree *t, int tpb) {
  const int N  = t->N;
  const int nn = t->n_nodes;
  int fr_blocks = (nn + tpb - 1) / tpb;

  int *d_flag, *d_pref, *d_depth, *d_bdepth;
  CUDA_CHECK(cudaMalloc(&d_flag,   nn * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_pref,   nn * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_depth,  nn * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_bdepth, nn * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&t->d_box_slot, nn * sizeof(int)));

  // depth/box_depth are only inputs/outputs of the shared box_slot_kernel; the source
  // sweep is handshake-driven (not level-synchronous) so neither is kept.
  node_depth_kernel<<<fr_blocks, tpb>>>(nn, t->d_parent, d_depth);
  CUDA_CHECK(cudaGetLastError());
  mark_box_kernel<<<fr_blocks, tpb>>>(nn, t->leaf_size, t->d_parent,
                                      t->d_first, t->d_last, d_flag);
  CUDA_CHECK(cudaGetLastError());
  { void *d_t = nullptr; size_t tb = 0;        // exclusive prefix sum of the box flags
    cub::DeviceScan::ExclusiveSum(d_t, tb, d_flag, d_pref, nn);
    CUDA_CHECK(cudaMalloc(&d_t, tb));
    cub::DeviceScan::ExclusiveSum(d_t, tb, d_flag, d_pref, nn);
    cudaFree(d_t); }
  box_slot_kernel<<<fr_blocks, tpb>>>(nn, d_flag, d_pref, d_depth,
                                      t->d_box_slot, d_bdepth);
  CUDA_CHECK(cudaGetLastError());
  { int last_pref, last_flag;                  // n_box = pref[nn-1] + flag[nn-1]
    CUDA_CHECK(cudaMemcpy(&last_pref, d_pref + (nn-1), sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&last_flag, d_flag + (nn-1), sizeof(int), cudaMemcpyDeviceToHost));
    t->n_box = last_pref + last_flag; }
  cudaFree(d_flag); cudaFree(d_pref); cudaFree(d_depth); cudaFree(d_bdepth);

  // Frontier list: seeds both the P2M and the upward M2M sweep.
  int *d_frontier, *d_ids, *d_num;
  CUDA_CHECK(cudaMalloc(&d_frontier, nn * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_ids,      nn * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_num,           sizeof(int)));
  mark_frontier_kernel<<<fr_blocks, tpb>>>(nn, N, t->leaf_size,
                                           t->d_first, t->d_last, t->d_parent, d_frontier);
  CUDA_CHECK(cudaGetLastError());
  iota_kernel<<<fr_blocks, tpb>>>(nn, d_ids);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMalloc(&t->d_leaf_nodes, nn * sizeof(int)));
  { void *d_t = nullptr; size_t sb = 0;
    cub::DeviceSelect::Flagged(d_t, sb, d_ids, d_frontier, t->d_leaf_nodes, d_num, nn);
    CUDA_CHECK(cudaMalloc(&d_t, sb));
    cub::DeviceSelect::Flagged(d_t, sb, d_ids, d_frontier, t->d_leaf_nodes, d_num, nn);
    cudaFree(d_t); }
  CUDA_CHECK(cudaMemcpy(&t->n_leaves, d_num, sizeof(int), cudaMemcpyDeviceToHost));
  cudaFree(d_frontier); cudaFree(d_ids); cudaFree(d_num);

  CUDA_CHECK(cudaMalloc(&t->d_moments, (size_t)t->n_box * t->comp * sizeof(double)));
  CUDA_CHECK(cudaMemset(t->d_moments, 0, (size_t)t->n_box * t->comp * sizeof(double)));
}

// Stages 4-5: internal nodes (Karras) + bottom-up AABB propagation over the FULL tree.
// The AABB half is split out of the old fused summarize_kernel because both of the
// stages that follow depend on it: mark_box_kernel needs final d_first/d_last, and
// P2M/M2M take their expansion centers from the AABBs.
// Caller guards on t->N > 1.
static void src_internal_aabb(fmm_tree *t, int tpb, const uint64_t *d_codes_sorted) {
  const int N  = t->N;
  const int nn = t->n_nodes;
  int in_blocks = (N - 1 + tpb - 1) / tpb;
  int mt_blocks = (N + tpb - 1) / tpb;

  build_internal_kernel<<<in_blocks, tpb>>>(N, d_codes_sorted,
                                            t->d_left, t->d_right, t->d_parent,
                                            t->d_first, t->d_last);
  CUDA_CHECK(cudaGetLastError());

  int *d_flags;
  CUDA_CHECK(cudaMalloc(&d_flags, nn * sizeof(int)));
  CUDA_CHECK(cudaMemset(d_flags, 0, nn * sizeof(int)));
  summarize_aabb_kernel<<<mt_blocks, tpb>>>(N, t->d_left, t->d_right, t->d_parent,
                                            t->d_min, t->d_max, d_flags);
  CUDA_CHECK(cudaGetLastError());
  cudaFree(d_flags);
}

// Stages 6-8: box set + compact moment allocation, P2M at the frontier, M2M upward.
// Runs for every N >= 1 (unlike stages 4-5): at N == 1 the single node is both root and
// frontier box, so it gets its moments from P2M and the M2M loop body never executes.
static void src_box_moments(fmm_tree *t, int tpb, int p) {
  src_box_storage(t, tpb);

  if (t->n_leaves <= 0) return;

  // P2M: grid = one block PER BOX, block = threads over that box's atoms. Parallelism has
  // to come from the atoms; a flat thread-per-box launch runs only n_leaves wide (~83 for
  // 6VYB at leaf_size=256) and is effectively serial.
  //
  // Block size follows leaf_size so each thread owns one atom -- rounded up to a whole
  // warp, and capped at 256 (above that threads stride, which is fine). n_leaves scales
  // as N/leaf_size, so n_leaves * p2m_blk ~= N either way: full atom-level parallelism at
  // leaf_size=32 (the default) and at 256 (what the H1N1 runs use).
  int p2m_blk = ((t->leaf_size + 31) / 32) * 32;
  if (p2m_blk < 32)  p2m_blk = 32;
  if (p2m_blk > 256) p2m_blk = 256;

  switch (p) {
    case 1: src_p2m_kernel<1 ><<<t->n_leaves, p2m_blk>>>(*t); break;
    case 2: src_p2m_kernel<2 ><<<t->n_leaves, p2m_blk>>>(*t); break;
    case 3: src_p2m_kernel<3 ><<<t->n_leaves, p2m_blk>>>(*t); break;
    case 4: src_p2m_kernel<4 ><<<t->n_leaves, p2m_blk>>>(*t); break;
    case 5: src_p2m_kernel<5 ><<<t->n_leaves, p2m_blk>>>(*t); break;
    case 6: src_p2m_kernel<6 ><<<t->n_leaves, p2m_blk>>>(*t); break;
    case 7: src_p2m_kernel<7 ><<<t->n_leaves, p2m_blk>>>(*t); break;
    case 8: src_p2m_kernel<8 ><<<t->n_leaves, p2m_blk>>>(*t); break;
    case 9: src_p2m_kernel<9 ><<<t->n_leaves, p2m_blk>>>(*t); break;
    default: src_p2m_kernel<10><<<t->n_leaves, p2m_blk>>>(*t); break;
  }
  CUDA_CHECK(cudaGetLastError());

  // M2M stays thread-per-frontier-box: there are only ~n_leaves interior boxes in total
  // and each walk is ~log(N) deep, so it is nowhere near the bottleneck.
  int lf_blocks = (t->n_leaves + tpb - 1) / tpb;

  int *d_bflags;
  CUDA_CHECK(cudaMalloc(&d_bflags, t->n_box * sizeof(int)));
  CUDA_CHECK(cudaMemset(d_bflags, 0, t->n_box * sizeof(int)));
  switch (p) {
    case 1: src_m2m_kernel<1 ><<<lf_blocks, tpb>>>(*t, d_bflags); break;
    case 2: src_m2m_kernel<2 ><<<lf_blocks, tpb>>>(*t, d_bflags); break;
    case 3: src_m2m_kernel<3 ><<<lf_blocks, tpb>>>(*t, d_bflags); break;
    case 4: src_m2m_kernel<4 ><<<lf_blocks, tpb>>>(*t, d_bflags); break;
    case 5: src_m2m_kernel<5 ><<<lf_blocks, tpb>>>(*t, d_bflags); break;
    case 6: src_m2m_kernel<6 ><<<lf_blocks, tpb>>>(*t, d_bflags); break;
    case 7: src_m2m_kernel<7 ><<<lf_blocks, tpb>>>(*t, d_bflags); break;
    case 8: src_m2m_kernel<8 ><<<lf_blocks, tpb>>>(*t, d_bflags); break;
    case 9: src_m2m_kernel<9 ><<<lf_blocks, tpb>>>(*t, d_bflags); break;
    default: src_m2m_kernel<10><<<lf_blocks, tpb>>>(*t, d_bflags); break;
  }
  CUDA_CHECK(cudaGetLastError());
  cudaFree(d_bflags);
}

// ========================== public C API ==========================
extern "C" {

void fmm_build_atom_tree(int num_atoms,
                        const double *d_atoms,
                        const double *d_charges,
                        double mac,
                        int p,
                        int leaf_size,
                        fmm_tree **out) {
  // Clamp p to the supported range with a stderr warning.
  if (p < 1) {
    fprintf(stderr, "[fmm] fmm_multipole_order=%d clamped to 1\n", p);
    p = 1;
  } else if (p > FMM_MAX_P) {
    fprintf(stderr, "[fmm] fmm_multipole_order=%d clamped to %d (FMM_MAX_P)\n", p, FMM_MAX_P);
    p = FMM_MAX_P;
  }

  init_constant_tables();

  fmm_tree *t = new fmm_tree();
  t->N       = num_atoms;
  t->n_nodes = (num_atoms > 0) ? (2 * num_atoms - 1) : 0;
  t->p       = p;
  t->comp    = comp_of(p);
  t->leaf_size = (leaf_size < 1) ? 1 : leaf_size;
  t->theta   = mac;
  *out = t;
  if (num_atoms <= 0) {
    t->d_pos = t->d_q = t->d_min = t->d_max = t->d_moments = nullptr;
    t->d_left = t->d_right = t->d_parent = nullptr;
    t->d_first = t->d_last = nullptr;
    t->d_box_slot = t->d_leaf_nodes = nullptr;
    t->n_box = t->n_leaves = 0;
    return;
  }

  const int tpb = 256;
  src_alloc_nodes(t);

  uint64_t *d_codes_sorted; int *d_idx_sorted;
  lbvh_morton_sort(t->N, d_atoms, tpb, &d_codes_sorted, &d_idx_sorted);  // stages 1-2

  src_reorder(t, tpb, d_idx_sorted, d_atoms, d_charges);                 // stage 3
  cudaFree(d_idx_sorted);

  if (t->N > 1) src_internal_aabb(t, tpb, d_codes_sorted);               // stages 4-5
  cudaFree(d_codes_sorted);
  src_box_moments(t, tpb, p);                                            // stages 6-8

  CUDA_CHECK(cudaDeviceSynchronize());
}

void fmm_free_tree(fmm_tree *t) {
  if (!t) return;
  cudaFree(t->d_pos);     cudaFree(t->d_q);
  cudaFree(t->d_min);     cudaFree(t->d_max);
  cudaFree(t->d_left);    cudaFree(t->d_right);   cudaFree(t->d_parent);
  cudaFree(t->d_first);   cudaFree(t->d_last);
  cudaFree(t->d_box_slot); cudaFree(t->d_leaf_nodes);
  cudaFree(t->d_moments);
  delete t;
}

double fmm_polarization_energy(fmm_tree *src, int num_pts,
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
  fmm_target_tree tt;
  build_target_tree(num_pts, d_V, tleaf, src->p, &tt);

  int block = FMM_BDIM;   // one target point per thread; >= target leaf_size
  int grid  = tt.n_leaves;
  if (grid > 0) {
    fmm_build_local(*src, tt, src->theta);   // M2L+L2L -> d_local (far); P2P CSR (near)
    launch_l2p_p2p<false>(src->p, grid, block, *src, tt,
                          d_flux, nullptr, nullptr, nullptr, 0.0, d_partial);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  double r = reduce_sum_host(d_partial, num_pts);
  free_target_tree(&tt);
  cudaFree(d_V); cudaFree(d_flux); cudaFree(d_partial);
  return r;
}

double fmm_ionic_energy(fmm_tree *src, int num_tri_verts,
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
  fmm_target_tree tt;
  build_target_tree(num_tri_verts, d_vert, tleaf, src->p, &tt);

  int block = FMM_BDIM;   // one target point per thread; >= target leaf_size
  int grid  = tt.n_leaves;
  if (grid > 0) {
    fmm_build_local(*src, tt, src->theta);   // M2L+L2L -> d_local (far); P2P CSR (near)
    launch_l2p_p2p<true>(src->p, grid, block, *src, tt,
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

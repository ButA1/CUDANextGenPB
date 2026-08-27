/*
 *  Fast Multipole Method (FMM) for NGPB electrostatic energy kernels.
 *
 *  One octree (LBVH / Morton-radix binary radix tree) is built over the ATOMS,
 *  which carry scalar charges q_i and act as the SOURCES in all three energy
 *  terms. Each cell stores complex spherical-harmonic multipole moments up to
 *  order p. Targets are grouped into leaf cells (a geometry-only target tree
 *  built internally over the supplied points); a dual-tree traversal splits each
 *  source/target cell pair into near (P2P) and far (admissible) interactions.
 *  The far field is resolved by the full FMM operator chain (M2L + L2L), then
 *  evaluated at each target by L2P. Targets evaluate either:
 *
 *    - the potential   phi(r)   = sum_i q_i / |r - r_i|              (E_pol)
 *    - the field        g(r)     = sum_i q_i (r - r_i)/|r - r_i|^3   (E_ion double layer)
 *
 *  g(r) = -grad phi(r) is the gradient of the same scalar expansion, so no
 *  separate moment set is needed for the ionic (double-layer) term.
 *
 *  All arithmetic is FP64. The naive O(N^2)/O(N*M) kernels in energy_cuda.cu
 *  remain available; this module is selected at runtime via energy_method=2.
 */

#ifndef NGPB_FMM_H
#define NGPB_FMM_H

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle to a built atom-tree (device-resident). */
typedef struct fmm_tree fmm_tree;

/*
 * Build an octree over the atoms already resident on the device.
 *   num_atoms : number of (filtered, charged) atoms
 *   d_atoms   : device ptr, 3*num_atoms doubles, interleaved x,y,z
 *   d_charges : device ptr, num_atoms doubles
 *   mac       : multipole acceptance criterion / opening angle
 *               (accept a cell pair if size/dist < mac)  [fmm_mac]
 *   p         : multipole truncation order (1..FMM_MAX_P).  [fmm_multipole_order]
 *               Out-of-range values are clamped with a stderr warning.
 *   leaf_size : max atoms per terminal cluster of the SOURCE tree; traversal stops
 *               descending at subtrees of <= leaf_size atoms and resolves them as
 *               direct P2P.  [fmm_leaf_size]
 *               This bounds the SOURCE box radius, which is what the MAC compares
 *               against -- hence the near-field radius and the P2P work. The target
 *               tree's leaf size is a separate knob, passed per energy call below.
 *   out       : receives the built tree handle (owns its own sorted copies)
 */
void fmm_build_atom_tree(int num_atoms,
                         const double *d_atoms,
                         const double *d_charges,
                         double mac,
                         int p,
                         int leaf_size,
                         fmm_tree **out);

void fmm_free_tree(fmm_tree *tree);

/*
 * Polarization "first_int": sum_p flux_p * phi(V_p), with atoms as sources.
 * The supplied points are grouped into leaf cells (a geometry-only target tree
 * built internally) and one CUDA block per target leaf traverses the atom
 * (source) tree, sharing the descent + MAC test across all targets in the leaf;
 * the far field is resolved by M2L + L2L + L2P.
 *   num_pts : number of flux surface points
 *   h_V     : host ptr, 3*num_pts doubles (point positions)
 *   h_flux  : host ptr, num_pts doubles (per-point flux weight)
 *   target_leaf_size : max points per leaf of the target tree built here. 0 follows the
 *             source tree's leaf_size (the historical single-knob behaviour). Clamped to
 *             FMM_BDIM, since one block serves one target leaf at one point per thread.
 *             It is a per-call parameter, not a property of the atom tree: polarization
 *             and ionic build DIFFERENT target trees over different point sets (on 6VYB
 *             the ionic vertex set is ~1.6x the polarization set), so they need not share
 *             a value. Raising it cuts the target box count and so the M2L pair count,
 *             but enlarges the target box, which is the radius over which each leaf's
 *             local expansion must stay accurate -- so it trades speed against accuracy,
 *             and fmm_ionic_energy pays that trade harder (see its note).
 * (host pointers are uploaded internally, matching the existing _dev wrappers)
 */
double fmm_polarization_energy(fmm_tree *tree,
                               int num_pts,
                               const double *h_V,
                               const double *h_flux,
                               int target_leaf_size);

/*
 * Ionic "second_int": sum_v factor_v * (n_v . g(V_v)), atoms as sources, where
 * factor_v = phi_sup_v * inv_4pi * area[tri]/3 (matching the naive ionic_kernel).
 *   num_tri_verts : 3 * (number of triangles)
 *   h_vert        : host ptr, 3*num_tri_verts doubles (vertex positions)
 *   h_norms       : host ptr, 3*num_tri_verts doubles (vertex normals)
 *   h_phi_sup     : host ptr, num_tri_verts doubles (interpolated surface potential)
 *   h_area        : host ptr, num_tri_verts/3 doubles (per-triangle area)
 *   target_leaf_size : as in fmm_polarization_energy, and deliberately a separate
 *                 argument. This path evaluates E = -grad Phi, so L2P differentiates the
 *                 local expansion; a derivative of a truncated expansion converges one
 *                 order slower than the expansion itself. Enlarging the target leaf grows
 *                 the box radius the expansion must cover, so the ionic term loses
 *                 accuracy faster than the polarization term at the same target leaf
 *                 size. Expect to need a smaller value here than for polarization.
 */
double fmm_ionic_energy(fmm_tree *tree,
                        int num_tri_verts,
                        const double *h_vert,
                        const double *h_norms,
                        const double *h_phi_sup,
                        const double *h_area,
                        double inv_4pi,
                        int target_leaf_size);

#ifdef __cplusplus
}
#endif

// ====================================================================
//  Internal CUDA-only implementation types (structs, defines, operators,
//  and small device math helpers) shared across the fmm .cu source(s).
//  Guarded by __CUDACC__ so host translation units that pull in this
//  header (e.g. pb_class.cpp via mpicxx) only ever see the opaque
//  fmm_tree handle + C API above. cT<T> is a template, so this block must
//  live OUTSIDE the extern "C" section.
// ====================================================================
#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cstdio>         // fprintf in the CUDA error check
#include <cmath>          // sqrt used by tsqrt / the harmonic builders

// ====================================================================
//  CUDA error check shared by every fmm .cu. static inline: each TU that
//  includes this header gets its own copy with no multiple-definition or
//  unused-function fuss.
// ====================================================================
static inline void check_cuda(cudaError_t err, const char *msg, int line) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error at line %d (%s): %s\n",
            line, msg, cudaGetErrorString(err));
  }
}
#define CUDA_CHECK(call) check_cuda((call), #call, __LINE__)

// ====================================================================
//  FMM Stage 2: complex spherical-harmonic multipole moments (CGR'99,
//  J. Comput. Phys. 155:468). FMM_MAX_P is the largest multipole order
//  supported. Moments M_n^m are stored for the m>=0 half only (0<=m<=n<=p);
//  the m<0 half is recovered by M_n^{-m} = conj(M_n^m) (Eq. 4 convention,
//  no (-1)^m). The field (gradient) is obtained by forward-mode automatic
//  differentiation of the potential expansion (see solid_harmonics / dual),
//  so it needs no expansion above order p.
// ====================================================================
#define FMM_MAX_P        10
#define FMM_MAX_2P       (2 * FMM_MAX_P)        // 20; M2L builds irregular harmonics to order 2p
// Highest factorial index needed. M2L (Eq.17) and the order-2p solid harmonics both touch
// (l+|q|) with l <= 2p, |q| <= l, hence up to 4p:  A_{j+n}^{m-k} and sqrt((n-m)!/(n+m)!) at n=2p.
#define FMM_FACT_MAX     (4 * FMM_MAX_P)        // 40

// # complex moments per node for orders 0..p, m in [0,n] (the stored m>=0 half).
__host__ __device__ constexpr int nmom_sph(int p) { return (p + 1) * (p + 2) / 2; }
// # doubles per node = 2 * complex count (interleaved re,im at slot s -> [2s],[2s+1]).
__host__ __device__ constexpr int comp_sph(int p) { return (p + 1) * (p + 2); }
#define FMM_NMOM_MAX   nmom_sph(FMM_MAX_P)    // 66  (order-p harmonic buffer)
#define FMM_NMOM_2MAX  nmom_sph(FMM_MAX_2P)   // 231 (order-2p harmonic buffer, M2L only)

// runtime doubles-per-node for a given order (host alloc / fmm_tree.comp)
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
struct fmm_tree {
  int     N;          // number of atoms (leaves)
  int     n_nodes;    // 2N - 1
  int     p;          // multipole order in [1, FMM_MAX_P]
  int     comp;       // moments per node = COMP(p)
  int     leaf_size;  // max atoms per terminal cluster of THIS (source) tree (P2P cutoff)
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

  // ---- box storage (mirrors fmm_target_tree's d_box_slot/d_local below) ----
  // Multipole moments live ONLY on "box" nodes: the leaf-size frontier and its strict
  // ancestors (~2*n_leaves nodes), NOT the ~2N radix nodes beneath the frontier. That is
  // exactly the set fmm_pair_kernel can reach before it stops descending -- its cutoff
  // (count(B) <= leaf_size) is the same predicate as mark_box_kernel -- so no moment that
  // is ever read goes missing. Dense storage would cost n_nodes*comp = 2*(p+1)(p+2)*8
  // bytes per atom: 2112 B/atom at p=10, i.e. 30 GB for a 14M-atom virion, on EVERY rank.
  int    *d_box_slot;   // n_nodes : box-node -> [0,n_box) moment slot, else -1
  int     n_box;        // number of box nodes (frontier + strict ancestors)
  int    *d_leaf_nodes; // n_leaves : frontier node ids (drives the P2M/M2M launches)
  int     n_leaves;

  // per-BOX complex spherical-harmonic moments M_n^m, m>=0 half, interleaved
  // re,im at slot s=sph_slot(n,m): d_moments[d_box_slot[node]*comp + 2s], [..+2s+1].
  // INDEX THROUGH d_box_slot -- never by raw node id.
  double *d_moments;  // n_box * comp   (comp = comp_sph(p) doubles)
};

// ====================================================================
//  Target tree (geometry only) for the FMM energy path. Same 2N-1 LBVH
//  layout as fmm_tree but carries no charges or multipole moments -- targets
//  are only grouped into cells so one CUDA block can serve a whole leaf.
//  d_orig[sorted] -> original (caller) point index, so per-point weights
//  (flux / norms / phi_sup / area[i/3]) are read in the caller's order and
//  results scatter back to the original layout.
// ====================================================================
struct fmm_target_tree {
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

  // ---- FMM Stage 3.2 (downward pass) ----
  // Local expansions live only on "box" nodes: the leaf-size frontier and its ancestors
  // (~2*n_leaves nodes), NOT the ~N sub-frontier/single-point radix nodes. d_box_slot maps
  // a node id -> compact local-expansion slot (-1 if not a box node). This keeps d_local
  // O(n_leaves*comp) instead of O(N*comp) -- the difference between ~35 MB and ~9 GB on a
  // 10^7-vertex molecular surface.
  int     p;          // multipole order (matches the source tree)
  int     comp;       // doubles per local expansion = comp_sph(p)
  int    *d_box_slot; // n_nodes : box-node -> [0,n_box) local slot, else -1
  int     n_box;      // number of box nodes (frontier + ancestors)
  double *d_local;    // n_box * comp : per-box local expansion L_n^m (m>=0 half)
  int     max_depth;  // deepest BOX-node depth (host copy; bounds the L2L level loop)

  // ---- L2L worklist: the non-root box nodes, bucketed by depth ----
  // Same idea as d_leaf_nodes above -- resolve the "does this thread have work?" predicate ONCE
  // at build time and hand the kernel a packed list -- except the L2L predicate also depends on
  // the level, so the list is segmented by depth (a CSR keyed by depth, built by the same
  // count/scan/scatter as fmm_build_p2p_csr). Without it every L2L level scanned all n_nodes and
  // its survivors were so sparse that each one sat alone in its warp. Node depth itself is now a
  // build-time temporary and no longer stored.
  int    *d_l2l_nodes; // n_box - 1 : non-root box node ids, grouped by depth
  int    *h_l2l_off;   // max_depth + 2 : level lvl owns d_l2l_nodes[h_l2l_off[lvl]..[lvl+1]).
                       // HOST pointer -- this struct is passed BY VALUE into kernels; never
                       // dereference it on the device.

  // ---- near-field P2P interaction lists (CSR, grouped by target-leaf node A) ----
  // The pair traversal emits near (leaf,leaf) pairs; a counting sort by A groups them so the
  // fused L2P+P2P kernel walks each leaf's near source leaves directly -- no per-leaf
  // BFS descent, no shared frontier. d_p2p_off is indexed by node id (target leaf A);
  // d_p2p_val[off[A] .. off[A+1]) are that leaf's near SOURCE leaf node ids.
  int     n_p2p;      // total near (leaf,leaf) pairs
  int    *d_p2p_off;  // n_nodes + 1 : CSR row offsets, indexed by target node id
  int    *d_p2p_val;  // n_p2p       : source-leaf node ids, grouped by target leaf
};

// scalar value extractor (for value-only branch guards inside the AD path)
__host__ __device__ __forceinline__ double val(double a) { return a; }
__host__ __device__ __forceinline__ double val(dual a)   { return a.v; }

// Fetch one coefficient for any m in [-n,n] from the stored m>=0 half (conj folds m<0).
// get_M reads interleaved-double moments {M_n^m}; get_R reads a cmplx harmonic buffer --
// either the regular R_n^m or the irregular S_n^m (the m<0 conjugate fold is identical).
__device__ __forceinline__ cmplx get_M(const double* M, int n, int m) {
  if (m >= 0) { int s = sph_slot(n, m);  return cmplx(M[2 * s], M[2 * s + 1]); }
  int s = sph_slot(n, -m);               return cmplx(M[2 * s], -M[2 * s + 1]);
}
__device__ __forceinline__ cmplx get_R(const cmplx* R, int n, int m) {
  if (m >= 0) return R[sph_slot(n, m)];
  cmplx r = R[sph_slot(n, -m)];          return cmplx(r.re, -r.im);
}

// acc += c * (a * b), for complex a,b and a REAL scale c -- the inner term shared by all three
// translation operators (M2M Eq.13, M2L Eq.17, L2L Eq.21).
//
// Written as four FMAs on the accumulator rather than {complex multiply, scale by c, add}: the scale
// folds into the product and the add folds into the multiply. The term costs 2 DMUL + 4 DFMA instead
// of 4 DMUL + 2 DFMA + 2 DADD, which matters because these loops run 1716 (L2L/M2M) to 7986 (M2L)
// times per shift and the kernels are FP64-issue-bound on the 3080's 1:64 rate.
//
// NOT an accuracy tradeoff: an FMA rounds once where a separate multiply-then-add rounds twice, so
// this is strictly more accurate than the expression it replaces.
__device__ __forceinline__ void cfma_acc(cmplx &acc, double c, cmplx a, cmplx b) {
  const double cr = c * a.re, ci = c * a.im;
  acc.re = fma( cr, b.re, acc.re);
  acc.re = fma(-ci, b.im, acc.re);
  acc.im = fma( cr, b.im, acc.im);
  acc.im = fma( ci, b.re, acc.im);
}

// ====================================================================
//  double atomic min/max (CUDA has no native FP64 min/max atomic)
//
//  Instead of a CAS retry loop we exploit the fact that IEEE-754 doubles are
//  ordered by their bit pattern: read as a SIGNED int64 the non-negative ones
//  sort exactly like the doubles, while read as an UNSIGNED int64 every
//  negative one outranks every non-negative one and the negatives sort in
//  reverse. Branching on the sign of the incoming value therefore reduces each
//  update to a single native 64-bit atomic (CC >= 3.5) -- no retry loop, and no
//  non-atomic read of a location other threads are writing. Either branch is a
//  correct min/max against whatever is currently stored, whatever ITS sign, so
//  mixed-sign callers on one address still reduce correctly. -0.0 takes the
//  negative branch, which is harmless since -0.0 == 0.0. NaN is not handled
//  (all callers feed finite coordinates).
// ====================================================================
__device__ __forceinline__ double atomicMinDouble(double *addr, double val) {
  const long long i = __double_as_longlong(val);
  return i < 0
       ? __longlong_as_double((long long)atomicMax((unsigned long long *)addr,
                                                   (unsigned long long)i))
       : __longlong_as_double(atomicMin((long long *)addr, i));
}
__device__ __forceinline__ double atomicMaxDouble(double *addr, double val) {
  const long long i = __double_as_longlong(val);
  return i < 0
       ? __longlong_as_double((long long)atomicMin((unsigned long long *)addr,
                                                   (unsigned long long)i))
       : __longlong_as_double(atomicMax((long long *)addr, i));
}

// Upper bound on the target leaf size, and hence on the leaf-kernel block width. The tleaf locals in
// fmm_polarization_energy / fmm_ionic_energy clamp to this, so a leaf never holds more than FMM_BDIM
// points and one block always covers a leaf -- one target point per thread in the fused L2P+P2P
// kernel. Only the TARGET trees are capped; the source leaf size is unconstrained by this.
// This is a CAP, not the launch width: the actual width is ceil(tleaf/32)*32. Pinning the block at 256
// while fmm_leaf_size was 64 left ~84% of every block's threads permanently inactive.
constexpr int FMM_BDIM      = 256;

#endif  // __CUDACC__

#endif /* NGPB_FMM_H */

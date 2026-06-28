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
 *   leaf_size : max atoms per terminal cluster; traversal stops descending at
 *               subtrees of <= leaf_size atoms and resolves them as direct P2P.
 *               [fmm_leaf_size]
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
 * (host pointers are uploaded internally, matching the existing _dev wrappers)
 */
double fmm_polarization_energy(fmm_tree *tree,
                               int num_pts,
                               const double *h_V,
                               const double *h_flux);

/*
 * Ionic "second_int": sum_v factor_v * (n_v . g(V_v)), atoms as sources, where
 * factor_v = phi_sup_v * inv_4pi * area[tri]/3 (matching the naive ionic_kernel).
 *   num_tri_verts : 3 * (number of triangles)
 *   h_vert        : host ptr, 3*num_tri_verts doubles (vertex positions)
 *   h_norms       : host ptr, 3*num_tri_verts doubles (vertex normals)
 *   h_phi_sup     : host ptr, num_tri_verts doubles (interpolated surface potential)
 *   h_area        : host ptr, num_tri_verts/3 doubles (per-triangle area)
 */
double fmm_ionic_energy(fmm_tree *tree,
                        int num_tri_verts,
                        const double *h_vert,
                        const double *h_norms,
                        const double *h_phi_sup,
                        const double *h_area,
                        double inv_4pi);

#ifdef __cplusplus
}
#endif

#endif /* NGPB_FMM_H */

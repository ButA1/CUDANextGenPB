/*
 *  Barnes-Hut tree-code for NGPB electrostatic energy kernels.
 *
 *  One octree (LBVH / Morton-radix binary radix tree) is built over the ATOMS,
 *  which carry scalar charges q_i and act as the SOURCES in all three energy
 *  terms. Each cell stores scalar Coulomb multipole moments up to quadrupole
 *  (M, D, T). Targets traverse the tree and evaluate either:
 *
 *    - the potential   phi(r)   = sum_i q_i / |r - r_i|              (E_coul, E_pol)
 *    - the field        g(r)     = sum_i q_i (r - r_i)/|r - r_i|^3   (E_ion double layer)
 *
 *  g(r) = -grad phi(r) is the gradient of the same scalar expansion, so no
 *  separate moment set is needed for the ionic (double-layer) term.
 *
 *  All arithmetic is FP64. The naive O(N^2)/O(N*M) kernels in energy_cuda.cu
 *  remain available; this module is selected at runtime via energy_method=1.
 */

#ifndef NGPB_BARNES_HUT_H
#define NGPB_BARNES_HUT_H

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle to a built atom-tree (device-resident). */
typedef struct bh_tree bh_tree;

/*
 * Build an octree over the atoms already resident on the device.
 *   num_atoms : number of (filtered, charged) atoms
 *   d_atoms   : device ptr, 3*num_atoms doubles, interleaved x,y,z
 *   d_charges : device ptr, num_atoms doubles
 *   theta     : Barnes-Hut opening angle (MAC: accept cell if size/dist < theta)
 *   p         : multipole truncation order (1..BH_MAX_P, currently 6).
 *               Out-of-range values are clamped with a stderr warning.
 *   out       : receives the built tree handle (owns its own sorted copies)
 */
void bh_build_atom_tree(int num_atoms,
                        const double *d_atoms,
                        const double *d_charges,
                        double theta,
                        int p,
                        bh_tree **out);

void bh_free_tree(bh_tree *tree);

/*
 * Coulombic energy: sum_{i<j} q_i q_j / r_ij.
 * Each atom evaluates phi at its own location (self leaf excluded); the result
 * is 0.5 * sum_i q_i phi_i, which equals the i<j pair sum. Caller scales by den_in.
 */
double bh_coulombic_energy(bh_tree *tree);

/*
 * Polarization "first_int": sum_p flux_p * phi(V_p), with atoms as sources.
 *   num_pts : number of flux surface points
 *   h_V     : host ptr, 3*num_pts doubles (point positions)
 *   h_flux  : host ptr, num_pts doubles (per-point flux weight)
 * (host pointers are uploaded internally, matching the existing _dev wrappers)
 */
double bh_polarization_energy(bh_tree *tree,
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
double bh_ionic_energy(bh_tree *tree,
                       int num_tri_verts,
                       const double *h_vert,
                       const double *h_norms,
                       const double *h_phi_sup,
                       const double *h_area,
                       double inv_4pi);

#ifdef __cplusplus
}
#endif

#endif /* NGPB_BARNES_HUT_H */

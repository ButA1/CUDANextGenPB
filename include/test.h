#ifndef TEST_H
#define TEST_H

#include "raytracer.h"

int assemble_matrix_kernel(ray_cache_t & ray_cache);

extern "C" {
double polarization_energy_cuda(
    int    num_pts,
    const double *h_V,
    const double *h_flux,
    int    num_atoms,
    const double *h_atoms,
    const double *h_charges);

double ionic_energy_cuda(
    int    num_tri_verts,
    const double *h_vert,
    const double *h_norms,
    const double *h_phi_sup,
    const double *h_area,
    int    num_atoms,
    const double *h_atoms,
    const double *h_charges,
    double inv_4pi);
}

#endif // TEST_H
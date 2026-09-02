#ifndef TEST_H
#define TEST_H

#include "raytracer.h"

int assemble_matrix_kernel(ray_cache_t & ray_cache);

extern "C" {
void atoms_to_device(
    int num_atoms, const double *h_atoms, const double *h_charges,
    double **d_atoms_out, double **d_charges_out);

void atoms_free_device(double *d_atoms, double *d_charges);

double polarization_energy_cuda(
    int    num_pts,
    const double *h_V,
    const double *h_flux,
    int    num_atoms,
    const double *h_atoms,
    const double *h_charges);

double polarization_energy_cuda_dev(
    int    num_pts,
    const double *h_V,
    const double *h_flux,
    int    num_atoms,
    const double *d_atoms,
    const double *d_charges);

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

double ionic_energy_cuda_dev(
    int    num_tri_verts,
    const double *h_vert,
    const double *h_norms,
    const double *h_phi_sup,
    const double *h_area,
    int    num_atoms,
    const double *d_atoms,
    const double *d_charges,
    double inv_4pi);

double coulombic_energy_cuda(
    int    num_atoms,
    const double *h_atoms,
    const double *h_charges);

double coulombic_energy_cuda_dev(
    int    num_atoms,
    const double *d_atoms,
    const double *d_charges);
}

#endif // TEST_H
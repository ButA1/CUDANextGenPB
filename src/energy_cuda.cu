/*
 *  GPU-accelerated polarization and ionic energy kernels.
 *
 *  Polarization kernel:
 *    For each surface point ip, compute
 *      partial[ip] = sum_ia  charge[ia] * flux[ip] / |atom[ia] - V[ip]|
 *
 *  Ionic kernel:
 *    For each triangle vertex (itv), compute
 *      partial[itv] = sum_ia  charge[ia] * phi_sup[itv] * dot(dist, norm)
 *                      * inv_r3 * inv_4pi * area[tri] / 3
 */

#include <cuda_runtime.h>
#include <vector>
#include <cstdio>

static void check_cuda(cudaError_t err, const char *msg, int line) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error at line %d (%s): %s\n",
            line, msg, cudaGetErrorString(err));
  }
}
#define CUDA_CHECK(call) check_cuda((call), #call, __LINE__)

// --------------- polarization kernel ---------------
__global__ void
polarization_kernel(int num_pts,
                    const double * __restrict__ V,
                    const double * __restrict__ flux,
                    int num_atoms,
                    const double * __restrict__ atoms,
                    const double * __restrict__ charges,
                    double * __restrict__ partial)
{
  int ip = blockIdx.x * blockDim.x + threadIdx.x;
  if (ip >= num_pts) return;

  double vx = V[3*ip], vy = V[3*ip+1], vz = V[3*ip+2];
  double f  = flux[ip];
  double sum = 0.0;

  for (int ia = 0; ia < num_atoms; ++ia) {
    double dx = atoms[3*ia]   - vx;
    double dy = atoms[3*ia+1] - vy;
    double dz = atoms[3*ia+2] - vz;
    double r  = sqrt(dx*dx + dy*dy + dz*dz);
    sum += charges[ia] * f / r;
  }
  partial[ip] = sum;
}

// --------------- ionic kernel ---------------
__global__ void
ionic_kernel(int num_tri_verts,
             const double * __restrict__ vert,
             const double * __restrict__ norms,
             const double * __restrict__ phi_sup,
             const double * __restrict__ area,
             int num_atoms,
             const double * __restrict__ atoms,
             const double * __restrict__ charges,
             double inv_4pi,
             double * __restrict__ partial)
{
  int itv = blockIdx.x * blockDim.x + threadIdx.x;
  if (itv >= num_tri_verts) return;

  double vx = vert[3*itv], vy = vert[3*itv+1], vz = vert[3*itv+2];
  double nx = norms[3*itv], ny = norms[3*itv+1], nz = norms[3*itv+2];
  double ps = phi_sup[itv];
  double a  = area[itv / 3];
  double factor = ps * inv_4pi * a / 3.0;

  double sum = 0.0;
  for (int ia = 0; ia < num_atoms; ++ia) {
    double dx = vx - atoms[3*ia];
    double dy = vy - atoms[3*ia+1];
    double dz = vz - atoms[3*ia+2];
    double r2 = dx*dx + dy*dy + dz*dz;
    double r  = sqrt(r2);
    double inv_r3 = 1.0 / (r2 * r);
    double dot = dx*nx + dy*ny + dz*nz;
    sum += charges[ia] * dot * inv_r3 * factor;
  }
  partial[itv] = sum;
}

// --------------- coulombic kernel ---------------
__global__ void
coulombic_kernel(int num_atoms,
                 const double * __restrict__ atoms,
                 const double * __restrict__ charges,
                 double * __restrict__ partial)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_atoms) return;

  double qi = charges[i];
  double xi = atoms[3*i], yi = atoms[3*i+1], zi = atoms[3*i+2];
  double sum = 0.0;

  for (int j = i + 1; j < num_atoms; ++j) {
    double dx = xi - atoms[3*j];
    double dy = yi - atoms[3*j+1];
    double dz = zi - atoms[3*j+2];
    double r  = sqrt(dx*dx + dy*dy + dz*dz);
    sum += qi * charges[j] / r;
  }
  partial[i] = sum;
}

// --------------- host-side reduction helper ---------------
static double gpu_reduce_sum(const double *d_arr, int n) {
  std::vector<double> h(n);
  CUDA_CHECK(cudaMemcpy(h.data(), d_arr, n * sizeof(double),
                         cudaMemcpyDeviceToHost));
  double s = 0.0;
  for (int i = 0; i < n; ++i) s += h[i];
  return s;
}

// ========================== public C API ==========================

extern "C" {

double polarization_energy_cuda(
    int    num_pts,
    const double *h_V,
    const double *h_flux,
    int    num_atoms,
    const double *h_atoms,
    const double *h_charges)
{
  if (num_pts == 0 || num_atoms == 0) return 0.0;

  double *d_V, *d_flux, *d_atoms, *d_charges, *d_partial;
  CUDA_CHECK(cudaMalloc(&d_V,       num_pts   * 3 * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_flux,    num_pts       * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_atoms,   num_atoms * 3 * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_charges, num_atoms     * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_partial, num_pts       * sizeof(double)));

  CUDA_CHECK(cudaMemcpy(d_V,       h_V,       num_pts   * 3 * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_flux,    h_flux,    num_pts       * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_atoms,   h_atoms,   num_atoms * 3 * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_charges, h_charges, num_atoms     * sizeof(double), cudaMemcpyHostToDevice));

  int tpb = 256;
  int blocks = (num_pts + tpb - 1) / tpb;
  polarization_kernel<<<blocks, tpb>>>(num_pts, d_V, d_flux,
                                       num_atoms, d_atoms, d_charges,
                                       d_partial);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  double result = gpu_reduce_sum(d_partial, num_pts);

  cudaFree(d_V);
  cudaFree(d_flux);
  cudaFree(d_atoms);
  cudaFree(d_charges);
  cudaFree(d_partial);

  return result;
}

double ionic_energy_cuda(
    int    num_tri_verts,
    const double *h_vert,
    const double *h_norms,
    const double *h_phi_sup,
    const double *h_area,
    int    num_atoms,
    const double *h_atoms,
    const double *h_charges,
    double inv_4pi)
{
  if (num_tri_verts == 0 || num_atoms == 0) return 0.0;

  int num_tris = num_tri_verts / 3;

  double *d_vert, *d_norms, *d_phi, *d_area, *d_atoms, *d_charges, *d_partial;
  CUDA_CHECK(cudaMalloc(&d_vert,    num_tri_verts * 3 * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_norms,   num_tri_verts * 3 * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_phi,     num_tri_verts     * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_area,    num_tris          * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_atoms,   num_atoms     * 3 * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_charges, num_atoms         * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_partial, num_tri_verts     * sizeof(double)));

  CUDA_CHECK(cudaMemcpy(d_vert,    h_vert,     num_tri_verts * 3 * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_norms,   h_norms,    num_tri_verts * 3 * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_phi,     h_phi_sup,  num_tri_verts     * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_area,    h_area,     num_tris          * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_atoms,   h_atoms,    num_atoms     * 3 * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_charges, h_charges,  num_atoms         * sizeof(double), cudaMemcpyHostToDevice));

  int tpb = 256;
  int blocks = (num_tri_verts + tpb - 1) / tpb;
  ionic_kernel<<<blocks, tpb>>>(num_tri_verts, d_vert, d_norms, d_phi, d_area,
                                num_atoms, d_atoms, d_charges,
                                inv_4pi, d_partial);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  double result = gpu_reduce_sum(d_partial, num_tri_verts);

  cudaFree(d_vert);
  cudaFree(d_norms);
  cudaFree(d_phi);
  cudaFree(d_area);
  cudaFree(d_atoms);
  cudaFree(d_charges);
  cudaFree(d_partial);

  return result;
}

double coulombic_energy_cuda(
    int    num_atoms,
    const double *h_atoms,
    const double *h_charges)
{
  if (num_atoms < 2) return 0.0;

  double *d_atoms, *d_charges, *d_partial;
  CUDA_CHECK(cudaMalloc(&d_atoms,   num_atoms * 3 * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_charges, num_atoms     * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_partial, num_atoms     * sizeof(double)));

  CUDA_CHECK(cudaMemcpy(d_atoms,   h_atoms,   num_atoms * 3 * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_charges, h_charges, num_atoms     * sizeof(double), cudaMemcpyHostToDevice));

  int tpb = 256;
  int blocks = (num_atoms + tpb - 1) / tpb;
  coulombic_kernel<<<blocks, tpb>>>(num_atoms, d_atoms, d_charges, d_partial);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  double result = gpu_reduce_sum(d_partial, num_atoms);

  cudaFree(d_atoms);
  cudaFree(d_charges);
  cudaFree(d_partial);

  return result;
}

} // extern "C"

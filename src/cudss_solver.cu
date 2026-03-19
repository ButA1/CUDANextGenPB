/*
 *  Copyright (C) 2019-2025 Carlo de Falco
 *  Copyright (C) 2020-2021 Martina Politi
 *  Copyright (C) 2021-2025 Vincenzo Di Florio
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#include "pb_class.h"

#include <stdio.h>
#include <cuda_runtime.h>
#include <cudss.h>

#include <bim_distributed_vector.h>
#include <quad_operators_3d.h>

void
poisson_boltzmann::cudss_compute_electric_potential (ray_cache_t & ray_cache)
{
  int rank, size;
  MPI_Comm_size (mpicomm, &size);
  MPI_Comm_rank (mpicomm, &rank);

  if (size > 1)
    {
      if (rank == 0)
        fprintf (stderr, "cuDSS solver does not support distributed (MPI size > 1) runs.\n");
      return;
    }

  // CSR
  std::vector<double> vals;
  std::vector<int> irow, jcol;

  (*A).csr (vals, jcol, irow);

  int nnz  = (*A).owned_nnz ();
  int n    = tmsh.num_owned_nodes ();
  int nrhs = 1;

  A.reset ();

  // Device pointers
  int    *csr_offsets_d = nullptr;
  int    *csr_columns_d = nullptr;
  double *csr_values_d  = nullptr;
  double *b_values_d    = nullptr;
  double *x_values_d    = nullptr;

  // Allocate device memory
  if (cudaMalloc (&csr_offsets_d, (n + 1) * sizeof (int)) != cudaSuccess ||
      cudaMalloc (&csr_columns_d, nnz     * sizeof (int)) != cudaSuccess ||
      cudaMalloc (&csr_values_d,  nnz     * sizeof (double)) != cudaSuccess ||
      cudaMalloc (&b_values_d,    n       * sizeof (double)) != cudaSuccess ||
      cudaMalloc (&x_values_d,    n       * sizeof (double)) != cudaSuccess)
    {
      printf ("cuDSS: device memory allocation failed\n");
      cudaFree (csr_offsets_d);
      cudaFree (csr_columns_d);
      cudaFree (csr_values_d);
      cudaFree (b_values_d);
      cudaFree (x_values_d);
      return;
    }

  // Copy matrix to device
  cudaMemcpy (csr_offsets_d, irow.data (), (n + 1) * sizeof (int),    cudaMemcpyHostToDevice);
  cudaMemcpy (csr_columns_d, jcol.data (), nnz     * sizeof (int),    cudaMemcpyHostToDevice);
  cudaMemcpy (csr_values_d,  vals.data (), nnz     * sizeof (double), cudaMemcpyHostToDevice);

  // Copy RHS to device and free host-side storage
  cudaMemcpy (b_values_d, rhs->get_owned_data ().data (), n * sizeof (double), cudaMemcpyHostToDevice);
  rhs.reset ();

  // Create CUDA stream
  cudaStream_t stream = nullptr;
  cudaStreamCreate (&stream);

  // Create cuDSS handle
  cudssHandle_t handle;
  cudssCreate (&handle);
  cudssSetStream (handle, stream);

  // Create cuDSS config and data
  cudssConfig_t solverConfig;
  cudssData_t   solverData;
  cudssConfigCreate (&solverConfig);
  cudssDataCreate (handle, &solverData);

  // Wrap matrix and vectors as cuDSS objects
  cudssMatrix_t matA, matB, matX;
  int64_t nrows = n, ncols = n;

  cudssMatrixCreateDn (&matB, nrows, nrhs, nrows, b_values_d, CUDA_R_64F,
                       CUDSS_LAYOUT_COL_MAJOR);
  cudssMatrixCreateDn (&matX, nrows, nrhs, nrows, x_values_d, CUDA_R_64F,
                       CUDSS_LAYOUT_COL_MAJOR);
  cudssMatrixCreateCsr (&matA, nrows, ncols, nnz,
                        csr_offsets_d, nullptr, csr_columns_d, csr_values_d,
                        CUDA_R_32I, CUDA_R_64F,
                        CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO);

  // Symbolic factorization
  cudssStatus_t status;
  status = cudssExecute (handle, CUDSS_PHASE_ANALYSIS,
                         solverConfig, solverData, matA, matX, matB);
  if (status != CUDSS_STATUS_SUCCESS)
    printf ("cuDSS: analysis phase failed with status %d\n", status);

  // Numerical factorization
  status = cudssExecute (handle, CUDSS_PHASE_FACTORIZATION,
                         solverConfig, solverData, matA, matX, matB);
  if (status != CUDSS_STATUS_SUCCESS)
    printf ("cuDSS: factorization phase failed with status %d\n", status);

  // Solve
  status = cudssExecute (handle, CUDSS_PHASE_SOLVE,
                         solverConfig, solverData, matA, matX, matB);
  if (status != CUDSS_STATUS_SUCCESS)
    printf ("cuDSS: solve phase failed with status %d\n", status);

  cudaStreamSynchronize (stream);

  // Copy solution back to host
  phi = std::make_unique<distributed_vector> (n, mpicomm);
  cudaMemcpy (phi->get_owned_data ().data (), x_values_d, n * sizeof (double),
              cudaMemcpyDeviceToHost);

  // Destroy cuDSS objects
  cudssMatrixDestroy (matA);
  cudssMatrixDestroy (matB);
  cudssMatrixDestroy (matX);
  cudssDataDestroy (handle, solverData);
  cudssConfigDestroy (solverConfig);
  cudssDestroy (handle);

  // Destroy CUDA stream and free device memory
  cudaStreamDestroy (stream);
  cudaFree (csr_offsets_d);
  cudaFree (csr_columns_d);
  cudaFree (csr_values_d);
  cudaFree (b_values_d);
  cudaFree (x_values_d);

  if (size > 1)
    bim3a_solution_with_ghosts (tmsh, *phi, replace_op);
}

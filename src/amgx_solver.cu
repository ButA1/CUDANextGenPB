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
#include <amgx_c.h>

#include <bim_distributed_vector.h>
#include <quad_operators_3d.h>

void
poisson_boltzmann::amgx_compute_electric_potential (ray_cache_t & ray_cache)
{
  int rank, size;
  MPI_Comm_size (mpicomm, &size);
  MPI_Comm_rank (mpicomm, &rank);

  // CSR extraction (local rows, global column indices)
  std::vector<double> vals;
  std::vector<int> irow, jcol;

  (*A).csr (vals, jcol, irow);

  int nnz      = (*A).owned_nnz ();
  int n        = tmsh.num_owned_nodes ();
  int n_global = tmsh.num_global_nodes ();

  A.reset ();

  // Initialize AMGX
  AMGX_SAFE_CALL (AMGX_initialize ());
  AMGX_SAFE_CALL (AMGX_initialize_plugins ());

  // Create config for GMRES solver
  AMGX_config_handle cfg;
  AMGX_SAFE_CALL (AMGX_config_create (
    &cfg,
    "config_version=2, "
    "solver(main)=GMRES, "
    "main:gmres_n_restart=30, "
    "main:max_iters=1000, "
    "main:tolerance=1e-10, "
    "main:norm=L2, "
    "main:convergence=RELATIVE_INI_CORE, "
    "main:monitor_residual=1, "
    "main:print_solve_stats=1, "
    "main:obtain_timings=1, "
    "main:preconditioner(amg)=AMG, "
    "amg:algorithm=AGGREGATION, "
    "amg:selector=SIZE_2, "
    "amg:max_iters=1, "
    "amg:cycle=V, "
    "amg:smoother=JACOBI_L1, "
    "amg:presweeps=1, "
    "amg:postsweeps=1, "
    "amg:max_levels=25"
  ));

  // Create resources with MPI communicator
  AMGX_resources_handle rsrc;
  int device_id = 0;
  AMGX_SAFE_CALL (AMGX_resources_create (&rsrc, cfg, &mpicomm, 1, &device_id));

  // Create matrix, vectors, and solver
  AMGX_matrix_handle amgx_A;
  AMGX_vector_handle amgx_b;
  AMGX_vector_handle amgx_x;
  AMGX_solver_handle solver;

  AMGX_SAFE_CALL (AMGX_matrix_create (&amgx_A, rsrc, AMGX_mode_dDDI));
  AMGX_SAFE_CALL (AMGX_vector_create (&amgx_b, rsrc, AMGX_mode_dDDI));
  AMGX_SAFE_CALL (AMGX_vector_create (&amgx_x, rsrc, AMGX_mode_dDDI));
  AMGX_SAFE_CALL (AMGX_solver_create (&solver, rsrc, AMGX_mode_dDDI, cfg));

  // Build partition vector (maps each global row to its owning rank)
  std::vector<int> partition_offsets (size + 1);
  std::vector<int> all_n (size);
  MPI_Allgather (&n, 1, MPI_INT, all_n.data (), 1, MPI_INT, mpicomm);
  partition_offsets[0] = 0;
  for (int i = 0; i < size; ++i)
    partition_offsets[i + 1] = partition_offsets[i] + all_n[i];

  std::vector<int> partition_vector (n_global);
  for (int r = 0; r < size; ++r)
    for (int i = partition_offsets[r]; i < partition_offsets[r + 1]; ++i)
      partition_vector[i] = r;

  // Get number of import rings from config
  int nrings;
  AMGX_SAFE_CALL (AMGX_config_get_default_number_of_rings (cfg, &nrings));

  // Upload distributed matrix (CSR with global column indices)
  AMGX_SAFE_CALL (AMGX_matrix_upload_all_global_32 (
    amgx_A, n_global, n, nnz, 1, 1,
    irow.data (), jcol.data (), vals.data (), NULL,
    0, nrings, partition_vector.data ()));

  // Bind vectors to matrix so they inherit the distributed layout
  AMGX_SAFE_CALL (AMGX_vector_bind (amgx_b, amgx_A));
  AMGX_SAFE_CALL (AMGX_vector_bind (amgx_x, amgx_A));

  // Upload RHS vector (local portion)
  AMGX_SAFE_CALL (AMGX_vector_upload (amgx_b, n, 1, rhs->get_owned_data ().data ()));
  rhs.reset ();

  // Upload zero initial guess
  std::vector<double> x0 (n, 0.0);
  AMGX_SAFE_CALL (AMGX_vector_upload (amgx_x, n, 1, x0.data ()));

  // Setup solver with matrix and solve
  AMGX_SAFE_CALL (AMGX_solver_setup (solver, amgx_A));
  AMGX_SAFE_CALL (AMGX_solver_solve (solver, amgx_b, amgx_x));

  // Check convergence
  AMGX_SOLVE_STATUS solve_status;
  AMGX_SAFE_CALL (AMGX_solver_get_status (solver, &solve_status));
  if (solve_status != AMGX_SOLVE_SUCCESS)
    printf ("AMGX [rank %d]: solver did not converge (status = %d)\n", rank, solve_status);

  // Download solution (local portion)
  phi = std::make_unique<distributed_vector> (n, mpicomm);
  AMGX_SAFE_CALL (AMGX_vector_download (amgx_x, phi->get_owned_data ().data ()));

  // Cleanup AMGX
  AMGX_SAFE_CALL (AMGX_solver_destroy (solver));
  AMGX_SAFE_CALL (AMGX_vector_destroy (amgx_x));
  AMGX_SAFE_CALL (AMGX_vector_destroy (amgx_b));
  AMGX_SAFE_CALL (AMGX_matrix_destroy (amgx_A));
  AMGX_SAFE_CALL (AMGX_resources_destroy (rsrc));
  AMGX_SAFE_CALL (AMGX_config_destroy (cfg));

  AMGX_SAFE_CALL (AMGX_finalize_plugins ());
  AMGX_SAFE_CALL (AMGX_finalize ());

  if (size > 1)
    bim3a_solution_with_ghosts (tmsh, *phi, replace_op);
}

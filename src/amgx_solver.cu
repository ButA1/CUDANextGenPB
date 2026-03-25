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

  if (size > 1)
    {
      if (rank == 0)
        fprintf (stderr, "AMGX solver does not support distributed (MPI size > 1) runs.\n");
      return;
    }

  // CSR
  std::vector<double> vals;
  std::vector<int> irow, jcol;

  (*A).csr (vals, jcol, irow);

  int nnz = (*A).owned_nnz ();
  int n   = tmsh.num_owned_nodes ();

  A.reset ();

  printf ("AMGX: n = %d, nnz = %d (matrix memory ~ %.1f MB)\n",
          n, nnz, (nnz * (sizeof (double) + sizeof (int)) + (n + 1) * sizeof (int)) / (1024.0 * 1024.0));

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
    "amg:algorithm=CLASSICAL, "
    "amg:max_iters=1, "
    "amg:cycle=V, "
    "amg:smoother=BLOCK_JACOBI, "
    "amg:presweeps=1, "
    "amg:postsweeps=1, "
    "amg:max_levels=25"
  ));

  // Create resources, matrix, and vectors
  AMGX_resources_handle rsrc;
  AMGX_matrix_handle    amgx_A;
  AMGX_vector_handle    amgx_b;
  AMGX_vector_handle    amgx_x;
  AMGX_solver_handle    solver;

  AMGX_SAFE_CALL (AMGX_resources_create_simple (&rsrc, cfg));
  AMGX_SAFE_CALL (AMGX_matrix_create (&amgx_A, rsrc, AMGX_mode_dDDI));
  AMGX_SAFE_CALL (AMGX_vector_create (&amgx_b, rsrc, AMGX_mode_dDDI));
  AMGX_SAFE_CALL (AMGX_vector_create (&amgx_x, rsrc, AMGX_mode_dDDI));
  AMGX_SAFE_CALL (AMGX_solver_create (&solver, rsrc, AMGX_mode_dDDI, cfg));

  // Upload matrix (CSR format, zero-based indexing)
  AMGX_SAFE_CALL (AMGX_matrix_upload_all (
    amgx_A, n, nnz, 1, 1,
    irow.data (), jcol.data (), vals.data (), NULL
  ));

  // Upload RHS vector
  AMGX_SAFE_CALL (AMGX_vector_upload (amgx_b, n, 1, rhs->get_owned_data ().data ()));
  rhs.reset ();

  // Set initial guess to zero
  std::vector<double> x0 (n, 0.0);
  AMGX_SAFE_CALL (AMGX_vector_upload (amgx_x, n, 1, x0.data ()));

  // Setup solver with matrix
  AMGX_SAFE_CALL (AMGX_solver_setup (solver, amgx_A));

  // Solve
  AMGX_SAFE_CALL (AMGX_solver_solve (solver, amgx_b, amgx_x));

  // Check convergence
  AMGX_SOLVE_STATUS solve_status;
  AMGX_SAFE_CALL (AMGX_solver_get_status (solver, &solve_status));
  if (solve_status != AMGX_SOLVE_SUCCESS)
    printf ("AMGX: solver did not converge (status = %d)\n", solve_status);

  // Download solution to host
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

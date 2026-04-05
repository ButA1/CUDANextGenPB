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

  std::vector<double> rhs_local = rhs->get_owned_data ();
  rhs.reset ();

  // AMGX config string (shared between single- and multi-rank paths)
  const char *amgx_cfg_str =
    "config_version=2, "
    "solver(main)=PCG, "
    "main:max_iters=1000, "
    "main:tolerance=1e-10, "
    "main:norm=L2, "
    "main:convergence=RELATIVE_INI_CORE, "
    "main:monitor_residual=1, "
    "main:print_solve_stats=1, "
    "main:obtain_timings=1, "
    "main:preconditioner(amg)=BLOCK_JACOBI, "
    "amg:max_iters=1";

  if (size == 1)
    {
      // --- Single rank: solve directly ---
      AMGX_SAFE_CALL (AMGX_initialize ());
      AMGX_SAFE_CALL (AMGX_initialize_plugins ());

      AMGX_config_handle cfg;
      AMGX_SAFE_CALL (AMGX_config_create (&cfg, amgx_cfg_str));

      AMGX_resources_handle rsrc;
      int device_id = 0;
      AMGX_SAFE_CALL (AMGX_resources_create (&rsrc, cfg, &mpicomm, 1, &device_id));

      AMGX_matrix_handle amgx_A;
      AMGX_vector_handle amgx_b;
      AMGX_vector_handle amgx_x;
      AMGX_solver_handle solver;

      AMGX_SAFE_CALL (AMGX_matrix_create (&amgx_A, rsrc, AMGX_mode_dDDI));
      AMGX_SAFE_CALL (AMGX_vector_create (&amgx_b, rsrc, AMGX_mode_dDDI));
      AMGX_SAFE_CALL (AMGX_vector_create (&amgx_x, rsrc, AMGX_mode_dDDI));
      AMGX_SAFE_CALL (AMGX_solver_create (&solver, rsrc, AMGX_mode_dDDI, cfg));

      AMGX_SAFE_CALL (AMGX_matrix_upload_all (
        amgx_A, n, nnz, 1, 1,
        irow.data (), jcol.data (), vals.data (), NULL));

      AMGX_SAFE_CALL (AMGX_vector_upload (amgx_b, n, 1, rhs_local.data ()));

      std::vector<double> x0 (n, 0.0);
      AMGX_SAFE_CALL (AMGX_vector_upload (amgx_x, n, 1, x0.data ()));

      AMGX_SAFE_CALL (AMGX_solver_setup (solver, amgx_A));
      AMGX_SAFE_CALL (AMGX_solver_solve (solver, amgx_b, amgx_x));

      AMGX_SOLVE_STATUS solve_status;
      AMGX_SAFE_CALL (AMGX_solver_get_status (solver, &solve_status));
      if (solve_status != AMGX_SOLVE_SUCCESS)
        printf ("AMGX: solver did not converge (status = %d)\n", solve_status);

      phi = std::make_unique<distributed_vector> (n, mpicomm);
      AMGX_SAFE_CALL (AMGX_vector_download (amgx_x, phi->get_owned_data ().data ()));

      AMGX_SAFE_CALL (AMGX_solver_destroy (solver));
      AMGX_SAFE_CALL (AMGX_vector_destroy (amgx_x));
      AMGX_SAFE_CALL (AMGX_vector_destroy (amgx_b));
      AMGX_SAFE_CALL (AMGX_matrix_destroy (amgx_A));
      AMGX_SAFE_CALL (AMGX_resources_destroy (rsrc));
      AMGX_SAFE_CALL (AMGX_config_destroy (cfg));

      AMGX_SAFE_CALL (AMGX_finalize_plugins ());
      AMGX_SAFE_CALL (AMGX_finalize ());
    }
  else
    {
      // --- Multi-rank: gather to rank 0, solve on single GPU, scatter back ---
      // This avoids the GPU memory overhead of the AMGX distributed path
      // while still using multiple ranks for CPU-parallel assembly.

      std::vector<int> all_n (size), all_nnz (size);
      MPI_Gather (&n,   1, MPI_INT, all_n.data (),   1, MPI_INT, 0, mpicomm);
      MPI_Gather (&nnz, 1, MPI_INT, all_nnz.data (), 1, MPI_INT, 0, mpicomm);

      // Compute displacements (only meaningful at rank 0)
      std::vector<int> n_displs (size + 1, 0);
      std::vector<int> nnz_displs (size + 1, 0);
      if (rank == 0)
        for (int i = 0; i < size; ++i) {
          n_displs[i + 1]   = n_displs[i]   + all_n[i];
          nnz_displs[i + 1] = nnz_displs[i] + all_nnz[i];
        }

      int total_nnz = 0;
      if (rank == 0)
        total_nnz = nnz_displs[size];

      // Gather RHS
      std::vector<double> global_rhs;
      if (rank == 0) global_rhs.resize (n_global);
      MPI_Gatherv (rhs_local.data (), n, MPI_DOUBLE,
                   global_rhs.data (), all_n.data (), n_displs.data (),
                   MPI_DOUBLE, 0, mpicomm);
      std::vector<double> ().swap (rhs_local);

      // Gather column indices and values
      std::vector<int> global_jcol;
      std::vector<double> global_vals;
      if (rank == 0) {
        global_jcol.resize (total_nnz);
        global_vals.resize (total_nnz);
      }
      MPI_Gatherv (jcol.data (), nnz, MPI_INT,
                   global_jcol.data (), all_nnz.data (), nnz_displs.data (),
                   MPI_INT, 0, mpicomm);
      MPI_Gatherv (vals.data (), nnz, MPI_DOUBLE,
                   global_vals.data (), all_nnz.data (), nnz_displs.data (),
                   MPI_DOUBLE, 0, mpicomm);
      std::vector<int> ().swap (jcol);
      std::vector<double> ().swap (vals);

      // Gather row pointers (each rank sends first n entries of its n+1 irow)
      std::vector<int> gathered_irow;
      if (rank == 0) gathered_irow.resize (n_global);
      MPI_Gatherv (irow.data (), n, MPI_INT,
                   gathered_irow.data (), all_n.data (), n_displs.data (),
                   MPI_INT, 0, mpicomm);
      std::vector<int> ().swap (irow);

      // Build global row pointers by adjusting local offsets
      std::vector<int> global_irow;
      if (rank == 0) {
        global_irow.resize (n_global + 1);
        for (int r = 0; r < size; ++r)
          for (int i = 0; i < all_n[r]; ++i)
            global_irow[n_displs[r] + i] = gathered_irow[n_displs[r] + i] + nnz_displs[r];
        global_irow[n_global] = total_nnz;
        std::vector<int> ().swap (gathered_irow);
      }

      // Rank 0: solve with AMGX using MPI_COMM_SELF (single-rank mode)
      std::vector<double> global_phi;
      if (rank == 0) {
        AMGX_SAFE_CALL (AMGX_initialize ());
        AMGX_SAFE_CALL (AMGX_initialize_plugins ());

        AMGX_config_handle cfg;
        AMGX_SAFE_CALL (AMGX_config_create (&cfg, amgx_cfg_str));

        AMGX_resources_handle rsrc;
        int device_id = 0;
        MPI_Comm self_comm = MPI_COMM_SELF;
        AMGX_SAFE_CALL (AMGX_resources_create (&rsrc, cfg, &self_comm, 1, &device_id));

        AMGX_matrix_handle amgx_A;
        AMGX_vector_handle amgx_b;
        AMGX_vector_handle amgx_x;
        AMGX_solver_handle solver;

        AMGX_SAFE_CALL (AMGX_matrix_create (&amgx_A, rsrc, AMGX_mode_dDDI));
        AMGX_SAFE_CALL (AMGX_vector_create (&amgx_b, rsrc, AMGX_mode_dDDI));
        AMGX_SAFE_CALL (AMGX_vector_create (&amgx_x, rsrc, AMGX_mode_dDDI));
        AMGX_SAFE_CALL (AMGX_solver_create (&solver, rsrc, AMGX_mode_dDDI, cfg));

        AMGX_SAFE_CALL (AMGX_matrix_upload_all (
          amgx_A, n_global, total_nnz, 1, 1,
          global_irow.data (), global_jcol.data (), global_vals.data (), NULL));
        std::vector<int> ().swap (global_irow);
        std::vector<int> ().swap (global_jcol);
        std::vector<double> ().swap (global_vals);

        AMGX_SAFE_CALL (AMGX_vector_upload (amgx_b, n_global, 1, global_rhs.data ()));
        std::vector<double> ().swap (global_rhs);

        global_phi.resize (n_global, 0.0);
        AMGX_SAFE_CALL (AMGX_vector_upload (amgx_x, n_global, 1, global_phi.data ()));

        AMGX_SAFE_CALL (AMGX_solver_setup (solver, amgx_A));
        AMGX_SAFE_CALL (AMGX_solver_solve (solver, amgx_b, amgx_x));

        AMGX_SOLVE_STATUS solve_status;
        AMGX_SAFE_CALL (AMGX_solver_get_status (solver, &solve_status));
        if (solve_status != AMGX_SOLVE_SUCCESS)
          printf ("AMGX: solver did not converge (status = %d)\n", solve_status);

        AMGX_SAFE_CALL (AMGX_vector_download (amgx_x, global_phi.data ()));

        AMGX_SAFE_CALL (AMGX_solver_destroy (solver));
        AMGX_SAFE_CALL (AMGX_vector_destroy (amgx_x));
        AMGX_SAFE_CALL (AMGX_vector_destroy (amgx_b));
        AMGX_SAFE_CALL (AMGX_matrix_destroy (amgx_A));
        AMGX_SAFE_CALL (AMGX_resources_destroy (rsrc));
        AMGX_SAFE_CALL (AMGX_config_destroy (cfg));

        AMGX_SAFE_CALL (AMGX_finalize_plugins ());
        AMGX_SAFE_CALL (AMGX_finalize ());
      }

      // Scatter solution from rank 0 to all ranks
      phi = std::make_unique<distributed_vector> (n, mpicomm);
      MPI_Scatterv (global_phi.data (), all_n.data (), n_displs.data (), MPI_DOUBLE,
                    phi->get_owned_data ().data (), n, MPI_DOUBLE,
                    0, mpicomm);

      bim3a_solution_with_ghosts (tmsh, *phi, replace_op);
    }
}

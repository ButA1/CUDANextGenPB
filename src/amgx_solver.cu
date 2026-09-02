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
#include <cstdint>
#include <cuda_runtime.h>
#include <amgx_c.h>

#include <bim_distributed_vector.h>
#include <quad_operators_3d.h>

// Built-in fallback AMGX config (single source of truth for all paths).
static const char *NGPB_AMGX_DEFAULT_CFG =
    "config_version=2, solver(main)=PCG, main:max_iters=1000, "
    "main:tolerance=1e-10, main:norm=L2, main:convergence=RELATIVE_INI_CORE, "
    "main:monitor_residual=1, main:print_solve_stats=1, main:obtain_timings=1, "
    "main:preconditioner(amg)=BLOCK_JACOBI, amg:max_iters=1";

// Create an AMGX config handle. If cfg_file is non-empty, load it; on any load
// failure, warn (rank 0) and fall back to NGPB_AMGX_DEFAULT_CFG.
static AMGX_config_handle
create_amgx_config (const std::string &cfg_file, int rank)
{
  AMGX_config_handle cfg = nullptr;
  if (!cfg_file.empty ())
    {
      AMGX_RC rc = AMGX_config_create_from_file (&cfg, cfg_file.c_str ());
      if (rc == AMGX_RC_OK)
        {
          if (rank == 0)
            std::cout << "AMGX: using config file " << cfg_file << std::endl;
          return cfg;
        }
      if (rank == 0)
        std::cerr << "AMGX: could not load config file '" << cfg_file
                  << "' (rc=" << rc << "); falling back to built-in config"
                  << std::endl;
      cfg = nullptr;   // ensure clean handle before the fallback create
    }
  AMGX_SAFE_CALL (AMGX_config_create (&cfg, NGPB_AMGX_DEFAULT_CFG));
  return cfg;
}

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

  if (rank == 0)
    std::cout << "Sparse matrix size: n = " << n_global
              << ", nnz = " << nnz << std::endl;

  A.reset ();

  std::vector<double> rhs_local = rhs->get_owned_data ();
  rhs.reset ();

  if (size == 1)
    {
      // --- Single rank: solve directly ---
      AMGX_SAFE_CALL (AMGX_initialize ());
      AMGX_SAFE_CALL (AMGX_initialize_plugins ());

      AMGX_config_handle cfg = create_amgx_config (amgx_config_file, rank);

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

        AMGX_config_handle cfg = create_amgx_config (amgx_config_file, rank);

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

void
poisson_boltzmann::amgx_compute_electric_potential_dist (ray_cache_t & ray_cache)
{
  // Multi-GPU path (selected when gpu_topo.total_gpus > 1). Each physical GPU
  // owns a contiguous range of MPI ranks (block mapping, see gpu_topology.cu).
  // We "sub-gather" every group's CSR to that group's leader, then run a single
  // distributed AMGX solve across the G leaders (one MPI rank per GPU). Because
  // the CSR columns are already GLOBAL indices, a row owned by one group can
  // reference a column owned by another with no remapping -- AMGX builds the
  // halo exchange from the partition offsets. Assembly stays MPI-wide; only the
  // solve is grouped. This generalizes the gather-to-rank-0 path: a 1-rank group
  // makes the sub-gather a no-op, and a single group reduces to the old behavior.

  MPI_Comm group_comm  = gpu_topo.group_comm;
  MPI_Comm solver_comm = gpu_topo.solver_comm;   // MPI_COMM_NULL on non-leaders

  int grank = 0, gsize = 1;
  MPI_Comm_rank (group_comm, &grank);
  MPI_Comm_size (group_comm, &gsize);
  const bool is_leader = (grank == 0);

  int world_rank = 0;
  MPI_Comm_rank (mpicomm, &world_rank);

  // --- CSR extraction (local rows, global column indices) ---
  std::vector<double> vals;
  std::vector<int> irow, jcol;
  (*A).csr (vals, jcol, irow);

  int nnz      = (*A).owned_nnz ();
  int n        = tmsh.num_owned_nodes ();
  int n_global = tmsh.num_global_nodes ();

  if (world_rank == 0)
    std::cout << "Sparse matrix size: n = " << n_global
              << ", nnz (rank 0) = " << nnz
              << "  [distributed AMGX across " << gpu_topo.total_gpus
              << " GPUs]" << std::endl;

  A.reset ();

  std::vector<double> rhs_local = rhs->get_owned_data ();
  rhs.reset ();

  // ================================================================
  // Sub-gather: within group_comm, collect all members' CSR at the leader.
  // Same index arithmetic as the gather-to-rank-0 path, scoped to the group.
  // ================================================================
  std::vector<int> all_n (gsize, 0), all_nnz (gsize, 0);
  MPI_Gather (&n,   1, MPI_INT, all_n.data (),   1, MPI_INT, 0, group_comm);
  MPI_Gather (&nnz, 1, MPI_INT, all_nnz.data (), 1, MPI_INT, 0, group_comm);

  std::vector<int> n_displs (gsize + 1, 0);
  std::vector<int> nnz_displs (gsize + 1, 0);
  if (is_leader)
    for (int i = 0; i < gsize; ++i) {
      n_displs[i + 1]   = n_displs[i]   + all_n[i];
      nnz_displs[i + 1] = nnz_displs[i] + all_nnz[i];
    }

  const int group_n   = is_leader ? n_displs[gsize]   : 0;
  const int group_nnz = is_leader ? nnz_displs[gsize] : 0;

  // Gather RHS
  std::vector<double> group_rhs;
  if (is_leader) group_rhs.resize (group_n);
  MPI_Gatherv (rhs_local.data (), n, MPI_DOUBLE,
               group_rhs.data (), all_n.data (), n_displs.data (),
               MPI_DOUBLE, 0, group_comm);
  std::vector<double> ().swap (rhs_local);

  // Gather column indices (already GLOBAL -- no adjustment) and values.
  // AMGX's upload_distributed reads global column indices as 64-bit by default,
  // and the known-good classical example (capi_cla.c) uses 64-bit columns +
  // 64-bit partition offsets. Convert the local columns to int64 BEFORE gathering
  // so the leader never holds both an int32 and an int64 copy at once.
  std::vector<int64_t> jcol64 (jcol.begin (), jcol.end ());
  std::vector<int> ().swap (jcol);

  std::vector<int64_t> group_jcol;
  std::vector<double> group_vals;
  if (is_leader) {
    group_jcol.resize (group_nnz);
    group_vals.resize (group_nnz);
  }
  MPI_Gatherv (jcol64.data (), nnz, MPI_INT64_T,
               group_jcol.data (), all_nnz.data (), nnz_displs.data (),
               MPI_INT64_T, 0, group_comm);
  MPI_Gatherv (vals.data (), nnz, MPI_DOUBLE,
               group_vals.data (), all_nnz.data (), nnz_displs.data (),
               MPI_DOUBLE, 0, group_comm);
  std::vector<int64_t> ().swap (jcol64);
  std::vector<double> ().swap (vals);

  // Gather row pointers (each rank sends the first n entries of its n+1 irow)
  std::vector<int> gathered_irow;
  if (is_leader) gathered_irow.resize (group_n);
  MPI_Gatherv (irow.data (), n, MPI_INT,
               gathered_irow.data (), all_n.data (), n_displs.data (),
               MPI_INT, 0, group_comm);
  std::vector<int> ().swap (irow);

  // Rebuild the group's row pointers by adding each member's nnz offset.
  std::vector<int> group_irow;
  if (is_leader) {
    group_irow.resize (group_n + 1);
    for (int r = 0; r < gsize; ++r)
      for (int i = 0; i < all_n[r]; ++i)
        group_irow[n_displs[r] + i] = gathered_irow[n_displs[r] + i] + nnz_displs[r];
    group_irow[group_n] = group_nnz;
    std::vector<int> ().swap (gathered_irow);
  }

  // ================================================================
  // Distributed AMGX solve across the G leaders (solver_comm).
  // ================================================================
  std::vector<double> group_phi;
  if (is_leader) {
    group_phi.resize (group_n, 0.0);

    int G = 0, my_group = 0;
    MPI_Comm_size (solver_comm, &G);
    MPI_Comm_rank (solver_comm, &my_group);

    // Partition offsets (int64, as AMGX_DIST_PARTITION_OFFSETS expects): global
    // row start of each leader (length G+1). Leaders are ordered by ascending
    // world rank, which (with block mapping + p4est's per-rank contiguous global
    // numbering) equals ascending global row start.
    std::vector<int64_t> partition_offsets (G + 1, 0);
    int64_t group_n64 = group_n;
    MPI_Allgather (&group_n64, 1, MPI_INT64_T,
                   partition_offsets.data () + 1, 1, MPI_INT64_T, solver_comm);
    for (int i = 0; i < G; ++i)
      partition_offsets[i + 1] += partition_offsets[i];

    // Contiguity guard (R1): the partition must tile exactly [0, n_global).
    if (partition_offsets[G] != (int64_t) n_global) {
      fprintf (stderr,
               "[amgx-dist] partition offsets sum (%lld) != n_global (%d): "
               "non-contiguous rank->row mapping. Block rank distribution is "
               "required (Slurm --distribution=block).\n",
               (long long) partition_offsets[G], n_global);
      MPI_Abort (mpicomm, 1);
    }

    AMGX_SAFE_CALL (AMGX_initialize ());
    AMGX_SAFE_CALL (AMGX_initialize_plugins ());

    AMGX_config_handle cfg = create_amgx_config (amgx_config_file, world_rank);

    AMGX_resources_handle rsrc;
    int device_id = gpu_topo.my_device;   // this leader's bound physical device
    AMGX_SAFE_CALL (AMGX_resources_create (&rsrc, cfg, &solver_comm, 1, &device_id));

    AMGX_matrix_handle amgx_A;
    AMGX_vector_handle amgx_b;
    AMGX_vector_handle amgx_x;
    AMGX_solver_handle solver;

    AMGX_SAFE_CALL (AMGX_matrix_create (&amgx_A, rsrc, AMGX_mode_dDDI));
    AMGX_SAFE_CALL (AMGX_vector_create (&amgx_b, rsrc, AMGX_mode_dDDI));
    AMGX_SAFE_CALL (AMGX_vector_create (&amgx_x, rsrc, AMGX_mode_dDDI));
    AMGX_SAFE_CALL (AMGX_solver_create (&solver, rsrc, AMGX_mode_dDDI, cfg));

    // Distribution: 64-bit GLOBAL column indices (AMGX default) + 64-bit
    // partition offsets, matching AMGX's classical distributed example
    // (examples/amgx_mpi_capi_cla.c). Do NOT set 32-bit col indices here -- the
    // columns were already widened to int64 above.
    AMGX_distribution_handle dist;
    AMGX_SAFE_CALL (AMGX_distribution_create (&dist, cfg));
    AMGX_SAFE_CALL (AMGX_distribution_set_partition_data (
      dist, AMGX_DIST_PARTITION_OFFSETS, partition_offsets.data ()));

    AMGX_SAFE_CALL (AMGX_matrix_upload_distributed (
      amgx_A, n_global, group_n, group_nnz, 1, 1,
      group_irow.data (), group_jcol.data (), group_vals.data (),
      NULL, dist));

    AMGX_SAFE_CALL (AMGX_distribution_destroy (dist));
    std::vector<int> ().swap (group_irow);
    std::vector<int64_t> ().swap (group_jcol);
    std::vector<double> ().swap (group_vals);

    // Bind the vectors to the distributed matrix BEFORE uploading their data, so
    // AMGX sizes them to (owned rows + halo/ghost elements). Without this the
    // vectors are only group_n long and the SpMV halo exchange overflows
    // ("Vector size too small: not enough space for halo elements").
    AMGX_SAFE_CALL (AMGX_vector_bind (amgx_b, amgx_A));
    AMGX_SAFE_CALL (AMGX_vector_bind (amgx_x, amgx_A));

    AMGX_SAFE_CALL (AMGX_vector_upload (amgx_b, group_n, 1, group_rhs.data ()));
    std::vector<double> ().swap (group_rhs);

    AMGX_SAFE_CALL (AMGX_vector_upload (amgx_x, group_n, 1, group_phi.data ()));

    AMGX_SAFE_CALL (AMGX_solver_setup (solver, amgx_A));
    AMGX_SAFE_CALL (AMGX_solver_solve (solver, amgx_b, amgx_x));

    AMGX_SOLVE_STATUS solve_status;
    AMGX_SAFE_CALL (AMGX_solver_get_status (solver, &solve_status));
    if (solve_status != AMGX_SOLVE_SUCCESS && world_rank == 0)
      printf ("AMGX: solver did not converge (status = %d)\n", solve_status);

    AMGX_SAFE_CALL (AMGX_vector_download (amgx_x, group_phi.data ()));

    AMGX_SAFE_CALL (AMGX_solver_destroy (solver));
    AMGX_SAFE_CALL (AMGX_vector_destroy (amgx_x));
    AMGX_SAFE_CALL (AMGX_vector_destroy (amgx_b));
    AMGX_SAFE_CALL (AMGX_matrix_destroy (amgx_A));
    AMGX_SAFE_CALL (AMGX_resources_destroy (rsrc));
    AMGX_SAFE_CALL (AMGX_config_destroy (cfg));

    AMGX_SAFE_CALL (AMGX_finalize_plugins ());
    AMGX_SAFE_CALL (AMGX_finalize ());
  }

  // ================================================================
  // Scatter each leader's solution back to its group members.
  // (sendcounts/displs are only read at the group root = leader.)
  // ================================================================
  phi = std::make_unique<distributed_vector> (n, mpicomm);
  MPI_Scatterv (is_leader ? group_phi.data () : nullptr,
                all_n.data (), n_displs.data (), MPI_DOUBLE,
                phi->get_owned_data ().data (), n, MPI_DOUBLE,
                0, group_comm);

  bim3a_solution_with_ghosts (tmsh, *phi, replace_op);
}

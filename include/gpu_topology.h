/*
 *  Node-local GPU topology and rank->device binding for multi-GPU runs.
 *
 *  Computed once at startup (setup_gpu_topology, called from main() right after
 *  MPI_Init) and then read by both the AMGX solver (to auto-select gather-to-
 *  rank-0 vs. the distributed path) and the CUDA energy path (which simply
 *  inherits the bound device).
 *
 *  Ranks are BLOCK-mapped onto the GPUs of their node so that each GPU owns a
 *  CONTIGUOUS range of MPI ranks -> a contiguous range of global matrix rows.
 *  That contiguity is required by AMGX_DIST_PARTITION_OFFSETS. Many ranks may
 *  share one GPU (e.g. 16 ranks / 2 GPUs -> 8 ranks per GPU): the first rank of
 *  each GPU's range is the "leader" that owns the distributed solve for that GPU.
 */

#ifndef NGPB_GPU_TOPOLOGY_H
#define NGPB_GPU_TOPOLOGY_H

#include <mpi.h>

struct gpu_topology
{
  int total_gpus = 1;   // distinct GPUs in use across the whole job (drives AMGX path choice)
  int my_device  = 0;   // physical CUDA device this rank was bound to
  int is_leader  = 1;   // 1 if this rank leads its GPU group (owns the distributed solve slice)
  int group_id   = 0;   // this rank's GPU-group index == leader's rank in solver_comm (0..G-1)

  MPI_Comm node_comm   = MPI_COMM_NULL;  // ranks sharing this physical node
  MPI_Comm group_comm  = MPI_COMM_NULL;  // ranks sharing this physical GPU (leader == rank 0)
  MPI_Comm solver_comm = MPI_COMM_NULL;  // the G leaders (MPI_COMM_NULL on non-leaders)
};

/*
 * Detect the node-local GPU count, bind this rank's CUDA device (block mapping),
 * and build the node / group / solver communicators. Returns the populated
 * topology by value.
 *
 * Env override NGPB_GPUS_PER_NODE: when set, the topology math (grouping,
 * leaders, total_gpus) pretends the node has that many GPUs, while the physical
 * cudaSetDevice is clamped to the real device count. This lets the full
 * multi-GPU / distributed-AMGX code path be exercised on a single-GPU box
 * (everything lands on device 0) for correctness testing without the cluster.
 */
gpu_topology setup_gpu_topology (MPI_Comm world);

#endif /* NGPB_GPU_TOPOLOGY_H */

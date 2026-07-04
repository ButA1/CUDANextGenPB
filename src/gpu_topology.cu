/*
 *  Node-local GPU topology + rank->device binding. See include/gpu_topology.h
 *  for the design rationale (block mapping -> contiguous global-row ranges per
 *  GPU, leaders own the distributed AMGX slice, NGPB_GPUS_PER_NODE override for
 *  single-GPU correctness testing).
 */

#include "gpu_topology.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Multiplier separating node id from local GPU index in the group color.
// Must exceed the largest plausible GPUs-per-node; kept small so that
// node_id * MAX_GPUS_PER_NODE stays well inside int for any realistic job.
static const int MAX_GPUS_PER_NODE = 1024;

gpu_topology
setup_gpu_topology (MPI_Comm world)
{
  gpu_topology t;

  int world_rank = 0, world_size = 1;
  MPI_Comm_rank (world, &world_rank);
  MPI_Comm_size (world, &world_size);

  // --- node-local communicator (ranks that share physical memory) ---
  MPI_Comm node_comm;
  MPI_Comm_split_type (world, MPI_COMM_TYPE_SHARED, world_rank,
                       MPI_INFO_NULL, &node_comm);
  int local_rank = 0, local_size = 1;
  MPI_Comm_rank (node_comm, &local_rank);
  MPI_Comm_size (node_comm, &local_size);
  t.node_comm = node_comm;

  // --- physical GPU count on this node (respects CUDA_VISIBLE_DEVICES/Slurm) ---
  int real_n_dev = 0;
  if (cudaGetDeviceCount (&real_n_dev) != cudaSuccess)
    real_n_dev = 0;

  // NGPB_GPUS_PER_NODE overrides the count used for the topology math ONLY, so
  // the full multi-GPU code path can be exercised on a single-GPU box. The
  // physical device (below) is clamped to the real count regardless.
  int n_dev = real_n_dev;
  if (const char *env = std::getenv ("NGPB_GPUS_PER_NODE"))
    {
      int v = std::atoi (env);
      if (v > 0)
        n_dev = v;
    }
  if (n_dev < 1)
    n_dev = 1;   // no-GPU / fallback: one logical device -> single-GPU (gather) path

  // --- BLOCK mapping: each GPU owns a contiguous range of local ranks ---
  int ranks_per_gpu = (local_size + n_dev - 1) / n_dev;   // ceil
  if (ranks_per_gpu < 1)
    ranks_per_gpu = 1;
  int my_gpu = local_rank / ranks_per_gpu;                // 0..n_dev-1
  if (my_gpu > n_dev - 1)
    my_gpu = n_dev - 1;                                   // defensive (ceil already guarantees this)

  // Physical device: clamp to the real GPU count so a faked NGPB_GPUS_PER_NODE
  // still runs -- everything simply lands on device 0 when only one GPU exists.
  int phys_dev = my_gpu;
  if (real_n_dev > 0)
    {
      if (phys_dev > real_n_dev - 1)
        phys_dev = real_n_dev - 1;
      cudaSetDevice (phys_dev);
    }
  t.my_device = phys_dev;

  // --- unique node id = world rank of this node's rank-0, shared within node ---
  int node_id = world_rank;
  MPI_Bcast (&node_id, 1, MPI_INT, 0, node_comm);

  // --- group_comm: all ranks sharing one physical GPU (leader = rank 0) ---
  int group_color = node_id * MAX_GPUS_PER_NODE + my_gpu;
  MPI_Comm group_comm;
  MPI_Comm_split (world, group_color, world_rank, &group_comm);
  t.group_comm = group_comm;

  int group_rank = 0;
  MPI_Comm_rank (group_comm, &group_rank);
  int is_leader = (group_rank == 0) ? 1 : 0;
  t.is_leader = is_leader;

  // --- solver_comm: the G leaders, ordered by world rank (== ascending global
  //     row start, since ranks are block-mapped and rows are numbered by rank) ---
  MPI_Comm solver_comm;
  MPI_Comm_split (world, is_leader ? 0 : MPI_UNDEFINED, world_rank, &solver_comm);
  t.solver_comm = solver_comm;   // MPI_COMM_NULL on non-leaders

  // --- distinct GPUs in use across the whole job ---
  int total_gpus = 0;
  MPI_Allreduce (&is_leader, &total_gpus, 1, MPI_INT, MPI_SUM, world);
  t.total_gpus = total_gpus;

  // --- this group's global index (leader's rank among all leaders) ---
  int group_id = 0;
  if (is_leader)
    MPI_Comm_rank (solver_comm, &group_id);
  MPI_Bcast (&group_id, 1, MPI_INT, 0, group_comm);   // share to the whole group
  t.group_id = group_id;

  if (world_rank == 0)
    {
      std::printf ("[gpu_topology] world_size=%d  gpus_in_use=%d  "
                   "(physical GPUs/node=%d", world_size, total_gpus, real_n_dev);
      if (n_dev != real_n_dev)
        std::printf (", NGPB_GPUS_PER_NODE override=%d", n_dev);
      std::printf (", ranks/gpu~%d)\n", ranks_per_gpu);
    }

  return t;
}

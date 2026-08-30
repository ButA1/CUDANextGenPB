#!/bin/bash
#SBATCH --job-name=ngpb-scale-2node
#SBATCH --partition=gpu,gpu_short
#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=8            # 8 ranks + 1 GPU per node -> 8 ranks/GPU, the SAME
#SBATCH --gres=gpu:a100:1              # per node -> 2 GPUs total, 2 leaders on 2 NODES,
#SBATCH --cpus-per-task=1              # which puts the AMGX halo exchange on the network
#SBATCH --distribution=block           # contiguous ranks per node -- required, see below
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=scale-2node-%j.out
#SBATCH --error=scale-2node-%j.err
#
# 3 of 3, and the fragile one. It differs from the 2gpu job ONLY in where the second
# GPU lives, so the pair isolates network cost from multi-GPU cost.
#
# --distribution=block is load-bearing, not tidiness: setup_gpu_topology block-maps
# ranks onto node-local GPUs and AMGX_DIST_PARTITION_OFFSETS needs each GPU to own a
# CONTIGUOUS range of global rows. A cyclic distribution interleaves ranks across
# nodes and the partition stops being contiguous -- which does not crash, it returns
# a wrong polarization energy. See check 3 in the job output.
#
# The launch is srun + one container per rank with pml=ucx, NOT mpirun-in-container;
# scaling_body.sh explains why the default hangs.
#
# This is the only topology that needs a HOST python >= 3.8 (bench_sweep.py forks srun,
# which a containerised process cannot do). scaling_body.sh loads it itself; override the
# module name with PYTHON_MODULE= if python/3.12 is not the right one here.

TOPO=2node
MPI_MODE=srun
source "$SLURM_SUBMIT_DIR/scaling_body.sh"

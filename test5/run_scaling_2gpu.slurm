#!/bin/bash
#SBATCH --job-name=ngpb-scale-2gpu
#SBATCH --partition=gpu,gpu_short      # 2xA100 nodes: gpu022 (gpu) or gpu026 (gpu_short)
#SBATCH --gres=gpu:a100:2              # both GPUs, ONE node -- no network in the AMGX halo
#SBATCH --nodes=1
#SBATCH --ntasks=16                    # -> 8 ranks per GPU, matching the 2-node layout so
#SBATCH --ntasks-per-node=16           # those two differ only in where the second GPU is
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=scale-2gpu-%j.out
#SBATCH --error=scale-2gpu-%j.err
#
# 2 of 3. Do NOT add --gpu-bind: both GPUs must stay visible to all 16 ranks so
# setup_gpu_topology can block-map them itself (contiguous ranks per GPU is what
# AMGX_DIST_PARTITION_OFFSETS requires).

TOPO=2gpu
MPI_MODE=mpirun
source "$SLURM_SUBMIT_DIR/scaling_body.sh"

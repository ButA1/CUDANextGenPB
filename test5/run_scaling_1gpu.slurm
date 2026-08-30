#!/bin/bash
#SBATCH --job-name=ngpb-scale-1gpu
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:a100:1              # the BASELINE every speedup is divided by
#SBATCH --nodes=1
#SBATCH --ntasks=16                    # held constant across all three topologies, so the
#SBATCH --ntasks-per-node=16           # only thing changing is the number of GPUs
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=scale-1gpu-%j.out
#SBATCH --error=scale-1gpu-%j.err
#
# 1 of 3. Submit all three; they are independent.
#   sbatch run_scaling_1gpu.slurm ; sbatch run_scaling_2gpu.slurm ; sbatch run_scaling_2node.slurm
# Knobs: MOL= PRM= SIF= RUNDIR= REPEATS= RESUME_RUN=1 AMGX_CFG=

TOPO=1gpu
MPI_MODE=mpirun
source "$SLURM_SUBMIT_DIR/scaling_body.sh"

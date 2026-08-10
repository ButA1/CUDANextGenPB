#!/bin/bash
#SBATCH --job-name=ngpb-h1n1
#SBATCH --partition=gpu,gpu_short      # widen the pool: gpu022 (gpu) OR gpu025/gpu026 (gpu_short)
#SBATCH --gres=gpu:a100:2              # 2x A100 40GB -> distributed AMGX (~19 GB/card at 150M
                                       # unknowns). One card would force the gather path and not fit.
#SBATCH --nodes=1                      # NanoShaper cannot distribute across nodes (threads only)
#SBATCH --ntasks=16                    # 16 ranks for the MPI assembly phases (the bulk of runtime)
#SBATCH --cpus-per-task=2              # 32 of the node's 40 logical CPUs
#SBATCH --distribution=block           # REQUIRED: distributed AMGX assumes each GPU group owns a
                                       # CONTIGUOUS global row range. A cyclic mapping returns a
                                       # silently wrong answer; the guard at amgx_solver.cu:409 is
                                       # only a sum check and cannot detect it.
#SBATCH --mem=0                        # ALL memory on the node. Never write 512G: that is 524288
                                       # MiB, above the node's configured RealMemory, which is what
                                       # "Memory specification can not be satisfied" meant.
#SBATCH --exclusive                    # whole node -> the step's cgroup holds every core, which is
                                       # what lets NanoShaper's threads spread (see below)
#SBATCH --time=08:00:00                # short enough to backfill quickly; bump if it TIMEOUTs
#SBATCH --output=h1n1-%j.out
#SBATCH --error=h1n1-%j.err

set -euo pipefail

module load singularity

cd "$SLURM_SUBMIT_DIR"
SIF="${SIF:-$SLURM_SUBMIT_DIR/ngpb.sif}"
PRM="${PRM:-options_h1n1.prm}"

echo "Node   : $(hostname)"
echo "CPUs   : ${SLURM_NTASKS} ranks x ${SLURM_CPUS_PER_TASK}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo

# OMP_NUM_THREADS is hard-set to 1 at recipe_cuda.def:23 in %environment, which is sourced
# AFTER any --env flag -- so it can only be overridden from inside, here.
#
# 16 threads on rank 0 deliberately exceeds --cpus-per-task=2: NanoShaper runs alone while
# the other 15 ranks sit in MPI_Barrier (poisson_boltzmann.cpp:119-138), and --bind-to none
# leaves them unpinned, so those threads spread over the whole cgroup instead of sharing two
# cores. Keep this equal to number_of_threads in the prm.
singularity exec --nv "$SIF" bash -c "
    export OMP_NUM_THREADS=16
    exec mpirun -n ${SLURM_NTASKS} --bind-to none ngpb --prmfile ${PRM}
"

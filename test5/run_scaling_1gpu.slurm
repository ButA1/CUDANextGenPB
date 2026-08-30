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
BODY="$SLURM_SUBMIT_DIR/scaling_body.sh"
if [ ! -r "$BODY" ]; then
    echo "MISSING: $BODY" >&2
    echo "" >&2
    echo "This wrapper only sets the SBATCH directives; the actual job lives in" >&2
    echo "scaling_body.sh, which all three scaling jobs share. Copy it across:" >&2
    echo "" >&2
    echo "    scp scaling_body.sh run_scaling_*.slurm <cluster>:$SLURM_SUBMIT_DIR/" >&2
    echo "    scp -r scripts <cluster>:$SLURM_SUBMIT_DIR/" >&2
    echo "" >&2
    echo "scripts/ too: --sweeps s and --mpi-cmd are recent additions to" >&2
    echo "bench_sweep.py and an older copy rejects both." >&2
    exit 1
fi
source "$BODY"

#!/bin/bash
#SBATCH --job-name=ngpb-2gpu
#SBATCH --partition=gpu,gpu_short      # both general 2xA100 nodes: gpu022 (gpu) OR gpu026 (gpu_short)
                                       # NO --nodelist: let Slurm take whichever 2xA100 node frees first.
                                       # (Then pin the baseline to whatever node THIS run reports.)
#SBATCH --gres=gpu:a100:2              # <-- multi-GPU: BOTH A100s (only gres differs from baseline)
#SBATCH --nodes=1
#SBATCH --ntasks=16                    # same rank count as baseline -> 8 ranks/group, one group per GPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=01:00:00                # short -> backfills into gaps faster. Bump if it TIMEOUTs.
#SBATCH --output=multigpu-%j.out
#SBATCH --error=multigpu-%j.err

set -euo pipefail
module load singularity                # verify name once: module avail singularity

BASE="$SLURM_SUBMIT_DIR"               # the test5 folder
SIF="${SIF:-$BASE/ngpb.sif}"
RUNDIR="$BASE/multigpu_2gpu"           # own dir so outputs don't clash with the baseline run

mkdir -p "$RUNDIR"
cp "$BASE/6VYB.pqr" "$BASE/options.prm" "$BASE/amgx_pcg_amg_cluster.json" "$RUNDIR/"
cd "$RUNDIR"

echo "host=$(hostname)  ntasks=$SLURM_NTASKS  gres=2xA100"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true   # expect TWO GPUs

# Both GPUs visible to all 16 ranks; AMGX groups ranks by device and gathers per group,
# energy kernels launch per-rank. GPUs independent (no NVLink needed). Do NOT add --gpu-bind.
singularity exec --nv "$SIF" \
    mpirun -n "$SLURM_NTASKS" ngpb --prmfile options.prm
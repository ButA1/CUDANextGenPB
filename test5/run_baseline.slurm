#!/bin/bash
#SBATCH --job-name=ngpb-1gpu
# IMPORTANT: set BOTH lines below to match the node the multigpu run landed on:
#   gpu022 -> --partition=gpu ;  gpu026 -> --partition=gpu_short
#SBATCH --partition=gpu_short          # <-- match multigpu's node's partition
#SBATCH --nodelist=gpu026              # <-- match multigpu's node -> only GPU count varies
#SBATCH --gres=gpu:a100:1              # <-- baseline: ONE A100
#SBATCH --nodes=1
#SBATCH --ntasks=16                    # held constant vs multigpu: isolates the GPU-count effect
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=baseline-%j.out
#SBATCH --error=baseline-%j.err

set -euo pipefail
module load singularity                # verify name once: module avail singularity

BASE="$SLURM_SUBMIT_DIR"               # the test5 folder
SIF="${SIF:-$BASE/ngpb.sif}"
RUNDIR="$BASE/baseline_1gpu"           # own dir so outputs don't clash with the 2-GPU run

mkdir -p "$RUNDIR"
cp "$BASE/6VYB.pqr" "$BASE/options.prm" "$BASE/amgx_pcg_amg_cluster.json" "$RUNDIR/"
cd "$RUNDIR"

echo "host=$(hostname)  ntasks=$SLURM_NTASKS  gres=1xA100"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true   # expect ONE GPU

# Both GPUs (here: one) left visible to all ranks -- the code does its own cudaSetDevice grouping.
# Do NOT add --gpu-bind.
singularity exec --nv "$SIF" \
    mpirun -n "$SLURM_NTASKS" ngpb --prmfile options.prm
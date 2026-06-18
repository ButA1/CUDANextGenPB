#!/bin/bash
#SBATCH --job-name=ngpb-leafsweep
#SBATCH --partition=gpu_short          # 24h cap; good for a sweep of short runs
#SBATCH --gres=gpu:a100:1              # each array task gets its own A100
#SBATCH --nodes=1
#SBATCH --ntasks=16                    # CPU ranks for assembly, per task
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --array=0-4%2                  # 5 values; %2 = at most 2 running at once (A100s are scarce)
#SBATCH --output=sweep-%A_%a.out       # %A=array job id, %a=task id
#SBATCH --error=sweep-%A_%a.err

set -euo pipefail
module load singularity

# --- The sweep: one bh_leaf_size per array index ---
LEAF_SIZES=(64 128 256 512 1024)
LEAF=${LEAF_SIZES[$SLURM_ARRAY_TASK_ID]}

BASE="$SLURM_SUBMIT_DIR"               # the test5 folder
SIF="${SIF:-$BASE/ngpb.sif}"
RUNDIR="$BASE/leaf_${LEAF}"

# Per-task working dir: own options.prm + own copy of the pqr, so outputs never collide.
mkdir -p "$RUNDIR"
cp "$BASE/6VYB.pqr" "$RUNDIR/"
sed "s/^bh_leaf_size.*/bh_leaf_size  = ${LEAF}/" "$BASE/options.prm" > "$RUNDIR/options.prm"

cd "$RUNDIR"
echo "Task $SLURM_ARRAY_TASK_ID : bh_leaf_size=$LEAF  host=$(hostname)"

singularity exec --nv "$SIF" \
    mpirun -n "$SLURM_NTASKS" ngpb --prmfile options.prm

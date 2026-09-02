#!/bin/bash
#SBATCH --job-name=ngpb-6vyb
#SBATCH --partition=gpu_short          # gpu_short=24h (iteration) | gpu=7d (long runs)
#SBATCH --gres=gpu:a100:1              # A100 40GB (general access). P100: gpu:tesla:1
#SBATCH --nodes=1                      # single node: AMGX solves on 1 GPU (gather-to-rank-0)
#SBATCH --ntasks=16                    # MPI ranks for the CPU assembly phase (A100 nodes = 20c/40t)
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G                     # ~1.8 GB/rank assembly; 512G available on the node
#SBATCH --time=04:00:00
#SBATCH --output=ngpb-%j.out
#SBATCH --error=ngpb-%j.err

set -euo pipefail

# The cluster provides singularity/apptainer as a MODULE -- not on PATH by default in a
# batch job (tutorial slide warns about this). Load it before any singularity call.
# Verify the exact name once with: module avail singularity ; module avail apptainer
module load singularity

# Self-contained case dir (this folder, test5): holds ngpb.sif, options.prm, 6VYB.pqr.
# Outputs are written here too (cwd is on writable scratch/home, not the read-only image).
cd "$SLURM_SUBMIT_DIR"
SIF="${SIF:-$SLURM_SUBMIT_DIR/ngpb.sif}"

echo "Node: $(hostname)  |  GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# Run mpirun INSIDE the container (its own OpenMPI) -- robust for single node.
# --nv bind-mounts the host NVIDIA driver; the CUDA toolkit comes from the image.
# singularity auto-mounts $PWD, so 6VYB.pqr (referenced relatively in options.prm) is visible.
singularity exec --nv "$SIF" \
    mpirun -n "$SLURM_NTASKS" ngpb --prmfile options.prm
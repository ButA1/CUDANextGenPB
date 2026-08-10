#!/bin/bash
#SBATCH --job-name=ngpb-h1n1
#SBATCH --partition=gpu,gpu_short      # widen the pool: gpu022 (gpu) OR gpu025/gpu026 (gpu_short)
#SBATCH --gres=gpu:a100:2              # 2x A100 40GB -> 80GB device for the DISTRIBUTED AMGX path.
                                       # ~150M unknowns needs ~37GB; the gather-to-rank-0 path
                                       # (amgx_solver.cu:142) would not fit on a single 40GB card.
#SBATCH --nodes=1                      # NanoShaper cannot distribute across nodes (v1.5 is
                                       # threads-only; the MPI version is future work in the paper).
#SBATCH --ntasks=8                     # 8 ranks x 5 cpus = 40 logical CPUs on a 20c/40t A100 node
#SBATCH --cpus-per-task=5              # rank 0 needs these for NanoShaper's OpenMP phase
#SBATCH --distribution=block           # REQUIRED by the distributed AMGX path: leaders are ordered
                                       # by world rank and their partition offsets assume each GPU
                                       # group owns a CONTIGUOUS global row range. A cyclic mapping
                                       # would build wrong halos and return a silently wrong answer
                                       # -- the guard at amgx_solver.cu:409 is only a sum check and
                                       # cannot detect it. Contiguous by construction on 1 node, but
                                       # set explicitly so this survives a move to >1 node.
#SBATCH --mem=0                        # ALL memory on the node. Do NOT write 512G: that is
                                       # 524288 MiB, above the node's configured RealMemory, which
                                       # is what "Memory specification can not be satisfied" means.
#SBATCH --exclusive
#SBATCH --time=12:00:00                # valid under both gpu (7d) and gpu_short (24h) caps
#SBATCH --output=h1n1-%j.out
#SBATCH --error=h1n1-%j.err

set -euo pipefail

module load singularity

cd "$SLURM_SUBMIT_DIR"
SIF="${SIF:-$SLURM_SUBMIT_DIR/ngpb.sif}"
PRM="${PRM:-options_h1n1.prm}"

# --- Pre-flight: H1N1.pqr lives in ../data, not here. singularity auto-mounts $PWD only,
# --- and the prm references the file relatively, so it must be in this directory.
for f in "$SIF" "$PRM" H1N1.pqr; do
    [ -e "$f" ] || { echo "MISSING: $f" >&2
                     [ "$f" = "H1N1.pqr" ] && echo "  -> cp ../data/H1N1.pqr ." >&2
                     exit 1; }
done

echo "Node    : $(hostname)"
echo "Memory  : $(awk '/MemTotal/{printf "%.0f GB total", $2/1048576}' /proc/meminfo)"
echo "CPUs    : ${SLURM_NTASKS} ranks x ${SLURM_CPUS_PER_TASK} cpus"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo

# recipe_cuda.def:23 hard-sets OMP_NUM_THREADS=1 in %environment, which is sourced AFTER any
# --env flag -- so it cannot be overridden from outside. Re-exporting inside `bash -c` runs
# after that sourcing and does stick. NanoShaper's thread count is also set explicitly from
# the prm file's number_of_threads, so keep the two values in agreement.
#
# --bind-to none: OpenMPI would otherwise pin each rank so tightly that rank 0's NanoShaper
# threads all timeshare one core. Slurm's cgroup still confines the step to its 40 CPUs.
/usr/bin/time -v singularity exec --nv "$SIF" bash -c "
    export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
    exec mpirun -n ${SLURM_NTASKS} --bind-to none ngpb --prmfile ${PRM}
"

echo
echo "=== peak memory: check 'Maximum resident set size' above (rank 0 / NanoShaper) ==="
echo "=== job-wide:  sacct -j ${SLURM_JOB_ID} --format=JobID,MaxRSS,MaxVMSize,Elapsed ==="
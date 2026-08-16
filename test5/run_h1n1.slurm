#!/bin/bash
#SBATCH --job-name=ngpb-h1n1-1n
#SBATCH --partition=h200_short
#SBATCH --gres=gpu:h200:4              # 4x H200 NVL. Selects gpu068 automatically -- it is the
                                       # only node with 4. Measured: 226.3M rows/leader = 79.7 GB
                                       # of ~150 GB usable. 2 GPUs would be 452.6M rows = 159 GB,
                                       # ~6% over, so 4 is the minimum that fits at h=1.3.
#SBATCH --nodes=1
#SBATCH --ntasks=4                     # ONE RANK PER GPU. This is the fix for the energy-stage
                                       # OOM: fmm_build_atom_tree (pb_class.cpp:3589) has no rank
                                       # guard, so every rank builds a full private LBVH. At 16
                                       # ranks that was 4 trees per card. AMGX was unaffected
                                       # because it only runs on the 4 group leaders.
                                       # Also drops set_ranges replication: 444 GiB -> 201 GiB.
#SBATCH --cpus-per-task=16             # 4 x 16 = 64 cores on the EPYC 9555
#SBATCH --distribution=block           # REQUIRED: distributed AMGX assumes each GPU group owns a
                                       # CONTIGUOUS global row range. Cyclic returns a silently
                                       # wrong answer; amgx_solver.cu:409 cannot detect it.
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=04:00:00                # measured last run: 31.5 min NanoShaper (rank 0, unchanged
                                       # by rank count) + 9.5 min MPI phases at 16 ranks + 0.5 min
                                       # solve. At 4 ranks the MPI phases are ~38 min -> ~70 min,
                                       # plus <20 min for the energy stage (6VYB FMM was 3.3 s at
                                       # p=10; ~32x on targets, ~325x on sources). ~1.5 h total.
                                       # Kept short on purpose: gpu068 is the only 4-GPU node, so
                                       # a smaller window is far easier for Slurm to backfill.
#SBATCH --output=h1n1-%j.out
#SBATCH --error=h1n1-%j.err

set -euo pipefail

module load singularity

cd "$SLURM_SUBMIT_DIR"
SIF="${SIF:-$SLURM_SUBMIT_DIR/ngpb.sif}"
PRM="${PRM:-options_h1n1.prm}"

# Pre-flight. singularity auto-mounts $PWD only, and the prm names its files relatively,
# so everything has to sit next to this script.
for f in "$SIF" "$PRM" H1N1.pqr amgx_pcgf_amg_poly_cluster.json; do
    [ -e "$f" ] || { echo "MISSING: $f" >&2; exit 1; }
done

# H1N1.pqr must start with an ATOM record: the reader at pb_class.cpp:1236 parses tokens with
# no record-type check, so a leading REMARK yields zero atoms and pb_class.cpp:1125 then
# dereferences max_element() over an empty vector -> segfault at (nil).
head -c 4 H1N1.pqr | grep -q ATOM || {
    echo "H1N1.pqr does not start with an ATOM record -- reader would parse 0 atoms" >&2
    exit 1
}

echo "Node   : $(hostname)"
echo "CPUs   : ${SLURM_NTASKS} ranks x ${SLURM_CPUS_PER_TASK}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo

# OMP_NUM_THREADS is hard-set to 1 at recipe_cuda.def:23 in %environment, which is sourced
# AFTER any --env flag -- so it can only be overridden from inside, here.
#
# 64 threads on rank 0 deliberately exceeds --cpus-per-task=16: NanoShaper runs alone while the
# other 3 ranks sit in MPI_Barrier (poisson_boltzmann.cpp:119-138), and --bind-to none leaves
# them unpinned, so those threads spread over all 64 cores. Matches number_of_threads in the prm.
singularity exec --nv "$SIF" bash -c "
    export OMP_NUM_THREADS=64
    exec mpirun -n ${SLURM_NTASKS} --bind-to none ngpb --prmfile ${PRM}
"
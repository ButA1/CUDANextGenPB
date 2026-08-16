#!/bin/bash
#SBATCH --job-name=ngpb-h1n1-2n
#SBATCH --partition=h200_short
#SBATCH --gres=gpu:h200:2              # PER NODE -> 2 nodes x 2 = 4 GPUs total, the same leader
                                       # layout that measured 79.7 GB/card. Targets gpu069-072,
                                       # which should queue faster than the single 4-GPU gpu068.
#SBATCH --nodes=2
#SBATCH --ntasks=4                     # ONE RANK PER GPU (see run_h1n1.slurm for why: the FMM
                                       # energy stage builds a full private LBVH on every rank).
#SBATCH --ntasks-per-node=2            # forces 2+2, not 3+1 -- ranks_per_gpu must come out as 1
#SBATCH --cpus-per-task=32             # 2 x 32 = 64 cores per node
#SBATCH --distribution=block           # REQUIRED, and more so here than on one node: leaders are
                                       # ordered by world rank and their partition offsets assume
                                       # each group owns a CONTIGUOUS global row range.
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=04:00:00                # ~1.5 h measured+estimated (see run_h1n1.slurm), with
                                       # margin for the inter-node halo exchange. Short on purpose
                                       # so Slurm can backfill it. Verify h200_short's cap.
#SBATCH --output=h1n1-2n-%j.out
#SBATCH --error=h1n1-2n-%j.err

set -euo pipefail

module load singularity

cd "$SLURM_SUBMIT_DIR"
SIF="${SIF:-$SLURM_SUBMIT_DIR/ngpb.sif}"
PRM="${PRM:-options_h1n1.prm}"

for f in "$SIF" "$PRM" H1N1.pqr amgx_pcgf_amg_poly_cluster.json; do
    [ -e "$f" ] || { echo "MISSING: $f" >&2; exit 1; }
done

head -c 4 H1N1.pqr | grep -q ATOM || {
    echo "H1N1.pqr does not start with an ATOM record -- reader would parse 0 atoms" >&2
    exit 1
}

echo "Nodes  : $SLURM_JOB_NODELIST"
echo "CPUs   : ${SLURM_NTASKS} ranks x ${SLURM_CPUS_PER_TASK}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo

# ---------------------------------------------------------------------------------------
# MULTI-NODE LAUNCH DIFFERS FROM THE 1-NODE SCRIPT AND IS THE ONE UNVERIFIED PIECE HERE.
#
# recipe_cuda.def:131 deliberately runs `mpirun` INSIDE the container ("robust for single
# node, no host MPI interop needed"). That does not extend across nodes: the container's own
# OpenMPI would have to ssh between compute nodes to spawn remote ranks.
#
# The portable fix is to let Slurm do the launching -- srun starts one container per task and
# wires them together over PMI, so the container's OpenMPI never spawns anything itself. That
# requires the container's OpenMPI to be ABI-compatible with the host's PMIx.
#
# VERIFY BEFORE RELYING ON THIS:
#   srun --mpi=list                       # is pmix (or pmix_v4/v5) offered?
#   srun -N2 -n4 --mpi=pmix singularity exec "$SIF" \
#        bash -c 'echo rank $OMPI_COMM_WORLD_RANK of $OMPI_COMM_WORLD_SIZE on $(hostname)'
# That should print 4 distinct ranks across 2 hostnames. If it prints four rank-0-of-1 lines,
# PMI is not wiring up and this script will run 4 independent serial jobs -- which would waste
# another queue slot without obviously failing.
#
# --cpu-bind=none mirrors mpirun's --bind-to none so rank 0's NanoShaper threads can use all 64
# cores of its node rather than just its own 32.
# ---------------------------------------------------------------------------------------
export SINGULARITYENV_OMP_NUM_THREADS=64
export APPTAINERENV_OMP_NUM_THREADS=64

srun --mpi=pmix --cpu-bind=none singularity exec --nv "$SIF" \
     bash -c "export OMP_NUM_THREADS=64; exec ngpb --prmfile ${PRM}"
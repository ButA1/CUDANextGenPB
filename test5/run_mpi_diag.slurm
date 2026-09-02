#!/bin/bash
#SBATCH --job-name=ngpb-mpidiag
#SBATCH --partition=gpu_short
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=mpidiag2-%j.out
#SBATCH --error=mpidiag2-%j.err

# Round 2. What round 1 established:
#   - host + container BOTH see /dev/infiniband; image has openib/ucx/ofi/vader/tcp; pmix_v5
#   - intra-node with one container per rank HANGS on the first Send/Recv (vader has no
#     working single-copy path: CMA blocked by user namespaces, no knem/xpmem)
#   - the old section A was invalid: nested `srun ... mpirun` makes OpenMPI pick the Slurm
#     launcher, which cannot find srun inside the image. Production calls mpirun directly.
#
# Every test is wrapped in `timeout` so a hang costs 60 s, not the whole allocation.

module load singularity 2>/dev/null || true
cd "$SLURM_SUBMIT_DIR"
SIF="${SIF:-$SLURM_SUBMIT_DIR/ngpb.sif}"
T=60                                   # per-test timeout, seconds

singularity exec "$SIF" mpicc -O2 -o mpi_bench mpi_bench.c 2>&1 || exit 1
echo "benchmark built; nodes: $SLURM_JOB_NODELIST"; echo

run() {                                # run <label> <extra-env> <srun-args...>
    local label="$1"; local env="$2"; shift 2
    echo "===================== $label ====================="
    timeout "$T" "$@" 2>&1 | grep -vE "^INFO:|^WARNING: group:" | tail -25
    local rc=${PIPESTATUS[0]}
    [ "$rc" = "124" ] && echo ">>> TIMED OUT after ${T}s (hung)"
    echo
}

# ---- A: the production path. mpirun DIRECTLY, no enclosing srun, one container. ----
# `--mca plm isolated` forces the local-fork launcher. Without it, mpirun sees SLURM_NNODES=2
# from THIS job's allocation, decides it must start remote orted daemons, reaches for `srun`
# -- which does not exist inside the image -- and dies at plm_slurm_module.c:475. That is the
# FORCE-TERMINATE from round 1. Production (--nodes=1) never takes that branch, so `isolated`
# is what faithfully reproduces it here.
echo "===================== A. intra-node, mpirun inside ONE container (production) ====="
timeout "$T" singularity exec "$SIF" mpirun --mca plm isolated -n 2 ./mpi_bench 2>&1 \
    | grep -vE "^INFO:|^WARNING: group:" | tail -25
[ "${PIPESTATUS[0]}" = "124" ] && echo ">>> TIMED OUT after ${T}s (hung)"
echo

# ---- Intra-node, one container per rank: default, then each candidate fix ----
run "B. intra-node, srun+pmix, DEFAULT (expected to hang)" "" \
    srun -N1 -n2 --mpi=pmix singularity exec "$SIF" ./mpi_bench

run "C. intra-node, vader single_copy_mechanism=emulated" "" \
    srun -N1 -n2 --mpi=pmix singularity exec "$SIF" bash -c \
    'export OMPI_MCA_btl_vader_single_copy_mechanism=emulated; exec ./mpi_bench'

run "D. intra-node, btl=tcp,self (slow but should not hang)" "" \
    srun -N1 -n2 --mpi=pmix singularity exec "$SIF" bash -c \
    'export OMPI_MCA_btl=tcp,self; exec ./mpi_bench'

run "E. intra-node, pml=ucx (UCX picks its own shm/rdma path)" "" \
    srun -N1 -n2 --mpi=pmix singularity exec "$SIF" bash -c \
    'export OMPI_MCA_pml=ucx OMPI_MCA_osc=ucx; exec ./mpi_bench'

run "F. intra-node, pml=ucx with shm DISABLED (force IB even on-node)" "" \
    srun -N1 -n2 --mpi=pmix singularity exec "$SIF" bash -c \
    'export OMPI_MCA_pml=ucx OMPI_MCA_osc=ucx UCX_TLS=rc,ud,self; exec ./mpi_bench'

# ---- Inter-node ----
run "G. inter-node, DEFAULT" "" \
    srun -N2 -n2 --ntasks-per-node=1 --mpi=pmix singularity exec "$SIF" ./mpi_bench

run "H. inter-node, pml=ucx" "" \
    srun -N2 -n2 --ntasks-per-node=1 --mpi=pmix singularity exec "$SIF" bash -c \
    'export OMPI_MCA_pml=ucx OMPI_MCA_osc=ucx; exec ./mpi_bench'

run "I. inter-node, btl=tcp,self (control: this is the slow floor)" "" \
    srun -N2 -n2 --ntasks-per-node=1 --mpi=pmix singularity exec "$SIF" bash -c \
    'export OMPI_MCA_btl=tcp,self; exec ./mpi_bench'

# ---- 4 ranks over 2 nodes: the real topology, with the best-guess fix applied ----
run "J. 2 nodes x 2 ranks, pml=ucx (the configuration H1N1 would actually use)" "" \
    srun -N2 -n4 --ntasks-per-node=2 --mpi=pmix singularity exec "$SIF" bash -c \
    'export OMPI_MCA_pml=ucx OMPI_MCA_osc=ucx; exec ./mpi_bench'

echo "===================== DONE ====================="
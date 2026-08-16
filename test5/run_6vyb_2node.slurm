#!/bin/bash
#SBATCH --job-name=ngpb-6vyb-2n
#SBATCH --partition=gpu,gpu_short      # A100 nodes are far less contended than gpu068
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4            # 4 ranks + 1 GPU per node -> ranks_per_gpu=4, so this
                                       # also exercises the multi-rank sub-gather inside each
                                       # group, not just the 1-rank-per-GPU case.
#SBATCH --cpus-per-task=8              # 4 x 8 = 32 of the A100 nodes' 40 logical CPUs
#SBATCH --gres=gpu:a100:1              # per node -> 2 GPUs, 2 groups, 2 leaders on 2 NODES.
                                       # That puts the AMGX halo exchange on the network, which
                                       # is the part the single-node runs have never tested.
#SBATCH --distribution=block
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=6vyb-2n-%j.out
#SBATCH --error=6vyb-2n-%j.err

set -euo pipefail

module load singularity

cd "$SLURM_SUBMIT_DIR"
SIF="${SIF:-$SLURM_SUBMIT_DIR/ngpb.sif}"
PRM="${PRM:-options.prm}"              # the known-good 6VYB config (ngpb-1699791.out)

for f in "$SIF" "$PRM" 6VYB.pqr; do
    [ -e "$f" ] || { echo "MISSING: $f" >&2; exit 1; }
done

echo "Nodes  : $SLURM_JOB_NODELIST"
echo "CPUs   : ${SLURM_NTASKS} ranks x ${SLURM_CPUS_PER_TASK}"
echo

# ---------------------------------------------------------------------------------------
# GATE 1: does srun+PMI actually build one MPI_COMM_WORLD across both nodes?
#
# This is the whole point of the test. The failure mode is silent -- without PMI you get N
# independent rank-0-of-1 processes, each happily solving the whole problem, and nothing
# reports an error. So assert COMM_WORLD_SIZE before spending the allocation.
#
# Note OMPI_COMM_WORLD_* is set by mpirun/orted, NOT by srun+PMIx, so probing env vars would
# give a false negative. The only honest check is a real MPI_Init.
# ---------------------------------------------------------------------------------------
cat > mpi_probe.c <<'EOF'
#include <mpi.h>
#include <stdio.h>
#include <unistd.h>
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, size; char host[256];
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  gethostname(host, sizeof host);
  printf("PROBE rank %d of %d on %s\n", rank, size, host);
  MPI_Finalize();
  return 0;
}
EOF
singularity exec "$SIF" mpicc -o mpi_probe mpi_probe.c

echo "=== GATE 1: PMI wire-up ==="
probe=$(srun --mpi=pmix singularity exec "$SIF" ./mpi_probe 2>&1) || {
    echo "$probe"
    echo "FAIL: srun --mpi=pmix did not run. Check 'srun --mpi=list'; try --mpi=pmi2." >&2
    exit 1
}
echo "$probe"
sizes=$(echo "$probe" | sed -n 's|^PROBE rank [0-9]* of \([0-9]*\) .*|\1|p' | sort -u | tr '\n' ' ')
hosts=$(echo "$probe" | sed -n 's|^PROBE rank [0-9]* of [0-9]* \(.*\)|\1|p' | sort -u | wc -l)
if [ "$(echo $sizes)" != "$SLURM_NTASKS" ]; then
    echo "FAIL: COMM_WORLD_SIZE reported as '$sizes', expected $SLURM_NTASKS." >&2
    echo "      PMI is not wiring up -- these are independent serial processes." >&2
    exit 1
fi
[ "$hosts" -eq 2 ] || { echo "FAIL: ranks landed on $hosts host(s), expected 2." >&2; exit 1; }
echo "GATE 1 OK: one COMM_WORLD of $sizes ranks across $hosts nodes."
echo

# ---------------------------------------------------------------------------------------
# GATE 2: the real run. Watch for two things in the output:
#   [gpu_topology] world_size=8  gpus_in_use=2   <- topology sees both nodes' GPUs
#   "distributed, multi-GPU"                     <- took the dist path, not gather
#
# Then diff the energies against the known-good single-node result in ngpb-1699791.out:
#   Polarization energy  [kT]: -45764.41444882513
#   Direct ionic energy  [kT]:     -45.43064923234952
#   Coulombic energy     [kT]: -540545.7921479711
#   Sum                  [kT]: -586355.6372460286
# A cross-node partition bug would most likely show up as a wrong polarization term while
# everything else still looks plausible -- so compare digits, not just "did it finish".
# ---------------------------------------------------------------------------------------
echo "=== GATE 2: 6VYB across 2 nodes ==="
srun --mpi=pmix --cpu-bind=none singularity exec --nv "$SIF" \
     bash -c "export OMP_NUM_THREADS=8; exec ngpb --prmfile ${PRM}"

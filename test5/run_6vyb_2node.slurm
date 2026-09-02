#!/bin/bash
#SBATCH --job-name=ngpb-6vyb-2n
#SBATCH --partition=gpu,gpu_short      # A100 nodes are far less contended than gpu068
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4            # 4 ranks + 1 GPU per node -> ranks_per_gpu=4, so this
                                       # exercises the multi-rank sub-gather inside each group
                                       # AND cross-node leader ordering. Strictly harder than
                                       # H1N1's 1-rank-per-GPU layout: if this is right, that is.
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1              # per node -> 2 GPUs, 2 groups, 2 leaders on 2 NODES,
                                       # putting the AMGX halo exchange on the network.
#SBATCH --distribution=block
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=6vyb-2n-%j.out
#SBATCH --error=6vyb-2n-%j.err

set -uo pipefail

module load singularity

cd "$SLURM_SUBMIT_DIR"
SIF="${SIF:-$SLURM_SUBMIT_DIR/ngpb.sif}"
PRM="${PRM:-options.prm}"

for f in "$SIF" "$PRM" 6VYB.pqr; do
    [ -e "$f" ] || { echo "MISSING: $f" >&2; exit 1; }
done

echo "Nodes  : $SLURM_JOB_NODELIST"
echo "CPUs   : ${SLURM_NTASKS} ranks x ${SLURM_CPUS_PER_TASK}"
echo

# ---------------------------------------------------------------------------------------
# pml=ucx is MANDATORY, not tuning. Measured on gpu026/027 (mpidiag2-1883214):
#   intra-node, default vader ....... HANGS (CMA blocked across user namespaces, no
#                                     knem/xpmem fallback). This is what stalled the last
#                                     6VYB run at p8est_refine -- 4 ranks/node deadlocked.
#   intra-node, pml=ucx ............. 0.37 us latency, 0.39 us allreduce
#   inter-node, default ............. 48.4 / 52.6 us -- indistinguishable from forced TCP
#   inter-node, pml=ucx ............. 6.10 / 5.73 us
#
# UCX prints a few "mm_posix.c:233 open(/proc/PID/fd/N) Permission denied" lines at startup:
# one shared-memory mechanism failing across namespaces. It falls back on its own and still
# reaches 0.31 us, so those are noise, not a fault.
#
# What to check in the output, in order:
#   1. [gpu_topology] world_size=8  gpus_in_use=2
#   2. "distributed, multi-GPU"  (took the dist path, not gather)
#   3. the energies, against the known-good single-node result in ngpb-1699791.out:
#        Polarization energy [kT]: -45764.41444882513
#        Direct ionic energy [kT]:     -45.43064923234952
#        Coulombic energy    [kT]: -540545.7921479711
#        Sum                 [kT]: -586355.6372460286
#
# (3) is the point of this run. With leaders on separate nodes, a bad row partition yields a
# PLAUSIBLE BUT WRONG polarization energy while the Coulombic term -- computed independently
# on rank 0 -- stays perfect. amgx_solver.cu:409 is only a sum check and cannot catch it.
# Compare digits, not just "did it finish".
# ---------------------------------------------------------------------------------------
srun --mpi=pmix --cpu-bind=none singularity exec --nv "$SIF" bash -c "
    export OMP_NUM_THREADS=8
    export OMPI_MCA_pml=ucx OMPI_MCA_osc=ucx
    exec ngpb --prmfile ${PRM}"
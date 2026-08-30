# ---------------------------------------------------------------------------------------
#  Shared body for the multi-GPU scaling jobs. Sourced, never executed directly.
#
#  The caller must have set, before sourcing:
#      TOPO       short tag for this topology, e.g. 1gpu / 2gpu / 2node
#      MPI_MODE   "mpirun"  -> one container, its own mpirun launches every rank
#                 "srun"    -> host srun launches one container per rank
#
#  Everything else has a default here so the three wrappers stay thin and cannot
#  drift apart in the part that matters -- the six configurations being compared.
#
#  WHY TWO LAUNCH MODES. Measured on gpu026/027 (mpidiag2-1883214):
#    intra-node, one container running mpirun ..... works, 0.37 us latency
#    multi-node, srun + one container per rank .... needs pml=ucx: the default
#        vader BTL HANGS because CMA is blocked across the per-rank user
#        namespaces and there is no knem/xpmem fallback. That is what stalled an
#        earlier 6VYB run at p8est_refine.
#    inter-node, default ...... 48.4 us   |   inter-node, pml=ucx ...... 6.10 us
#  So the launch mode is not a preference: a 2-node run without srun+pmix+ucx
#  deadlocks, and a 1-node run does not need any of it.
#
#  UCX prints a few "mm_posix.c:233 open(/proc/PID/fd/N) Permission denied" lines
#  at startup -- one shared-memory mechanism failing across namespaces. It falls
#  back on its own. Noise, not a fault.
# ---------------------------------------------------------------------------------------

set -euo pipefail

module load singularity

BASE="$SLURM_SUBMIT_DIR"
SIF="${SIF:-$BASE/ngpb.sif}"
SCRIPTS="${SCRIPTS:-$BASE/scripts}"
AMGX_CFG="${AMGX_CFG:-$BASE/amgx_pcgf_amg_block_jacobi.json}"

MOL="${MOL:-6VYB}"
PQR="${PQR:-$BASE/$MOL.pqr}"
RUNDIR="${RUNDIR:-$BASE/scaling_${MOL}_${TOPO}}"
REPEATS="${REPEATS:-3}"

RESUME=""
[ -n "${RESUME_RUN:-}" ] && RESUME="--resume"

if [ -z "${PRM:-}" ]; then
    if [ -f "$BASE/options_$MOL.prm" ]; then PRM="$BASE/options_$MOL.prm"
    else PRM="$BASE/options.prm"; fi
fi

for f in "$SIF" "$PRM" "$PQR" "$AMGX_CFG" "$SCRIPTS/bench_sweep.py"; do
    [ -e "$f" ] || { echo "MISSING: $f" >&2; exit 1; }
done

echo "topology : $TOPO   (launch mode: $MPI_MODE)"
echo "nodes    : ${SLURM_JOB_NODELIST:-?}"
echo "ranks    : $SLURM_NTASKS   ($((SLURM_NTASKS / SLURM_JOB_NUM_NODES)) per node)"
echo "molecule : $MOL"
echo "rundir   : $RUNDIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo

# --- Case directory ----------------------------------------------------------------------
mkdir -p "$RUNDIR"
cp "$PQR" "$AMGX_CFG" "$RUNDIR/"

# bench_sweep.py owns every key it manages, so only the ones it does NOT touch have to be
# right here: the pqr name, and amgx_config resolving inside $RUNDIR. Deleted-then-inserted
# rather than substituted because the two options.prm in circulation disagree about whether
# amgx_config exists at all (the cluster copy names amgx_pcgf_amg_poly_cluster.json, the one
# in git has no such line) and a substitution is a silent no-op on the second.
sed -E '/^[[:space:]]*amgx_config[[:space:]]*=/d' "$PRM" \
  | sed -E "s|^[[:space:]]*filename[[:space:]]*=.*|filename = $(basename "$PQR")|" \
  | awk -v cfg="$(basename "$AMGX_CFG")" '
        { print }
        /^\[algorithm\]/ { print "amgx_config = " cfg }' > "$RUNDIR/options.prm"

# --- Launch mode -------------------------------------------------------------------------
if [ "$MPI_MODE" = "srun" ]; then
    # One container per rank. The MCA settings have to reach INSIDE each container, and
    # SINGULARITYENV_* is the documented way to get them there -- exporting the bare
    # OMPI_MCA_* names relies on singularity's default env passthrough, which --cleanenv
    # or a future image could take away without any visible error, just a hang.
    export SINGULARITYENV_OMPI_MCA_pml=ucx
    export SINGULARITYENV_OMPI_MCA_osc=ucx
    export SINGULARITYENV_OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
    # --bind "$BASE" explicitly: bench_sweep.py runs each rank with cwd=$RUNDIR, and
    # singularity only auto-binds $HOME and the cwd it was invoked from. Without this the
    # ranks start in a directory that has no options.prm.
    LAUNCH=(--mpi-cmd "srun --mpi=pmix --cpu-bind=none singularity exec --nv --bind $BASE $SIF")
    # bench_sweep.py drives the runs, and it must itself run somewhere with a python3 >= 3.8.
    # The host's is a module (TU Berlin ships 3.7 among others), so use the image's -- but
    # WITHOUT --nv and as a plain exec: this instance only orchestrates, it never touches a
    # GPU, and it has to be able to fork srun on the host... which it cannot from inside a
    # container. So in srun mode bench_sweep.py runs on the HOST python.
    DRIVER=(python3)
    # Loaded HERE, not before sbatch. A submit-time "module load" does usually survive
    # (sbatch defaults to --export=ALL, so PATH propagates), but it fails silently if the
    # site or the user sets --export=NONE, if the module tree is not mounted the same way
    # on the compute node, or simply if someone forgets it on a resubmit. The failure then
    # lands ~40 min in, when bench_sweep.py reaches statistics.fmean on python 3.7.
    # "|| true": a wrong module NAME must not kill the job before the version check below
    # gets to print something useful.
    module load "${PYTHON_MODULE:-python/3.12}" 2>/dev/null || true

    if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,8) else 1)" 2>/dev/null; then
        echo "ERROR: no host python3 >= 3.8 after 'module load ${PYTHON_MODULE:-python/3.12}'." >&2
        echo "       srun mode cannot fall back to the image's python: bench_sweep.py has to" >&2
        echo "       fork srun, and a containerised process cannot." >&2
        echo "       Set PYTHON_MODULE= to a name from this list and resubmit:" >&2
        module avail python 2>&1 | sed 's/^/         /' >&2
        exit 1
    fi
    echo "driver python: $(python3 -V 2>&1) at $(command -v python3)"
else
    # One container, its own mpirun inside it. bench_sweep.py runs in the image too, on the
    # container's python 3.12.
    LAUNCH=(--launcher "singularity exec --nv --bind $BASE $SIF")
    DRIVER=(singularity exec --bind "$BASE" "$SIF" python3)
    if ! singularity exec --bind "$BASE" "$SIF" test -r "$SCRIPTS/bench_sweep.py"; then
        echo "ERROR: $SCRIPTS is not visible inside the container" >&2
        exit 1
    fi
fi

# --- The matrix ----------------------------------------------------------------------------
# --sweeps s is the six-row scaling matrix defined in bench_sweep.py's build_plan:
#   {naive energy, FMM 0.3/p9 at 8/1024, FMM 0.4/p11 at 16/1024} x {tuned AMGX, default AMGX}
# The same six rows run at every topology, so speedup is a ratio of matching config_ids.
echo "=== scaling matrix at $TOPO ==="
"${DRIVER[@]}" "$SCRIPTS/bench_sweep.py" "$RUNDIR" \
    --np "$SLURM_NTASKS" --repeats "$REPEATS" --sweeps s $RESUME \
    "${LAUNCH[@]}" \
    -o "$RUNDIR/scaling_$TOPO.csv"

echo
echo "results : $RUNDIR/scaling_$TOPO.csv"
echo
echo "CHECK BEFORE TRUSTING ANY OF IT:"
echo "  1. grep '\[gpu_topology\]' in this log -- gpus_in_use must be what $TOPO claims"
echo "  2. grep 'distributed, multi-GPU' -- confirms AMGX took the distributed path"
echo "     rather than silently gathering to rank 0, which would make the scaling"
echo "     number meaningless while still finishing and still printing energies"
echo "  3. the energies, against the known-good single-node result in ngpb-1699791.out:"
echo "        Polarization energy [kT]: -45764.41444882513"
echo "        Sum                 [kT]: -586355.6372460286"
echo "     With leaders on separate nodes a bad row partition gives a PLAUSIBLE BUT"
echo "     WRONG polarization energy while the Coulombic term -- computed on rank 0"
echo "     alone -- stays perfect. amgx_solver.cu:409 is a sum check and cannot catch"
echo "     it. Compare digits."

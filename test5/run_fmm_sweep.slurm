#!/bin/bash
#SBATCH --job-name=ngpb-fmmsweep
#SBATCH --partition=gpu_short          # 24h cap; the whole job is ~1-2h
#SBATCH --gres=gpu:a100:1              # 1 GPU / 16 ranks -- SAME geometry as run.slurm,
                                       # so t_total_s is directly comparable to the
                                       # "Compute energy" line of ngpb-1699791.out (6.22 s).
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=fmmsweep-%j.out
#SBATCH --error=fmmsweep-%j.err

# ---------------------------------------------------------------------------------------
#  Sweep the fmm_* parameters on 6VYB without re-running the pipeline per configuration.
#
#  ONE ngpb run dumps the phase-1 energy inputs (the flux points, the surface triangles
#  and the atoms -- see include/energy_dump.h), then fmm_replay evaluates the whole
#  fmm_mac x fmm_multipole_order x fmm_leaf_size grid against that dump.
#
#  Through ngpb the same grid is 140 configs x 3 repeats x 157.7 s ~ 18 h, which does not
#  fit in one gpu_short allocation. Here it is one 157.7 s run plus ~30-90 min of replay,
#  because everything up to the energy stage is identical across the grid.
#
#  NEEDS AN IMAGE BUILT FROM THE CURRENT recipe_cuda.def -- one whose %post ran
#  "make distclean all fmm_replay". The pre-existing ngpb.sif does NOT qualify: its
#  ngpb has no energy_dump key and it has no fmm_replay at all. Locally:
#
#      make -C src distclean                       # keep stale .o out of %files
#      apptainer build --fakeroot ngpb.sif recipe_cuda.def
#      scp ngpb.sif run_fmm_sweep.slurm <cluster>:~/test5/
#
#  The preflight below fails fast with that message rather than dying mid-job.
#
#  SRC= is the iteration escape hatch: point it at a source tree on the cluster and the
#  job bind-mounts it over /usr/local/nextgenPB and rebuilds ngpb + fmm_replay in-image
#  (~10 min of nvcc) instead of needing a fresh 6.2 GB upload for every source change.
#  Leave it unset for the normal "run exactly what is in the .sif" path.
#
#  Knobs:  SIF= SRC= RUNDIR= REBUILD=1 MAC= ORDER= LEAF= REPEATS=
# ---------------------------------------------------------------------------------------

set -euo pipefail

module load singularity

BASE="$SLURM_SUBMIT_DIR"
SIF="${SIF:-$BASE/ngpb.sif}"
SRC="${SRC:-}"                         # empty = run the image as built (the normal case)
RUNDIR="${RUNDIR:-$BASE/fmm_sweep}"

MAC="${MAC:-0.2:0.8:0.1}"
ORDER="${ORDER:-6:10}"
LEAF="${LEAF:-32,64,128,256}"
REPEATS="${REPEATS:-3}"

for f in "$SIF" "$BASE/options.prm" "$BASE/6VYB.pqr"; do
    [ -e "$f" ] || { echo "MISSING: $f" >&2; exit 1; }
done

if [ -n "$SRC" ]; then
    # The bind mount replaces /usr/local/nextgenPB wholesale, so $SRC must carry
    # src/ + include/ + addons/ -- exactly what recipe_cuda.def's %files copies in.
    [ -e "$SRC/src/Makefile" ] || { echo "MISSING: $SRC/src/Makefile" >&2; exit 1; }
    RUN_IN_IMAGE=(singularity exec --nv --bind "$SRC:/usr/local/nextgenPB" "$SIF")
else
    RUN_IN_IMAGE=(singularity exec --nv "$SIF")
fi

echo "host    : $(hostname)"
echo "ranks   : $SLURM_NTASKS"
echo "sif     : $SIF"
echo "src     : ${SRC:-(none -- using the image as built)}"
echo "rundir  : $RUNDIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo

# --- 1. Make sure the binaries exist ----------------------------------------------------
if [ -z "$SRC" ]; then
    # Cheap proxy for "this image postdates the dump/replay work": fmm_replay and the
    # energy_dump-aware ngpb come out of the same %post. Fail here, not 3 minutes into
    # a pipeline run that would silently ignore the energy_dump key.
    if ! "${RUN_IN_IMAGE[@]}" bash -c 'command -v fmm_replay >/dev/null 2>&1'; then
        cat >&2 <<'EOF'
ERROR: no fmm_replay in the image -- it predates the dump/replay work.

  Rebuild it from the current recipe_cuda.def (its %post now runs
  "make distclean all fmm_replay"), or point SRC= at a source tree on
  the cluster to bind-mount and rebuild in-image instead.
EOF
        exit 1
    fi
else
    # -arch=native resolves to sm_80 here because the A100 is visible under --nv; do NOT
    # move this to a login node. distclean first: any .o copied from the dev box was
    # compiled against CUDA 13.1 for sm_86 and would link silently wrong.
    if [ ! -x "$SRC/src/fmm_replay" ] || [ ! -x "$SRC/src/ngpb" ] || [ -n "${REBUILD:-}" ]; then
        echo "=== building ngpb + fmm_replay from $SRC ==="
        "${RUN_IN_IMAGE[@]}" bash -c '
            set -e
            cd /usr/local/nextgenPB/src
            # The container toolkit is at /usr/local/cuda, not the dev box'"'"'s cuda-13.1.
            # Idempotent -- recipe_cuda.def applies the same substitution at build time.
            sed -i "s|cuda-13\.1|cuda|g" local_settings.mk
            make distclean
            make -j"$(nproc)" all fmm_replay
        '
        echo
    else
        echo "=== reusing $SRC/src/{ngpb,fmm_replay} (REBUILD=1 to force) ==="
        echo
    fi
fi

# --- 2. Case directory ------------------------------------------------------------------
mkdir -p "$RUNDIR"
cp "$BASE/6VYB.pqr" "$RUNDIR/"

# energy_dump goes in right after [algorithm] so it does not depend on key order;
# energy_method is pinned to 2 so the dump run also produces one FMM number to eyeball.
awk '{ print } /^\[algorithm\]/ { print "energy_dump = fmm_inputs" }' "$BASE/options.prm" \
    | sed 's/^energy_method.*/energy_method = 2/' > "$RUNDIR/options.prm"

cd "$RUNDIR"

# --- 3. One pipeline run -> the dump -----------------------------------------------------
# ~650 MB total across 16 .rank<K>.bin files (measured on this case at 4 ranks -- the
# totals are mesh properties, so the rank count only changes how they are split). Left in
# place on purpose: re-sweeping a different grid needs no further ngpb run, just step 4.
echo "=== [1/2] pipeline run, dumping phase-1 energy inputs ==="
"${RUN_IN_IMAGE[@]}" mpirun -n "$SLURM_NTASKS" ngpb --prmfile options.prm
echo

# --- 4. The sweep ------------------------------------------------------------------------
# Rank count MUST match the dump's -- fmm_replay refuses a mismatch rather than
# reducing a silent partial sum. The CSV is flushed after every configuration, so a
# TIMEOUT still leaves usable partial data.
echo "=== [2/2] replaying the fmm_* grid ==="
"${RUN_IN_IMAGE[@]}" mpirun -n "$SLURM_NTASKS" fmm_replay fmm_inputs \
    --mac "$MAC" --order "$ORDER" --leaf "$LEAF" \
    --repeats "$REPEATS" --csv fmm_sweep.csv

echo
echo "results : $RUNDIR/fmm_sweep.csv"
echo "plot    : python3 scripts/plot_fmm_sweep.py $RUNDIR --csv $RUNDIR/fmm_sweep.csv"

# ---------------------------------------------------------------------------------------
#  Equivalent via the local driver, if you copied scripts/ over and the node has a host
#  python3 -- same CSV, and it restores options.prm on any exit path:
#
#      python3 scripts/bench_sweep.py "$RUNDIR" \
#          --np "$SLURM_NTASKS" --fmm-sweep \
#          --launcher "singularity exec --nv $SIF" \
#          --fmm-mac "$MAC" --fmm-order "$ORDER" --fmm-leaf "$LEAF" \
#          --fmm-repeats "$REPEATS"
#
#  Plotting is a pure-python step with no GPU in it -- scp fmm_sweep.csv back and run
#  scripts/plot_fmm_sweep.py locally rather than burning allocation on it.
# ---------------------------------------------------------------------------------------

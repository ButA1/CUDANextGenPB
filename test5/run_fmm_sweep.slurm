#!/bin/bash
#SBATCH --job-name=ngpb-leafsweep
#SBATCH --partition=gpu_short          # 24h cap; this job is ~1.5h
#SBATCH --gres=gpu:a100:1              # 1 GPU / 16 ranks -- SAME geometry as run.slurm,
                                       # so t_total_s is directly comparable to the
                                       # "Compute energy" line of ngpb-1699791.out (6.22 s).
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=leafsweep-%j.out
#SBATCH --error=leafsweep-%j.err

# ---------------------------------------------------------------------------------------
#  JOB 1 of 2 -- the leaf-size sweep. One submission per molecule (MOL=, default 6VYB).
#
#  Part A: the usual pipeline sweeps, one full ngpb run each.
#      sweep A  linear_solver = amgx+config / amgx bare / lis   (energy_method 1)
#      sweep B  energy_method = 0 / 1 / 2                       (amgx + config)
#      The energy_method=2 run uses OPT_SRC/OPT_TGT below, so it is the end-to-end
#      wall-clock number AT the believed optimum, not at some stale prm value.
#
#  Part B: the source x target leaf grid, at FIXED mac and order. One ngpb run dumps
#      the phase-1 energy inputs, then fmm_replay evaluates every leaf pair against
#      that dump -- 24 configurations in ~8 min instead of ~1.5 h of pipeline runs.
#
#  The two leaf sizes pull opposite ways: the SOURCE leaf bounds the box radius the MAC
#  tests against (near-field/P2P work), the TARGET leaf sets the target box count (M2L
#  pair count). Sweeping them as one number only samples the diagonal, which is why
#  64 and 256 looked near-equal in the old sweeps -- 256 won the far field, 64 the near.
#
#  p and theta are swept separately, at the winning leaf pairs, by run_fmm_ptheta.slurm.
#
#  NEEDS AN IMAGE BUILT FROM THE CURRENT recipe_cuda.def -- one whose %post ran
#  "make distclean all fmm_replay". The pre-existing ngpb.sif does NOT qualify: its
#  ngpb has no energy_dump key and it has no fmm_replay at all. Locally:
#
#      make -C src distclean                       # keep stale .o out of %files
#      apptainer build --fakeroot ngpb.sif recipe_cuda.def
#      scp ngpb.sif run_fmm_sweep.slurm run_fmm_ptheta.slurm <cluster>:~/test5/
#      scp -r scripts data/amgx_pcgf_amg_block_jacobi.json <cluster>:~/test5/
#
#  The preflight below fails fast rather than dying mid-job.
#
#  SRC= is the iteration escape hatch: point it at a source tree on the cluster and the
#  job bind-mounts it over /usr/local/nextgenPB and rebuilds ngpb + fmm_replay in-image
#  (~10 min of nvcc) instead of needing a fresh 6.2 GB upload for every source change.
#  Leave it unset for the normal "run exactly what is in the .sif" path.
#
#  Knobs:  MOL= PQR= PRM= SIF= SRC= RUNDIR= REBUILD=1 RESUME_RUN=1 REPEATS=
#          MAC= ORDER= LEAF= TLEAF=
#          OPT_SRC= OPT_TGT= SKIP_AB=1 SKIP_LEAF=1
# ---------------------------------------------------------------------------------------

set -euo pipefail

module load singularity

BASE="$SLURM_SUBMIT_DIR"
SIF="${SIF:-$BASE/ngpb.sif}"
SRC="${SRC:-}"                         # empty = run the image as built (the normal case)
SCRIPTS="${SCRIPTS:-$BASE/scripts}"
AMGX_CFG="${AMGX_CFG:-$BASE/amgx_pcgf_amg_block_jacobi.json}"

# One submission per molecule:  sbatch run_fmm_sweep.slurm
#                          MOL=1VSZ sbatch run_fmm_sweep.slurm
# Each gets its own RUNDIR, so the two never collide and can run concurrently.
MOL="${MOL:-6VYB}"
PQR="${PQR:-$BASE/$MOL.pqr}"
RUNDIR="${RUNDIR:-$BASE/leafsweep_$MOL}"

# A per-molecule prm if there is one, else the shared one. Only the [input] filename
# has to differ between molecules and this script rewrites that anyway, so a second
# prm is needed only when the MESH settings differ (scale, perfil, refinement).
if [ -z "${PRM:-}" ]; then
    if [ -f "$BASE/options_$MOL.prm" ]; then PRM="$BASE/options_$MOL.prm"
    else PRM="$BASE/options.prm"; fi
fi

# Set RESUME=1 on a resubmit: bench_sweep.py flushes its CSV after every run and skips
# configurations already in it, so a TIMEOUT costs only the run it was in the middle of.
RESUME=""
[ -n "${RESUME_RUN:-}" ] && RESUME="--resume"

REPEATS="${REPEATS:-3}"

# The leaf grid: mac and order are FIXED here so the two axes under test are the only
# things moving. theta=0.3 / p=9 rather than the p=11 ceiling because l2p_p2p spills
# registers from p=10 up, and a spilling kernel would confound the leaf comparison.
MAC="${MAC:-0.3}"
ORDER="${ORDER:-9}"
LEAF="${LEAF:-8,16,32,64}"                          # SOURCE tree -- small
TLEAF="${TLEAF:-128,256,512,1024,2048,4096}"        # TARGET tree -- large
                                                    # 4096 == FMM_MAX_TLEAF, a hard
                                                    # ceiling: if it wins, the optimum
                                                    # is not bracketed and fmm.h needs
                                                    # raising before you can claim one.

# What options.prm carries for part A's energy_method=2 run.
OPT_SRC="${OPT_SRC:-16}"
OPT_TGT="${OPT_TGT:-1024}"

# --- Preflight --------------------------------------------------------------------------
for f in "$SIF" "$PRM" "$PQR" "$AMGX_CFG" \
         "$SCRIPTS/bench_sweep.py"; do
    [ -e "$f" ] || { echo "MISSING: $f" >&2; exit 1; }
done

# bench_sweep.py runs INSIDE the image, on the container's python3 (Ubuntu 24.04 ->
# 3.12). Deliberately not the host's: TU Berlin serves python as a module, and the
# versions on offer include 3.7, on which bench_sweep.py would run the whole sweep and
# only then die in statistics.fmean (3.8+) while writing the summary. Nothing about
# this job needs a host python, so it does not ask for one.

if [ -n "$SRC" ]; then
    # The bind mount replaces /usr/local/nextgenPB wholesale, so $SRC must carry
    # src/ + include/ + addons/ -- exactly what recipe_cuda.def's %files copies in.
    [ -e "$SRC/src/Makefile" ] || { echo "MISSING: $SRC/src/Makefile" >&2; exit 1; }
    RUN_IN_IMAGE=(singularity exec --nv --bind "$BASE" --bind "$SRC:/usr/local/nextgenPB" "$SIF")
else
    RUN_IN_IMAGE=(singularity exec --nv --bind "$BASE" "$SIF")
fi

echo "host    : $(hostname)"
echo "ranks   : $SLURM_NTASKS"
echo "sif     : $SIF"
echo "src     : ${SRC:-(none -- using the image as built)}"
echo "molecule: $MOL  ($PQR)"
echo "prm     : $PRM"
echo "rundir  : $RUNDIR"
echo "leaf    : src {$LEAF} x tgt {$TLEAF}  at mac=$MAC order=$ORDER"
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
fi

# The container's python3 drives the sweep, and it must be able to SEE the bind-mounted
# scripts. Checking both together catches a $BASE that did not bind as well as an image
# built without python3.
if ! "${RUN_IN_IMAGE[@]}" python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,8) else 1)" 2>/dev/null; then
    echo "ERROR: no usable python3 (>=3.8) inside $SIF" >&2
    exit 1
fi
if ! "${RUN_IN_IMAGE[@]}" test -r "$SCRIPTS/bench_sweep.py"; then
    echo "ERROR: $SCRIPTS is not visible inside the container -- add --bind for it" >&2
    exit 1
fi

if [ -n "$SRC" ]; then
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
cp "$PQR" "$AMGX_CFG" "$RUNDIR/"

# bench_sweep.py owns the whole [algorithm] section, so the fmm_* values written here
# are what its energy_method=2 run uses. Pin them to the believed optimum: that run is
# the end-to-end wall-clock datapoint for the thesis table.
# Delete-then-insert rather than insert-only: a plain insert leaves the file's own
# fmm_* lines in place below the new ones, and which of the duplicates wins is then a
# question about ngpb's parser that nothing here should depend on.
sed -E '/^[[:space:]]*(energy_method|fmm_mac|fmm_multipole_order|fmm_leaf_size|fmm_target_leaf_size|energy_dump)[[:space:]]*=/d' \
    "$PRM" \
  | sed -E "s|^[[:space:]]*filename[[:space:]]*=.*|filename = $(basename "$PQR")|" \
  | awk -v s="$OPT_SRC" -v t="$OPT_TGT" -v m="$MAC" -v o="$ORDER" '
        { print }
        /^\[algorithm\]/ {
            print "energy_method = 2"
            print "fmm_mac = " m
            print "fmm_multipole_order = " o
            print "fmm_leaf_size = " s
            print "fmm_target_leaf_size = " t
        }' > "$RUNDIR/options.prm"

cd "$RUNDIR"

# --- 3. Part A: the pipeline sweeps -------------------------------------------------------
# 3 solver configs + 3 energy methods, x REPEATS, + 1 reference run. At 157.7 s each that
# is ~50 min of the allocation and it is the only part that cannot be replayed.
if [ -z "${SKIP_AB:-}" ]; then
    echo "=== [1/2] sweeps A + B (linear solver, energy method) ==="
    "${RUN_IN_IMAGE[@]}" python3 "$SCRIPTS/bench_sweep.py" "$RUNDIR" \
        --np "$SLURM_NTASKS" --repeats "$REPEATS" --sweeps ab $RESUME \
        -o "$RUNDIR/bench_runs.csv"
    echo
fi

# --- 4. Part B: the leaf grid --------------------------------------------------------------
# One dump run (~650 MB across 16 .rank<K>.bin files, left in place so a re-sweep needs no
# further ngpb run), then 24 leaf pairs off that dump. The CSV is flushed per configuration,
# so a TIMEOUT still leaves usable partial data.
if [ -z "${SKIP_LEAF:-}" ]; then
    echo "=== [2/2] source x target leaf grid ==="
    "${RUN_IN_IMAGE[@]}" python3 "$SCRIPTS/bench_sweep.py" "$RUNDIR" \
        --np "$SLURM_NTASKS" --fmm-sweep \
        --fmm-mac "$MAC" --fmm-order "$ORDER" \
        --fmm-leaf "$LEAF" --fmm-tleaf "$TLEAF" \
        --fmm-repeats "$REPEATS" \
        -o "$RUNDIR/leaf_sweep.csv"
fi

echo
echo "results : $RUNDIR/bench_runs.csv   (sweeps A+B, full pipeline)"
echo "          $RUNDIR/leaf_sweep.csv   (source x target leaf grid)"
echo
echo "Plotting is pure python with no GPU in it -- scp the CSVs back and run"
echo "scripts/plot_fmm_sweep.py locally rather than burning allocation on it."
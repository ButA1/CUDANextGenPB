#!/bin/bash
#SBATCH --job-name=ngpb-ptheta
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=16                    # must match run_fmm_sweep.slurm: the replay refuses
                                       # a rank count the dump was not written at
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=01:00:00                # deliberately SHORT so it backfills into gaps that
                                       # a 4h allocation would wait behind
#SBATCH --output=ptheta-%j.out
#SBATCH --error=ptheta-%j.err

# ---------------------------------------------------------------------------------------
#  JOB 2 of 2 -- the accuracy-vs-work curves, at the leaf pairs that already won.
#
#  Sweeps fmm_mac x fmm_multipole_order over an EXPLICIT list of (source, target) leaf
#  pairs rather than their cross product. Crossing the full leaf grid back against p and
#  theta would multiply the point count by 24 for configurations already known to lose;
#  --pairs walks only the list.
#
#  This is the figure with e_rel on one axis and t on the other, one line per leaf pair.
#  Including the pre-split 64/64 makes the split itself visible as a shifted curve rather
#  than a claim in the text.
#
#  SELF-CONTAINED: it makes its own dump in its own RUNDIR, so it can be submitted at the
#  same time as run_fmm_sweep.slurm (or before it) with no ordering dependency and no risk
#  of reading a half-written dump. That costs one 157.7 s pipeline run. To reuse job 1's
#  dump instead, once job 1 has FINISHED:  RUNDIR=$SLURM_SUBMIT_DIR/leafsweep_$MOL sbatch ...
#
#  Same image requirement as job 1 -- see run_fmm_sweep.slurm's header.
#
#  Knobs:  MOL= PQR= PRM= SIF= SRC= RUNDIR= REBUILD=1 REPEATS= MAC= ORDER= PAIRS=
# ---------------------------------------------------------------------------------------

set -euo pipefail

module load singularity

BASE="$SLURM_SUBMIT_DIR"
SIF="${SIF:-$BASE/ngpb.sif}"
SRC="${SRC:-}"
# The dump run is a real ngpb solve, so it needs its AMGX config like any other. The
# 2026-08-30 submission had no such handling here: the prm carried over
# "amgx_config = amgx_pcgf_amg_poly_cluster.json" from the H1N1 scripts, nothing copied a
# matching json, and ngpb silently fell back to its built-in configuration -- a different
# solve, hence a slightly different phi in the dump. Harmless for timings and for errors
# measured against the naive baseline on that same dump, but not something to leave in.
AMGX_CFG="${AMGX_CFG:-$BASE/amgx_pcgf_amg_block_jacobi.json}"
MOL="${MOL:-6VYB}"
PQR="${PQR:-$BASE/$MOL.pqr}"
RUNDIR="${RUNDIR:-$BASE/ptheta_$MOL}"

if [ -z "${PRM:-}" ]; then
    if [ -f "$BASE/options_$MOL.prm" ]; then PRM="$BASE/options_$MOL.prm"
    else PRM="$BASE/options.prm"; fi
fi

REPEATS="${REPEATS:-3}"
MAC="${MAC:-0.2:0.8:0.1}"
ORDER="${ORDER:-6:11}"                 # 11 is the ceiling: p=12 measured slower AND less
                                       # accurate (unscaled M2L conditioning)

# src/tgt pairs. 16/1024 is the measured optimum; 16/512 and 32/1024 bracket it on each
# axis; 64/64 is the pre-split configuration, carried so the figure shows what decoupling
# the two trees actually bought.
PAIRS="${PAIRS:-16/1024,16/512,32/1024,64/64}"

for f in "$SIF" "$PRM" "$PQR" "$AMGX_CFG"; do
    [ -e "$f" ] || { echo "MISSING: $f" >&2; exit 1; }
done

if [ -n "$SRC" ]; then
    [ -e "$SRC/src/Makefile" ] || { echo "MISSING: $SRC/src/Makefile" >&2; exit 1; }
    RUN_IN_IMAGE=(singularity exec --nv --bind "$BASE" --bind "$SRC:/usr/local/nextgenPB" "$SIF")
else
    RUN_IN_IMAGE=(singularity exec --nv --bind "$BASE" "$SIF")
fi

echo "host    : $(hostname)"
echo "ranks   : $SLURM_NTASKS"
echo "molecule: $MOL  ($PQR)"
echo "rundir  : $RUNDIR"
echo "grid    : mac {$MAC} x order {$ORDER} x pairs {$PAIRS}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo

# --- 1. Binaries --------------------------------------------------------------------------
if [ -z "$SRC" ]; then
    if ! "${RUN_IN_IMAGE[@]}" bash -c 'command -v fmm_replay >/dev/null 2>&1'; then
        echo "ERROR: no fmm_replay in the image -- see run_fmm_sweep.slurm's header." >&2
        exit 1
    fi
elif [ ! -x "$SRC/src/fmm_replay" ] || [ ! -x "$SRC/src/ngpb" ] || [ -n "${REBUILD:-}" ]; then
    echo "=== building ngpb + fmm_replay from $SRC ==="
    "${RUN_IN_IMAGE[@]}" bash -c '
        set -e
        cd /usr/local/nextgenPB/src
        sed -i "s|cuda-13\.1|cuda|g" local_settings.mk
        make distclean
        make -j"$(nproc)" all fmm_replay
    '
    echo
fi

# --- 2. Case directory ---------------------------------------------------------------------
mkdir -p "$RUNDIR"
cp "$PQR" "$AMGX_CFG" "$RUNDIR/"

# Delete-then-insert: see the same block in run_fmm_sweep.slurm. The fmm_* values are
# irrelevant here -- the dump is taken in phase 1, before any of them apply. amgx_config is
# deleted and reinserted rather than substituted, because the cluster's options.prm has the
# key and the one in git does not; a substitution would be a silent no-op on the latter.
sed -E '/^[[:space:]]*(energy_method|energy_dump|amgx_config)[[:space:]]*=/d' "$PRM" \
  | sed -E "s|^[[:space:]]*filename[[:space:]]*=.*|filename = $(basename "$PQR")|" \
  | awk -v cfg="$(basename "$AMGX_CFG")" '
        { print }
        /^\[algorithm\]/ {
            print "amgx_config = " cfg
            print "energy_method = 2"
            print "energy_dump = fmm_inputs"
        }' > "$RUNDIR/options.prm"

cd "$RUNDIR"

# --- 3. Dump, unless one is already here ----------------------------------------------------
# rank0 alone is a weak check, but a dump is written by all ranks within one job step: if
# rank0 exists the others do too, unless a previous job was killed mid-write. Delete the
# .rank*.bin files to force a fresh dump.
if [ -f "fmm_inputs.rank0.bin" ]; then
    echo "=== [1/2] reusing existing dump in $RUNDIR ==="
else
    echo "=== [1/2] pipeline run, dumping phase-1 energy inputs ==="
    "${RUN_IN_IMAGE[@]}" mpirun -n "$SLURM_NTASKS" ngpb --prmfile options.prm
fi
echo

# --- 4. The p x theta sweep at each leaf pair -------------------------------------------------
echo "=== [2/2] mac x order at $PAIRS ==="
"${RUN_IN_IMAGE[@]}" mpirun -n "$SLURM_NTASKS" fmm_replay fmm_inputs \
    --mac "$MAC" --order "$ORDER" --pairs "$PAIRS" \
    --repeats "$REPEATS" --csv ptheta.csv

echo
echo "results : $RUNDIR/ptheta.csv"
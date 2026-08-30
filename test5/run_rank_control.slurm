#!/bin/bash
#SBATCH --job-name=ngpb-rank4
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=4                     # THE POINT OF THIS JOB -- matches the local runs
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=02:00:00                # short enough to backfill; measured need is ~65 min
#SBATCH --output=rank4-%j.out
#SBATCH --error=rank4-%j.err

# ---------------------------------------------------------------------------------------
#  THE CONTROL JOB. Everything measured on the cluster so far ran at 16 ranks on one A100,
#  while every local number is 4 ranks on one RTX 3080. Two variables moved at once, so no
#  cluster-vs-local difference could be attributed to the hardware. This job holds the rank
#  count fixed at 4 and changes only the GPU.
#
#  It answers two questions that are currently open:
#
#    1. AMGX. The 16-rank runs show the AMGX solve kernel 2.1-2.3x faster on the A100
#       (6VYB tuned 2.080 s -> 0.927 s) while t_solve barely moves (6.276 s -> 5.431 s),
#       because the host-side remainder is 12-25% SLOWER on the cluster's weaker cores.
#       That reading assumes the rank count is not what did it. At 4 ranks it is testable.
#
#    2. The leaf optimum. Local 6VYB peaks at src 8 / tgt 1024, the 16-rank cluster run at
#       src 16 / tgt 512 -- one step up in source leaf on both molecules. The targets are
#       distributed across ranks, so 16 ranks means a quarter of the target points per rank
#       and a different near/far balance; that alone could produce the shift. Same grid at
#       4 ranks separates the two explanations.
#
#  Sweep B is deliberately NOT run. Its energy_method=0 leg is the CPU reference, 515 s at
#  16 ranks; at 4 it would be ~35 min per repeat and it answers nothing this job is asking.
#  Sweep A gives the solver comparison, and --fmm-sweep takes its own dump.
#
#  Same image requirement as run_fmm_sweep.slurm -- see its header. Submit as:
#
#      sbatch run_rank_control.slurm                 # 6VYB
#      MOL=1VSZ sbatch run_rank_control.slurm        # 1VSZ, own RUNDIR, can run concurrently
#
#  Knobs:  MOL= PQR= PRM= SIF= SRC= RUNDIR= REBUILD=1 RESUME_RUN=1 REPEATS=
#          MAC= ORDER= LEAF= TLEAF= AMGX_CFG= SKIP_A=1 SKIP_LEAF=1
# ---------------------------------------------------------------------------------------

set -euo pipefail

module load singularity

BASE="$SLURM_SUBMIT_DIR"
SIF="${SIF:-$BASE/ngpb.sif}"
SRC="${SRC:-}"
SCRIPTS="${SCRIPTS:-$BASE/scripts}"
AMGX_CFG="${AMGX_CFG:-$BASE/amgx_pcgf_amg_block_jacobi.json}"

MOL="${MOL:-6VYB}"
PQR="${PQR:-$BASE/$MOL.pqr}"
# A separate RUNDIR from leafsweep_$MOL on purpose: this data must never be mixed into the
# 16-rank CSVs, since the whole point is that the two rank counts are not comparable.
RUNDIR="${RUNDIR:-$BASE/rank4_$MOL}"

if [ -z "${PRM:-}" ]; then
    if [ -f "$BASE/options_$MOL.prm" ]; then PRM="$BASE/options_$MOL.prm"
    else PRM="$BASE/options.prm"; fi
fi

RESUME=""
[ -n "${RESUME_RUN:-}" ] && RESUME="--resume"

REPEATS="${REPEATS:-3}"

# Identical to run_fmm_sweep.slurm and to the local runs. Do not "improve" these: the grid
# is only a control if it matches what it is being compared against.
MAC="${MAC:-0.3}"
ORDER="${ORDER:-9}"
LEAF="${LEAF:-8,16,32,64}"
TLEAF="${TLEAF:-128,256,512,1024,2048,4096}"

# --- Preflight --------------------------------------------------------------------------
for f in "$SIF" "$PRM" "$PQR" "$AMGX_CFG" "$SCRIPTS/bench_sweep.py"; do
    [ -e "$f" ] || { echo "MISSING: $f" >&2; exit 1; }
done

if [ -n "$SRC" ]; then
    [ -e "$SRC/src/Makefile" ] || { echo "MISSING: $SRC/src/Makefile" >&2; exit 1; }
    RUN_IN_IMAGE=(singularity exec --nv --bind "$BASE" --bind "$SRC:/usr/local/nextgenPB" "$SIF")
else
    RUN_IN_IMAGE=(singularity exec --nv --bind "$BASE" "$SIF")
fi

echo "host    : $(hostname)"
echo "ranks   : $SLURM_NTASKS   (control: matches the local 4-rank runs)"
echo "sif     : $SIF"
echo "molecule: $MOL  ($PQR)"
echo "rundir  : $RUNDIR"
echo "leaf    : src {$LEAF} x tgt {$TLEAF}  at mac=$MAC order=$ORDER"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo

if [ "$SLURM_NTASKS" -ne 4 ]; then
    echo "WARNING: --ntasks is $SLURM_NTASKS, not 4. This job is only a control at 4." >&2
fi

# --- 1. Binaries --------------------------------------------------------------------------
if [ -z "$SRC" ]; then
    if ! "${RUN_IN_IMAGE[@]}" bash -c 'command -v fmm_replay >/dev/null 2>&1'; then
        echo "ERROR: no fmm_replay in the image -- see run_fmm_sweep.slurm's header." >&2
        exit 1
    fi
fi
if ! "${RUN_IN_IMAGE[@]}" python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,8) else 1)" 2>/dev/null; then
    echo "ERROR: no usable python3 (>=3.8) inside $SIF" >&2
    exit 1
fi
if ! "${RUN_IN_IMAGE[@]}" test -r "$SCRIPTS/bench_sweep.py"; then
    echo "ERROR: $SCRIPTS is not visible inside the container -- add --bind for it" >&2
    exit 1
fi

if [ -n "$SRC" ] && { [ ! -x "$SRC/src/fmm_replay" ] || [ ! -x "$SRC/src/ngpb" ] || [ -n "${REBUILD:-}" ]; }; then
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

# --- 2. Case directory ------------------------------------------------------------------
mkdir -p "$RUNDIR"
cp "$PQR" "$AMGX_CFG" "$RUNDIR/"

# amgx_config goes through the same delete-then-insert as the rest, NOT a rewrite-in-place:
# the two options.prm in play disagree about whether the key exists at all (the cluster copy
# carries "amgx_config = amgx_pcgf_amg_poly_cluster.json" from the H1N1 scripts, the one in
# git has no such line), and a substitution would silently do nothing on the second. Writing
# the basename of the copy made above makes it resolve inside $RUNDIR either way -- the
# omission that made the 2026-08-30 ptheta jobs fall back to AMGX's built-in configuration.
sed -E '/^[[:space:]]*(energy_method|fmm_mac|fmm_multipole_order|fmm_leaf_size|fmm_target_leaf_size|energy_dump|amgx_config)[[:space:]]*=/d' \
    "$PRM" \
  | sed -E "s|^[[:space:]]*filename[[:space:]]*=.*|filename = $(basename "$PQR")|" \
  | awk -v m="$MAC" -v o="$ORDER" -v cfg="$(basename "$AMGX_CFG")" '
        { print }
        /^\[algorithm\]/ {
            print "amgx_config = " cfg
            print "energy_method = 1"
            print "fmm_mac = " m
            print "fmm_multipole_order = " o
            print "fmm_leaf_size = 16"
            print "fmm_target_leaf_size = 1024"
        }' > "$RUNDIR/options.prm"

cd "$RUNDIR"

# --- 3. Sweep A: the linear solvers --------------------------------------------------------
# 3 configs x REPEATS + 1 reference run. energy_method=1 throughout, so the energy stage is
# a fixed cost across the three and t_solve is the only thing under comparison.
if [ -z "${SKIP_A:-}" ]; then
    echo "=== [1/2] sweep A (linear solver) at $SLURM_NTASKS ranks ==="
    "${RUN_IN_IMAGE[@]}" python3 "$SCRIPTS/bench_sweep.py" "$RUNDIR" \
        --np "$SLURM_NTASKS" --repeats "$REPEATS" --sweeps a $RESUME \
        -o "$RUNDIR/bench_runs.csv"
    echo
fi

# --- 4. The leaf grid, at 4 ranks ------------------------------------------------------------
# The dump is written at $SLURM_NTASKS ranks and fmm_replay refuses any other rank count, so
# this dump is usable only by 4-rank replays. That is the intent -- do not point a 16-rank
# job at this RUNDIR.
if [ -z "${SKIP_LEAF:-}" ]; then
    echo "=== [2/2] source x target leaf grid at $SLURM_NTASKS ranks ==="
    "${RUN_IN_IMAGE[@]}" python3 "$SCRIPTS/bench_sweep.py" "$RUNDIR" \
        --np "$SLURM_NTASKS" --fmm-sweep \
        --fmm-mac "$MAC" --fmm-order "$ORDER" \
        --fmm-leaf "$LEAF" --fmm-tleaf "$TLEAF" \
        --fmm-repeats "$REPEATS" \
        -o "$RUNDIR/leaf_sweep.csv"
fi

echo
echo "results : $RUNDIR/bench_runs.csv   (sweep A, 4 ranks -- compare against test1/test2)"
echo "          $RUNDIR/leaf_sweep.csv   (leaf grid, 4 ranks -- the hardware-only comparison)"
echo
echo "Compare against the 16-rank runs with:"
echo "  scripts/plot_leaf_sweep.py $RUNDIR --stat min --tag leaf_sweep_rank4_$MOL"

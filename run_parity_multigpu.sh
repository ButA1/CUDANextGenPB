#!/usr/bin/env bash
#
# Local parity check for the multi-GPU AMGX rewrite (WS-E).
#
# Runs the SAME case twice at the same rank count, both on the local (single) GPU
# via the NGPB_GPUS_PER_NODE fake-count override, so BOTH code paths actually
# execute and their energies can be compared:
#   1. distributed multi-GPU AMGX  (sub-gather + AMGX_matrix_upload_distributed)
#   2. gather-to-rank-0 AMGX        (NGPB_AMGX_FORCE_GATHER=1, the pre-existing path)
#
# The linear solver is forced to amgx in a temp copy of the case's options.prm
# (test1 ships with linear_solver=lis). Everything else is left untouched.
#
# Usage:  ./run_parity_multigpu.sh [TESTDIR] [NRANKS] [FAKE_GPUS]
#   TESTDIR    case directory (default: test0 -- small enough for a single GPU)
#   NRANKS     MPI ranks       (default: 8)
#   FAKE_GPUS  pretended GPUs/node for the topology math (default: 2)
# Env: NGPB (binary path), TOL (relative tolerance, default 1e-6).
#
# NOTE: the fake-count override splits the matrix LOGICALLY into G partitions but
# they all still live on the one physical GPU, so it does NOT reduce device
# memory. Large cases (test1/6VYB, test2/1vsz) therefore OOM here -- validate
# those on the real 2-GPU cluster node. Use the SMALL analytical cases locally:
# test0 (1CCM), test3/test4 (Kirkwood, which also check the analytical energy).
set -u

NGPB="${NGPB:-/usr/local/nextgenPB/src/ngpb}"
TESTDIR="${1:-/usr/local/nextgenPB/test0}"
NRANKS="${2:-8}"
FAKE_GPUS="${3:-2}"
TOL="${TOL:-1e-6}"

cd "$TESTDIR" || { echo "no such dir: $TESTDIR"; exit 1; }
[ -x "$NGPB" ] || { echo "ngpb not built at $NGPB (run: cd src && make clean && make all)"; exit 1; }

# AMGX variant of this case's options.prm.
PRM_AMGX="$(mktemp --suffix=.prm)"
sed 's/^[[:space:]]*linear_solver[[:space:]]*=.*/linear_solver = amgx/' options.prm > "$PRM_AMGX"
grep -qiE '^[[:space:]]*linear_solver[[:space:]]*=[[:space:]]*amgx' "$PRM_AMGX" \
  || echo "linear_solver = amgx" >> "$PRM_AMGX"

OUT_DIST="$(mktemp)"; OUT_GATH="$(mktemp)"

echo ">> [1/2] distributed  (NGPB_GPUS_PER_NODE=$FAKE_GPUS, -n $NRANKS)"
NGPB_GPUS_PER_NODE="$FAKE_GPUS" \
  mpirun -n "$NRANKS" "$NGPB" --prmfile "$PRM_AMGX" > "$OUT_DIST" 2>&1

echo ">> [2/2] gather-to-0  (NGPB_AMGX_FORCE_GATHER=1, -n $NRANKS)"
NGPB_GPUS_PER_NODE="$FAKE_GPUS" NGPB_AMGX_FORCE_GATHER=1 \
  mpirun -n "$NRANKS" "$NGPB" --prmfile "$PRM_AMGX" > "$OUT_GATH" 2>&1

# Confirm each run took the intended path (banner printed by main()).
grep -q "distributed, multi-GPU" "$OUT_DIST" || echo "WARN: run 1 did not report the distributed path"
grep -q "single-GPU"            "$OUT_GATH" || echo "WARN: run 2 did not report the single-GPU path"

echo "----------------------------------------------------------------"
LABELS=( "Flux charge" "Polarization energy" "Direct ionic energy" \
         "Coulombic energy" "Sum of electrostatic energy" )
fail=0
for lbl in "${LABELS[@]}"; do
  a=$(grep -F "$lbl" "$OUT_DIST" | tail -1 | grep -oE '[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?' | tail -1)
  b=$(grep -F "$lbl" "$OUT_GATH" | tail -1 | grep -oE '[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?' | tail -1)
  if [ -z "$a" ] || [ -z "$b" ]; then
    printf "  %-30s : (absent in output)\n" "$lbl"; continue
  fi
  rel=$(awk -v x="$a" -v y="$b" 'BEGIN{d=x-y; if(d<0)d=-d; s=(x<0?-x:x); if(s<1)s=1; printf "%.2e", d/s}')
  ok=$(awk -v r="$rel" -v t="$TOL" 'BEGIN{print (r+0<=t+0)?"ok":"FAIL"}')
  [ "$ok" = "FAIL" ] && fail=1
  printf "  %-30s : dist=%s  gather=%s  relΔ=%s  [%s]\n" "$lbl" "$a" "$b" "$rel" "$ok"
done
echo "----------------------------------------------------------------"
echo "full logs: $OUT_DIST (dist)   $OUT_GATH (gather)"
rm -f "$PRM_AMGX"
if [ "$fail" -eq 0 ]; then echo "PARITY: PASS (tol=$TOL)"; else echo "PARITY: FAIL (see logs above)"; fi
exit $fail

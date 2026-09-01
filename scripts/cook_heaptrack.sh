#!/bin/sh
# Interpret raw heaptrack captures (from run_heaptrack.sh -r) into normal ones.
#
#     scripts/cook_heaptrack.sh test1/profiles/ngpb.rank*.raw.zst
#
# THE memory fix is unsetting DEBUGINFOD_URLS below. This machine has
#     DEBUGINFOD_URLS=https://debuginfod.ubuntu.com
# set globally, and heaptrack_interpret honors it: libdwfl then downloads the
# FULL DWARF for every unresolved module -- libc, libstdc++, libmpi, libcuda,
# libsc, libp4est -- and holds it in RAM while resolving. That is what grows to
# ~30 GB and gets OOM-killed, not the trace data. Measured on rank 1:
#     with debuginfod:     OOM-killed at ~28-30 GB
#     without:             211 MB peak RSS, 76 s, exit 0
# Nothing we care about is lost: ngpb and the bimpp libs carry their own local
# debug info, so our symbols still resolve (verified -- assemble_system_matrix
# still resolves to boost::container::new_allocator at 1.02 G). Only file/line
# *inside system libraries* goes missing, which we never look at.
# If heaptrack_gui is also slow or hungry when opening a file, unset it there
# too: env -u DEBUGINFOD_URLS heaptrack_gui foo.zst
#
# Serial is still the right default: one resolver at a time, and because the
# raw file stays on disk a failed cook costs a re-cook, not another 12-minute
# profiling run.
set -e

# HARD MEMORY CAP -- do not remove. Uncapped, a runaway heaptrack_interpret
# does not just fail: it triggers the global OOM killer, which takes down the
# desktop session, the editor and anything else running. That happened on
# 2026-09-01 cooking pre-opt/ngpb.rank0.raw.zst. Capped, it dies alone with
# bad_alloc and everything else survives.
# Raise deliberately (MAXMEM_KB=16777216 for 16 GiB) only when you are willing
# to sit and watch it.
MAXMEM_KB=${MAXMEM_KB:-8388608}
ulimit -v "$MAXMEM_KB" || echo "warning: could not set memory cap" >&2

# NOT the fix -- kept because it is cheap and harmless, but the rank 0 blow-up
# happens with these set, so do not credit them with anything. Unsetting them
# was measured on rank 1 only, without a with-debuginfod control, so their
# actual effect is UNKNOWN.
unset DEBUGINFOD_URLS
export HEAPTRACK_ENABLE_DEBUGINFOD=0

INTERPRET=${INTERPRET:-/usr/lib/heaptrack/libexec/heaptrack_interpret}

if [ ! -x "$INTERPRET" ]; then
    echo "no heaptrack_interpret at $INTERPRET (override with INTERPRET=)" >&2
    exit 1
fi

if [ $# -eq 0 ]; then
    echo "usage: $0 <file.raw.zst> [...]" >&2
    exit 1
fi

for raw in "$@"; do
    case "$raw" in
        *.raw.zst) cooked=$(printf '%s' "$raw" | sed 's/\.raw\.zst$/.zst/') ;;
        *) echo "skipping $raw: not a .raw.zst" >&2; continue ;;
    esac

    if [ -e "$cooked" ]; then
        echo "skipping $raw: $cooked already exists" >&2
        continue
    fi

    echo "=> $raw -> $cooked"
    # Write to a temporary first so an OOM mid-cook cannot leave a truncated
    # file that heaptrack_gui will happily open and mis-attribute -- that is
    # exactly how the rank 0 opt.amgx capture ended up showing all its memory
    # under one frame.
    if zstd -dc < "$raw" | "$INTERPRET" | zstd -c > "$cooked.part"; then
        mv "$cooked.part" "$cooked"
    else
        rm -f "$cooked.part"
        echo "FAILED on $raw (raw file kept, safe to retry)" >&2
        exit 1
    fi
done

echo "done"

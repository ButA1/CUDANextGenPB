#!/bin/sh
# Record raw; interpret afterwards with cook_heaptrack.sh.
#
# heaptrack's default pipeline is
#     heaptrack_interpret < pipe | zstd > out.zst
# i.e. a symbol resolver running *alongside* the job. With 4 ranks that is 4
# resolvers resident next to 4 copies of NGPB on a 31 GiB box, and the resolver
# is what grows without bound -- it accumulates the IP->symbol cache, the
# interned strings and the trace tree for the whole run. The OOM kills name
# "heaptrack_inter" (comm truncates heaptrack_interpret), never ngpb, which is
# why NGPB on its own is fine.
#
# -r drops the resolver from the recording: only zstd sits between the app and
# the disk, at constant memory. Interpretation then happens offline, one file
# at a time, with the whole machine to itself -- and if a cook does OOM it can
# be retried for free instead of costing another 12-minute run.
set -e

RANKS=${RANKS:-4}
OUT=${OUT:-$PWD/profiles}
export OUT

mkdir -p "$OUT"

# $OMPI_COMM_WORLD_RANK is expanded by each rank's own shell (single quotes),
# so the files come out named by rank rather than by pid.
mpirun -n "$RANKS" -x OUT sh -c \
    'exec heaptrack -r -o "$OUT/ngpb.rank$OMPI_COMM_WORLD_RANK" ngpb --prmfile options.prm'

echo
echo "Raw captures written to $OUT. Now interpret them:"
echo "    ../scripts/cook_heaptrack.sh $OUT/ngpb.rank*.raw.zst"

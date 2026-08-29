#!/usr/bin/env python3
"""
Shared CSV layer for the FMM sweep plotters.

Three schemas have to land in one shape, because the sweeps that produced them
were written months apart:

  bench_sweep.py      one row per full ngpb run: status / sweep / energy_method,
                      the fmm_* parameter names, t_energy for the whole stage.

  fmm_replay (old)    one row per replayed phase-2 evaluation, back when the two
                      trees shared a leaf size: method / mac / order / leaf.

  fmm_replay (new)    the same, after the source and target leaf sizes were
                      decoupled: src_leaf / tgt_leaf in place of leaf.

An old-schema row is the src_leaf == tgt_leaf diagonal of the new grid, and is
loaded as such -- that is what it measured. It is NOT interchangeable with a new
row at the same nominal leaf size, so do not concatenate the two CSVs.

Everything downstream sees: mac, p, sleaf, tleaf, time, err{pol,ionic,sum}.

Living here rather than in one of the plotters because the replay schema has
changed once already and broke its only consumer silently; the next change
should break one place, not two.
"""

import csv
import math
import statistics
import sys
from collections import defaultdict

ERR_FLOOR = 1e-16  # log axes cannot show an exact zero

METRICS = {
    "pol": ("energy_pol", "polarisation energy"),
    "ionic": ("energy_ionic", "direct ionic energy"),
    "sum": ("energy_sum", "total electrostatic energy"),
}


def _leaf_columns(row):
    """(source, target) leaf out of whichever replay schema this row is in."""
    if "src_leaf" in row:
        return row["src_leaf"], row["tgt_leaf"]
    # Pre-split: one leaf size drove both trees.
    return row["leaf"], row["leaf"]


def normalise_replay(rows):
    """Rewrite fmm_replay rows into the bench_sweep.py column names.

    One consequence worth knowing: the replay's only baseline is the naive GPU
    kernel, so it stands in for BOTH energy_method=0 (the error denominator) and
    energy_method=1 (the reference marks). A naive-vs-CPU floor band therefore
    collapses on replay-sourced data -- there is no independent reference for the
    naive path to deviate from. The meaningful floor there is the FMM's own
    run-to-run spread across repeats.
    """
    out = []

    for r in rows:
        pol = float(r["energy_pol"])
        ionic = float(r["energy_ionic"])
        coul = float(r.get("energy_coul") or 0.0)

        common = {
            "status": "ok",
            "sweep": "B",
            # Phase 2 only: tree build + polarisation + ionic. The phase-1 mesh
            # sweep is not re-run, so this is slightly below a pipeline t_energy.
            "t_energy": r["t_total_s"],
            "t_build": r.get("t_build_s", ""),
            "t_pol": r.get("t_pol_s", ""),
            "t_ionic": r.get("t_ionic_s", ""),
            "energy_pol": r["energy_pol"],
            "energy_ionic": r["energy_ionic"],
            "energy_sum": repr(pol + ionic + coul),
        }

        if r.get("method") == "fmm":
            sleaf, tleaf = _leaf_columns(r)
            out.append(dict(common,
                            energy_method="2",
                            fmm_mac=r["mac"],
                            fmm_multipole_order=r["order"],
                            fmm_leaf_size=sleaf,
                            fmm_target_leaf_size=tleaf))
        elif r.get("method") == "naive":
            out.append(dict(common, energy_method="0"))
            out.append(dict(common, energy_method="1"))

    return out


def load(csv_path):
    """Rows from either schema, normalised to bench_sweep.py column names."""
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))

    if rows and "method" in rows[0]:
        return normalise_replay(rows)

    return [r for r in rows if r.get("status") == "ok"]


def _tleaf_of(row):
    """Target leaf, defaulting to the source leaf.

    A bench_sweep row from before the split has no fmm_target_leaf_size at all,
    and one written with the key omitted has it empty; both mean "the target
    tree followed the source tree", which is what 0 means to fmm_replay too.
    """
    raw = (row.get("fmm_target_leaf_size") or "").strip()
    if not raw or int(float(raw)) == 0:
        return int(float(row["fmm_leaf_size"]))
    return int(float(raw))


def aggregate(rows, metric_col="energy_pol", require_baseline=True):
    """Collapse repeats -> one point per configuration.

    Time is the median over repeats (robust to a stray slow run); the energy is
    the mean (it barely moves, and averaging keeps the error estimate from riding
    on one sample).

    Returns (points, info). Each point carries mac / p / sleaf / tleaf / time /
    err / n, plus the phase split (t_build, t_pol, t_ionic) where the source CSV
    had one. info carries the baseline and naive reference values, any of which
    may be None when require_baseline is False.
    """
    sweep_b = [r for r in rows if r.get("sweep") == "B"]
    if not sweep_b:
        sys.exit("no sweep B rows in the CSV -- nothing to plot")

    base = [r for r in sweep_b if r.get("energy_method") == "0"]
    if not base and require_baseline:
        sys.exit("no energy_method=0 rows -- cannot form the FMM error baseline")

    ref = statistics.fmean(float(r[metric_col]) for r in base) if base else None

    naive = [r for r in sweep_b if r.get("energy_method") == "1"]
    naive_t = statistics.median(float(r["t_energy"]) for r in naive) if naive else None
    naive_e = None
    if naive and ref:
        naive_e = abs(statistics.fmean(float(r[metric_col]) for r in naive) - ref)
        naive_e = max(naive_e / abs(ref), ERR_FLOOR)

    groups = defaultdict(list)
    for r in sweep_b:
        if r.get("energy_method") != "2":
            continue
        key = (float(r["fmm_mac"]), int(r["fmm_multipole_order"]),
               int(float(r["fmm_leaf_size"])), _tleaf_of(r))
        groups[key].append(r)

    def med(grp, col):
        vals = [float(r[col]) for r in grp if (r.get(col) or "").strip()]
        return statistics.median(vals) if vals else None

    points = []
    for (mac, p, sleaf, tleaf), grp in sorted(groups.items()):
        errs = {}
        for name, (col, _) in METRICS.items():
            if not base:
                errs[name] = None
                continue
            val = statistics.fmean(float(r[col]) for r in grp)
            base_val = statistics.fmean(float(b[col]) for b in base)
            errs[name] = (max(abs(val - base_val) / abs(base_val), ERR_FLOOR)
                          if base_val else ERR_FLOOR)
        points.append({
            "mac": mac, "p": p, "sleaf": sleaf, "tleaf": tleaf,
            "pair": "{}/{}".format(sleaf, tleaf),
            "time": statistics.median(float(r["t_energy"]) for r in grp),
            "t_build": med(grp, "t_build"),
            "t_pol": med(grp, "t_pol"),
            "t_ionic": med(grp, "t_ionic"),
            "err": errs, "n": len(grp),
        })

    info = {
        "ref": ref,
        "naive_t": naive_t,
        "naive_e": naive_e,
        "base_t": (statistics.median(float(r["t_energy"]) for r in base)
                   if base else None),
        "molecule": (rows[0].get("molecule", "") if rows else "") or "",
        "natoms": (rows[0].get("num_atoms", "") if rows else "") or "",
    }
    return points, info


# --------------------------------------------------------------------------
#  Axis ticks. Both plotters need these and pgfplots' defaults are unreadable
#  on the narrow panels these figures use.
# --------------------------------------------------------------------------

def x_ticks(lo, hi):
    """A 1-2-5 ladder over [lo, hi]. The times span well under a decade, so the
    default log ticks come out as 10^{-1.5}, which is unreadable in print."""
    vals = []
    e = math.floor(math.log10(lo))
    while 10 ** e <= hi * 10:
        for m in (1, 2, 5):
            v = m * 10 ** e
            if lo <= v <= hi:
                vals.append(v)
        e += 1
    if len(vals) < 3:  # very narrow range: fall back to a finer ladder
        vals = []
        e = math.floor(math.log10(lo))
        while 10 ** e <= hi * 10:
            for m in (1, 1.5, 2, 3, 5, 7):
                v = m * 10 ** e
                if lo <= v <= hi:
                    vals.append(v)
            e += 1
    return sorted(vals)


def fmt_tick(v):
    if v >= 1:
        return "{:g}".format(v)
    return "{:.6f}".format(v).rstrip("0").rstrip(".")


def y_ticks(lo, hi):
    """Decade ticks, thinned to keep about five labels on a 5.4 cm panel."""
    e0, e1 = math.ceil(math.log10(lo)), math.floor(math.log10(hi))
    span = max(e1 - e0, 1)
    step = max(1, round(span / 5))
    return [10.0 ** e for e in range(e1, e0 - 1, -step)][::-1]
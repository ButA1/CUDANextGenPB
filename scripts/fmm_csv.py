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
            # Carried so a caption can state the rank count without being told
            # it. The replay calls it "ranks", bench_sweep.py calls it "np";
            # both end up here under the bench_sweep name.
            "np": r.get("ranks", ""),
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


TIME_STATS = {"median": statistics.median, "min": min, "mean": statistics.fmean}


def aggregate(rows, metric_col="energy_pol", require_baseline=True,
              stat="median"):
    """Collapse repeats -> one point per configuration.

    `stat` picks how the repeats of a timing collapse to one number.  "median"
    is the default and is right for a quiet machine.  Use "min" for shared
    cluster nodes: contention from a co-scheduled job only ever makes a run
    slower, so the fastest repeat is the closest thing to an uncontended
    measurement.  On the TU Berlin A100 runs 11 of 168 ptheta configurations had
    one repeat 3-6x slower than its siblings, enough to move a median and flip
    the winner in a cell; the leaf-sweep jobs on the same hardware had none.

    The energy is always the mean -- it barely moves between repeats, and
    averaging keeps the error estimate from riding on a single sample.

    Returns (points, info). Each point carries mac / p / sleaf / tleaf / time /
    err / n, plus the phase split (t_build, t_pol, t_ionic) where the source CSV
    had one. info carries the baseline and naive reference values, any of which
    may be None when require_baseline is False.
    """
    try:
        collapse = TIME_STATS[stat]
    except KeyError:
        sys.exit("unknown --stat %r (want one of %s)"
                 % (stat, ", ".join(sorted(TIME_STATS))))
    sweep_b = [r for r in rows if r.get("sweep") == "B"]
    if not sweep_b:
        sys.exit("no sweep B rows in the CSV -- nothing to plot")

    base = [r for r in sweep_b if r.get("energy_method") == "0"]
    if not base and require_baseline:
        sys.exit("no energy_method=0 rows -- cannot form the FMM error baseline")

    ref = statistics.fmean(float(r[metric_col]) for r in base) if base else None

    naive = [r for r in sweep_b if r.get("energy_method") == "1"]
    naive_t = collapse(float(r["t_energy"]) for r in naive) if naive else None
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
        return collapse(vals) if vals else None

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
            "time": collapse(float(r["t_energy"]) for r in grp),
            "t_build": med(grp, "t_build"),
            "t_pol": med(grp, "t_pol"),
            "t_ionic": med(grp, "t_ionic"),
            "err": errs, "n": len(grp),
        })

    info = {
        "ref": ref,
        "naive_t": naive_t,
        "naive_e": naive_e,
        "base_t": (collapse(float(r["t_energy"]) for r in base)
                   if base else None),
        "molecule": (rows[0].get("molecule", "") if rows else "") or "",
        "natoms": (rows[0].get("num_atoms", "") if rows else "") or "",
        "ranks": ranks_of(rows),
    }
    return points, info


def _f(x):
    x = (x or "").strip()
    return float(x) if x else None


def stock_floor(bench_csv, replay_csv=None, metric_col="energy_pol"):
    r"""How well stock NextGenPB reproduces its own polarisation energy.

    The sweep figures plot FMM error against the naive $O(N^2)$ GPU sum, which
    invites the question the number alone cannot answer: how accurate is the code
    being replaced?  This measures that, as the largest disagreement between
    evaluations that are supposed to give the identical answer:

      cpu_vs_naive        the stock CPU path (energy_method=0) against the naive
                          GPU sum (energy_method=1) of the SAME pipeline run, so
                          both see the same solved potential.  This is the pure
                          summation-order difference.
      spread_cpu          run-to-run spread of energy_method=0 at one fixed
                          configuration, and likewise spread_naive.  Non-zero
                          because AMGX is not bitwise reproducible, so phi -- and
                          with it the energy -- moves slightly between runs.
      pipeline_vs_replay  the naive result of the pipeline run against the naive
                          result the replay computes from the dump.  Different
                          pipeline runs, so this again carries the solver's
                          non-determinism; it bounds how far the replay's y-axis
                          reference can sit from the pipeline's.

    The floor is the max of whichever of these the CSVs support.  Reporting only
    cpu_vs_naive would over-claim: on 1VSZ it is 2.9e-13 while the repeat spread
    at a fixed configuration is 2.8e-12, i.e. an order of magnitude larger, so a
    line drawn at 2.9e-13 would sit well inside the noise of its own measurement.

    Only a full-pipeline bench_sweep.py CSV can supply this.  A replay CSV cannot:
    its only reference IS the naive kernel, so the CPU path is never evaluated
    there (see normalise_replay).

    Returns a dict of the components plus "floor", or None if bench_csv holds no
    solver configuration with both energy methods.
    """
    with open(bench_csv, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("status") == "ok"]

    # Grouped by solver configuration on purpose: energy_pol rides on the solved
    # potential, so comparing an LIS run against an AMGX one measures the linear
    # solve and not the energy summation at all.
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        method = r.get("energy_method")
        if method not in ("0", "1"):
            continue
        val = _f(r.get(metric_col))
        if val is None:
            continue
        key = (r.get("np"), r.get("linear_solver"), r.get("amgx_config"),
               r.get("sweep"))
        groups[key][method].append(val)

    usable = [(k, d) for k, d in groups.items() if d["0"] and d["1"]]
    if not usable:
        return None
    # The best-sampled configuration, so the spreads rest on as many repeats as
    # the data allows.
    _, best = max(usable, key=lambda kd: len(kd[1]["0"]) + len(kd[1]["1"]))

    cpu, naive = best["0"], best["1"]
    mean_cpu, mean_naive = statistics.fmean(cpu), statistics.fmean(naive)

    out = {
        "cpu_vs_naive": abs(mean_naive - mean_cpu) / abs(mean_cpu),
        "spread_cpu": (max(cpu) - min(cpu)) / abs(mean_cpu),
        "spread_naive": (max(naive) - min(naive)) / abs(mean_naive),
        "n_cpu": len(cpu), "n_naive": len(naive),
        "cpu_energy": mean_cpu, "naive_energy": mean_naive,
        "pipeline_vs_replay": None,
    }

    if replay_csv:
        with open(replay_csv, newline="") as fh:
            vals = [_f(r.get(metric_col)) for r in csv.DictReader(fh)
                    if r.get("method") == "naive"]
        vals = [v for v in vals if v is not None]
        if vals:
            mean_replay = statistics.fmean(vals)
            out["pipeline_vs_replay"] = (abs(mean_naive - mean_replay)
                                         / abs(mean_replay))

    out["floor"] = max(v for k, v in out.items()
                       if k in ("cpu_vs_naive", "spread_cpu", "spread_naive",
                                "pipeline_vs_replay") and v is not None)
    return out


def machine_note(machine, ranks):
    r"""The "measured on X with N ranks" sentence every figure caption carries.

    A reviewer asked for it on every figure, so it is built in one place and the
    plotters only supply the machine name. `ranks` comes from the CSV, never from
    the caller, so it cannot disagree with the data it labels.
    """
    where = machine or "\\textbf{[machine not recorded]}"
    if ranks:
        return " Measured on %s with %s MPI ranks." % (where, ranks)
    return (" Measured on %s; the rank count is not recorded in the source CSV."
            % where)


def add_machine_argument(ap):
    """The --machine flag, worded the same way for every plotter."""
    ap.add_argument("--machine", default=None,
                    help="where the data was captured, inserted verbatim into "
                         "the caption, e.g. \"the local system "
                         "(Table~\\ref{tab:test-systems})\" or \"an A100 node of "
                         "the TU Berlin HPC\". The rank count is read from the "
                         "CSV and appended automatically. Omitting this puts a "
                         "visible [machine not recorded] in the caption rather "
                         "than quietly leaving it out.")


def ranks_of(rows):
    """The MPI rank count these rows were measured at, or None if not one value.

    Captions have to state it -- the energy stage is distributed over the ranks,
    so a time measured at 4 ranks is not comparable to one at 16 -- and deriving
    it from the CSV is the only way it cannot silently disagree with the data.
    Mixed rank counts return None rather than a number, because a single figure
    drawn across two of them has a problem no caption can fix.
    """
    seen = {(r.get("np") or "").strip() for r in rows}
    seen.discard("")
    return seen.pop() if len(seen) == 1 else None


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
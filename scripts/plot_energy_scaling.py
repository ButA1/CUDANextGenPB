#!/usr/bin/env python3
r"""
How the energy stage scales across GPUs and topologies.

The companion to scripts/plot_amgx_scaling.py, on the same five configurations
and the same x axis, so the two figures can be read against each other:

    RTX 3080, 1 GPU, 4 ranks   ->  A100, 1 GPU, 4 ranks     only the GPU
    A100, 1 GPU, 4 ranks       ->  A100, 1 GPU, 16 ranks    only the rank count
    A100, 1 GPU, 16 ranks      ->  A100, 2 GPUs, 1 node     only the GPU count
    A100, 2 GPUs, 1 node       ->  A100, 2 GPUs, 2 nodes    only where GPU 2 sits

The rank-count step means something different here than it does for AMGX. There
the whole matrix is gathered onto one GPU whatever fed it, so the step is a
control; the energy stage keeps its work distributed across ranks, so it is a
real scaling step.

Two panels because the FMM and the naive kernel differ by ~7x on the local GPU,
and a shared y axis would flatten the FMM panel into the axis. A series with no
run for a panel's method is drawn as a gap and marked, not silently omitted.

Both panels are logarithmic. The naive panel spans 19x on its own -- the local
GPU's FP64 rate, not the topology -- and on a linear axis the four A100 bars
that the figure is actually about collapse into stubs against the local one.
The sibling figure (figures/energy_methods) is log for the same reason.

Each bar is the `t_energy` stage timer -- the whole stage, so it includes the
host-side collection of the flux points and surface triangles, not just the
kernel.

Rows are pooled over every run in the CSV that shares an energy method and, for
the FMM, an (mac, order, leaf, target) tuple -- across linear solvers and
repeats, since the solver runs before the energy stage and cannot reach it.
That premise is checked rather than assumed: --max-pool-spread aborts if the
pooled CONFIGURATIONS disagree in their medians by more than that fraction.

Writes, relative to the thesis directory:
    figures/data/<tag>_m<i>.dat   one row per series, one file per method panel
    figures/data/<tag>_ref.tex    caption values as macros
    figures/<tag>.tex             the figure
    figures/<tag>_table.tex       the same numbers plus the pipeline context

The table carries `t_report_total_s` and the stage's share of it on purpose.
The energy stage gaining 1.4x does not make the run 1.4x faster, and the figure
alone invites exactly that reading. It also carries the CPU path, which was
measured in every topology but is 300x off the panel scale.
"""

import argparse
import csv
import os
import re
import statistics
import sys

# The same two blues as figures/energy_methods.tex, so the FMM is the same
# colour in both energy figures and the naive kernel is the same lighter one.
COLOUR_FMM = "2C5F8D"
COLOUR_NAIVE = "9DC3E0"

ENERGY_MACROS = ["energyScalingMolecule", "energyScalingRef",
                 "energyScalingRepeats", "energyScalingGap"]

# energy_method column value -> (key, panel title, colour macro, table heading).
METHODS = {
    "2": ("fmm", "FMM", "escalefmm", "FMM"),
    "1": ("naive", r"naive $O(NM)$", "escalenaive", r"naive GPU $O(NM)$"),
    "0": ("cpu", "CPU", None, "CPU"),
}
PANEL_ORDER = ["fmm", "naive"]

FMM_COLS = ["fmm_mac", "fmm_multipole_order", "fmm_leaf_size",
            "fmm_target_leaf_size"]


def localise(text, tag):
    r"""Suffix this figure's macros and labels so several can coexist."""
    sfx = "".join(c for c in tag if c.isalpha())
    # (?![A-Za-z]) because a control sequence ends at the first non-letter.
    for name in ENERGY_MACROS:
        text = re.sub(r"\\" + name + r"(?![A-Za-z])", "\\\\" + name + sfx, text)
    return text.replace("SFX", sfx).replace("TAG", tag)


def texify(s):
    """Escape the few characters that appear in these labels and break TeX."""
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def fmm_tuple(row):
    """The FMM parameters of a row, or None if it did not run the FMM."""
    vals = [(row.get(c) or "").strip() for c in FMM_COLS]
    return tuple(vals) if any(vals) else None


def fmm_label(t):
    r"""(mac, order, leaf, target) -> a compact slash tuple for the table.

    The spelled-out form ($\theta=0.4$, $P=11$, $n_{leaf}=16/1024$) runs the
    seven-column table 44pt past the text block, so the parameter names live in
    the column header and the cells carry values only.
    """
    if t is None:
        return "---"
    return "/".join(v for v in t if v)


def log_axis(lo, hi):
    """(ymin, ymax, [ticks]) covering [lo, hi] on the 1-2-5 decade series.

    Explicit ticks because a log axis spanning well under a decade -- which the
    FMM panel does -- otherwise gets one labelled tick and a scatter of unnamed
    minor ones.
    """
    steps = []
    e = -6
    while 10.0 ** e <= hi * 10:
        for m in (1, 2, 5):
            steps.append(m * 10.0 ** e)
        e += 1
    below = [s for s in steps if s <= lo]
    above = [s for s in steps if s >= hi]
    ymin = below[-1] if below else steps[0]
    ymax = above[0] if above else steps[-1]
    return ymin, ymax, [s for s in steps if ymin <= s <= ymax]


def fmt_tick(v):
    return ("%g" % v) if v >= 1 else ("%g" % v).lstrip("0")


def fmt_spread(s, pct="\\%"):
    """Repeat-to-repeat spread, with the warm figure in brackets if one row of
    the pool was a cold start. A single run has no spread and says so."""
    if s["spread"] is None:
        return "---"
    if s["warm_spread"] is None:
        return "%.0f%s" % (100 * s["spread"], pct)
    return "%.0f%s (%.0f%s)" % (100 * s["spread"], pct,
                                100 * s["warm_spread"], pct)


def stats_for(rs):
    """One bar's worth of numbers from a pool of rows, sorted by timestamp."""
    rs = sorted(rs, key=lambda r: r.get("timestamp") or "")
    t = [float(r["t_energy"]) for r in rs]
    total = [float(r["t_report_total_s"]) for r in rs
             if (r.get("t_report_total_s") or "").strip()]

    # Per-config_id medians. Every config_id pooled into one bar differs only in
    # its linear solver, which finishes before the energy stage starts -- so
    # these medians agreeing is the evidence that the pool is one measurement,
    # and it is what --max-pool-spread gates on. It deliberately does not gate
    # on repeat-to-repeat spread, which is a property of the machine; the table
    # reports that per row.
    per_cfgid = {}
    for r in rs:
        per_cfgid.setdefault(r["config_id"], []).append(float(r["t_energy"]))
    med = [statistics.median(v) for v in per_cfgid.values()]

    def spread_of(v):
        # None, not 0.0: a single run has no repeat-to-repeat spread to report,
        # and "0%" would claim it was perfectly reproducible.
        return ((max(v) - min(v)) / statistics.median(v)) if len(v) > 1 else None

    # Cold start, tested rather than assumed: the slowest run must BE the
    # earliest one, and dropping it must bring the rest into a normal band.
    # Where that does not hold, nothing is excluded and the full spread stands.
    warm = t[1:]
    cold = (len(t) > 2 and t[0] == max(t)
            and spread_of(warm) < 0.15 <= spread_of(t))

    return {
        "t": statistics.median(t),
        "spread": spread_of(t),
        "warm_spread": spread_of(warm) if cold else None,
        "pool_spread": ((max(med) - min(med)) / statistics.median(med)
                        if len(med) > 1 else 0.0),
        "n_cfgid": len(per_cfgid),
        "n": len(rs),
        "pipeline": statistics.median(total) if total else None,
    }


def load_series(spec, fmm_pick):
    """"LABEL=path" -> (label, ranks, {method: stats}, molecule).

    LABEL may contain '|', which becomes a line break in the tick label.

    The FMM parameters are not part of the pool key by accident: a folder may
    hold several swept configurations, which are different measurements and
    must not be averaged together. One is chosen per series -- the fastest,
    unless --fmm names one -- and recorded so the table can state it.
    """
    if "=" not in spec:
        sys.exit("series must be LABEL=path/to/bench_runs.csv, got %r" % spec)
    label, path = spec.split("=", 1)
    label, path = label.strip(), path.strip()
    if not os.path.exists(path):
        sys.exit("no such CSV: %s" % path)

    with open(path) as fh:
        rows = [r for r in csv.DictReader(fh)
                if (r.get("status") or "ok") == "ok"
                and (r.get("t_energy") or "").strip()]
    if not rows:
        sys.exit("no successful rows with a t_energy timer in %s" % path)

    ranks = {(r.get("np") or "").strip() for r in rows}
    ranks.discard("")
    if len(ranks) != 1:
        sys.exit("%s mixes rank counts %s -- one series is one rank count"
                 % (path, sorted(ranks)))

    by_method = {}
    for r in rows:
        m = METHODS.get((r.get("energy_method") or "").strip())
        if m:
            by_method.setdefault(m[0], []).append(r)

    out = {}
    for key, rs in by_method.items():
        if key != "fmm":
            out[key] = stats_for(rs)
            out[key]["config"] = None
            continue
        by_tuple = {}
        for r in rs:
            by_tuple.setdefault(fmm_tuple(r), []).append(r)
        if fmm_pick and fmm_pick in by_tuple:
            chosen = fmm_pick
        elif fmm_pick:
            sys.exit("%s has no FMM rows at %s -- it has %s"
                     % (path, ",".join(fmm_pick),
                        "; ".join(",".join(t) for t in by_tuple)))
        else:
            chosen = min(by_tuple,
                         key=lambda t: statistics.median(
                             float(r["t_energy"]) for r in by_tuple[t]))
        out[key] = stats_for(by_tuple[chosen])
        out[key]["config"] = chosen
        out[key]["n_configs"] = len(by_tuple)

    molecules = {(r.get("molecule") or "").strip() for r in rows}
    return (label, ranks.pop(), out,
            os.path.splitext(sorted(molecules)[0])[0] if molecules else "?")


FIGURE_TEMPLATE = r"""% ------------------------------------------------------------------------
%  Energy-stage scaling across GPUs and topologies.
%  Generated by scripts/plot_energy_scaling.py -- regenerate rather than edit.
%  Include with \input{figures/TAG}
%  Needs \usepackage{pgfplots} in the preamble.
% ------------------------------------------------------------------------
\input{figures/data/TAG_ref}

\pgfplotsset{compat=1.16}
\usepgfplotslibrary{groupplots}

\definecolor{escalefmm}{HTML}{COLFMM}
\definecolor{escalenaive}{HTML}{COLNAIVE}

\begin{figure}[!htb]
\centering
\begin{tikzpicture}
\begin{groupplot}[
    group style={
      group size=2 by 1,
      horizontal sep=1.5cm,
    },
    width=7.4cm, height=6.2cm,
    enlarge x limits=0.14,
    ymode=log, log basis y=10,
    log ticks with fixed point,
    ylabel={energy stage [s]},
    ylabel style={font=\footnotesize},
    symbolic x coords={SYMCOORDS},
    xtick={SYMCOORDS},
    xticklabels={XTICKLABELS},
    xticklabel style={font=\tiny, align=center, yshift=-0.2ex},
    tick label style={font=\scriptsize},
    title style={font=\footnotesize, yshift=-0.4ex},
    ymajorgrids=true,
    major grid style={black!12},
]
PANELS
\end{groupplot}
\end{tikzpicture}

\caption[Energy-stage scaling across GPUs and topologies]{\textbf{Energy-stage
    time on \energyScalingMolecule{} over five GPU configurations.} Median over
    at least \energyScalingRepeats{} runs per bar, from the \texttt{t\_energy}
    stage timer. The figure above each bar is its time relative to
    \energyScalingRef{}. The two panels have their own $y$ scales.\energyScalingGap{}}
\label{fig:TAG}
\end{figure}
"""

# `ybar` and `bar width` live here rather than in the shared groupplot options:
# the groupplot library rejects `bar width` at that level, because the bar
# handler that defines the key has not been installed yet when the shared list
# is parsed. That holds for plain `ybar`, not just `ybar stacked`.
PANEL_TEMPLATE = r"""
\nextgroupplot[ybar, bar width=13pt, title={TITLE}, ymin=YMIN, ymax=YMAX,
               ytick={YTICKS}, yticklabels={YTICKLABELS}]
  \addplot[fill=COLOUR, draw=COLOUR!70!black, line width=0.4pt]
    table[x=sx, y=t] {figures/data/DATFILE.dat};
  % Explicit nodes rather than `nodes near coords`, so a series with no bar in
  % this panel gets its "not measured" mark in the same pass and at the same
  % anchor as the ratios.
LABELS"""

TABLE_TEMPLATE = r"""% ------------------------------------------------------------------------
%  Generated by scripts/plot_energy_scaling.py -- regenerate rather than edit.
%  Include with \input{figures/TAG_table}
%  Needs \usepackage{booktabs}.
% ------------------------------------------------------------------------
\begin{table}[!htb]
\centering
\footnotesize
\setlength{\tabcolsep}{4.5pt}
\caption[Energy-stage scaling against the whole pipeline]{\textbf{The numbers
    behind figure~\ref{fig:TAG}.} \emph{energy} is the \texttt{t\_energy} stage
    timer, \emph{pipeline} is \texttt{t\_report\_total\_s}, the sum of every
    reported stage, and \emph{share} is the first as a fraction of the second.
    \emph{spread} is the widest repeat-to-repeat variation in the row.}
\label{tab:TAG}
\begin{tabular}{llrrrrr}
\toprule
platform & FMM config & energy [s] & vs.\ ref. & spread & pipeline [s] & share \\
         & {\footnotesize $\theta$/$P$/$n_{leaf,src}$/$n_{leaf,tgt}$} & & & & & \\
ROWS\bottomrule
\end{tabular}
\end{table}
"""


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("series", nargs="+",
                    help="LABEL=path/to/bench_runs.csv, in the order they should "
                         "be drawn. '|' inside LABEL becomes a line break in the "
                         "tick label")
    ap.add_argument("--thesis", default="thesis")
    ap.add_argument("--tag", default="energy_scaling")
    ap.add_argument("--reference", type=int, default=None,
                    help="0-based index of the series every bar is divided by. "
                         "Default is the LAST single-GPU cluster series, i.e. the "
                         "point the topology sweep actually starts from")
    ap.add_argument("--fmm", default=None,
                    help="mac,order,leaf,target to pin the FMM configuration to "
                         "in every series (default: the fastest one each series "
                         "measured)")
    ap.add_argument("--max-pool-spread", type=float, default=0.15,
                    help="abort if the CONFIGURATIONS pooled into one bar "
                         "disagree in their medians by more than this fraction "
                         "(default 0.15). Pooling across linear solvers is only "
                         "valid because the solver finishes before the energy "
                         "stage; this is the check on that. It is not a gate on "
                         "repeat-to-repeat noise, which the table reports per row")
    ap.add_argument("--molecule", default=None)
    args = ap.parse_args()

    fmm_pick = tuple(c.strip() for c in args.fmm.split(",")) if args.fmm else None
    if fmm_pick and len(fmm_pick) != len(FMM_COLS):
        sys.exit("--fmm needs %d comma-separated values (%s)"
                 % (len(FMM_COLS), ",".join(FMM_COLS)))

    loaded = [load_series(s, fmm_pick) for s in args.series]
    if len(loaded) < 2:
        sys.exit("need at least two series to compare")

    molecules = {mol for _, _, _, mol in loaded}
    if len(molecules) != 1 and not args.molecule:
        sys.exit("the series disagree on the molecule (%s); pass --molecule to "
                 "override if that is intended" % sorted(molecules))

    for label, _, st, _ in loaded:
        for key, s in st.items():
            if s["pool_spread"] > args.max_pool_spread:
                sys.exit("series %r, method %r: the %d pooled configurations "
                         "disagree by %.1f%% in their medians, over the %.1f%% "
                         "limit -- the linear solver IS reaching the energy "
                         "stage, or these are not the same measurement.\n"
                         "  Raise --max-pool-spread only after checking why."
                         % (label, key, s["n_cfgid"], 100 * s["pool_spread"],
                            100 * args.max_pool_spread))

    ref = args.reference
    if ref is None:
        # The last series before the GPU count leaves 1: the topology sweep's own
        # starting point, which is what "scales by N" should be measured against.
        ref = 0
        for i, (label, _, _, _) in enumerate(loaded):
            if "2 GPU" in label or "2 gpu" in label.lower():
                break
            ref = i
    if not 0 <= ref < len(loaded):
        sys.exit("--reference %d is outside 0..%d" % (ref, len(loaded) - 1))

    panels_present = [k for k in PANEL_ORDER
                      if any(k in st for _, _, st, _ in loaded)]
    for key in panels_present:
        if key not in loaded[ref][2]:
            sys.exit("the reference series %r has no %s run, so the %s panel has "
                     "nothing to divide by -- pass --reference"
                     % (loaded[ref][0].replace("|", ", "), key, key))

    data_dir = os.path.join(args.thesis, "figures", "data")
    os.makedirs(data_dir, exist_ok=True)
    reftex = os.path.join(data_dir, args.tag + "_ref.tex")
    fig = os.path.join(args.thesis, "figures", args.tag + ".tex")
    tab = os.path.join(args.thesis, "figures", args.tag + "_table.tex")

    sym = ["s%d" % i for i in range(len(loaded))]
    ticklabels = ", ".join("{%s}" % texify(lab).replace("|", r"\\")
                           for lab, _, _, _ in loaded)

    panels, gaps = [], []
    for mi, key in enumerate(panels_present):
        datname = "%s_m%d" % (args.tag, mi)
        refval = loaded[ref][2][key]["t"]
        vals = [st[key]["t"] for _, _, st, _ in loaded if key in st]
        # 1.25x headroom above the tallest bar so its ratio label clears the top.
        ymin, ymax, ticks = log_axis(min(vals), max(vals) * 1.25)
        labels = []
        with open(os.path.join(data_dir, datname + ".dat"), "w") as fh:
            fh.write("sx t\n")
            for i, (lab, _, st, _) in enumerate(loaded):
                if key not in st:
                    # No bar, but the tick stays (xtick is an explicit list, not
                    # `xtick=data`), so the gap is visible and is labelled as a
                    # gap rather than reading as a zero.
                    gaps.append((key, lab))
                    labels.append(
                        "  \\node[font=\\tiny, text=black!45, anchor=west, "
                        "rotate=90, xshift=1pt] at (axis cs:%s,%.6g) "
                        "{not measured};\n" % (sym[i], ymin))
                    continue
                fh.write("%s %.6g\n" % (sym[i], st[key]["t"]))
                labels.append(
                    "  \\node[font=\\tiny, text=black!55, anchor=south, "
                    "yshift=1pt] at (axis cs:%s,%.6g) {$\\times$%.2f};\n"
                    % (sym[i], st[key]["t"], refval / st[key]["t"]))
        panels.append(PANEL_TEMPLATE
                      .replace("TITLE", METHODS_TITLE[key])
                      .replace("YMIN", "%.6g" % ymin)
                      .replace("YMAX", "%.6g" % ymax)
                      .replace("YTICKS", ",".join("%.6g" % t for t in ticks))
                      .replace("YTICKLABELS",
                               ",".join(fmt_tick(t) for t in ticks))
                      .replace("COLOUR", METHODS_COLOUR[key])
                      .replace("LABELS", "".join(labels))
                      .replace("DATFILE", datname))

    npool = min(s["n"] for _, _, st, _ in loaded
                for k, s in st.items() if k in panels_present)
    # One short clause, only when a panel actually has a hole in it.
    gap_note = ""
    if gaps:
        clause = "; ".join(
            "the %s was not run on %s" % (METHODS_TITLE[k].replace("$", ""),
                                          texify(lab).replace("|", ", "))
            for k, lab in gaps)
        # It follows a full stop in the caption, so it is its own sentence.
        gap_note = " " + clause[0].upper() + clause[1:] + "."
    ref_buf = [
        "% generated by scripts/plot_energy_scaling.py -- do not edit\n",
        "\\def\\energyScalingMolecule{%s}\n"
        % (args.molecule or sorted(molecules)[0]),
        "\\def\\energyScalingRef{%s}\n"
        % texify(loaded[ref][0]).replace("|", ", "),
        "\\def\\energyScalingRepeats{%d}\n" % npool,
        "\\def\\energyScalingGap{%s}\n" % gap_note,
    ]
    with open(reftex, "w") as fh:
        fh.write(localise("".join(ref_buf), args.tag))

    with open(fig, "w") as fh:
        fh.write(localise(FIGURE_TEMPLATE
                          .replace("COLFMM", COLOUR_FMM)
                          .replace("COLNAIVE", COLOUR_NAIVE)
                          .replace("SYMCOORDS", ",".join(sym))
                          .replace("XTICKLABELS", ticklabels)
                          .replace("PANELS", "".join(panels)), args.tag))

    # The CPU path gets a block in the table but no panel: it was measured in
    # every topology, and leaving it out would hide that it is the only path
    # whose cost the extra ranks actually change.
    rows = []
    for key in PANEL_ORDER + ["cpu"]:
        if not any(key in st for _, _, st, _ in loaded):
            continue
        rows.append("\\midrule\n\\multicolumn{7}{l}{\\emph{%s}} \\\\\n"
                    % METHODS_HEAD[key])
        refval = loaded[ref][2][key]["t"] if key in loaded[ref][2] else None
        for label, ranks, st, _ in loaded:
            plat = texify(label).replace("|", ", ")
            if key not in st:
                rows.append("%s & \\multicolumn{6}{c}{not measured} \\\\\n" % plat)
                continue
            s = st[key]
            rows.append(
                "%s & %s & %.3f & %s & %s & %s & %s \\\\\n"
                % (plat, fmm_label(s["config"]), s["t"],
                   "---" if refval is None else "$\\times$%.2f" % (refval / s["t"]),
                   fmt_spread(s),
                   "---" if s["pipeline"] is None else "%.1f" % s["pipeline"],
                   "---" if s["pipeline"] is None
                   else "%.1f\\%%" % (100 * s["t"] / s["pipeline"])))
    with open(tab, "w") as fh:
        fh.write(localise(TABLE_TEMPLATE.replace("ROWS", "".join(rows)), args.tag))

    print("reference series: [%d] %s\n" % (ref, loaded[ref][0].replace("|", ", ")))
    for key in PANEL_ORDER + ["cpu"]:
        if not any(key in st for _, _, st, _ in loaded):
            continue
        print("--- energy_method %s (%s)" % (
            next(k for k, v in METHODS.items() if v[0] == key), key))
        refval = loaded[ref][2][key]["t"] if key in loaded[ref][2] else None
        for label, ranks, st, _ in loaded:
            if key not in st:
                print("   %-34s np=%-3s not measured" % (
                    label.replace("|", ", "), ranks))
                continue
            s = st[key]
            print("   %-34s np=%-3s n=%-2d t_energy %8.3f  %-7s spread %-14s "
                  "pipeline %-7s share %-6s %s"
                  % (label.replace("|", ", "), ranks, s["n"], s["t"],
                     "---" if refval is None else "x%.2f" % (refval / s["t"]),
                     fmt_spread(s, pct="%"),
                     "---" if s["pipeline"] is None else "%.1f" % s["pipeline"],
                     "---" if s["pipeline"] is None
                     else "%.1f%%" % (100 * s["t"] / s["pipeline"]),
                     "" if s["config"] is None else
                     "fmm=%s (%d swept)" % (",".join(s["config"]),
                                            s.get("n_configs", 1))))
        print()
    print("wrote %s\n      %s\n      %s\n      %s"
          % (os.path.join(data_dir, args.tag + "_m*.dat"), reftex, fig, tab))


METHODS_TITLE = {v[0]: v[1] for v in METHODS.values()}
METHODS_COLOUR = {v[0]: v[2] for v in METHODS.values()}
METHODS_HEAD = {v[0]: v[3] for v in METHODS.values()}


if __name__ == "__main__":
    sys.exit(main())

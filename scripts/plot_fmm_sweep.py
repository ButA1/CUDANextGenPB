#!/usr/bin/env python3
"""
Turn a bench_sweep.py FMM sweep into a pgfplots accuracy-vs-work figure.

The sweep varies three knobs (fmm_mac, fmm_multipole_order, fmm_leaf_size), which
is one dimension too many to put on axes.  So the axes are the *outcome* --
energy-calculation time against relative error, both log -- and the knobs become visual
encoding:

    panel  = fmm_leaf_size          (2x2 group plot)
    colour = fmm_multipole_order    (one line per p)
    points along each line = fmm_mac, 0.2 -> 0.8

Every configuration in the sweep appears exactly once.  Each panel also carries
the Pareto front computed over the *whole* sweep (grey), so it is visible at a
glance which leaf size owns the frontier, plus two reference marks: a vertical
line at the naive (energy_method=1) energy time, taken from the same CSV, and a
shaded band up to the error stock NextGenPB itself reaches on the analytical
Kirkwood sphere, which comes from test3 and has to be passed in.

Writes, relative to the thesis directory:
    figures/data/<tag>.dat          one row per configuration
    figures/data/<tag>_pareto.dat   the front, ordered by time
    figures/data/<tag>_ref.tex      reference values as macros
    figures/<tag>.tex               the figure, \\input-able like the others
    figures/<tag>_table.tex         best configuration per error target

Errors are relative to the baseline of the same sweep, so what is plotted is FMM
truncation error and not solver error.  On a replay CSV that baseline is the
naive GPU sum over the same dumped inputs -- there is no CPU path in a replay,
so do not describe it as one.
"""

import argparse
import math
import os
import re
import statistics
import sys

import fmm_csv
from fmm_csv import ERR_FLOOR, METRICS

# A cool-to-warm ramp covering p = 6..12 (FMM_MAX_P is 12). Ordered by lightness
# so it survives greyscale printing, and each step stays distinguishable under the
# common CVD types.
#
# Keyed on p rather than on position in the drawn set, so p = 9 is the same colour
# in every figure regardless of which orders a particular sweep happened to cover.
P_RAMP = ["1B3A6B", "1F77B4", "17A398", "7EA83D", "E8A33D", "D95F02", "C1272D"]
P_MARK_RAMP = ["*", "square*", "triangle*", "diamond*", "pentagon*",
               "oplus*", "otimes*"]


def p_style(p):
    """(colour name, hex, mark) for multipole order p."""
    i = (p - 6) % len(P_RAMP)
    return "fmmp{}".format(p), P_RAMP[i], P_MARK_RAMP[i]


def group_size(n):
    """Panel grid for n leaf pairs -- pgfplots "C by R", C columns per row."""
    if n <= 1:
        return 1, 1
    if n == 2:
        return 2, 1
    if n <= 4:
        return 2, 2
    if n <= 6:
        return 3, 2
    return 3, (n + 2) // 3


def pareto(points, metric):
    """Non-dominated set under (minimise time, minimise error)."""
    front = []
    for a in points:
        dominated = any(
            b is not a
            and b["time"] <= a["time"] and b["err"][metric] <= a["err"][metric]
            and (b["time"] < a["time"] or b["err"][metric] < a["err"][metric])
            for b in points
        )
        if not dominated:
            front.append(a)
    return sorted(front, key=lambda d: d["time"])


def write_dat(path, points, front_ids):
    # pairid is what the pgfplots filter tests. Filtering on sleaf AND tleaf AND p
    # would need a three-column filter; collapsing the leaf pair to one integer
    # keeps the existing two-column style working.
    with open(path, "w") as fh:
        fh.write("mac p pairid sleaf tleaf time err_pol err_ionic err_sum pareto\n")
        for d in points:
            fh.write("{:.1f} {} {} {} {} {:.6g} {:.6g} {:.6g} {:.6g} {}\n".format(
                d["mac"], d["p"], d["pairid"], d["sleaf"], d["tleaf"], d["time"],
                d["err"]["pol"], d["err"]["ionic"], d["err"]["sum"],
                1 if id(d) in front_ids else 0))


def write_pareto_dat(path, front, metric):
    with open(path, "w") as fh:
        fh.write("time err mac p sleaf tleaf\n")
        for d in front:
            fh.write("{:.6g} {:.6g} {:.1f} {} {} {}\n".format(
                d["time"], d["err"][metric], d["mac"], d["p"],
                d["sleaf"], d["tleaf"]))


FMM_MACROS = ["fmmMolecule", "fmmNatoms", "fmmNaiveTime", "fmmNaiveErr",
              "fmmCpuTime", "fmmMetric", "fmmRepeats", "fmmSubsetNote",
              "fmmMachine", "fmmKirkErr", "fmmKirkClause", "fmmPanelClause"]


def macro_suffix(tag):
    """Letters of the tag -- a control sequence may not contain digits or _."""
    return "".join(c for c in tag if c.isalpha())


def localise(text, tag):
    r"""Make this figure's macros and labels unique to it.

    Several of these figures now coexist in one document. Without this every one
    of them defines \fmmMolecule and \label{fig:fmm-sweep}, so the labels clash
    and the captions all read whichever \def happened to run last.
    """
    sfx = macro_suffix(tag)
    # (?![A-Za-z]) is load-bearing: a LaTeX control sequence ends at the
    # first non-letter, so a plain replace of \\leafMac also rewrites the
    # \\leafMachine that starts with it -- which it did, silently producing
    # \\leafMacleafsweepvybhine and an undefined macro in the caption.
    for name in FMM_MACROS:
        text = re.sub(r"\\" + name + r"(?![A-Za-z])",
                      "\\\\" + name + sfx, text)
    return text.replace("TAG", tag)


def sci(x):
    r"""1.246641e-10 -> $1.25\times10^{-10}$."""
    mant, exp = "{:.2e}".format(x).split("e")
    return "$%s\\times10^{%d}$" % (mant, int(exp))


def kirkwood_clause(err):
    """The caption sentence for the Kirkwood band, or nothing.

    Descriptive only: what the line IS, not what it means for the choice of
    configuration. That reading belongs in the body text.
    """
    if err is None:
        return ""
    return (" The horizontal line is stock NextGenPB's own error on the "
            "Kirkwood sphere, %s." % sci(err))


def panel_clause(npanels):
    """The caption sentence naming what the panels are, if there are several."""
    if npanels <= 1:
        return ""
    return " Panels are the leaf size, shared by both trees."


def write_ref_tex(path, molecule, natoms, naive_t, naive_e, base_t, metric, nrep,
                  shown=None, n_total=None, tag="", machine_note="",
                  kirkwood_err=None, npanels=1):
    def num(x, default="0"):
        return default if x is None else "{:.6g}".format(x)

    # When the figure draws a subset, the caption has to say so, and say that the
    # front behind it still covers the whole sweep.
    if shown is None:
        note = ""
    else:
        ps = sorted({d["p"] for d in shown})
        macs = sorted({d["mac"] for d in shown})
        note = (" For legibility the figure draws %d of the %d configurations: "
                "$p \\in \\{%s\\}$ and $\\theta \\in \\{%s\\}$. The Pareto front "
                "is still taken over all %d, so it can run through gaps between "
                "the drawn lines."
                % (len(shown), n_total, ",\\,".join(str(p) for p in ps),
                   ",\\,".join("{:g}".format(m) for m in macs), n_total))

    with open(path, "w") as fh:
        fh.write("% generated by scripts/plot_fmm_sweep.py -- do not edit\n")
        fh.write(localise("\\def\\fmmMolecule{%s}\n" % molecule, tag))
        fh.write(localise("\\def\\fmmNatoms{%s}\n" % natoms, tag))
        fh.write(localise("\\def\\fmmNaiveTime{%s}\n" % num(naive_t), tag))
        fh.write(localise("\\def\\fmmNaiveErr{%s}\n" % num(naive_e, str(ERR_FLOOR)), tag))
        fh.write(localise("\\def\\fmmCpuTime{%s}\n" % num(base_t), tag))
        fh.write(localise("\\def\\fmmMetric{%s}\n" % METRICS[metric][1], tag))
        fh.write(localise("\\def\\fmmRepeats{%s}\n" % nrep, tag))
        fh.write(localise("\\def\\fmmSubsetNote{%s}\n" % note, tag))
        fh.write(localise("\\def\\fmmMachine{%s}\n" % machine_note, tag))
        # Not measurable from any sweep CSV: it is the error against the
        # analytical Kirkwood sphere, a different test system entirely. It has
        # to be handed in with --kirkwood-err; without it the band is not drawn.
        fh.write(localise("\\def\\fmmKirkErr{%s}\n"
                          % num(kirkwood_err, str(ERR_FLOOR)), tag))
        fh.write(localise("\\def\\fmmKirkClause{%s}\n"
                          % kirkwood_clause(kirkwood_err), tag))
        fh.write(localise("\\def\\fmmPanelClause{%s}\n"
                          % panel_clause(npanels), tag))


FIGURE_TEMPLATE = r"""% ------------------------------------------------------------------------
%  FMM parameter sweep: accuracy against work.
%  Generated by scripts/plot_fmm_sweep.py -- regenerate rather than edit.
%  Include with \input{figures/TAG}
%  Needs \usepackage{pgfplots} in the preamble.
% ------------------------------------------------------------------------
\input{figures/data/TAG_ref}

\pgfplotsset{compat=1.16}
\usepgfplotslibrary{groupplots}

% Keep only the rows where column #1 equals #2 AND column #3 equals #4. Lets
% every series come from the one data table instead of twenty little files.
%
% It has to test both columns in a single style: two separate one-column filters
% on the same \addplot would both define x filter/.code, and the second silently
% replaces the first.
\pgfplotsset{
  discard if not two/.style n args={4}{
    x filter/.code={%
      \edef\tempa{\thisrow{#1}}\edef\tempb{#2}%
      \edef\tempc{\thisrow{#3}}\edef\tempd{#4}%
      \ifx\tempa\tempb
        \ifx\tempc\tempd\else\def\pgfmathresult{inf}\fi
      \else\def\pgfmathresult{inf}\fi
    }
  }
}

COLORDEFS
\definecolor{fmmfront}{HTML}{9AA0A6}

\begin{figure}[!htb]
\centering
\begin{tikzpicture}
\begin{groupplot}[
    group style={
      group size=GROUPSIZE,
      horizontal sep=0.9cm,
      vertical sep=1.0cm,
      xlabels at=edge bottom,
      ylabels at=edge left,
      xticklabels at=edge bottom,
      yticklabels at=edge left,
    },
    width=PLOTW, height=PLOTH,
    xmode=log, ymode=log,
    xlabel={energy calculation [s]},
    ylabel={relative error},
    xlabel style={font=\footnotesize},
    ylabel style={font=\footnotesize},
    tick label style={font=\scriptsize},
    title style={font=\footnotesize, yshift=-0.4ex},
    grid=both,
    major grid style={black!12},
    minor grid style={black!5},
    minor tick num=0,
    ymin=YMIN, ymax=YMAX,
    xmin=XMIN, xmax=XMAX,
    log basis x=10, log basis y=10,
    xtick={XTICKS}, xticklabels={XTICKLABELS},
    ytick={YTICKS},
    legend columns=LEGENDCOLS,
    legend style={draw=none, font=\footnotesize, column sep=0.6em},
]
PANELS
\end{groupplot}
\end{tikzpicture}

\vspace{0.4ex}
\ref{LEGENDNAME}

\caption[FMM parameter sweep: accuracy against work]{\textbf{FMM accuracy against
    work on \fmmMolecule{} (\fmmNatoms{} atoms).} Median energy-calculation time
    over \fmmRepeats{} repeats against relative error in the
    \fmmMetric{}.\fmmPanelClause\fmmKirkClause\fmmMachine\fmmSubsetNote}
\label{fig:TAG}
\end{figure}
"""

PANEL_TEMPLATE = r"""
\nextgroupplot[title={PANELTITLE}LEGENDOPT]
KIRKBAND  % naive O(N^2) energy time
  \addplot[black!55, dashed, line width=0.8pt, forget plot]
    coordinates {(\fmmNaiveTime,YMIN) (\fmmNaiveTime,YMAX)};
  % Pareto front over the whole sweep
  \addplot[fmmfront, line width=1.6pt, forget plot]
    table[x=time, y=err] {figures/data/TAG_pareto.dat};
SERIESLEGEND"""

# How accurate stock NextGenPB is against the analytical Kirkwood sphere -- the
# accuracy of the method being replaced, and so the level an approximation has to
# reach to be free. It comes from a different test system (test3) and cannot be
# derived from any sweep CSV, so it is passed in with --kirkwood-err.
KIRKWOOD_BAND = r"""  % stock NextGenPB error on the analytical Kirkwood sphere
  \addplot[draw=none, fill=black!7, forget plot]
    coordinates {(XMIN,YMIN) (XMAX,YMIN) (XMAX,\fmmKirkErr) (XMIN,\fmmKirkErr)}
    \closedcycle;
  \addplot[black!45, densely dotted, line width=1.0pt, forget plot]
    coordinates {(XMIN,\fmmKirkErr) (XMAX,\fmmKirkErr)};
"""

# Only the first panel feeds the shared legend; the other three repeat the same
# five series and must not add duplicate entries.
# PORDER, not P: the placeholders are substituted by plain string replace, so a
# one-letter one would also rewrite any literal P the other substitutions put in
# -- which it did, turning the legend entry "$P=6$" into "$6=6$".
SERIES_TEMPLATE = r"""  \addplot[color=COLOR, mark=MARK, mark size=MARKSIZE, line width=SERIESLW FORGET]
    table[x=time, y=ERRCOL, discard if not two={pairid}{PAIRID}{p}{PORDER}]
      {figures/data/TAG.dat};
ENTRY"""

LEGEND_EXTRA = r"""  \addlegendimage{fmmfront, line width=1.6pt}
  \addlegendentry{Pareto front}
  \addlegendimage{black!55, dashed, line width=0.8pt}
  \addlegendentry{naive $O(N^2)$}
"""

LEGEND_KIRKWOOD = r"""  \addlegendimage{black!45, densely dotted, line width=1.0pt}
  \addlegendentry{stock NGPB error}
"""


def build_figure(tag, points, metric, pairs, naive_t, naive_e, out_path,
                 kirkwood_err=None):
    errcol = "err_" + metric
    times = [d["time"] for d in points]
    errs = [d["err"][metric] for d in points]

    # Data-driven limits with a multiplicative margin. Rounding to decades wastes
    # most of the axis here: the times span well under one decade.
    lo_t = min(times + ([naive_t] if naive_t else []))
    hi_t = max(times + ([naive_t] if naive_t else []))
    xmin, xmax = lo_t / 1.35, hi_t * 1.35
    extra_e = [e for e in (naive_e, kirkwood_err) if e]
    ymin = min(errs + extra_e) / 3.0
    ymax = max(errs + extra_e) * 4.0

    # A single-panel figure has the whole text width to itself, so the 7.2 cm
    # that four panels have to share leaves it a postage stamp. Everything that
    # scales with it -- marks, line widths -- grows with it, or the plot just
    # gets emptier rather than more readable.
    single = len(pairs) == 1
    plotw, ploth = ("14.2cm", "8.8cm") if single else ("7.2cm", "5.4cm")
    marksize, serieslw = ("1.7pt", "1.1pt") if single else ("1.15pt", "0.9pt")

    # Only the orders actually being drawn get a series, a colour and a legend
    # entry -- the sweep may hold more than the figure shows.
    p_shown = sorted({d["p"] for d in points})

    colordefs = "\n".join(
        "\\definecolor{%s}{HTML}{%s}" % (name, hexv)
        for name, hexv, _ in (p_style(p) for p in p_shown)
    )
    legend_name = "fmmlegend" + tag.replace("_", "")

    xt = fmm_csv.x_ticks(xmin, xmax)
    yt = fmm_csv.y_ticks(ymin, ymax)
    xticks = ",".join("{:.6g}".format(v) for v in xt)
    xticklabels = ",".join(fmm_csv.fmt_tick(v) for v in xt)
    yticks = ",".join("{:.6g}".format(v) for v in yt)

    panels = []
    for idx, (pairid, sleaf, tleaf) in enumerate(pairs):
        first = idx == 0
        series = "".join(
            SERIES_TEMPLATE
            .replace("COLOR", p_style(p)[0])
            .replace("MARKSIZE", marksize)
            .replace("SERIESLW", serieslw)
            .replace("MARK", p_style(p)[2])
            .replace("ERRCOL", errcol)
            .replace("FORGET", "" if first else ", forget plot")
            .replace("ENTRY", "  \\addlegendentry{$P=%d$}\n" % p if first else "")
            .replace("PAIRID", str(pairid))
            .replace("PORDER", str(p))
            .replace("TAG", tag)
            for p in p_shown
        )
        # Before the split one leaf size drove both trees, so naming both here
        # would invent a distinction the run did not have.
        title = ("$n_{leaf} = %d$" % sleaf if sleaf == tleaf
                 else "$n_{leaf,src}/n_{leaf,tgt} = %d/%d$" % (sleaf, tleaf))
        legend = (LEGEND_EXTRA + (LEGEND_KIRKWOOD if kirkwood_err else "")
                  if first else "")
        panel = (PANEL_TEMPLATE
                 .replace("SERIES", series)
                 .replace("LEGENDOPT",
                          ", legend to name=" + legend_name if first else "")
                 .replace("LEGEND", legend)
                 .replace("KIRKBAND", KIRKWOOD_BAND if kirkwood_err else "")
                 .replace("PANELTITLE", title)
                 .replace("TAG", tag)
                 .replace("XMIN", "{:.6g}".format(xmin))
                 .replace("XMAX", "{:.6g}".format(xmax))
                 .replace("YMIN", "{:.6g}".format(ymin))
                 .replace("YMAX", "{:.6g}".format(ymax)))
        panels.append(panel)

    cols, rows = group_size(len(pairs))
    body = (FIGURE_TEMPLATE
            .replace("GROUPSIZE", "{} by {}".format(cols, rows))
            # Five is the most that fits the text block. A pgfplots legend sizes
            # each column to its widest cell over ALL rows, so the three wide
            # reference entries stretch three columns no matter which row they
            # land on -- six columns overflows by 38pt and seven by 105pt, which
            # is what clipped "naive O(N^2)" off the right margin.
            .replace("LEGENDCOLS", str(min(max(len(p_shown), 3), 5)))
            .replace("PLOTW", plotw)
            .replace("PLOTH", ploth)
            .replace("COLORDEFS", colordefs)
            .replace("PANELS", "\n".join(panels))
            .replace("LEGENDNAME", legend_name)
            .replace("XTICKLABELS", xticklabels)
            .replace("XTICKS", xticks)
            .replace("YTICKS", yticks)
            .replace("XMIN", "{:.6g}".format(xmin))
            .replace("XMAX", "{:.6g}".format(xmax))
            .replace("YMIN", "{:.6g}".format(ymin))
            .replace("YMAX", "{:.6g}".format(ymax))
            .replace("TAG", tag))

    with open(out_path, "w") as fh:
        fh.write(localise(body, tag))


def write_table(path, points, metric, naive_t, naive_e, kirkwood_err, top=3,
                tag=""):
    """The `top` fastest configurations that satisfy the accuracy criterion.

    Drawn from every configuration, NOT from the Pareto front: the second and
    third fastest qualifier can each be dominated by the first (slower and less
    accurate) and so need not be on the front at all. Restricting to the front
    would silently drop exactly the near-misses this table exists to show.
    """
    ok = ([d for d in points if d["err"][metric] <= kirkwood_err]
          if kirkwood_err else list(points))
    rows = sorted(ok, key=lambda d: d["time"])[:top]

    if kirkwood_err:
        criterion = ("A configuration qualifies when its relative error is at "
                     "or below stock NextGenPB's own on the Kirkwood sphere, %s"
                     % sci(kirkwood_err))
        qualified = " %d of the %d configurations in the sweep qualify." % (
            len(ok), len(points))
    else:
        criterion = ("No accuracy criterion was applied, so these are simply "
                     "the fastest configurations in the sweep")
        qualified = ""

    buf = ["% generated by scripts/plot_fmm_sweep.py -- do not edit\n",
           "\\begin{table}[!htb]\n\\centering\n",
           "\\caption[Fastest FMM configurations meeting the accuracy criterion]"
           "{\\textbf{The %d fastest FMM configurations on \\fmmMolecule{} that "
           "meet the accuracy criterion.} %s. Error in the \\fmmMetric{} and "
           "speed-up are both against the naive path "
           "(\\fmmNaiveTime\\,s).%s From figure~\\ref{fig:TAG}."
           "\\fmmMachine}\n" % (len(rows), criterion, qualified),
           "\\label{tab:TAG-pareto}\n",
           "\\begin{tabular}{rrrrrrr}\n\\toprule\n",
           "$\\theta$ & $P$ & $n_{leaf,src}$ & $n_{leaf,tgt}$ & "
           "time [s] & rel. error & speed-up \\\\\n\\midrule\n"]

    for d in rows:
        buf.append("%.1f & %d & %d & %d & %.4f & %s & $%.2f\\times$ \\\\\n" % (
            d["mac"], d["p"], d["sleaf"], d["tleaf"], d["time"],
            sci(d["err"][metric]),
            naive_t / d["time"] if naive_t else float("nan")))

    buf.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    with open(path, "w") as fh:
        fh.write(localise("".join(buf), tag))

    # main() prints these back to the console. Lost when this function was
    # rewritten to buffer-then-localise, which crashed AFTER every file was
    # already written -- so the outputs were correct and the exit status was not.
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="test folder holding bench_runs.csv")
    ap.add_argument("--csv", default=None, help="override the CSV path")
    ap.add_argument("--thesis", default="thesis",
                    help="thesis directory to write into (default: thesis)")
    ap.add_argument("--tag", default=None,
                    help="basename for the generated files (default: fmm_sweep_<folder>)")
    fmm_csv.add_machine_argument(ap)
    ap.add_argument("--stat", default="median",
                    choices=sorted(fmm_csv.TIME_STATS),
                    help="how to collapse timing repeats; use min for shared "
                         "cluster nodes, where contention only slows runs down")
    ap.add_argument("--metric", default="pol", choices=sorted(METRICS),
                    help="which energy the error is measured on (default: pol)")
    ap.add_argument("--top", type=int, default=3,
                    help="how many of the fastest configurations meeting the "
                         "accuracy criterion the table lists (default: 3). The "
                         "criterion is --kirkwood-err; without it the table "
                         "falls back to the fastest overall and says so.")
    ap.add_argument("--molecule", default=None,
                    help="molecule name for the caption (the fmm_replay CSV has "
                         "no molecule column, so it must be given here)")
    ap.add_argument("--natoms", default=None,
                    help="atom count for the caption, as above")
    ap.add_argument("--p-values", default=None,
                    help="only draw these multipole orders, e.g. 6,8,10. "
                         "Display only -- the sweep data is not touched, and the "
                         "Pareto front is still taken over every configuration.")
    ap.add_argument("--mac-values", default=None,
                    help="only draw these mac values, e.g. 0.2,0.4,0.6,0.8. "
                         "Display only, as above.")
    ap.add_argument("--kirkwood-err", type=float, default=None,
                    help="relative error stock NextGenPB reaches in this energy "
                         "on the analytical Kirkwood sphere, drawn as a "
                         "horizontal band. It comes from a different test system "
                         "(test3) and cannot be derived from a sweep CSV, so it "
                         "has to be given here; without it no band is drawn.")
    ap.add_argument("--stock-csv", default=None,
                    help="full-pipeline bench_sweep.py CSV of the same molecule. "
                         "Diagnostic only, NOT drawn: prints how well the "
                         "pipeline reproduces its own energy (CPU vs naive GPU "
                         "sum, plus the run-to-run spread), which bounds how "
                         "finely two configurations can be told apart at all.")
    args = ap.parse_args()

    csv_path = args.csv or os.path.join(args.folder, "bench_runs.csv")
    if not os.path.exists(csv_path):
        sys.exit("no such CSV: {}".format(csv_path))

    tag = args.tag or "fmm_sweep_" + os.path.basename(os.path.normpath(args.folder))
    metric_col = METRICS[args.metric][0]

    rows = fmm_csv.load(csv_path)
    points, info = fmm_csv.aggregate(rows, metric_col, stat=args.stat)
    if not points:
        sys.exit("no energy_method=2 rows -- nothing to plot")

    naive_t, naive_e = info["naive_t"], info["naive_e"]
    base_t = info["base_t"]

    # One integer per distinct (source, target) leaf pair; the pgfplots filter
    # tests it instead of the two columns separately.
    pair_ids = {pr: i for i, pr in
                enumerate(sorted({(d["sleaf"], d["tleaf"]) for d in points}))}
    for d in points:
        d["pairid"] = pair_ids[(d["sleaf"], d["tleaf"])]

    molecule = (args.molecule
                or os.path.splitext(info["molecule"])[0] or "?")
    natoms = args.natoms or info["natoms"] or "?"
    nrep = statistics.median(d["n"] for d in points)

    # The front is deliberately taken over EVERY configuration, before any
    # display thinning: it is a claim about the sweep, not about the subset that
    # happens to be drawn. So a thinned figure can show the front passing through
    # gaps between its own lines, which is correct and is stated in the caption.
    front = pareto(points, args.metric)
    front_ids = {id(d) for d in front}

    shown = points
    if args.p_values:
        keep = {int(v) for v in args.p_values.split(",")}
        shown = [d for d in shown if d["p"] in keep]
    if args.mac_values:
        keep = {round(float(v), 3) for v in args.mac_values.split(",")}
        shown = [d for d in shown if round(d["mac"], 3) in keep]
    if not shown:
        sys.exit("--p-values/--mac-values selected nothing")
    thinned = len(shown) < len(points)

    pairs = sorted({(d["pairid"], d["sleaf"], d["tleaf"]) for d in shown})
    cols, rows = group_size(len(pairs))
    if cols * rows != len(pairs):
        print("note: {} leaf pairs laid out as {}x{} -- {} cell(s) will be blank"
              .format(len(pairs), cols, rows, cols * rows - len(pairs)),
              file=sys.stderr)

    data_dir = os.path.join(args.thesis, "figures", "data")
    os.makedirs(data_dir, exist_ok=True)

    dat = os.path.join(data_dir, tag + ".dat")
    par = os.path.join(data_dir, tag + "_pareto.dat")
    reftex = os.path.join(data_dir, tag + "_ref.tex")
    fig = os.path.join(args.thesis, "figures", tag + ".tex")
    tab = os.path.join(args.thesis, "figures", tag + "_table.tex")

    kirkwood_err = args.kirkwood_err
    stock = fmm_csv.stock_floor(args.stock_csv, csv_path, metric_col) \
        if args.stock_csv else None

    write_dat(dat, shown, front_ids)
    write_pareto_dat(par, front, args.metric)
    write_ref_tex(reftex, molecule, natoms, naive_t, naive_e, base_t,
                  args.metric, int(nrep), shown if thinned else None, len(points),
                  tag=tag,
                  machine_note=fmm_csv.machine_note(args.machine, info["ranks"]),
                  kirkwood_err=kirkwood_err, npanels=len(pairs))
    build_figure(tag, shown, args.metric, pairs, naive_t, naive_e, fig,
                 kirkwood_err=kirkwood_err)
    table_rows = write_table(tab, points, args.metric, naive_t, naive_e,
                             kirkwood_err, args.top, tag)

    print("configurations : {}  ({} repeats each)".format(len(points), int(nrep)))
    print("Pareto front   : {} points".format(len(front)))
    print("naive O(N^2)   : {:.4f} s, own error {:.2e}".format(naive_t or 0, naive_e or 0))
    print("CPU reference  : {:.4f} s".format(base_t))
    if kirkwood_err:
        print("Kirkwood band  : {:.4e}  (drawn)".format(kirkwood_err))
    else:
        print("Kirkwood band  : not drawn -- pass --kirkwood-err")
    if stock:
        # Not drawn. Reported because it bounds how finely two configurations
        # can be told apart at all: below it the pipeline does not reproduce
        # itself, so an accuracy ranking there is ranking noise.
        print("reproducibility: {:.2e}  (diagnostic, from {})".format(
            stock["floor"], args.stock_csv))
        print("    CPU vs naive, same run        {:.2e}  (n={}/{})".format(
            stock["cpu_vs_naive"], stock["n_cpu"], stock["n_naive"]))
        print("    repeat spread, CPU / naive    {:.2e} / {:.2e}".format(
            stock["spread_cpu"], stock["spread_naive"]))
        if stock["pipeline_vs_replay"] is not None:
            print("    pipeline naive vs replay      {:.2e}".format(
                stock["pipeline_vs_replay"]))
    print()
    print("wrote {}".format(dat))
    print("      {}".format(par))
    print("      {}".format(reftex))
    print("      {}".format(fig))
    print("      {}".format(tab))
    print()
    if kirkwood_err:
        n_ok = sum(1 for d in points if d["err"][args.metric] <= kirkwood_err)
        print("fastest {} of the {} configurations meeting relerr <= {:.3e}:"
              .format(len(table_rows), n_ok, kirkwood_err))
    else:
        print("fastest {} configurations (NO accuracy criterion applied):"
              .format(len(table_rows)))
    print("  theta   p    src/tgt      time   rel. error   speed-up vs naive")
    for d in table_rows:
        print("   {:>4.1f}  {:>2d}  {:>9s}  {:>8.4f}   {:>9.2e}   {:>6.2f}x".format(
            d["mac"], d["p"], d["pair"], d["time"], d["err"][args.metric],
            naive_t / d["time"] if naive_t else float("nan")))
    print()
    print("add \\usepackage{pgfplots} to thesis/main.tex, then "
          "\\input{figures/%s}" % tag)


if __name__ == "__main__":
    sys.exit(main())
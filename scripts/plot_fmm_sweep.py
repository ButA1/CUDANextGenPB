#!/usr/bin/env python3
"""
Turn a bench_sweep.py FMM sweep into a pgfplots accuracy-vs-work figure.

The sweep varies three knobs (fmm_mac, fmm_multipole_order, fmm_leaf_size), which
is one dimension too many to put on axes.  So the axes are the *outcome* --
energy-stage time against relative error, both log -- and the knobs become visual
encoding:

    panel  = fmm_leaf_size          (2x2 group plot)
    colour = fmm_multipole_order    (one line per p)
    points along each line = fmm_mac, 0.2 -> 0.8

Every configuration in the sweep appears exactly once.  Each panel also carries
the Pareto front computed over the *whole* sweep (grey), so it is visible at a
glance which leaf size owns the frontier, plus two reference marks taken from the
same CSV: a vertical line at the naive (energy_method=1) energy time, and a shaded
band below the naive path's own deviation from the CPU reference -- the floor
under which an accuracy difference is not resolvable.

Writes, relative to the thesis directory:
    figures/data/<tag>.dat          one row per configuration
    figures/data/<tag>_pareto.dat   the front, ordered by time
    figures/data/<tag>_ref.tex      reference values as macros
    figures/<tag>.tex               the figure, \\input-able like the others
    figures/<tag>_table.tex         best configuration per error target

Errors are relative to the energy_method=0 run *of the same sweep* (same linear
solver), so what is plotted is FMM truncation error and not solver error.
"""

import argparse
import math
import os
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
              "fmmCpuTime", "fmmMetric", "fmmRepeats", "fmmSubsetNote"]


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
    for name in FMM_MACROS:
        text = text.replace("\\" + name, "\\" + name + sfx)
    return text.replace("TAG", tag)


def write_ref_tex(path, molecule, natoms, naive_t, naive_e, base_t, metric, nrep,
                  shown=None, n_total=None, tag=""):
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
    width=7.2cm, height=5.4cm,
    xmode=log, ymode=log,
    xlabel={energy stage [s]},
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
    legend columns=7,
    legend style={draw=none, font=\footnotesize, column sep=0.6em},
]
PANELS
\end{groupplot}
\end{tikzpicture}

\vspace{0.4ex}
\ref{LEGENDNAME}

\caption[FMM parameter sweep: accuracy against work]{\textbf{FMM accuracy against
    work on \fmmMolecule{} (\fmmNatoms{} atoms).} Each point is one
    $(\theta,\,p,\,n_{\mathrm{leaf}})$ configuration; the horizontal axis is the
    median energy-stage time over \fmmRepeats{} repeats and the vertical axis the
    relative error in the \fmmMetric{} against the CPU path
    (\texttt{energy\_method\,=\,0}) of the same sweep, so solver error cancels.
    Panels are the leaf size, colour the multipole order $p$, and the points
    along each line the multipole acceptance criterion $\theta$, increasing from
    $0.2$ at the accurate end to $0.8$. The grey line is the Pareto front over
    the whole sweep, repeated in every panel. The dashed vertical line is the
    naive $O(N^2)$ GPU path: only configurations to its left are worth using.
    The shaded band is that path's own deviation from the CPU reference --
    differences below it are not resolvable.\fmmSubsetNote}
\label{fig:TAG}
\end{figure}
"""

PANEL_TEMPLATE = r"""
\nextgroupplot[title={$n_{\mathrm{src}}/n_{\mathrm{tgt}} = PAIRLABEL$}LEGENDOPT]
  % resolution floor of the comparison
  \addplot[draw=none, fill=black!7, forget plot]
    coordinates {(XMIN,YMIN) (XMAX,YMIN) (XMAX,\fmmNaiveErr) (XMIN,\fmmNaiveErr)}
    \closedcycle;
  % naive O(N^2) energy time
  \addplot[black!55, dashed, line width=0.8pt, forget plot]
    coordinates {(\fmmNaiveTime,YMIN) (\fmmNaiveTime,YMAX)};
  % Pareto front over the whole sweep
  \addplot[fmmfront, line width=1.6pt, forget plot]
    table[x=time, y=err] {figures/data/TAG_pareto.dat};
SERIESLEGEND"""

# Only the first panel feeds the shared legend; the other three repeat the same
# five series and must not add duplicate entries.
SERIES_TEMPLATE = r"""  \addplot[color=COLOR, mark=MARK, mark size=1.15pt, line width=0.9pt FORGET]
    table[x=time, y=ERRCOL, discard if not two={pairid}{PAIRID}{p}{P}]
      {figures/data/TAG.dat};
ENTRY"""

LEGEND_EXTRA = r"""  \addlegendimage{fmmfront, line width=1.6pt}
  \addlegendentry{Pareto front}
  \addlegendimage{black!55, dashed, line width=0.8pt}
  \addlegendentry{naive $O(N^2)$}
"""


def build_figure(tag, points, metric, pairs, naive_t, naive_e, out_path):
    errcol = "err_" + metric
    times = [d["time"] for d in points]
    errs = [d["err"][metric] for d in points]

    # Data-driven limits with a multiplicative margin. Rounding to decades wastes
    # most of the axis here: the times span well under one decade.
    lo_t = min(times + ([naive_t] if naive_t else []))
    hi_t = max(times + ([naive_t] if naive_t else []))
    xmin, xmax = lo_t / 1.35, hi_t * 1.35
    lo_e = min(errs + ([naive_e] if naive_e else []))
    ymin, ymax = lo_e / 3.0, max(errs) * 4.0

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
            .replace("MARK", p_style(p)[2])
            .replace("ERRCOL", errcol)
            .replace("FORGET", "" if first else ", forget plot")
            .replace("ENTRY", "  \\addlegendentry{$p=%d$}\n" % p if first else "")
            .replace("PAIRID", str(pairid))
            .replace("P", str(p))
            .replace("TAG", tag)
            for p in p_shown
        )
        panel = (PANEL_TEMPLATE
                 .replace("SERIES", series)
                 .replace("LEGENDOPT",
                          ", legend to name=" + legend_name if first else "")
                 .replace("LEGEND", LEGEND_EXTRA if first else "")
                 .replace("PAIRLABEL", "%d/%d" % (sleaf, tleaf))
                 .replace("TAG", tag)
                 .replace("XMIN", "{:.6g}".format(xmin))
                 .replace("XMAX", "{:.6g}".format(xmax))
                 .replace("YMIN", "{:.6g}".format(ymin))
                 .replace("YMAX", "{:.6g}".format(ymax)))
        panels.append(panel)

    cols, rows = group_size(len(pairs))
    body = (FIGURE_TEMPLATE
            .replace("GROUPSIZE", "{} by {}".format(cols, rows))
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


def write_table(path, front, metric, naive_t, naive_e, targets, tag=""):
    """Cheapest configuration that reaches each error target."""
    rows = []
    for target in targets:
        ok = [d for d in front if d["err"][metric] <= target]
        if not ok:
            continue
        best = min(ok, key=lambda d: d["time"])
        speedup = naive_t / best["time"] if naive_t else float("nan")
        rows.append((target, best, speedup))

    buf = ["% generated by scripts/plot_fmm_sweep.py -- do not edit\n",
           "\\begin{table}[!htb]\n\\centering\n",
           "\\caption[Cheapest FMM configuration per error target]"
           "{\\textbf{Cheapest FMM configuration reaching each error "
           "target on \\fmmMolecule{}.} Taken from the Pareto front of "
           "figure~\\ref{fig:TAG}; the speed-up is against the "
           "naive $O(N^2)$ GPU path (\\fmmNaiveTime\\,s).}\n",
           "\\label{tab:TAG-pareto}\n",
           "\\begin{tabular}{lrrrrrr}\n\\toprule\n",
           "error target & $\\theta$ & $p$ & $n_{\\mathrm{src}}$ & "
           "$n_{\\mathrm{tgt}}$ & time [s] & speed-up \\\\\n\\midrule\n"]

    for target, best, speedup in rows:
        buf.append("$10^{%d}$ & %.1f & %d & %d & %d & %.4f & $%.2f\\times$ \\\\\n" % (
            round(math.log10(target)), best["mac"], best["p"],
            best["sleaf"], best["tleaf"], best["time"], speedup))

    buf.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    with open(path, "w") as fh:
        fh.write(localise("".join(buf), tag))

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="test folder holding bench_runs.csv")
    ap.add_argument("--csv", default=None, help="override the CSV path")
    ap.add_argument("--thesis", default="thesis",
                    help="thesis directory to write into (default: thesis)")
    ap.add_argument("--tag", default=None,
                    help="basename for the generated files (default: fmm_sweep_<folder>)")
    ap.add_argument("--metric", default="pol", choices=sorted(METRICS),
                    help="which energy the error is measured on (default: pol)")
    ap.add_argument("--targets", default="1e-10,1e-9,1e-8,1e-7,1e-6",
                    help="comma-separated error targets for the Pareto table")
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
    args = ap.parse_args()

    csv_path = args.csv or os.path.join(args.folder, "bench_runs.csv")
    if not os.path.exists(csv_path):
        sys.exit("no such CSV: {}".format(csv_path))

    tag = args.tag or "fmm_sweep_" + os.path.basename(os.path.normpath(args.folder))
    metric_col = METRICS[args.metric][0]

    rows = fmm_csv.load(csv_path)
    points, info = fmm_csv.aggregate(rows, metric_col)
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

    write_dat(dat, shown, front_ids)
    write_pareto_dat(par, front, args.metric)
    write_ref_tex(reftex, molecule, natoms, naive_t, naive_e, base_t,
                  args.metric, int(nrep), shown if thinned else None, len(points),
                  tag=tag)
    build_figure(tag, shown, args.metric, pairs, naive_t, naive_e, fig)
    targets = [float(t) for t in args.targets.split(",")]
    table_rows = write_table(tab, front, args.metric, naive_t, naive_e, targets, tag)

    print("configurations : {}  ({} repeats each)".format(len(points), int(nrep)))
    print("Pareto front   : {} points".format(len(front)))
    print("naive O(N^2)   : {:.4f} s, own error {:.2e}".format(naive_t or 0, naive_e or 0))
    print("CPU reference  : {:.4f} s".format(base_t))
    print()
    print("wrote {}".format(dat))
    print("      {}".format(par))
    print("      {}".format(reftex))
    print("      {}".format(fig))
    print("      {}".format(tab))
    print()
    print("cheapest configuration per error target:")
    print("  target      theta   p    src/tgt      time    speed-up vs naive")
    for target, best, speedup in table_rows:
        print("  {:<10.0e}  {:>4.1f}  {:>2d}  {:>9s}  {:>8.4f}   {:>6.2f}x".format(
            target, best["mac"], best["p"], best["pair"], best["time"], speedup))
    print()
    print("add \\usepackage{pgfplots} to thesis/main.tex, then "
          "\\input{figures/%s}" % tag)


if __name__ == "__main__":
    sys.exit(main())
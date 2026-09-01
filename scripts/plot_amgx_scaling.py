#!/usr/bin/env python3
r"""
How the AMGX solve scales across GPUs -- and where that stops working.

The x axis is a chain in which each adjacent pair changes exactly one thing:

    RTX 3080, 1 GPU, 4 ranks   ->  A100, 1 GPU, 4 ranks     only the GPU
    A100, 1 GPU, 4 ranks       ->  A100, 1 GPU, 16 ranks    only the rank count
    A100, 1 GPU, 16 ranks      ->  A100, 2 GPUs, 1 node     only the GPU count
    A100, 2 GPUs, 1 node       ->  A100, 2 GPUs, 2 nodes    only where GPU 2 sits

The middle step is the control that makes the first bar admissible at all: the
local runs are at 4 ranks and the topology runs at 16, and on one GPU the whole
matrix is gathered onto that GPU regardless of how many ranks fed it, so the
rank count should not reach AMGX. The bar pair says whether it does.

Two panels because the two AMGX configurations differ by ~2.5x in absolute time
and by an order of magnitude in iteration count, so a shared y axis would flatten
one of them. They are not redundant: the multigrid configuration is the one whose
scaling INVERTS across nodes, and the single-level one is the control that shows
the inversion is a property of the coarse-grid hierarchy, not of the network
alone.

Each bar is stacked setup + solve, both taken from AMGX's own timers
(`amgx_setup_s` / `amgx_solve_s`), not from the wall clock.

Rows are pooled over every `linear_solver=amgx` run in the CSV that shares an
`amgx_config`, across energy methods and repeats -- the energy method cannot
reach the solver, and pooling is what gives the median a sample size. That
premise is checked rather than assumed: --max-pool-spread aborts if the pooled
CONFIGURATIONS disagree in their medians by more than that fraction, which is
the thing that would invalidate the pool. It deliberately does not gate on
repeat-to-repeat spread, which is a property of the machine and not of the
pooling -- that number is reported per row in the table instead. (It matters:
the A100 4-rank series has a cold first repeat, setup 0.356 s against 0.152 and
0.174, which the median rejects and a max-minus-min gate would not survive.)

Writes, relative to the thesis directory:
    figures/data/<tag>_c<i>.dat   one row per series, one file per AMGX config
    figures/data/<tag>_ref.tex    caption values as macros
    figures/<tag>.tex             the figure
    figures/<tag>_table.tex       the same numbers plus the t_solve context

The table carries `t_solve` and the non-AMGX remainder on purpose. AMGX gaining
1.6x does not make the solve stage 1.6x faster, and the figure alone invites
exactly that reading.
"""

import argparse
import csv
import os
import re
import statistics
import sys

# Two segments of one quantity, so one hue at two lightnesses rather than two
# hues -- setup and solve are parts of the AMGX total, not competing series.
COLOUR_SETUP = "9DC3E0"
COLOUR_SOLVE = "2C5F8D"

AMGX_MACROS = ["amgxMolecule", "amgxRef", "amgxTunedName",
               "amgxTunedIters", "amgxDefaultIters", "amgxRepeats"]


def localise(text, tag):
    r"""Suffix this figure's macros and labels so several can coexist."""
    sfx = "".join(c for c in tag if c.isalpha())
    # (?![A-Za-z]) because a control sequence ends at the first non-letter.
    for name in AMGX_MACROS:
        text = re.sub(r"\\" + name + r"(?![A-Za-z])", "\\\\" + name + sfx, text)
    return text.replace("SFX", sfx).replace("TAG", tag)


def texify(s):
    """Escape the few characters that appear in these labels and break TeX."""
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def config_key(row):
    """Which AMGX configuration a row used. Empty column = AMGX's own default."""
    cfg = (row.get("amgx_config") or "").strip()
    return os.path.splitext(os.path.basename(cfg))[0] if cfg else ""


def load_series(spec):
    """"LABEL=path" -> (label, ranks, {config: stats}).

    LABEL may contain '|', which becomes a line break in the tick label.
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
                and (r.get("linear_solver") or "").strip() == "amgx"
                and (r.get("amgx_solve_s") or "").strip()]
    if not rows:
        sys.exit("no successful linear_solver=amgx rows with AMGX timers in %s\n"
                 "  (the LIS rows carry no amgx_setup_s/amgx_solve_s)" % path)

    ranks = {(r.get("np") or "").strip() for r in rows}
    ranks.discard("")
    if len(ranks) != 1:
        sys.exit("%s mixes rank counts %s -- one series is one rank count"
                 % (path, sorted(ranks)))

    by_cfg = {}
    for r in rows:
        by_cfg.setdefault(config_key(r), []).append(r)

    stats = {}
    for cfg, rs in by_cfg.items():
        rs.sort(key=lambda r: r.get("timestamp") or "")
        setup = [float(r["amgx_setup_s"]) for r in rs]
        solve = [float(r["amgx_solve_s"]) for r in rs]
        total = [a + b for a, b in zip(setup, solve)]
        iters = {(r.get("solver_iters") or "").strip() for r in rs}
        iters.discard("")
        tsolve = [float(r["t_solve"]) for r in rs if (r.get("t_solve") or "").strip()]

        # Per-config_id medians. Every config_id sharing an amgx_config differs
        # only in its energy method, which runs after the solve and cannot reach
        # it -- so these medians agreeing is the evidence that the pool is one
        # measurement, and it is what --max-pool-spread gates on.
        per_cfgid = {}
        for r in rs:
            per_cfgid.setdefault(r["config_id"], []).append(
                float(r["amgx_setup_s"]) + float(r["amgx_solve_s"]))
        med = [statistics.median(v) for v in per_cfgid.values()]

        # Cold start, tested rather than assumed. On some jobs the first AMGX
        # call of the whole run is far slower than every later one (clock ramp
        # and first-touch inside the library), which makes max-minus-min a
        # statement about that one call. The test is deliberately narrow: the
        # slowest run must BE the earliest one, and dropping it must bring the
        # rest into a normal band. Where it does not hold, nothing is excluded
        # and the full spread stands.
        def spread_of(v):
            return (max(v) - min(v)) / statistics.median(v)
        warm = total[1:]
        cold = (len(total) > 2 and total[0] == max(total)
                and spread_of(warm) < 0.15 <= spread_of(total))

        stats[cfg] = {
            "setup": statistics.median(setup),
            "solve": statistics.median(solve),
            "total": statistics.median(total),
            "spread": spread_of(total),
            "warm_spread": spread_of(warm) if cold else None,
            "pool_spread": ((max(med) - min(med)) / statistics.median(med)
                            if len(med) > 1 else 0.0),
            "n_cfgid": len(per_cfgid),
            "n": len(rs),
            # A single value means every pooled run took the same path through the
            # solver; several means the pool is not one measurement and the median
            # is meaningless. Reported, so the table can show it either way.
            "iters": sorted(iters, key=lambda s: float(s)) if iters else [],
            "tsolve": statistics.median(tsolve) if tsolve else None,
        }
    molecules = {(r.get("molecule") or "").strip() for r in rows}
    return (label, ranks.pop(), stats,
            os.path.splitext(sorted(molecules)[0])[0] if molecules else "?")


FIGURE_TEMPLATE = r"""% ------------------------------------------------------------------------
%  AMGX scaling across GPUs and topologies.
%  Generated by scripts/plot_amgx_scaling.py -- regenerate rather than edit.
%  Include with \input{figures/TAG}
%  Needs \usepackage{pgfplots} in the preamble.
% ------------------------------------------------------------------------
\input{figures/data/TAG_ref}

\pgfplotsset{compat=1.16}
\usepgfplotslibrary{groupplots}

\definecolor{amgxsetup}{HTML}{COLSETUP}
\definecolor{amgxsolve}{HTML}{COLSOLVE}

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
    ymin=0,
    ylabel={AMGX time [s]},
    ylabel style={font=\footnotesize},
    symbolic x coords={SYMCOORDS},
    xtick=data,
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

\vspace{0.4ex}
\ref{amgxlegendSFX}

\caption[AMGX scaling across GPUs and topologies]{\textbf{AMGX setup and solve
    time on \amgxMolecule{} as the GPU, the rank count and the GPU topology are
    changed one at a time.} Median over at least \amgxRepeats{} runs per bar,
    from AMGX's own timers rather than the wall clock; the figure above each bar
    is its total relative to \amgxRef{}. Adjacent pairs differ in one variable:
    GPU, then rank count, then number of GPUs, then whether the second GPU sits
    in the same node. \emph{Left:} the tuned configuration \amgxTunedName{},
    \amgxTunedIters{} iterations. \emph{Right:} AMGX's default,
    \amgxDefaultIters{} iterations, on its own y scale. The rank-count pair is
    the control that lets the 4-rank local bars stand beside the 16-rank cluster
    bars: on one GPU the matrix is gathered onto that GPU whatever the rank
    count. Table~\ref{tab:TAG} sets these times against the full
    \texttt{t\_solve} stage, of which AMGX is only a part.}
\label{fig:TAG}
\end{figure}
"""

# `ybar stacked` and `bar width` live here rather than in the shared groupplot
# options: the groupplot library rejects `bar width` at that level, because the
# bar handler that defines the key has not been installed yet when the shared
# list is parsed.
PANEL_TEMPLATE = r"""
\nextgroupplot[ybar stacked, bar width=13pt, title={TITLE}, ymax=YMAX, LEGEND]
  \addplot[fill=amgxsetup, draw=amgxsetup!70!black, line width=0.4pt]
    table[x=sx, y=setup] {figures/data/DATFILE.dat};
  ENTRYSETUP
  \addplot[fill=amgxsolve, draw=amgxsolve!70!black, line width=0.4pt]
    table[x=sx, y=solve] {figures/data/DATFILE.dat};
  ENTRYSOLVE
  % Explicit nodes rather than `nodes near coords`: inside `ybar stacked` the
  % latter anchors each label at its own segment's height, i.e. partway up the
  % bar, and `stack plots=false` on an overlay plot earns a bar shift instead.
  % The totals are known here, so the placement can just be stated.
RATIOLABELS"""

TABLE_TEMPLATE = r"""% ------------------------------------------------------------------------
%  Generated by scripts/plot_amgx_scaling.py -- regenerate rather than edit.
%  Include with \input{figures/TAG_table}
%  Needs \usepackage{booktabs}.
% ------------------------------------------------------------------------
\begin{table}[!htb]
\centering
\footnotesize
\caption[AMGX scaling against the whole solve stage]{\textbf{The numbers behind
    figure~\ref{fig:TAG}, set against the solve stage they sit inside.}
    \emph{AMGX} is setup${}+{}$solve from AMGX's own timers and is what the
    figure draws; \emph{rest} is \texttt{t\_solve} minus that, i.e. the
    assembly-to-solver handoff, which is not on the GPU and does not scale.
    RESTCLAIM \emph{spread} is the widest repeat-to-repeat variation among the
    runs pooled into that row; a bracketed second figure excludes the job's
    first AMGX call, a cold start that the median already rejects.}
\label{tab:TAG}
\begin{tabular}{lrrrrrrr}
\toprule
platform & iters & setup [s] & solve [s] & AMGX [s] & vs.\ ref. & spread & rest [s] \\
ROWS\bottomrule
\end{tabular}
\end{table}
"""


def fmt_iters(stat):
    if not stat["iters"]:
        return "---"
    if len(stat["iters"]) == 1:
        return stat["iters"][0]
    return "%s--%s" % (stat["iters"][0], stat["iters"][-1])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("series", nargs="+",
                    help="LABEL=path/to/bench_runs.csv, in the order they should "
                         "be drawn. '|' inside LABEL becomes a line break in the "
                         "tick label")
    ap.add_argument("--thesis", default="thesis")
    ap.add_argument("--tag", default="amgx_scaling")
    ap.add_argument("--reference", type=int, default=None,
                    help="0-based index of the series every bar is divided by. "
                         "Default is the LAST single-GPU cluster series, i.e. the "
                         "point the topology sweep actually starts from")
    ap.add_argument("--configs", default=None,
                    help="comma-separated amgx_config basenames to draw, left "
                         "panel first; '' means AMGX's built-in default. Default "
                         "is every config present in every series")
    ap.add_argument("--max-pool-spread", type=float, default=0.15,
                    help="abort if the CONFIGURATIONS pooled into one bar "
                         "disagree in their medians by more than this fraction "
                         "(default 0.15). Pooling across energy methods is only "
                         "valid because the energy method cannot reach the "
                         "solver; this is the check on that. It is not a gate on "
                         "repeat-to-repeat noise, which the table reports per row")
    ap.add_argument("--molecule", default=None)
    args = ap.parse_args()

    loaded = [load_series(s) for s in args.series]
    if len(loaded) < 2:
        sys.exit("need at least two series to compare")

    molecules = {mol for _, _, _, mol in loaded}
    if len(molecules) != 1 and not args.molecule:
        sys.exit("the series disagree on the molecule (%s); pass --molecule to "
                 "override if that is intended" % sorted(molecules))

    if args.configs is not None:
        configs = [c.strip() for c in args.configs.split(",")]
    else:
        common = set.intersection(*[set(st) for _, _, st, _ in loaded])
        if not common:
            sys.exit("no amgx_config is present in every series -- pass --configs")
        # Tuned first: named configs before AMGX's unnamed default.
        configs = sorted(common, key=lambda c: (c == "", c))
    if len(configs) != 2:
        sys.exit("this figure is two panels; got %d configs: %s"
                 % (len(configs), configs))

    for cfg in configs:
        for label, _, stats, _ in loaded:
            if cfg not in stats:
                sys.exit("series %r has no rows for amgx_config %r" % (label, cfg))
            if stats[cfg]["pool_spread"] > args.max_pool_spread:
                sys.exit("series %r, config %r: the %d pooled configurations "
                         "disagree by %.1f%% in their medians, over the %.1f%% "
                         "limit -- the energy method IS reaching the solver, or "
                         "these are not the same solve.\n"
                         "  Raise --max-pool-spread only after checking why."
                         % (label, cfg, stats[cfg]["n_cfgid"],
                            100 * stats[cfg]["pool_spread"],
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

    data_dir = os.path.join(args.thesis, "figures", "data")
    os.makedirs(data_dir, exist_ok=True)
    reftex = os.path.join(data_dir, args.tag + "_ref.tex")
    fig = os.path.join(args.thesis, "figures", args.tag + ".tex")
    tab = os.path.join(args.thesis, "figures", args.tag + "_table.tex")

    sym = ["s%d" % i for i in range(len(loaded))]
    ticklabels = ", ".join("{%s}" % texify(lab).replace("|", r"\\")
                           for lab, _, _, _ in loaded)

    panels = []
    for ci, cfg in enumerate(configs):
        datname = "%s_c%d" % (args.tag, ci)
        refval = loaded[ref][2][cfg]["total"]
        peak = 0.0
        labels = []
        with open(os.path.join(data_dir, datname + ".dat"), "w") as fh:
            fh.write("sx setup solve total\n")
            for i, (_, _, stats, _) in enumerate(loaded):
                s = stats[cfg]
                peak = max(peak, s["total"])
                fh.write("%s %.6g %.6g %.6g\n"
                         % (sym[i], s["setup"], s["solve"], s["total"]))
                labels.append(
                    "  \\node[font=\\tiny, text=black!55, anchor=south, "
                    "yshift=1pt] at (axis cs:%s,%.6g) {$\\times$%.2f};\n"
                    % (sym[i], s["total"], refval / s["total"]))
        title = (r"tuned: \texttt{%s}" % texify(cfg)) if cfg else "AMGX default"
        # One legend for both panels; the second panel must not draw its own.
        legend = ("legend style={draw=none, font=\\footnotesize, "
                  "column sep=0.8em}, legend columns=2, "
                  "legend to name=amgxlegendSFX" if ci == 0 else
                  "legend style={draw=none}")
        panels.append(PANEL_TEMPLATE
                      .replace("TITLE", title)
                      .replace("YMAX", "%.6g" % (peak * 1.16))
                      .replace("LEGEND", legend)
                      .replace("RATIOLABELS", "".join(labels))
                      .replace("DATFILE", datname)
                      .replace("ENTRYSETUP",
                               r"\addlegendentry{setup}" if ci == 0 else "")
                      .replace("ENTRYSOLVE",
                               r"\addlegendentry{solve}" if ci == 0 else ""))

    npool = min(st[cfg]["n"] for _, _, st, _ in loaded for cfg in configs)
    tuned, default = configs[0], configs[1]
    ref_buf = [
        "% generated by scripts/plot_amgx_scaling.py -- do not edit\n",
        "\\def\\amgxMolecule{%s}\n" % (args.molecule or sorted(molecules)[0]),
        "\\def\\amgxRef{%s}\n" % texify(loaded[ref][0]).replace("|", ", "),
        "\\def\\amgxRepeats{%d}\n" % npool,
        "\\def\\amgxTunedName{\\texttt{%s}}\n" % texify(tuned or "default"),
        "\\def\\amgxTunedIters{%s}\n"
        % fmt_iters(loaded[ref][2][tuned]).replace("--", "\\,--\\,"),
        "\\def\\amgxDefaultIters{%s}\n"
        % fmt_iters(loaded[ref][2][default]).replace("--", "\\,--\\,"),
    ]
    with open(reftex, "w") as fh:
        fh.write(localise("".join(ref_buf), args.tag))

    with open(fig, "w") as fh:
        fh.write(localise(FIGURE_TEMPLATE
                          .replace("COLSETUP", COLOUR_SETUP)
                          .replace("COLSOLVE", COLOUR_SOLVE)
                          .replace("SYMCOORDS", ",".join(sym))
                          .replace("XTICKLABELS", ticklabels)
                          .replace("PANELS", "".join(panels)), args.tag))

    rows = []
    for ci, cfg in enumerate(configs):
        head = (r"\texttt{%s}" % texify(cfg)) if cfg else "AMGX default"
        rows.append("\\midrule\n\\multicolumn{8}{l}{\\emph{%s}} \\\\\n" % head)
        refval = loaded[ref][2][cfg]["total"]
        # No ranks or GPUs column: the platform label already carries both, and
        # at eight columns this table is already at the text width.
        for i, (label, ranks, stats, _) in enumerate(loaded):
            s = stats[cfg]
            rest = ("---" if s["tsolve"] is None
                    else "%.2f" % (s["tsolve"] - s["total"]))
            rows.append("%s & %s & %.3f & %.3f & %.3f & "
                        "$\\times$%.2f & %s & %s \\\\\n"
                        % (texify(label).replace("|", ", "),
                           fmt_iters(s), s["setup"], s["solve"], s["total"],
                           refval / s["total"],
                           "%.0f\\%%" % (100 * s["spread"])
                           if s["warm_spread"] is None else
                           "%.0f\\%% (%.0f\\%%)" % (100 * s["spread"],
                                                    100 * s["warm_spread"]),
                           rest))
    # The reason the table exists is that `rest` dominates, so the claim is
    # checked against the numbers rather than written into the template: if a
    # future dataset stops supporting it, the sentence goes rather than lying.
    tuned_rest = [(st[tuned]["tsolve"] - st[tuned]["total"], st[tuned]["total"])
                  for _, _, st, _ in loaded if st[tuned]["tsolve"] is not None]
    if tuned_rest and all(r > a for r, a in tuned_rest):
        claim = ("It is the larger of the two in every "
                 "\\texttt{%s} row, so a speedup on \\emph{AMGX} is not a "
                 "speedup on the stage." % texify(tuned or "default"))
    else:
        claim = "A speedup on \\emph{AMGX} is therefore not a speedup on the stage."
    with open(tab, "w") as fh:
        fh.write(localise(TABLE_TEMPLATE
                          .replace("RESTCLAIM", claim)
                          .replace("ROWS", "".join(rows)), args.tag))

    print("reference series: [%d] %s\n" % (ref, loaded[ref][0].replace("|", ", ")))
    for cfg in configs:
        print("--- amgx_config = %r" % (cfg or "<AMGX default>"))
        refval = loaded[ref][2][cfg]["total"]
        for label, ranks, stats, _ in loaded:
            s = stats[cfg]
            print("   %-34s np=%-3s n=%-2d setup %6.3f  solve %6.3f  "
                  "total %6.3f  x%.2f  spread %-14s iters %-6s t_solve %s"
                  % (label.replace("|", ", "), ranks, s["n"], s["setup"],
                     s["solve"], s["total"], refval / s["total"],
                     "%.1f%%" % (100 * s["spread"]) if s["warm_spread"] is None
                     else "%.1f%% (%.1f%% warm)" % (100 * s["spread"],
                                                    100 * s["warm_spread"]),
                     fmt_iters(s),
                     "---" if s["tsolve"] is None else "%.3f" % s["tsolve"]))
        print()
    print("wrote %s\n      %s\n      %s\n      %s"
          % (os.path.join(data_dir, args.tag + "_c*.dat"), reftex, fig, tab))


if __name__ == "__main__":
    sys.exit(main())

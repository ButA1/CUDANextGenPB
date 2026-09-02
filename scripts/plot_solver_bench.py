#!/usr/bin/env python3
"""
Turn the sweep-A (linear solver) rows of a bench_sweep.py run into a pgfplots
column chart and a booktabs table.

Sweep A is three configurations -- LIS, AMGX on its built-in defaults, and AMGX
with the tuned config file -- each run ten times with energy_method pinned to 1.
The chart compares the solve stage across them; the table carries the detail that
explains the difference (iteration count, the AMGX setup/solve split, and the
residual each solver actually reached, which is not the same for all three).

Give it one folder for a single-molecule chart, or several to get one group of
columns per molecule:

    python3 scripts/plot_solver_bench.py test0
    python3 scripts/plot_solver_bench.py test0 test1 test2 --relative

Writes, relative to the thesis directory:
    figures/data/<tag>.dat        one row per molecule, one column trio per config
    figures/data/<tag>_ref.tex    reference values as macros
    figures/<tag>.tex             the figure, \\input-able like the others
    figures/<tag>_table.tex       the accompanying table
"""

import argparse
import csv
import math
import os
import re
import statistics
import sys
from collections import defaultdict

# This script has its own CSV handling -- it reads bench_runs.csv directly rather
# than through the FMM normalisation -- but the machine/rank caption note is
# shared so every figure in the chapter words it identically.
import fmm_csv

SOLVER_MACROS = ["solverMetric", "solverRepeats", "solverUnit", "solverMolecules",
                  "solverEnergyMethod", "solverMachine", "solverCaveats"]


def localise(text, tag):
    r"""Suffix this figure's macros and resolve REFLABEL, so a local-only and a
    cluster-only solver_bench figure can \input into the same chapter.

    Without it every --tag emits the same \def and the same \label, and the
    second \input silently redefines the first figure's caption -- floats are
    typeset where they land, not where they are declared, so the collision is
    not even reliably in the reader's favour. At the default tag this is the
    identity on the label: "solver_bench" -> "solver-bench".
    """
    sfx = "".join(c for c in tag if c.isalpha())
    # (?![A-Za-z]) because a control sequence ends at the first non-letter.
    for name in SOLVER_MACROS:
        text = re.sub(r"\\" + name + r"(?![A-Za-z])", "\\\\" + name + sfx, text)
    return text.replace("REFLABEL", tag.replace("_", "-"))

# The two AMGX entries are the same solver under different configuration, so they
# share a hue and separate by lightness; LIS is a different solver and gets a
# different hue. Blue against orange stays legible under the common CVD types,
# and the three differ enough in lightness to survive a greyscale print.
CONFIGS = [
    ("lis",     "LIS",             "solverlis",  "D95F02"),
    ("amgx",    "AMGX (default)",  "solveramgx", "9DC3E0"),
    ("amgxcfg", "AMGX (tuned)",    "solvercfg",  "2C5F8D"),
]
CONFIG_ORDER = [c[0] for c in CONFIGS]

METRICS = {
    "solve": ("t_solve", "solve stage"),
    "wall": ("wall_s", "total run"),
    "assemble": ("t_assemble", "matrix assembly"),
}


def classify(row):
    """Map a sweep-A row onto one of the three configuration keys."""
    solver = (row.get("linear_solver") or "").strip().lower()
    if solver == "lis":
        return "lis"
    if solver == "amgx":
        return "amgxcfg" if (row.get("amgx_config") or "").strip() else "amgx"
    return None


def _single_value(rows, col):
    """The one value column `col` takes across `rows`, or None if it varies."""
    seen = {(r.get(col) or "").strip() for r in rows}
    seen.discard("")
    return seen.pop() if len(seen) == 1 else None


def collect(folder, csv_path, metric_col):
    # Filter on the folder column, not just the file, when the file could hold
    # runs from several test folders -- bench_sweep.py writes wherever -o points,
    # so a sweep of test2 with -o test1/bench_runs.csv lands both molecules in
    # one file. Every row records the folder it came from, so the split is exact;
    # without this the two molecules would be averaged together and labelled
    # with whichever one happened to be first in the file.
    #
    # The `folder` column holds the absolute path of the machine that ran the
    # sweep, which for a cluster run is never this repo's checkout path -- so the
    # filter is skipped when the file holds only one folder's rows (the common
    # case, <folder>/bench_runs.csv read on its own); a single-folder file is
    # this folder's by construction, whatever path it was written on.
    want = os.path.abspath(folder)
    with open(csv_path, newline="") as fh:
        ok = [r for r in csv.DictReader(fh)
              if r.get("status") == "ok" and r.get("sweep") == "A"]
    multi = len({r.get("folder") for r in ok}) > 1
    rows = [r for r in ok if not multi or (r.get("folder") or want) == want]
    if not rows:
        return None

    molecule = os.path.splitext(rows[0].get("molecule", "") or "")[0] or \
        os.path.basename(os.path.normpath(folder))

    groups = defaultdict(list)
    for r in rows:
        key = classify(r)
        if key:
            groups[key].append(r)

    def stat(grp, col, fn=statistics.median):
        vals = []
        for r in grp:
            raw = r.get(col, "")
            if raw not in ("", None):
                try:
                    vals.append(float(raw))
                except ValueError:
                    pass
        return fn(vals) if vals else None

    out = {"molecule": molecule, "folder": folder, "configs": {},
           "ranks": fmm_csv.ranks_of(rows),
           "energy_method": _single_value(rows, "energy_method")}
    for key, grp in groups.items():
        times = [float(r[metric_col]) for r in grp if r.get(metric_col)]
        if not times:
            continue
        out["configs"][key] = {
            "n": len(grp),
            "med": statistics.median(times),
            "min": min(times),
            "max": max(times),
            "iters": stat(grp, "solver_iters"),
            "resid": stat(grp, "solver_final_residual"),
            "setup": stat(grp, "amgx_setup_s"),
            "solve": stat(grp, "amgx_solve_s"),
            "total": stat(grp, "amgx_total_s"),
            "wall": stat(grp, "wall_s"),
            "amgx_config": _single_value(grp, "amgx_config"),
        }
    return out if out["configs"] else None


def write_dat(path, datasets, relative):
    """Wide layout: one row per molecule, one (value, err-, err+, iters) column
    group per configuration. Every series then reads its own columns and no row
    filtering is needed, which keeps the pgfplots side trivial."""
    header = ["molecule"]
    for key, _, _, _ in CONFIGS:
        header += [key, key + "_em", key + "_ep", key + "_it"]
    with open(path, "w") as fh:
        fh.write(" ".join(header) + "\n")
        for ds in datasets:
            for cells in [_dat_row(ds, relative)]:
                fh.write(" ".join(cells) + "\n")


def _dat_row(ds, relative):
    present = [c["med"] for c in ds["configs"].values()]
    scale = min(present) if (relative and present) else 1.0
    cells = [ds["molecule"].replace(" ", "")]
    for key, _, _, _ in CONFIGS:
        c = ds["configs"].get(key)
        if not c:
            # pgfplots draws nothing for a nan coordinate
            cells += ["nan", "0", "0", "nan"]
            continue
        med, lo, hi = c["med"] / scale, c["min"] / scale, c["max"] / scale
        cells += ["{:.6g}".format(med), "{:.6g}".format(med - lo),
                  "{:.6g}".format(hi - med),
                  "nan" if c["iters"] is None else "{:.0f}".format(c["iters"])]
    return cells


def write_long_dat(path, ds, relative):
    """Long layout for the single-molecule chart: one row per configuration, with
    a numeric index so the bars can sit on a numeric x axis."""
    present = [c["med"] for c in ds["configs"].values()]
    scale = min(present) if (relative and present) else 1.0
    with open(path, "w") as fh:
        fh.write("idx config value em ep iters\n")
        for idx, (key, _, _, _) in enumerate(CONFIGS):
            c = ds["configs"].get(key)
            if not c:
                continue
            med, lo, hi = c["med"] / scale, c["min"] / scale, c["max"] / scale
            fh.write("{} {} {:.6g} {:.6g} {:.6g} {}\n".format(
                idx, key, med, med - lo, hi - med,
                "nan" if c["iters"] is None else "{:.0f}".format(c["iters"])))


def parse_machine_overrides(pairs):
    """--machine-override MOLECULE=TEXT, repeatable -- one molecule measured
    somewhere other than --machine. Returns {molecule: machine text}."""
    overrides = {}
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(
                "--machine-override needs MOLECULE=TEXT, got {!r}".format(item))
        mol, text = item.split("=", 1)
        overrides[mol.strip()] = text
    return overrides


def dataset_repeats(ds):
    """The repeat count shared by every config in `ds`, or None if it varies."""
    ns = {c["n"] for c in ds["configs"].values()}
    return ns.pop() if len(ns) == 1 else None


def build_caption_notes(datasets, default_machine, overrides, nrep):
    """\\solverMachine covers only the datasets NOT in `overrides`. A molecule
    measured on different hardware, with a different repeat count, or with a
    different AMGX config does not fit that one blanket sentence -- folding it
    in anyway would misdescribe it, so it gets its own clause in
    \\solverCaveats instead. The repeat-count check runs over every dataset,
    not just overridden ones: \\solverRepeats is a single number (the median
    across all of them), which silently hides a molecule that used a very
    different count. Also returns the base energy_method (None if even the
    non-overridden datasets disagree)."""
    base = [ds for ds in datasets if ds["molecule"] not in overrides]

    base_ranks = {ds["ranks"] for ds in base}
    ranks = base_ranks.pop() if len(base_ranks) == 1 else None
    machine_note = fmm_csv.machine_note(default_machine, ranks)

    base_em = {ds["energy_method"] for ds in base}
    energy_method = base_em.pop() if len(base_em) == 1 else None

    base_cfg = {ds["configs"]["amgxcfg"]["amgx_config"] for ds in base
                if "amgxcfg" in ds["configs"]}
    base_cfg = base_cfg.pop() if len(base_cfg) == 1 else None

    clauses = []
    for ds in datasets:
        mol = ds["molecule"]
        bits = []
        if mol in overrides:
            bits.append("measured on {} with {} MPI ranks".format(
                overrides[mol], ds["ranks"] or "an unrecorded number of"))
        reps = dataset_repeats(ds)
        if reps is not None and reps != nrep:
            # Not "(hence no whiskers)" even at reps == 1: the error-bar cap is
            # still drawn at zero spread, so it reads as a (flat) whisker anyway.
            bits.append("{} run{} per configuration, not {}".format(
                reps, "" if reps == 1 else "s", nrep))
        cfg = ds["configs"].get("amgxcfg", {}).get("amgx_config")
        if cfg and base_cfg and cfg != base_cfg:
            bits.append("AMGX (tuned) using \\texttt{{{}}} instead of \\texttt{{{}}}"
                        .format(cfg.replace("_", "\\_"), base_cfg.replace("_", "\\_")))
        if bits:
            clauses.append("{}: {}.".format(mol, "; ".join(bits)))
    caveats = (" " + " ".join(clauses)) if clauses else ""
    return machine_note, energy_method, caveats


def write_ref_tex(path, datasets, metric, relative, nrep, machine_note="",
                   energy_method=None, caveats=""):
    tag = os.path.basename(path)[:-len("_ref.tex")]
    body = "".join([
        "% generated by scripts/plot_solver_bench.py -- do not edit\n",
        "\\def\\solverMetric{%s}\n" % METRICS[metric][1],
        "\\def\\solverRepeats{%s}\n" % nrep,
        "\\def\\solverUnit{%s}\n" %
        ("relative to the fastest" if relative else "seconds"),
        "\\def\\solverMolecules{%s}\n" %
        ", ".join(ds["molecule"] for ds in datasets),
        "\\def\\solverEnergyMethod{%s}\n" %
        (energy_method if energy_method is not None else "\\textbf{[mixed]}"),
        "\\def\\solverMachine{%s}\n" % machine_note,
        "\\def\\solverCaveats{%s}\n" % caveats,
    ])
    with open(path, "w") as fh:
        fh.write(localise(body, tag))


FIGURE_TEMPLATE = r"""% ------------------------------------------------------------------------
%  Linear solver comparison (bench_sweep.py sweep A).
%  Generated by scripts/plot_solver_bench.py -- regenerate rather than edit.
%  Include with \input{figures/TAG}
%  Needs \usepackage{pgfplots} in the preamble.
% ------------------------------------------------------------------------
\input{figures/data/TAG_ref}

\pgfplotsset{compat=1.16}

% Keep only the rows whose column #1 equals #2.
\pgfplotsset{
  discard if not/.style 2 args={
    x filter/.code={%
      \edef\tempa{\thisrow{#1}}\edef\tempb{#2}%
      \ifx\tempa\tempb\else\def\pgfmathresult{inf}\fi
    }
  }
}

COLORDEFS

\begin{figure}[!htb]
\centering
\begin{tikzpicture}
\begin{axis}[
    ybar,
    % The iteration-count label sits above the top whisker by NEARSHIFT, which
    % ymax's headroom does not always cover -- clip=false stops that label from
    % being cut off instead of chasing a headroom multiplier that varies with
    % how many digits the label has.
    clip=false,
    width=WIDTH, height=6.4cm,
    bar width=BARWIDTH,
    enlarge x limits=ENLARGE,
    YRANGE
    ylabel={YLABEL},
    ylabel style={font=\footnotesize},
    AXISX
    xticklabel style={font=\footnotesize, align=center},
    tick label style={font=\scriptsize},
    ymajorgrids=true,
    major grid style={black!12},
    LEGENDSTYLE
    % Run-to-run spread is only a few percent, so the upper whisker is 1-2 mm and
    % the lower half is buried in the bar fill. Caps nearly as wide as the bar
    % are what make it read as an error bar rather than a stray tick.
    % T-caps. rotate=90 is required: pgfplots draws the cap perpendicular to the
    % error direction by itself, but any mark size given here clobbers that, and
    % the cap then renders vertically -- which reads as a longer whisker rather
    % than a wider cap. Setting error mark explicitly does not help; the rotation
    % is what matters.
    error bars/error bar style={black!85, line width=0.8pt},
    error bars/error mark options={
      mark size=CAPSIZE, rotate=90, line width=0.8pt, black!85,
    },
    every node near coord/.append style={
      font=\tiny, color=black!60, anchor=south, yshift=NEARSHIFT,
    },
]
SERIES
\end{axis}
\end{tikzpicture}

\caption[Linear solver comparison]{\textbf{Linear solver comparison on
    \solverMolecules{}.} Median \solverMetric{} time over \solverRepeats{} runsAXISNOTE,
    whiskers spanning fastest to slowest; the figure above each column is the
    iteration count.\solverMachine{}\solverCaveats{}}
\label{fig:REFLABEL}
\end{figure}
"""

# Grouped layout (several molecules): one series per configuration, each reading
# its own column group. nodes near coords carries the iteration count as explicit
# meta, so pgfplots places each label over its own bar -- positioning them by hand
# would need the per-series bar shift.
SERIES_TEMPLATE = r"""  \addplot+[
    ybar, fill=COLOR, draw=COLOR!70!black, line width=0.4pt,
    nodes near coords, point meta=explicit symbolic,
    error bars/.cd, y dir=both, y explicit,
  ] table[x=molecule, y=KEY, y error minus=KEY_em, y error plus=KEY_ep,
          meta=KEY_it] {figures/data/TAG.dat};
  \addlegendentry{LABEL}
"""

# Single-molecule layout: configurations go on the x axis, one bar each, so the
# axis is not mostly empty and the tick labels do the legend's job.
#
# bar shift=0pt is load-bearing. ybar treats several \addplot as a group and
# shifts each series sideways so they sit next to each other -- which is what the
# multi-molecule layout wants, but here each series is already its own x
# position, so the shift would slide every bar off its own tick.
SERIES_ONE_TEMPLATE = r"""  \addplot+[
    ybar, bar shift=0pt, fill=COLOR, draw=COLOR!70!black, line width=0.4pt,
    nodes near coords, point meta=explicit symbolic,
    error bars/.cd, y dir=both, y explicit,
  ] table[x=idx, y=value, y error minus=em, y error plus=ep, meta=iters,
          discard if not={config}{KEY}] {figures/data/TAG.dat};
"""


def build_figure(tag, datasets, metric, relative, out_path, log_scale=False):
    colordefs = "\n".join(
        "\\definecolor{%s}{HTML}{%s}" % (name, hexv)
        for _, _, name, hexv in CONFIGS
    )
    single = len(datasets) == 1
    present_keys = [k for k, _, _, _ in CONFIGS
                    if any(k in ds["configs"] for ds in datasets)]

    # Headroom for the iteration label sitting above the top whisker.
    tops, bottoms = [], []
    for ds in datasets:
        vals = [c["med"] for c in ds["configs"].values()]
        scale = min(vals) if (relative and vals) else 1.0
        tops += [c["max"] / scale for c in ds["configs"].values()]
        bottoms += [c["min"] / scale for c in ds["configs"].values()]
    ymax = max(tops) * 1.14

    if log_scale:
        # log origin y=infty is what makes ybar draw down to ymin rather than to
        # 1, which is where pgfplots puts the base of a log-axis bar by default.
        # The configs here rarely span a full decade, so bare powers of ten would
        # leave only one or two ticks on the axis -- a 1-2-5 ladder fills it in.
        ymin = min(bottoms) / 1.3
        # 1.08 measured short: at 6VYB/1vsz's ~15x headroom the iteration label
        # (yshift plus the \tiny node's own box) landed with its top half behind
        # the axis frame line instead of above it. 1.24 was checked against an
        # actual pdflatex render of this exact dataset (labels clear the frame
        # with a visible gap); it is a log-space multiplier, so it does not carry
        # over exactly to a dataset spanning a very different number of decades.
        ymax *= 1.24
        lo_dec, hi_dec = math.floor(math.log10(ymin)), math.ceil(math.log10(ymax))
        yticks = [m * 10 ** e for e in range(lo_dec, hi_dec + 1) for m in (1, 2, 5)
                  if ymin <= m * 10 ** e <= ymax]
        yrange = ("ymode=log, log origin y=infty,\n"
                  "    ymin={:.6g}, ymax={:.6g},\n"
                  "    ytick={{{}}},\n"
                  "    log ticks with fixed point,").format(
                      ymin, ymax, ",".join("{:.6g}".format(t) for t in yticks))
        axisnote = " on a logarithmic axis"
    else:
        yrange = "ymin=0, ymax={:.6g},".format(ymax)
        axisnote = ""

    if single:
        labels = {k: lbl for k, lbl, _, _ in CONFIGS}
        axis_x = (
            "xtick={%s},\n    xticklabels={%s},\n    xmin=-0.7, xmax=%.1f,"
            % (",".join(str(i) for i, (k, _, _, _) in enumerate(CONFIGS)
                        if k in present_keys),
               ",".join("{%s}" % labels[k].replace(" (", "\\\\(")
                        for k in present_keys),
               len(CONFIGS) - 0.3)
        )
        series = "".join(
            SERIES_ONE_TEMPLATE
            .replace("COLOR", name).replace("KEY", key).replace("TAG", tag)
            for key, _, name, _ in CONFIGS if key in present_keys
        )
        width, bar_width, enlarge, legend = "9.6cm", "26pt", "0.18", ""
        cap = "6pt"
    else:
        coords = ",".join(ds["molecule"].replace(" ", "") for ds in datasets)
        axis_x = "symbolic x coords={%s},\n    xtick=data," % coords
        series = "".join(
            SERIES_TEMPLATE
            .replace("COLOR", name).replace("KEY", key)
            .replace("LABEL", label).replace("TAG", tag)
            for key, label, name, _ in CONFIGS if key in present_keys
        )
        n_mol = len(datasets)
        width = "{:.1f}cm".format(min(15.0, max(8.0, 3.2 * n_mol + 3.5)))
        bar_width = "16pt" if n_mol <= 2 else "10pt"
        enlarge = "0.28"
        cap = "3.5pt" if n_mol <= 2 else "2.5pt"
        legend = ("legend style={\n"
                  "      at={(0.5,-0.16)}, anchor=north, draw=none,\n"
                  "      legend columns=3, font=\\footnotesize, column sep=0.8em,\n"
                  "    },")

    ylabel = ("\\solverMetric{} time, relative to the fastest"
              if relative else "\\solverMetric{} time [s]")

    body = (FIGURE_TEMPLATE
            .replace("COLORDEFS", colordefs)
            .replace("SERIES", series)
            .replace("AXISX", axis_x)
            .replace("LEGENDSTYLE", legend)
            .replace("YRANGE", yrange)
            .replace("AXISNOTE", axisnote)
            # BARWIDTH before WIDTH: the latter is a substring of the former
            .replace("BARWIDTH", bar_width)
            .replace("WIDTH", width)
            .replace("ENLARGE", enlarge)
            .replace("CAPSIZE", cap)
            .replace("NEARSHIFT", "5pt")
            .replace("YLABEL", ylabel)
            .replace("TAG", tag))

    # An omitted substitution leaves a line of bare indentation, and TeX reads a
    # whitespace-only line as \par -- which ends the axis options early. Drop
    # those; genuinely empty lines (between top-level blocks) are kept.
    body = "\n".join(ln for ln in body.split("\n") if ln.strip() or ln == "")

    with open(out_path, "w") as fh:
        fh.write(localise(body, tag))


def fmt(v, spec="{:.4f}", dash="--"):
    return dash if v is None or (isinstance(v, float) and math.isnan(v)) \
        else spec.format(v)


def fmt_sci(v, dash="--"):
    """Residuals as proper math: 8.8e-07 -> $8.8\\times10^{-7}$."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return dash
    if v == 0:
        return "$0$"
    exp = int(math.floor(math.log10(abs(v))))
    mant = v / 10 ** exp
    return "${:.1f}\\times10^{{{}}}$".format(mant, exp)


def write_table(path, datasets, metric, out_tex):
    lines = [
        "% generated by scripts/plot_solver_bench.py -- do not edit",
        "\\begin{table}[!htb]",
        "\\centering",
        # Nine columns overflow \textwidth at the body size. \small is not enough:
        # \tabcolsep is a fixed 6pt per gutter and does not scale with the font, so
        # 18 gutters carry ~108pt that no font change touches -- \small lands at
        # ~459pt against a 455pt textwidth. \footnotesize clears it with room and
        # matches the other generated tables.
        "\\footnotesize",
        "\\caption[Linear solver comparison]{\\textbf{Linear solver comparison.} "
        "Median over \\solverRepeats{} runs. \\emph{stage} is the whole "
        "\\solverMetric{}, split into the part AMGX spends on the GPU (its own "
        "\\emph{setup} and \\emph{solve} timers) and the \\emph{host} remainder. "
        "\\emph{rel.} is the stage time relative to the fastest configuration "
        "for that molecule.\\solverMachine{}\\solverCaveats{}}",
        "\\label{tab:REFLABEL}",
        "\\begin{tabular}{llrrrrrrl}",
        "\\toprule",
        "& & & \\multicolumn{2}{c}{GPU [s]} & & & & \\\\",
        "\\cmidrule(lr){4-5}",
        "molecule & configuration & iters & setup & solve & host [s] "
        "& stage [s] & rel. & residual \\\\",
        "\\midrule",
    ]
    for di, ds in enumerate(datasets):
        if di:
            lines.append("\\midrule")
        present = [c["med"] for c in ds["configs"].values()]
        best = min(present) if present else None
        first = True
        for key, label, _, _ in CONFIGS:
            c = ds["configs"].get(key)
            if not c:
                continue
            mol = ds["molecule"].replace("_", "\\_") if first else ""
            first = False
            rel = c["med"] / best if best else None
            # Whatever the stage cost that AMGX's own timers do not account for.
            # For LIS both timers are absent and the whole stage is host time,
            # which is the honest reading -- it runs entirely on the CPU.
            host = c["med"]
            if host is not None:
                host -= (c["setup"] or 0.0) + (c["solve"] or 0.0)
            lines.append(
                "{} & {} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
                    mol, label,
                    fmt(c["iters"], "{:.0f}"),
                    fmt(c["setup"]),
                    fmt(c["solve"]),
                    fmt(host),
                    fmt(c["med"]),
                    fmt(rel, "{:.2f}$\\times$"),
                    fmt_sci(c["resid"]),
                ))
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    tag = os.path.basename(path)[:-len("_table.tex")]
    with open(path, "w") as fh:
        fh.write(localise("\n".join(lines), tag))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+", help="test folders holding bench_runs.csv")
    ap.add_argument("--thesis", default="thesis",
                    help="thesis directory to write into (default: thesis)")
    ap.add_argument("--tag", default="solver_bench",
                    help="basename for the generated files (default: solver_bench)")
    fmm_csv.add_machine_argument(ap)
    ap.add_argument("--machine-override", action="append", default=None,
                    metavar="MOLECULE=TEXT",
                    help="one molecule was measured on different hardware than "
                         "--machine, e.g. --machine-override H1N1='an H200 node "
                         "of the TU Berlin HPC'. Repeatable. That molecule's "
                         "rank count, repeat count, and AMGX config are compared "
                         "against the rest and any difference is spelled out in "
                         "its own caption clause instead of folded into the one "
                         "blanket \\solverMachine sentence.")
    ap.add_argument("--metric", default="solve", choices=sorted(METRICS),
                    help="which stage the columns show (default: solve)")
    ap.add_argument("--csv", default=None,
                    help="read every folder's rows from this one CSV instead of "
                         "<folder>/bench_runs.csv; rows are still split by their "
                         "folder column, so one mixed file can feed several molecules")
    ap.add_argument("--relative", action="store_true",
                    help="plot each molecule relative to its fastest configuration "
                         "instead of absolute seconds")
    ap.add_argument("--log", action="store_true",
                    help="plot the y axis on a logarithmic scale instead of "
                         "linear -- useful when the slowest configuration is an "
                         "order of magnitude or more above the fastest")
    args = ap.parse_args()

    metric_col = METRICS[args.metric][0]
    datasets = []
    for folder in args.folders:
        csv_path = args.csv or os.path.join(folder, "bench_runs.csv")
        if not os.path.exists(csv_path):
            print("skipping {}: no bench_runs.csv".format(folder), file=sys.stderr)
            continue
        ds = collect(folder, csv_path, metric_col)
        if ds is None:
            print("skipping {}: no sweep A rows".format(folder), file=sys.stderr)
            continue
        datasets.append(ds)

    if not datasets:
        sys.exit("no sweep A data found -- run bench_sweep.py --sweeps a first")

    scales = [min(c["med"] for c in ds["configs"].values()) for ds in datasets]
    if not args.relative and len(scales) > 1 and max(scales) / min(scales) > 5:
        print("note: the molecules' {} times differ by {:.0f}x -- the small ones "
              "will be unreadable on a shared linear axis. Consider --relative."
              .format(args.metric, max(scales) / min(scales)), file=sys.stderr)

    nrep = int(statistics.median(
        c["n"] for ds in datasets for c in ds["configs"].values()))

    data_dir = os.path.join(args.thesis, "figures", "data")
    os.makedirs(data_dir, exist_ok=True)
    dat = os.path.join(data_dir, args.tag + ".dat")
    reftex = os.path.join(data_dir, args.tag + "_ref.tex")
    fig = os.path.join(args.thesis, "figures", args.tag + ".tex")
    tab = os.path.join(args.thesis, "figures", args.tag + "_table.tex")

    if len(datasets) == 1:
        write_long_dat(dat, datasets[0], args.relative)
    else:
        write_dat(dat, datasets, args.relative)
    overrides = parse_machine_overrides(args.machine_override)
    machine_note, energy_method, caveats = build_caption_notes(
        datasets, args.machine, overrides, nrep)
    write_ref_tex(reftex, datasets, args.metric, args.relative, nrep,
                  machine_note, energy_method, caveats)
    build_figure(args.tag, datasets, args.metric, args.relative, fig, args.log)
    write_table(tab, datasets, args.metric, tab)

    label = {k: lbl for k, lbl, _, _ in CONFIGS}
    for ds in datasets:
        ds_nrep = dataset_repeats(ds)
        print("\n{}  ({} runs per configuration)".format(
            ds["molecule"], ds_nrep if ds_nrep is not None else "mixed"))
        print("  %-16s %9s %8s %7s %10s %10s %11s" % (
            "configuration", "stage s", "spread", "iters", "setup s", "solve s",
            "residual"))
        best = min(c["med"] for c in ds["configs"].values())
        for key, _, _, _ in CONFIGS:
            c = ds["configs"].get(key)
            if not c:
                continue
            print("  %-16s %9.4f %8s %7s %10s %10s %11s   %.2fx" % (
                label[key], c["med"],
                "+{:.3f}/-{:.3f}".format(c["max"] - c["med"], c["med"] - c["min"]),
                fmt(c["iters"], "{:.0f}"), fmt(c["setup"]), fmt(c["solve"]),
                fmt(c["resid"], "{:.1e}"), c["med"] / best))

    print("\nwrote {}\n      {}\n      {}\n      {}".format(dat, reftex, fig, tab))
    print("\n\\input{figures/%s}\n\\input{figures/%s_table}"
          % (args.tag, args.tag))


if __name__ == "__main__":
    sys.exit(main())
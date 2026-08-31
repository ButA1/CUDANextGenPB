#!/usr/bin/env python3
r"""
Energy-stage cost of the three energy paths, side by side.

One grouped column chart over the molecules, three bars each:

    energy_method = 0   the stock CPU path
                  = 1   the naive O(NM) GPU sum
                  = 2   the FMM

The local molecules come out of a bench_sweep.py sweep B, where all three
methods ran back to back under one discretisation, so the bars differ only in
the energy path. A molecule that was only ever run on the cluster has no such
sweep -- it gets one bar per log file instead, via --extra, and every way its
run differs from the sweep (machine, ranks, grid scale, repeat count, FMM
configuration) is spelled out in its own caption clause rather than folded into
the blanket "measured on X" sentence.

A run that was killed before the energy stage finished is not dropped and is
not extrapolated. Its bar is drawn to the top of the axis and clipped, and the
caption states the lower bound the log actually supports: the job's time limit
minus every stage the log did report. Pass --dnf-limit to say what that limit
was.

    python3 scripts/plot_energy_methods.py test1 test2 \
        --extra H1N1:cpu=test7/h1n1-0-1884559.out \
        --extra H1N1:naive=test7/h1n1-1-1884560.out \
        --extra H1N1:fmm=test7/h1n1-2n-1883766.out \
        --extra-fmm 'H1N1=$P=10$, $\theta=0.4$, $n_{leaf}=256$' \
        --dnf-limit 4 \
        --machine 'the local system (Table~\ref{tab:test-systems})' \
        --machine-override 'H1N1=four H200 GPUs of the TU Berlin HPC'

Writes, relative to the thesis directory:
    figures/data/<tag>.dat        one row per molecule, one column group per method
    figures/data/<tag>_ref.tex    caption macros
    figures/<tag>.tex             the figure
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

import fmm_csv

# Ordered slowest to fastest, which is also the order the bars appear in. The
# CPU path is the odd one out and gets the odd hue; the two GPU paths share a
# hue and separate by lightness, so the figure reads as "CPU versus GPU" first
# and "naive versus FMM" second. Same three colours as the solver figure.
METHODS = [
    ("cpu",   "0", "CPU",                "energycpu",   "D95F02"),
    ("naive", "1", "naive GPU $O(NM)$",  "energynaive", "9DC3E0"),
    ("fmm",   "2", "FMM",                "energyfmm",   "2C5F8D"),
]
METHOD_KEYS = [m[0] for m in METHODS]
BY_EM = {m[1]: m[0] for m in METHODS}

# Bars are plotted in seconds on a log axis, so the DNF bar needs a finite
# value to be drawn at all. It is set to the axis maximum and clipped there --
# never to the lower bound, which would read as a measurement.
DNF_SENTINEL = "dnf"


# --------------------------------------------------------------------------
#  bench_sweep.py CSVs (the local molecules)
# --------------------------------------------------------------------------

def _floats(rows, col):
    out = []
    for r in rows:
        raw = (r.get(col) or "").strip()
        if raw:
            try:
                out.append(float(raw))
            except ValueError:
                pass
    return out


def _single_value(rows, col):
    seen = {(r.get(col) or "").strip() for r in rows}
    seen.discard("")
    return seen.pop() if len(seen) == 1 else None


def fmm_config_tex(rows):
    r"""$P=11$, $\theta=0.4$, $n_{leaf,src}=16$, $n_{leaf,tgt}=1024$."""
    mac = _single_value(rows, "fmm_mac")
    order = _single_value(rows, "fmm_multipole_order")
    src = _single_value(rows, "fmm_leaf_size")
    tgt = _single_value(rows, "fmm_target_leaf_size")
    bits = []
    if order:
        bits.append("$P = %s$" % order)
    if mac:
        bits.append("$\\theta = %s$" % mac)
    # The two leaf sizes are only worth distinguishing once they differ; before
    # the source/target split there was one knob and one symbol for it.
    if src and tgt and src != tgt:
        bits.append("$n_{leaf,src} = %s$" % src)
        bits.append("$n_{leaf,tgt} = %s$" % tgt)
    elif src:
        bits.append("$n_{leaf} = %s$" % src)
    return ", ".join(bits)


def prm_scale(folder):
    """The mesh scale from <folder>/options.prm, so the caption can say what a
    differently-meshed molecule is being compared against. Absent is fine -- the
    caveat then states the odd one out's scale without a comparison."""
    path = os.path.join(folder, "options.prm")
    if not os.path.exists(path):
        return None
    with open(path, errors="replace") as fh:
        for line in fh:
            m = re.match(r"\s*scale\s*=\s*([0-9.eE+-]+)", line)
            if m:
                return m.group(1)
    return None


def config_key(row):
    return tuple((row.get(c) or "").strip() for c in
                 ("fmm_mac", "fmm_multipole_order", "fmm_leaf_size",
                  "fmm_target_leaf_size"))


def pick_fmm_config(groups, ref_pol, accuracy):
    """The fastest FMM configuration that met the accuracy criterion.

    A folder whose sweep B is the whole parameter grid (test0 is 140
    configurations) has no single "the FMM run" to plot. Picking the outright
    fastest would put a bar on the figure that the sweep chapter had already
    rejected on accuracy, and picking by hand would not survive the data
    changing -- so the same criterion the chapter ranks on selects it here, and
    the caption says how many configurations it was selected from.

    Returns (rows, n_configs, n_ok) or None when nothing qualifies.
    """
    scored = []
    for key, grp in groups.items():
        times = _floats(grp, "t_energy")
        pols = _floats(grp, "energy_pol")
        if not times or not pols:
            continue
        err = (abs(statistics.mean(pols) - ref_pol) / abs(ref_pol)
               if ref_pol else None)
        scored.append((statistics.median(times), err, key, grp))
    if not scored:
        return None
    ok = [s for s in scored
          if accuracy is None or (s[1] is not None and s[1] <= accuracy)]
    if not ok:
        return None
    ok.sort(key=lambda s: s[0])
    return ok[0][3], len(scored), len(ok)


def collect_folder(folder, csv_path, sweep, metric_col, accuracy=None):
    """One dataset from the sweep of `folder`, keyed by method."""
    want = os.path.abspath(folder)
    with open(csv_path, newline="") as fh:
        rows = [r for r in csv.DictReader(fh)
                if r.get("status") == "ok" and r.get("sweep") == sweep
                and (r.get("folder") or want) == want]
    if not rows:
        return None

    molecule = os.path.splitext(rows[0].get("molecule", "") or "")[0] or \
        os.path.basename(os.path.normpath(folder))
    # 1vsz.pqr and 1VSZ.pqr are the same molecule; the PDB id is upper case.
    if re.fullmatch(r"[0-9][A-Za-z0-9]{3}", molecule):
        molecule = molecule.upper()

    groups = defaultdict(list)
    for r in rows:
        key = BY_EM.get((r.get("energy_method") or "").strip())
        if key:
            groups[key].append(r)

    # A sweep folder holds one row group per FMM configuration where the others
    # hold one run. Collapse it to a single bar before anything else looks at it.
    n_configs = n_ok = None
    if groups.get("fmm"):
        by_config = defaultdict(list)
        for r in groups["fmm"]:
            by_config[config_key(r)].append(r)
        if len(by_config) > 1:
            ref = (_floats(groups.get("cpu") or groups.get("naive") or [],
                           "energy_pol") or [None])
            ref = statistics.mean(ref) if ref[0] is not None else None
            if accuracy is None:
                raise SystemExit(
                    "%s holds %d FMM configurations in sweep %s and no rule for "
                    "choosing between them -- pass --accuracy to select the "
                    "fastest one that met the criterion."
                    % (folder, len(by_config), sweep))
            picked = pick_fmm_config(by_config, ref, accuracy)
            if picked is None:
                raise SystemExit(
                    "%s: none of its %d FMM configurations met --accuracy %g"
                    % (folder, len(by_config), accuracy))
            groups["fmm"], n_configs, n_ok = picked

    out = {"molecule": molecule, "source": csv_path, "methods": {},
           "n_configs": n_configs, "n_ok": n_ok,
           "ranks": fmm_csv.ranks_of(rows), "machine": None, "notes": [],
           "atoms": _single_value(rows, "num_atoms"),
           "scale": prm_scale(folder)}
    for key, grp in groups.items():
        times = _floats(grp, metric_col)
        if not times:
            continue
        out["methods"][key] = {
            "n": len(times),
            "med": statistics.median(times),
            "min": min(times),
            "max": max(times),
            "pol": statistics.mean(_floats(grp, "energy_pol")) or None,
            "ionic": statistics.mean(_floats(grp, "energy_ionic")) or None,
            "config": fmm_config_tex(grp) if key == "fmm" else "",
            "ranks": fmm_csv.ranks_of(grp),
            "dnf": False,
        }
    return out if out["methods"] else None


# --------------------------------------------------------------------------
#  ngpb stdout logs (the cluster molecule)
# --------------------------------------------------------------------------

RE_EVENT = re.compile(
    r"^Event:\s*Compute energy,\s*total hits:\s*(\d+),\s*total time:\s*"
    r"([0-9.eE+-]+)\s*s\.")
RE_STAGE = re.compile(r"^Elapsed time\s*:\s*([0-9.eE+-]+)\s*ms")
RE_POL = re.compile(r"Polarization energy \[kT\]:\s*(-?[0-9.eE+-]+)")
RE_IONIC = re.compile(r"Direct ionic energy \[kT\]:\s*(-?[0-9.eE+-]+)")
RE_RANKS = re.compile(r"^CPUs\s*:\s*(\d+)\s*ranks")
RE_SCALE = re.compile(r"^\s*Scale:\s*([0-9.eE+-]+)")
RE_ATOMS = re.compile(r"^\s*Number of atoms\s*:\s*(\d+)")
RE_NODES = re.compile(r"^Nodes?\s*:\s*(\S+)")
RE_GPU = re.compile(r"^(NVIDIA [^,]+|Tesla [^,]+),")


def parse_log(path):
    """Everything the figure needs out of one ngpb stdout log.

    `energy` is None when the stage never reported, which is how a killed run is
    recognised -- there is no separate "it died" marker to look for. The stage
    times that did report are kept so the caller can turn a job time limit into
    a lower bound.
    """
    info = {"energy": None, "hits": None, "stages_ms": [], "pol": None,
            "ionic": None, "ranks": None, "scale": None, "atoms": None,
            "nodes": None, "gpus": [], "path": path}
    with open(path, errors="replace") as fh:
        for line in fh:
            m = RE_EVENT.match(line)
            if m:
                info["hits"], info["energy"] = int(m.group(1)), float(m.group(2))
                continue
            m = RE_STAGE.match(line)
            if m:
                info["stages_ms"].append(float(m.group(1)))
                continue
            m = RE_GPU.match(line)
            if m:
                info["gpus"].append(m.group(1).strip())
                continue
            for key, rx in (("ranks", RE_RANKS), ("scale", RE_SCALE),
                            ("atoms", RE_ATOMS), ("nodes", RE_NODES)):
                if info[key] is None:
                    m = rx.match(line)
                    if m:
                        info[key] = m.group(1)
            for key, rx in (("pol", RE_POL), ("ionic", RE_IONIC)):
                m = rx.search(line)
                if m:
                    info[key] = float(m.group(1))
    return info


def method_from_log(info, dnf_limit_s):
    """One bar's worth of data from a parsed log."""
    if info["energy"] is not None:
        return {
            "n": 1,
            "med": info["energy"], "min": info["energy"], "max": info["energy"],
            "pol": info["pol"], "ionic": info["ionic"],
            "config": "", "ranks": info["ranks"], "dnf": False,
        }
    # Killed before the stage reported. The reported stages are wall time the
    # energy stage demonstrably did not get, so the limit minus their sum is a
    # lower bound on how long it ran -- an understatement, since it also ignores
    # process start-up, input parsing and the final I/O.
    accounted = sum(info["stages_ms"]) / 1000.0
    floor = (dnf_limit_s - accounted) if dnf_limit_s else None
    return {
        "n": 0, "med": DNF_SENTINEL, "min": None, "max": None,
        "pol": None, "ionic": None, "config": "", "ranks": info["ranks"],
        "dnf": True, "floor": floor, "accounted": accounted,
    }


def parse_extra(specs, dnf_limit_s):
    """--extra MOLECULE:METHOD=PATH, repeatable. Returns datasets in first-seen
    molecule order."""
    order, byname = [], {}
    for item in specs or []:
        if "=" not in item or ":" not in item.split("=", 1)[0]:
            raise SystemExit(
                "--extra needs MOLECULE:METHOD=PATH, got {!r}".format(item))
        head, path = item.split("=", 1)
        mol, method = head.split(":", 1)
        mol, method = mol.strip(), method.strip().lower()
        if method not in METHOD_KEYS:
            raise SystemExit("--extra method must be one of {}, got {!r}"
                             .format("/".join(METHOD_KEYS), method))
        if not os.path.exists(path):
            raise SystemExit("--extra log not found: {}".format(path))
        info = parse_log(path)
        if mol not in byname:
            byname[mol] = {"molecule": mol, "source": path, "methods": {},
                           "ranks": None, "machine": None, "notes": [],
                           "atoms": None, "scale": None, "gpus": [],
                           "nodes": None, "logs": {}}
            order.append(byname[mol])
        ds = byname[mol]
        ds["methods"][method] = method_from_log(info, dnf_limit_s)
        ds["logs"][method] = info
        for key in ("ranks", "atoms", "scale", "nodes"):
            if not ds.get(key) and info.get(key):
                ds[key] = info[key]
        if info["gpus"] and not ds["gpus"]:
            ds["gpus"] = info["gpus"]
    return order


# --------------------------------------------------------------------------
#  Accuracy
# --------------------------------------------------------------------------

def relative_errors(ds):
    """Each method's energies against the most exact one present.

    The CPU path is the reference where it ran. Where it did not, the naive sum
    is -- it is a direct summation over the same inputs, so it differs from the
    CPU path only by summation order, and the local molecules put a number on
    that (see the table). Which one was used is recorded per dataset, because a
    caption that does not say cannot be checked.
    """
    for ref_key in ("cpu", "naive"):
        ref = ds["methods"].get(ref_key)
        if ref and ref.get("pol"):
            break
    else:
        return None
    ds["ref_method"] = ref_key
    for key, m in ds["methods"].items():
        for col in ("pol", "ionic"):
            base, got = ref.get(col), m.get(col)
            # The reference against itself is 0, which is not a measurement of
            # anything and reads in the table as an accuracy claim. Leave it out.
            m["err_" + col] = (abs(got - base) / abs(base)
                               if base and got and key != ref_key else None)
    return ref_key


# --------------------------------------------------------------------------
#  Output
# --------------------------------------------------------------------------

def bar_label(value):
    """Three significant figures, no unit (the axis carries it) -- short enough
    to sit over a 13pt bar, and the same rule at every magnitude, which matters
    once the axis spans from 0.0181 s to 3163 s. Above 1000 the %g form would
    switch to an exponent, so those keep plain digits."""
    return ("{:.0f}" if value >= 1000 else "{:.3g}").format(value)


def write_dat(path, datasets, dnf_top):
    header = ["molecule"]
    for key, _, _, _, _ in METHODS:
        header += [key, key + "_em", key + "_ep", key + "_lbl"]
    with open(path, "w") as fh:
        fh.write(" ".join(header) + "\n")
        for ds in datasets:
            cells = [ds["molecule"].replace(" ", "")]
            for key, _, _, _, _ in METHODS:
                m = ds["methods"].get(key)
                if not m:
                    # pgfplots draws nothing for a nan coordinate.
                    cells += ["nan", "0", "0", "{~}"]
                elif m["dnf"]:
                    # Clipped at the axis top, unlabelled -- the figure annotates
                    # it by hand so the wording is not squeezed into a bar label.
                    cells += ["{:.6g}".format(dnf_top), "0", "0", "{~}"]
                else:
                    med, lo, hi = m["med"], m["min"], m["max"]
                    cells += ["{:.6g}".format(med),
                              "{:.6g}".format(med - lo),
                              "{:.6g}".format(hi - med),
                              bar_label(med)]
            fh.write(" ".join(cells) + "\n")


def thousands(n):
    r"""14442610 -> 14\,442\,610. A thin space, not a comma: the caption already
    uses commas to separate molecules."""
    return "\\,".join(re.findall(r"\d{1,3}(?=(?:\d{3})*$)", str(int(n))))


def tick_label(ds):
    """Molecule over its atom count, as one centred two-line tick label."""
    if not ds.get("atoms"):
        return "{%s}" % ds["molecule"]
    return "{%s\\\\{\\scriptsize %s atoms}}" % (
        ds["molecule"], thousands(ds["atoms"]))


def _and(items):
    """a, b and c"""
    items = list(items)
    if len(items) < 2:
        return "".join(items)
    return "%s and %s" % (", ".join(items[:-1]), items[-1])


def hours(seconds):
    if seconds >= 3600:
        return "{:.0f}\\,h".format(math.floor(seconds / 3600.0))
    return "{:.0f}\\,min".format(math.floor(seconds / 60.0))


def build_caption_notes(datasets, default_machine, overrides, extra_fmm,
                        notes, nrep, dnf_limit_s):
    r"""\energyMachine covers only the datasets NOT in `overrides`; everything
    that makes another dataset incomparable to those gets its own clause in
    \energyCaveats. Folding a differently-measured molecule into the blanket
    sentence would misdescribe it, and this figure exists to be compared across
    columns, so every reason not to is stated."""
    base = [ds for ds in datasets if ds["molecule"] not in overrides]
    base_ranks = {ds["ranks"] for ds in base}
    machine = fmm_csv.machine_note(
        default_machine, base_ranks.pop() if len(base_ranks) == 1 else None)

    # The FMM configuration sentence. One shared config is a sentence; several
    # are listed per molecule, since the whole point of the sweep chapter is
    # that the optimum is molecule-specific.
    configs = {}
    for ds in datasets:
        cfg = extra_fmm.get(ds["molecule"]) or \
            ds["methods"].get("fmm", {}).get("config", "")
        if cfg:
            configs[ds["molecule"]] = cfg
    shared = defaultdict(list)
    for mol, cfg in configs.items():
        shared[cfg].append(mol)
    if len(shared) == 1 and len(configs) == len(datasets):
        fmm_note = " The FMM ran at %s." % next(iter(shared))
    elif configs:
        fmm_note = " The FMM ran at %s." % "; ".join(
            "%s for %s" % (cfg, _and(mols)) for cfg, mols in shared.items())
    else:
        fmm_note = ""

    clauses = []
    for ds in datasets:
        mol, bits = ds["molecule"], []
        if mol in overrides:
            bits.append("measured on %s with %s MPI ranks" % (
                overrides[mol], ds["ranks"] or "an unrecorded number of"))
            if ds.get("scale"):
                base_scale = next((d.get("scale") for d in base if d.get("scale")),
                                  None)
                bits.append("at a grid scale of %s%s" % (
                    ds["scale"],
                    " instead of %s" % base_scale if base_scale else ""))
        reps = {m["n"] for m in ds["methods"].values() if not m["dnf"]}
        reps = reps.pop() if len(reps) == 1 else None
        if reps is not None and reps != nrep:
            bits.append("%d run%s per method, not %d%s" % (
                reps, "" if reps == 1 else "s", nrep,
                " (hence no whiskers)" if reps == 1 else ""))
        for key, m in ds["methods"].items():
            if not m["dnf"]:
                continue
            label = dict((k, lbl) for k, _, lbl, _, _ in METHODS)[key]
            floor = m.get("floor")
            bits.append(
                "the %s run did not finish -- it was killed by the %s job "
                "limit with the energy stage still running, and the stages it "
                "did report account for only %s of that, so its bar is a lower "
                "bound of over %s and is drawn clipped at the top of the axis"
                % (label, hours(dnf_limit_s) if dnf_limit_s else "job",
                   hours(m.get("accounted", 0.0)),
                   hours(floor) if floor else "the remainder"))
        if ds.get("n_configs"):
            bits.append(
                "its FMM bar is the fastest of the %d configurations swept, of "
                "which %d met the criterion" % (ds["n_configs"], ds["n_ok"]))
        if notes.get(mol):
            bits.append(notes[mol])
        if bits:
            clauses.append("%s: %s." % (mol, "; ".join(bits)))
    return machine, fmm_note, (" " + " ".join(clauses)) if clauses else ""


def write_ref_tex(path, datasets, nrep, machine, fmm_note, caveats, refs):
    with open(path, "w") as fh:
        fh.write("% generated by scripts/plot_energy_methods.py -- do not edit\n")
        fh.write("\\def\\energyMolecules{%s}\n" %
                 ", ".join(ds["molecule"] for ds in datasets))
        fh.write("\\def\\energyRepeats{%s}\n" % nrep)
        fh.write("\\def\\energyMachine{%s}\n" % machine)
        fh.write("\\def\\energyFmmConfig{%s}\n" % fmm_note)
        fh.write("\\def\\energyReference{%s}\n" % refs)
        fh.write("\\def\\energyCaveats{%s}\n" % caveats)


FIGURE_TEMPLATE = r"""% ------------------------------------------------------------------------
%  Energy stage cost: CPU vs naive GPU sum vs FMM.
%  Generated by scripts/plot_energy_methods.py -- regenerate rather than edit.
%  Include with \input{figures/TAG}
%  Needs \usepackage{pgfplots} and \usetikzlibrary{arrows.meta} in the preamble.
% ------------------------------------------------------------------------
\input{figures/data/TAG_ref}

\pgfplotsset{compat=1.16}

COLORDEFS

\begin{figure}[!htb]
\centering
\begin{tikzpicture}
\begin{axis}[
    ybar,
    width=15.0cm, height=7.4cm,
    bar width=BARWIDTH,
    enlarge x limits=0.30,
    % A shared linear axis would put every bar but the slowest on the baseline:
    % the times span RANGEx. log origin y=infty is what makes the bars start at
    % ymin rather than at 1, which on a log axis is where ybar puts them by
    % default -- without it the sub-second bars would hang below the frame.
    ymode=log, log origin y=infty,
    ymin=YMIN, ymax=YMAX,
    ytick={YTICKS},
    % 1, 10, 100 rather than 10^0, 10^1, 10^2. The times are wall clock and get
    % read off as such; the exponents make a reader translate before comparing.
    log ticks with fixed point,
    ylabel={energy stage time [s]},
    ylabel style={font=\footnotesize},
    symbolic x coords={XCOORDS},
    xtick=data,
    % The atom count belongs on the axis, not in the caption: the three columns
    % are three problem sizes, and without it the H1N1 column reads as the same
    % work taking longer rather than as two orders of magnitude more of it.
    xticklabels={XTICKLABELS},
    xticklabel style={font=\footnotesize, align=center},
    tick label style={font=\scriptsize},
    ymajorgrids=true,
    major grid style={black!12},
    % Clipping stays ON. log origin y=infty gives every bar a base far below the
    % frame -- a large finite log coordinate, not a real infinity -- and the clip
    % is the only thing keeping those bases off the page. The DNF annotation
    % therefore cannot be drawn in here; it is anchored to a \coordinate and
    % drawn after \end{axis}, where clipping no longer applies.
    % -0.22, not the -0.14 the other figures use: the tick labels here are two
    % lines, and the legend has to clear the atom count as well as the name.
    legend style={
      at={(0.5,-0.22)}, anchor=north, draw=none,
      legend columns=3, font=\footnotesize, column sep=0.8em,
    },
    error bars/error bar style={black!85, line width=0.8pt},
    error bars/error mark options={
      mark size=3pt, rotate=90, line width=0.8pt, black!85,
    },
    % The white backing is load-bearing on the H1N1 group: a four-digit label is
    % wider than a 13pt bar, so it overhangs onto the neighbouring bar, which
    % there is a full-height block of colour.
    every node near coord/.append style={
      font=\tiny, color=black!55, anchor=south, yshift=4pt,
      inner sep=1pt, fill=white, fill opacity=0.85, text opacity=1,
    },
]
SERIES
DNFANCHORS
\end{axis}
DNFMARKS
\end{tikzpicture}

\caption[Energy stage cost by method]{\textbf{Cost of the energy stage on
    \energyMolecules{}.} Median wall time of the \texttt{Compute energy} stage
    over \energyRepeats{} runs, on a logarithmic axis, with whiskers spanning
    the fastest and slowest of those runs; the number above each bar is that
    median in seconds. The three bars are the stock CPU path
    (\texttt{energy\_method\,=\,0}), the naive $O(NM)$ GPU sum
    (\texttt{\,=\,1}) and the FMM (\texttt{\,=\,2}), run on the same
    discretisation so that they differ only in the energy path.\energyFmmConfig
    \energyMachine\energyCaveats}
\label{fig:energy-methods}
\end{figure}
"""

SERIES_TEMPLATE = r"""  \addplot+[
    ybar, bar shift=SHIFT, fill=COLOR, draw=COLOR!70!black, line width=0.4pt,
    nodes near coords, point meta=explicit symbolic,
    error bars/.cd, y dir=both, y explicit,
  ] table[x=molecule, y=KEY, y error minus=KEY_em, y error plus=KEY_ep,
          meta=KEY_lbl] {figures/data/TAG.dat};
  \addlegendentry{LABEL}
"""

# The clipped bar on its own reads as a measurement that happens to be tall, so
# it is labelled where it leaves the axis. Re-applying the series' own bar shift
# to the symbolic coordinate is what keeps the arrow over the right bar; there
# is no way to ask pgfplots for a single bar's centre. \coordinate draws
# nothing, so it is safe inside the clipped axis; the arrow that does draw is
# emitted after \end{axis}.
DNF_ANCHOR_TEMPLATE = r"""  \coordinate (NAME) at ([xshift=SHIFT]axis cs:MOLECULE,YTOP);
"""

DNF_TEMPLATE = r"""\draw[-{Stealth[length=2.6mm,width=2.2mm]}, line width=1pt, COLOR!75!black]
  (NAME) -- ++(0,0.62cm)
  node[above, inner sep=1pt, font=\scriptsize\bfseries, text=black!70]{DNF};
"""


def build_figure(tag, datasets, out_path, ymin, ymax, yticks, dnf_top, span):
    colordefs = "\n".join("\\definecolor{%s}{HTML}{%s}" % (name, hexv)
                          for _, _, _, name, hexv in METHODS)
    present = [k for k, _, _, _, _ in METHODS
               if any(k in ds["methods"] for ds in datasets)]
    bar_width, gap = 13, 14  # pt; explicit shifts beat pgfplots' automatic ones
    shifts = {k: (i - (len(present) - 1) / 2.0) * gap
              for i, k in enumerate(present)}

    series = "".join(
        SERIES_TEMPLATE
        .replace("SHIFT", "{:.1f}pt".format(shifts[key]))
        .replace("COLOR", name).replace("KEY", key)
        .replace("LABEL", label).replace("TAG", tag)
        for key, _, label, name, _ in METHODS if key in present)

    dnf = [(ds, key, name) for ds in datasets
           for key, _, _, name, _ in METHODS
           if ds["methods"].get(key, {}).get("dnf")]
    anchors = "".join(
        DNF_ANCHOR_TEMPLATE
        .replace("NAME", "dnf-%s-%s" % (ds["molecule"], key))
        .replace("SHIFT", "{:.1f}pt".format(shifts[key]))
        .replace("MOLECULE", ds["molecule"].replace(" ", ""))
        .replace("YTOP", "{:.6g}".format(dnf_top))
        for ds, key, _ in dnf)
    marks = "".join(
        DNF_TEMPLATE
        .replace("NAME", "dnf-%s-%s" % (ds["molecule"], key))
        .replace("COLOR", name)
        for ds, key, name in dnf)

    body = (FIGURE_TEMPLATE
            .replace("COLORDEFS", colordefs)
            .replace("SERIES", series)
            .replace("DNFANCHORS", anchors)
            .replace("DNFMARKS", marks)
            .replace("XCOORDS", ",".join(ds["molecule"].replace(" ", "")
                                         for ds in datasets))
            .replace("XTICKLABELS", ",".join(tick_label(ds) for ds in datasets))
            .replace("BARWIDTH", "{:d}pt".format(bar_width))
            .replace("YTICKS", ",".join("{:.6g}".format(t) for t in yticks))
            .replace("YMIN", "{:.6g}".format(ymin))
            .replace("YMAX", "{:.6g}".format(ymax))
            .replace("RANGE", "{:.0f}".format(span))
            .replace("TAG", tag))
    body = "\n".join(ln for ln in body.split("\n") if ln.strip() or ln == "")
    with open(out_path, "w") as fh:
        fh.write(body)


def fmt_sci(v, dash="--"):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return dash
    if v == 0:
        return "$0$"
    exp = int(math.floor(math.log10(abs(v))))
    return "${:.2f}\\times10^{{{}}}$".format(v / 10 ** exp, exp)


def fmt_ratio(base, value, dash="--"):
    """A speed-up. The sub-unit entries are the slower paths measured against a
    faster one, and 0.0x tells the reader nothing, so they keep two significant
    figures where the speed-ups above 1 keep one decimal."""
    if not base or not value:
        return dash
    r = base / value
    spec = "{:.0f}" if r >= 100 else ("{:.1f}" if r >= 1 else "{:.2g}")
    return (spec + "$\\times$").format(r)


def write_table(path, datasets, dnf_limit_s):
    labels = {k: lbl for k, _, lbl, _, _ in METHODS}
    lines = [
        "% generated by scripts/plot_energy_methods.py -- do not edit",
        "\\begin{table}[!htb]",
        "\\centering",
        "\\footnotesize",
        "\\caption[Energy stage cost by method]{\\textbf{Cost and accuracy of "
        "the energy stage by method.} Median over \\energyRepeats{} runs, with "
        "the spread to the fastest and slowest run. \\emph{vs.\\ CPU} and "
        "\\emph{vs.\\ naive} are the speed-ups over those two paths. The "
        "relative errors are against \\energyReference{}, and are the "
        "quantities the FMM parameter sweep "
        "(Section~\\ref{sec:evaluation-results-energy-calculation}) was ranked "
        "on.\\energyFmmConfig\\energyMachine\\energyCaveats}",
        "\\label{tab:energy-methods}",
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        "& & & & \\multicolumn{2}{c}{speed-up} & "
        "\\multicolumn{2}{c}{rel.\\ error} \\\\",
        "\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}",
        "molecule & method & ranks & stage [s] & vs.\\ CPU & vs.\\ naive "
        "& polarization & ionic \\\\",
        "\\midrule",
    ]
    for di, ds in enumerate(datasets):
        if di:
            lines.append("\\midrule")
        cpu = ds["methods"].get("cpu")
        naive = ds["methods"].get("naive")
        cpu_t = cpu["med"] if cpu and not cpu["dnf"] else None
        naive_t = naive["med"] if naive and not naive["dnf"] else None
        first = True
        for key, _, _, _, _ in METHODS:
            m = ds["methods"].get(key)
            if not m:
                continue
            mol = ds["molecule"].replace("_", "\\_") if first else ""
            first = False
            if m["dnf"]:
                floor = m.get("floor")
                stage = ("$>%s$ (DNF)" % hours(floor).replace("\\,", "\\,")
                         if floor else "DNF")
                lines.append("{} & {} & {} & {} & -- & -- & -- & -- \\\\".format(
                    mol, labels[key], m["ranks"] or "--", stage))
                continue
            spread = ("" if m["n"] < 2 else
                      "\\,$+{:.3g}/-{:.3g}$".format(m["max"] - m["med"],
                                                    m["med"] - m["min"]))
            lines.append(
                # 4 significant figures, not 3 decimals: the column spans
                # 0.0181 s to 3163 s and a fixed decimal count drops the last
                # digit off one end or pads noise onto the other.
                "{} & {} & {} & {:.4g}{} & {} & {} & {} & {} \\\\".format(
                    mol, labels[key], m["ranks"] or "--", m["med"], spread,
                    fmt_ratio(cpu_t, m["med"]), fmt_ratio(naive_t, m["med"]),
                    fmt_sci(m.get("err_pol")),
                    fmt_sci(m.get("err_ionic"))))
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


def parse_pairs(items, flag):
    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit("%s needs MOLECULE=TEXT, got %r" % (flag, item))
        key, text = item.split("=", 1)
        out[key.strip()] = text
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="*",
                    help="test folders holding bench_runs.csv")
    ap.add_argument("--thesis", default="thesis")
    ap.add_argument("--tag", default="energy_methods")
    ap.add_argument("--sweep", default="B",
                    help="which bench_sweep.py sweep holds the energy_method "
                         "comparison (default: B)")
    ap.add_argument("--metric", default="t_energy",
                    help="CSV column to plot (default: t_energy)")
    ap.add_argument("--csv", default=None,
                    help="read every folder's rows from this one CSV instead "
                         "of <folder>/bench_runs.csv")
    fmm_csv.add_machine_argument(ap)
    ap.add_argument("--machine-override", action="append", metavar="MOLECULE=TEXT",
                    help="a molecule measured on hardware other than --machine. "
                         "Its rank count, grid scale and repeat count are then "
                         "compared against the rest and every difference gets "
                         "its own caption clause. Repeatable.")
    ap.add_argument("--extra", action="append", metavar="MOLECULE:METHOD=LOG",
                    help="one bar read from an ngpb stdout log rather than from "
                         "a sweep CSV, for a molecule that was only ever run "
                         "once. METHOD is cpu, naive or fmm. Repeatable.")
    ap.add_argument("--extra-fmm", action="append", metavar="MOLECULE=TEXT",
                    help="the FMM configuration an --extra molecule ran at; "
                         "logs do not record it. Inserted verbatim, so it can "
                         "carry math. Repeatable.")
    ap.add_argument("--note", action="append", metavar="MOLECULE=TEXT",
                    help="an extra caption clause for one molecule, appended "
                         "after the differences this script detects itself. "
                         "Repeatable.")
    ap.add_argument("--accuracy", type=float, default=None, metavar="RELERR",
                    help="the polarization-energy criterion the sweep chapter "
                         "ranks on. Required for a folder whose sweep holds the "
                         "whole FMM parameter grid rather than one chosen "
                         "configuration: the bar is then the fastest "
                         "configuration whose error against the CPU path is at "
                         "or under this, and the caption records how many it "
                         "was chosen from.")
    ap.add_argument("--dnf-limit", type=float, default=None, metavar="HOURS",
                    help="the job time limit a killed run hit. Turns the stage "
                         "times the log did report into a lower bound on the "
                         "energy stage. Without it the bar is still drawn, but "
                         "the caption cannot put a number on it.")
    args = ap.parse_args()

    dnf_limit_s = args.dnf_limit * 3600.0 if args.dnf_limit else None

    datasets = []
    for folder in args.folders:
        csv_path = args.csv or os.path.join(folder, "bench_runs.csv")
        if not os.path.exists(csv_path):
            print("skipping %s: no bench_runs.csv" % folder, file=sys.stderr)
            continue
        ds = collect_folder(folder, csv_path, args.sweep, args.metric,
                            args.accuracy)
        if ds is None:
            print("skipping %s: no sweep %s rows" % (folder, args.sweep),
                  file=sys.stderr)
            continue
        datasets.append(ds)
    datasets += parse_extra(args.extra, dnf_limit_s)

    if not datasets:
        sys.exit("no data -- give test folders, --extra logs, or both")

    for ds in datasets:
        missing = [k for k, _, _, _, _ in METHODS if k not in ds["methods"]]
        if missing:
            print("note: %s has no %s bar" % (ds["molecule"], "/".join(missing)),
                  file=sys.stderr)
        relative_errors(ds)

    nrep = int(statistics.median(
        [m["n"] for ds in datasets for m in ds["methods"].values()
         if not m["dnf"]] or [1]))

    # Axis range. ymin sits below the smallest bar so it is still a visible
    # block rather than a line on the baseline, and ymax a little above the
    # tallest real bar so its label fits -- the DNF bar is then drawn to ymax
    # and clipped there, which is the whole point of it.
    reals = [m["med"] for ds in datasets for m in ds["methods"].values()
             if not m["dnf"]]
    lo_dec = math.floor(math.log10(min(reals)))
    hi_dec = math.ceil(math.log10(max(reals)))
    ymin = 10.0 ** lo_dec / 2.0
    ymax = 10.0 ** hi_dec * 2.0
    yticks = [10.0 ** e for e in range(int(lo_dec), int(hi_dec) + 1)]
    span = max(reals) / min(reals)

    data_dir = os.path.join(args.thesis, "figures", "data")
    os.makedirs(data_dir, exist_ok=True)
    dat = os.path.join(data_dir, args.tag + ".dat")
    reftex = os.path.join(data_dir, args.tag + "_ref.tex")
    fig = os.path.join(args.thesis, "figures", args.tag + ".tex")
    tab = os.path.join(args.thesis, "figures", args.tag + "_table.tex")

    overrides = parse_pairs(args.machine_override, "--machine-override")
    extra_fmm = parse_pairs(args.extra_fmm, "--extra-fmm")
    notes = parse_pairs(args.note, "--note")

    refs = {ds.get("ref_method") for ds in datasets}
    refs.discard(None)
    ref_text = {"cpu": "the CPU path", "naive": "the naive GPU sum"}
    reference = (ref_text.get(refs.pop(), "the most exact path present")
                 if len(refs) == 1
                 else "the CPU path where it ran and the naive GPU sum otherwise")

    write_dat(dat, datasets, ymax)
    machine, fmm_note, caveats = build_caption_notes(
        datasets, args.machine, overrides, extra_fmm, notes, nrep, dnf_limit_s)
    write_ref_tex(reftex, datasets, nrep, machine, fmm_note, caveats, reference)
    build_figure(args.tag, datasets, fig, ymin, ymax, yticks, ymax, span)
    write_table(tab, datasets, dnf_limit_s)

    labels = {k: lbl for k, _, lbl, _, _ in METHODS}
    for ds in datasets:
        print("\n%s  (reference: %s)" % (ds["molecule"],
                                         ds.get("ref_method") or "none"))
        print("  %-20s %6s %12s %12s %12s %12s" % (
            "method", "runs", "stage s", "vs CPU", "rel err pol", "rel err ion"))
        cpu = ds["methods"].get("cpu")
        cpu_t = cpu["med"] if cpu and not cpu["dnf"] else None
        for key, _, _, _, _ in METHODS:
            m = ds["methods"].get(key)
            if not m:
                continue
            if m["dnf"]:
                print("  %-20s %6s %12s   (killed; %s accounted for elsewhere)"
                      % (labels[key], "0",
                         ">%.0f" % m["floor"] if m.get("floor") else "DNF",
                         "%.0f s" % m.get("accounted", 0.0)))
                continue
            print("  %-20s %6d %12.4f %12s %12s %12s" % (
                labels[key], m["n"], m["med"],
                "%.1fx" % (cpu_t / m["med"]) if cpu_t else "--",
                "%.3g" % m["err_pol"] if m.get("err_pol") else "--",
                "%.3g" % m["err_ionic"] if m.get("err_ionic") else "--"))

    print("\nwrote %s\n      %s\n      %s\n      %s" % (dat, reftex, fig, tab))
    print("\n\\input{figures/%s}\n\\input{figures/%s_table}" % (args.tag, args.tag))


if __name__ == "__main__":
    sys.exit(main())

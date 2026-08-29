#!/usr/bin/env python3
"""
Parameter sweep driver for NextGenPB.

Runs `mpirun -n <np> ngpb --prmfile options.prm` repeatedly inside one or more
test folders, varying the [algorithm] section of options.prm, and records every
per-stage timing plus the final energies into a CSV (one row per run).

Two sweeps, never crossed:

  sweep A (linear solver)   energy_method pinned to 1
      linear_solver = amgx, amgx_config = <path>
      linear_solver = amgx, amgx_config omitted
      linear_solver = lis

  sweep B (energy method)   linear_solver pinned to amgx + amgx_config
      energy_method = 0
      energy_method = 1
      energy_method = 2 (at whatever fmm_* values options.prm already carries;
                         the fmm_* grid lives in --fmm-sweep, see below)

The --fmm-sweep grid is fmm_mac x fmm_multipole_order x fmm_leaf_size (SOURCE
tree) x fmm_target_leaf_size (TARGET tree). The two leaf sizes are independent
knobs pulling opposite ways -- the source leaf bounds the box radius the MAC
tests against, hence the near-field/P2P work; the target leaf sets the target
box count, hence the M2L pair count. Sweeping them on one value (what this
script did before fmm_target_leaf_size existed) only samples the diagonal.

A single reference run (linear_solver = lis, energy_method = 0) is done first;
relative errors of every other run's energies are computed against it.

options.prm is backed up before the first run and restored afterwards, including
on Ctrl-C or an unhandled error.  The backup is only deleted once the restore has
succeeded, so a hard kill still leaves options.prm.bench_backup on disk --
recover it with `--restore`.
"""

import argparse
import atexit
import csv
import json
import math
import os
import re
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

PRM = "options.prm"
BACKUP = "options.prm.bench_backup"

# Keys this script owns inside [algorithm]. Anything else in that section is
# passed through untouched.
MANAGED_KEYS = [
    "linear_solver",
    "amgx_config",
    "energy_method",
    "fmm_mac",
    "fmm_multipole_order",
    "fmm_leaf_size",
    "fmm_target_leaf_size",
    "energy_dump",
]

# TOC() label -> CSV column. These are the stages bim_timing.h reports.
STAGE_COLUMNS = [
    ("create_mesh", "t_create_mesh"),
    ("Building Surface with NanoShaper", "t_surface"),
    ("Building Grid", "t_build_grid"),
    ("refine the box", "t_refine_box"),
    ("create element markers", "t_element_markers"),
    ("create density map", "t_density_map"),
    ("Assemble system matrix", "t_assemble"),
    ("Compute numerical solution", "t_solve"),
    ("Write potential on atoms", "t_potential_on_atoms"),
    ("Compute energy", "t_energy"),
    ("Write potential on the surface", "t_potential_on_surface"),
    ("export potential map", "t_export_potential_map"),
    ("export epsilon map", "t_export_epsilon_map"),
    ("export potential map new", "t_export_potential_map_new"),
    ("export epsilon map new", "t_export_epsilon_map_new"),
]
STAGE_BY_LABEL = dict(STAGE_COLUMNS)

ENERGY_KEYS = ["energy_pol", "energy_ionic", "energy_coul", "energy_sum"]

FIELDNAMES = (
    [
        "folder", "molecule", "sweep", "config_id", "repeat", "status",
        "exit_code", "timestamp", "np",
        "linear_solver", "amgx_config", "energy_method",
        "fmm_mac", "fmm_multipole_order", "fmm_leaf_size", "fmm_target_leaf_size",
        "wall_s", "t_report_total_s",
    ]
    + [col for _, col in STAGE_COLUMNS]
    + [
        "net_charge", "flux_charge",
        "energy_pol", "energy_ionic", "energy_coul", "energy_sum",
        "relerr_pol", "relerr_ionic", "relerr_coul", "relerr_sum",
        "solver_iters", "solver_final_residual",
        "amgx_total_s", "amgx_setup_s", "amgx_solve_s", "amgx_per_iter_s",
        "amgx_reduction", "amgx_max_mem_gb", "amgx_levels",
        "matrix_n", "matrix_nnz", "num_atoms", "grid_nx", "grid_ny", "grid_nz",
        "log_file",
    ]
)


# --------------------------------------------------------------------------
# options.prm rewriting
# --------------------------------------------------------------------------

def render_prm(original_text, settings):
    """Return options.prm content with [algorithm] rewritten from `settings`.

    settings maps managed key -> value, or key -> None to omit the key entirely
    (that is how "amgx without a config file" is expressed).  Unmanaged keys in
    [algorithm] and every other section are preserved verbatim.
    """
    lines = original_text.splitlines(keepends=True)
    out = []
    i = 0
    n = len(lines)
    seen_algorithm = False

    while i < n:
        if lines[i].strip() == "[algorithm]":
            seen_algorithm = True
            out.append(lines[i])
            i += 1
            body = []
            while i < n and lines[i].strip() != "[../]":
                body.append(lines[i])
                i += 1
            for raw in body:
                stripped = raw.strip()
                if not stripped:
                    continue
                key = stripped.split("=", 1)[0].strip() if "=" in stripped else None
                if key in MANAGED_KEYS:
                    continue
                out.append(raw if raw.endswith("\n") else raw + "\n")
            for key in MANAGED_KEYS:
                value = settings.get(key)
                if value is None:
                    continue
                out.append("{:<19} = {}\n".format(key, value))
            if i < n:  # the [../] that closes the section
                out.append(lines[i])
                i += 1
            continue
        out.append(lines[i])
        i += 1

    if not seen_algorithm:
        block = ["\n[algorithm]\n"]
        for key in MANAGED_KEYS:
            value = settings.get(key)
            if value is not None:
                block.append("{:<19} = {}\n".format(key, value))
        block.append("[../]\n")
        out.extend(block)

    return "".join(out)


def read_prm_value(text, section, key):
    """Pull a single key out of a section, for reporting (e.g. the molecule)."""
    in_section = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "[{}]".format(section):
            in_section = True
            continue
        if in_section:
            if stripped == "[../]":
                break
            if "=" in stripped:
                k, v = stripped.split("=", 1)
                if k.strip() == key:
                    return v.split("#", 1)[0].strip()
    return ""


# --------------------------------------------------------------------------
# output parsing
# --------------------------------------------------------------------------

NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"

RE_TIMING_EVENT = re.compile(
    r"^Event:\s*(?P<label>.+?),\s*total hits:\s*(?P<hits>\d+),"
    r"\s*total time:\s*(?P<secs>" + NUM + r")\s*s\."
)
RE_ENERGY = {
    "net_charge": re.compile(r"^\s*Net charge \[e\]:\s*(" + NUM + r")"),
    "flux_charge": re.compile(r"^\s*Flux charge \[e\]:\s*(" + NUM + r")"),
    "energy_pol": re.compile(r"^\s*Polarization energy \[kT\]:\s*(" + NUM + r")"),
    "energy_ionic": re.compile(r"^\s*Direct ionic energy \[kT\]:\s*(" + NUM + r")"),
    "energy_coul": re.compile(r"^\s*Coulombic energy \[kT\]:\s*(" + NUM + r")"),
    "energy_sum": re.compile(
        r"^\s*Sum of electrostatic energy contributions \[kT\]:\s*(" + NUM + r")"
    ),
}
RE_AMGX_ITERS = re.compile(r"Total Iterations:\s*(\d+)")
RE_AMGX_FINAL_RES = re.compile(r"Final Residual:\s*(" + NUM + r")")
RE_AMGX_REDUCTION = re.compile(r"Total Reduction in Residual:\s*(" + NUM + r")")
RE_AMGX_MAXMEM = re.compile(r"Maximum Memory Usage:\s*(" + NUM + r")\s*GB")
RE_AMGX_LEVELS = re.compile(r"Number of Levels:\s*(\d+)")
RE_AMGX_TOTAL = re.compile(r"^Total Time:\s*(" + NUM + r")")
RE_AMGX_SETUP = re.compile(r"^\s*setup:\s*(" + NUM + r")\s*s")
RE_AMGX_SOLVE = re.compile(r"^\s*solve:\s*(" + NUM + r")\s*s")
RE_AMGX_PERIT = re.compile(r"^\s*solve\(per iteration\):\s*(" + NUM + r")\s*s")
RE_LIS_ITER = re.compile(
    r"^\s*iteration:\s*(\d+)\s*relative residual\s*=\s*(" + NUM + r")"
)
RE_MATRIX = re.compile(
    r"Sparse matrix size:\s*n\s*=\s*(\d+),\s*nnz(?:\s*\(rank 0\))?\s*=\s*(\d+)"
)
RE_ATOMS = re.compile(r"Number of atoms\s*:\s*(\d+)")
RE_GRID = re.compile(r"nx\s*=\s*(\d+)\s+ny\s*=\s*(\d+)\s+nz\s*=\s*(\d+)")


def parse_log(text):
    """Extract every metric we care about from one run's combined stdout/stderr."""
    res = {}
    stage_totals = {}
    report_total = 0.0

    for line in text.splitlines():
        m = RE_TIMING_EVENT.match(line)
        if m:
            label = m.group("label").strip()
            secs = float(m.group("secs"))
            report_total += secs
            col = STAGE_BY_LABEL.get(label)
            if col:
                stage_totals[col] = secs
            else:
                # Unknown stage: keep it visible rather than silently dropping it.
                stage_totals.setdefault("_unknown", {})
                stage_totals["_unknown"][label] = secs
            continue

        for key, rx in RE_ENERGY.items():
            m = rx.match(line)
            if m:
                res[key] = float(m.group(1))
                break
        else:
            for rx, key, cast in (
                (RE_AMGX_ITERS, "solver_iters", int),
                (RE_AMGX_FINAL_RES, "solver_final_residual", float),
                (RE_AMGX_REDUCTION, "amgx_reduction", float),
                (RE_AMGX_MAXMEM, "amgx_max_mem_gb", float),
                (RE_AMGX_LEVELS, "amgx_levels", int),
                (RE_AMGX_TOTAL, "amgx_total_s", float),
                (RE_AMGX_SETUP, "amgx_setup_s", float),
                (RE_AMGX_SOLVE, "amgx_solve_s", float),
                (RE_AMGX_PERIT, "amgx_per_iter_s", float),
            ):
                m = rx.search(line)
                if m:
                    res[key] = cast(m.group(1))
                    break
            else:
                m = RE_LIS_ITER.match(line)
                if m:
                    # LIS prints one line per iteration; the last one wins.
                    res["solver_iters"] = int(m.group(1))
                    res["solver_final_residual"] = float(m.group(2))
                    continue
                m = RE_MATRIX.search(line)
                if m:
                    res["matrix_n"] = int(m.group(1))
                    res["matrix_nnz"] = int(m.group(2))
                    continue
                m = RE_ATOMS.search(line)
                if m:
                    res["num_atoms"] = int(m.group(1))
                    continue
                m = RE_GRID.search(line)
                if m:
                    res["grid_nx"] = int(m.group(1))
                    res["grid_ny"] = int(m.group(2))
                    res["grid_nz"] = int(m.group(3))

    unknown = stage_totals.pop("_unknown", None)
    res.update(stage_totals)
    if report_total:
        res["t_report_total_s"] = report_total
    if unknown:
        res["_unknown_stages"] = unknown
    return res


def relerr(value, reference):
    if value is None or reference is None:
        return ""
    if not math.isfinite(value) or not math.isfinite(reference):
        return ""
    if reference == 0.0:
        return "" if value == 0.0 else "inf"
    return abs(value - reference) / abs(reference)


# --------------------------------------------------------------------------
# run plan
# --------------------------------------------------------------------------

def build_plan(amgx_config, sweeps, original_text=""):
    """Return a list of (sweep_name, settings dict) in execution order.

    original_text is the untouched options.prm; sweep B's energy_method=2 entry
    is run at the fmm_* values it already carries.  render_prm strips every
    managed key from the section before writing the new ones, so passing None
    here does not preserve a key -- it deletes it and lets ngpb fall back to its
    compiled default, which is not the same configuration the file describes.
    """
    plan = []

    if "a" in sweeps:
        for solver, cfg in (
            ("amgx", amgx_config),
            ("amgx", None),
            ("lis", None),
        ):
            plan.append(("A", {
                "linear_solver": solver,
                "amgx_config": cfg,
                "energy_method": 1,
                "fmm_mac": None,
                "fmm_multipole_order": None,
                "fmm_leaf_size": None,
                "fmm_target_leaf_size": None,
            }))

    if "b" in sweeps:
        # energy_method only. The fmm_mac x fmm_multipole_order x fmm_leaf_size
        # grid used to live here, at 140 configurations x N repeats of a FULL
        # pipeline run each -- ~18 h on 6VYB to measure a stage that is under 4%
        # of the wall time. src/tools/fmm_replay.cu does that grid in minutes by
        # replaying the energy stage from a dump; see its header comment.
        # energy_method=2 is still run once, at whatever fmm_* values options.prm
        # already carries, so the three methods stay comparable in one table.
        base = {"linear_solver": "amgx", "amgx_config": amgx_config}

        # Carried through verbatim so the energy_method=2 row is measured at the
        # configuration options.prm actually specifies. Absent keys stay absent.
        fmm_now = {k: (read_prm_value(original_text, "algorithm", k) or None)
                   for k in ("fmm_mac", "fmm_multipole_order",
                             "fmm_leaf_size", "fmm_target_leaf_size")}

        for method in (0, 1, 2):
            # The fmm_* keys are meaningless for methods 0 and 1 and would only
            # clutter the config_id, so they ride along on method 2 alone.
            extra = fmm_now if method == 2 else dict.fromkeys(fmm_now, None)
            plan.append(("B", dict(base, energy_method=method, **extra)))
    return plan


def config_id(sweep, settings):
    parts = [sweep]
    for key in MANAGED_KEYS:
        value = settings.get(key)
        if value is None:
            continue
        if key == "amgx_config":
            value = os.path.basename(str(value))
        parts.append("{}={}".format(key, value))
    return "|".join(parts)


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------

class PrmGuard:
    """Backs up options.prm and restores it on any exit path we can catch."""

    def __init__(self, folder):
        self.prm = os.path.join(folder, PRM)
        self.backup = os.path.join(folder, BACKUP)
        self.original = None
        self.mode = None
        self.active = False

    def acquire(self):
        if os.path.exists(self.backup):
            raise SystemExit(
                "{} already exists -- a previous sweep did not restore cleanly.\n"
                "Inspect it, then run with --restore to put it back (or delete it)."
                .format(self.backup)
            )
        with open(self.prm, "rb") as fh:
            self.original = fh.read()
        self.mode = os.stat(self.prm).st_mode
        with open(self.backup, "wb") as fh:
            fh.write(self.original)
        os.chmod(self.backup, self.mode)
        self.active = True

    def restore(self):
        if not self.active:
            return
        with open(self.prm, "wb") as fh:
            fh.write(self.original)
        os.chmod(self.prm, self.mode)
        self.active = False
        # Only now is it safe to drop the backup.
        if os.path.exists(self.backup):
            os.remove(self.backup)

    def write(self, text):
        with open(self.prm, "w") as fh:
            fh.write(text)
        os.chmod(self.prm, self.mode)


def run_once(folder, np_ranks, timeout_s, env, launcher=()):
    # The launcher wraps the WHOLE command, mpirun included -- that is the
    # container idiom the cluster jobs already use ("singularity exec --nv SIF
    # mpirun -n N ngpb"), where the MPI doing the launching must be the one
    # inside the image, not the host's.
    cmd = list(launcher) + ["mpirun", "-n", str(np_ranks), "ngpb", "--prmfile", PRM]
    start = time.perf_counter()
    proc = subprocess.Popen(
        cmd, cwd=folder, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, errors="replace", start_new_session=True, env=env,
    )
    try:
        out, _ = proc.communicate(timeout=timeout_s)
        status = "ok" if proc.returncode == 0 else "fail"
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except OSError:
            pass
        out, _ = proc.communicate()
        status = "timeout"
    wall = time.perf_counter() - start
    return status, proc.returncode, wall, out or ""


def main():
    ap = argparse.ArgumentParser(
        description="Sweep NextGenPB solver / energy-method parameters and log timings to CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("folders", nargs="+",
                    help="test folders to run in (each holds one molecule's options.prm)")
    ap.add_argument("-n", "--np", type=int, default=4,
                    help="MPI ranks per run (default: 4)")
    ap.add_argument("-r", "--repeats", type=int, default=10,
                    help="repeats per configuration (default: 10)")
    ap.add_argument("--sweeps", default="ab", choices=["a", "b", "ab"],
                    help="which sweeps to run: a=linear solver, b=energy method (default: ab)")
    ap.add_argument("--amgx-config", default=None,
                    help="path to amgx_pcgf_amg_block_jacobi.json "
                         "(default: the copy in the test folder, else data/)")
    ap.add_argument("-o", "--out", default=None,
                    help="raw CSV path (default: <folder>/bench_runs.csv)")
    ap.add_argument("--timeout", type=float, default=3600.0,
                    help="per-run timeout in seconds (default: 3600)")
    ap.add_argument("--keep-logs", default="failed",
                    choices=["none", "failed", "first", "all"],
                    help="which run logs to keep on disk (default: failed)")
    ap.add_argument("--resume", action="store_true",
                    help="skip configurations already completed in the existing CSV")
    ap.add_argument("--limit", type=int, default=None,
                    help="run only the first N configurations of the plan "
                         "(smoke-testing a long sweep)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the run plan and exit without running anything")
    ap.add_argument("--no-reference", action="store_true",
                    help="skip the LIS + energy_method=0 reference run "
                         "(relative errors are left blank)")
    ap.add_argument("--restore", action="store_true",
                    help="restore options.prm from a leftover backup and exit")
    ap.add_argument("--launcher", default="",
                    help="command prefix for every run, e.g. "
                         "\"singularity exec --nv --bind $SRC:/usr/local/nextgenPB ngpb.sif\". "
                         "Wraps mpirun too, so the container's own MPI does the launching; "
                         "the host-side mpirun/ngpb checks are skipped when this is set.")
    ap.add_argument("--replay-bin", default=None,
                    help="path to fmm_replay (default: PATH, then ../src/fmm_replay; "
                         "with --launcher, the bare name resolved inside the container)")

    fmm = ap.add_argument_group(
        "FMM parameter sweep",
        "One ngpb run dumps the phase-1 energy inputs, then src/tools/fmm_replay "
        "sweeps the fmm_* grid over that dump. Everything up to the energy stage "
        "is identical across the grid, so re-running it per configuration is "
        "wasted work -- on 6VYB that is ~96% of the wall time.")
    fmm.add_argument("--fmm-sweep", action="store_true",
                     help="run the dump-and-replay FMM sweep instead of sweeps a/b")
    fmm.add_argument("--fmm-mac", default="0.2:0.8:0.1",
                     help="fmm_mac spec passed to fmm_replay (default: 0.2:0.8:0.1)")
    fmm.add_argument("--fmm-order", default="9:11",
                     help="fmm_multipole_order spec (default: 9:11). 12 is the compiled "
                          "maximum but REGRESSES: p=12 measured slower AND less accurate "
                          "than p=11 (unscaled M2L conditioning), so 11 is the ceiling")
    fmm.add_argument("--fmm-leaf", default="8,16,32",
                     help="fmm_leaf_size (SOURCE tree) spec (default: 8,16,32; "
                          "16 is the measured optimum, interior)")
    fmm.add_argument("--fmm-tleaf", default="512,1024,2048",
                     help="fmm_target_leaf_size spec; 0 follows --fmm-leaf "
                          "(default: 512,1024,2048; 1024 is the measured optimum, "
                          "interior). Capped at FMM_MAX_TLEAF by the energy entry points")
    fmm.add_argument("--fmm-pairs", default=None,
                     help="explicit source/target leaf pairs, e.g. 16/1024,16/512. "
                          "REPLACES --fmm-leaf x --fmm-tleaf. Use this to sweep mac "
                          "and order at the leaf combinations that already won, "
                          "instead of re-crossing every leaf pair against them")
    fmm.add_argument("--fmm-repeats", type=int, default=3,
                     help="timed repeats per configuration (default: 3)")

    args = ap.parse_args()
    args.launcher = shlex.split(args.launcher)

    if args.restore:
        for folder in args.folders:
            backup = os.path.join(folder, BACKUP)
            if os.path.exists(backup):
                shutil.move(backup, os.path.join(folder, PRM))
                print("restored {}".format(os.path.join(folder, PRM)))
            else:
                print("no backup in {}".format(folder))
        return 0

    # Both live inside the image when a launcher is set; nothing on the host
    # PATH is expected to match, so there is nothing worth pre-checking.
    if not args.launcher:
        if shutil.which("mpirun") is None:
            return _die("mpirun not found on PATH")
        if shutil.which("ngpb") is None:
            return _die("ngpb not found on PATH")

    for folder in args.folders:
        rc = run_fmm_sweep(folder, args) if args.fmm_sweep else run_folder(folder, args)
        if rc:
            return rc
    return 0


def _die(msg):
    print("error: " + msg, file=sys.stderr)
    return 1


def find_fmm_replay():
    """Locate the replay binary: PATH first, then this repo's src/ build."""
    found = shutil.which("fmm_replay")
    if found:
        return found

    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(os.path.dirname(here), "src", "fmm_replay")

    return candidate if os.path.exists(candidate) else None


def run_fmm_sweep(folder, args):
    """One ngpb run to dump the phase-1 energy inputs, then replay the FMM grid.

    The dump files are left in the folder on purpose: re-sweeping a different
    grid, or re-running at a different repeat count, needs no further ngpb run.
    """
    folder = os.path.abspath(folder)
    prm_path = os.path.join(folder, PRM)

    if not os.path.exists(prm_path):
        return _die("no {} in {}".format(PRM, folder))

    if args.replay_bin:
        replay = args.replay_bin
    elif args.launcher:
        # Resolved on the container's PATH: the image puts src/ there, and a
        # host-side absolute path would only be valid inside if the bind mount
        # happened to land at the same location.
        replay = "fmm_replay"
    else:
        replay = find_fmm_replay()
        if replay is None:
            return _die("fmm_replay not found -- build it with 'make -C src fmm_replay'")

    with open(prm_path) as fh:
        original_text = fh.read()

    molecule = os.path.basename(read_prm_value(original_text, "input", "filename"))
    amgx_config = resolve_amgx_config(folder, args.amgx_config)
    # Absolute, because fmm_replay is spawned with cwd=folder: a relative --out
    # given on the command line is relative to where the USER stands, not to the
    # test folder, and would otherwise resolve to <folder>/<folder>/name.csv.
    out_csv = os.path.abspath(args.out) if args.out \
        else os.path.join(folder, "fmm_sweep.csv")
    prefix = "fmm_inputs"

    # energy_method=2 so the dump run also produces one FMM evaluation to eyeball;
    # the fmm_* values it uses are irrelevant, the dump is taken before they apply.
    settings = {
        "linear_solver": "amgx",
        "amgx_config": amgx_config,
        "energy_method": 2,
        "fmm_mac": None,
        "fmm_multipole_order": None,
        "fmm_leaf_size": None,
        "fmm_target_leaf_size": None,
        "energy_dump": prefix,
    }

    # --pairs replaces both leaf axes rather than adding a third, so the two are
    # mutually exclusive on the command line as well.
    if args.fmm_pairs:
        leaf_args = ["--pairs", args.fmm_pairs]
    else:
        leaf_args = ["--leaf", args.fmm_leaf, "--tleaf", args.fmm_tleaf]

    replay_cmd = list(args.launcher) + \
                 ["mpirun", "-n", str(args.np), replay, prefix,
                  "--mac", args.fmm_mac, "--order", args.fmm_order] + \
                 leaf_args + \
                 ["--repeats", str(args.fmm_repeats),
                  "--csv", out_csv]

    if args.dry_run:
        print("folder     : {}".format(folder))
        print("molecule   : {}".format(molecule))
        print("replay     : {}".format(replay))
        print("csv        : {}".format(out_csv))
        if args.launcher:
            print("launcher   : {}".format(" ".join(args.launcher)))
        print("step 1     : {}".format(" ".join(
            list(args.launcher) + ["mpirun", "-n", str(args.np), "ngpb",
                                   "--prmfile", PRM])))
        print("             (with energy_dump = {})".format(prefix))
        print("step 2     : {}".format(" ".join(replay_cmd)))
        return 0

    env = dict(os.environ)
    guard = PrmGuard(folder)
    guard.acquire()
    atexit.register(guard.restore)

    def on_signal(signum, _frame):
        guard.restore()
        print("\ninterrupted -- {} restored".format(prm_path), file=sys.stderr)
        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    try:
        print("[1/2] dumping energy inputs ...", end=" ", flush=True)
        guard.write(render_prm(original_text, settings))
        status, code, wall, out = run_once(folder, args.np, args.timeout, env,
                                           args.launcher)
        print("{} ({:.1f}s)".format(status, wall))

        if status != "ok":
            log = os.path.join(folder, "fmm_dump_failed.log")
            with open(log, "w") as fh:
                fh.write(out)
            return _die("dump run failed (exit {}); see {}".format(code, log))

        for line in out.splitlines():
            if "[energy_dump]" in line:
                print("      " + line.strip())
    finally:
        guard.restore()

    print("\n[2/2] " + " ".join(replay_cmd))
    rc = subprocess.call(replay_cmd, cwd=folder, env=env)

    if rc != 0:
        return _die("fmm_replay exited {}".format(rc))

    print("\nresults: {}".format(out_csv))
    print("plot   : python3 scripts/plot_fmm_sweep.py {} --csv {}"
          .format(folder, out_csv))
    return 0


def resolve_amgx_config(folder, override):
    if override:
        return os.path.abspath(override)
    local = os.path.join(folder, "amgx_pcgf_amg_block_jacobi.json")
    if os.path.exists(local):
        # Relative on purpose: ngpb runs with cwd = folder, and this keeps the
        # CSV readable when the same sweep is repeated elsewhere.
        return "amgx_pcgf_amg_block_jacobi.json"
    shared = "/usr/local/nextgenPB/data/amgx_pcgf_amg_block_jacobi.json"
    if os.path.exists(shared):
        return shared
    return None


def run_folder(folder, args):
    folder = os.path.abspath(folder)
    prm_path = os.path.join(folder, PRM)
    if not os.path.exists(prm_path):
        return _die("no {} in {}".format(PRM, folder))

    with open(prm_path) as fh:
        original_text = fh.read()
    molecule = os.path.basename(read_prm_value(original_text, "input", "filename"))

    amgx_config = resolve_amgx_config(folder, args.amgx_config)
    if amgx_config is None and args.sweeps != "":
        return _die("no amgx_pcgf_amg_block_jacobi.json found for {} "
                    "(pass --amgx-config)".format(folder))

    plan = build_plan(amgx_config, args.sweeps, original_text)
    if args.limit is not None:
        plan = plan[:args.limit]
    out_csv = args.out or os.path.join(folder, "bench_runs.csv")
    log_dir = os.path.join(folder, "bench_logs")

    if args.dry_run:
        print("folder     : {}".format(folder))
        print("molecule   : {}".format(molecule))
        print("amgx config: {}".format(amgx_config))
        print("csv        : {}".format(out_csv))
        print("configs    : {}  x {} repeats = {} runs"
              .format(len(plan), args.repeats, len(plan) * args.repeats))
        if not args.no_reference:
            print("             + 1 reference run (lis, energy_method=0)")
        for sweep, settings in plan:
            print("  " + config_id(sweep, settings))
        return 0

    done = Counter()
    if args.resume and os.path.exists(out_csv):
        with open(out_csv, newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("status") == "ok" and row.get("folder") == folder:
                    done[row.get("config_id", "")] += 1
        print("resume: {} configurations have prior successful runs"
              .format(len([k for k, v in done.items() if v])))

    env = dict(os.environ)
    guard = PrmGuard(folder)
    guard.acquire()
    atexit.register(guard.restore)

    def on_signal(signum, _frame):
        guard.restore()
        print("\ninterrupted -- {} restored".format(prm_path), file=sys.stderr)
        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    if args.keep_logs != "none":
        os.makedirs(log_dir, exist_ok=True)

    new_file = not os.path.exists(out_csv)
    if not new_file:
        # The header is only written for a new file, so appending to a CSV produced by an
        # older schema (before fmm_target_leaf_size split off fmm_leaf_size) would write
        # every subsequent value one column to the left -- silently, and only detectable
        # later as nonsense timings. Compare and refuse instead.
        with open(out_csv, newline="") as fh:
            existing = next(csv.reader(fh), None)
        if existing is not None and existing != FIELDNAMES:
            missing = [c for c in FIELDNAMES if c not in existing]
            print("error: {} was written with a different column set; appending would "
                  "misalign\n       every row. Missing here: {}\n"
                  "       Move it aside, or pass -o with a new path."
                  .format(out_csv, ", ".join(missing) or "(column order differs)"),
                  file=sys.stderr)
            sys.exit(1)

    csv_fh = open(out_csv, "a", newline="")
    writer = csv.DictWriter(csv_fh, fieldnames=FIELDNAMES, extrasaction="ignore")
    if new_file:
        writer.writeheader()
        csv_fh.flush()

    write_meta(folder, molecule, amgx_config, args)

    reference = None
    total_runs = len(plan) * args.repeats
    counter = 0
    t_start = time.time()

    try:
        if not args.no_reference:
            print("[reference] lis + energy_method=0 ...", end=" ", flush=True)
            ref_settings = {
                "linear_solver": "lis", "amgx_config": None, "energy_method": 0,
                "fmm_mac": None, "fmm_multipole_order": None, "fmm_leaf_size": None,
            }
            row, parsed = execute(guard, folder, original_text, ref_settings,
                                  "REF", 0, args, env, log_dir, molecule, None)
            writer.writerow(row)
            csv_fh.flush()
            print("{} ({:.1f}s)".format(row["status"], row["wall_s"]))
            if row["status"] == "ok":
                reference = {k: parsed.get(k) for k in ENERGY_KEYS}
                print("            reference energies: " + ", ".join(
                    "{}={:.12g}".format(k, v) for k, v in reference.items()
                    if v is not None))
            else:
                print("            reference FAILED -- relative errors will be blank",
                      file=sys.stderr)

        for sweep, settings in plan:
            cid = config_id(sweep, settings)
            already = done.get(cid, 0)
            for rep in range(args.repeats):
                counter += 1
                if rep < already:
                    continue
                label = "[{:>{w}}/{}] {}".format(counter, total_runs, cid,
                                                 w=len(str(total_runs)))
                print("{} rep {} ...".format(label, rep), end=" ", flush=True)
                row, _ = execute(guard, folder, original_text, settings, sweep,
                                 rep, args, env, log_dir, molecule, reference)
                writer.writerow(row)
                csv_fh.flush()
                print("{} {:.2f}s".format(row["status"], row["wall_s"]))
    finally:
        csv_fh.close()
        guard.restore()

    elapsed = time.time() - t_start
    print("\ndone in {:.0f}s -- {} restored".format(elapsed, prm_path))
    print("raw results: {}".format(out_csv))
    summary_csv = os.path.splitext(out_csv)[0] + "_summary.csv"
    write_summary(out_csv, summary_csv)
    print("summary    : {}".format(summary_csv))
    return 0


def execute(guard, folder, original_text, settings, sweep, rep, args, env,
            log_dir, molecule, reference):
    guard.write(render_prm(original_text, settings))
    status, code, wall, out = run_once(folder, args.np, args.timeout, env,
                                       args.launcher)
    parsed = parse_log(out)

    cid = config_id(sweep, settings)
    log_file = ""
    keep = (
        args.keep_logs == "all"
        or (args.keep_logs == "failed" and status != "ok")
        or (args.keep_logs == "first" and rep == 0)
        or (args.keep_logs == "failed" and sweep == "REF")
    )
    if keep and args.keep_logs != "none":
        safe = re.sub(r"[^A-Za-z0-9._=-]+", "_", cid)
        log_file = os.path.join(log_dir, "{}__rep{}.log".format(safe, rep))
        with open(log_file, "w") as fh:
            fh.write(out)

    row = {
        "folder": folder,
        "molecule": molecule,
        "sweep": sweep,
        "config_id": cid,
        "repeat": rep,
        "status": status,
        "exit_code": code,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "np": args.np,
        "wall_s": round(wall, 4),
        "log_file": os.path.relpath(log_file, folder) if log_file else "",
    }
    for key in MANAGED_KEYS:
        value = settings.get(key)
        row[key] = "" if value is None else value
    for key, value in parsed.items():
        if key.startswith("_"):
            continue
        row[key] = value
    if reference:
        for key in ENERGY_KEYS:
            row["relerr_" + key.replace("energy_", "")] = relerr(
                parsed.get(key), reference.get(key))

    unknown = parsed.get("_unknown_stages")
    if unknown:
        print("\n  note: unrecognised timing stage(s) {} -- not in the CSV schema"
              .format(sorted(unknown)), file=sys.stderr)
    return row, parsed


def write_meta(folder, molecule, amgx_config, args):
    meta = {
        "host": socket.gethostname(),
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "folder": folder,
        "molecule": molecule,
        "amgx_config": amgx_config,
        "np": args.np,
        "repeats": args.repeats,
        "sweeps": args.sweeps,
    }
    try:
        meta["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=folder, capture_output=True,
            text=True, check=True).stdout.strip()
        meta["git_dirty"] = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=folder, capture_output=True,
            text=True, check=True).stdout.strip())
    except Exception:
        pass
    try:
        meta["gpus"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"], capture_output=True, text=True,
            check=True).stdout.strip().splitlines()
    except Exception:
        pass
    with open(os.path.join(folder, "bench_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------

TIMING_COLS = ["wall_s", "t_report_total_s"] + [col for _, col in STAGE_COLUMNS]


def write_summary(raw_csv, summary_csv):
    """Aggregate repeats per configuration: median/mean/std of timings, energies,
    and relative errors -- both against the LIS reference and, for sweep B,
    against that sweep's own energy_method=0 run."""
    rows = []
    with open(raw_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") == "ok":
                rows.append(row)
    if not rows:
        return

    groups = defaultdict(list)
    for row in rows:
        groups[(row["folder"], row["config_id"])].append(row)

    # Per-folder sweep-B baseline (energy_method=0, same solver settings).
    em0_baseline = {}
    for (folder, cid), grp in groups.items():
        if grp[0]["sweep"] == "B" and grp[0]["energy_method"] == "0":
            em0_baseline[folder] = {
                k: _mean_of(grp, k) for k in ENERGY_KEYS
            }

    fields = ["folder", "molecule", "sweep", "config_id",
              "linear_solver", "amgx_config", "energy_method",
              "fmm_mac", "fmm_multipole_order", "fmm_leaf_size", "n_runs"]
    for col in TIMING_COLS:
        fields += [col + "_median", col + "_mean", col + "_std", col + "_min"]
    fields += ["solver_iters_median", "amgx_setup_s_median", "amgx_solve_s_median"]
    for key in ENERGY_KEYS:
        fields += [key + "_mean", key + "_spread"]
    for key in ENERGY_KEYS:
        short = key.replace("energy_", "")
        fields += ["relerr_" + short + "_mean", "relerr_" + short + "_vs_em0"]

    with open(summary_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for (folder, cid), grp in sorted(groups.items(), key=lambda kv: kv[0][1]):
            first = grp[0]
            out = {k: first.get(k, "") for k in fields[:10]}
            out["n_runs"] = len(grp)
            for col in TIMING_COLS:
                vals = _floats(grp, col)
                if not vals:
                    continue
                out[col + "_median"] = statistics.median(vals)
                out[col + "_mean"] = statistics.fmean(vals)
                out[col + "_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
                out[col + "_min"] = min(vals)
            for col in ("solver_iters", "amgx_setup_s", "amgx_solve_s"):
                vals = _floats(grp, col)
                if vals:
                    out[col + "_median"] = statistics.median(vals)
            base = em0_baseline.get(folder, {})
            for key in ENERGY_KEYS:
                vals = _floats(grp, key)
                short = key.replace("energy_", "")
                if vals:
                    out[key + "_mean"] = statistics.fmean(vals)
                    out[key + "_spread"] = max(vals) - min(vals)
                    out["relerr_" + short + "_vs_em0"] = relerr(
                        statistics.fmean(vals), base.get(key))
                errs = _floats(grp, "relerr_" + short)
                if errs:
                    out["relerr_" + short + "_mean"] = statistics.fmean(errs)
            writer.writerow(out)


def _floats(rows, key):
    out = []
    for row in rows:
        raw = row.get(key, "")
        if raw in ("", None):
            continue
        try:
            out.append(float(raw))
        except ValueError:
            pass
    return out


def _mean_of(rows, key):
    vals = _floats(rows, key)
    return statistics.fmean(vals) if vals else None


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
cif2pqr.py -- direct mmCIF -> PQR converter for NGPB, built for LARGE structures.

Why this exists
---------------
pdb2pqr 3.6.1 (the apt `python3-pdb2pqr`) round-trips mmCIF through fixed-column
legacy PDB internally, so it silently drops/corrupts any structure with
>99,999 atoms or multi-character chain IDs (ribosomes, capsids). This bypasses
that entirely: it parses the `_atom_site` loop directly (whitespace mmCIF, no
size limit, any chain-ID width) and assigns charge/radius from pdb2pqr's own
forcefield tables, emitting a header-free, whitespace-delimited PQR that NGPB's
`operator>>` reads directly.

Scope / honesty
---------------
Intended for the naive-vs-FMM TIMING / scaling study, where the result depends
on N and geometry -- NOT on exact protonation. This does NOT add hydrogens or
compute pKa, so energies from these large rungs are APPROXIMATE (heavy-atom
charges, standard protonation). Fine for timing/crossover; not publication-grade
solvation numbers. For small rungs where you want accurate energies too, keep
using pdb2pqr proper.

Column mapping is read from the `_atom_site.` header, so it works for any mmCIF,
not just 4V6X.

Usage
-----
    python3 cif2pqr.py input.cif output.pqr [--ff AMBER|PARSE|CHARMM]

Output is already header-free ATOM records -- no need to grep for '^ATOM'.
"""
import sys
import os
import argparse

DAT_DIR = "/usr/lib/python3/dist-packages/pdb2pqr/dat"

# element -> default radius (Angstrom, Bondi-ish) for atoms absent from the FF table
ELEM_RADIUS = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "P": 1.80, "S": 1.80,
    "MG": 1.73, "NA": 2.27, "K": 2.75, "CL": 1.75, "ZN": 1.39,
    "FE": 1.80, "CA": 2.31, "MN": 1.79, "F": 1.47,
}
DEFAULT_RADIUS = 1.70

# mmCIF single-letter RNA comp_id -> AMBER.DAT residue naming (DNA DA/DT/DG/DC already match)
RES_ALIAS = {"A": "RA", "U": "RU", "G": "RG", "C": "RC"}

# modern PDBv3/mmCIF atom names -> AMBER.DAT naming (tried when the direct key misses).
# The phosphate oxygens carry the nucleic-acid backbone charge, so this matters a lot.
ATOM_ALIAS = {"OP1": "O1P", "OP2": "O2P", "OP3": "O3P", "OXT": "OXT"}

DROP_RES = {"HOH", "WAT", "DOD"}   # waters: continuum solvent is implicit


def unquote(tok):
    if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in "'\"":
        return tok[1:-1]
    return tok


def load_ff(ff):
    """Load pdb2pqr's `RES ATOM CHARGE RADIUS TYPE` table -> {(res, atom): (q, r)}."""
    path = os.path.join(DAT_DIR, ff.upper() + ".DAT")
    table = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            f = line.split()
            if len(f) < 4:
                continue
            try:
                table[(f[0], f[1])] = (float(f[2]), float(f[3]))
            except ValueError:
                continue
    return table


def convert(cif_path, pqr_path, ff):
    table = load_ff(ff)
    cols = []          # ordered _atom_site.<field> names
    idx = None         # field name -> column index (built once header ends)
    serial = 0
    matched = 0
    fallback = 0

    def col(name, *alts):
        for n in (name,) + alts:
            if n in idx:
                return idx[n]
        return None

    with open(cif_path) as fh, open(pqr_path, "w") as out:
        for line in fh:
            s = line.strip()

            if s.startswith("_atom_site."):
                cols.append(s.split(".", 1)[1].split()[0])
                continue

            # first non-header line after the header block: finalize the index map
            if cols and idx is None:
                idx = {name: i for i, name in enumerate(cols)}
                c_group = col("group_PDB")
                c_elem = col("type_symbol")
                c_atom = col("auth_atom_id", "label_atom_id")
                c_res = col("auth_comp_id", "label_comp_id")
                c_seq = col("auth_seq_id", "label_seq_id")
                c_alt = col("label_alt_id")
                c_model = col("pdbx_PDB_model_num")
                c_x = col("Cartn_x")
                c_y = col("Cartn_y")
                c_z = col("Cartn_z")
                ncol = len(cols)

            if idx is None:
                continue

            # end of the atom_site loop
            if s == "#" or s.startswith("loop_") or (s.startswith("_") and not s.startswith("_atom_site")):
                break

            f = s.split()
            if len(f) < ncol:
                continue
            if f[c_group] not in ("ATOM", "HETATM"):
                continue
            # first model only
            if c_model is not None and f[c_model] not in ("1", "."):
                continue
            # keep only the first alternate conformation
            if c_alt is not None and unquote(f[c_alt]) not in (".", "?", "A"):
                continue

            res = unquote(f[c_res])
            if res in DROP_RES:
                continue
            atom = unquote(f[c_atom])
            elem = unquote(f[c_elem]).upper()
            x, y, z = f[c_x], f[c_y], f[c_z]
            seq = unquote(f[c_seq])
            if not seq.lstrip("-").isdigit():
                seq = "0"   # NGPB reads resNum as int; never emit a non-numeric here

            res_ff = RES_ALIAS.get(res, res)
            hit = table.get((res_ff, atom)) or table.get((res_ff, ATOM_ALIAS.get(atom, atom)))
            if hit is not None:
                q, r = hit
                matched += 1
            else:
                q = 0.0
                r = ELEM_RADIUS.get(elem, DEFAULT_RADIUS)
                fallback += 1

            serial += 1
            # No chain column -> resNum is unambiguously the numeric token after resName,
            # matching NGPB's chain-vs-resNum heuristic. Whitespace-delimited, no fixed cols.
            out.write("ATOM %d %s %s %s %s %s %s %.4f %.4f\n"
                      % (serial, atom, res, seq, x, y, z, q, r))

    return serial, matched, fallback


def main():
    ap = argparse.ArgumentParser(description="Direct mmCIF -> PQR for NGPB (large structures).")
    ap.add_argument("input_cif")
    ap.add_argument("output_pqr")
    ap.add_argument("--ff", default="AMBER", help="forcefield table: AMBER (default), PARSE, CHARMM, ...")
    args = ap.parse_args()

    n, m, fb = convert(args.input_cif, args.output_pqr, args.ff)
    if n == 0:
        sys.stderr.write("ERROR: no atoms written -- is this an mmCIF with an _atom_site loop?\n")
        sys.exit(1)
    pct = 100.0 * m / n
    sys.stderr.write("wrote %d atoms to %s  (%.1f%% matched %s table, %d fell back to element radii)\n"
                     % (n, args.output_pqr, pct, args.ff.upper(), fb))


if __name__ == "__main__":
    main()

/*
 *  Copyright (C) 2021-2025 Vincenzo Di Florio
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

/*
 *  fmm_replay -- sweep the FMM parameters by replaying phase 2 of the energy
 *  stage from a dump, instead of re-running the whole pipeline per configuration.
 *
 *  On 6VYB the energy stage is 6.2 s of a 157.7 s pipeline, so a 140-point
 *  fmm_mac x fmm_multipole_order x fmm_leaf_size grid at 3 repeats costs ~18 h
 *  through ngpb and ~30 min here.
 *
 *  Since the source and target leaf sizes were decoupled, --leaf and --tleaf are
 *  independent axes and the grid is |mac|x|order|x|leaf|x|tleaf|. Cross them fully
 *  and the point count multiplies -- prefer a staged sweep (near-field knobs first,
 *  then the target/order knobs at the winner) over one large cross product.
 *
 *  Produce the dump by setting, in the [algorithm] section of options.prm:
 *
 *      energy_dump = fmm_inputs
 *
 *  then replay it at the SAME rank count that produced it:
 *
 *      mpirun -n 4 fmm_replay fmm_inputs \
 *          --mac 0.4:0.6:0.1 --order 9,10 --leaf 8,16,32,64 --tleaf 128,256 \
 *          --repeats 3 --csv sweep.csv
 *
 *  Each rank loads its own <prefix>.rank<K>.bin, so the sweep runs under the
 *  identical domain decomposition as the original solve and its timings are
 *  directly comparable to the "Compute energy" line of the timing report.
 *
 *  Accuracy is reported against the naive O(N*M) GPU kernels, evaluated once on
 *  the same inputs. That is the same comparison energy_method=1 vs 2 makes
 *  in-pipeline; the naive kernels agree with the CPU path to ~2e-11, well below
 *  the 1e-9..1e-4 range the FMM truncation error spans.
 */

#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "energy_dump.h"
#include "fmm.h"
#include "gpu_topology.h"

// ====================================================================
//  Naive GPU energy kernels. Declared here rather than including test.h,
//  which pulls in raytracer.h -> nanoshaper.h + json.hpp; this tool needs
//  none of that and stays a CUDA-only link. Keep in sync with include/test.h.
// ====================================================================
extern "C" {
void atoms_to_device (int num_atoms, const double *h_atoms, const double *h_charges,
                      double **d_atoms_out, double **d_charges_out);

void atoms_free_device (double *d_atoms, double *d_charges);

double polarization_energy_cuda_dev (int num_pts, const double *h_V, const double *h_flux,
                                     int num_atoms, const double *d_atoms,
                                     const double *d_charges);

double ionic_energy_cuda_dev (int num_tri_verts, const double *h_vert, const double *h_norms,
                              const double *h_phi_sup, const double *h_area,
                              int num_atoms, const double *d_atoms, const double *d_charges,
                              double inv_4pi);
}

// ====================================================================
//  Parameter-spec parsing.
//
//  Two forms, so a full sweep and a single probe read the same way:
//    "a,b,c"     explicit list
//    "lo:hi"     inclusive range, step 1
//    "lo:hi:s"   inclusive range, step s (float-safe: the count is computed
//                up front rather than accumulating the step)
// ====================================================================
static bool
parse_spec (const std::string &spec, std::vector<double> &out)
{
  out.clear ();

  if (spec.empty ())
    return false;

  if (spec.find (':') != std::string::npos) {
    double lo = 0.0, hi = 0.0, step = 1.0;
    int n = std::sscanf (spec.c_str (), "%lf:%lf:%lf", &lo, &hi, &step);

    if (n < 2 || step <= 0.0 || hi < lo)
      return false;

    // Round rather than truncate: 0.2:0.8:0.1 must yield 7 values, not 6.
    const int count = (int) std::llround ((hi - lo) / step) + 1;

    for (int i = 0; i < count; ++i)
      out.push_back (lo + step * i);

    return true;
  }

  size_t pos = 0;

  while (pos <= spec.size ()) {
    size_t comma = spec.find (',', pos);
    std::string tok = spec.substr (pos, comma == std::string::npos
                                        ? std::string::npos : comma - pos);

    if (!tok.empty ()) {
      char *end = nullptr;
      double v = std::strtod (tok.c_str (), &end);

      if (end == tok.c_str () || *end != '\0')
        return false;

      out.push_back (v);
    }

    if (comma == std::string::npos)
      break;

    pos = comma + 1;
  }

  return !out.empty ();
}

static bool
parse_spec_int (const std::string &spec, std::vector<int> &out)
{
  std::vector<double> tmp;

  if (!parse_spec (spec, tmp))
    return false;

  out.clear ();
  for (double v : tmp)
    out.push_back ((int) std::llround (v));

  return true;
}

// --------------------------------------------------------------------
//  "src/tgt,src/tgt,..." -- an EXPLICIT list of (source, target) leaf pairs,
//  replacing the --leaf x --tleaf cross product.
//
//  Wanted because the two sweeps ask different questions. Finding the leaf
//  optimum needs the full cross product; sweeping p and theta AT that optimum
//  needs only the handful of pairs that won, and crossing those back out would
//  multiply the point count by |leaf|x|tleaf| for configurations already known
//  to be beaten.
//
//  '/' separates the pair, not ':' -- ':' already means "range" in a SPEC.
// --------------------------------------------------------------------
static bool
parse_pairs (const std::string &spec, std::vector<std::pair<int, int>> &out)
{
  out.clear ();

  size_t pos = 0;

  while (pos <= spec.size ()) {
    size_t comma = spec.find (',', pos);
    std::string tok = spec.substr (pos, comma == std::string::npos
                                        ? std::string::npos : comma - pos);

    if (!tok.empty ()) {
      size_t slash = tok.find ('/');

      if (slash == std::string::npos)
        return false;

      char *end = nullptr;
      long src = std::strtol (tok.substr (0, slash).c_str (), &end, 10);
      if (end == nullptr || *end != '\0' || src < 1)
        return false;

      std::string tgt_tok = tok.substr (slash + 1);
      long tgt = std::strtol (tgt_tok.c_str (), &end, 10);
      if (end == tgt_tok.c_str () || *end != '\0' || tgt < 0)
        return false;

      out.emplace_back ((int) src, (int) tgt);
    }

    if (comma == std::string::npos)
      break;

    pos = comma + 1;
  }

  return !out.empty ();
}

static double
relerr (double value, double ref)
{
  if (ref == 0.0)
    return (value == 0.0) ? 0.0 : NAN;

  return std::fabs (value - ref) / std::fabs (ref);
}

static void
usage (const char *argv0)
{
  std::fprintf (stderr,
    "usage: mpirun -n <ranks> %s <dump-prefix> [options]\n"
    "\n"
    "  <dump-prefix>       matches [algorithm] energy_dump in options.prm;\n"
    "                      each rank reads <prefix>.rank<K>.bin\n"
    "\n"
    "  --mac SPEC          fmm_mac values            (default 0.4)\n"
    "  --order SPEC        fmm_multipole_order       (default 11, max 12)\n"
    "  --leaf SPEC         fmm_leaf_size (SOURCE)    (default 16)\n"
    "  --tleaf SPEC        fmm_target_leaf_size      (default 1024; 0 = follow --leaf)\n"
    "  --pairs SRC/TGT,... explicit (source,target) leaf pairs; REPLACES the\n"
    "                      --leaf x --tleaf cross product\n"
    "  --repeats N         timed repeats per config  (default 3)\n"
    "  --warmup N          discarded runs per config (default 1)\n"
    "  --csv PATH          write results here        (default stdout only)\n"
    "  --no-naive          skip the naive baseline (relative errors become NaN)\n"
    "\n"
    "  --leaf sizes the SOURCE (atom) tree: it bounds the box radius the MAC tests\n"
    "  against, hence the near-field radius and the P2P work. --tleaf sizes the\n"
    "  TARGET tree: it sets the target box count, hence the M2L pair count. They\n"
    "  pull opposite ways, so sweeping them together on one --leaf value (what this\n"
    "  tool did before --tleaf existed) only samples the diagonal of the real grid.\n"
    "  --tleaf is capped at FMM_MAX_TLEAF inside the energy entry points; the value\n"
    "  actually used is what lands in the tgt_leaf CSV column.\n"
    "\n"
    "  Measured optima (6VYB + 1VSZ, RTX 3080): src 16, tgt 1024 -- interior on BOTH\n"
    "  axes (8/32 and 512/4096 are worse). --order past 11 REGRESSES: p=12 is slower\n"
    "  and less accurate than p=11 (unscaled M2L conditioning), so 11 is the ceiling.\n"
    "\n"
    "  SPEC is \"a,b,c\", \"lo:hi\" (step 1) or \"lo:hi:step\", e.g.\n"
    "    --mac 0.4:0.6:0.1 --order 10,11 --leaf 8,16,32 --tleaf 512,1024,2048\n",
    argv0);
}

int
main (int argc, char **argv)
{
  MPI_Init (&argc, &argv);

  int rank = 0, size = 1;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &size);

  std::string prefix, csv_path;
  std::vector<double> macs {0.4};
  std::vector<int> orders {11}, leaves {16};   // measured optimum; see usage()
  std::vector<int> tleaves {1024};   // 0 would follow the source leaf (pre-decoupling behaviour)
  std::vector<std::pair<int, int>> leaf_pairs;   // non-empty => --pairs overrides leaf x tleaf
  int repeats = 3, warmup = 1;
  bool do_naive = true;
  bool bad_args = false;

  for (int i = 1; i < argc && !bad_args; ++i) {
    std::string a = argv[i];
    auto next = [&] (std::string &dst) {
      if (i + 1 >= argc) { bad_args = true; return false; }
      dst = argv[++i];
      return true;
    };

    if (a == "-h" || a == "--help") {
      if (rank == 0) usage (argv[0]);
      MPI_Finalize ();
      return 0;
    } else if (a == "--mac") {
      std::string s;
      if (next (s) && !parse_spec (s, macs)) bad_args = true;
    } else if (a == "--order") {
      std::string s;
      if (next (s) && !parse_spec_int (s, orders)) bad_args = true;
    } else if (a == "--leaf") {
      std::string s;
      if (next (s) && !parse_spec_int (s, leaves)) bad_args = true;
    } else if (a == "--pairs") {
      std::string s;
      if (next (s) && !parse_pairs (s, leaf_pairs)) bad_args = true;
    } else if (a == "--tleaf") {
      std::string s;
      if (next (s) && !parse_spec_int (s, tleaves)) bad_args = true;
    } else if (a == "--repeats") {
      std::string s;
      if (next (s)) repeats = std::atoi (s.c_str ());
    } else if (a == "--warmup") {
      std::string s;
      if (next (s)) warmup = std::atoi (s.c_str ());
    } else if (a == "--csv") {
      next (csv_path);
    } else if (a == "--no-naive") {
      do_naive = false;
    } else if (!a.empty () && a[0] == '-') {
      bad_args = true;
    } else if (prefix.empty ()) {
      prefix = a;
    } else {
      bad_args = true;
    }
  }

  if (bad_args || prefix.empty () || repeats < 1 || warmup < 0) {
    if (rank == 0) usage (argv[0]);
    MPI_Finalize ();
    return 2;
  }

  // Without --pairs the leaf axes are a full cross product, which is what the
  // leaf-optimum sweep wants. Building the list here means the sweep loop has
  // exactly one shape to walk either way.
  if (leaf_pairs.empty ())
    for (int l : leaves)
      for (int t : tleaves)
        leaf_pairs.emplace_back (l, t);

  // ----------------------------------------------------------------
  //  Bind this rank's CUDA device exactly as ngpb's main() does. On a
  //  single-GPU node this changes nothing -- every rank is on device 0 either
  //  way -- but on a multi-GPU node the pipeline block-maps its ranks across
  //  the GPUs, and without this the whole replay would pile onto device 0 and
  //  its timings would not be comparable to that run's "Compute energy" line.
  //  Must precede every CUDA call, i.e. atoms_to_device below.
  // ----------------------------------------------------------------
  setup_gpu_topology (MPI_COMM_WORLD);

  // ----------------------------------------------------------------
  //  Load this rank's slice of the dump.
  // ----------------------------------------------------------------
  energy_inputs_t in;
  std::string err;
  int dump_size = 0;
  int ok = read_energy_inputs (in, energy_dump_path (prefix, rank), dump_size, err) ? 1 : 0;

  // Every rank must succeed; otherwise the reduced energy would be a silent
  // partial sum. Agree on failure before anyone reaches a collective.
  int all_ok = 0;
  MPI_Allreduce (&ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  if (!ok)
    std::fprintf (stderr, "rank %d: %s\n", rank, err.c_str ());

  if (!all_ok) {
    MPI_Finalize ();
    return 1;
  }

  if (dump_size != size) {
    if (rank == 0)
      std::fprintf (stderr,
                    "error: dump was written by %d ranks, this replay has %d.\n"
                    "       Re-run with: mpirun -n %d %s %s ...\n",
                    dump_size, size, dump_size, argv[0], prefix.c_str ());
    MPI_Finalize ();
    return 1;
  }

  const int num_atoms     = in.num_atoms ();
  const int num_pts       = in.num_pts ();
  const int num_tri_verts = in.num_tri_verts ();

  {
    long long tot_pts = 0, tot_tri = 0;
    long long my_pts = num_pts, my_tri = num_tri_verts / 3;
    MPI_Reduce (&my_pts, &tot_pts, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce (&my_tri, &tot_tri, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
      std::printf ("loaded %s: %d ranks, %d atoms, %lld flux points, %lld triangles, ionic=%s\n",
                   prefix.c_str (), size, num_atoms, tot_pts, tot_tri,
                   in.do_ionic ? "yes" : "no");
  }

  // ----------------------------------------------------------------
  //  Upload the atoms once. Hoisted out of the sweep on purpose: the tree
  //  is rebuilt per configuration, but the source data never changes.
  // ----------------------------------------------------------------
  double *d_atoms = nullptr, *d_charges = nullptr;
  atoms_to_device (num_atoms, in.atoms.data (), in.charges.data (), &d_atoms, &d_charges);

  FILE *csv = nullptr;

  if (rank == 0 && !csv_path.empty ()) {
    csv = std::fopen (csv_path.c_str (), "w");

    if (!csv) {
      std::fprintf (stderr, "error: cannot open %s for writing\n", csv_path.c_str ());
      atoms_free_device (d_atoms, d_charges);
      MPI_Finalize ();
      return 1;
    }

    // energy_coul is constant across the sweep, but scripts/plot_fmm_sweep.py
    // needs it to form energy_sum = pol + ionic + coul on the same scale as a
    // bench_sweep.py CSV -- it is part of the relative-error denominator.
    // src_leaf/tgt_leaf replace the old single `leaf` column. Older CSVs with a `leaf`
    // column were produced when one value sized BOTH trees, i.e. they sample only the
    // src_leaf == tgt_leaf diagonal -- do not concatenate them with new rows.
    std::fprintf (csv, "method,mac,order,src_leaf,tgt_leaf,repeat,ranks,"
                       "t_build_s,t_pol_s,t_ionic_s,t_total_s,"
                       "energy_pol,energy_ionic,energy_coul,"
                       "relerr_pol,relerr_ionic\n");
  }

  // Sum the per-rank contributions exactly the way energy_cuda_fast does:
  // form energy_pol / energy_react locally, then MPI_Reduce the results.
  auto reduce_energies = [&] (double first_int, double second_int,
                              double &energy_pol, double &energy_ionic) {
    double loc_pol   = 0.5 * in.constant_pol * first_int;
    double loc_ionic = in.do_ionic ? 0.5 * (second_int - first_int * in.constant_react) : 0.0;

    MPI_Reduce (&loc_pol,   &energy_pol,   1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce (&loc_ionic, &energy_ionic, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  };

  // The stage is only as fast as its slowest rank.
  auto reduce_time = [&] (double t) {
    double out = 0.0;
    MPI_Reduce (&t, &out, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    return out;
  };

  // ----------------------------------------------------------------
  //  Naive baseline: the accuracy reference, and a timing datapoint of its own.
  // ----------------------------------------------------------------
  double ref_pol = NAN, ref_ionic = NAN;

  if (do_naive) {
    for (int rep = 0; rep < warmup + repeats; ++rep) {
      MPI_Barrier (MPI_COMM_WORLD);
      double t0 = MPI_Wtime ();

      double first_int = polarization_energy_cuda_dev (num_pts, in.V_pol.data (),
                                                       in.flux_pol.data (),
                                                       num_atoms, d_atoms, d_charges);
      cudaDeviceSynchronize ();
      double t1 = MPI_Wtime ();

      double second_int = 0.0;
      if (in.do_ionic)
        second_int = ionic_energy_cuda_dev (num_tri_verts, in.vert_ion.data (),
                                            in.norms_ion.data (), in.phi_ion.data (),
                                            in.area_ion.data (), num_atoms,
                                            d_atoms, d_charges, in.inv_4pi);
      cudaDeviceSynchronize ();
      double t2 = MPI_Wtime ();

      double e_pol = 0.0, e_ionic = 0.0;
      reduce_energies (first_int, second_int, e_pol, e_ionic);

      double t_pol   = reduce_time (t1 - t0);
      double t_ionic = reduce_time (t2 - t1);

      if (rank == 0) {
        ref_pol   = e_pol;
        ref_ionic = e_ionic;

        if (rep >= warmup) {
          if (csv)
            std::fprintf (csv, "naive,,,,,%d,%d,0,%.9f,%.9f,%.9f,%.17g,%.17g,%.17g,0,0\n",
                          rep - warmup, size, t_pol, t_ionic, t_pol + t_ionic,
                          e_pol, e_ionic, in.coul_energy);
        }
      }
    }

    // Every rank needs the reference to report relative errors consistently.
    MPI_Bcast (&ref_pol,   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast (&ref_ionic, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0)
      std::printf ("naive reference: energy_pol = %.17g  energy_ionic = %.17g\n\n",
                   ref_pol, ref_ionic);
  }

  // ----------------------------------------------------------------
  //  The sweep.
  // ----------------------------------------------------------------
  long done = 0;

  if (rank == 0)
    std::printf ("%-6s %-5s %-6s %-6s %10s %10s %10s %12s %12s\n",
                 "mac", "order", "sleaf", "tleaf", "build[s]", "pol[s]", "ionic[s]",
                 "relerr_pol", "relerr_ion");

  for (double mac : macs) {
    for (int order : orders) {
      for (const std::pair<int, int> &lp : leaf_pairs) {
       {
        const int leaf  = lp.first;
        const int tleaf = lp.second;

        // Record what actually RAN, not what was asked for: 0 follows the source leaf, and the
        // energy entry points clamp to FMM_MAX_TLEAF. Resolving both here means the CSV can never
        // disagree with the configuration the timing came from. Must mirror resolve_tleaf() in
        // src/fmm.cu -- note it clamps CAPACITY only; the launch width is a separate constant
        // (FMM_L2P_BLOCK) now that l2p_p2p_kernel strides over its leaf's points.
        int tleaf_eff = (tleaf < 1) ? leaf : tleaf;
        if (tleaf_eff > FMM_MAX_TLEAF) tleaf_eff = FMM_MAX_TLEAF;

        double last_relerr_pol = NAN, last_relerr_ion = NAN;
        double last_build = 0.0, last_pol_t = 0.0, last_ion_t = 0.0;

        for (int rep = 0; rep < warmup + repeats; ++rep) {
          MPI_Barrier (MPI_COMM_WORLD);
          double t0 = MPI_Wtime ();

          fmm_tree *tree = nullptr;
          fmm_build_atom_tree (num_atoms, d_atoms, d_charges, mac, order, leaf, &tree);
          cudaDeviceSynchronize ();
          double t1 = MPI_Wtime ();

          double first_int = fmm_polarization_energy (tree, num_pts, in.V_pol.data (),
                                                      in.flux_pol.data (), tleaf_eff);
          cudaDeviceSynchronize ();
          double t2 = MPI_Wtime ();

          double second_int = 0.0;
          if (in.do_ionic)
            second_int = fmm_ionic_energy (tree, num_tri_verts, in.vert_ion.data (),
                                           in.norms_ion.data (), in.phi_ion.data (),
                                           in.area_ion.data (), in.inv_4pi, tleaf_eff);
          cudaDeviceSynchronize ();
          double t3 = MPI_Wtime ();

          fmm_free_tree (tree);

          double e_pol = 0.0, e_ionic = 0.0;
          reduce_energies (first_int, second_int, e_pol, e_ionic);

          double t_build = reduce_time (t1 - t0);
          double t_pol   = reduce_time (t2 - t1);
          double t_ionic = reduce_time (t3 - t2);

          if (rank == 0 && rep >= warmup) {
            double t_total = t_build + t_pol + t_ionic;
            double rp = do_naive ? relerr (e_pol, ref_pol) : NAN;
            double ri = (do_naive && in.do_ionic) ? relerr (e_ionic, ref_ionic) : NAN;

            if (csv)
              std::fprintf (csv, "fmm,%g,%d,%d,%d,%d,%d,"
                                 "%.9f,%.9f,%.9f,%.9f,%.17g,%.17g,%.17g,%.6e,%.6e\n",
                            mac, order, leaf, tleaf_eff, rep - warmup, size,
                            t_build, t_pol, t_ionic, t_total,
                            e_pol, e_ionic, in.coul_energy, rp, ri);

            last_build = t_build;
            last_pol_t = t_pol;
            last_ion_t = t_ionic;
            last_relerr_pol = rp;
            last_relerr_ion = ri;
          }
        }

        if (rank == 0) {
          std::printf ("%-6g %-5d %-6d %-6d %10.4f %10.4f %10.4f %12.3e %12.3e\n",
                       mac, order, leaf, tleaf_eff, last_build, last_pol_t, last_ion_t,
                       last_relerr_pol, last_relerr_ion);
          std::fflush (stdout);

          if (csv)
            std::fflush (csv);
        }

        ++done;
       }
      }
    }
  }

  if (rank == 0) {
    std::printf ("\n%ld configurations x %d repeats done\n", done, repeats);

    if (csv) {
      std::fclose (csv);
      std::printf ("results: %s\n", csv_path.c_str ());
    }
  }

  atoms_free_device (d_atoms, d_charges);
  MPI_Finalize ();

  return 0;
}

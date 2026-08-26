/*
 *  Copyright (C) 2019-2025 Carlo de Falco
 *  Copyright (C) 2020-2021 Martina Politi
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
 *  On-disk form of the inputs to the electrostatic energy kernels.
 *
 *  energy_cuda_fast splits cleanly in two:
 *
 *    phase 1  sweep border_quad over the mesh and collect, per rank, the flux
 *             surface points and the marching-cubes triangle data. Depends on
 *             the mesh, phi, the ray cache and the MPI decomposition; does NOT
 *             depend on energy_method or on any fmm_* parameter.
 *
 *    phase 2  evaluate the two integrals against the atom charges, either with
 *             the naive O(N*M) kernels or with the FMM. Depends ONLY on the
 *             arrays phase 1 produced -- no mesh, no p8est, no phi, no ray
 *             cache, no MPI topology.
 *
 *  energy_inputs_t is that interface. Dumping it after phase 1 lets the whole
 *  FMM parameter space be swept by replaying phase 2 alone (see
 *  src/tools/fmm_replay.cpp), instead of re-running a pipeline in which the
 *  energy stage is under 4% of the wall time.
 *
 *  The format is raw host-endian doubles with a fixed-size header. It is a
 *  scratch format for benchmarking on one machine, not an archival one: there
 *  is no byte-order or float-format negotiation, and a dump is only meaningful
 *  to a replay running at the same rank count that produced it.
 *
 *  This header is deliberately free of bimpp/p8est/MPI dependencies so the
 *  replay driver can include it and link against CUDA alone.
 */

#ifndef NGPB_ENERGY_DUMP_H
#define NGPB_ENERGY_DUMP_H

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

// ====================================================================
//  Everything phase 2 reads. Sizes are implied by the array lengths:
//    num_atoms     = charges.size()      atoms     is 3 * num_atoms
//    num_pts       = flux_pol.size()     V_pol     is 3 * num_pts
//    num_tri_verts = phi_ion.size()      vert_ion  is 3 * num_tri_verts
//                                        norms_ion is 3 * num_tri_verts
//                                        area_ion  is num_tri_verts / 3
//  (num_tri_verts counts triangle VERTICES, i.e. 3 per triangle, matching the
//  argument of fmm_ionic_energy / ionic_energy_cuda_dev.)
// ====================================================================
struct energy_inputs_t
{
  bool   do_ionic   = false;
  double charge_pol = 0.0;   // accumulated flux, printed as charge_pol/(4 pi)

  // Scalars derived from the prm parameters. Carried so the replay can report
  // energies in the same units without re-deriving the physical constants.
  double inv_4pi        = 0.0;
  double constant_pol   = 0.0;
  double constant_react = 0.0;
  double den_in         = 0.0;
  double net_charge     = 0.0;
  double coul_energy    = 0.0;   // rank 0 only; 0 when calc_coulombic != 1

  std::vector<double> atoms;       // 3 * num_atoms, interleaved x,y,z
  std::vector<double> charges;     // num_atoms

  std::vector<double> V_pol;       // 3 * num_pts
  std::vector<double> flux_pol;    // num_pts

  std::vector<double> vert_ion;    // 3 * num_tri_verts
  std::vector<double> norms_ion;   // 3 * num_tri_verts
  std::vector<double> phi_ion;     // num_tri_verts
  std::vector<double> area_ion;    // num_tri_verts / 3

  int num_atoms     () const { return (int) charges.size (); }
  int num_pts       () const { return (int) flux_pol.size (); }
  int num_tri_verts () const { return (int) phi_ion.size (); }
};

// ====================================================================
//  Fixed 96-byte header. Explicitly padded so the doubles are 8-aligned.
// ====================================================================
struct energy_dump_header
{
  char     magic[8];        // "NGPBFMM1", not NUL-terminated
  int32_t  version;
  int32_t  rank;
  int32_t  size;            // MPI world size that produced the dump
  int32_t  num_atoms;
  int32_t  num_pts;
  int32_t  num_tri_verts;
  int32_t  do_ionic;
  int32_t  pad;
  double   charge_pol;
  double   coul_energy;
  double   inv_4pi;
  double   constant_pol;
  double   constant_react;
  double   den_in;
  double   net_charge;
};

static constexpr char    NGPB_ENERGY_DUMP_MAGIC[8] = {'N','G','P','B','F','M','M','1'};
static constexpr int32_t NGPB_ENERGY_DUMP_VERSION  = 1;

// Path of one rank's dump, given the user-supplied prefix.
inline std::string
energy_dump_path (const std::string &prefix, int rank)
{
  return prefix + ".rank" + std::to_string (rank) + ".bin";
}

namespace ngpb_dump_detail
{
  inline void
  put (std::ofstream &os, const std::vector<double> &v)
  {
    if (!v.empty ())
      os.write (reinterpret_cast<const char *> (v.data ()),
                (std::streamsize) (v.size () * sizeof (double)));
  }

  inline bool
  get (std::ifstream &is, std::vector<double> &v, size_t n)
  {
    v.resize (n);
    if (n == 0)
      return true;
    is.read (reinterpret_cast<char *> (v.data ()),
             (std::streamsize) (n * sizeof (double)));
    return (bool) is;
  }
}

// ====================================================================
//  Write one rank's inputs. Returns false (with err set) on any I/O failure;
//  callers treat that as a warning, never as a reason to abort the solve.
// ====================================================================
inline bool
write_energy_inputs (const energy_inputs_t &in, const std::string &path,
                     int rank, int size, std::string &err)
{
  std::ofstream os (path, std::ios::binary | std::ios::trunc);

  if (!os) {
    err = "cannot open " + path + " for writing";
    return false;
  }

  energy_dump_header h{};
  std::memcpy (h.magic, NGPB_ENERGY_DUMP_MAGIC, sizeof (h.magic));
  h.version        = NGPB_ENERGY_DUMP_VERSION;
  h.rank           = rank;
  h.size           = size;
  h.num_atoms      = in.num_atoms ();
  h.num_pts        = in.num_pts ();
  h.num_tri_verts  = in.num_tri_verts ();
  h.do_ionic       = in.do_ionic ? 1 : 0;
  h.charge_pol     = in.charge_pol;
  h.coul_energy    = in.coul_energy;
  h.inv_4pi        = in.inv_4pi;
  h.constant_pol   = in.constant_pol;
  h.constant_react = in.constant_react;
  h.den_in         = in.den_in;
  h.net_charge     = in.net_charge;

  os.write (reinterpret_cast<const char *> (&h), sizeof (h));

  ngpb_dump_detail::put (os, in.atoms);
  ngpb_dump_detail::put (os, in.charges);
  ngpb_dump_detail::put (os, in.V_pol);
  ngpb_dump_detail::put (os, in.flux_pol);
  ngpb_dump_detail::put (os, in.vert_ion);
  ngpb_dump_detail::put (os, in.norms_ion);
  ngpb_dump_detail::put (os, in.phi_ion);
  ngpb_dump_detail::put (os, in.area_ion);

  os.close ();

  if (!os) {
    err = "write failed on " + path;
    return false;
  }

  return true;
}

// ====================================================================
//  Read one rank's inputs back. `size_out` receives the world size the dump
//  was produced at, so the replay can refuse a mismatched rank count rather
//  than silently reporting a partial energy.
// ====================================================================
inline bool
read_energy_inputs (energy_inputs_t &in, const std::string &path,
                    int &size_out, std::string &err)
{
  std::ifstream is (path, std::ios::binary);

  if (!is) {
    err = "cannot open " + path;
    return false;
  }

  energy_dump_header h{};
  is.read (reinterpret_cast<char *> (&h), sizeof (h));

  if (!is) {
    err = path + " is shorter than its header";
    return false;
  }

  if (std::memcmp (h.magic, NGPB_ENERGY_DUMP_MAGIC, sizeof (h.magic)) != 0) {
    err = path + " is not an ngpb energy dump";
    return false;
  }

  if (h.version != NGPB_ENERGY_DUMP_VERSION) {
    err = path + ": dump version " + std::to_string (h.version)
        + ", this build expects " + std::to_string (NGPB_ENERGY_DUMP_VERSION);
    return false;
  }

  if (h.num_atoms < 0 || h.num_pts < 0 || h.num_tri_verts < 0
      || h.num_tri_verts % 3 != 0) {
    err = path + ": implausible header counts";
    return false;
  }

  size_out          = h.size;
  in.do_ionic       = (h.do_ionic != 0);
  in.charge_pol     = h.charge_pol;
  in.coul_energy    = h.coul_energy;
  in.inv_4pi        = h.inv_4pi;
  in.constant_pol   = h.constant_pol;
  in.constant_react = h.constant_react;
  in.den_in         = h.den_in;
  in.net_charge     = h.net_charge;

  const size_t na = (size_t) h.num_atoms;
  const size_t np = (size_t) h.num_pts;
  const size_t nt = (size_t) h.num_tri_verts;

  bool ok = ngpb_dump_detail::get (is, in.atoms,     3 * na)
         && ngpb_dump_detail::get (is, in.charges,       na)
         && ngpb_dump_detail::get (is, in.V_pol,     3 * np)
         && ngpb_dump_detail::get (is, in.flux_pol,      np)
         && ngpb_dump_detail::get (is, in.vert_ion,  3 * nt)
         && ngpb_dump_detail::get (is, in.norms_ion, 3 * nt)
         && ngpb_dump_detail::get (is, in.phi_ion,       nt)
         && ngpb_dump_detail::get (is, in.area_ion,  nt / 3);

  if (!ok) {
    err = path + " is truncated";
    return false;
  }

  return true;
}

#endif /* NGPB_ENERGY_DUMP_H */

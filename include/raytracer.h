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

#ifndef RAYTRACER_H
#define RAYTRACER_H

#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <array>
#include <cstring>
#include <mpi.h>
#include <nanoshaper.h>
// #include <raytracer_datatype.h>

#include "json.hpp"
#include "serialize.h"

using int_coord_t = unsigned long long int;

// map_compare uses tol=1e-3, so quantise to that grid before hashing/comparing.
// NanoShaper and p4est compute the same geometric point via different arithmetic;
// they can differ by a small FP amount that is << tol but != 0.
struct ray_hash {
  size_t operator()(const std::array<double, 2>& a) const noexcept {
    auto qx = static_cast<int64_t>(std::llround(a[0] * 1000.0));
    auto qy = static_cast<int64_t>(std::llround(a[1] * 1000.0));
    uint64_t ux = static_cast<uint64_t>(qx);
    uint64_t uy = static_cast<uint64_t>(qy);
    ux ^= uy * 0x9e3779b97f4a7c15ULL;
    ux ^= ux >> 30; ux *= 0xbf58476d1ce4e5b9ULL;
    ux ^= ux >> 27; ux *= 0x94d049bb133111ebULL;
    return static_cast<size_t>(ux ^ (ux >> 31));
  }
};

struct ray_equal {
  bool operator()(const std::array<double,2>& a, const std::array<double,2>& b) const noexcept {
    return std::llround(a[0] * 1000.0) == std::llround(b[0] * 1000.0) &&
           std::llround(a[1] * 1000.0) == std::llround(b[1] * 1000.0);
  }
};

using fast_rays_t = std::array<std::unordered_map<std::array<double,2>, crossings_t, ray_hash, ray_equal>, 3>;

struct
  ray_cache_t {

  fast_rays_t rays; //map that contains all the rays in the 3 direction

  static int_coord_t count_cache;
  static int_coord_t count_new;
  int count_new_dir[3] = {0, 0, 0};
  int count_cache_dir[3] = {0, 0, 0};
  int count = 0;
  // unsigned dir; //direction of the ray
  std::unique_ptr<NS::NanoShaper> ns;
  // NS::NanoShaper ns;
  double l_c[3] = {0, 0, 0};
  double r_c[3] = {0, 0, 0};
  // int num_req_rays = 0;
  int num_req_rays[3] = {0, 0, 0};
  std::array<std::unordered_set<std::array<double, 2>, ray_hash, ray_equal>, 3> rays_list; //list of req rays

  crossings_t &
  operator () (double x0, double x1, unsigned dir = 1);

  void
  fill_cache ();

  void
  init_analytical_surf (const std::vector<NS::Atom> & atoms, const NS::surface_type & surf_type,
                        const double & surf_param, const double & stern_layer, const unsigned & num_threads,const std::string* configFile=nullptr);

  void
  init_analytical_surf_ns (const std::vector<NS::Atom> & atoms, const NS::surface_type & surf_type,
                           const double & surf_param, const double & stern_layer, const unsigned & num_threads,
                           double* l_cr, double* r_cr, double scale,const std::string* configFile=nullptr);
  void
  compute_ns_inters (crossings_t & ct);

  void
  ensure_ray (double x0, double x1, unsigned dir);

  void
  compute_pending_rays ();

  std::vector<unsigned char>
  write_map (const std::map<std::array<double, 2>, crossings_t, map_compare>& container);

  void
  read_map (const std::vector<unsigned char>& data,
            std::map<std::array<double, 2>, crossings_t, map_compare>& container);

  std::vector<unsigned char>
  write_ct (const crossings_t& ct);

  void
  read_ct (const std::vector<unsigned char>& data, crossings_t& ct);
};

#endif //RAYTRACER_H

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

#include "raytracer.h"

crossings_t &
ray_cache_t::operator () (double x0, double x1, unsigned direct)
{

  int rank;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);

  static crossings_t cr_t;

  static std::array<double, 2> start_point;
  start_point = {x0, x1};

  auto it0 = rays[direct].find (start_point);

  if (it0 != rays[direct].end () && it0->second.init == 1) {
    return rays[direct][start_point];
  }

  else if (it0 != rays[direct].end () && it0->second.init == 0) {
    count_new++;
    count_new_dir[direct]++;

    if (rank == 0)
      compute_ns_inters (it0->second);
  } else {
    // create a new ray
    crossings_t tmp;
    tmp.point[0] = x0;
    tmp.point[1] = x1;
    tmp.dir = direct;

    if (rank == 0)
      compute_ns_inters (tmp);

    rays[direct][start_point] = std::move (tmp);
    cr_t = rays[direct][start_point];
    count_new++;
    count_new_dir[direct]++;
  }

  return rays[direct][start_point];
}

void
ray_cache_t::fill_cache ()
{
  int rank, size;
  MPI_Comm mpicomm = MPI_COMM_WORLD;
  MPI_Comm_rank (mpicomm, &rank);
  MPI_Comm_size (mpicomm, &size);

  for (unsigned idir = 0; idir < 3; ++idir) {
    std::vector<std::array<double, 2>> rays_vector (rays_list[idir].begin (), rays_list[idir].end ()); //copy the set content inside a vector

    num_req_rays[idir] = rays_vector.size (); //numb of req rays from each proc
    MPI_Barrier (mpicomm);
    // std::cout << "Sending " << num_req_rays[idir] << " rays requested from rank " << rank << std::endl;

    std::vector<unsigned char> local_ser_rays_vec = serialize::write (rays_vector); //vector of char for the rays_vector
    std::vector<unsigned char> global_ser_rays_vec; //vector with all the rays from all proc

    int ser_rays_len = local_ser_rays_vec.size (); //length of each vect of char
    int global_ser_rays_len; //variable with total number of rays (sum all the proc rays)

    auto proc_ser_rays_len = std::make_unique<int[]> (size);
    auto proc_num_req_rays = std::make_unique<int[]> (size);
    auto displ_ser_rays_vec = std::make_unique<int[]> (size);
    auto cum_rays_vec = std::make_unique<int[]> (size);


    std::vector<unsigned char> global_ser_map; //vector of char for the map of rank 0
    std::vector<unsigned char> local_ser_map; //vector of char for the local map to send to each rank

    int global_ser_map_len;
    int local_ser_map_len;

    auto proc_ser_map_len = std::make_unique<int[]> (size);
    auto displ_ser_map = std::make_unique<int[]> (size);


    MPI_Reduce (&ser_rays_len, &global_ser_rays_len, 1, MPI_INT, MPI_SUM, 0, mpicomm); //fill global_ser_rays_len

    MPI_Gather (&ser_rays_len, 1, MPI_INT, proc_ser_rays_len.get (), 1, MPI_INT, 0, mpicomm); //fill proc_ser_rays_len
    MPI_Gather (&num_req_rays[idir], 1, MPI_INT, proc_num_req_rays.get (), 1, MPI_INT, 0, mpicomm); //fill proc_num_req_rays

    if (rank == 0) {
      for (int i = 1; i < size; i++) {
        displ_ser_rays_vec[i] = displ_ser_rays_vec[i-1] + proc_ser_rays_len[i-1];
        cum_rays_vec[i] = cum_rays_vec[i-1] + proc_num_req_rays[i];
      }

      global_ser_rays_vec.resize (global_ser_rays_len);
    }

    // Send the requested rays to rank 0
    MPI_Gatherv (&local_ser_rays_vec[0], ser_rays_len, MPI_CHAR, &global_ser_rays_vec[0],
                 proc_ser_rays_len.get (), displ_ser_rays_vec.get (), MPI_CHAR, 0, mpicomm);

    std::map<std::array<double, 2>, crossings_t, map_compare> local_req_rays_map;

    if (rank == 0) {
      // Transform the char rays in vector<array<double,2>>
      serialize::read (global_ser_rays_vec, rays_vector);
      //std::copy (rays_vector.begin (), rays_vector.end (),
      //std::inserter(rays_list[idir], rays_list[idir].begin ())); //copy the req rays in a set, to avoid repetitions in for loop

      // Add the rays to rank 0 map:
      //for (auto it = rays_list[idir].begin (); it != rays_list[idir].end (); it ++)
      //crossings_t & ct = (*this)((*it)[0], (*it)[1]);

      for (int i = 1; i < size; i++) {
        local_req_rays_map.clear ();

        std::transform (rays_vector.begin () + cum_rays_vec[i-1],
                        rays_vector.begin () + cum_rays_vec[i],
                        std::inserter (local_req_rays_map, end (local_req_rays_map)),
        [this,idir] (std::array<double,2> arr) {
          return (std::pair<std::array<double,2>, crossings_t> (arr, (*this) (arr[0], arr[1],idir)));
        }); //return ((this->rays).find(arr));

        local_ser_map = write_map (local_req_rays_map);
        proc_ser_map_len[i] = local_ser_map.size ();
        global_ser_map.insert (global_ser_map.end (), local_ser_map.begin (), local_ser_map.end ());
        displ_ser_map[i] = displ_ser_map[i-1] + proc_ser_map_len[i-1];
      }

      global_ser_map_len = global_ser_map.size ();

    }

    MPI_Scatter (proc_ser_map_len.get (), 1, MPI_INT, &local_ser_map_len, 1, MPI_INT, 0, mpicomm);

    local_ser_map.resize (local_ser_map_len);
    MPI_Scatterv (&global_ser_map[0], proc_ser_map_len.get (), displ_ser_map.get (), MPI_CHAR,
                  &local_ser_map[0], local_ser_map_len, MPI_CHAR, 0, mpicomm);

    if (rank != 0) {
      local_req_rays_map.clear ();
      read_map (local_ser_map, local_req_rays_map);
      (this->rays)[idir].insert (local_req_rays_map.begin (), local_req_rays_map.end ());
    }

    //MPI_Barrier (mpicomm);
    // std::cout << "Rays created in rank " << rank << std::endl;
  }
}

/*
void
ray_cache_t::init_analytical_surf (const std::vector<NS::Atom> & atoms, const NS::surface_type & surf_type,
                                   const double & surf_param, const double & stern_layer, const unsigned & num_threads,
                                   const std::string* configFile)
{
  // NS::NanoShaper ns0 (atoms, surf_type, surf_param, stern_layer, num_threads);
  // ns = ns0;
  ns.initConstructor (atoms, surf_type, surf_param, stern_layer, num_threads,configFile);

  ns.setConfig<double> ("Grid_scale", 2.0);
  ns.setConfig<double> ("Self_Intersections_Grid_Coefficient", 1.5);
  ns.buildAnalyticalSurface();
  std::cout << "\n" << std::endl;

  std::vector<unsigned> idx;
  std::vector<double> coords;

  // right-upper corner
  ns.getGridSize (idx);
  ns.getGridPointCoordinates (idx[0]-1, idx[1]-1, idx[2]-1, coords);
  std::copy (coords.cbegin(), coords.cend(), crossings_t::end);

  // left-bottom corner
  ns.getGridPointCoordinates (0, 0, 0, coords);
  std::copy (coords.cbegin(), coords.cend(), crossings_t::start);
}


void
ray_cache_t::init_analytical_surf_ns (const std::vector<NS::Atom> & atoms, const NS::surface_type & surf_type,
                                      const double & surf_param, const double & stern_layer, const unsigned & num_threads,
                                      double* l_cr, double* r_cr, double scale, const std::string* configFile)
{

  ns.initConstructor (atoms, surf_type, surf_param, stern_layer, num_threads,configFile);
  // set here a consistent grid scale
  ns.setConfig<double> ("Grid_scale", scale );
  ns.setConfig<bool> ("Accurate_Triangulation",true);
  ns.setConfig<double> ("Self_Intersections_Grid_Coefficient", 1.5);

  ns.setConfig<bool> ("Build_epsilon_maps",true);

  // build the grid in the new mode
  ns.setConfig<bool> ("PB_grid_mode",true);

  // impose here the min and max of the cube
  ns.setConfig<double> ("xmin",l_cr[0]);
  ns.setConfig<double> ("ymin",l_cr[1]);
  ns.setConfig<double> ("zmin",l_cr[2]);

  ns.setConfig<double> ("xmax",r_cr[0]);
  ns.setConfig<double> ("ymax",r_cr[1]);
  ns.setConfig<double> ("zmax",r_cr[2]);
  ns.buildAnalyticalSurface();

  // remember to set this to true in order to collect rays intersections data
  ns.setCollectGridRays (true);
  ns.colourGrid();
  // retrieve intersections data from the map
  rays = ns.getRaysMap();

  // crossings_t::start[0] = l_cr[0];
  // crossings_t::start[1] = l_cr[1];
  // crossings_t::start[2] = l_cr[2];
  // crossings_t::end[0] = r_cr[0];
  // crossings_t::end[1] = r_cr[1];
  // crossings_t::end[2] = r_cr[2];

  l_c[0] = l_cr[0];
  l_c[1] = l_cr[1];
  l_c[2] = l_cr[2];
  r_c[0] = r_cr[0];
  r_c[1] = r_cr[1];
  r_c[2] = r_cr[2];
}


void
ray_cache_t::compute_ns_inters (crossings_t & ct)
{

  if (ct.dir == 0) {
    double start_ray[3] = {l_c[ct.dir], ct.point[0], ct.point[1]};
    bool compute_normals = true;


    if (ct.point[0] < l_c[1] || ct.point[1] < l_c[2] || ct.point[0] > r_c[1] || ct.point[1] > r_c[2]) { //out of the molecule
      ct.init = 1;
      return;
    }

    std::cout << "Sending new ray in x direction for NS!" << std::endl;
    ns.setDirection (ct.dir);
    std::vector<std::pair<double,double*>> ints_norms; //intersections and normals
    ns.castAxisOrientedRay (start_ray, r_c[ct.dir], ints_norms, ct.dir, compute_normals);

    if (ints_norms.size() != 0) {
      for (unsigned i = 0; i < ints_norms.size(); i++) {
        ct.inters.push_back (ints_norms[i].first);

        ct.normals.push_back (* (ints_norms[i].second));
        ct.normals.push_back (* (ints_norms[i].second+1));
        ct.normals.push_back (* (ints_norms[i].second+2));
      }
    }

    ct.init = 1; //ray is now initialized
  }

  if (ct.dir == 1) {
    double start_ray[3] = {ct.point[0], l_c[ct.dir], ct.point[1]};
    bool compute_normals = true;


    if (ct.point[0] < l_c[0] || ct.point[1] < l_c[2] || ct.point[0] > r_c[0] || ct.point[1] > r_c[2]) { //out of the molecule
      ct.init = 1;
      return;
    }
    std::cout << "Sending new ray in y direction for NS!" << std::endl;
    ns.setDirection (ct.dir);
    std::vector<std::pair<double,double*>> ints_norms; //intersections and normals
    ns.castAxisOrientedRay (start_ray, r_c[ct.dir], ints_norms, ct.dir, compute_normals);

    if (ints_norms.size() != 0) {
      for (unsigned i = 0; i < ints_norms.size(); i++) {
        ct.inters.push_back (ints_norms[i].first);

        ct.normals.push_back (* (ints_norms[i].second));
        ct.normals.push_back (* (ints_norms[i].second+1));
        ct.normals.push_back (* (ints_norms[i].second+2));
      }
    }

    ct.init = 1; //ray is now initialized
  }

  if (ct.dir == 2) {
    double start_ray[3] = {ct.point[0], ct.point[1], l_c[ct.dir]};
    bool compute_normals = true;


    if (ct.point[0] < l_c[0] || ct.point[1] < l_c[1] || ct.point[0] > r_c[0] || ct.point[1] > r_c[1]) { //out of the molecule
      ct.init = 1;
      return;
    }
    std::cout << "Sending new ray in z direction for NS!" << std::endl;
    ns.setDirection (ct.dir);
    std::vector<std::pair<double,double*>> ints_norms; //intersections and normals
    ns.castAxisOrientedRay (start_ray, r_c[ct.dir], ints_norms, ct.dir, compute_normals);

    if (ints_norms.size() != 0) {
      for (unsigned i = 0; i < ints_norms.size(); i++) {
        ct.inters.push_back (ints_norms[i].first);

        ct.normals.push_back (* (ints_norms[i].second));
        ct.normals.push_back (* (ints_norms[i].second+1));
        ct.normals.push_back (* (ints_norms[i].second+2));
      }
    }

    ct.init = 1; //ray is now initialized
  }

}
*/


void
ray_cache_t::init_analytical_surf_ns (const std::vector<NS::Atom> & atoms, const NS::surface_type & surf_type,
                                      const double & surf_param, const double & stern_layer, const unsigned & num_threads,
                                      double* l_cr, double* r_cr, double scale, const std::string* configFile)
{
  ns = std::make_unique<NS::NanoShaper> (atoms, surf_type, surf_param, stern_layer, num_threads,configFile);
  // set here a consistent grid scale
  ns->setConfig<double> ("Grid_scale", scale);
  ns->setConfig<bool> ("Accurate_Triangulation",true);
  ns->setConfig<double> ("Self_Intersections_Grid_Coefficient", 1.5);

  ns->setConfig<bool> ("Build_epsilon_maps",true);

  // build the grid in the new mode
  ns->setConfig<bool> ("PB_grid_mode",true);

  // impose here the min and max of the cube
  ns->setConfig<double> ("xmin",l_cr[0]);
  ns->setConfig<double> ("ymin",l_cr[1]);
  ns->setConfig<double> ("zmin",l_cr[2]);

  ns->setConfig<double> ("xmax",r_cr[0]);
  ns->setConfig<double> ("ymax",r_cr[1]);
  ns->setConfig<double> ("zmax",r_cr[2]);
  ns->buildAnalyticalSurface ();

  // remember to set this to true in order to collect rays intersections data
  ns->setCollectGridRays (true);
  ns->colourGrid ();
  // retrieve intersections data from the map

  rays = ns->getRaysMap ();

  l_c[0] = l_cr[0];
  l_c[1] = l_cr[1];
  l_c[2] = l_cr[2];
  r_c[0] = r_cr[0];
  r_c[1] = r_cr[1];
  r_c[2] = r_cr[2];
}


void
ray_cache_t::compute_ns_inters (crossings_t & ct)
{

  if (ct.dir == 0) {
    double start_ray[3] = {l_c[ct.dir], ct.point[0], ct.point[1]};
    bool compute_normals = true;


    if (ct.point[0] < l_c[1] || ct.point[1] < l_c[2] || ct.point[0] > r_c[1] || ct.point[1] > r_c[2]) { //out of the molecule
      ct.init = 1;
      return;
    }

    std::cout << "Sending new ray in x direction for NS!" << std::endl;
    ns->setDirection (ct.dir);
    std::vector<std::pair<double,double*>> ints_norms; //intersections and normals
    ns->castAxisOrientedRay (start_ray, r_c[ct.dir], ints_norms, ct.dir, compute_normals);

    if (ints_norms.size () != 0) {
      for (unsigned i = 0; i < ints_norms.size (); i++) {
        ct.inters.push_back (ints_norms[i].first);

        ct.normals.push_back (* (ints_norms[i].second));
        ct.normals.push_back (* (ints_norms[i].second+1));
        ct.normals.push_back (* (ints_norms[i].second+2));
      }
    }

    ct.init = 1; //ray is now initialized
  }

  if (ct.dir == 1) {
    double start_ray[3] = {ct.point[0], l_c[ct.dir], ct.point[1]};
    bool compute_normals = true;


    if (ct.point[0] < l_c[0] || ct.point[1] < l_c[2] || ct.point[0] > r_c[0] || ct.point[1] > r_c[2]) { //out of the molecule
      ct.init = 1;
      return;
    }

    std::cout << "Sending new ray in y direction for NS!" << std::endl;
    ns->setDirection (ct.dir);
    std::vector<std::pair<double,double*>> ints_norms; //intersections and normals
    ns->castAxisOrientedRay (start_ray, r_c[ct.dir], ints_norms, ct.dir, compute_normals);

    if (ints_norms.size () != 0) {
      for (unsigned i = 0; i < ints_norms.size (); i++) {
        ct.inters.push_back (ints_norms[i].first);

        ct.normals.push_back (* (ints_norms[i].second));
        ct.normals.push_back (* (ints_norms[i].second+1));
        ct.normals.push_back (* (ints_norms[i].second+2));
      }
    }

    ct.init = 1; //ray is now initialized
  }

  if (ct.dir == 2) {
    double start_ray[3] = {ct.point[0], ct.point[1], l_c[ct.dir]};
    bool compute_normals = true;


    if (ct.point[0] < l_c[0] || ct.point[1] < l_c[1] || ct.point[0] > r_c[0] || ct.point[1] > r_c[1]) { //out of the molecule
      ct.init = 1;
      return;
    }

    std::cout << "Sending new ray in z direction for NS!" << std::endl;
    ns->setDirection (ct.dir);
    std::vector<std::pair<double,double*>> ints_norms; //intersections and normals
    ns->castAxisOrientedRay (start_ray, r_c[ct.dir], ints_norms, ct.dir, compute_normals);

    if (ints_norms.size () != 0) {
      for (unsigned i = 0; i < ints_norms.size (); i++) {
        ct.inters.push_back (ints_norms[i].first);

        ct.normals.push_back (* (ints_norms[i].second));
        ct.normals.push_back (* (ints_norms[i].second+1));
        ct.normals.push_back (* (ints_norms[i].second+2));
      }
    }

    ct.init = 1; //ray is now initialized
  }

}


//! Convert an crossings_t object
//!  to a vector of bytes that can
//!  be saved to a binary file or transmitted through
//!  a channel such as a socket or sent as an MPI message.
std::vector<unsigned char>
ray_cache_t::write_ct (const crossings_t& ct)
{

  size_t N = sizeof (unsigned) + sizeof (bool) + sizeof (double)*2;
  std::vector<unsigned char> res (N, 0);

  std::copy (&ct.dir, &ct.dir+1, reinterpret_cast<unsigned*> (& (res[0])));
  std::copy (&ct.init, &ct.init+1, reinterpret_cast<bool*> (& (res[sizeof (unsigned)])));
  std::copy (ct.point, ct.point+2, reinterpret_cast<double*> (& (res[sizeof (bool)+sizeof (unsigned)])));

  std::vector<unsigned char> tmp = serialize::write (ct.inters);
  res.insert (res.end (), tmp.begin (), tmp.end ());
  tmp = serialize::write (ct.normals);
  res.insert (res.end (), tmp.begin (), tmp.end ());
  return res;
}

void
ray_cache_t::read_ct (const std::vector<unsigned char>& data, crossings_t& ct)
{

  size_t size_fixed = sizeof (unsigned) + sizeof (bool) + sizeof (double)*2;
  size_t size_non_fixed = 4*sizeof (double); // + sizeof(bool);
  size_t numel_flags = ( (data.size () * sizeof (unsigned char)) - size_fixed) / size_non_fixed;

  std::copy (data.begin (), data.begin () + sizeof (unsigned),
             reinterpret_cast<unsigned char*> (& (ct.dir)));

  auto init = data.begin () + sizeof (unsigned);
  std::copy (init, init + sizeof (bool),
             reinterpret_cast<unsigned char*> (& (ct.init)));

  init = init + sizeof (bool);
  std::copy (init, init + sizeof (double)*2,
             reinterpret_cast<unsigned char*> (& (ct.point)));

  if (numel_flags != 0) {
    init = init + sizeof (double)*2;
    //ct.flags.resize (numel_flags);
    //std::copy (init, init + sizeof(bool)*numel_flags,
    //reinterpret_cast<unsigned char*> (&(ct.flags[0])));

    //init = init + sizeof(bool)*numel_flags;
    ct.inters.resize (numel_flags);
    std::copy (init, init + sizeof (double)*numel_flags,
               reinterpret_cast<unsigned char*> (& (ct.inters[0])));

    init = init + sizeof (double)*numel_flags;
    ct.normals.resize (3*numel_flags);
    std::copy (init, data.end (),
               reinterpret_cast<unsigned char*> (& (ct.normals[0])));
  }
}

std::vector<unsigned char>
ray_cache_t::write_map (const std::map<std::array<double, 2>, crossings_t, map_compare>& container)
{
  size_t numel = container.size (); //number of elements inside the container
  size_t size = numel * (sizeof (double)*2); //array of double

  std::vector<unsigned char> res (size, 0);

  auto destf = reinterpret_cast<std::array<double, 2>*> (& (res[0]));

  for (auto ii = container.begin (); ii != container.end (); ++ii, ++destf)
    *destf = ii->first;

  for (auto ii = container.begin (); ii != container.end (); ++ii) {
    std::vector<unsigned char> tmp = write_ct (ii->second);
    res.insert (res.end (), tmp.begin (), tmp.end ());
  }

  std::vector<size_t> num_flags (numel + 1);
  int pos = 1;
  num_flags[0] = numel;

  for (auto it = container.cbegin (); it != container.cend (); it++) {
    num_flags[pos] = it->second.inters.size ();
    pos++;
  }

  std::vector<unsigned char> tmp2 = serialize::write (num_flags);
  res.insert (res.begin (), tmp2.begin (), tmp2.end ()); //at the begin the info about map size

  return res;
}

void
ray_cache_t::read_map (const std::vector<unsigned char>& data,
                       std::map<std::array<double, 2>, crossings_t, map_compare>& container)
{
  const size_t *psize = (reinterpret_cast<const size_t*> (& (data[0])));
  size_t num_maps = *psize;

  std::vector<size_t> num_flags (num_maps);
  size_t size_num_flags = sizeof (size_t) * num_maps + sizeof (size_t); //da dove iniziano i dati

  psize++;

  for (size_t jj = 0; jj < num_maps; jj++, psize++)
    num_flags[jj] = (*psize);

  auto destf = reinterpret_cast<const std::array<double, 2>*> (psize);

  for (size_t ii = 0; ii < num_maps; ++ii, ++destf)
    container[*destf];

  size_t jj = 0;
  auto destdir = reinterpret_cast<const unsigned*> (destf);

  for (auto ii = container.begin (); ii != container.end (); ++ii) {
    ii->second.dir = *destdir;
    destdir++;
    auto destinit = reinterpret_cast<const bool*> (destdir);
    ii->second.init = *destinit;
    destinit++;
    auto destpoint = reinterpret_cast<const double*> (destinit);
    ii->second.point[0] = *destpoint;
    destpoint++;
    ii->second.point[1] = *destpoint;
    destpoint++;

    if (num_flags[jj] != 0) {
      auto end = destpoint + num_flags[jj];
      ii->second.inters.resize (num_flags[jj]);
      std::copy (destpoint, end, ii->second.inters.begin ());
      destpoint = end;
      end = destpoint + 3*num_flags[jj];
      ii->second.normals.resize (3*num_flags[jj]);
      std::copy (destpoint, end, ii->second.normals.begin ());
      destpoint = end;
    }

    destdir = reinterpret_cast<const unsigned*> (destpoint);
    jj++;
  }
}

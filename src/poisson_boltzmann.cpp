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

#include <mpi.h>

#include "pb_class.h"
#include "readpdb.h"
#include "vtk_class.h"

// #include "wrapper_search.h"

// int
// cerca_atomo_wrapper (p8est_t *p4est,
//                      p4est_topidx_t which_tree,
//                      p8est_quadrant_t *quadrant,
//                      p4est_locidx_t local_num,
//                      void *point)
// {

//   poisson_boltzmann * pb_wrapper = static_cast<poisson_boltzmann*> (pb_global);
//   return pb_wrapper->cerca_atomo (p4est, which_tree, quadrant, local_num,point);
// }

// static char filename[255];

double crossings_t::start[3] = {0., 0., 0.};
double crossings_t::end[3] = {0., 0., 0.};

int_coord_t ray_cache_t::count_cache = 0;
int_coord_t ray_cache_t::count_new = 0;


void
print_map (const std::array<std::map<std::array<double, 2>, crossings_t, map_compare>, 3>& r);

void
print_point (const std::array<std::vector<std::array<double, 2>>,3>& r);
void
save_ray_cache (nlohmann::json& j, const std::map<std::array<double, 2>, crossings_t, map_compare>& r);


int
main (int argc, char **argv)
{

  MPI_Init (&argc, &argv);

  // int recursive, partforcoarsen, balance;
  MPI_Comm mpicomm = MPI_COMM_WORLD;
  int rank, size;
  tmesh_3d tmsh;

  MPI_Comm_rank (mpicomm, &rank);
  MPI_Comm_size (mpicomm, &size);

  poisson_boltzmann pb;
  ray_cache_t ray_cache;

  // pb_global = (void *) (&pb);

  if (pb.parse_options (argc, argv))
    return 1;

  if (rank == 0) {
    std::ifstream inputfile (pb.pqrfilename);

    if (pb.filetype == "pdb") {
      std::cout << "Reading PDB file: " << pb.pqrfilename << std::endl;
      read_pdb (pb.pqrfilename, pb.atoms);
      load_radii (pb.radiusfilename, pb.atoms);
      load_charges (pb.chargefilename, pb.atoms);

      if (pb.write_pqr == 1) {
        write_pqr (pb.name_pqr, pb.atoms);
        std::cout << "Wrote PQR file: " << pb.name_pqr << std::endl;
      }
    } else
      pb.read_atoms_from_pqr (inputfile);

    inputfile.close ();
    pb.read_atoms_from_class ();
  }

  MPI_Barrier (mpicomm);

  if (size > 1) {
    pb.broadcast_vectors ();
  }

  // if (rank == 0) {
  // std::cout << "Atom : " << std::endl;
  // pb.write_atoms_to_pqr (std::cout);
  // pb.print_options ();
  // }

  MPI_Barrier (mpicomm);

  // TIC ();
  // pb.create_mesh_ns ();
  pb.create_mesh ();
  // TOC ("create_mesh");

  std::vector<double> ().swap (pb.r_atoms);

  TIC ();

  if (rank == 0) {
    std::cout << "\n=== [ Building Surface with NanoShaper ] ===\n";
    ray_cache.init_analytical_surf_ns (pb.atoms, pb.surf_type, pb.surf_param, pb.stern_layer, pb.num_threads, pb.l_cr, pb.r_cr, pb.scale);
    std::vector<NS::Atom> ().swap (pb.atoms);
    std::cout << "\n============================================\n";
  }

  TOC ("Building Surface with NanoShaper");

  MPI_Barrier (mpicomm);

  TIC ();

  if (rank == 0) {
    std::cout << "\n============ [ Building Grid ] =============\n";
  }

  // Initialize the mesh depending on the settings
  if (pb.mesh_shape == 0 || pb.refine_box == 1 || pb.mesh_shape == 4) {
    pb.init_tmesh_with_refine_scale();
  } else if (pb.mesh_shape == 3) {
    pb.init_tmesh_with_refine_box_scale();
  } else {
    pb.init_tmesh();
  }

  int local_nodes = pb.tmsh.num_local_nodes();
  int local_quads = pb.tmsh.num_local_quadrants();

  std::vector<int> all_nodes, all_quads;

  if (rank == 0) {
    all_nodes.resize (size);
    all_quads.resize (size);
  }

  MPI_Gather (&local_nodes, 1, MPI_INT,
              all_nodes.data(), 1, MPI_INT,
              0, mpicomm);

  MPI_Gather (&local_quads, 1, MPI_INT,
              all_quads.data(), 1, MPI_INT,
              0, mpicomm);

  if (rank == 0) {
    for (int r = 0; r < size; ++r) {
      std::cout << "  [Rank " << r << "] Local nodes     : " << all_nodes[r] << '\n';
      std::cout << "  [Rank " << r << "] Local quadrants : " << all_quads[r] << '\n';
    }

    std::cout << "  [Global] Total nodes     : " << pb.tmsh.num_global_nodes() << '\n';
    std::cout << "  [Global] Total quadrants : " << pb.tmsh.num_global_quadrants() << '\n';
    std::cout << "============================================\n";
  }

  TOC ("Building Grid");




  if (pb.loc_refinement == 1 || pb.mesh_shape == 4) {
    TIC ();
    pb.refine_surface (ray_cache);
    TOC ("refine the box");
  }



  TIC ();

  if (rank == 0) {
    std::cout << "\n========= [ Building Epsilon Map ] =========\n";
  }

  pb.create_markers (ray_cache);

  if (rank == 0)
    std::cout << "============================================\n";

  TOC ("create element markers");



  if ( rank == 0) ray_cache.ns->clean();

  TIC ();

  if (rank == 0) {
    std::cout << "\n========= [ Creating Density Map ] =========\n";
  }

  pb.create_density_map (ray_cache);

  if (rank == 0)
    std::cout << "============================================\n";

  TOC ("create density map");

  TIC();

  if (rank == 0) {
    std::cout << "\n========== [ Assemble Matricies ] ==========\n";
    std::cout << "Selected BCs          : ";

    if (pb.bc ==1)
      std::cout << "Null\n";
    else if (pb.bc ==2)
      std::cout << "Coulombic\n";
    else
      std::cout << "Neumann\n";
  }

  pb.assemple_system_matrix (ray_cache);

  if (rank == 0)
    std::cout << "============================================\n";

  TOC ("Assemble system matrix");

  MPI_Barrier (mpicomm);

  TIC ();

  if (pb.linear_solver_name == "mumps") {
    if (rank == 0)
      std::cout << "\n== [ Starting numerical solution using MUMPS ] ==\n";

    pb.mumps_compute_electric_potential (ray_cache);
  } else if (pb.linear_solver_name == "lis") {
    if (rank == 0)
      std::cout << "\n== [ Starting numerical solution using LIS ] ==\n";

    pb.lis_compute_electric_potential (ray_cache);
  } else {
    std::cerr << "Invalid linear solver selected" << std::endl;
    return 1;
  }

  if (rank == 0)
    std::cout << "============================================\n";

  TOC ("Compute numerical solution");

  if (pb.atoms_write == 1) {
    TIC ();
    pb.write_potential_on_atoms_fast ();
    TOC ("Write potential on atoms")
  }

  if (pb.calc_potential_term > 0 || pb.calc_field_term > 0 || pb.calc_energy > 0) {
    TIC ();
    const bool refined = (pb.loc_refinement == 1 || pb.mesh_shape > 2 || (pb.mesh_shape == 2 && pb.refine_box == 1));
    const bool pot_field_bool = (pb.calc_potential_term > 0 || pb.calc_field_term > 0);

    if (pot_field_bool) {
      if (refined) {
        pb.pot_field (ray_cache);

        if (pb.calc_energy > pb.calc_potential_term && pb.calc_energy > pb.calc_field_term)
          pb.energy (ray_cache);
      } else {
        pb.pot_field_fast (ray_cache);

        if (pb.calc_energy > pb.calc_potential_term && pb.calc_energy > pb.calc_field_term)
          pb.energy_fast (ray_cache);
      }
    } else {
      if (refined)
        pb.energy (ray_cache);
      else
        // pb.pot_field_fast (ray_cache);
        pb.energy_fast (ray_cache);
    }


    TOC ("Compute energy")
  }

  if (pb.surf_write == 1) {
    TIC ();
    pb.write_potential_on_surface (ray_cache);
    TOC ("Write potential on the surface")
  }

  if (pb.map_type == "oct") {
    if (pb.potential_map == 1) {
      TIC ();
      pb.export_potential_map (ray_cache);
      TOC ("export potential map");
    }

    if (pb.eps_map == 1) {
      TIC ();
      pb.export_tmesh (ray_cache);
      TOC ("export epsilon map");
    }
  } else if (pb.map_type == "vtu") {
    std::vector<std::string> fieldNames;
    std::vector<std::string> baseNames;
    VTKWriter vtk ("");

    if (pb.potential_map == 1) {
      TIC ();
      std::string fieldname = "phi";
      std::string basename = "potential_map";
      fieldNames.push_back (fieldname);
      baseNames.push_back (basename);
      vtk.setBaseName (basename, rank);
      vtk.writeFieldVtuBinary (pb.tmsh, *pb.phi, fieldname);
      TOC ("export potential map new");
    }

    if (pb.eps_map == 1) {
      TIC ();
      std::string fieldname = "eps";
      std::string basename = "eps_map";
      fieldNames.push_back (fieldname);
      baseNames.push_back (basename);
      vtk.setBaseName (basename, rank);
      vtk.writeFieldVtuBinary (pb.tmsh, *pb.epsilon_nodes, fieldname);
      TOC ("export epsilon map new");
    }

    MPI_Barrier (mpicomm);

    if (rank == 0)
      if (pb.potential_map == 1 || pb.eps_map == 1)
        vtk.createPvtuFile (fieldNames, baseNames, size);
  } else
    std::cout << "\n Wrong type of map output! "<<std::endl;



  if (rank == 0) {
    std::cout<<std::endl;
    print_timing_report ();

    // if (pb.surf_type != 2)
    // {
    // //Save ray_cache:
    // for (int i = 0; i < 3; ++i)
    // {
    // // std::cout << "Direzione: "<< i <<std::endl;
    // // std::cout << std::endl;

    // nlohmann::json j;
    // save_ray_cache (j, ray_cache.rays[i]);

    // // std::cout << "Count cached rays: " << ray_cache.count_cache_dir[i] << std::endl;
    // // std::cout << "Count new rays: " << ray_cache.count_new_dir[i] << std::endl;
    // // std::cout << std::endl;
    // std::ofstream ray_cached_file;
    // std::string filename = "ray_cache_";
    // std::string extension = ".json";
    // filename += std::to_string(i);
    // filename += extension;
    // ray_cached_file.open (filename.c_str ());

    // if (ray_cached_file.is_open ())
    // ray_cached_file << j;

    // ray_cached_file.close ();

    // //Alternative way to save results:
    // print_map (ray_cache.rays);
    // }
    // }
  }

  MPI_Barrier (mpicomm);

  MPI_Finalize ();

  return 0;

}


void
print_point (const std::array<std::vector<std::array<double, 2>>,3>& r)
{
  std::ofstream ray_cached_file;

  // ray_cached_file.open ("ray_cache_ns.txt");
  for (int i = 0; i < 3; ++i) {
    std::string filename = "ray_point_ns_";
    std::string extension = ".txt";
    filename += std::to_string (i);
    filename += extension;
    ray_cached_file.open (filename.c_str ());

    if (ray_cached_file.is_open ()) {
      for (auto it = r[i].begin (); it !=r[i].end (); ++it) {
        ray_cached_file <<std::setprecision (9)<< "[[" << (*it)[0] << ", " << (*it)[1] << "]" << std::endl;
      }
    }

    ray_cached_file.close ();
  }

}


void
print_map (const std::array<std::map<std::array<double, 2>, crossings_t, map_compare>, 3>& r)
{
  std::ofstream ray_cached_file;

  // ray_cached_file.open ("ray_cache_ns.txt");
  for (int i = 0; i < 3; ++i) {
    std::string filename = "ray_cache_ns_";
    std::string extension = ".txt";
    filename += std::to_string (i);
    filename += extension;
    ray_cached_file.open (filename.c_str ());

    if (ray_cached_file.is_open ()) {
      ray_cached_file << "Count cached rays: " << ray_cache_t::count_cache << std::endl;
      ray_cached_file << "Count new rays: " << ray_cache_t::count_new << std::endl;
      int count = 0;

      for (auto it : r[i]) {
        ray_cached_file <<std::setprecision (9)<< "[[" << it.first.at (0) << ", " << it.first.at (1) << "]";

        if (it.second.inters.size ()>0) {
          count++;
        }

        for (int i = 0; i < it.second.inters.size (); i++)
          ray_cached_file << ", " << it.second.inters[i];

        ray_cached_file << "]" << std::endl;
      }

      ray_cached_file << std::endl;
      ray_cached_file << "num raggi inters: " << count << std::endl;
    }

    ray_cached_file.close ();
  }

}

/*
void
save_ray_cache (nlohmann::json& j, const std::map<std::array<double, 2>, crossings_t, map_compare>& r)
{
  std::vector<double> ints;

  for (auto it : r) {
    ints.clear();

    j += nlohmann::json{{"ray", it.first}, {"inters", (it.second.inters)}, {"normals", (it.second.normals)}};

    // if (it.second.inters.size()>0)
    // {
    // for (int i = 0; i < it.second.inters.size(); ++i)
    // {
    // std::cout<< std::hypot(it.first[0],it.first[1], it.second.inters[i]) << "  ";
    // }
    // std::cout<<std::endl;
    // }
  }
}

*/
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

#include "pb_class.h"
#include "GetPot"

#include <bim_distributed_vector.h>
#include <quad_operators_3d.h>
#include <mumps_class.h>
#include <lis_class.h>
// #include <lis_class_distributed.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdio.h>

#include <p8est.h>
#include <random>


void
poisson_boltzmann::create_mesh ()
{
  int rank;
  MPI_Comm_rank (mpicomm, &rank);

  double eps_out = 4.0*pi*e_0*e_out*kb*T*Angs/ (e*e); //adim e_out
  double C_0 = 1.0e3*N_av*ionic_strength; //Bulk concentration of monovalent species
  double k2 = 2.0*C_0*Angs*Angs*e*e/ (e_0*e_out*kb*T);
  double k = std::sqrt (k2);

  int nx, ny, nz;
  double scale_tmp, scale_x, scale_y, scale_z;
  double lmax = 0;
  double max_len = 0;
  double l[3];
  scale_level = unilevel;
  double maxradius = *std::max_element (r_atoms.begin (), r_atoms.end ());

  auto comp_pos_x = [] (const std::array<double, 3>& a1, const std::array<double, 3>& a2) -> bool {
    return a1[0] < a2[0];
  };
  auto comp_pos_y = [] (const std::array<double, 3>& a1, const std::array<double, 3>& a2) -> bool {
    return a1[1] < a2[1];
  };
  auto comp_pos_z = [] (const std::array<double, 3>& a1, const std::array<double, 3>& a2) -> bool {
    return a1[2] < a2[2];
  };


  auto minmax_x = std::minmax_element (pos_atoms.begin (), pos_atoms.end (), comp_pos_x);
  auto minmax_y = std::minmax_element (pos_atoms.begin (), pos_atoms.end (), comp_pos_y);
  auto minmax_z = std::minmax_element (pos_atoms.begin (), pos_atoms.end (), comp_pos_z);



  net_charge = std::accumulate (charge_atoms.begin(), charge_atoms.end(), 0.0);
  int num_atoms = charge_atoms.size ();

  if (rank == 0) {
    std::cout << "\n========== [ System Information ] ==========\n";
    std::cout << "  Number of atoms    : " << num_atoms << '\n';
    std::cout << "  Size protein [Å]   : ";
    std::cout << "[" << (*minmax_x.second)[0] - (*minmax_x.first)[0] + 2*maxradius << ", "
              << (*minmax_y.second)[1] - (*minmax_y.first)[1] + 2*maxradius << ", "
              << (*minmax_z.second)[2] - (*minmax_z.first)[2] + 2*maxradius << "]\n";

    if (std::fabs (net_charge - std::round (net_charge)) > 1.e-5)
      std::cerr << "  [WARNING] Net charge is not an integer: " << net_charge << '\n';

    std::cout << "  Solute epsilon     : " << e_in << '\n';
    std::cout << "  Solvent epsilon    : " << e_out << '\n';
    std::cout << "  Temperature        : " << T << " [K] \n";
    std::cout << "  Ionic strength     : " << ionic_strength << " [mol/L] \n";
    std::cout << "============================================\n\n";
  }

  if (mesh_shape !=2) {
    l_c[0] = (*minmax_x.first)[0] - maxradius - 2*prb_radius;
    l_c[1] = (*minmax_y.first)[1] - maxradius - 2*prb_radius;
    l_c[2] = (*minmax_z.first)[2] - maxradius - 2*prb_radius;
    r_c[0] = (*minmax_x.second)[0] + maxradius + 2*prb_radius;
    r_c[1] = (*minmax_y.second)[1] + maxradius + 2*prb_radius;
    r_c[2] = (*minmax_z.second)[2] + maxradius + 2*prb_radius;

    for (int kk = 0; kk < 3; ++kk) {
      l[kk] = (r_c[kk] - l_c[kk]);
      cc[kk] = (r_c[kk] + l_c[kk])*0.5;
      lmax = l[kk] > lmax ? l[kk] : lmax;
    }

    // For random displacement of the grid
    if (rand_center == 1) {
      std::random_device rd; // Will be used to obtain a seed for the random number engine
      std::mt19937 gen (rd ()); // Standard mersenne_twister_engine seeded with rd()
      std::uniform_real_distribution<> dis (-1./scale*0.5, 1./scale*0.5);
      double tmp_cc[3];

      for (int n = 0; n < 3; ++n) {
        tmp_cc[n] = cc[n];
        cc[n] = tmp_cc[n] + dis (gen);
      }

      std::cout << tmp_cc[0] << "  " << cc[0] << "  " << std::abs (tmp_cc[0] -cc[0]) << std::endl;
      std::cout << tmp_cc[1] << "  " << cc[1] << "  " << std::abs (tmp_cc[1] -cc[1]) << std::endl;
      std::cout << tmp_cc[2] << "  " << cc[2] << "  " << std::abs (tmp_cc[2] -cc[2]) << std::endl;
    }

    for (int kk = 0; kk < 3; ++kk) {
      ll[kk] = cc[kk] - lmax*0.5;
      rr[kk] = cc[kk] + lmax*0.5;
    }

    //stretched box with perfil1
    l_c[0] -= l[0]*0.5* (1.0/perfil1 - 1);
    l_c[1] -= l[1]*0.5* (1.0/perfil1 - 1);
    l_c[2] -= l[2]*0.5* (1.0/perfil1 - 1);
    r_c[0] += l[0]*0.5* (1.0/perfil1 - 1);
    r_c[1] += l[1]*0.5* (1.0/perfil1 - 1);
    r_c[2] += l[2]*0.5* (1.0/perfil1 - 1);

    //cubic box with perfil1
    ll[0] -= lmax*0.5* (1.0/perfil1 - 1);
    ll[1] -= lmax*0.5* (1.0/perfil1 - 1);
    ll[2] -= lmax*0.5* (1.0/perfil1 - 1);
    rr[0] += lmax*0.5* (1.0/perfil1 - 1);
    rr[1] += lmax*0.5* (1.0/perfil1 - 1);
    rr[2] += lmax*0.5* (1.0/perfil1 - 1);

    l_cr[0] = l_c[0];
    r_cr[0] = r_c[0];
    l_cr[1] = l_c[1];
    r_cr[1] = r_c[1];
    l_cr[2] = l_c[2];
    r_cr[2] = r_c[2];
  }

  if (mesh_shape == 0) {

    //cubic box with max perfil2
    double size = 1.0/scale;
    scale_level = 0;

    for (int ii = 0; ii<29; ++ii) {
      size *= 2;
      scale_level ++;

      if (lmax/size < perfil2)
        break;
    }

    ll[0] = cc[0] - size/2;
    ll[1] = cc[1] - size/2;
    ll[2] = cc[2] - size/2;
    rr[0] = cc[0] + size/2;
    rr[1] = cc[1] + size/2;
    rr[2] = cc[2] + size/2;


    nx = (int) ( (r_cr[0] - cc[0])*scale + 0.5);
    ny = (int) ( (r_cr[1] - cc[1])*scale + 0.5);
    nz = (int) ( (r_cr[2] - cc[2])*scale + 0.5);

    //refined box
    l_cr[0] = cc[0] - nx*1.0/scale;
    l_cr[1] = cc[1] - ny*1.0/scale;
    l_cr[2] = cc[2] - nz*1.0/scale;
    r_cr[0] = cc[0] + nx*1.0/scale;
    r_cr[1] = cc[1] + ny*1.0/scale;
    r_cr[2] = cc[2] + nz*1.0/scale;

    double dist = size/2 - lmax*0.5;
    pot_bc = std::exp (-k*dist)/ (dist*eps_out);

    if (rank == 0) {
      std::cout << "========== [ Domain Information ] ==========\n";
      std::cout << "  Scale:  " << scale << "\n";

      std::cout << "  Center of the System [Å]:";
      std::cout << "  [" << cc[0] << ", " << cc[1] << ", " << cc[2] << "]\n";

      std::cout << "  Perfil outer box:  " << perfil2 << "\n";
      std::cout << "  Complete Domain Box Size [Å]:\n";
      std::cout << "      x = [" << ll[0] << ", " << rr[0] << "]\n";
      std::cout << "      y = [" << ll[1] << ", " << rr[1] << "]\n";
      std::cout << "      z = [" << ll[2] << ", " << rr[2] << "]\n";

      std::cout << "  Perfil uniform grid:  " << perfil1 << "\n";
      std::cout << "  Uniform grid Size [Å]:\n";
      std::cout << "      x = [" << l_cr[0] << ", " << r_cr[0] << "]\n";
      std::cout << "      y = [" << l_cr[1] << ", " << r_cr[1] << "]\n";
      std::cout << "      z = [" << l_cr[2] << ", " << r_cr[2] << "]\n";

      std::cout << "  Number of Subdivisions in the Uniform grid:";
      std::cout << "  nx = " << nx * 2 << "  ny = " << ny * 2 << "  nz = " << nz * 2 << '\n';

      std::cout << "============================================\n";
    }

    simple_conn_num_vertices = 8;
    simple_conn_num_trees = 1;

    simple_conn_p = std::make_unique<double[]> (simple_conn_num_vertices*3);
    simple_conn_t = std::make_unique<p4est_topidx_t[]> (simple_conn_num_vertices);

    auto tmp_p = {ll[0], ll[1], ll[2],
                  rr[0], ll[1], ll[2],
                  ll[0], rr[1], ll[2],
                  rr[0], rr[1], ll[2],
                  ll[0], ll[1], rr[2],
                  rr[0], ll[1], rr[2],
                  ll[0], rr[1], rr[2],
                  rr[0], rr[1], rr[2]
                 };
    auto tmp_t = {1, 2, 3, 4, 5, 6, 7, 8};

    std::copy (tmp_p.begin (), tmp_p.end (), simple_conn_p.get ());
    std::copy (tmp_t.begin (), tmp_t.end (), simple_conn_t.get ());

    for (int i = 0; i<6; i++)
      bcells.push_back (std::make_pair (0, i));
  } else if (mesh_shape == 1) {
    l_cr[0] = ll[0];
    l_cr[1] = ll[1];
    l_cr[2] = ll[2];
    r_cr[0] = rr[0];
    r_cr[1] = rr[1];
    r_cr[2] = rr[2];
    scale = (1<<unilevel)/ (rr[0]-ll[0]);

    if (refine_box == 1) {
      //cubic box with perfil2
      ll[0] = cc[0] - lmax*0.5*1.0/perfil2;
      ll[1] = cc[1] - lmax*0.5*1.0/perfil2;
      ll[2] = cc[2] - lmax*0.5*1.0/perfil2;
      rr[0] = cc[0] + lmax*0.5*1.0/perfil2;
      rr[1] = cc[1] + lmax*0.5*1.0/perfil2;
      rr[2] = cc[2] + lmax*0.5*1.0/perfil2;
      scale_tmp = (1<<unilevel)/ (rr[0]-ll[0]);

      nx = (int) ( (r_cr[0] - cc[0])*scale_tmp + 0.5);
      ny = (int) ( (r_cr[1] - cc[1])*scale_tmp + 0.5);
      nz = (int) ( (r_cr[2] - cc[2])*scale_tmp + 0.5);

      //refined stretched box
      l_cr[0] = cc[0] - nx*1.0/scale_tmp;
      l_cr[1] = cc[1] - ny*1.0/scale_tmp;
      l_cr[2] = cc[2] - nz*1.0/scale_tmp;
      r_cr[0] = cc[0] + nx*1.0/scale_tmp;
      r_cr[1] = cc[1] + ny*1.0/scale_tmp;
      r_cr[2] = cc[2] + nz*1.0/scale_tmp;
    }

    double dist = ((rr[0]-ll[0]) - lmax)*0.5;
    pot_bc = std::exp (-k*dist)/ (dist*eps_out);

    if (rank == 0) {
      std::cout << "========== [ Domain Information ] ==========\n";
      std::cout << "  Scale:  " << scale << "\n";

      std::cout << "  Center of the System [Å]:";
      std::cout << "  [" << cc[0] << ", " << cc[1] << ", " << cc[2] << "]\n";

      std::cout << "  Perfil box:  " << perfil1 << "\n";
      std::cout << "  Complete Domain Box Size [Å]:\n";
      std::cout << "      x = [" << ll[0] << ", " << rr[0] << "]\n";
      std::cout << "      y = [" << ll[1] << ", " << rr[1] << "]\n";
      std::cout << "      z = [" << ll[2] << ", " << rr[2] << "]\n";

      std::cout << "  Number of Subdivisions:";
      std::cout << "  nx = " << nx * 2 << "  ny = " << ny * 2 << "  nz = " << nz * 2 << '\n';

      std::cout << "============================================\n";
    }

    simple_conn_num_vertices = 8;
    simple_conn_num_trees = 1;

    simple_conn_p = std::make_unique<double[]> (simple_conn_num_vertices*3);
    simple_conn_t = std::make_unique<p4est_topidx_t[]> (simple_conn_num_vertices);

    auto tmp_p = {ll[0], ll[1], ll[2],
                  rr[0], ll[1], ll[2],
                  ll[0], rr[1], ll[2],
                  rr[0], rr[1], ll[2],
                  ll[0], ll[1], rr[2],
                  rr[0], ll[1], rr[2],
                  ll[0], rr[1], rr[2],
                  rr[0], rr[1], rr[2]
                 };
    auto tmp_t = {1, 2, 3, 4, 5, 6, 7, 8};

    std::copy (tmp_p.begin (), tmp_p.end (), simple_conn_p.get ());
    std::copy (tmp_t.begin (), tmp_t.end (), simple_conn_t.get ());

    for (int i = 0; i<6; i++)
      bcells.push_back (std::make_pair (0, i));
  } else if (mesh_shape == 2) {

    l_cr[0] = l_c[0];
    l_cr[1] = l_c[1];
    l_cr[2] = l_c[2];
    r_cr[0] = r_c[0];
    r_cr[1] = r_c[1];
    r_cr[2] = r_c[2];
    scale = (1<<unilevel)/ (r_c[0]-l_c[0]);

    double dist = ((r_cr[0]-l_cr[0]) - lmax)*0.5;
    pot_bc = std::exp (-k*dist)/ (dist*eps_out);

    if (rank == 0) {
      std::cout << "========== [ Domain Information ] ==========\n";

      std::cout << "  Scale:  " << scale << "\n";

      std::cout << "  Center of the System [Å]:";
      std::cout << "  [" << (r_c[0] + l_c[0])*0.5 << ", "
                << (r_c[1] + l_c[1])*0.5 << ", "
                << (r_c[2] + l_c[2])*0.5 << "]\n";

      std::cout << "  Complete Domain Box Size [Å]:\n";
      std::cout << "      x = [" << l_cr[0] << ", " << r_cr[0] << "]\n";
      std::cout << "      y = [" << l_cr[1] << ", " << r_cr[1] << "]\n";
      std::cout << "      z = [" << l_cr[2] << ", " << r_cr[2] << "]\n";

      std::cout << "  Number of Subdivisions:";
      std::cout << "  nx = " << (1<<unilevel) << "  ny = " << (1<<unilevel) << "  nz = " << (1<<unilevel) << '\n';

      std::cout << "============================================\n";
    }

    if (refine_box == 1) {
      double nx_tmp, ny_tmp, nz_tmp;
      double cc_tmp[3];

      scale_x = (1<<unilevel)/ (r_c[0]-l_c[0]);
      scale_y = (1<<unilevel)/ (r_c[1]-l_c[1]);
      scale_z = (1<<unilevel)/ (r_c[2]-l_c[2]);

      cc[0] = (r_c[0]+l_c[0])*0.5;
      cc[1] = (r_c[1]+l_c[1])*0.5;
      cc[2] = (r_c[2]+l_c[2])*0.5;

      cc_tmp[0] = (r_cr[0]+l_cr[0])*0.5;
      cc_tmp[1] = (r_cr[1]+l_cr[1])*0.5;
      cc_tmp[2] = (r_cr[2]+l_cr[2])*0.5;

      nx_tmp = (int) ( (cc[0] - cc_tmp[0])*scale_x + 0.5);
      ny_tmp = (int) ( (cc[1] - cc_tmp[1])*scale_y + 0.5);
      nz_tmp = (int) ( (cc[2] - cc_tmp[2])*scale_z + 0.5);

      cc_tmp[0] = cc[0] + nx_tmp*1.0/scale_x;
      cc_tmp[1] = cc[1] + ny_tmp*1.0/scale_y;
      cc_tmp[2] = cc[2] + nz_tmp*1.0/scale_z;

      nx = (int) ( (r_cr[0] - cc_tmp[0])*scale_x + 0.5);
      ny = (int) ( (r_cr[1] - cc_tmp[1])*scale_y + 0.5);
      nz = (int) ( (r_cr[2] - cc_tmp[2])*scale_z + 0.5);
      //refined stretched box
      l_cr[0] = cc_tmp[0] - nx*1.0/scale_x;
      l_cr[1] = cc_tmp[1] - ny*1.0/scale_y;
      l_cr[2] = cc_tmp[2] - nz*1.0/scale_z;
      r_cr[0] = cc_tmp[0] + nx*1.0/scale_x;
      r_cr[1] = cc_tmp[1] + ny*1.0/scale_y;
      r_cr[2] = cc_tmp[2] + nz*1.0/scale_z;

      std::cout << "xb: " << l_cr[0] << ", " << r_cr[0] << std::endl;
      std::cout << "yb: " << l_cr[1] << ", " << r_cr[1] << std::endl;
      std::cout << "zb: " << l_cr[2] << ", " << r_cr[2] << "\n" << std::endl;
    }

    simple_conn_num_vertices = 8;
    simple_conn_num_trees = 1;

    simple_conn_p = std::make_unique<double[]> (simple_conn_num_vertices*3);
    simple_conn_t = std::make_unique<p4est_topidx_t[]> (simple_conn_num_vertices);

    auto tmp_p = {l_c[0], l_c[1], l_c[2],
                  r_c[0], l_c[1], l_c[2],
                  l_c[0], r_c[1], l_c[2],
                  r_c[0], r_c[1], l_c[2],
                  l_c[0], l_c[1], r_c[2],
                  r_c[0], l_c[1], r_c[2],
                  l_c[0], r_c[1], r_c[2],
                  r_c[0], r_c[1], r_c[2]
                 };
    auto tmp_t = {1, 2, 3, 4, 5, 6, 7, 8};

    std::copy (tmp_p.begin (), tmp_p.end (), simple_conn_p.get ());
    std::copy (tmp_t.begin (), tmp_t.end (), simple_conn_t.get ());

    for (int i = 0; i<6; i++)
      bcells.push_back (std::make_pair (0, i));

  } else if (mesh_shape == 3) {
    //cubic box with max perfil2
    double size = 1.0/scale;
    double scale_min_box = 0.5;
    scale_level = 0;
    scale_level_min_box = 0;


    for (int ii = 0; ii<28; ++ii) {
      size *= 2;
      scale_level ++;

      if (lmax/size < perfil2)
        break;
    }

    for (int ii = 0; ii<scale_level; ++ii) {
      scale_level_min_box ++;

      if ( (std::pow (2,scale_level_min_box)+1)/size >= scale_min_box)
        break;
    }

    ll[0] = cc[0] - size/2;
    ll[1] = cc[1] - size/2;
    ll[2] = cc[2] - size/2;
    rr[0] = cc[0] + size/2;
    rr[1] = cc[1] + size/2;
    rr[2] = cc[2] + size/2;

    //refined box NS
    nx = (int) ( (r_cr[0] - cc[0])*scale + 0.5);
    ny = (int) ( (r_cr[1] - cc[1])*scale + 0.5);
    nz = (int) ( (r_cr[2] - cc[2])*scale + 0.5);

    l_cr[0] = cc[0] - nx*1.0/scale;
    l_cr[1] = cc[1] - ny*1.0/scale;
    l_cr[2] = cc[2] - nz*1.0/scale;
    r_cr[0] = cc[0] + nx*1.0/scale;
    r_cr[1] = cc[1] + ny*1.0/scale;
    r_cr[2] = cc[2] + nz*1.0/scale;

    //refined box FOCUS

    double err_l;
    l_box[0] = cc_focusing[0] - std::round (n_grid/2.)/scale;
    l_box[1] = cc_focusing[1] - std::round (n_grid/2.)/scale;
    l_box[2] = cc_focusing[2] - std::round (n_grid/2.)/scale;
    r_box[0] = cc_focusing[0] + std::round (n_grid/2.)/scale;
    r_box[1] = cc_focusing[1] + std::round (n_grid/2.)/scale;
    r_box[2] = cc_focusing[2] + std::round (n_grid/2.)/scale;


    if ( (r_box[0]- l_box[0])>= (r_cr[0] - l_cr[0])) {
      r_box[0] = r_cr[0];
      l_box[0] = l_cr[0];
    } else {
      if (l_box[0] <= l_cr[0]) {
        err_l = l_cr[0]-l_box[0];
        l_box[0] = l_box[0] + err_l;
        r_box[0] = r_box[0] + err_l;
      }

      if (r_box[0] >= r_cr[0]) {
        err_l = r_box[0]-r_cr[0];
        l_box[0] = l_box[0] - err_l;
        r_box[0] = r_box[0] - err_l;
      }
    }

    if ( (r_box[1]- l_box[1])>= (r_cr[1] - l_cr[1])) {
      r_box[1] = r_cr[1];
      l_box[1] = l_cr[1];
    } else {
      if (l_box[1] <= l_cr[1]) {
        err_l = l_cr[1]-l_box[1];
        l_box[1] = l_box[1] + err_l;
        r_box[1] = r_box[1] + err_l;
      }

      if (r_box[1] >= r_cr[1]) {
        err_l = r_box[1]-r_cr[1];
        l_box[1] = l_box[1] - err_l;
        r_box[1] = r_box[1] - err_l;
      }
    }

    if ( (r_box[2]- l_box[2])>= (r_cr[2] - l_cr[2])) {
      r_box[2] = r_cr[2];
      l_box[2] = l_cr[2];
    } else {
      if (l_box[2] <= l_cr[2]) {
        err_l = l_cr[2]-l_box[2];
        l_box[2] = l_box[2] + err_l;
        r_box[2] = r_box[2] + err_l;
      }

      if (r_box[2] >= r_cr[2]) {
        err_l = r_box[2]-r_cr[2];
        r_box[2] = r_box[2] - err_l;
      }
    }

    //calcolo outlevel
    unsigned int ratio_l_b;
    double len_box_foc = n_grid/scale;
    double len_box = rr[0]-ll[0];

    for (int ii = 1; ii<=5; ++ii) {
      len_box /= 2.0;

      if (len_box <= len_box_foc) {
        ratio_l_b =ii;
        break;
      }
    }

    outlevel = ratio_l_b;
    // outlevel = 1;
    ////////////////////////////////////////


    if (rank == 0) {
      std::cout << "========== [ Domain Information ] ==========\n";


      std::cout << "  Center of the System [Å]:";
      std::cout << "  [" << cc[0] << ", " << cc[1] << ", " << cc[2] << "]\n";

      std::cout << "  Center of the focusing [Å]:";
      std::cout << "  [" << cc_focusing[0] << ", " << cc_focusing[1] << ", " << cc_focusing[2] << "]\n";

      std::cout << "  Scale in the box focusing:  " << scale << "\n";

      std::cout << "  Complete Domain Box Size [Å]:\n";
      std::cout << "      x = [" << ll[0] << ", " << rr[0] << "]\n";
      std::cout << "      y = [" << ll[1] << ", " << rr[1] << "]\n";
      std::cout << "      z = [" << ll[2] << ", " << rr[2] << "]\n";

      std::cout << "  Focusing Box Size [Å]:\n";
      std::cout << "      x = [" << l_box[0] << ", " << r_box[0] << "]\n";
      std::cout << "      y = [" << l_box[1] << ", " << r_box[1] << "]\n";
      std::cout << "      z = [" << l_box[2] << ", " << r_box[2] << "]\n";


      std::cout << "  Number of Subdivisions in the focusing Box:";
      std::cout << "  nx = " << n_grid << "  ny = " << n_grid << "  nz = " << n_grid << '\n';

      std::cout << "============================================\n";
    }


    simple_conn_num_vertices = 8;
    simple_conn_num_trees = 1;

    simple_conn_p = std::make_unique<double[]> (simple_conn_num_vertices*3);
    simple_conn_t = std::make_unique<p4est_topidx_t[]> (simple_conn_num_vertices);

    auto tmp_p = {ll[0], ll[1], ll[2],
                  rr[0], ll[1], ll[2],
                  ll[0], rr[1], ll[2],
                  rr[0], rr[1], ll[2],
                  ll[0], ll[1], rr[2],
                  rr[0], ll[1], rr[2],
                  ll[0], rr[1], rr[2],
                  rr[0], rr[1], rr[2]
                 };
    auto tmp_t = {1, 2, 3, 4, 5, 6, 7, 8};

    std::copy (tmp_p.begin (), tmp_p.end (), simple_conn_p.get ());
    std::copy (tmp_t.begin (), tmp_t.end (), simple_conn_t.get ());

    for (int i = 0; i<6; i++)
      bcells.push_back (std::make_pair (0, i));
  } else if (mesh_shape == 4) {

    //cubic box with max perfil2

    double size = 1.0/scale_max;

    scale = scale_max;
    maxlevel = 0;

    for (int ii = 0; ii<28; ++ii) {
      std::cout << size << "  " << maxlevel << std::endl;
      size *= 2;
      maxlevel ++;

      if (lmax/size < perfil2)
        break;
    }

    minlevel = (int) (maxlevel - std::sqrt (scale_max/scale_min));

    unilevel = (int) ( (maxlevel + minlevel)*0.5);
    unilevel = unilevel + 1;
    outlevel = minlevel;
    scale_level = maxlevel;

    ll[0] = cc[0] - size/2;
    ll[1] = cc[1] - size/2;
    ll[2] = cc[2] - size/2;
    rr[0] = cc[0] + size/2;
    rr[1] = cc[1] + size/2;
    rr[2] = cc[2] + size/2;

    //refined box NS
    nx = (int) ( (r_cr[0] - cc[0])*scale + 0.5);
    ny = (int) ( (r_cr[1] - cc[1])*scale + 0.5);
    nz = (int) ( (r_cr[2] - cc[2])*scale + 0.5);

    l_cr[0] = cc[0] - nx*1.0/scale;
    l_cr[1] = cc[1] - ny*1.0/scale;
    l_cr[2] = cc[2] - nz*1.0/scale;
    r_cr[0] = cc[0] + nx*1.0/scale;
    r_cr[1] = cc[1] + ny*1.0/scale;
    r_cr[2] = cc[2] + nz*1.0/scale;


    if (rank == 0) {
      std::cout << "cx: " << cc[0]
                << " , cy: " << cc[1]
                << " , cz: " << cc[2] << std::endl;

      std::cout << "x: " << ll[0] << ", " << rr[0] << std::endl;
      std::cout << "y: " << ll[1] << ", " << rr[1] << std::endl;
      std::cout << "z: " << ll[2] << ", " << rr[2] << std::endl;

      std::cout << minlevel <<" "<<maxlevel << "  " << unilevel <<std::endl;
    }

    simple_conn_num_vertices = 8;
    simple_conn_num_trees = 1;

    simple_conn_p = std::make_unique<double[]> (simple_conn_num_vertices*3);
    simple_conn_t = std::make_unique<p4est_topidx_t[]> (simple_conn_num_vertices);

    auto tmp_p = {ll[0], ll[1], ll[2],
                  rr[0], ll[1], ll[2],
                  ll[0], rr[1], ll[2],
                  rr[0], rr[1], ll[2],
                  ll[0], ll[1], rr[2],
                  rr[0], ll[1], rr[2],
                  ll[0], rr[1], rr[2],
                  rr[0], rr[1], rr[2]
                 };
    auto tmp_t = {1, 2, 3, 4, 5, 6, 7, 8};

    std::copy (tmp_p.begin (), tmp_p.end (), simple_conn_p.get ());
    std::copy (tmp_t.begin (), tmp_t.end (), simple_conn_t.get ());

    for (int i = 0; i<6; i++)
      bcells.push_back (std::make_pair (0, i));
  } else if (mesh_shape == 5) {
    l_cr[0] = cc[0] - len/2;
    l_cr[1] = cc[1] - len/2;
    l_cr[2] = cc[2] - len/2;
    r_cr[0] = cc[0] + len/2;
    r_cr[1] = cc[1] + len/2;
    r_cr[2] = cc[2] + len/2;


    if (unilevel == 0)
      scale = (num_trees[0])/ (len);
    else
      scale = (num_trees[0]* (1<<unilevel))/ (len);

    //////////////////////////////
    num_trees[1] = num_trees[0];
    num_trees[2] = num_trees[0];

    /////////////////////////////
    if (rank == 0) {
      std::cout << "x: " << l_cr[0] << ", " << r_cr[0] << std::endl;
      std::cout << "y: " << l_cr[1] << ", " << r_cr[1] << std::endl;
      std::cout << "z: " << l_cr[2] << ", " << r_cr[2] << "\n" << std::endl;
      std::cout << "Number of trees: " << num_trees[0] << std::endl;
    }

    double bound_x = std::abs (r_cr[0]-l_cr[0]);
    double bound_y = std::abs (r_cr[1]-l_cr[1]);
    double bound_z = std::abs (r_cr[2]-l_cr[2]);
    double step[3] = { bound_x/num_trees[0],
                       bound_y/num_trees[1],
                       bound_z/num_trees[2]
                     };
    double bound[3] = {bound_x, bound_y, bound_z};
    make_connectivity_3d (num_trees, step, simple_conn_p,
                          simple_conn_num_vertices, simple_conn_t,
                          simple_conn_num_trees, bcells);

    for (p4est_topidx_t i =0; i < simple_conn_num_vertices; ++i) {
      p4est_topidx_t j = 0;
      simple_conn_p[3*i + j++] += l_cr[0];
      simple_conn_p[3*i + j++] += l_cr[1];
      simple_conn_p[3*i + j] += l_cr[2];
    }
  } else {
    if (rank == 0) {
      std::cout << "x: " << ll[0] << ", " << rr[0] << std::endl;
      std::cout << "y: " << ll[1] << ", " << rr[1] << std::endl;
      std::cout << "z: " << ll[2] << ", " << rr[2] << "\n" << std::endl;
    }

    simple_conn_num_vertices = 8;
    simple_conn_num_trees = 1;

    simple_conn_p = std::make_unique<double[]> (simple_conn_num_vertices*3);
    simple_conn_t = std::make_unique<p4est_topidx_t[]> (simple_conn_num_vertices);

    auto tmp_p = {ll[0], ll[1], ll[2],
                  rr[0], ll[1], ll[2],
                  ll[0], rr[1], ll[2],
                  rr[0], rr[1], ll[2],
                  ll[0], ll[1], rr[2],
                  rr[0], ll[1], rr[2],
                  ll[0], rr[1], rr[2],
                  rr[0], rr[1], rr[2]
                 };
    auto tmp_t = {1, 2, 3, 4, 5, 6, 7, 8};

    std::copy (tmp_p.begin (), tmp_p.end (), simple_conn_p.get ());
    std::copy (tmp_t.begin (), tmp_t.end (), simple_conn_t.get ());

    for (int i = 0; i<6; i++)
      bcells.push_back (std::make_pair (0, i));
  }

  tmsh.read_connectivity (simple_conn_p.get (), simple_conn_num_vertices,
                          simple_conn_t.get (), simple_conn_num_trees);

}



double
poisson_boltzmann::levelsetfun (double x, double y, double z)
{
  double dist = 0.0;

  for (const NS::Atom& i : atoms) {

    dist += std::exp (surf_param * ( (std::pow (x - i.pos[0], 2) +
                                      std::pow (y - i.pos[1], 2) +
                                      std::pow (z - i.pos[2], 2)) /
                                     (i.radius*i.radius) - 1.0));

    if (dist > 1.5)
      break;

  }

  return dist;
}


double
poisson_boltzmann::is_in_ns_surf (ray_cache_t & ray_cache, double x, double y, double z, int dir)
{
  int rank;
  MPI_Comm_rank (mpicomm, &rank);
  double x1 = x;
  double x2 = y;
  double x3 = z;

  if (dir == 0) {
    x1 = y;
    x2 = z;
    x3 = x;
  } else if (dir == 1) {
    x1 = x;
    x2 = z;
    x3 = y;
  }

  crossings_t & ct = ray_cache (x1, x2, dir);

  if (!ct.init && rank != 0) {
    std::array<double, 2> ray = {x1, x2};
    ray_cache.rays[dir].erase (ray);
    return -1.;
  }


  int i = 0;

  if (ct.inters.size () == 0 || x3 < ct.inters[i])
    return 0; //if there are no inters or y_the coord is before the first intersection, the point is outside.

  while (i < ct.inters.size () && x3 >= ct.inters[i]) //go on until the inters is passed
    i++;

  return (i % 2);
}


int
poisson_boltzmann::parse_options (int argc, char **argv)
{
  int rank;
  MPI_Comm_rank (mpicomm, &rank);

  GetPot g (argc, argv);

  // =============================
  // 1. Leggi file dei parametri
  // =============================
  if (!g.search ("--prmfile") && !g.search ("--potfile")) {
    if (rank == 0)
      std::cout << "Warning: No parameters file selected, using the default one."
                << "\nTo select one use --prmfile or --potfile option followed by the desired file.\n";
  }

  // Cerca il file, dando precedenza a --prmfile se entrambi sono presenti
  if (g.search ("--prmfile")) {
    optionsfilename = g.next ("../../data/options.prm");
  } else if (g.search ("--potfile")) {
    optionsfilename = g.next ("../../data/options.prm");
  }

  if (rank == 0)
    std::cout << "Selected parameters file: " << optionsfilename << std::endl;

  std::ifstream optionsfile (optionsfilename);

  if (!optionsfile) {
    if (rank == 0)
      std::cerr << "Cannot find the options file" << std::endl;

    return 1;
  }

  GetPot g2 (optionsfilename.c_str());

  // =============================
  // 2. Leggi parametri input/
  // =============================
  const std::string input_section = "input/";
  std::string filename_from_file;

  filetype = g2 ((input_section + "filetype").c_str(), "pqr");
  pqrfilename = g2 ((input_section + "filename").c_str(), "../../data/1CCM.pqr");
  radiusfilename = g2 ((input_section + "radius_file").c_str(), "../../data/radius.siz");
  chargefilename = g2 ((input_section + "charge_file").c_str(), "../../data/charge.crg");
  write_pqr = g2 ((input_section + "write_pqr").c_str(), 0);
  name_pqr = g2 ((input_section + "name_pqr").c_str(), "output.pqr");



  // =============================
  // 3. Override da riga di comando: --pqrfile
  // =============================
  if (g.search ("--pqrfile")) {
    pqrfilename = g.next ("");
    filetype = "pqr"; // Forza il filetype a pqr se viene specificato un file
  }

  if (rank == 0)
    std::cout << "Selected molecule file:   " << pqrfilename << std::endl;

  std::ifstream pqrfile (pqrfilename);

  if (!pqrfile) {
    if (rank == 0) {
      std::cerr << "Cannot find the pqr file" << std::endl;
      return 1;
    }
  }

  const std::string mesh_options = "mesh/";
  maxlevel = g2 ( (mesh_options + "maxlevel").c_str (), 9);
  minlevel = g2 ( (mesh_options + "minlevel").c_str (), 3);
  unilevel = g2 ( (mesh_options + "unilevel").c_str (), 5);
  outlevel = g2 ( (mesh_options + "outlevel").c_str (), 1);
  loc_refinement = g2 ( (mesh_options + "loc_refinement").c_str (), 0);
  mesh_shape = g2 ( (mesh_options + "mesh_shape").c_str (), 1);
  refine_box = g2 ( (mesh_options + "refine_box").c_str (), 0);
  rand_center = g2 ( (mesh_options + "rand_center").c_str (), 0);
  aligned = g2 ( (mesh_options + "aligned").c_str (), 0);

  if (mesh_shape < 2) {
    perfil1 = g2 ( (mesh_options + "perfil1").c_str (), 0.8);
    perfil2 = g2 ( (mesh_options + "perfil2").c_str (), 0.2);
    scale = g2 ( (mesh_options + "scale").c_str (), 2.0);
  }

  if (mesh_shape == 2) {
    l_c[0] = g2 ( (mesh_options + "x1").c_str (), -128.0);
    r_c[0] = g2 ( (mesh_options + "x2").c_str (), 128.0);
    l_c[1] = g2 ( (mesh_options + "y1").c_str (), -128.0);
    r_c[1] = g2 ( (mesh_options + "y2").c_str (), 128.0);
    l_c[2] = g2 ( (mesh_options + "z1").c_str (), -128.0);
    r_c[2] = g2 ( (mesh_options + "z2").c_str (), 128.0);
    l_cr[0] = g2 ( (mesh_options + "refine_x1").c_str (), -64.0);
    r_cr[0] = g2 ( (mesh_options + "refine_x2").c_str (), 64.0);
    l_cr[1] = g2 ( (mesh_options + "refine_y1").c_str (), -64.0);
    r_cr[1] = g2 ( (mesh_options + "refine_y2").c_str (), 64.0);
    l_cr[2] = g2 ( (mesh_options + "refine_z1").c_str (), -64.0);
    r_cr[2] = g2 ( (mesh_options + "refine_z2").c_str (), 64.0);
  }

  if (mesh_shape == 3) {
    perfil1 = g2 ( (mesh_options + "perfil1").c_str (), 0.8);
    perfil2 = g2 ( (mesh_options + "perfil2").c_str (), 0.2);
    scale = g2 ( (mesh_options + "scale").c_str (), 2.0);
    cc_focusing[0] = g2 ( (mesh_options + "cx_foc").c_str (), 0.0);
    cc_focusing[1] = g2 ( (mesh_options + "cy_foc").c_str (), 0.0);
    cc_focusing[2] = g2 ( (mesh_options + "cz_foc").c_str (), 0.0);
    n_grid = g2 ( (mesh_options + "n_grid").c_str (), 8);
  }

  if (mesh_shape == 4) {
    perfil1 = g2 ( (mesh_options + "perfil1").c_str (), 0.8);
    perfil2 = g2 ( (mesh_options + "perfil2").c_str (), 0.2);
    scale_min = g2 ( (mesh_options + "scale_min").c_str (), 0.5);
    scale_max = g2 ( (mesh_options + "scale_max").c_str (), 2.0);
  }

  if (mesh_shape == 5) {
    num_trees [0] = g2 ( (mesh_options + "num_trees_x").c_str (), 10);
    num_trees [1] = g2 ( (mesh_options + "num_trees_y").c_str (), 10);
    num_trees [2] = g2 ( (mesh_options + "num_trees_z").c_str (), 10);
    len = g2 ( (mesh_options + "lato").c_str (), 50.0);
    perfil1 = g2 ( (mesh_options + "perfil1").c_str (), 0.8);
  }

  const std::string model_options = "model/";
  linearized = g2 ( (model_options + "linearized").c_str (), 1);
  bc = g2 ( (model_options + "bc_type").c_str (), 1);
  ionic_strength = g2 ( (model_options + "ionic_strength").c_str (), 0.145);
  e_in = g2 ( (model_options + "molecular_dielectric_constant").c_str (), 2.);
  e_out = g2 ( (model_options + "solvent_dielectric_constant").c_str (), 80.);
  T = g2 ( (model_options + "T").c_str (), 298.15);
  calc_energy = g2 ( (model_options + "calc_energy").c_str (), 2);
  calc_coulombic = g2 ( (model_options + "calc_coulombic").c_str (), 0);
  calc_potential_term = g2 ( (model_options + "calc_potential_terms").c_str (), 0);
  calc_field_term = g2 ( (model_options + "calc_field_terms").c_str (), 0);
  atoms_write = g2 ( (model_options + "atoms_write").c_str (), 0);
  surf_write = g2 ( (model_options + "surf_write").c_str (), 0);
  surf_write = g2 ( (model_options + "surf_write").c_str (), 0);
  map_type = g2 ( (model_options + "map_type").c_str (), "vtu");
  potential_map = g2 ( (model_options + "potential_map").c_str (), 0);
  eps_map = g2 ( (model_options + "eps_map").c_str (), 0);
  const std::string surf_options = "surface/";
  surf_type_num = g2 ( (surf_options + "surface_type").c_str (), 0);

  if (surf_type_num == 1) surf_type = NS::skin;
  else if (surf_type_num == 0) surf_type = NS::ses;
  // else if (surf_type_num == 2) surf_type = NS::blobby;
  else surf_type = NS::ses;

  surf_param = g2 ( (surf_options + "surface_parameter").c_str (), 0.45);
  stern_layer_surf = g2 ( (surf_options + "stern_layer_surf").c_str (), 0);
  stern_layer = g2 ( (surf_options + "stern_layer_thickness").c_str (), 2.);
  num_threads = g2 ( (surf_options + "number_of_threads").c_str (), 1);

  const std::string alg_options = "algorithm/";
  linear_solver_name = g2 ( (alg_options + "linear_solver").c_str (), "lis");
  linear_solver_options = g2 ( (alg_options + "solver_options").c_str (), "-p ssor -ssor_omega 0.51 -i cgs -tol 1.e-6 -print 2 -conv_cond 2 -tol_w 0");

  const std::string out_options = "output/";
  p4estfilename = g2 ( (out_options + "p4estfilename").c_str (), "poisson_boltzmann_p4est");
  markerfilename = g2 ( (out_options + "markerfilename").c_str (), "poisson_boltzmann_marker_0");
  surffilename = g2 ( (out_options + "surffilename").c_str (), "poisson_boltzmann_surface_0");

  return 0;
}

// ====================================
// Autovettore dominante per matrice 3x3 simmetrica
// Usa il metodo di Jacobi (iterativo, robusto)
// ====================================
void compute_dominant_eigenvector (double cov[3][3], double axis[3])
{
  // Inizializza axis = (1,0,0)
  axis[0] = 1.0;
  axis[1] = 0.0;
  axis[2] = 0.0;

  // Potenza iterativa per il vettore principale
  for (int iter = 0; iter < 20; ++iter) {
    double x = cov[0][0]*axis[0] + cov[0][1]*axis[1] + cov[0][2]*axis[2];
    double y = cov[1][0]*axis[0] + cov[1][1]*axis[1] + cov[1][2]*axis[2];
    double z = cov[2][0]*axis[0] + cov[2][1]*axis[1] + cov[2][2]*axis[2];

    double norm = std::sqrt (x*x + y*y + z*z);

    if (norm < 1e-12) break;

    axis[0] = x / norm;
    axis[1] = y / norm;
    axis[2] = z / norm;
  }
}

void align_atoms_to_Z (std::vector<NS::Atom> &atoms)
{
  if (atoms.empty()) return;

  // 1. Centro geometrico (NON pesato)
  double center[3] = {0.0, 0.0, 0.0};

  for (const auto &a : atoms) {
    center[0] += a.pos[0];
    center[1] += a.pos[1];
    center[2] += a.pos[2];
  }

  center[0] /= atoms.size();
  center[1] /= atoms.size();
  center[2] /= atoms.size();

  // Traslazione
  for (auto &a : atoms) {
    a.pos[0] -= center[0];
    a.pos[1] -= center[1];
    a.pos[2] -= center[2];
  }

  // 2. Matrice di covarianza simmetrica
  double cov[3][3] = {{0.0}};

  for (const auto &a : atoms) {
    cov[0][0] += a.pos[0] * a.pos[0];
    cov[0][1] += a.pos[0] * a.pos[1];
    cov[0][2] += a.pos[0] * a.pos[2];
    cov[1][1] += a.pos[1] * a.pos[1];
    cov[1][2] += a.pos[1] * a.pos[2];
    cov[2][2] += a.pos[2] * a.pos[2];
  }

  cov[1][0] = cov[0][1];
  cov[2][0] = cov[0][2];
  cov[2][1] = cov[1][2];

  // 3. Autovettore principale (metodo potenza)
  double axis[3];
  compute_dominant_eigenvector (cov, axis);

  // Normalizza
  double norm = std::sqrt (axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);

  if (norm > 1e-12) {
    axis[0] /= norm;
    axis[1] /= norm;
    axis[2] /= norm;
  }

  // 4. Rotazione: porta axis su (0,0,1) usando rotazione di Rodrigues
  double z_axis[3] = {0.0, 0.0, 1.0};
  double v[3] = {
    axis[1]*z_axis[2] - axis[2]*z_axis[1],
    axis[2]*z_axis[0] - axis[0]*z_axis[2],
    axis[0]*z_axis[1] - axis[1]*z_axis[0]
  };
  double s = std::sqrt (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
  double c = axis[0]*z_axis[0] + axis[1]*z_axis[1] + axis[2]*z_axis[2];

  double R[3][3];

  if (s < 1e-8) {
    R[0][0] = R[1][1] = R[2][2] = 1.0;
    R[0][1] = R[0][2] = R[1][0] = R[1][2] = R[2][0] = R[2][1] = 0.0;
  } else {
    double vx = v[0]/s, vy = v[1]/s, vz = v[2]/s;
    double k = 1.0 - c;
    R[0][0] = c + vx*vx*k;
    R[0][1] = vx*vy*k - vz*s;
    R[0][2] = vx*vz*k + vy*s;
    R[1][0] = vy*vx*k + vz*s;
    R[1][1] = c + vy*vy*k;
    R[1][2] = vy*vz*k - vx*s;
    R[2][0] = vz*vx*k - vy*s;
    R[2][1] = vz*vy*k + vx*s;
    R[2][2] = c + vz*vz*k;
  }

  // 5. Applica rotazione
  for (auto &a : atoms) {
    double x = a.pos[0], y = a.pos[1], z = a.pos[2];
    a.pos[0] = R[0][0]*x + R[0][1]*y + R[0][2]*z;
    a.pos[1] = R[1][0]*x + R[1][1]*y + R[1][2]*z;
    a.pos[2] = R[2][0]*x + R[2][1]*y + R[2][2]*z;
  }
}

void
poisson_boltzmann::read_atoms_from_pqr (std::basic_istream<char> &inputfile)
{
  static NS::Atom a;
  atoms.clear ();

  while (inputfile >> a)
    atoms.push_back (a);

  if (aligned == 1) {
    align_atoms_to_Z (atoms);
  }

  if (atoms.size() < 4) {
    auto comp = [] (const NS::Atom &a1, const NS::Atom &a2) -> bool {
      return a1.radius < a2.radius;
    };

    auto max_iter = std::max_element (atoms.begin(), atoms.end(), comp);
    std::array<double, 3> max_pos = {
      max_iter->pos[0],
      max_iter->pos[1],
      max_iter->pos[2]
    };

    const double epsilon = 0.001;

    std::array<std::array<int, 3>, 6> directions = {{
        {{-1, 0, 0}},
        {{+1, 0, 0}},
        {{0, -1, 0}},
        {{0, +1, 0}},
        {{0, 0, -1}},
        {{0, 0, +1}}
      }
    };

    for (int ii = 0; ii < 6; ++ii) {
      std::array<double, 3> new_pos = {
        max_pos[0] + epsilon * directions[ii][0],
        max_pos[1] + epsilon * directions[ii][1],
        max_pos[2] + epsilon * directions[ii][2]
      };
      NS::Atom dummy;
      dummy.pos[0] = new_pos[0];
      dummy.pos[1] = new_pos[1];
      dummy.pos[2] = new_pos[2];
      dummy.charge = 0.0;
      dummy.radius = 0.0;
      atoms.push_back (dummy);
    }
  }
}

void
poisson_boltzmann::read_atoms_from_class ()
{
  static std::array<double,3> pos;

  if (atoms_write == 1) {
    int atom_number = 1;

    for (const NS::Atom& i : atoms) {
      pos[0] = i.pos[0];
      pos[1] = i.pos[1];
      pos[2] = i.pos[2];
      pos_atoms.push_back (pos);
      index_atoms.push_back (atom_number);
      charge_atoms.push_back (i.charge);
      r_atoms.push_back (i.radius);
      atom_number++;
    }
  } else {
    for (const NS::Atom& i : atoms) {
      pos[0] = i.pos[0];
      pos[1] = i.pos[1];
      pos[2] = i.pos[2];
      pos_atoms.push_back (pos);
      charge_atoms.push_back (i.charge);
      r_atoms.push_back (i.radius);
    }
  }
}

void
poisson_boltzmann::broadcast_vectors ()
{
  int size_vec = charge_atoms.size ();
  MPI_Bcast (&size_vec, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Ogni processo alloca il vettore
  pos_atoms.resize (size_vec);
  charge_atoms.resize (size_vec);
  r_atoms.resize (size_vec);
  index_atoms.resize (size_vec);

  // Effettuare il broadcast del vettore
  MPI_Bcast (pos_atoms.data (), size_vec * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast (charge_atoms.data (), size_vec, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast (r_atoms.data (), size_vec, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast (index_atoms.data (), size_vec, MPI_INT, 0, MPI_COMM_WORLD);
}

void
poisson_boltzmann::write_atoms_to_pqr (std::basic_ostream<char> &outputfile)
{
  int Atom_number = 1;

  outputfile << std::setw (10) << std::left << "fieldname" << std::setw (12)
             << std::left <<"Atom_number" << std::setw (12) << std::left << "Atom_name" << std::setw (16) << std::left
             << "Residue_name" << std::setw (16) << std::left << "Residue_number" << std::setw (10) << std::left << "X"
             << std::setw (10) << std::left << "Y" << std::setw (10) << std::left
             << "Z" << std::setw (10) << std::left << "Charge" << std::setw (10) << std::left << "Radius" << std::endl;

  for (auto & ii : atoms)
    outputfile << std::setw (10) << std::left << "ATOM" << std::setw (12)
               << std::left << Atom_number++ << std::setw (12) << std::left << ii.ai.name << std::setw (16) << std::left
               << ii.ai.resName << std::setw (16) << std::left << ii.ai.resNum << std::setw (10) << std::left << ii.pos[0]
               << std::setw (10) << std::left << ii.pos[1] << std::setw (10) << std::left
               << ii.pos[2] << std::setw (10) << std::left << ii.charge << std::setw (10) << std::left << ii.radius << std::endl;

}



std::basic_istream<char>&
operator>> (std::basic_istream<char>& inputfile, NS::Atom &a)
{
  int Atom_number;
  std::string Field_name;

  inputfile >> Field_name
            >> Atom_number
            >> a.ai.name
            >> a.ai.resName;

  std::string token;
  inputfile >> token;

  // Verifica se è un numero (resNum) oppure una stringa (chain)
  bool is_number = !token.empty() &&
                   (std::isdigit (token[0]) || token[0] == '-' || token[0] == '+');

  if (is_number) {
    // Era resNum - metti solo il token indietro nello stream
    for (auto it = token.rbegin(); it != token.rend(); ++it) {
      inputfile.putback (*it);
    }

    a.ai.chain.clear(); // Nessuna catena specificata
  } else {
    // Era chain - leggi il resNum successivo
    a.ai.chain = token;
  }

  // Ora leggi i dati numerici
  inputfile >> a.ai.resNum >> a.pos[0] >> a.pos[1] >> a.pos[2]
            >> a.charge >> a.radius;

  if (a.radius < 1.e-5)
    a.radius = 1.0;

  return inputfile;
}

std::basic_istream<char>&
operator>> (std::basic_istream<char>& inputfile, std::array<float,5> &a)
{
  int Atom_number;
  std::string Field_name;
  std::string name;
  std::string resName;
  int resNum;

  inputfile >> Field_name
            >> Atom_number
            >> name
            >> resName
            >> resNum
            >> a[0] // x_pos
            >> a[1] // y_pos
            >> a[2] // z_pos
            >> a[3] // charge
            >> a[4]; // radius

  if (a[4] < 1.e-5)
    a[4] = 1.0;

  return inputfile;
}

void
poisson_boltzmann::init_tmesh ()
{
  for (auto i = 0; i < unilevel; ++i) {
    tmsh.set_refine_marker (uniform_refinement);
    tmsh.refine (0, 1);
  }
}


void
poisson_boltzmann::init_tmesh_with_refine_scale ()
{

  for (auto i = 0; i < outlevel; ++i) {
    tmsh.set_refine_marker (uniform_refinement);
    tmsh.refine (0, 1);
  }

  auto refinement = [this]
  (tmesh_3d::quadrant_iterator q) -> int {
    int currentlevel = static_cast<int> (q->the_quadrant->level);
    int retval = 0;
    double x1, y1, z1;

    if (currentlevel >= this->scale_level)
      retval = 0;
    else {
      for (int ii = 0; ii < 8; ++ii) {
        if (! q->is_hanging (ii)) {
          x1 = q -> p (0, ii);
          y1 = q -> p (1, ii);
          z1 = q -> p (2, ii);

          if ( (x1 >= this->l_c[0]) && (x1 <= this->r_c[0])
               && (y1 >= this->l_c[1]) && (y1 <= this->r_c[1])
               && (z1 >= this->l_c[2]) && (z1 <= this->r_c[2])) {
            retval = 1;
            break;
          }
        }
      }
    }
    return (retval);
  };
  for (auto i = 0; i < scale_level; ++i) {
    tmsh.set_refine_marker (refinement);
    tmsh.refine (0, 1);
  }
  
}


void
poisson_boltzmann::init_tmesh_with_refine_box_scale ()
{

  for (auto i = 0; i < outlevel; ++i) {
    tmsh.set_refine_marker (uniform_refinement);
    tmsh.refine (0, 1);
  }

  auto refinement = [this]
  (tmesh_3d::quadrant_iterator q) -> int {
    int currentlevel = static_cast<int> (q->the_quadrant->level);
    int retval = 0;
    double x1, y1, z1;

    if (currentlevel >= this->scale_level)
      retval = 0;
    else {
      for (int ii = 0; ii < 8; ++ii) {
        if (! q->is_hanging (ii)) {
          x1 = q -> p (0, ii);
          y1 = q -> p (1, ii);
          z1 = q -> p (2, ii);

          if ( (x1 >= this->l_box[0]) && (x1 <= this->r_box[0])
               && (y1 >= this->l_box[1]) && (y1 <= this->r_box[1])
               && (z1 >= this->l_box[2]) && (z1 <= this->r_box[2])) {
            retval = 1;
            break;
          }

          // else if ((x1 >= this->l_c[0]) && (x1 <= this->r_c[0])
          // && (y1 >= this->l_c[1]) && (y1 <= this->r_c[1])
          // && (z1 >= this->l_c[2]) && (z1 <= this->r_c[2])
          // &&currentlevel < this->scale_level_min_box) {
          // retval = 1;
          // break;
          // }
        }
      }
    }

    return (retval);
  };

  for (auto i = 0; i < scale_level; ++i) {
    tmsh.set_refine_marker (refinement);
    tmsh.refine (0, 1);
  }
}

bool
poisson_boltzmann::is_in (const NS::Atom& i,
                          tmesh_3d::quadrant_iterator q)
{
  double tol = p4esttol * (rr[0]-ll[0]);

  if (mesh_shape == 2)
    tol = p4esttol * (r_c[0]-l_c[0]);

  bool retval = false;
  double l, r, t, b, f, bk;

  l = q->p (0, 0);
  r = q->p (0, 7);

  f = q->p (1, 0);
  bk = q->p (1, 7);

  b = q->p (2, 0);
  t = q->p (2, 7);


  // for (int ii = 1; ii < 8; ++ii) {
  // l = q->p (0, ii) < l ? q->p (0, ii) : l;
  // r = q->p (0, ii) > r ? q->p (0, ii) : r;
  // f = q->p (1, ii) < f ? q->p (1, ii) : f;
  // bk = q->p (1, ii) > bk ? q->p (1, ii) : bk;
  // b = q->p (2, ii) < b ? q->p (2, ii) : b;
  // t = q->p (2, ii) > t ? q->p (2, ii) : t;

  // }

  retval = (i.pos[0] > l - tol) && (i.pos[0] <= r - tol); //make sure that the charge is assigned only once
  retval = retval && (i.pos[1] > f - tol) && (i.pos[1] <= bk - tol);
  retval = retval && (i.pos[2] > b - tol) && (i.pos[2] <= t - tol);

  return retval;
}

bool
poisson_boltzmann::is_in_ref (const NS::Atom& i,
                              tmesh_3d::quadrant_iterator q)
{

  bool retval = false;
  double l, r, t, b, f, bk;

  l = q->p (0, 0);
  r = q->p (0, 7);

  f = q->p (1, 0);
  bk = q->p (1, 7);

  b = q->p (2, 0);
  t = q->p (2, 7);


  // for (int ii = 1; ii < 8; ++ii) {
  // l = q->p (0, ii) < l ? q->p (0, ii) : l;
  // r = q->p (0, ii) > r ? q->p (0, ii) : r;
  // f = q->p (1, ii) < f ? q->p (1, ii) : f;
  // bk = q->p (1, ii) > bk ? q->p (1, ii) : bk;
  // b = q->p (2, ii) < b ? q->p (2, ii) : b;
  // t = q->p (2, ii) > t ? q->p (2, ii) : t;

  // }

  retval = (i.pos[0] >= l) && (i.pos[0] <= r);
  retval = retval && (i.pos[1] >= f) && (i.pos[1] <= bk);
  retval = retval && (i.pos[2] >= b) && (i.pos[2] <= t);

  return retval;
}

void
poisson_boltzmann::refine_surface (ray_cache_t & ray_cache)
{
  int rank, size;
  MPI_Comm_size (mpicomm, &size);
  MPI_Comm_rank (mpicomm, &rank);

  int num_cycles = 2;

  if (size == 1 || surf_type_num == 2)
    num_cycles = 1;

  int coars_ref_cycles = (maxlevel - unilevel) > (unilevel - minlevel) ? (maxlevel - unilevel) : (unilevel - minlevel);

  for (int kk = 0; kk < coars_ref_cycles; ++kk) {
    // REFINEMENT
    {
      distributed_vector rcoeff (tmsh.num_owned_nodes ());

      for (int jj = 0; jj < num_cycles; jj++) {
        ray_cache.num_req_rays[0] = 0; //zero at each ref/coarsen cycle
        ray_cache.num_req_rays[1] = 0; //zero at each ref/coarsen cycle
        ray_cache.num_req_rays[2] = 0; //zero at each ref/coarsen cycle
        ray_cache.rays_list[0].clear ();
        ray_cache.rays_list[1].clear ();
        ray_cache.rays_list[2].clear ();

        if (rank == 0 && jj == 0)
          std::cout << "Refinement: " << kk << std::endl;

        for (auto quadrant = tmsh.begin_quadrant_sweep ();
             quadrant != tmsh.end_quadrant_sweep ();
             ++quadrant) {

          for (int ii = 0; ii < 8; ++ii) {

            if (! quadrant->is_hanging (ii)) {
              if (surf_type_num == 2)
                rcoeff[quadrant->gt (ii)] = levelsetfun (quadrant->p (0, ii),
                                            quadrant->p (1, ii),
                                            quadrant->p (2, ii));
              else {
                for (int idir = 0; idir < 3; ++idir) {
                  rcoeff[quadrant->gt (ii)] = is_in_ns_surf (ray_cache,
                                              quadrant->p (0, ii),
                                              quadrant->p (1, ii),
                                              quadrant->p (2, ii), idir);

                  if (rcoeff[quadrant->gt (ii)] < -0.5) {
                    ray_cache.num_req_rays[idir]++;

                    std::array<double, 2> ray;

                    std::vector<int> direzioni = {0,1,2};
                    direzioni.erase (direzioni.begin ()+idir);

                    for (unsigned i = 0; i < direzioni.size (); ++i) {
                      ray[i] = quadrant->p (direzioni[i], ii);
                    }

                    ray_cache.rays_list[idir].insert (ray);
                  }
                }
              }
            } else {
              for (int idir = 0; idir < 3; ++idir) {
                double pippo = is_in_ns_surf (ray_cache,
                                              quadrant->p (0, ii),
                                              quadrant->p (1, ii),
                                              quadrant->p (2, ii), idir);

                if (pippo < -0.5) {
                  ray_cache.num_req_rays[idir]++;

                  std::array<double, 2> ray;

                  std::vector<int> direzioni = {0,1,2};
                  direzioni.erase (direzioni.begin ()+idir);

                  for (unsigned i = 0; i < direzioni.size (); ++i) {
                    ray[i] = quadrant->p (direzioni[i], ii);
                  }

                  ray_cache.rays_list[idir].insert (ray);
                }
              }

              for (int jj = 0; jj < quadrant->num_parents (ii); ++jj) {
                rcoeff[quadrant->gparent (jj, ii)] += 0.;
              }
            }
          }
        }

        MPI_Barrier (mpicomm);
        ray_cache.fill_cache ();
      }

      auto refinement = [&rcoeff,this]
      (tmesh_3d::quadrant_iterator q) -> int {
        int currentlevel = static_cast<int> (q->the_quadrant->level);
        int retval = 0;
        double min = 2.0;
        double max = 0.0;
        double tmp = 0.0;

        if (currentlevel >= this->maxlevel)
          retval = 0;
        else {
          for (int ii = 0; ii < 8; ++ii) {

            if (! q->is_hanging (ii)) {
              tmp = rcoeff[q->gt (ii)];

              if (tmp > max) max = tmp;

              if (tmp < min) min = tmp;
            }

          }

          if (this->surf_type_num == 2) {
            if (max > 1 && min < 1)
              retval = this->maxlevel - currentlevel;
            else
              for (const NS::Atom& i : atoms)
                if (is_in_ref (i, q)) {
                  retval = this->maxlevel - currentlevel;
                  break;
                }
          } else {
            if (max > 0.5 && min < 0.5)
              retval = this->maxlevel - currentlevel;
            else
              for (const NS::Atom& i : atoms)
                if (is_in_ref (i, q)) {
                  retval = this->maxlevel - currentlevel;
                  break;
                }
          }
        }

        return (retval);
      };

      tmsh.set_refine_marker (refinement);
      tmsh.refine (0, 1);
    }

    // COARSENING
    {
      distributed_vector rcoeff (tmsh.num_owned_nodes ());

      for (int jj = 0; jj < num_cycles; jj++) {
        ray_cache.num_req_rays[0] = 0; //zero at each ref/coarsen cycle
        ray_cache.num_req_rays[1] = 0; //zero at each ref/coarsen cycle
        ray_cache.num_req_rays[2] = 0; //zero at each ref/coarsen cycle
        ray_cache.rays_list[0].clear ();
        ray_cache.rays_list[1].clear ();
        ray_cache.rays_list[2].clear ();

        if (rank == 0 && jj == 0)
          std::cout << "Coarsening: " << kk << std::endl;

        for (auto quadrant = tmsh.begin_quadrant_sweep ();
             quadrant != tmsh.end_quadrant_sweep ();
             ++quadrant) {

          for (int ii = 0; ii < 8; ++ii) {

            if (! quadrant->is_hanging (ii)) {
              if (surf_type_num == 2)
                rcoeff[quadrant->gt (ii)] = levelsetfun (quadrant->p (0, ii),
                                            quadrant->p (1, ii),
                                            quadrant->p (2, ii));
              else {
                for (int idir = 0; idir < 3; ++idir) {
                  rcoeff[quadrant->gt (ii)] = is_in_ns_surf (ray_cache,
                                              quadrant->p (0, ii),
                                              quadrant->p (1, ii),
                                              quadrant->p (2, ii), idir);

                  if (rcoeff[quadrant->gt (ii)] < -0.5) {
                    ray_cache.num_req_rays[idir]++;

                    std::array<double, 2> ray;

                    std::vector<int> direzioni = {0,1,2};
                    direzioni.erase (direzioni.begin ()+idir);

                    for (unsigned i = 0; i < direzioni.size (); ++i) {
                      ray[i] = quadrant->p (direzioni[i], ii);
                    }

                    // ray_cache.rays[idir].erase(ray);
                    ray_cache.rays_list[idir].insert (ray);
                  }
                }
              }
            } else {
              for (int idir = 0; idir < 3; ++idir) {
                double pippo = is_in_ns_surf (ray_cache,
                                              quadrant->p (0, ii),
                                              quadrant->p (1, ii),
                                              quadrant->p (2, ii), idir);

                if (pippo < -0.5) {
                  ray_cache.num_req_rays[idir]++;

                  std::array<double, 2> ray;

                  std::vector<int> direzioni = {0,1,2};
                  direzioni.erase (direzioni.begin ()+idir);

                  for (unsigned i = 0; i < direzioni.size (); ++i) {
                    ray[i] = quadrant->p (direzioni[i], ii);
                  }

                  ray_cache.rays_list[idir].insert (ray);
                }
              }

              for (int jj = 0; jj < quadrant->num_parents (ii); ++jj) {
                rcoeff[quadrant->gparent (jj, ii)] += 0.;
              }
            }
          }
        }

        MPI_Barrier (mpicomm);
        ray_cache.fill_cache ();
      }

      auto coarsening = [&rcoeff,this]
      (tmesh_3d::quadrant_iterator q) -> int {
        int currentlevel = static_cast<int> (q->the_quadrant->level);
        int retval = 0;
        double min = 2.0;
        double max = 0.0;
        double tmp = 0.0;

        if (currentlevel <= this->minlevel)
          retval = 0;
        else {
          for (int ii = 0; ii < 8; ++ii) {

            if (! q->is_hanging (ii)) {
              tmp = rcoeff[q->gt (ii)];

              if (tmp > max) max = tmp;

              if (tmp < min) min = tmp;
            }

          }

          if (this->surf_type_num == 2) {
            if (min > 1 || max < 1)
              retval = currentlevel - this->minlevel;
          } else {
            if (min > 0.5 || max < 0.5)
              retval = currentlevel - this->minlevel;
          }

          for (const NS::Atom& i : atoms)
            if (is_in_ref (i, q)) {
              retval = 0;
              break;
            }
        }

        return (retval);
      };

      tmsh.set_coarsen_marker (coarsening);
      tmsh.coarsen (0, 1);
    }
  }
}



double
poisson_boltzmann::is_in_ns_surf_stern (ray_cache_t & ray_cache, double x, double y, double z, int dir)
{
  int rank;
  MPI_Comm_rank (mpicomm, &rank);
  double x1 = x;
  double x2 = y;
  double x3 = z;

  if (dir == 0) {
    x1 = y;
    x2 = z;
    x3 = x;
  } else if (dir == 1) {
    x1 = x;
    x2 = z;
    x3 = y;
  }

  crossings_t & ct = ray_cache (x1, x2, dir);

  if (!ct.init && rank != 0) {
    std::array<double, 2> ray = {x1, x2};
    ray_cache.rays[dir].erase (ray);
    return -1.;
  }

  int i = 0;
  int sign = 1;

  if (ct.inters.size () == 0 || x3 < (ct.inters[i]- stern_layer))
    return 0; //if there are no inters or y_the coord is before the first intersection, the point is outside.

  while (i < ct.inters.size () && x3 > (ct.inters[i] - stern_layer*sign)) {
    //go on until the inters is passed
    i++;
    sign *= -1;
  }

  return (i % 2);
}


void
poisson_boltzmann::create_markers (ray_cache_t & ray_cache)
{

  int size, rank;
  MPI_Comm_size (mpicomm, &size);
  MPI_Comm_rank (mpicomm, &rank);

  double eps_in = 4.0*pi*e_0*e_in*kb*T*Angs/ (e*e); //adim e_in
  double eps_out = 4.0*pi*e_0*e_out*kb*T*Angs/ (e*e); //adim e_out

  /////////////////////////////////////////////////////////
  //reactions
  double C_0 = 1.0e3*N_av*ionic_strength; //Bulk concentration of monovalent species
  double k2 = 2.0*C_0*Angs*Angs*e*e/ (e_0*e_out*kb*T);

  this->marker.assign (this->tmsh.num_local_quadrants (), 0.0); //marker = 0 -> in

  if (stern_layer_surf == 1) {
    this->marker_k.assign (this->tmsh.num_local_quadrants (), 1.0); //marker = 1 -> out stern
    this->reaction.assign (tmsh.num_local_quadrants (), eps_out*k2);
  }

  epsilon_nodes = std::make_unique<distributed_vector> (tmsh.num_owned_nodes (),mpicomm);
  epsilon_nodes->get_owned_data ().assign (tmsh.num_owned_nodes (), eps_out);

  reaction_nodes = std::make_unique<distributed_vector> (tmsh.num_owned_nodes (),mpicomm);
  reaction_nodes->get_owned_data ().assign (tmsh.num_owned_nodes (), eps_out*k2);

  int num_cycles = 2;

  if (size == 1) {
    num_cycles = 1;
  }

  int local_num;

  for (int jj = 0; jj < num_cycles; ++jj) {
    ray_cache.num_req_rays[0] = 0; //zero at each ref/coarsen cycle
    ray_cache.num_req_rays[1] = 0; //zero at each ref/coarsen cycle
    ray_cache.num_req_rays[2] = 0; //zero at each ref/coarsen cycle
    ray_cache.rays_list[0].clear ();
    ray_cache.rays_list[1].clear ();
    ray_cache.rays_list[2].clear ();


    for (auto quadrant = this->tmsh.begin_quadrant_sweep ();
         quadrant != this->tmsh.end_quadrant_sweep ();
         ++quadrant) {
      int num_int_nodes = 0;
      int num_int_nodes_stern = 0;
      int num_hanging[3] = {0, 0, 0};
      double x,y,z;

      for (int ii = 0; ii < 8; ++ii) {
        local_num =quadrant->gt (ii);
        x = quadrant->p (0, ii);
        y = quadrant->p (1, ii);
        z = quadrant->p (2, ii);

        if (! quadrant->is_hanging (ii)) {
          if (this->is_in_ns_surf (ray_cache, x, y, z, 2) > 0.5) { //inside the molecule
            ++num_int_nodes;
            ++num_int_nodes_stern;
            (*epsilon_nodes)[local_num] = eps_in;
            (*reaction_nodes)[local_num] = 0.0;
          } else if (this->is_in_ns_surf (ray_cache, x, y, z, 2) < -0.5) {
            ray_cache.num_req_rays[2]++;

            // std::array<double, 2> ray {x,y};

            ray_cache.rays_list[2].insert ({x,y});
          }

          if (this->is_in_ns_surf (ray_cache, x, y, z, 0) < -0.5) {
            ray_cache.num_req_rays[0]++;

            // std::array<double, 2> ray {y,z};

            ray_cache.rays_list[0].insert ({y,z});
          }

          if (this->is_in_ns_surf (ray_cache, x, y, z, 1) < -0.5) {
            ray_cache.num_req_rays[1]++;

            std::array<double, 2> ray {x,z};

            ray_cache.rays_list[1].insert ({x,z});
          }

          if (stern_layer_surf == 1) {
            if (this->is_in_ns_surf_stern (ray_cache, x, y, z, 2) > 0.5) { //inside the stern layer
              ++num_int_nodes_stern;
            }
          }


        } else
          for (int idir = 0; idir < 3; ++idir) {
            ++num_hanging[idir];

            if (this->is_in_ns_surf (ray_cache, x, y, z, idir) < -0.5) {
              ray_cache.num_req_rays[idir]++;
              std::array<double, 2> ray;

              std::vector<int> direzioni {0,1,2};
              direzioni.erase (direzioni.begin ()+idir);

              for (unsigned i = 0; i < direzioni.size (); ++i) {
                ray[i] = quadrant->p (direzioni[i], ii);
              }

              ray_cache.rays_list[idir].insert (ray);

            }
          }
      }

      if (jj != 0 || num_cycles == 1) {
        if (num_int_nodes == 0) { //if there's no node inside the molecule
          this->marker[quadrant->get_forest_quad_idx ()] = 1.0; //quadrant is out
        } else if (num_int_nodes < (8 - num_hanging[2])) { //if the non hanging nodes are not all inside
          this->marker[quadrant->get_forest_quad_idx ()] = 1.0/2.0; //"border"
          border_quad.push_back (quadrant->get_forest_quad_idx ());
        }

        //else: all the nodes are inside: the quadrant is inside and the marker value is 0
        if (stern_layer_surf == 1) {
          for (int idir = 0; idir < 3; ++idir)
            if (num_int_nodes_stern != 0) { //if there is at least on node inside the stern layer along idir-axis
              this->marker_k[quadrant->get_forest_quad_idx ()] = 0.0; //quadrant is in
              this->reaction[quadrant->get_forest_quad_idx ()] = 0.0; //quadrant is in
            }
        }

      }

    }

    MPI_Barrier (mpicomm);
    ray_cache.fill_cache ();
  }

  if (size >1) {
    bim3a_solution_with_ghosts (tmsh, *epsilon_nodes, replace_op);

    if (stern_layer_surf == 0)
      bim3a_solution_with_ghosts (tmsh, (*reaction_nodes), replace_op);
  }
}

void
poisson_boltzmann::create_density_map (ray_cache_t & ray_cache)
{
  int size, rank;
  MPI_Comm_size (mpicomm, &size);
  MPI_Comm_rank (mpicomm, &rank);

  // ------------------------------------------------------------
  // Early exit: if rho_fixed is already initialized, skip the whole procedure.
  // The density map must be created only once.
  // ------------------------------------------------------------
  if (this->rho_fixed) {
    if (rank == 0)
      std::cout << "[INFO] Density map already initialized. Skipping.\n";

    return;
  }

  // ------------------------------------------------------------
  // Allocate and initialize nodal density vector (rho)
  // ------------------------------------------------------------
  this->rho_fixed = std::make_unique<distributed_vector> (tmsh.num_owned_nodes(), mpicomm);
  this->rho_fixed->get_owned_data().assign (tmsh.num_owned_nodes(), 0.0);

  // ------------------------------------------------------------
  // Vector of ones at nodes
  // ------------------------------------------------------------
  this->ones = std::make_unique<distributed_vector> (tmsh.num_owned_nodes(), mpicomm);
  this->ones->get_owned_data().assign (tmsh.num_owned_nodes(), 1.0);

  // ------------------------------------------------------------
  // Per-cell constant field used for computing patch volumes
  // ------------------------------------------------------------
  this->const_ones.assign (tmsh.num_local_quadrants(), 1.0);

  // ------------------------------------------------------------
  // vol_patch will store the nodal patch volumes obtained via BIM
  // ------------------------------------------------------------
  std::unique_ptr<distributed_vector> vol_patch =
    std::make_unique<distributed_vector> (tmsh.num_owned_nodes(), mpicomm);

  // ------------------------------------------------------------
  // Update ghost values if running in parallel
  // ------------------------------------------------------------
  if (size > 1)
    bim3a_solution_with_ghosts (tmsh, *ones, replace_op);

  // ------------------------------------------------------------
  // Compute patch volumes by integrating the constant field = 1
  // ------------------------------------------------------------
  bim3a_rhs (tmsh, const_ones, *ones, *vol_patch);

  if (size > 1)
    vol_patch->assemble();

  // ------------------------------------------------------------
  // Locate the atom positions inside the adaptive mesh
  // ------------------------------------------------------------
  search_points();

  // ------------------------------------------------------------
  // Distribute the atomic charges to the surrounding nodes
  // using a linear volumetric approximation.
  // ------------------------------------------------------------
  for (auto it = lookup_table.begin(); it != lookup_table.end(); ++it) {

    // Compute cell volume (Cartesian and axis-aligned)
    double volume =
      (it->second.p (0, 7) - it->second.p (0, 0)) *
      (it->second.p (1, 7) - it->second.p (1, 0)) *
      (it->second.p (2, 7) - it->second.p (2, 0));

    for (int ii = 0; ii < 8; ++ii) {

      // Linear interpolation weight based on opposite corner distances
      double weight = std::abs (
                        (pos_atoms[it->first][0] - it->second.p (0, 7 - ii)) *
                        (pos_atoms[it->first][1] - it->second.p (1, 7 - ii)) *
                        (pos_atoms[it->first][2] - it->second.p (2, 7 - ii))) / volume;

      // Regular node (not hanging)
      if (!it->second.is_hanging (ii)) {
        (*rho_fixed)[it->second.gt (ii)] +=
          charge_atoms[it->first] * 4.0 * pi * weight /
          (*vol_patch)[it->second.gt (ii)];
      }
      // Hanging node → distribute to parents
      else {
        for (int jj = 0; jj < it->second.num_parents (ii); ++jj) {
          double denom =
            it->second.num_parents (ii) *
            (*vol_patch)[it->second.gparent (jj, ii)];
          (*rho_fixed)[it->second.gparent (jj, ii)] +=
            charge_atoms[it->first] * 4.0 * pi * weight / denom;
        }
      }
    }
  }

  // Release temporary vector
  vol_patch.reset();

  // ------------------------------------------------------------
  // Sync ghost nodes of rho_fixed in parallel runs
  // ------------------------------------------------------------
  if (size > 1)
    bim3a_solution_with_ghosts (tmsh, *rho_fixed);
}

void
poisson_boltzmann::assemple_system_matrix (ray_cache_t & ray_cache)
{
  int size, rank;
  MPI_Comm_size (mpicomm, &size);
  MPI_Comm_rank (mpicomm, &rank);

  // ------------------------------------------------------------
  // 1) CHECK / BUILD REQUIRED DATA STRUCTURES
  // ------------------------------------------------------------

  // If the density map has not been created yet,
  // the system matrix cannot be assembled. Build it first.
  if (!rho_fixed) {
    create_density_map (ray_cache);
  }

  // Function computing fractional volume intersections (for cut-cells)
  auto func_frac = [&] (tmesh_3d::quadrant_iterator& quadrant) {
    return cube_fraction_intersection (quadrant, ray_cache);
  };

  // Allocate sparse matrix A and RHS vector
  A = std::make_unique<distributed_sparse_matrix> (mpicomm);
  A->set_ranges (tmsh.num_owned_nodes());

  rhs = std::make_unique<distributed_vector> (tmsh.num_owned_nodes(), mpicomm);

  // ------------------------------------------------------------
  // 2) ASSEMBLE THE LINEAR SYSTEM (RHS + STIFFNESS MATRIX)
  // ------------------------------------------------------------

  // Assemble RHS using previously computed fixed charge density
  bim3a_rhs (tmsh, const_ones, *rho_fixed, *rhs);

  // rho_fixed and const_ones are no longer needed after building RHS
  rho_fixed.reset();
  std::vector<double>().swap (const_ones);

  // Assemble Laplace operator (with fractional cell treatment)
  bim3a_laplacian_frac (tmsh, *epsilon_nodes, *A, func_frac);

  // Add reaction term (Stern layer present or fractional formulation)
  if (stern_layer_surf == 1) {
    // Standard Stern-layer reaction
    bim3a_reaction (tmsh, reaction, *ones, *A);
  } else {
    // Fractional reaction for intersected cells
    bim3a_reaction_frac (tmsh, (*reaction_nodes), *ones, *A, func_frac);
  }

  // Reaction-related vectors are no longer required
  reaction_nodes.reset();
  ones.reset();
  std::vector<double>().swap (reaction);
  std::vector<double>().swap (marker);

  // ------------------------------------------------------------
  // 3) APPLY BOUNDARY CONDITIONS
  // ------------------------------------------------------------
  // Build Dirichlet boundary list
  dirichlet_bcs3 bcs;

  if (bc == 1) { // Homogeneous Dirichlet BC
    if (std::fabs (pot_bc) > 1.e-5 && rank == 0)
      std::cerr << "[WARNING] Boundary conditions may be inaccurate!!\n";

    for (auto const& ibc : bcells) {
      bcs.emplace_back (ibc.first, ibc.second,
      [] (double, double, double) {
        return 0.0;
      });
    }

    bim3a_dirichlet_bc (tmsh, bcs, *A, *rhs);
  }

  if (bc == 2) { // Coulombic Dirichlet BC
    for (auto const& ibc : bcells) {
      bcs.emplace_back (ibc.first, ibc.second,
      [&] (double x, double y, double z) {
        return coulomb_boundary_conditions (x, y, z);
      });
    }

    bim3a_dirichlet_bc (tmsh, bcs, *A, *rhs);
  }

  if (bc == 3) { // Analytic Dirichlet BC (sphere test case)
    for (auto const& ibc : bcells) {
      bcs.emplace_back (ibc.first, ibc.second,
      [&] (double x, double y, double z) {
        return analytic_solution (x, y, z);
      });
    }

    bim3a_dirichlet_bc (tmsh, bcs, *A, *rhs);
  }

  // ------------------------------------------------------------
  // 4) PARALLEL ASSEMBLY (MPI)
  // ------------------------------------------------------------

  if (size > 1) {
    A->assemble();
    rhs->assemble();
  }
}


void
poisson_boltzmann::export_tmesh (ray_cache_t & ray_cache)
{
  int size, rank;
  MPI_Comm_size (mpicomm, &size);
  MPI_Comm_rank (mpicomm, &rank);
  bim3a_solution_with_ghosts (tmsh, (*epsilon_nodes), replace_op);
  tmsh.octbin_export ("eps_map_0", (*epsilon_nodes));
}

void
poisson_boltzmann::export_potential_map (ray_cache_t & ray_cache)
{
  int size, rank;
  MPI_Comm_size (mpicomm, &size);
  MPI_Comm_rank (mpicomm, &rank);
  tmsh.octbin_export ("potential_map_0", (*phi));
}

void
poisson_boltzmann::export_marked_tmesh ()
{
  tmsh.octbin_export_quadrant (markerfilename.c_str (), marker);

  if (stern_layer_surf == 1) {
    tmsh.octbin_export_quadrant ("mark_stern_0", marker_k);
  }
}

void
poisson_boltzmann::export_p4est ()
{
  tmsh.save (p4estfilename.c_str ());
}



void
poisson_boltzmann::mumps_compute_electric_potential (ray_cache_t & ray_cache)
{
  int rank, size;
  MPI_Comm_size (mpicomm, &size);
  MPI_Comm_rank (mpicomm, &rank);


  mumps mumps_solver;

  std::vector<double> vals;
  std::vector<int> irow, jcol;

  (*A).aij (vals, irow, jcol, mumps_solver.get_index_base ());

  mumps_solver.set_lhs_distributed ();
  mumps_solver.set_distributed_lhs_structure (tmsh.num_global_nodes (), irow, jcol);
  mumps_solver.set_distributed_lhs_data (vals);
  mumps_solver.set_rhs_distributed (*rhs);
  rhs.reset ();
  A.reset ();
  std::cout << "mumps_solver.analyze () = "
            << mumps_solver.analyze ()
            << std::endl;
  std::cout << "mumps_solver.factorize () = "
            << mumps_solver.factorize ()
            << std::endl;
  std::cout << "mumps_solver.solve () = "
            << mumps_solver.solve ()
            << std::endl;

  phi = std::make_unique<distributed_vector> (tmsh.num_owned_nodes ());
  (*phi) = mumps_solver.get_distributed_solution ();

  if (size > 1)
    bim3a_solution_with_ghosts (tmsh, *phi, replace_op);

  ///////

  mumps_solver.cleanup ();
}



void
poisson_boltzmann::lis_compute_electric_potential (ray_cache_t & ray_cache)
{
  int rank, size;
  MPI_Comm_size (mpicomm, &size);
  MPI_Comm_rank (mpicomm, &rank);

  //CSR
  std::vector<double> vals;
  std::vector<int> irow, jcol;


  (*A).csr (vals, jcol, irow);

  // lis RHS
  LIS_INT i, is, ie, n_rhs, ln;
  LIS_VECTOR rhs_lis;
  //n_rhs = tmsh.num_global_nodes();
  ln = rhs->get_owned_data ().size ();

  lis_vector_create (mpicomm, &rhs_lis);
  lis_vector_set_size (rhs_lis, ln, 0);
  lis_vector_get_range (rhs_lis, &is, &ie);

  for (i=is; i<ie; i++)
    lis_vector_set_value (LIS_INS_VALUE, i, rhs->get_owned_data ()[i-is], rhs_lis);

  //cleaning of rhs
  rhs.reset ();

  //lis_vector_print(rhs_lis);
  // lis PHI
  LIS_VECTOR phi_lis;

  lis_vector_create (mpicomm, &phi_lis);
  lis_vector_set_size (phi_lis, ln, 0);
  //lis_vector_set_size(phi_lis, 0, n_rhs);
  lis_vector_get_range (phi_lis, &is, &ie);

  // lis MATRIX
  LIS_INT n, nnz; //n: matrix dim ; nnz: numb of non zero elems
  LIS_INT *index; //array of integer containing the col index of non zero elems
  LIS_INT *ptr; //array of integer with starting points of rows
  LIS_SCALAR *value; //array of double stores non-zero elements of matrix A along the row
  LIS_MATRIX A_lis; //array of integer containing the col index of non zero elems

  nnz = (*A).owned_nnz ();
  n = tmsh.num_owned_nodes ();

  A.reset ();

  lis_matrix_create (mpicomm, &A_lis);

  lis_matrix_set_size (A_lis, n, 0);
  ptr = &irow[0];
  index = &jcol[0];
  value = &vals[0];

  lis_matrix_set_csr (nnz, ptr, index, value, A_lis);


  lis_matrix_assemble (A_lis);

  //Solve linear system
  LIS_SOLVER solver;

  lis_solver_create (&solver);

  std::string opts = linear_solver_options;

  lis_solver_set_option (&opts[0], solver);

  lis_solve (A_lis, rhs_lis, phi_lis, solver);

  lis_solver_destroy (solver);
  lis_vector_destroy (rhs_lis);

  phi = std::make_unique<distributed_vector> (tmsh.num_owned_nodes (), mpicomm);

  lis_vector_get_values (phi_lis, is, ln, phi->get_owned_data ().data ());

  lis_vector_destroy (phi_lis);

  if (size > 1)
    bim3a_solution_with_ghosts (tmsh, *phi, replace_op);

}


void
poisson_boltzmann::write_potential_on_atoms_fast ()
{
  int rank, size;
  MPI_Comm_rank (mpicomm, &rank);
  MPI_Comm_size (mpicomm, &size);

  std::vector<std::string> local_lines;

  double phi_on_atom;
  double phi_hang_nodes = 0.0;

  // Costruisci stringhe localmente
  for (auto it = lookup_table.begin(); it != lookup_table.end(); ++it) {
    phi_on_atom = 0.0;
    double volume = (it->second.p (0, 7) - it->second.p (0, 0)) *
                    (it->second.p (1, 7) - it->second.p (1, 0)) *
                    (it->second.p (2, 7) - it->second.p (2, 0));

    for (int ii = 0; ii < 8; ++ii) {
      double weigth = std::abs ((pos_atoms[it->first][0] - it->second.p (0, 7-ii)) *
                                (pos_atoms[it->first][1] - it->second.p (1, 7-ii)) *
                                (pos_atoms[it->first][2] - it->second.p (2, 7-ii))) / volume;

      if (!it->second.is_hanging (ii)) {
        phi_on_atom += (*phi)[it->second.gt (ii)] * weigth;
      } else {
        phi_hang_nodes = 0.0;

        for (int jj = 0; jj < it->second.num_parents (ii); ++jj) {
          phi_hang_nodes += (*phi)[it->second.gparent (jj, ii)] / it->second.num_parents (ii);
        }

        phi_on_atom += phi_hang_nodes * weigth;
      }
    }

    std::ostringstream oss;
    oss << std::setw (8) << index_atoms[it->first]
        << std::fixed << std::setprecision (3)
        << std::setw (8) << pos_atoms[it->first][0]
        << std::setw (8) << pos_atoms[it->first][1]
        << std::setw (8) << pos_atoms[it->first][2]
        << std::fixed << std::setprecision (4)
        << "  " << phi_on_atom << "\n";

    local_lines.push_back (oss.str());
  }

  // Serializzazione delle stringhe
  std::string local_data;

  for (const auto& line : local_lines)
    local_data += line;

  int local_size = local_data.size();
  std::vector<int> all_sizes (size);

  MPI_Gather (&local_size, 1, MPI_INT, all_sizes.data(), 1, MPI_INT, 0, mpicomm);

  std::vector<int> displs (size);
  std::string global_data;

  if (rank == 0) {
    int total_size = 0;

    for (int i = 0; i < size; ++i) {
      displs[i] = total_size;
      total_size += all_sizes[i];
    }

    global_data.resize (total_size);
  }

  MPI_Gatherv (local_data.data(), local_size, MPI_CHAR,
               rank == 0 ? &global_data[0] : nullptr,
               all_sizes.data(), displs.data(), MPI_CHAR,
               0, mpicomm);

  // Solo rank 0 scrive sul file
  if (rank == 0) {
    std::ofstream phi_atoms ("phi_on_atoms.txt");
    phi_atoms << global_data;
    phi_atoms.close();
  }
}


/////////////////////////////////////////////////////////////////////////////////////////////////////

std::array<double,12>
poisson_boltzmann::cube_fraction_intersection (tmesh_3d::quadrant_iterator& quadrant,
    const ray_cache_t & ray_cache)


// v6_________e7_________v7
// /|                  /|
// e8 / |                 / |
// /  |             e6 /  |
// /   | e12           /   | e11
// v4/____|_____e5_______/v5  |
// |    |              |    |
// |  v2|______e3______|____|v3
// e9  |   /               |   /
// |  /            e10 |  /
// | /  e4             | / e2
// |/                  |/
// v0/_________e1________/v1
{
  std::array<double,12> fraction = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};

  int dir;
  int i1, i2;
  double x1, x2;
  std::array<double,2> ray;

  if (marker[quadrant->get_forest_quad_idx ()] == 0.5) {
    for (int j: {
           0,2,4,6
         })
      //x-axis edge
    {
      dir = 0;
      i1 = edge2nodes[2 * j ];
      i2 = edge2nodes[2 * j + 1];
      x1 = quadrant->p (dir, i1);
      x2 = quadrant->p (dir, i2);
      std::vector<int> direzioni {0,1,2};
      direzioni.erase (direzioni.begin ()+dir);

      for (unsigned i = 0; i < direzioni.size (); ++i) {
        ray[i] = quadrant->p (direzioni[i], i1);
      }

      auto it0 = ray_cache.rays[dir].find (ray);
      auto inters = it0->second.inters;

      for (int ii =0; ii<inters.size (); ii++) {
        if (inters[ii]>= x1 && inters[ii] <=x2) {
          fraction[j] = (inters[ii] - x1)/ (x2 - x1);
        }
      }
    }

    for (int j: {
           1,3,5,7
         })
      //y-axis edge
    {
      dir = 1;
      i1 = edge2nodes[2 * j ];
      i2 = edge2nodes[2 * j + 1];
      x1 = quadrant->p (dir, i1);
      x2 = quadrant->p (dir, i2);
      std::vector<int> direzioni {0,1,2};
      direzioni.erase (direzioni.begin ()+dir);

      for (unsigned i = 0; i < direzioni.size (); ++i) {
        ray[i] = quadrant->p (direzioni[i], i1);
      }

      auto it0 = ray_cache.rays[dir].find (ray);
      auto inters = it0->second.inters;

      for (int ii =0; ii<inters.size (); ii++) {
        if (inters[ii]>= x1 && inters[ii] <=x2) {
          fraction[j] = (inters[ii] - x1)/ (x2 - x1);
        }
      }
    }

    for (int j: {
           8,9,10,11
         })
      //z-axis edge
    {
      dir = 2;
      i1 = edge2nodes[2 * j ];
      i2 = edge2nodes[2 * j + 1];
      x1 = quadrant->p (dir, i1);
      x2 = quadrant->p (dir, i2);
      std::vector<int> direzioni {0,1,2};
      direzioni.erase (direzioni.begin ()+dir);

      for (unsigned i = 0; i < direzioni.size (); ++i) {
        ray[i] = quadrant->p (direzioni[i], i1);
      }

      auto it0 = ray_cache.rays[dir].find (ray);
      auto inters = it0->second.inters;

      for (int ii =0; ii<inters.size (); ii++) {
        if (inters[ii]>= x1 && inters[ii] <=x2) {
          fraction[j] = (inters[ii] - x1)/ (x2 - x1);
        }
      }
    }
  }

  return fraction;
}

void
poisson_boltzmann::normal_intersection (tmesh_3d::quadrant_iterator& quadrant,
                                        const ray_cache_t & ray_cache,
                                        int edge, std::array<double,3> &norm,
                                        double &frac)
{
  int dir = edge_axis[edge];
  int i1 = edge2nodes[2*edge ];
  int i2 = edge2nodes[2*edge + 1];
  double x1 = quadrant->p (dir, i1),
         x2 = quadrant->p (dir, i2);

  std::array<double,2> ray;
  std::vector<int> direzioni = {0,1,2};
  direzioni.erase (direzioni.begin ()+dir);

  for (unsigned i = 0; i < direzioni.size (); ++i) {
    ray[i] = quadrant->p (direzioni[i], i1);
  }

  auto it0 = ray_cache.rays[dir].find (ray);
  auto normali = it0->second.normals;
  auto inters = it0->second.inters;

  frac = 0.5;

  for (int ii =0; ii<inters.size (); ii++) {
    if (inters[ii]>= x1 && inters[ii] <=x2) {
      norm[0] = normali[0 + 3*ii];
      norm[1] = normali[1 + 3*ii];
      norm[2] = normali[2 + 3*ii];
      frac = (inters[ii] - x1)/ (x2 - x1);
    }
  }
}

int
poisson_boltzmann::classifyCube (tmesh_3d::quadrant_iterator& quadrant,
                                 double isolevel)
{
  int cubeindex = 0;
  int index = 1;
  double tmp = 0;
  constexpr double EPSILON = 1e-10;

  for (int ii : {
         0,1,3,2,4,5,7,6
       }) {
    if (! quadrant->is_hanging (ii)) {
      if ( (*epsilon_nodes)[quadrant->gt (ii)] < (isolevel - EPSILON)) cubeindex |= index;
    } else {
      for (int jj = 0; jj < quadrant->num_parents (ii); ++jj) {
        tmp += (*epsilon_nodes)[quadrant->gparent (jj, ii)] / quadrant->num_parents (ii);
      }

      if (tmp < (isolevel - EPSILON)) cubeindex |= index;
    }

    tmp = 0;
    index *= 2;
  }

  // Cube is entirely in/out of the surface
  if (edgeTable[cubeindex] == 0)
    return -1;

  return cubeindex;
}

int
poisson_boltzmann::classifyCube_fast (tmesh_3d::quadrant_iterator& quadrant,
                                      double isolevel)
{
  int cubeindex = 0;
  int index = 1;
  double tmp = 0;
  constexpr double EPSILON = 1e-10;

  for (int ii : {
         0,1,3,2,4,5,7,6
       }) {
    if ( (*epsilon_nodes)[quadrant->gt (ii)] < (isolevel - EPSILON)) cubeindex |= index;

    index *= 2;
  }

  // Cube is entirely in/out of the surface
  if (edgeTable[cubeindex] == 0)
    return -1;

  return cubeindex;
}

std::tuple<std::array<double,8>, std::array<double,8>, std::vector<int>,std::vector<int> >
poisson_boltzmann::classifyCube_flux (tmesh_3d::quadrant_iterator& quadrant,
                                      std::array<double,8>& tmp_phi,
                                      std::array<double,8>& tmp_eps)
{
  std::vector<int> edges {};
  std::vector<int> flux {};

  for (int ii = 0; ii < 8; ++ii) {
    if (! quadrant->is_hanging (ii)) {
      tmp_eps[ii]= (*epsilon_nodes)[quadrant->gt (ii)];
      tmp_phi[ii]= (*phi)[quadrant->gt (ii)];
    } else {
      tmp_eps[ii] = 0.0;
      tmp_phi[ii] = 0.0;

      for (int jj = 0; jj < quadrant->num_parents (ii); ++jj) {
        tmp_eps[ii] += (*epsilon_nodes)[quadrant->gparent (jj, ii)] / quadrant->num_parents (ii);
        tmp_phi[ii] += (*phi)[quadrant->gparent (jj, ii)] / quadrant->num_parents (ii);
      }
    }
  }

  for (int ii = 0; ii < 12; ++ii) {
    if (tmp_eps[edge2nodes[2*ii]] < tmp_eps[edge2nodes[2*ii +1]]) {
      flux.push_back (1);
      edges.push_back (ii);
    } else if (tmp_eps[edge2nodes[2*ii]] > tmp_eps[edge2nodes[2*ii +1]]) {
      flux.push_back (-1);
      edges.push_back (ii);
    }
  }

  return make_tuple (tmp_phi, tmp_eps, edges, flux);
}

std::tuple<std::array<double,8>, std::array<double,8>, std::vector<int>,std::vector<int> >
poisson_boltzmann::classifyCube_flux_fast (tmesh_3d::quadrant_iterator& quadrant,
    std::array<double,8>& tmp_phi,
    std::array<double,8>& tmp_eps)
{
  std::vector<int> edges {};
  std::vector<int> flux {};


  for (int ii = 0; ii < 8; ++ii) {
    tmp_eps[ii]= (*epsilon_nodes)[quadrant->gt (ii)];
    tmp_phi[ii]= (*phi)[quadrant->gt (ii)];
  }

  for (int ii = 0; ii < 12; ++ii) {
    if (tmp_eps[edge2nodes[2*ii]] < tmp_eps[edge2nodes[2*ii +1]]) {
      flux.push_back (1);
      edges.push_back (ii);
    } else if (tmp_eps[edge2nodes[2*ii]] > tmp_eps[edge2nodes[2*ii +1]]) {
      flux.push_back (-1);
      edges.push_back (ii);
    }
  }

  return make_tuple (tmp_phi, tmp_eps, edges, flux);
}

double wha (double eps1, double eps2, double frac)
{
  return 1.0/ (frac/eps1 + (1-frac)/eps2);
}

double flux_dir (double eps1, double eps2)
{
  return eps1 < eps2 ? 1 : -1;
}

double phi0 (double eps1, double eps2,
             double phi1, double phi2, double frac)
{
  return phi1 + frac*eps2* (phi2-phi1)/ (eps2*frac + eps1* (1-frac));
}

double areaTriangle (const std::array<std::array<double,3>,3> &triangle)
{

  double area;
  std::array<double,3> ab;
  std::array<double,3> ac;

  for (int i = 0; i < 3; ++i) {
    ab[i] = triangle[0][i] - triangle[1][i];
    ac[i] = triangle[0][i] - triangle[2][i];
  }

  area = 0.5* std::hypot (ab[1]*ac[2] - ab[2]*ac[1],
                          ab[0]*ac[2] - ab[2]*ac[0],
                          ab[1]*ac[0] - ab[0]*ac[1]);
  return area;
}

double SphercalAreaTriangle (const std::array<std::array<double,3>,3> &triangle)
{

  double area;
  double rs =2;
  double a,b,c, aa, bb, cc, E;
  double nA,nB,nC;
  std::array<double,3> A = triangle[0];
  std::array<double,3> B = triangle[1];
  std::array<double,3> C = triangle[2];

  nA = std::hypot (A[0], A[1], A[2]);
  nB = std::hypot (B[0], B[1], B[2]);
  nC = std::hypot (C[0], C[1], C[2]);

  a = std::inner_product (std::begin (A), std::end (A), std::begin (B), 0.0);
  b = std::inner_product (std::begin (A), std::end (A), std::begin (C), 0.0);
  c = std::inner_product (std::begin (B), std::end (B), std::begin (C), 0.0);

  a = a/ (nA*nB);
  b = b/ (nA*nC);
  c = c/ (nC*nB);

  a = std::acos (a);
  b = std::acos (b);
  c = std::acos (c);

  if (a>pi/2.0)
    a = pi - a;

  if (b>pi/2.0)
    b = pi - b;

  if (c>pi/2.0)
    c = pi - c;

  aa = std::acos ( (cos (a)-cos (b)*cos (c))/ (sin (b)*sin (c)));
  bb = std::acos ( (cos (b)-cos (c)*cos (a))/ (sin (a)*sin (c)));
  cc = std::acos ( (cos (c)-cos (b)*cos (a))/ (sin (b)*sin (a)));

  E = aa + bb + cc - pi;

  area = rs*rs * E;

  return area;
}

int
poisson_boltzmann::getTriangles (int cubeindex,
                                 std::array<std::array<int,3>,5> &triangles)
{
  int ntriang = 0;
  int i;
  triangles.fill ({}); //set matrix to zero

  for (i=0; triTable[cubeindex][i]!=-1; i+=3) {
    // save all the assigned indexes
    triangles[ntriang][0] = triTable[cubeindex][i ];
    triangles[ntriang][1] = triTable[cubeindex][i+1];
    triangles[ntriang][2] = triTable[cubeindex][i+2];
    ntriang++;
  }

  return ntriang;
}

void
poisson_boltzmann::energy (ray_cache_t & ray_cache)
{
  int rank;
  MPI_Comm_rank (mpicomm, &rank);

  if (rank == 0)
    std::cout << "\n================ [ Electrostatic Energy ] =================\n";


  // ===========================
  // Costanti fisiche e scalari
  // ===========================
  const double inv_4pi = 1.0 / (4.0 * pi);
  const double eps0 = e_0; // Permittività del vuoto
  const double eps_in = 4.0 * pi * eps0 * e_in * kb * T * Angs / (e * e);
  const double eps_out = 4.0 * pi * eps0 * e_out * kb * T * Angs / (e * e);

  const double C0 = 1.0e3 * N_av * ionic_strength; // [mol/m^3]
  const double k2 = 2.0 * C0 * Angs * Angs * e * e / (eps0 * e_out * kb * T);
  const double k = std::sqrt (k2);

  const double den_in = 1.0 / eps_in;
  const double constant_pol = (1.0 / eps_out - 1.0 / eps_in) * inv_4pi;
  const double constant_react = (1.0 / eps_out) * inv_4pi;

  // ===========================
  // Variabili locali
  // ===========================
  // double energy_pol = 0.0;
  // double energy_react = 0.0;
  // double coul_energy = 0.0;
  double charge_pol = 0.0;

  // --- Filtra gli atomi caricati ---
  std::vector<double> charge_atoms_tmp;
  std::vector<std::array<double, 3>> pos_atoms_tmp;
  // charge_atoms_tmp.reserve(charge_atoms.size());
  // pos_atoms_tmp.reserve(pos_atoms.size());

  for (size_t ii = 0; ii < charge_atoms.size(); ++ii) {
    if (std::fabs (charge_atoms[ii]) > 1.e-5) {
      charge_atoms_tmp.push_back (charge_atoms[ii]);
      pos_atoms_tmp.push_back (pos_atoms[ii]);
    }
  }

  const size_t num_atoms = charge_atoms_tmp.size();
  std::vector<double>().swap (charge_atoms);
  std::vector<std::array<double, 3>>().swap (pos_atoms);

  // --- Energia Coulombiana ---
  if (calc_coulombic == 1) {
    for (size_t i = 0; i < num_atoms; ++i) {
      const double qi = charge_atoms_tmp[i];
      const std::array<double,3>& ri = pos_atoms_tmp[i];

      for (size_t j = i + 1; j < num_atoms; ++j) {
        const std::array<double,3>& rj = pos_atoms_tmp[j];
        const double dx = ri[0] - rj[0];
        const double dy = ri[1] - rj[1];
        const double dz = ri[2] - rj[2];
        const double r = std::sqrt (dx * dx + dy * dy + dz * dz);
        this->coul_energy += qi * charge_atoms_tmp[j] / r;
      }
    }

    this->coul_energy *= den_in;
  }

  ////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////
  double first_int = 0.0, second_int = 0.0;

  std::array<double,3> h{0}, area_h{0};
  std::array<double,3> V, N;
  std::array<double,8> tmp_eps, tmp_phi;
  std::vector<int> edg, fl_dir;


  auto quadrant = this->tmsh.begin_quadrant_sweep ();

  // polarization energy
  if (calc_energy==1 || (calc_energy==2 && k < 1.e-5)) {
    for (const int ii : border_quad) {
      quadrant[ii];

      for (int d = 0; d < 3; ++d)
        h[d] = quadrant->p (d, 7) - quadrant->p (d, 0);

      area_h = {h[1]*h[2]/h[0]*0.25, h[0]*h[2]/h[1]*0.25, h[0]*h[1]/h[2]*0.25};

      std::tie (tmp_phi, tmp_eps, edg, fl_dir) = classifyCube_flux (quadrant, tmp_phi, tmp_eps);

      for (int ip = 0; ip < edg.size (); ++ip) {
        const int edge = edg[ip];
        const int axis = edge_axis[edge];
        const int i1 = edge2nodes[2 * edge];
        const int i2 = edge2nodes[2 * edge + 1];

        double fract = 0.0;
        normal_intersection (quadrant, ray_cache, edge, N, fract);

        V = {quadrant->p (0, i1), quadrant->p (1, i1), quadrant->p (2, i1)};
        V[axis] += fract * h[axis];

        const double tmp_flux =
          - (tmp_phi[i2] - tmp_phi[i1]) * wha (tmp_eps[i1], tmp_eps[i2], fract)
          * fl_dir[ip] * area_h[axis];

        charge_pol += tmp_flux;

        for (size_t ia = 0; ia < num_atoms; ++ia) {
          const std::array<double,3> &ra = pos_atoms_tmp[ia];
          const double dx = ra[0] - V[0];
          const double dy = ra[1] - V[1];
          const double dz = ra[2] - V[2];
          const double r = std::sqrt (dx * dx + dy * dy + dz * dz);
          const double inv_r3 = 1.0 / (r * r * r);
          const double qflux = tmp_flux / r;

          first_int += charge_atoms_tmp[ia] * qflux;
        }
      }
    }

    energy_pol = 0.5*constant_pol*first_int;
  }


  //polarization energy + ionic energy
  if (calc_energy==2 && k > 1.e-5) {
    int cubeindex = -1;
    std::array<std::array<double,3>,3> vert_triangles, norms_vert;
    std::array<double,3> dist_vert, phi_sup;
    int ntriang = 0;

    for (const int ii : border_quad) {
      quadrant[ii];
      cubeindex = classifyCube (quadrant, eps_out);

      for (int d = 0; d < 3; ++d)
        h[d] = quadrant->p (d, 7) - quadrant->p (d, 0);

      area_h = {h[1]*h[2]/h[0]*0.25, h[0]*h[2]/h[1]*0.25, h[0]*h[1]/h[2]*0.25};

      std::tie (tmp_phi, tmp_eps, edg, fl_dir) = classifyCube_flux (quadrant, tmp_phi, tmp_eps);
      ntriang = getTriangles (cubeindex, triangles);

      // --- flussi
      for (int ip = 0; ip < edg.size (); ++ip) {
        const int edge = edg[ip];
        const int axis = edge_axis[edge];
        const int i1 = edge2nodes[2 * edge];
        const int i2 = edge2nodes[2 * edge + 1];

        double fract = 0.0;
        normal_intersection (quadrant, ray_cache, edge, N, fract);

        V = {quadrant->p (0, i1), quadrant->p (1, i1), quadrant->p (2, i1)};
        V[axis] += fract * h[axis];

        const double tmp_flux =
          - (tmp_phi[i2] - tmp_phi[i1]) * wha (tmp_eps[i1], tmp_eps[i2], fract)
          * fl_dir[ip] * area_h[axis];

        charge_pol += tmp_flux;

        for (size_t ia = 0; ia < num_atoms; ++ia) {
          const std::array<double,3> &ra = pos_atoms_tmp[ia];
          const double dx = ra[0] - V[0];
          const double dy = ra[1] - V[1];
          const double dz = ra[2] - V[2];
          const double r = std::sqrt (dx * dx + dy * dy + dz * dz);
          const double inv_r3 = 1.0 / (r * r * r);

          first_int += charge_atoms_tmp[ia] * tmp_flux / r;
        }
      }

      // --- triangoli (componente ionica)
      for (int itri = 0; itri < ntriang; ++itri) {
        for (int jj = 0; jj < 3; ++jj) {
          const int edge = triangles[itri][jj];
          const int axis = edge_axis[edge];
          const int i1 = edge2nodes[2 * edge];
          const int i2 = edge2nodes[2 * edge + 1];

          double fract = 0.0;
          normal_intersection (quadrant, ray_cache, edge, N, fract);

          V = {quadrant->p (0, i1), quadrant->p (1, i1), quadrant->p (2, i1)};
          V[axis] += fract * h[axis];

          vert_triangles[jj] = V;
          norms_vert[jj] = N;

          phi_sup[jj] = phi0 (tmp_eps[i1], tmp_eps[i2], tmp_phi[i1], tmp_phi[i2], fract);
        }

        const double area = areaTriangle (vert_triangles);

        for (size_t ia = 0; ia < num_atoms; ++ia) {
          const double qi = charge_atoms_tmp[ia];
          const std::array<double,3> &ra = pos_atoms_tmp[ia];

          for (int kk = 0; kk < 3; ++kk) {
            dist_vert = {vert_triangles[kk][0] - ra[0],
                         vert_triangles[kk][1] - ra[1],
                         vert_triangles[kk][2] - ra[2]
                        };
            const double r2 = dist_vert[0]*dist_vert[0] + dist_vert[1]*dist_vert[1] + dist_vert[2]*dist_vert[2];
            const double r = std::sqrt (r2);
            const double inv_r3 = 1.0 / (r2 * r);
            const double inv_r5 = inv_r3 / r2;
            const double dot = dist_vert[0]*norms_vert[kk][0] + dist_vert[1]*norms_vert[kk][1] + dist_vert[2]*norms_vert[kk][2];
            const double factor = phi_sup[kk] * inv_4pi * area / 3.0;

            second_int += qi * phi_sup[kk] * dot * inv_r3 * inv_4pi * area / 3.0;
          }
        }
      }
    }

    energy_pol = 0.5 * constant_pol * first_int;
    energy_react = 0.5 * (second_int - first_int * constant_react);
  }



  auto reduce_double = [&] (double &x) {
    MPI_Reduce (rank == 0 ? MPI_IN_PLACE : &x, &x, 1, MPI_DOUBLE, MPI_SUM, 0, mpicomm);
  };

  auto reduce_vec = [&] (std::vector<double> &v) {
    MPI_Reduce (rank == 0 ? MPI_IN_PLACE : v.data(),
                rank == 0 ? v.data() : nullptr,
                (int)v.size(), MPI_DOUBLE, MPI_SUM, 0, mpicomm);
  };

  reduce_double (charge_pol);
  reduce_double (energy_pol);
  reduce_double (energy_react);

  if (rank == 0) {
    constexpr int label_width = 50;
    constexpr int precision = 16;

    std::cout << std::left << std::setw (label_width) << "  Net charge [e]:"
              << std::setprecision (precision) << net_charge << "\n";

    std::cout << std::left << std::setw (label_width) << "  Flux charge [e]:"
              << std::setprecision (precision) << charge_pol / (4.0 * pi) << "\n";

    std::cout << std::left << std::setw (label_width) << "  Polarization energy [kT]:"
              << std::setprecision (precision) << energy_pol << "\n";

    if (calc_energy == 2) {
      std::cout << std::left << std::setw (label_width) << "  Direct ionic energy [kT]:"
                << std::setprecision (precision) << energy_react << "\n";
    }

    if (calc_coulombic == 1) {
      std::cout << std::left << std::setw (label_width) << "  Coulombic energy [kT]:"
                << std::setprecision (precision) << coul_energy << "\n";
    }

    std::cout << std::left << std::setw (label_width) << "  Sum of electrostatic energy contributions [kT]:"
              << std::setprecision (precision)
              << (energy_pol + energy_react + coul_energy) << "\n";

    std::cout << "===========================================================\n";
  }
}

void
poisson_boltzmann::energy_fast (ray_cache_t & ray_cache)
{
  int rank;
  MPI_Comm_rank (mpicomm, &rank);

  if (rank == 0)
    std::cout << "\n================ [ Electrostatic Energy ] =================\n";

  // ===========================
  // Costanti fisiche e scalari
  // ===========================
  const double inv_4pi = 1.0 / (4.0 * pi);
  const double eps0 = e_0; // Permittività del vuoto
  const double eps_in = 4.0 * pi * eps0 * e_in * kb * T * Angs / (e * e);
  const double eps_out = 4.0 * pi * eps0 * e_out * kb * T * Angs / (e * e);

  const double C0 = 1.0e3 * N_av * ionic_strength; // [mol/m^3]
  const double k2 = 2.0 * C0 * Angs * Angs * e * e / (eps0 * e_out * kb * T);
  const double k = std::sqrt (k2);

  const double den_in = 1.0 / eps_in;
  const double constant_pol = (1.0 / eps_out - 1.0 / eps_in) * inv_4pi;
  const double constant_react = (1.0 / eps_out) * inv_4pi;

  // ===========================
  // Variabili locali
  // ===========================
  // double energy_pol = 0.0;
  // double energy_react = 0.0;
  // double coul_energy = 0.0;
  double charge_pol = 0.0;

  // --- Filtra gli atomi caricati ---
  std::vector<double> charge_atoms_tmp;
  std::vector<std::array<double, 3>> pos_atoms_tmp;
  // charge_atoms_tmp.reserve(charge_atoms.size());
  // pos_atoms_tmp.reserve(pos_atoms.size());

  for (size_t ii = 0; ii < charge_atoms.size(); ++ii) {
    if (std::fabs (charge_atoms[ii]) > 1.e-5) {
      charge_atoms_tmp.push_back (charge_atoms[ii]);
      pos_atoms_tmp.push_back (pos_atoms[ii]);
    }
  }

  const size_t num_atoms = charge_atoms_tmp.size();
  std::vector<double>().swap (charge_atoms);
  std::vector<std::array<double, 3>>().swap (pos_atoms);

  // --- Energia Coulombiana ---
  if (calc_coulombic == 1) {
    for (size_t i = 0; i < num_atoms; ++i) {
      const double qi = charge_atoms_tmp[i];
      const std::array<double,3>& ri = pos_atoms_tmp[i];

      for (size_t j = i + 1; j < num_atoms; ++j) {
        const std::array<double,3>& rj = pos_atoms_tmp[j];
        const double dx = ri[0] - rj[0];
        const double dy = ri[1] - rj[1];
        const double dz = ri[2] - rj[2];
        const double r = std::sqrt (dx * dx + dy * dy + dz * dz);
        this->coul_energy += qi * charge_atoms_tmp[j] / r;
      }
    }

    this->coul_energy *= den_in;
  }

  ////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////

  double first_int = 0.0, second_int = 0.0;

  std::array<double,3> h{0}, area_h{0};
  std::array<double,3> V, N;
  std::array<double,8> tmp_eps, tmp_phi;
  std::vector<int> edg, fl_dir;

  auto quadrant = this->tmsh.begin_quadrant_sweep ();

  if (!border_quad.empty()) {
    quadrant[border_quad[0]];

    for (int d = 0; d < 3; ++d)
      h[d] = quadrant->p (d, 7) - quadrant->p (d, 0);

    area_h = {h[1]*h[2]/h[0]*0.25, h[0]*h[2]/h[1]*0.25, h[0]*h[1]/h[2]*0.25};
  }

  // flux and polarization energy calculation
  if (calc_energy==1 || (calc_energy == 2 && k < 1.e-5)) {
    for (const int ii : border_quad) {
      quadrant[ii];
      std::tie (tmp_phi, tmp_eps, edg, fl_dir) = classifyCube_flux_fast (quadrant, tmp_phi, tmp_eps);

      for (int ip = 0; ip < edg.size (); ++ip) {
        const int edge = edg[ip];
        const int axis = edge_axis[edge];
        const int i1 = edge2nodes[2 * edge];
        const int i2 = edge2nodes[2 * edge + 1];

        double fract = 0.0;
        normal_intersection (quadrant, ray_cache, edge, N, fract);

        V = {quadrant->p (0, i1), quadrant->p (1, i1), quadrant->p (2, i1)};
        V[axis] += fract * h[axis];

        const double tmp_flux =
          - (tmp_phi[i2] - tmp_phi[i1]) * wha (tmp_eps[i1], tmp_eps[i2], fract)
          * fl_dir[ip] * area_h[axis];

        charge_pol += tmp_flux;

        for (size_t ia = 0; ia < num_atoms; ++ia) {
          const std::array<double,3> &ra = pos_atoms_tmp[ia];
          const double dx = ra[0] - V[0];
          const double dy = ra[1] - V[1];
          const double dz = ra[2] - V[2];
          const double r = std::sqrt (dx * dx + dy * dy + dz * dz);
          const double inv_r3 = 1.0 / (r * r * r);
          const double qflux = tmp_flux / r;

          first_int += charge_atoms_tmp[ia] * qflux;
        }
      }
    }

    this->energy_pol = 0.5*constant_pol*first_int;
  }

  //polarization energy + ionic energy
  if (calc_energy==2 && k > 1.e-5) {
    int cubeindex = -1;
    std::array<std::array<double,3>,3> vert_triangles, norms_vert;
    std::array<double,3> dist_vert, phi_sup;
    int ntriang = 0;

    for (const int ii : border_quad) {
      quadrant[ii];
      cubeindex = classifyCube_fast (quadrant, eps_out);
      std::tie (tmp_phi, tmp_eps, edg, fl_dir) = classifyCube_flux_fast (quadrant, tmp_phi, tmp_eps);
      ntriang = getTriangles (cubeindex, triangles);

      // --- flussi
      for (int ip = 0; ip < edg.size (); ++ip) {
        const int edge = edg[ip];
        const int axis = edge_axis[edge];
        const int i1 = edge2nodes[2 * edge];
        const int i2 = edge2nodes[2 * edge + 1];

        double fract = 0.0;
        normal_intersection (quadrant, ray_cache, edge, N, fract);

        V = {quadrant->p (0, i1), quadrant->p (1, i1), quadrant->p (2, i1)};
        V[axis] += fract * h[axis];

        const double tmp_flux =
          - (tmp_phi[i2] - tmp_phi[i1]) * wha (tmp_eps[i1], tmp_eps[i2], fract)
          * fl_dir[ip] * area_h[axis];

        charge_pol += tmp_flux;

        for (size_t ia = 0; ia < num_atoms; ++ia) {
          const std::array<double,3> &ra = pos_atoms_tmp[ia];
          const double dx = ra[0] - V[0];
          const double dy = ra[1] - V[1];
          const double dz = ra[2] - V[2];
          const double r = std::sqrt (dx * dx + dy * dy + dz * dz);
          const double inv_r3 = 1.0 / (r * r * r);

          first_int += charge_atoms_tmp[ia] * tmp_flux / r;
        }
      }

      // --- triangoli (componente ionica)
      for (int itri = 0; itri < ntriang; ++itri) {
        for (int jj = 0; jj < 3; ++jj) {
          const int edge = triangles[itri][jj];
          const int axis = edge_axis[edge];
          const int i1 = edge2nodes[2 * edge];
          const int i2 = edge2nodes[2 * edge + 1];

          double fract = 0.0;
          normal_intersection (quadrant, ray_cache, edge, N, fract);

          V = {quadrant->p (0, i1), quadrant->p (1, i1), quadrant->p (2, i1)};
          V[axis] += fract * h[axis];

          vert_triangles[jj] = V;
          norms_vert[jj] = N;

          phi_sup[jj] = phi0 (tmp_eps[i1], tmp_eps[i2], tmp_phi[i1], tmp_phi[i2], fract);
        }

        const double area = areaTriangle (vert_triangles);

        for (size_t ia = 0; ia < num_atoms; ++ia) {
          const double qi = charge_atoms_tmp[ia];
          const std::array<double,3> &ra = pos_atoms_tmp[ia];

          for (int kk = 0; kk < 3; ++kk) {
            dist_vert = {vert_triangles[kk][0] - ra[0],
                         vert_triangles[kk][1] - ra[1],
                         vert_triangles[kk][2] - ra[2]
                        };
            const double r2 = dist_vert[0]*dist_vert[0] + dist_vert[1]*dist_vert[1] + dist_vert[2]*dist_vert[2];
            const double r = std::sqrt (r2);
            const double inv_r3 = 1.0 / (r2 * r);
            const double inv_r5 = inv_r3 / r2;
            const double dot = dist_vert[0]*norms_vert[kk][0] + dist_vert[1]*norms_vert[kk][1] + dist_vert[2]*norms_vert[kk][2];
            const double factor = phi_sup[kk] * inv_4pi * area / 3.0;

            second_int += qi * phi_sup[kk] * dot * inv_r3 * inv_4pi * area / 3.0;
          }
        }
      }
    }

    this->energy_pol = 0.5 * constant_pol * first_int;
    this->energy_react = 0.5 * (second_int - first_int * constant_react);
  }


  auto reduce_double = [&] (double &x) {
    MPI_Reduce (rank == 0 ? MPI_IN_PLACE : &x, &x, 1, MPI_DOUBLE, MPI_SUM, 0, mpicomm);
  };

  auto reduce_vec = [&] (std::vector<double> &v) {
    MPI_Reduce (rank == 0 ? MPI_IN_PLACE : v.data(),
                rank == 0 ? v.data() : nullptr,
                (int)v.size(), MPI_DOUBLE, MPI_SUM, 0, mpicomm);
  };

  reduce_double (charge_pol);
  reduce_double (energy_pol);
  reduce_double (energy_react);

  // Print the result
  if (rank == 0) {
    constexpr int label_width = 50;
    constexpr int precision = 16;

    std::cout << std::left << std::setw (label_width) << "  Net charge [e]:"
              << std::setprecision (precision) << net_charge << "\n";

    std::cout << std::left << std::setw (label_width) << "  Flux charge [e]:"
              << std::setprecision (precision) << charge_pol / (4.0 * pi) << "\n";

    // std::cout << std::left << std::setw(label_width)
    // << "    Error w.r.t. net charge [%]:"
    // << std::setprecision(6)
    // << ((charge_pol / (4.0 * pi) - net_charge) / net_charge * 100.0) << "\n";

    std::cout << std::left << std::setw (label_width) << "  Polarization energy [kT]:"
              << std::setprecision (precision) << energy_pol << "\n";

    if (calc_energy == 2) {
      std::cout << std::left << std::setw (label_width) << "  Direct ionic energy [kT]:"
                << std::setprecision (precision) << energy_react << "\n";
    }

    if (calc_coulombic == 1) {
      std::cout << std::left << std::setw (label_width) << "  Coulombic energy [kT]:"
                << std::setprecision (precision) << coul_energy << "\n";
    }

    std::cout << std::left << std::setw (label_width) << "  Sum of electrostatic energy contributions [kT]:"
              << std::setprecision (precision)
              << (energy_pol + energy_react + coul_energy) << "\n";

    std::cout << "===========================================================\n";
  }
}

void
poisson_boltzmann::write_potential_on_surface (ray_cache_t & ray_cache)
{
  int rank, size;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &size);

  const double eps_in = 4.0 * pi * e_0 * e_in * kb * T * Angs / (e * e);
  const double eps_out = 4.0 * pi * e_0 * e_out * kb * T * Angs / (e * e);
  const double C_0 = 1.0e3 * N_av * ionic_strength;
  const double k2 = 2.0 * C_0 * Angs * Angs * e * e / (e_0 * e_out * kb * T);
  const double k = std::sqrt (k2);

  std::array<double, 3> V, N, h, area_h;
  std::array<double, 8> tmp_eps, tmp_phi;
  std::vector<int> edg, fl_dir;
  std::array<std::array<double, 3>, 3> vert_triangles, norms_vert;
  std::array<double, 3> phi_sup;

  int cubeindex = -1;
  int edge, i1 = 0, i2 = 0, ntriang = 0;
  double fract;
  double tmp_phi_1 = 0.0, tmp_phi_2 = 0.0;
  double tmp_eps_1 = 0.0, tmp_eps_2 = 0.0;

  auto quadrant = this->tmsh.begin_quadrant_sweep();

  // File setup
  // std::string filename_nodes = "phi_nodes_" + pqrfilename + ".txt";
  // std::string filename_surf  = "phi_surf_" + pqrfilename + ".txt";

  // std::ofstream phi_nodes_txt(filename_nodes);
  // std::ofstream phi_surf_txt(filename_surf);
  // FILE* phi_nod_delphi = std::fopen("filename_nodes_delphi.txt", "w");
  // FILE* phi_sup_delphi = std::fopen("filename_sup_delphi.txt", "w");
  std::vector<std::string> phi_nodes_local;
  std::vector<std::string> phi_surf_local;
  // std::vector<std::string> phi_nodes_delphi_local;
  // std::vector<std::string> phi_sup_delphi_local;

  for (const int ii : border_quad) {
    quadrant[ii];
    cubeindex = classifyCube (quadrant, eps_out);

    // Compute edge lengths and area scale factors
    h[0] = quadrant->p (0, 7) - quadrant->p (0, 0);
    h[1] = quadrant->p (1, 7) - quadrant->p (1, 0);
    h[2] = quadrant->p (2, 7) - quadrant->p (2, 0);
    area_h[0] = h[1] * h[2] / h[0] * 0.25;
    area_h[1] = h[0] * h[2] / h[1] * 0.25;
    area_h[2] = h[0] * h[1] / h[2] * 0.25;

    std::tie (tmp_phi, tmp_eps, edg, fl_dir) = classifyCube_flux (quadrant, tmp_phi, tmp_eps);
    ntriang = getTriangles (cubeindex, triangles);

    for (int t = 0; t < ntriang; ++t) {
      for (int j = 0; j < 3; ++j) {
        edge = triangles[t][j];
        i1 = edge2nodes[2 * edge];
        i2 = edge2nodes[2 * edge + 1];

        // Intersection and vertex
        V[0] = quadrant->p (0, i1);
        V[1] = quadrant->p (1, i1);
        V[2] = quadrant->p (2, i1);

        normal_intersection (quadrant, ray_cache, edge, N, fract);
        V[edge_axis[edge]] += fract * h[edge_axis[edge]];

        vert_triangles[j] = V;
        norms_vert[j] = N;

        // Interpolate phi/epsilon at i1
        if (!quadrant->is_hanging (i1)) {
          tmp_phi_1 = (*phi)[quadrant->gt (i1)];
          tmp_eps_1 = (*epsilon_nodes)[quadrant->gt (i1)];
        } else {
          tmp_phi_1 = tmp_eps_1 = 0.0;
          int np = quadrant->num_parents (i1);

          for (int k = 0; k < np; ++k) {
            tmp_phi_1 += (*phi)[quadrant->gparent (k, i1)] / np;
            tmp_eps_1 += (*epsilon_nodes)[quadrant->gparent (k, i1)] / np;
          }
        }

        // Interpolate phi/epsilon at i2
        if (!quadrant->is_hanging (i2)) {
          tmp_phi_2 = (*phi)[quadrant->gt (i2)];
          tmp_eps_2 = (*epsilon_nodes)[quadrant->gt (i2)];
        } else {
          tmp_phi_2 = tmp_eps_2 = 0.0;
          int np = quadrant->num_parents (i2);

          for (int k = 0; k < np; ++k) {
            tmp_phi_2 += (*phi)[quadrant->gparent (k, i2)] / np;
            tmp_eps_2 += (*epsilon_nodes)[quadrant->gparent (k, i2)] / np;
          }
        }

        // Interpolate phi on surface
        phi_sup[j] = phi0 (tmp_eps_1, tmp_eps_2, tmp_phi_1, tmp_phi_2, fract);

        std::ostringstream oss;
        oss << std::scientific << std::setprecision (5)
            << quadrant->p (0, i1) << " "
            << quadrant->p (1, i1) << " "
            << quadrant->p (2, i1) << " "
            << tmp_phi_1 << "\n"
            << quadrant->p (0, i2) << " "
            << quadrant->p (1, i2) << " "
            << quadrant->p (2, i2) << " "
            << tmp_phi_2;
        phi_nodes_local.push_back (oss.str());
        oss.str ("");
        oss.clear();
        oss << std::scientific << std::setprecision (5)
            << V[0] << " "
            << V[1] << " "
            << V[2] << " "
            << phi_sup[j];
        phi_surf_local.push_back (oss.str());
        oss.str ("");
        oss.clear();
        // // Write to ASCII
        // phi_nodes_txt << quadrant->p(0, i1) << "  " << quadrant->p(1, i1) << "  " << quadrant->p(2, i1) << "  " << tmp_phi_1 << "\n";
        // phi_nodes_txt << quadrant->p(0, i2) << "  " << quadrant->p(1, i2) << "  " << quadrant->p(2, i2) << "  " << tmp_phi_2 << "\n";

        // phi_surf_txt  << V[0] << "  " << V[1] << "  " << V[2] << "  " << phi_sup[j] << "\n";

        // // Write to Delphi PDB-like format
        // std::fprintf(phi_nod_delphi,
        // "\nATOM  %5d %-4s %3s %s%4d    %8.3f%8.3f%8.3f%8.4f%8.4f",
        // 1, "X", "XXX", " ", 0,
        // quadrant->p(0, i1), quadrant->p(1, i1), quadrant->p(2, i1), tmp_phi_1, tmp_phi_2);

        // std::fprintf(phi_nod_delphi,
        // "\nATOM  %5d %-4s %3s %s%4d    %8.3f%8.3f%8.3f%8.4f%8.4f",
        // 1, "X", "XXX", " ", 0,
        // quadrant->p(0, i2), quadrant->p(1, i2), quadrant->p(2, i2), tmp_phi_1, tmp_phi_2);

        // std::fprintf(phi_sup_delphi,
        // "\nATOM  %5d %-4s %3s %s%4d    %8.3f%8.3f%8.3f%8.4f%8.4f",
        // 1, "X", "XXX", " ", 0,
        // V[0], V[1], V[2], phi_sup[j], 0.0);
      }
    }
  }

  auto gather_and_write = [&] (const std::string& filename,
  const std::vector<std::string>& local_lines) {
    if (rank == 0) {
      std::ofstream ofs (filename);

      for (const auto& line : local_lines)
        ofs << line << "\n";

      for (int r = 1; r < size; ++r) {
        int n_lines;
        MPI_Recv (&n_lines, 1, MPI_INT, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int i = 0; i < n_lines; ++i) {
          char buf[512];
          MPI_Recv (buf, 512, MPI_CHAR, r, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          ofs << buf << "\n";
        }
      }

      ofs.close();
    } else {
      int n_lines = static_cast<int> (local_lines.size());
      MPI_Send (&n_lines, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

      for (const auto& line : local_lines) {
        MPI_Send (line.c_str(), static_cast<int> (line.size()) + 1,
                  MPI_CHAR, 0, 1, MPI_COMM_WORLD);
      }
    }
  };

  gather_and_write ("phi_nodes.txt", phi_nodes_local);
  gather_and_write ("phi_surf.txt", phi_surf_local);
  // Close files
  // phi_nodes_txt.close();
  // phi_surf_txt.close();
  // std::fclose(phi_nod_delphi);
  // std::fclose(phi_sup_delphi);
}


double
poisson_boltzmann::coulomb_boundary_conditions (double x, double y, double z)
{
  double eps_out = 4.0*pi*e_0*e_out*kb*T*Angs/ (e*e); //adim e_out
  double C_0 = 1.0e3*N_av*ionic_strength; //Bulk concentration of monovalent species
  double k2 = 2.0*C_0*Angs*Angs*e*e/ (e_0*e_out*kb*T);

  double dist = 0.0;
  double pot = 0.0;
  double k = std::sqrt (k2);

  for (const NS::Atom& i : atoms) {
    dist = std::hypot ( (i.pos[0] - x), (i.pos[1] - y), (i.pos[2] - z));
    pot += i.charge*exp (-k*dist)/ (dist*eps_out);
  }

  return pot;
}

double
poisson_boltzmann::analytic_solution (double x, double y, double z)
{
  double eps_out = 4.0*pi*e_0*e_out*kb*T*Angs/ (e*e); //adim e_out
  double eps_in = 4.0*pi*e_0*e_in*kb*T*Angs/ (e*e); //adim e_in
  double C_0 = 1.0e3*N_av*ionic_strength; //Bulk concentration of monovalent species
  double k2 = 2.0*C_0*Angs*Angs*e*e/ (e_0*e_out*kb*T);
  double k = std::sqrt (k2);
  double rs = 2.0;
  double pippo_out = eps_out* (1.0+k*rs);
  double pippo_in = eps_in*rs;
  double dist= 0.0;
  double pot = 0.0;

  dist = std::hypot (x, y, z);

  if (dist <= rs)
    pot = 1.0/ (pippo_out*rs) + (rs-dist)/ (pippo_in*dist);
  else
    pot = exp (k* (rs-dist))/ (dist*pippo_out);

  return pot;
}



void
poisson_boltzmann::analitic_potential ()
{
  double eps_out = 4.0*pi*e_0*e_out*kb*T*Angs/ (e*e); //adim e_out
  double eps_in = 4.0*pi*e_0*e_in*kb*T*Angs/ (e*e); //adim e_in
  double C_0 = 1.0e3*N_av*ionic_strength; //Bulk concentration of monovalent species
  double k2 = 2.0*C_0*Angs*Angs*e*e/ (e_0*e_out*kb*T);
  double k = std::sqrt (k2);
  double rs = 2.0;
  double pippo_out = eps_out* (1.0+k*rs);
  double pippo_in = eps_in*rs;
  double dist= 0.0;


  distributed_vector phi_an (tmsh.num_owned_nodes (), mpicomm);
  // distributed_vector field_an (tmsh.num_owned_nodes (), mpicomm);
  bim3a_solution_with_ghosts (tmsh, phi_an, replace_op);
  // bim3a_solution_with_ghosts (tmsh, field_an, replace_op);

  for (auto quadrant = this->tmsh.begin_quadrant_sweep ();
       quadrant != this->tmsh.end_quadrant_sweep ();
       ++quadrant) {
    for (const NS::Atom& i : atoms) {
      if (std::fabs (i.charge) > std::numeric_limits<double>::epsilon ()) {
        for (int ii = 0; ii < 8; ++ii) {
          if (! quadrant->is_hanging (ii)) {
            dist = std::hypot ( (i.pos[0] - quadrant->p (0,ii)),
                                (i.pos[1] - quadrant->p (1,ii)),
                                (i.pos[2] - quadrant->p (2,ii)));

            if (dist <= rs) {
              phi_an[quadrant->gt (ii)] = 1.0/ (pippo_out*rs) +
                                          i.charge* (rs-dist)/ (pippo_in*dist);
              // field_an[quadrant->gt (ii)] = (1.0 + (10.0-dist)/dist)*
              // i.charge/(pippo_in*dist);
            } else
              phi_an[quadrant->gt (ii)] = i.charge*exp (k* (rs-dist))/ (dist*pippo_out);

            // field_an[quadrant->gt (ii)] = phi_an[quadrant->gt (ii)]*(k +1.0/dist);
          }

          // else{
          // for (int jj = 0; jj < quadrant->num_parents (ii); ++jj){
          // phi_an[quadrant->gparent (jj, ii)] += 0.0;
          // field_an[quadrant->gparent (jj, ii)] += 0.0;
          // }
          // }
        }
      }
    }
  }

  phi_an.assemble (replace_op);
  tmsh.octbin_export ("phi_an_0", phi_an);
  // field_an.assemble (replace_op);
  // tmsh.octbin_export ("field_an_0", field_an);
}

bool
poisson_boltzmann::controlla_coordinate (int i, const p8est_quadrant_t *quadrant)
{

  double tol = p4esttol * (rr[0]-ll[0]);

  if (mesh_shape == 2)
    tol = p4esttol * (r_c[0]-l_c[0]);


  bool retval = false;

  p8est_quadrant_t node;
  int ii, jj;
  double vxyz [3 * 8] = {0,0,0, 0,0,0, 0,0,0, 0,0,0,
                         0,0,0, 0,0,0, 0,0,0, 0,0,0
                        };

  for (ii = 0; ii < 8; ++ii) {
    p8est_quadrant_corner_node (quadrant, ii, &node);
    p8est_qcoord_to_vertex (tmsh.p8est->connectivity, 0,
                            node.x, node.y, node.z, & (vxyz[3 * ii]));
  }

  double l, r, t, b, f, bk;

  l = vxyz[0];
  r = vxyz[3*7];

  f = vxyz[1];
  bk = vxyz[3*7 +1];

  b = vxyz[2];
  t = vxyz[3*7 +2];

  // retval = (atoms[i].pos[0] > l- tol) && (atoms[i].pos[0] <= r- tol); //make sure that the charge is assigned only once
  // retval = retval && (atoms[i].pos[1] > f- tol) && (atoms[i].pos[1] <= bk- tol);
  // retval = retval && (atoms[i].pos[2] > b- tol) && (atoms[i].pos[2] <= t- tol);
  retval = (pos_atoms[i][0] > l- tol) && (pos_atoms[i][0] <= r- tol); //make sure that the charge is assigned only once
  retval = retval && (pos_atoms[i][1] > f- tol) && (pos_atoms[i][1] <= bk- tol);
  retval = retval && (pos_atoms[i][2] > b- tol) && (pos_atoms[i][2] <= t- tol);

  return retval;
}


int
poisson_boltzmann::cerca_atomo (p8est_t * p4est,
                                p4est_topidx_t which_tree,
                                p8est_quadrant_t * quadrant,
                                p4est_locidx_t local_num,
                                void *point)
{
  int *pt = (int *) point;
  // std::cout << "\n pt: " <<*pt <<"\n"<< std::endl;
  bool tf = controlla_coordinate (*pt, quadrant);

  if (tf) {
    if (local_num >= 0) {
      auto quadrant = this->tmsh.begin_quadrant_sweep ();
      quadrant[local_num];
      tmesh_3d::quadrant_t qi = tmsh.current_quadrant;
      lookup_table.emplace (*pt, qi);

    }

    return 1;
  }

  return 0;

}

void 
poisson_boltzmann::search_points()
{
    size_t count = charge_atoms.size();
    auto base = std::make_unique<int[]>(count);

    for (size_t ii = 0; ii < count; ++ii)
        base[ii] = ii;

    sc_array_t* points = sc_array_new_data(
        base.get(), sizeof(int), count);

    // Set the global pointer for the callback
    pb_global_wrapper = this;

    // call the search function
    p8est_search_local(
        tmsh.p8est,
        0,
        NULL,
        cerca_atomo_wrapper,
        points
    );
}

void
poisson_boltzmann::pot_field_fast (ray_cache_t & ray_cache)
{
  int rank;
  MPI_Comm_rank (mpicomm, &rank);

  if (rank == 0)
    std::cout << "\n================ [ Calculating Potential & Field Components ] =================\n";

  // ===========================
  // Costanti fisiche e scalari
  // ===========================
  const double inv_4pi = 1.0 / (4.0 * pi);
  const double eps0 = e_0; // Permittività del vuoto
  const double eps_in = 4.0 * pi * eps0 * e_in * kb * T * Angs / (e * e);
  const double eps_out = 4.0 * pi * eps0 * e_out * kb * T * Angs / (e * e);

  const double C0 = 1.0e3 * N_av * ionic_strength; // [mol/m^3]
  const double k2 = 2.0 * C0 * Angs * Angs * e * e / (eps0 * e_out * kb * T);
  const double k = std::sqrt (k2);

  const double den_in = 1.0 / eps_in;
  const double constant_pol = (1.0 / eps_out - 1.0 / eps_in) * inv_4pi;
  const double constant_react = (1.0 / eps_out) * inv_4pi;

  // ===========================
  // Variabili locali
  // ===========================
  // double energy_pol = 0.0;
  // double energy_react = 0.0;
  // double coul_energy = 0.0;
  double charge_pol = 0.0;
  const size_t num_atoms = charge_atoms.size();

  std::vector<double> phi_c, phi_p, phi_i;
  std::vector<double> field_cx, field_px, field_ix;
  std::vector<double> field_cy, field_py, field_iy;
  std::vector<double> field_cz, field_pz, field_iz;

  phi_c.assign (num_atoms, 0.0);
  field_cx.assign (num_atoms, 0.0);
  field_cy.assign (num_atoms, 0.0);
  field_cz.assign (num_atoms, 0.0);

  for (size_t i = 0; i < num_atoms; ++i) {
    const std::array<double,3> &ri = pos_atoms[i];
    const double qi = charge_atoms[i];

    for (size_t j = i + 1; j < num_atoms; ++j) {
      const std::array<double,3> &rj = pos_atoms[j];
      const double qj = charge_atoms[j];

      const double dx = ri[0] - rj[0];
      const double dy = ri[1] - rj[1];
      const double dz = ri[2] - rj[2];
      const double r2 = dx * dx + dy * dy + dz * dz;
      const double r = std::sqrt (r2);
      const double inv_r3 = 1.0 / (r2 * r);

      // Coulomb energy
      this->coul_energy += (qi * qj) / r * den_in;
      // Coulomb potential
      phi_c[i] += qj / r * den_in;
      phi_c[j] += qi / r * den_in;


      // Coulomb field
      std::array<double, 3> eij = {
        dx * inv_r3 * qj * den_in,
        dy * inv_r3 * qj * den_in,
        dz * inv_r3 * qj * den_in
      };
      std::array<double, 3> eji = {
        -dx * inv_r3 * qi * den_in,
        -dy * inv_r3 * qi * den_in,
        -dz * inv_r3 * qi * den_in
      };

      field_cx[i] += eij[0];
      field_cx[j] += eji[0];
      field_cy[i] += eij[1];
      field_cy[j] += eji[1];
      field_cz[i] += eij[2];
      field_cz[j] += eji[2];
    }
  }

  ////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////

  double first_int = 0.0, second_int = 0.0, distance = 0.0;

  std::array<double,3> h{0}, area_h{0};
  std::array<double,3> V, N;
  std::array<double,8> tmp_eps, tmp_phi;
  std::vector<int> edg, fl_dir;

  auto quadrant = this->tmsh.begin_quadrant_sweep ();

  if (!border_quad.empty()) {
    quadrant[border_quad[0]];

    for (int d = 0; d < 3; ++d)
      h[d] = quadrant->p (d, 7) - quadrant->p (d, 0);

    area_h = {h[1]*h[2]/h[0]*0.25, h[0]*h[2]/h[1]*0.25, h[0]*h[1]/h[2]*0.25};
  }

  // flux and polarization energy calculation
  if ((calc_field_term==1 || (calc_field_term == 2 && k < 1.e-5)) && calc_potential_term < 2) {
    phi_p.assign (num_atoms, 0.0);
    field_px.assign (num_atoms, 0.0);
    field_py.assign (num_atoms, 0.0);
    field_pz.assign (num_atoms, 0.0);

    for (const int ii : border_quad) {
      quadrant[ii];
      std::tie (tmp_phi, tmp_eps, edg, fl_dir) = classifyCube_flux_fast (quadrant, tmp_phi, tmp_eps);

      for (int ip = 0; ip < edg.size (); ++ip) {
        const int edge = edg[ip];
        const int axis = edge_axis[edge];
        const int i1 = edge2nodes[2 * edge];
        const int i2 = edge2nodes[2 * edge + 1];
        double fract = 0.0;
        normal_intersection (quadrant, ray_cache, edge, N, fract);
        V = {quadrant->p (0, i1), quadrant->p (1, i1), quadrant->p (2, i1)};
        V[axis] += fract * h[axis];
        const double tmp_flux =
          - (tmp_phi[i2] - tmp_phi[i1]) * wha (tmp_eps[i1], tmp_eps[i2], fract)
          * fl_dir[ip] * area_h[axis];

        charge_pol += tmp_flux;

        for (size_t ia = 0; ia < num_atoms; ++ia) {
          const std::array<double,3> &ra = pos_atoms[ia];
          const double dx = ra[0] - V[0];
          const double dy = ra[1] - V[1];
          const double dz = ra[2] - V[2];
          const double r = std::sqrt (dx * dx + dy * dy + dz * dz);
          const double inv_r3 = 1.0 / (r * r * r);
          const double qflux = tmp_flux / r;

          first_int += charge_atoms[ia] * qflux;
          phi_p[ia] += qflux * constant_pol;
          field_px[ia] += dx * inv_r3 * tmp_flux * constant_pol;
          field_py[ia] += dy * inv_r3 * tmp_flux * constant_pol;
          field_pz[ia] += dz * inv_r3 * tmp_flux * constant_pol;
        }
      }
    }

    this->energy_pol = 0.5*constant_pol*first_int;
  } else if ((calc_potential_term == 1 || (calc_potential_term == 2 && k < 1.e-5)) && calc_field_term == 0) {
    phi_p.assign (num_atoms, 0.0);

    for (const int ii : border_quad) {
      quadrant[ii];
      std::tie (tmp_phi, tmp_eps, edg, fl_dir) = classifyCube_flux_fast (quadrant, tmp_phi, tmp_eps);

      for (int ip = 0; ip < edg.size (); ++ip) {
        const int edge = edg[ip];
        const int axis = edge_axis[edge];
        const int i1 = edge2nodes[2 * edge];
        const int i2 = edge2nodes[2 * edge + 1];
        double fract = 0.0;
        normal_intersection (quadrant, ray_cache, edge, N, fract);
        V = {quadrant->p (0, i1), quadrant->p (1, i1), quadrant->p (2, i1)};
        V[axis] += fract * h[axis];
        const double tmp_flux =
          - (tmp_phi[i2] - tmp_phi[i1]) * wha (tmp_eps[i1], tmp_eps[i2], fract)
          * fl_dir[ip] * area_h[axis];

        charge_pol += tmp_flux;

        for (size_t ia = 0; ia < num_atoms; ++ia) {
          const std::array<double,3> &ra = pos_atoms[ia];
          const double dx = ra[0] - V[0];
          const double dy = ra[1] - V[1];
          const double dz = ra[2] - V[2];
          const double r = std::sqrt (dx * dx + dy * dy + dz * dz);
          const double inv_r3 = 1.0 / (r * r * r);
          const double qflux = tmp_flux / r;

          first_int += charge_atoms[ia] * qflux;
          phi_p[ia] += qflux * constant_pol;

        }
      }
    }

    this->energy_pol = 0.5*constant_pol*first_int;
  }

  auto allocate_potential_fields = [&] (void) {
    phi_p.assign (num_atoms, 0.0);
    phi_i.assign (num_atoms, 0.0);
    field_px.assign (num_atoms, 0.0);
    field_py.assign (num_atoms, 0.0);
    field_pz.assign (num_atoms, 0.0);
    field_ix.assign (num_atoms, 0.0);
    field_iy.assign (num_atoms, 0.0);
    field_iz.assign (num_atoms, 0.0);
  };

  //polarization energy + ionic energy
  if (calc_field_term==2 && k > 1.e-5) {
    allocate_potential_fields();
    int cubeindex = -1;
    std::array<std::array<double,3>,3> vert_triangles, norms_vert;
    std::array<double,3> dist_vert, phi_sup;
    int ntriang = 0;

    for (const int ii : border_quad) {
      quadrant[ii];
      cubeindex = classifyCube_fast (quadrant, eps_out);
      std::tie (tmp_phi, tmp_eps, edg, fl_dir) = classifyCube_flux_fast (quadrant, tmp_phi, tmp_eps);
      ntriang = getTriangles (cubeindex, triangles);

      // --- flussi
      for (int ip = 0; ip < edg.size (); ++ip) {
        const int edge = edg[ip];
        const int axis = edge_axis[edge];
        const int i1 = edge2nodes[2 * edge];
        const int i2 = edge2nodes[2 * edge + 1];

        double fract = 0.0;
        normal_intersection (quadrant, ray_cache, edge, N, fract);

        V = {quadrant->p (0, i1), quadrant->p (1, i1), quadrant->p (2, i1)};
        V[axis] += fract * h[axis];

        const double tmp_flux =
          - (tmp_phi[i2] - tmp_phi[i1]) * wha (tmp_eps[i1], tmp_eps[i2], fract)
          * fl_dir[ip] * area_h[axis];
        charge_pol += tmp_flux;

        for (size_t ia = 0; ia < num_atoms; ++ia) {
          const std::array<double,3> &ra = pos_atoms[ia];
          const double dx = ra[0] - V[0];
          const double dy = ra[1] - V[1];
          const double dz = ra[2] - V[2];
          const double r = std::sqrt (dx * dx + dy * dy + dz * dz);
          const double inv_r3 = 1.0 / (r * r * r);

          first_int += charge_atoms[ia] * tmp_flux / r;
          phi_p[ia] += tmp_flux / r * constant_pol;
          field_px[ia] += dx * inv_r3 * tmp_flux * constant_pol;
          field_py[ia] += dy * inv_r3 * tmp_flux * constant_pol;
          field_pz[ia] += dz * inv_r3 * tmp_flux * constant_pol;
        }
      }

      // --- triangoli (componente ionica)
      for (int itri = 0; itri < ntriang; ++itri) {
        for (int jj = 0; jj < 3; ++jj) {
          const int edge = triangles[itri][jj];
          const int axis = edge_axis[edge];
          const int i1 = edge2nodes[2 * edge];
          const int i2 = edge2nodes[2 * edge + 1];

          double fract = 0.0;
          normal_intersection (quadrant, ray_cache, edge, N, fract);

          V = {quadrant->p (0, i1), quadrant->p (1, i1), quadrant->p (2, i1)};
          V[axis] += fract * h[axis];

          vert_triangles[jj] = V;
          norms_vert[jj] = N;

          phi_sup[jj] = phi0 (tmp_eps[i1], tmp_eps[i2], tmp_phi[i1], tmp_phi[i2], fract);
        }

        const double area = areaTriangle (vert_triangles);

        for (size_t ia = 0; ia < num_atoms; ++ia) {
          const double qi = charge_atoms[ia];
          const std::array<double,3> &ra = pos_atoms[ia];

          for (int kk = 0; kk < 3; ++kk) {
            dist_vert = {vert_triangles[kk][0] - ra[0],
                         vert_triangles[kk][1] - ra[1],
                         vert_triangles[kk][2] - ra[2]
                        };
            const double r2 = dist_vert[0]*dist_vert[0] + dist_vert[1]*dist_vert[1] + dist_vert[2]*dist_vert[2];
            const double r = std::sqrt (r2);
            const double inv_r3 = 1.0 / (r2 * r);
            const double inv_r5 = inv_r3 / r2;
            const double dot = dist_vert[0]*norms_vert[kk][0] + dist_vert[1]*norms_vert[kk][1] + dist_vert[2]*norms_vert[kk][2];
            const double factor = phi_sup[kk] * inv_4pi * area / 3.0;

            second_int += qi * phi_sup[kk] * dot * inv_r3 * inv_4pi * area / 3.0;
            phi_i[ia] += phi_sup[kk] * dot * inv_r3 * inv_4pi * area / 3.0;

            field_ix[ia] += factor * (-3 * dist_vert[0] * inv_r5 * dot + norms_vert[kk][0] * inv_r3);
            field_iy[ia] += factor * (-3 * dist_vert[1] * inv_r5 * dot + norms_vert[kk][1] * inv_r3);
            field_iz[ia] += factor * (-3 * dist_vert[2] * inv_r5 * dot + norms_vert[kk][2] * inv_r3);
          }
        }
      }
    }

    this->energy_pol = 0.5 * constant_pol * first_int;
    this->energy_react = 0.5 * (second_int - first_int * constant_react);
  } else if ((calc_potential_term==2 && k > 1.e-5) && calc_field_term == 0) {
    phi_p.assign (num_atoms, 0.0);
    phi_i.assign (num_atoms, 0.0);
    int cubeindex = -1;
    std::array<std::array<double,3>,3> vert_triangles, norms_vert;
    std::array<double,3> dist_vert, phi_sup;
    int ntriang = 0;

    for (const int ii : border_quad) {
      quadrant[ii];
      cubeindex = classifyCube_fast (quadrant, eps_out);
      std::tie (tmp_phi, tmp_eps, edg, fl_dir) = classifyCube_flux_fast (quadrant, tmp_phi, tmp_eps);
      ntriang = getTriangles (cubeindex, triangles);

      // --- flussi
      for (int ip = 0; ip < edg.size (); ++ip) {
        const int edge = edg[ip];
        const int axis = edge_axis[edge];
        const int i1 = edge2nodes[2 * edge];
        const int i2 = edge2nodes[2 * edge + 1];

        double fract = 0.0;
        normal_intersection (quadrant, ray_cache, edge, N, fract);

        V = {quadrant->p (0, i1), quadrant->p (1, i1), quadrant->p (2, i1)};
        V[axis] += fract * h[axis];

        const double tmp_flux =
          - (tmp_phi[i2] - tmp_phi[i1]) * wha (tmp_eps[i1], tmp_eps[i2], fract)
          * fl_dir[ip] * area_h[axis];
        charge_pol += tmp_flux;

        for (size_t ia = 0; ia < num_atoms; ++ia) {
          const std::array<double,3> &ra = pos_atoms[ia];
          const double dx = ra[0] - V[0];
          const double dy = ra[1] - V[1];
          const double dz = ra[2] - V[2];
          const double r = std::sqrt (dx * dx + dy * dy + dz * dz);
          const double inv_r3 = 1.0 / (r * r * r);
          first_int += charge_atoms[ia] * tmp_flux / r;
          phi_p[ia] += tmp_flux / r * constant_pol;
        }
      }

      // --- triangoli (componente ionica)
      for (int itri = 0; itri < ntriang; ++itri) {
        for (int jj = 0; jj < 3; ++jj) {
          const int edge = triangles[itri][jj];
          const int axis = edge_axis[edge];
          const int i1 = edge2nodes[2 * edge];
          const int i2 = edge2nodes[2 * edge + 1];

          double fract = 0.0;
          normal_intersection (quadrant, ray_cache, edge, N, fract);

          V = {quadrant->p (0, i1), quadrant->p (1, i1), quadrant->p (2, i1)};
          V[axis] += fract * h[axis];

          vert_triangles[jj] = V;
          norms_vert[jj] = N;

          phi_sup[jj] = phi0 (tmp_eps[i1], tmp_eps[i2], tmp_phi[i1], tmp_phi[i2], fract);
        }

        const double area = areaTriangle (vert_triangles);

        for (size_t ia = 0; ia < num_atoms; ++ia) {
          const double qi = charge_atoms[ia];
          const std::array<double,3> &ra = pos_atoms[ia];

          for (int kk = 0; kk < 3; ++kk) {
            dist_vert = {vert_triangles[kk][0] - ra[0],
                         vert_triangles[kk][1] - ra[1],
                         vert_triangles[kk][2] - ra[2]
                        };
            const double r2 = dist_vert[0]*dist_vert[0] + dist_vert[1]*dist_vert[1] + dist_vert[2]*dist_vert[2];
            const double r = std::sqrt (r2);
            const double inv_r3 = 1.0 / (r2 * r);
            const double inv_r5 = inv_r3 / r2;
            const double dot = dist_vert[0]*norms_vert[kk][0] + dist_vert[1]*norms_vert[kk][1] + dist_vert[2]*norms_vert[kk][2];
            const double factor = phi_sup[kk] * inv_4pi * area / 3.0;
            second_int += qi * phi_sup[kk] * dot * inv_r3 * inv_4pi * area / 3.0;
            phi_i[ia] += phi_sup[kk] * dot * inv_r3 * inv_4pi * area / 3.0;
          }
        }
      }
    }

    this->energy_pol = 0.5 * constant_pol * first_int;
    this->energy_react = 0.5 * (second_int - first_int * constant_react);
  } else if ((calc_potential_term==2 && k > 1.e-5) && calc_field_term == 1) {
    phi_p.assign (num_atoms, 0.0);
    phi_i.assign (num_atoms, 0.0);
    field_px.assign (num_atoms, 0.0);
    field_py.assign (num_atoms, 0.0);
    field_pz.assign (num_atoms, 0.0);
    int cubeindex = -1;
    std::array<std::array<double,3>,3> vert_triangles, norms_vert;
    std::array<double,3> dist_vert, phi_sup;
    int ntriang = 0;

    for (const int ii : border_quad) {
      quadrant[ii];
      cubeindex = classifyCube_fast (quadrant, eps_out);
      std::tie (tmp_phi, tmp_eps, edg, fl_dir) = classifyCube_flux_fast (quadrant, tmp_phi, tmp_eps);
      ntriang = getTriangles (cubeindex, triangles);

      // --- flussi
      for (int ip = 0; ip < edg.size (); ++ip) {
        const int edge = edg[ip];
        const int axis = edge_axis[edge];
        const int i1 = edge2nodes[2 * edge];
        const int i2 = edge2nodes[2 * edge + 1];

        double fract = 0.0;
        normal_intersection (quadrant, ray_cache, edge, N, fract);

        V = {quadrant->p (0, i1), quadrant->p (1, i1), quadrant->p (2, i1)};
        V[axis] += fract * h[axis];

        const double tmp_flux =
          - (tmp_phi[i2] - tmp_phi[i1]) * wha (tmp_eps[i1], tmp_eps[i2], fract)
          * fl_dir[ip] * area_h[axis];
        charge_pol += tmp_flux;

        for (size_t ia = 0; ia < num_atoms; ++ia) {
          const std::array<double,3> &ra = pos_atoms[ia];
          const double dx = ra[0] - V[0];
          const double dy = ra[1] - V[1];
          const double dz = ra[2] - V[2];
          const double r = std::sqrt (dx * dx + dy * dy + dz * dz);
          const double inv_r3 = 1.0 / (r * r * r);
          first_int += charge_atoms[ia] * tmp_flux / r;
          phi_p[ia] += tmp_flux / r * constant_pol;
          field_px[ia] += dx * inv_r3 * tmp_flux * constant_pol;
          field_py[ia] += dy * inv_r3 * tmp_flux * constant_pol;
          field_pz[ia] += dz * inv_r3 * tmp_flux * constant_pol;
        }
      }

      // --- triangoli (componente ionica)
      for (int itri = 0; itri < ntriang; ++itri) {
        for (int jj = 0; jj < 3; ++jj) {
          const int edge = triangles[itri][jj];
          const int axis = edge_axis[edge];
          const int i1 = edge2nodes[2 * edge];
          const int i2 = edge2nodes[2 * edge + 1];

          double fract = 0.0;
          normal_intersection (quadrant, ray_cache, edge, N, fract);

          V = {quadrant->p (0, i1), quadrant->p (1, i1), quadrant->p (2, i1)};
          V[axis] += fract * h[axis];

          vert_triangles[jj] = V;
          norms_vert[jj] = N;

          phi_sup[jj] = phi0 (tmp_eps[i1], tmp_eps[i2], tmp_phi[i1], tmp_phi[i2], fract);
        }

        const double area = areaTriangle (vert_triangles);

        for (size_t ia = 0; ia < num_atoms; ++ia) {
          const double qi = charge_atoms[ia];
          const std::array<double,3> &ra = pos_atoms[ia];

          for (int kk = 0; kk < 3; ++kk) {
            dist_vert = {vert_triangles[kk][0] - ra[0],
                         vert_triangles[kk][1] - ra[1],
                         vert_triangles[kk][2] - ra[2]
                        };
            const double r2 = dist_vert[0]*dist_vert[0] + dist_vert[1]*dist_vert[1] + dist_vert[2]*dist_vert[2];
            const double r = std::sqrt (r2);
            const double inv_r3 = 1.0 / (r2 * r);
            const double inv_r5 = inv_r3 / r2;
            const double dot = dist_vert[0]*norms_vert[kk][0] + dist_vert[1]*norms_vert[kk][1] + dist_vert[2]*norms_vert[kk][2];
            const double factor = phi_sup[kk] * inv_4pi * area / 3.0;
            second_int += qi * phi_sup[kk] * dot * inv_r3 * inv_4pi * area / 3.0;
            phi_i[ia] += phi_sup[kk] * dot * inv_r3 * inv_4pi * area / 3.0;
          }
        }
      }
    }

    this->energy_pol = 0.5 * constant_pol * first_int;
    this->energy_react = 0.5 * (second_int - first_int * constant_react);
  }


  auto reduce_double = [&] (double &x) {
    MPI_Reduce (rank == 0 ? MPI_IN_PLACE : &x, &x, 1, MPI_DOUBLE, MPI_SUM, 0, mpicomm);
  };

  auto reduce_vec = [&] (std::vector<double> &v) {
    MPI_Reduce (rank == 0 ? MPI_IN_PLACE : v.data(),
                rank == 0 ? v.data() : nullptr,
                (int)v.size(), MPI_DOUBLE, MPI_SUM, 0, mpicomm);
  };

  reduce_double (charge_pol);
  reduce_double (energy_pol);
  reduce_double (energy_react);
  reduce_vec (phi_p);
  reduce_vec (phi_i);
  reduce_vec (field_px);
  reduce_vec (field_py);
  reduce_vec (field_pz);
  reduce_vec (field_ix);
  reduce_vec (field_iy);
  reduce_vec (field_iz);

  // Print the result
  if (rank == 0) {
    constexpr int label_width = 50;
    constexpr int precision = 16;


    if (calc_potential_term == 1 || calc_field_term == 1 || calc_potential_term == 2 || calc_field_term == 2) {
      std::cout << std::left << std::setw (label_width) << "  Polarization energy [kT]:"
                << std::setprecision (precision) << energy_pol << "\n";
    }

    if (calc_potential_term == 2 || calc_field_term == 2) {
      std::cout << std::left << std::setw (label_width) << "  Direct ionic energy [kT]:"
                << std::setprecision (precision) << energy_react << "\n";
    }


    std::cout << std::left << std::setw (label_width) << "  Coulombic energy [kT]:"
              << std::setprecision (precision) << coul_energy << "\n";


    std::cout << std::left << std::setw (label_width) << "  Sum of electrostatic energy contributions [kT]:"
              << std::setprecision (precision)
              << (energy_pol + energy_react + coul_energy) << "\n";

    std::cout << "===========================================================\n";


    // ============================
    // Scrittura su file risultati
    // ============================
    std::ofstream fout ("pot_field.dat");
    std::cout << "Writing potentials and fields to pot_field.dat\n";
    fout << "# index    x    y    z    ";

    if (calc_field_term==2 && k > 1.e-5) {
      fout <<"phi_c    phi_p    phi_i    Ex_c    Ey_c    Ez_c   Ex_p    Ey_p    Ez_p   Ex_i    Ey_i    Ez_i\n";

      for (size_t i = 0; i < num_atoms; ++i) {
        phi_i[i] -= phi_p[i] / constant_pol * constant_react;
        field_ix[i] -= field_px[i] / constant_pol * constant_react;
        field_iy[i] -= field_py[i] / constant_pol * constant_react;
        field_iz[i] -= field_pz[i] / constant_pol * constant_react;
        fout << std::setw (5) << i + 1 << "  "
             << std::setw (8) << pos_atoms[i][0] << "  "
             << std::setw (8) << pos_atoms[i][1] << "  "
             << std::setw (8) << pos_atoms[i][2] << "  "
             << std::setw (8) << phi_c[i] << "  "
             << std::setw (8) << phi_p[i] << "  "
             << std::setw (8) << phi_i[i] << "  ";
        fout << std::setw (8) << field_cx[i] << "  ";
        fout << std::setw (8) << field_cy[i] << "  ";
        fout << std::setw (8) << field_cz[i] << "  ";
        fout << std::setw (8) << field_px[i] << "  ";
        fout << std::setw (8) << field_py[i] << "  ";
        fout << std::setw (8) << field_pz[i] << "  ";
        fout << std::setw (8) << field_ix[i] << "  ";
        fout << std::setw (8) << field_iy[i] << "  ";
        fout << std::setw (8) << field_iz[i] << "  ";
        fout << "\n";
      }
    } else if ((calc_potential_term==2 && k > 1.e-5) && calc_field_term == 0) {
      fout <<"phi_c    phi_p    phi_i    Ex_c    Ey_c    Ez_c\n";

      for (size_t i = 0; i < num_atoms; ++i) {
        phi_i[i] -= phi_p[i] / constant_pol * constant_react;

        fout << std::setw (5) << i + 1 << "  "
             << std::setw (8) << pos_atoms[i][0] << "  "
             << std::setw (8) << pos_atoms[i][1] << "  "
             << std::setw (8) << pos_atoms[i][2] << "  "
             << std::setw (8) << phi_c[i] << "  "
             << std::setw (8) << phi_p[i] << "  "
             << std::setw (8) << phi_i[i] << "  ";
        fout << std::setw (8) << field_cx[i] << "  ";
        fout << std::setw (8) << field_cy[i] << "  ";
        fout << std::setw (8) << field_cz[i] << "  ";
        fout << "\n";
      }
    } else if ((calc_potential_term==2 && k > 1.e-5) && calc_field_term == 1) {
      fout <<"phi_c    phi_p    phi_i    Ex_c    Ey_c    Ez_c   Ex_p    Ey_p    Ez_p\n";

      for (size_t i = 0; i < num_atoms; ++i) {
        phi_i[i] -= phi_p[i] / constant_pol * constant_react;

        fout << std::setw (5) << i + 1 << "  "
             << std::setw (8) << pos_atoms[i][0] << "  "
             << std::setw (8) << pos_atoms[i][1] << "  "
             << std::setw (8) << pos_atoms[i][2] << "  "
             << std::setw (8) << phi_c[i] << "  "
             << std::setw (8) << phi_p[i] << "  "
             << std::setw (8) << phi_i[i] << "  ";
        fout << std::setw (8) << field_cx[i] << "  ";
        fout << std::setw (8) << field_cy[i] << "  ";
        fout << std::setw (8) << field_cz[i] << "  ";
        fout << std::setw (8) << field_px[i] << "  ";
        fout << std::setw (8) << field_py[i] << "  ";
        fout << std::setw (8) << field_pz[i] << "  ";

        fout << "\n";
      }
    } else if ((calc_potential_term == 1 || (calc_potential_term == 2 && k < 1.e-5)) && calc_field_term == 0) {
      fout <<"phi_c    phi_p    Ex_c    Ey_c    Ez_c\n";

      for (size_t i = 0; i < num_atoms; ++i) {
        fout << std::setw (5) << i + 1 << "  "
             << std::setw (8) << pos_atoms[i][0] << "  "
             << std::setw (8) << pos_atoms[i][1] << "  "
             << std::setw (8) << pos_atoms[i][2] << "  "
             << std::setw (8) << phi_c[i] << "  "
             << std::setw (8) << phi_p[i] << "  ";
        fout << std::setw (8) << field_cx[i] << "  ";
        fout << std::setw (8) << field_cy[i] << "  ";
        fout << std::setw (8) << field_cz[i] << "  ";
        fout << "\n";
      }
    } else if ((calc_field_term==1 || (calc_field_term == 2 && k < 1.e-5)) && calc_potential_term < 2) {
      fout <<"phi_c    phi_p    Ex_c    Ey_c    Ez_c   Ex_p    Ey_p    Ez_p\n";

      for (size_t i = 0; i < num_atoms; ++i) {
        fout << std::setw (5) << i + 1 << "  "
             << std::setw (8) << pos_atoms[i][0] << "  "
             << std::setw (8) << pos_atoms[i][1] << "  "
             << std::setw (8) << pos_atoms[i][2] << "  "
             << std::setw (8) << phi_c[i] << "  "
             << std::setw (8) << phi_p[i] << "  ";
        fout << std::setw (8) << field_cx[i] << "  ";
        fout << std::setw (8) << field_cy[i] << "  ";
        fout << std::setw (8) << field_cz[i] << "  ";
        fout << std::setw (8) << field_px[i] << "  ";
        fout << std::setw (8) << field_py[i] << "  ";
        fout << std::setw (8) << field_pz[i] << "  ";

        fout << "\n";
      }
    }


    fout.close();
    std::cout << "Atom potentials and fields written to 'pot_field.dat'\n";
  }
}

void
poisson_boltzmann::pot_field (ray_cache_t & ray_cache)
{
  int rank;
  MPI_Comm_rank (mpicomm, &rank);

  if (rank == 0)
    std::cout << "\n================ [ Calculating Potential & Field Components ] =================\n";

  // ===========================
  // Costanti fisiche e scalari
  // ===========================
  const double inv_4pi = 1.0 / (4.0 * pi);
  const double eps0 = e_0; // Permittività del vuoto
  const double eps_in = 4.0 * pi * eps0 * e_in * kb * T * Angs / (e * e);
  const double eps_out = 4.0 * pi * eps0 * e_out * kb * T * Angs / (e * e);

  const double C0 = 1.0e3 * N_av * ionic_strength; // [mol/m^3]
  const double k2 = 2.0 * C0 * Angs * Angs * e * e / (eps0 * e_out * kb * T);
  const double k = std::sqrt (k2);

  const double den_in = 1.0 / eps_in;
  const double constant_pol = (1.0 / eps_out - 1.0 / eps_in) * inv_4pi;
  const double constant_react = (1.0 / eps_out) * inv_4pi;

  // ===========================
  // Variabili locali
  // ===========================
  // double energy_pol = 0.0;
  // double energy_react = 0.0;
  // double coul_energy = 0.0;
  double charge_pol = 0.0;
  const size_t num_atoms = charge_atoms.size();

  std::vector<double> phi_c, phi_p, phi_i;
  std::vector<double> field_cx, field_px, field_ix;
  std::vector<double> field_cy, field_py, field_iy;
  std::vector<double> field_cz, field_pz, field_iz;

  phi_c.assign (num_atoms, 0.0);
  field_cx.assign (num_atoms, 0.0);
  field_cy.assign (num_atoms, 0.0);
  field_cz.assign (num_atoms, 0.0);

  for (size_t i = 0; i < num_atoms; ++i) {
    const std::array<double,3> &ri = pos_atoms[i];
    const double qi = charge_atoms[i];

    for (size_t j = i + 1; j < num_atoms; ++j) {
      const std::array<double,3> &rj = pos_atoms[j];
      const double qj = charge_atoms[j];

      const double dx = ri[0] - rj[0];
      const double dy = ri[1] - rj[1];
      const double dz = ri[2] - rj[2];
      const double r2 = dx * dx + dy * dy + dz * dz;
      const double r = std::sqrt (r2);
      const double inv_r3 = 1.0 / (r2 * r);

      // Coulomb energy
      this->coul_energy += (qi * qj) / r * den_in;
      // Coulomb potential
      phi_c[i] += qj / r * den_in;
      phi_c[j] += qi / r * den_in;


      // Coulomb field
      std::array<double, 3> eij = {
        dx * inv_r3 * qj * den_in,
        dy * inv_r3 * qj * den_in,
        dz * inv_r3 * qj * den_in
      };
      std::array<double, 3> eji = {
        -dx * inv_r3 * qi * den_in,
        -dy * inv_r3 * qi * den_in,
        -dz * inv_r3 * qi * den_in
      };

      field_cx[i] += eij[0];
      field_cx[j] += eji[0];
      field_cy[i] += eij[1];
      field_cy[j] += eji[1];
      field_cz[i] += eij[2];
      field_cz[j] += eji[2];
    }
  }

  ////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////

  double first_int = 0.0, second_int = 0.0, distance = 0.0;

  std::array<double,3> h{0}, area_h{0};
  std::array<double,3> V, N;
  std::array<double,8> tmp_eps, tmp_phi;
  std::vector<int> edg, fl_dir;

  auto quadrant = this->tmsh.begin_quadrant_sweep ();

  // flux and polarization energy calculation
  if ((calc_field_term==1 || (calc_field_term == 2 && k < 1.e-5)) && calc_potential_term < 2) {
    phi_p.assign (num_atoms, 0.0);
    field_px.assign (num_atoms, 0.0);
    field_py.assign (num_atoms, 0.0);
    field_pz.assign (num_atoms, 0.0);

    for (const int ii : border_quad) {
      quadrant[ii];

      for (int d = 0; d < 3; ++d)
        h[d] = quadrant->p (d, 7) - quadrant->p (d, 0);

      area_h = {h[1]*h[2]/h[0]*0.25, h[0]*h[2]/h[1]*0.25, h[0]*h[1]/h[2]*0.25};
      std::tie (tmp_phi, tmp_eps, edg, fl_dir) = classifyCube_flux (quadrant, tmp_phi, tmp_eps);

      for (int ip = 0; ip < edg.size (); ++ip) {
        const int edge = edg[ip];
        const int axis = edge_axis[edge];
        const int i1 = edge2nodes[2 * edge];
        const int i2 = edge2nodes[2 * edge + 1];
        double fract = 0.0;
        normal_intersection (quadrant, ray_cache, edge, N, fract);
        V = {quadrant->p (0, i1), quadrant->p (1, i1), quadrant->p (2, i1)};
        V[axis] += fract * h[axis];
        const double tmp_flux =
          - (tmp_phi[i2] - tmp_phi[i1]) * wha (tmp_eps[i1], tmp_eps[i2], fract)
          * fl_dir[ip] * area_h[axis];

        charge_pol += tmp_flux;

        for (size_t ia = 0; ia < num_atoms; ++ia) {
          const std::array<double,3> &ra = pos_atoms[ia];
          const double dx = ra[0] - V[0];
          const double dy = ra[1] - V[1];
          const double dz = ra[2] - V[2];
          const double r = std::sqrt (dx * dx + dy * dy + dz * dz);
          const double inv_r3 = 1.0 / (r * r * r);
          const double qflux = tmp_flux / r;

          first_int += charge_atoms[ia] * qflux;
          phi_p[ia] += qflux * constant_pol;
          field_px[ia] += dx * inv_r3 * tmp_flux * constant_pol;
          field_py[ia] += dy * inv_r3 * tmp_flux * constant_pol;
          field_pz[ia] += dz * inv_r3 * tmp_flux * constant_pol;
        }
      }
    }

    this->energy_pol = 0.5*constant_pol*first_int;
  } else if ((calc_potential_term == 1 || (calc_potential_term == 2 && k < 1.e-5)) && calc_field_term == 0) {
    phi_p.assign (num_atoms, 0.0);

    for (const int ii : border_quad) {
      quadrant[ii];

      for (int d = 0; d < 3; ++d)
        h[d] = quadrant->p (d, 7) - quadrant->p (d, 0);

      area_h = {h[1]*h[2]/h[0]*0.25, h[0]*h[2]/h[1]*0.25, h[0]*h[1]/h[2]*0.25};
      std::tie (tmp_phi, tmp_eps, edg, fl_dir) = classifyCube_flux (quadrant, tmp_phi, tmp_eps);

      for (int ip = 0; ip < edg.size (); ++ip) {
        const int edge = edg[ip];
        const int axis = edge_axis[edge];
        const int i1 = edge2nodes[2 * edge];
        const int i2 = edge2nodes[2 * edge + 1];
        double fract = 0.0;
        normal_intersection (quadrant, ray_cache, edge, N, fract);
        V = {quadrant->p (0, i1), quadrant->p (1, i1), quadrant->p (2, i1)};
        V[axis] += fract * h[axis];
        const double tmp_flux =
          - (tmp_phi[i2] - tmp_phi[i1]) * wha (tmp_eps[i1], tmp_eps[i2], fract)
          * fl_dir[ip] * area_h[axis];

        charge_pol += tmp_flux;

        for (size_t ia = 0; ia < num_atoms; ++ia) {
          const std::array<double,3> &ra = pos_atoms[ia];
          const double dx = ra[0] - V[0];
          const double dy = ra[1] - V[1];
          const double dz = ra[2] - V[2];
          const double r = std::sqrt (dx * dx + dy * dy + dz * dz);
          const double inv_r3 = 1.0 / (r * r * r);
          const double qflux = tmp_flux / r;

          first_int += charge_atoms[ia] * qflux;
          phi_p[ia] += qflux * constant_pol;

        }
      }
    }

    this->energy_pol = 0.5*constant_pol*first_int;
  }

  auto allocate_potential_fields = [&] (void) {
    phi_p.assign (num_atoms, 0.0);
    phi_i.assign (num_atoms, 0.0);
    field_px.assign (num_atoms, 0.0);
    field_py.assign (num_atoms, 0.0);
    field_pz.assign (num_atoms, 0.0);
    field_ix.assign (num_atoms, 0.0);
    field_iy.assign (num_atoms, 0.0);
    field_iz.assign (num_atoms, 0.0);
  };

  //polarization energy + ionic energy
  if (calc_field_term==2 && k > 1.e-5) {
    allocate_potential_fields();
    int cubeindex = -1;
    std::array<std::array<double,3>,3> vert_triangles, norms_vert;
    std::array<double,3> dist_vert, phi_sup;
    int ntriang = 0;

    for (const int ii : border_quad) {
      quadrant[ii];

      for (int d = 0; d < 3; ++d)
        h[d] = quadrant->p (d, 7) - quadrant->p (d, 0);

      area_h = {h[1]*h[2]/h[0]*0.25, h[0]*h[2]/h[1]*0.25, h[0]*h[1]/h[2]*0.25};
      cubeindex = classifyCube (quadrant, eps_out);
      std::tie (tmp_phi, tmp_eps, edg, fl_dir) = classifyCube_flux (quadrant, tmp_phi, tmp_eps);
      ntriang = getTriangles (cubeindex, triangles);

      // --- flussi
      for (int ip = 0; ip < edg.size (); ++ip) {
        const int edge = edg[ip];
        const int axis = edge_axis[edge];
        const int i1 = edge2nodes[2 * edge];
        const int i2 = edge2nodes[2 * edge + 1];

        double fract = 0.0;
        normal_intersection (quadrant, ray_cache, edge, N, fract);

        V = {quadrant->p (0, i1), quadrant->p (1, i1), quadrant->p (2, i1)};
        V[axis] += fract * h[axis];

        const double tmp_flux =
          - (tmp_phi[i2] - tmp_phi[i1]) * wha (tmp_eps[i1], tmp_eps[i2], fract)
          * fl_dir[ip] * area_h[axis];
        charge_pol += tmp_flux;

        for (size_t ia = 0; ia < num_atoms; ++ia) {
          const std::array<double,3> &ra = pos_atoms[ia];
          const double dx = ra[0] - V[0];
          const double dy = ra[1] - V[1];
          const double dz = ra[2] - V[2];
          const double r = std::sqrt (dx * dx + dy * dy + dz * dz);
          const double inv_r3 = 1.0 / (r * r * r);

          first_int += charge_atoms[ia] * tmp_flux / r;
          phi_p[ia] += tmp_flux / r * constant_pol;
          field_px[ia] += dx * inv_r3 * tmp_flux * constant_pol;
          field_py[ia] += dy * inv_r3 * tmp_flux * constant_pol;
          field_pz[ia] += dz * inv_r3 * tmp_flux * constant_pol;
        }
      }

      // --- triangoli (componente ionica)
      for (int itri = 0; itri < ntriang; ++itri) {
        for (int jj = 0; jj < 3; ++jj) {
          const int edge = triangles[itri][jj];
          const int axis = edge_axis[edge];
          const int i1 = edge2nodes[2 * edge];
          const int i2 = edge2nodes[2 * edge + 1];

          double fract = 0.0;
          normal_intersection (quadrant, ray_cache, edge, N, fract);

          V = {quadrant->p (0, i1), quadrant->p (1, i1), quadrant->p (2, i1)};
          V[axis] += fract * h[axis];

          vert_triangles[jj] = V;
          norms_vert[jj] = N;

          phi_sup[jj] = phi0 (tmp_eps[i1], tmp_eps[i2], tmp_phi[i1], tmp_phi[i2], fract);
        }

        const double area = areaTriangle (vert_triangles);

        for (size_t ia = 0; ia < num_atoms; ++ia) {
          const double qi = charge_atoms[ia];
          const std::array<double,3> &ra = pos_atoms[ia];

          for (int kk = 0; kk < 3; ++kk) {
            dist_vert = {vert_triangles[kk][0] - ra[0],
                         vert_triangles[kk][1] - ra[1],
                         vert_triangles[kk][2] - ra[2]
                        };
            const double r2 = dist_vert[0]*dist_vert[0] + dist_vert[1]*dist_vert[1] + dist_vert[2]*dist_vert[2];
            const double r = std::sqrt (r2);
            const double inv_r3 = 1.0 / (r2 * r);
            const double inv_r5 = inv_r3 / r2;
            const double dot = dist_vert[0]*norms_vert[kk][0] + dist_vert[1]*norms_vert[kk][1] + dist_vert[2]*norms_vert[kk][2];
            const double factor = phi_sup[kk] * inv_4pi * area / 3.0;

            second_int += qi * phi_sup[kk] * dot * inv_r3 * inv_4pi * area / 3.0;
            phi_i[ia] += phi_sup[kk] * dot * inv_r3 * inv_4pi * area / 3.0;

            field_ix[ia] += factor * (-3 * dist_vert[0] * inv_r5 * dot + norms_vert[kk][0] * inv_r3);
            field_iy[ia] += factor * (-3 * dist_vert[1] * inv_r5 * dot + norms_vert[kk][1] * inv_r3);
            field_iz[ia] += factor * (-3 * dist_vert[2] * inv_r5 * dot + norms_vert[kk][2] * inv_r3);
          }
        }
      }
    }

    this->energy_pol = 0.5 * constant_pol * first_int;
    this->energy_react = 0.5 * (second_int - first_int * constant_react);
  } else if ((calc_potential_term==2 && k > 1.e-5) && calc_field_term == 1) {
    phi_p.assign (num_atoms, 0.0);
    phi_i.assign (num_atoms, 0.0);
    field_px.assign (num_atoms, 0.0);
    field_py.assign (num_atoms, 0.0);
    field_pz.assign (num_atoms, 0.0);
    int cubeindex = -1;
    std::array<std::array<double,3>,3> vert_triangles, norms_vert;
    std::array<double,3> dist_vert, phi_sup;
    int ntriang = 0;

    for (const int ii : border_quad) {
      quadrant[ii];

      for (int d = 0; d < 3; ++d)
        h[d] = quadrant->p (d, 7) - quadrant->p (d, 0);

      area_h = {h[1]*h[2]/h[0]*0.25, h[0]*h[2]/h[1]*0.25, h[0]*h[1]/h[2]*0.25};
      cubeindex = classifyCube (quadrant, eps_out);
      std::tie (tmp_phi, tmp_eps, edg, fl_dir) = classifyCube_flux (quadrant, tmp_phi, tmp_eps);
      ntriang = getTriangles (cubeindex, triangles);

      // --- flussi
      for (int ip = 0; ip < edg.size (); ++ip) {
        const int edge = edg[ip];
        const int axis = edge_axis[edge];
        const int i1 = edge2nodes[2 * edge];
        const int i2 = edge2nodes[2 * edge + 1];

        double fract = 0.0;
        normal_intersection (quadrant, ray_cache, edge, N, fract);

        V = {quadrant->p (0, i1), quadrant->p (1, i1), quadrant->p (2, i1)};
        V[axis] += fract * h[axis];

        const double tmp_flux =
          - (tmp_phi[i2] - tmp_phi[i1]) * wha (tmp_eps[i1], tmp_eps[i2], fract)
          * fl_dir[ip] * area_h[axis];
        charge_pol += tmp_flux;

        for (size_t ia = 0; ia < num_atoms; ++ia) {
          const std::array<double,3> &ra = pos_atoms[ia];
          const double dx = ra[0] - V[0];
          const double dy = ra[1] - V[1];
          const double dz = ra[2] - V[2];
          const double r = std::sqrt (dx * dx + dy * dy + dz * dz);
          const double inv_r3 = 1.0 / (r * r * r);
          first_int += charge_atoms[ia] * tmp_flux / r;
          phi_p[ia] += tmp_flux / r * constant_pol;
          field_px[ia] += dx * inv_r3 * tmp_flux * constant_pol;
          field_py[ia] += dy * inv_r3 * tmp_flux * constant_pol;
          field_pz[ia] += dz * inv_r3 * tmp_flux * constant_pol;
        }
      }

      // --- triangoli (componente ionica)
      for (int itri = 0; itri < ntriang; ++itri) {
        for (int jj = 0; jj < 3; ++jj) {
          const int edge = triangles[itri][jj];
          const int axis = edge_axis[edge];
          const int i1 = edge2nodes[2 * edge];
          const int i2 = edge2nodes[2 * edge + 1];

          double fract = 0.0;
          normal_intersection (quadrant, ray_cache, edge, N, fract);

          V = {quadrant->p (0, i1), quadrant->p (1, i1), quadrant->p (2, i1)};
          V[axis] += fract * h[axis];

          vert_triangles[jj] = V;
          norms_vert[jj] = N;

          phi_sup[jj] = phi0 (tmp_eps[i1], tmp_eps[i2], tmp_phi[i1], tmp_phi[i2], fract);
        }

        const double area = areaTriangle (vert_triangles);

        for (size_t ia = 0; ia < num_atoms; ++ia) {
          const double qi = charge_atoms[ia];
          const std::array<double,3> &ra = pos_atoms[ia];

          for (int kk = 0; kk < 3; ++kk) {
            dist_vert = {vert_triangles[kk][0] - ra[0],
                         vert_triangles[kk][1] - ra[1],
                         vert_triangles[kk][2] - ra[2]
                        };
            const double r2 = dist_vert[0]*dist_vert[0] + dist_vert[1]*dist_vert[1] + dist_vert[2]*dist_vert[2];
            const double r = std::sqrt (r2);
            const double inv_r3 = 1.0 / (r2 * r);
            const double inv_r5 = inv_r3 / r2;
            const double dot = dist_vert[0]*norms_vert[kk][0] + dist_vert[1]*norms_vert[kk][1] + dist_vert[2]*norms_vert[kk][2];
            const double factor = phi_sup[kk] * inv_4pi * area / 3.0;
            second_int += qi * phi_sup[kk] * dot * inv_r3 * inv_4pi * area / 3.0;
            phi_i[ia] += phi_sup[kk] * dot * inv_r3 * inv_4pi * area / 3.0;
          }
        }
      }
    }

    this->energy_pol = 0.5 * constant_pol * first_int;
    this->energy_react = 0.5 * (second_int - first_int * constant_react);
  } else if ((calc_potential_term==2 && k > 1.e-5) && calc_field_term == 0) {
    phi_p.assign (num_atoms, 0.0);
    phi_i.assign (num_atoms, 0.0);
    int cubeindex = -1;
    std::array<std::array<double,3>,3> vert_triangles, norms_vert;
    std::array<double,3> dist_vert, phi_sup;
    int ntriang = 0;

    for (const int ii : border_quad) {
      quadrant[ii];

      for (int d = 0; d < 3; ++d)
        h[d] = quadrant->p (d, 7) - quadrant->p (d, 0);

      area_h = {h[1]*h[2]/h[0]*0.25, h[0]*h[2]/h[1]*0.25, h[0]*h[1]/h[2]*0.25};
      cubeindex = classifyCube (quadrant, eps_out);
      std::tie (tmp_phi, tmp_eps, edg, fl_dir) = classifyCube_flux (quadrant, tmp_phi, tmp_eps);
      ntriang = getTriangles (cubeindex, triangles);

      // --- flussi
      for (int ip = 0; ip < edg.size (); ++ip) {
        const int edge = edg[ip];
        const int axis = edge_axis[edge];
        const int i1 = edge2nodes[2 * edge];
        const int i2 = edge2nodes[2 * edge + 1];

        double fract = 0.0;
        normal_intersection (quadrant, ray_cache, edge, N, fract);

        V = {quadrant->p (0, i1), quadrant->p (1, i1), quadrant->p (2, i1)};
        V[axis] += fract * h[axis];

        const double tmp_flux =
          - (tmp_phi[i2] - tmp_phi[i1]) * wha (tmp_eps[i1], tmp_eps[i2], fract)
          * fl_dir[ip] * area_h[axis];
        charge_pol += tmp_flux;

        for (size_t ia = 0; ia < num_atoms; ++ia) {
          const std::array<double,3> &ra = pos_atoms[ia];
          const double dx = ra[0] - V[0];
          const double dy = ra[1] - V[1];
          const double dz = ra[2] - V[2];
          const double r = std::sqrt (dx * dx + dy * dy + dz * dz);
          const double inv_r3 = 1.0 / (r * r * r);
          first_int += charge_atoms[ia] * tmp_flux / r;
          phi_p[ia] += tmp_flux / r * constant_pol;
        }
      }

      // --- triangoli (componente ionica)
      for (int itri = 0; itri < ntriang; ++itri) {
        for (int jj = 0; jj < 3; ++jj) {
          const int edge = triangles[itri][jj];
          const int axis = edge_axis[edge];
          const int i1 = edge2nodes[2 * edge];
          const int i2 = edge2nodes[2 * edge + 1];

          double fract = 0.0;
          normal_intersection (quadrant, ray_cache, edge, N, fract);

          V = {quadrant->p (0, i1), quadrant->p (1, i1), quadrant->p (2, i1)};
          V[axis] += fract * h[axis];

          vert_triangles[jj] = V;
          norms_vert[jj] = N;

          phi_sup[jj] = phi0 (tmp_eps[i1], tmp_eps[i2], tmp_phi[i1], tmp_phi[i2], fract);
        }

        const double area = areaTriangle (vert_triangles);

        for (size_t ia = 0; ia < num_atoms; ++ia) {
          const double qi = charge_atoms[ia];
          const std::array<double,3> &ra = pos_atoms[ia];

          for (int kk = 0; kk < 3; ++kk) {
            dist_vert = {vert_triangles[kk][0] - ra[0],
                         vert_triangles[kk][1] - ra[1],
                         vert_triangles[kk][2] - ra[2]
                        };
            const double r2 = dist_vert[0]*dist_vert[0] + dist_vert[1]*dist_vert[1] + dist_vert[2]*dist_vert[2];
            const double r = std::sqrt (r2);
            const double inv_r3 = 1.0 / (r2 * r);
            const double inv_r5 = inv_r3 / r2;
            const double dot = dist_vert[0]*norms_vert[kk][0] + dist_vert[1]*norms_vert[kk][1] + dist_vert[2]*norms_vert[kk][2];
            const double factor = phi_sup[kk] * inv_4pi * area / 3.0;
            second_int += qi * phi_sup[kk] * dot * inv_r3 * inv_4pi * area / 3.0;
            phi_i[ia] += phi_sup[kk] * dot * inv_r3 * inv_4pi * area / 3.0;
          }
        }
      }
    }

    this->energy_pol = 0.5 * constant_pol * first_int;
    this->energy_react = 0.5 * (second_int - first_int * constant_react);
  }


  auto reduce_double = [&] (double &x) {
    MPI_Reduce (rank == 0 ? MPI_IN_PLACE : &x, &x, 1, MPI_DOUBLE, MPI_SUM, 0, mpicomm);
  };

  auto reduce_vec = [&] (std::vector<double> &v) {
    MPI_Reduce (rank == 0 ? MPI_IN_PLACE : v.data(),
                rank == 0 ? v.data() : nullptr,
                (int)v.size(), MPI_DOUBLE, MPI_SUM, 0, mpicomm);
  };

  reduce_double (charge_pol);
  reduce_double (energy_pol);
  reduce_double (energy_react);
  reduce_vec (phi_p);
  reduce_vec (phi_i);
  reduce_vec (field_px);
  reduce_vec (field_py);
  reduce_vec (field_pz);
  reduce_vec (field_ix);
  reduce_vec (field_iy);
  reduce_vec (field_iz);

  // Print the result
  if (rank == 0) {
    constexpr int label_width = 50;
    constexpr int precision = 16;


    if (calc_potential_term == 1 || calc_field_term == 1 || calc_potential_term == 2 || calc_field_term == 2) {
      std::cout << std::left << std::setw (label_width) << "  Polarization energy [kT]:"
                << std::setprecision (precision) << energy_pol << "\n";
    }

    if (calc_potential_term == 2 || calc_field_term == 2) {
      std::cout << std::left << std::setw (label_width) << "  Direct ionic energy [kT]:"
                << std::setprecision (precision) << energy_react << "\n";
    }


    std::cout << std::left << std::setw (label_width) << "  Coulombic energy [kT]:"
              << std::setprecision (precision) << coul_energy << "\n";


    std::cout << std::left << std::setw (label_width) << "  Sum of electrostatic energy contributions [kT]:"
              << std::setprecision (precision)
              << (energy_pol + energy_react + coul_energy) << "\n";

    std::cout << "===========================================================\n";


    // ============================
    // Scrittura su file risultati
    // ============================
    std::ofstream fout ("pot_field.dat");
    std::cout << "Writing potentials and fields to pot_field.dat\n";
    fout << "# index    x    y    z    ";

    if (calc_field_term==2 && k > 1.e-5) {
      fout <<"phi_c    phi_p    phi_i    Ex_c    Ey_c    Ez_c   Ex_p    Ey_p    Ez_p   Ex_i    Ey_i    Ez_i\n";

      for (size_t i = 0; i < num_atoms; ++i) {
        phi_i[i] -= phi_p[i] / constant_pol * constant_react;
        field_ix[i] -= field_px[i] / constant_pol * constant_react;
        field_iy[i] -= field_py[i] / constant_pol * constant_react;
        field_iz[i] -= field_pz[i] / constant_pol * constant_react;
        fout << std::setw (5) << i + 1 << "  "
             << std::setw (8) << pos_atoms[i][0] << "  "
             << std::setw (8) << pos_atoms[i][1] << "  "
             << std::setw (8) << pos_atoms[i][2] << "  "
             << std::setw (8) << phi_c[i] << "  "
             << std::setw (8) << phi_p[i] << "  "
             << std::setw (8) << phi_i[i] << "  ";
        fout << std::setw (8) << field_cx[i] << "  ";
        fout << std::setw (8) << field_cy[i] << "  ";
        fout << std::setw (8) << field_cz[i] << "  ";
        fout << std::setw (8) << field_px[i] << "  ";
        fout << std::setw (8) << field_py[i] << "  ";
        fout << std::setw (8) << field_pz[i] << "  ";
        fout << std::setw (8) << field_ix[i] << "  ";
        fout << std::setw (8) << field_iy[i] << "  ";
        fout << std::setw (8) << field_iz[i] << "  ";
        fout << "\n";
      }
    } else if ((calc_potential_term==2 && k > 1.e-5) && calc_field_term == 0) {
      fout <<"phi_c    phi_p    phi_i    Ex_c    Ey_c    Ez_c \n";

      for (size_t i = 0; i < num_atoms; ++i) {
        phi_i[i] -= phi_p[i] / constant_pol * constant_react;

        fout << std::setw (5) << i + 1 << "  "
             << std::setw (8) << pos_atoms[i][0] << "  "
             << std::setw (8) << pos_atoms[i][1] << "  "
             << std::setw (8) << pos_atoms[i][2] << "  "
             << std::setw (8) << phi_c[i] << "  "
             << std::setw (8) << phi_p[i] << "  "
             << std::setw (8) << phi_i[i] << "  ";
        fout << std::setw (8) << field_cx[i] << "  ";
        fout << std::setw (8) << field_cy[i] << "  ";
        fout << std::setw (8) << field_cz[i] << "  ";
        fout << "\n";
      }
    } else if ((calc_potential_term==2 && k > 1.e-5) && calc_field_term == 1) {
      fout <<"phi_c    phi_p    phi_i    Ex_c    Ey_c    Ez_c   Ex_p    Ey_p    Ez_p\n";

      for (size_t i = 0; i < num_atoms; ++i) {
        phi_i[i] -= phi_p[i] / constant_pol * constant_react;

        fout << std::setw (5) << i + 1 << "  "
             << std::setw (8) << pos_atoms[i][0] << "  "
             << std::setw (8) << pos_atoms[i][1] << "  "
             << std::setw (8) << pos_atoms[i][2] << "  "
             << std::setw (8) << phi_c[i] << "  "
             << std::setw (8) << phi_p[i] << "  "
             << std::setw (8) << phi_i[i] << "  ";
        fout << std::setw (8) << field_cx[i] << "  ";
        fout << std::setw (8) << field_cy[i] << "  ";
        fout << std::setw (8) << field_cz[i] << "  ";
        fout << std::setw (8) << field_px[i] << "  ";
        fout << std::setw (8) << field_py[i] << "  ";
        fout << std::setw (8) << field_pz[i] << "  ";

        fout << "\n";
      }
    } else if ((calc_potential_term == 1 || (calc_potential_term == 2 && k < 1.e-5)) && calc_field_term == 0) {
      fout <<"phi_c    phi_p    Ex_c    Ey_c    Ez_c\n";

      for (size_t i = 0; i < num_atoms; ++i) {
        fout << std::setw (5) << i + 1 << "  "
             << std::setw (8) << pos_atoms[i][0] << "  "
             << std::setw (8) << pos_atoms[i][1] << "  "
             << std::setw (8) << pos_atoms[i][2] << "  "
             << std::setw (8) << phi_c[i] << "  "
             << std::setw (8) << phi_p[i] << "  ";
        fout << std::setw (8) << field_cx[i] << "  ";
        fout << std::setw (8) << field_cy[i] << "  ";
        fout << std::setw (8) << field_cz[i] << "  ";
        fout << "\n";
      }
    } else if ((calc_field_term==1 || (calc_field_term == 2 && k < 1.e-5)) && calc_potential_term < 2) {
      fout <<"phi_c    phi_p    Ex_c    Ey_c    Ez_c   Ex_p    Ey_p    Ez_p\n";

      for (size_t i = 0; i < num_atoms; ++i) {
        fout << std::setw (5) << i + 1 << "  "
             << std::setw (8) << pos_atoms[i][0] << "  "
             << std::setw (8) << pos_atoms[i][1] << "  "
             << std::setw (8) << pos_atoms[i][2] << "  "
             << std::setw (8) << phi_c[i] << "  "
             << std::setw (8) << phi_p[i] << "  ";
        fout << std::setw (8) << field_cx[i] << "  ";
        fout << std::setw (8) << field_cy[i] << "  ";
        fout << std::setw (8) << field_cz[i] << "  ";
        fout << std::setw (8) << field_px[i] << "  ";
        fout << std::setw (8) << field_py[i] << "  ";
        fout << std::setw (8) << field_pz[i] << "  ";

        fout << "\n";
      }
    }


    fout.close();
    std::cout << "Atom potentials and fields written to 'pot_field.dat'\n";
  }
}
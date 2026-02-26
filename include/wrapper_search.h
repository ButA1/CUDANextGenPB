/*
 *  Copyright (C) 2024 Vincenzo Di Florio
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

#ifndef WRAPPER_SEARCH_H
#define WRAPPER_SEARCH_H

#include "p8est.h"

struct poisson_boltzmann;  // forward declaration

// l’unico modo: una variabile globale
extern poisson_boltzmann* pb_global_wrapper;

int cerca_atomo_wrapper(
    p8est_t* p4est,
    p4est_topidx_t which_tree,
    p8est_quadrant_t* quadrant,
    p4est_locidx_t local_num,
    void* point   // p4est gestisce sc_array_t
);

#endif
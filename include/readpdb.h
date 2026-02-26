#ifndef READPDB_H
#define READPDB_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include <cctype>
#include <cstdlib>

#include <nanoshaper.h>



void read_pdb (const std::string& filename, std::vector<NS::Atom>& atoms);
void load_radii (const std::string& filename, std::vector<NS::Atom>& atoms);
void load_charges (const std::string& filename, std::vector<NS::Atom>& atoms);
void write_pqr (const std::string& filename, const std::vector<NS::Atom>& atoms);

int parse_line_to_atom_radius (const std::string& line, NS::Atom& atom);
int parse_line_to_atom_charge (const std::string& line, NS::Atom& atom);

#endif
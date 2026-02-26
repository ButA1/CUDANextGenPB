
#include <cstdio>
#include <algorithm>
#include "readpdb.h"


void read_pdb (const std::string& filename, std::vector<NS::Atom>& atoms)
{
  std::ifstream infile (filename);

  if (!infile) {
    std::cerr << "Error opening PDB file: " << filename << "\n";
    exit (1);
  }

  std::string line;
  int count = 0;

  while (std::getline (infile, line)) {
    if (line.compare (0, 4, "ATOM") != 0 && line.compare (0, 6, "HETATM") != 0)
      continue;

    NS::Atom atom;
    // atom.ai.name    = line.substr(12, 4);
    std::string raw_name = line.substr (12, 4);
    raw_name.erase (std::remove_if (raw_name.begin(), raw_name.end(), ::isspace), raw_name.end());
    atom.ai.name = raw_name;
    atom.ai.resName = line.substr (17, 3);
    atom.ai.chain = line[21];
    atom.ai.resNum = std::stoi (line.substr (22, 4));
    atom.pos[0] = std::stof (line.substr (30, 8));
    atom.pos[1] = std::stof (line.substr (38, 8));
    atom.pos[2] = std::stof (line.substr (46, 8));
    atom.charge = 0.0f;
    atom.radius = 0.0f;
    atoms.push_back (atom);
  }

  std::cout << "Total atoms read: " << atoms.size() << "\n";
}

int parse_line_to_atom_radius (const std::string& line, NS::Atom& atom)
{
  std::istringstream iss (line);
  std::string tokens[6];
  int count = 0;

  while (iss >> tokens[count] && count < 6) {
    count++;
  }

  // Inizializza campi non presenti
  // atom.ai.resNum = 1;
  // atom.ai.chain = 'X';
  // atom.charge = 0.0f;
  // atom.radius = 0.0f;
  // atom.ai.resName = "XXX";
  // atom.ai.name = "X";

  if (count == 2) {
    // atom_name  radius
    atom.ai.name = tokens[0];
    atom.radius = std::stof (tokens[1]);
    return 2;
  } else if (count == 3) {
    // atom_name  res_name  radius
    atom.ai.name = tokens[0];
    atom.ai.resName = tokens[1];
    atom.radius = std::stof (tokens[2]);
    return 3;
  } else if (count == 4) {
    // atom_name  res_name  chain_id  radius
    atom.ai.name = tokens[0];
    atom.ai.resName = tokens[1];
    atom.ai.chain = tokens[2][0];
    atom.radius = std::stof (tokens[3]);
    return 4;
  } else if (count == 5) {
    // atom_name  res_name  res_num  chain_id  radius
    atom.ai.name = tokens[0];
    atom.ai.resName = tokens[1];
    atom.ai.resNum = std::stoi (tokens[2]);
    atom.ai.chain = tokens[3][0];
    atom.radius = std::stof (tokens[4]);
    return 5;
  }

  return 0;
}

int parse_line_to_atom_charge (const std::string& line, NS::Atom& atom)
{
  std::istringstream iss (line);
  std::string tokens[6];
  int count = 0;

  while (iss >> tokens[count] && count < 6) {
    count++;
  }

  // Inizializza campi non presenti
  // atom.ai.resNum = 1;
  // atom.ai.chain = 'X';
  // atom.charge = 0.0f;
  // atom.radius = 0.0f;
  // atom.ai.resName = "XXX";
  // atom.ai.name = "X";

  if (count == 2) {
    // atom_name  charge
    atom.ai.name = tokens[0];
    atom.charge = std::stof (tokens[1]);
    return 2;
  } else if (count == 3) {
    // atom_name  res_name  charge
    atom.ai.name = tokens[0];
    atom.ai.resName = tokens[1];
    atom.charge = std::stof (tokens[2]);
    return 3;
  } else if (count == 4) {
    // atom_name  res_name  chain_id  charge
    atom.ai.name = tokens[0];
    atom.ai.resName = tokens[1];
    atom.ai.chain = tokens[2][0];
    atom.charge = std::stof (tokens[3]);
    return 4;
  } else if (count == 5) {
    // atom_name  res_name  res_num  chain_id  charge
    atom.ai.name = tokens[0];
    atom.ai.resName = tokens[1];
    atom.ai.resNum = std::stoi (tokens[2]);
    atom.ai.chain = tokens[3][0];
    atom.charge = std::stof (tokens[4]);
    return 5;
  }

  return 0;
}


void load_radii (const std::string& filename, std::vector<NS::Atom>& atoms)
{
  std::ifstream infile (filename);

  if (!infile) {
    std::cerr << "Failed to open radius file: " << filename << "\n";
    std::exit (1);
  }

  std::string line;

  // Leggi le righe dati
  while (std::getline (infile, line)) {
    // Rimuovi commenti
    size_t excl = line.find ('!');

    if (excl != std::string::npos)
      line = line.substr (0, excl);

    // Rimuovi spazi iniziali e righe vuote
    line.erase (0, line.find_first_not_of (" \t\r\n"));

    if (line.empty()) continue;

    NS::Atom parsed;
    int matched = parse_line_to_atom_radius (line, parsed);

    if (matched >= 2) {
      for (auto& atom : atoms) {
        if (atom.ai.name.substr (0, 4) == parsed.ai.name.substr (0, 4) &&
            (matched < 3 || atom.ai.resName.substr (0, 3) == parsed.ai.resName.substr (0, 3)) &&
            (matched < 5 || atom.ai.resNum == parsed.ai.resNum) &&
            (matched < 4 || atom.ai.chain == parsed.ai.chain)) {
          atom.radius = parsed.radius;
        }
      }
    }
  }

  infile.close ();
}

void load_charges (const std::string& filename, std::vector<NS::Atom>& atoms)
{
  std::ifstream infile (filename);

  if (!infile) {
    std::cerr << "Failed to open charge file: " << filename << "\n";
    std::exit (1);
  }

  std::string line;

  // Leggi le righe dati
  while (std::getline (infile, line)) {
    // Rimuovi commenti
    size_t excl = line.find ('!');

    if (excl != std::string::npos)
      line = line.substr (0, excl);

    // Rimuovi spazi iniziali e righe vuote
    line.erase (0, line.find_first_not_of (" \t\r\n"));

    if (line.empty()) continue;

    NS::Atom parsed;
    int matched = parse_line_to_atom_charge (line, parsed);

    if (matched >= 2) {
      for (auto& atom : atoms) {
        if (atom.ai.name.substr (0, 4) == parsed.ai.name.substr (0, 4) &&
            (matched < 3 || atom.ai.resName.substr (0, 3) == parsed.ai.resName.substr (0, 3)) &&
            (matched < 5 || atom.ai.resNum == parsed.ai.resNum) &&
            (matched < 4 || atom.ai.chain == parsed.ai.chain)) {
          atom.charge = parsed.charge;
        }
      }
    }
  }

  infile.close ();
}

void write_pqr (const std::string& filename, const std::vector<NS::Atom>& atoms)
{
  FILE* fp = std::fopen (filename.c_str(), "w");

  if (!fp) {
    std::perror ("Failed to open file for writing");
    std::exit (1);
  }

  for (std::size_t i = 0; i < atoms.size(); ++i) {
    const NS::Atom& a = atoms[i];
    std::fprintf (fp,
                  "ATOM  %5zu %-4s %-3s %c%4d    %8.3f%8.3f%8.3f %7.4f %7.4f\n",
                  i + 1,
                  a.ai.name.c_str(),
                  a.ai.resName.c_str(),
                  a.ai.chain.empty() ? ' ' : a.ai.chain[0], // Usa solo il primo carattere
                  a.ai.resNum,
                  a.pos[0],
                  a.pos[1],
                  a.pos[2],
                  a.charge,
                  a.radius
                 );
    // std::fprintf(fp,
    // "ATOM  %5zu %-4s %-3s %4d    %8.3f%8.3f%8.3f %7.4f %7.4f\n",
    // i + 1,
    // a.ai.name.c_str(),
    // a.ai.resName.c_str(),
    // a.ai.resNum,
    // a.pos[0],
    // a.pos[1],
    // a.pos[2],
    // a.charge,
    // a.radius
    // );
  }

  std::fclose (fp);
}

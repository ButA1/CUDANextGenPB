/*
 *  Copyright (C) 2024-2025 Vincenzo Di Florio
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

#include "vtk_class.h"

void
VTKWriter:: writeFieldVtuBinary (tmesh_3d& tmsh,
                                 const distributed_vector& field,
                                 std::string &fieldname_)
{
  preparingData (field, tmsh);
  setFieldName (fieldname_);
  initializeFile();
  writePieceHeader();
  writeGrid();
  writeDataPoints();
  finalizeFile();
}

void
VTKWriter:: writeFieldVtuBinary_local (tmesh_3d& tmsh,
                                       const distributed_vector& field,
                                       std::string &fieldname_, double* l_cr, double* r_cr)
{
  preparingData_local (field, tmsh, l_cr, r_cr);
  setFieldName (fieldname_);
  initializeFile();
  writePieceHeader();
  writeGrid();
  writeDataPoints();
  finalizeFile();
}

void
VTKWriter::createPvtuFile (std::vector<std::string> &fieldNames, std::vector<std::string> &baseNames, int numRanks)
{
  // Nome del file pvtu
  if (fieldNames.size() != baseNames.size()) {
    throw std::runtime_error ("Error: different number of field and names! ");
  }

  for (size_t i = 0; i < baseNames.size(); ++i) {

    std::string pvtuFilename = "total_"+baseNames[i]+".pvtu";

    // Apri il file
    std::ofstream pvtuFile (pvtuFilename);

    if (!pvtuFile.is_open()) {
      throw std::runtime_error ("Error: impossible to create the file " + pvtuFilename);
    }

    // Scrivi l'intestazione
    pvtuFile << "<?xml version=\"1.0\"?>\n";
    pvtuFile << "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    pvtuFile << "  <PUnstructuredGrid>\n";

    // Scrivi le sezioni PPoints, PCells e PPointData
    pvtuFile << "    <PPoints>\n";
    pvtuFile << "      <PDataArray type=\"Float64\" NumberOfComponents=\"3\" Name=\"Array\"/>\n";
    pvtuFile << "    </PPoints>\n";

    pvtuFile << "    <PCells>\n";
    pvtuFile << "      <PDataArray type=\"Int32\" Name=\"connectivity\"/>\n";
    pvtuFile << "      <PDataArray type=\"Int32\" Name=\"offsets\"/>\n";
    pvtuFile << "      <PDataArray type=\"Int32\" Name=\"types\"/>\n";
    pvtuFile << "    </PCells>\n";

    // Aggiungi i dati per ogni campo
    pvtuFile << "    <PPointData>\n";
    pvtuFile << "      <PDataArray type=\"Float64\" Name=\"" << fieldNames[i] << "\"/>\n";
    pvtuFile << "    </PPointData>\n";

    // Aggiungi i riferimenti ai file .vtu per ogni campo e rank
    for (int rank = 0; rank < numRanks; ++rank)
      pvtuFile << "    <Piece Source=\""
               << baseNames[i] << "_" << std::setfill ('0') << std::setw (4) << rank << ".vtu\"/>\n";

    // Chiudi le sezioni
    pvtuFile << "  </PUnstructuredGrid>\n";
    pvtuFile << "</VTKFile>\n";

    // Chiudi il file
    pvtuFile.close();
    std::cout << "File .pvtu create: " << pvtuFilename << std::endl;
  }
}

void
VTKWriter::preparingData (const distributed_vector& field, tmesh_3d& tmsh)
{
  int real_node = tmsh.num_owned_nodes();
  nelems = tmsh.num_local_quadrants ();

  p.assign (3 * real_node,0);
  f_loc.assign (real_node,0);
  t.assign (8*nelems, 0);

  int ij = 0;
  int triang = 0;

  for (auto quadrant = tmsh.begin_quadrant_sweep ();
       quadrant != tmsh.end_quadrant_sweep ();
       ++quadrant) {
    ij = 0;

    for (int ii = 0; ii < 8; ++ii) {
      triang = quadrant->t (ii);

      if (! quadrant->is_hanging (ii))
        if (triang < real_node) {
          for (int jj = 0; jj < 3; ++jj)
            p[3 * triang + jj] = quadrant->p (jj, ii);

          f_loc[triang] = field[quadrant->gt (ii)];
          t[8 * quadrant->get_forest_quad_idx () + (ij++)] = triang;
        } else {
          for (int jj = 0; jj < 3; ++jj)
            p.push_back (quadrant->p (jj, ii));

          f_loc.push_back (field[quadrant->gt (ii)]);
          t[8 * quadrant->get_forest_quad_idx () + (ij++)] =
            f_loc.size () - 1;
        } else {
        for (int jj = 0; jj < 3; ++jj)
          p.push_back (quadrant->p (jj, ii));

        int pp;
        double fbuff = 0;

        for (pp = 0; pp < quadrant->num_parents (ii); ++pp)
          fbuff += field [quadrant->gparent (pp, ii)];

        f_loc.push_back (fbuff / pp);
        t[8 * quadrant->get_forest_quad_idx () + (ij++)] =
          f_loc.size () - 1;
      }
    }
  }

  nnodes = f_loc.size ();
}

void VTKWriter::preparingData_local (const distributed_vector& field,
                                     tmesh_3d& tmsh,
                                     double* l_cr,
                                     double* r_cr)
{
  nelems = 0;
  nnodes = 0;

  // Array di indici per i nodi univoci, inizializzati a -1
  std::vector<int> global_to_local (tmsh.num_owned_nodes(), -1);

  for (auto quadrant = tmsh.begin_quadrant_sweep();
       quadrant != tmsh.end_quadrant_sweep();
       ++quadrant) {

    // Coordinate dei nodi estremi del quadrante
    double x1 = quadrant->p (0, 0), y1 = quadrant->p (1, 0), z1 = quadrant->p (2, 0);
    double x2 = quadrant->p (0, 7), y2 = quadrant->p (1, 7), z2 = quadrant->p (2, 7);

    // Controllo se il quadrante è nella regione desiderata
    if ((x1 > l_cr[0]) && (x2 < r_cr[0]) &&
        (y1 > l_cr[1]) && (y2 < r_cr[1]) &&
        (z1 > l_cr[2]) && (z2 < r_cr[2])) {
      // Itera sui nodi del quadrante
      for (int ii = 0; ii < 8; ++ii) {
        int global_index = quadrant->gt (ii);

        // Verifica se il nodo è già stato aggiunto
        if (global_to_local[global_index] == -1) {
          if (rank==1) {
            std::cout << f_loc.size()<<" " << global_to_local[global_index] << "  " << global_index <<std::endl;
          }

          // Nodo nuovo: aggiungilo a p e aggiorna il mapping
          for (int jj = 0; jj < 3; ++jj) {
            p.push_back (quadrant->p (jj, ii));
          }

          // Calcola il valore del campo per il nodo
          double field_value = 0.0;

          if (!quadrant->is_hanging (ii)) {
            field_value = field[global_index];
          } else {
            // Media dei valori dei nodi parent
            int num_parents = quadrant->num_parents (ii);

            for (int pp = 0; pp < num_parents; ++pp) {
              field_value += field[quadrant->gparent (pp, ii)];
            }

            field_value /= num_parents;
          }

          f_loc.push_back (field_value);

          // Registra il nodo nella lista dei nodi univoci
          global_to_local[global_index] = nnodes++;
        }

        // Aggiungi l'indice del nodo al vettore di connettività
        t.push_back (global_to_local[global_index]);
      }

      // Incrementa il numero di elementi
      ++nelems;
    }
  }

  // Calcolo del numero di nodi
  nnodes = f_loc.size();
}

void
VTKWriter::setBaseName (std::string basename_, int rank_)
{
  // Se il basename è vuoto, usa 'default'
  basename = basename_.empty() ? "default" : basename_;
  rank = rank_;

  // Genera il nuovo nome del file
  snprintf (filename, sizeof (filename), "%s_%4.4d.vtu", basename.c_str(), rank);

  // Azzerare l'offset quando viene cambiato il nome del file
  offsetCounter = 0;
  // Azzera il datacontainer e gli alti ogetti
  ByteArray().swap (dataContainer);
  int nnodes;
  int nelems;
  std::vector<double>().swap (p);
  std::vector<double>().swap (f_loc);
  std::vector<int>().swap (t);
}

void
VTKWriter::setFieldName (std::string &fieldname_)
{
  if (fieldname_.empty()) {
    fieldname = "field";
    std::cerr << "Warning: fieldname is empty. Setting basename to 'fied'.\n";
  } else
    fieldname = fieldname_;
}


void
VTKWriter::initializeFile()
{
  // Apre il file in modalità sovrascrittura
  file.open (filename, std::ios::binary | std::ios::trunc);

  if (!file.is_open()) {
    throw std::runtime_error ("Errore: impossibile aprire il file " + std::string (filename));
  }

  // Scrive l'intestazione del file
  file << "<?xml version=\"1.0\"?>\n";
  file << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
  file << "<UnstructuredGrid>\n";
}


void
VTKWriter::writePieceHeader()
{
  file << "<Piece NumberOfPoints=\"" << nnodes << "\" NumberOfCells=\"" << nelems << "\">\n";
}


template <typename T>
ByteArray serializeData (const std::vector<T>& data)
{
  ByteArray rawData;
  rawData.reserve (data.size() * sizeof (T));

  for (const T& value : data) {
    const char* rawBytes = reinterpret_cast<const char*> (&value);
    rawData.insert (rawData.end(), rawBytes, rawBytes + sizeof (T));
  }

  return rawData;
}

template <typename T>
ByteArray serializeDataWithHeader (const std::vector<T>& data)
{
  ByteArray rawData;
  // Intestazione con la dimensione del blocco (in byte)
  int32_t blockSize = static_cast<int32_t> (data.size() * sizeof (T));
  const char* headerBytes = reinterpret_cast<const char*> (&blockSize);
  rawData.insert (rawData.end(), headerBytes, headerBytes + sizeof (int32_t));

  // Aggiungere i dati veri e propri
  const char* dataBytes = reinterpret_cast<const char*> (data.data());
  rawData.insert (rawData.end(), dataBytes, dataBytes + blockSize);

  return rawData;
}

void
VTKWriter::appendData (const ByteArray& rawData)
{
  // Aggiunge i dati binari al contenitore principale
  dataContainer.insert (dataContainer.end(), rawData.begin(), rawData.end());

  // Aggiorna l'offset (la dimensione totale dei dati scritti finora)
  offsetCounter += rawData.size();
  // size_t alignTo = 8; // Multiplo di 8 per l'allineamento
  // offsetCounter = (offsetCounter + alignTo - 1) & ~(alignTo - 1);
}

void
VTKWriter::writeDataArray (const std::string& type, const std::string& name, int numComponents,
                           const std::vector<double>& data, int numEntries)
{
  file << "<DataArray type=\"" << type << "\" Name=\"" << name
       << "\" NumberOfComponents=\"" << numComponents
       << "\" format=\"appended\" offset=\"" << offsetCounter << "\">\n";

  // ByteArray rawData = serializeData(data);
  ByteArray rawData = serializeDataWithHeader (data);
  appendData (rawData);

  file << "</DataArray>\n";
}

void VTKWriter::writeConnectivity()
{
  file << "<DataArray type=\"Int32\" Name=\"connectivity\" format=\"appended\" offset=\"" << offsetCounter << "\">\n";
  file << "</DataArray>\n";

  // ByteArray rawData = serializeData(t); // Serializza i dati come Int32
  ByteArray rawData = serializeDataWithHeader (t); // Serializza i dati come Int32
  appendData (rawData); // Aggiunge al contenitore binario
}

void VTKWriter::writeOffsets()
{
  std::vector<int> offsets (nelems);
  int nodesPerCell = 8; // Supponendo celle cubiche 3D

  for (int i = 0; i < nelems; ++i) {
    offsets[i] = (i + 1) * nodesPerCell;
  }

  file << "<DataArray type=\"Int32\" Name=\"offsets\" format=\"appended\" offset=\"" << offsetCounter << "\">\n";
  file << "</DataArray>\n";

  // ByteArray rawData = serializeData(offsets);
  ByteArray rawData = serializeDataWithHeader (offsets);
  appendData (rawData);
}

void VTKWriter::writeCellTypes()
{
  int cellType = 11; // Per celle cubiche 3D
  std::vector<int> types (nelems, cellType);

  file << "<DataArray type=\"Int32\" Name=\"types\" format=\"appended\" offset=\"" << offsetCounter << "\">\n";
  file << "</DataArray>\n";

  // ByteArray rawData = serializeData(types);
  ByteArray rawData = serializeDataWithHeader (types);
  appendData (rawData);
}

void
VTKWriter::writeGrid()
{
  file << "<Points>\n";
  writeDataArray ("Float64", "Array", 3, p, nnodes); // Scrive i nodi della mesh
  file << "</Points>\n";

  file << "<Cells>\n";
  writeConnectivity(); // Scrive la connettività
  writeOffsets(); // Scrive gli offset
  writeCellTypes(); // Scrive i tipi delle celle
  file << "</Cells>\n";
}

void
VTKWriter::writeDataPoints()
{

  file << "<PointData>\n";
  writeDataArray ("Float64", fieldname, 1, f_loc, nnodes);

  file << "</PointData>\n";

}



void
VTKWriter::finalizeFile()
{
  file << "</Piece>\n";
  file << "</UnstructuredGrid>\n";
  file << " <AppendedData encoding=\"raw\">\n";
  file << "_";
  file.write (reinterpret_cast<const char*> (dataContainer.data()), dataContainer.size());
  file << " </AppendedData>\n";
  file << "</VTKFile>\n";
  file.close();
}

bool
VTKWriter::fileExists (std::string& name)
{
  std::ifstream f (name.c_str());
  return f.good();
}

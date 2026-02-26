
import argparse
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

import glob

def merge_vtu_files(file_pattern):
    """
    Merges multiple .vtu files into a single vtkUnstructuredGrid.

    Parameters:
        file_pattern (str): Glob pattern to match multiple .vtu files (e.g., "potential_map_*.vtu").

    Returns:
        vtkUnstructuredGrid: Merged VTK unstructured grid.
    """
    files = sorted(glob.glob(file_pattern))  # Get all matching files
    if not files:
        raise FileNotFoundError(f"No .vtu files found matching pattern: {file_pattern}")

    append_filter = vtk.vtkAppendFilter()  # To merge unstructured grids

    for file in files:
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(file)
        reader.Update()
        append_filter.AddInputData(reader.GetOutput())

    append_filter.Update()
    return append_filter.GetOutput()

def read_pqr(file_path):
    """
    Legge un file PQR e restituisce il bounding box (size e centro) includendo i raggi atomici.
    Gestisce file con o senza chain ID.
    
    Parametri:
        file_path (str): percorso al file .pqr
    
    Restituisce:
        size (list di float): dimensioni del bounding box [dx, dy, dz]
        center (list di float): centro del bounding box [cx, cy, cz]
    """
    min_coords = np.full(3, np.inf)
    max_coords = np.full(3, -np.inf)

    with open(file_path, 'r') as f:
        for lineno, line in enumerate(f, 1):
            if not line.startswith(("ATOM", "HETATM")):
                continue

            parts = line.strip().split()
            try:
                # Prende le ultime 5 colonne: x, y, z, charge, radius
                x, y, z = map(float, parts[-5:-2])
                radius = float(parts[-1])
            except (IndexError, ValueError) as e:
                print(f"Warning: Riga malformata {lineno} in {file_path}: {e}")
                continue

            coords = np.array([x, y, z])
            min_coords = np.minimum(min_coords, coords - radius)
            max_coords = np.maximum(max_coords, coords + radius)

    size = max_coords - min_coords
    center = (max_coords + min_coords) * 0.5

    return size.tolist(), center.tolist()

def write_potential_cube(filename, scalar_field, scale, origin, dimensions):
    """
    Write a 3D potential map in Gaussian cube format.
    
    Parameters:
        filename (str): Output file name.
        scalar_field (numpy.ndarray): 3D array of potential values.
        scale (float): Conversion factor to compute the step size (1/scale).
        oldmid (list or numpy.ndarray): Midpoint coordinates in Angstrom [x, y, z].
    """
    coeff = 0.5291772108  # Conversion from Bohr to Angstrom
    stepsize = 1.0 / scale / coeff
    igrid = scalar_field.shape[0]  # Assuming scalar_field is a cube (NxNxN grid)

    # Convert origin from Angstrom to Bohr
    origin_in_bohr = [coord / coeff for coord in origin]
    
    with open(filename, 'w') as cube_file:
        # Write the header
        cube_file.write('qdiffxs4 with an improved surfacing routine\n')
        cube_file.write('Gaussian cube format phimap\n')
        
        # Write origin and grid size in Bohr
        cube_file.write(f" 1   {origin_in_bohr[0]:12.6f}   {origin_in_bohr[1]:12.6f}   {origin_in_bohr[2]:12.6f}\n")
        cube_file.write(f" {dimensions[0]}   {stepsize:.6f}   0.000000   0.000000\n")
        cube_file.write(f" {dimensions[1]}   0.000000   {stepsize:.6f}   0.000000\n")
        cube_file.write(f" {dimensions[2]}   0.000000   0.000000   {stepsize:.6f}\n")
        
        # Placeholder for atomic data
        cube_file.write(" 1   0.000000   0.000000   0.000000   0.000000\n")
        
        # Write the potential map
        nx, ny, nz = dimensions
        count = 0
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    value = scalar_field[ix, iy, iz]
                    cube_file.write(f"{value: .5E} ")
                    count += 1
                    if count % 6 == 0:  # 6 valori per riga
                        cube_file.write("\n")
                if count % 6 != 0:
                    cube_file.write("\n")
                count = 0   
        if count % 6 != 0:  # Se l'ultimo blocco non riempie la riga, aggiunge un ritorno a capo
            cube_file.write("\n")




# Parse command-line arguments
parser = argparse.ArgumentParser(description="Interpolate unstructured VTK grid onto a cube dataset.")
parser.add_argument('pqrfile', type=str, help="Input pqr file")
parser.add_argument('--scale', type=float, default=2, help="grid scale factor (default: 2.0)")
args = parser.parse_args()
name_pqr = args.pqrfile


# Read and merge multiple .vtu files
print("Reading and merging .vtu files")
unstructured_grid = merge_vtu_files("potential_map_*.vtu")
#print(f"Merged {unstructured_grid.GetNumberOfCells()} cells into a single grid")

# Get the bounds of the protein
size, center = read_pqr(name_pqr) 
max_size = max(size)
max_size = max_size / 0.9
origin = [center[0] - max_size * 0.5,
          center[1] - max_size * 0.5,
          center[2] - max_size * 0.5]

# Define the grid dimensions (adjust the resolution)
scale = args.scale
nx = round(max_size * scale) + 1
ny = round(max_size * scale) + 1
nz = round(max_size * scale) + 1
#print(f"Grid dimensions: {nx, ny, nz}")

# Create an ImageData object for STRUCTURED_POINTS
structured_points = vtk.vtkImageData()
structured_points.SetDimensions(nx, ny, nz)
structured_points.SetOrigin(origin[0], origin[1], origin[2])
structured_points.SetSpacing(1 / scale, 1 / scale, 1 / scale)
#print(f"Grid origin: {origin}")

# Interpolation: use vtkProbeFilter to interpolate unstructured data onto structured points
probe = vtk.vtkProbeFilter()
probe.SetInputData(structured_points)
probe.SetSourceData(unstructured_grid)
probe.Update()

# Get the interpolated data
interpolated_points = probe.GetOutput()
#print("Interpolation done!")

# Specify the name of the field you want to keep (e.g., "FieldData1")
desired_field_name = "phi"
point_data = interpolated_points.GetPointData()
if not point_data.HasArray(desired_field_name):
    raise ValueError(f"Il potenziale scalare '{desired_field_name}' non è presente nel dataset.")

# Controlla che sia scalare
array = point_data.GetArray(desired_field_name)
if array.GetNumberOfComponents() != 1:
    raise ValueError(f"L'array '{desired_field_name}' non è scalare, ha {array.GetNumberOfComponents()} componenti.")

# Convert the VTK array to a NumPy array
numpy_array = vtk_to_numpy(array)

print(f"Phi values (min, max): {numpy_array.min()}, {numpy_array.max()}")

# Convert the NumPy array back to a VTK array
new_array = numpy_to_vtk(numpy_array, deep=True)
new_array.SetName(desired_field_name) 



# Reshape and write the CUBE file
output = name_pqr.replace('.pqr', '.cube')
scalar_field = numpy_array.reshape((nx, ny, nz), order='F')
write_potential_cube(output, scalar_field, scale, origin, (nx, ny, nz))
print("File CUBE correctly saved!")

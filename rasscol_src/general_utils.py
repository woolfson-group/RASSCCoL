# general_utils.py

# builtins
import datetime

def pdb2seq(pdb_path: str) -> dict:
    """Extracts amino acid sequences from a PDB file."""
    
    AAdict = {
    'Ala': 'A', 'Val': 'V', 'Met': 'M',
    'Phe': 'F', 'Tyr': 'Y', 'Gln': 'Q',
    'Thr': 'T', 'Gly': 'G', 'Leu': 'L',
    'Ile': 'I', 'Pro': 'P', 'Ser': 'S',
    'Cys': 'C', 'Trp': 'W', 'Asp': 'D',
    'Asn': 'N', 'Glu': 'E', 'Lys': 'K',
    'Arg': 'R', 'His': 'H'
    }
    
    sequences = dict()
    with open(pdb_path, 'r') as pdb_obj:
        
        # Iterate over each line in the PDB file.
        for line in pdb_obj:
            
            if ' CA ' in line:  # Check if the line represents an alpha carbon atom.
                
                # Extract the chain identifier.
                chain = line[22]
                
                # Extract and format the residue name.
                residue = line[18:20].strip().title()  
                
                # Ensure a key for the chain exists.
                sequences.setdefault(chain, '')  
                
                # Append the single-letter code for the residue to the sequence.
                if not residue in list(AAdict.keys()):
                    print(f'Non-canonical amino acid: {residue}, replacing with X')
                    sequences[chain] += 'X'
                else:
                    sequences[chain] += AAdict[residue]
    return sequences

def get_timestamp():
    # Get the current local date and time
    return datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S').split()

def get_centroid(coords: list) -> list:
    """Calculates the centroid of the given coordinates."""
    # Calculate the mean of each column (x, y, z)
    num_points = len(coords)
    centroid = [sum(coord[i] for coord in coords) / num_points for i in range(3)]
    return centroid

def euclidean_distance(xyz1:list, xyz2:list) -> float:
    """Function to compute Euclidean distance"""
    return sum((xyz1[i]-xyz2[i])**2 for i in range(3))**0.5

def get_mol_len(coords:list) -> float:
    """Function to calculate length (maximum Euclidean distance) of atoms in a molecule"""
    
    # Initialize the maximum distance
    max_distance = 0

    # Compute pairwise distances
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            dist = euclidean_distance(coords[i], coords[j])
            if dist > max_distance:
                
                max_distance = round(dist,2)
                start_atom = coords[i]
                end_atom = coords[j]
            
    return max_distance, start_atom, end_atom

def get_pdbqt_coords(pdbqt_path: str, ca_only:bool=False) -> list:
    """
    Extracts the x, y, z coordinates of all atoms from a PDB file.

    Parameters:
    pdb_path (str): Path to the PDB file.

    Returns:
    np.ndarray: A NumPy array with shape (n_atoms, 3) containing x, y, z coordinates of all atoms.
    """
    # Initialise a list to store coordinates
    coords = []

    # Open and read the PDB file
    with open(pdbqt_path, 'r') as pdb_file:
        for line in pdb_file:
            if line.startswith("ATOM") and (not ca_only or ' CA ' in line):
                # Extract the x, y, z coordinates from columns 31-54
                x, y, z = map(float, [line[30:38].strip(), line[38:46].strip(), line[46:54].strip()])
                coords.append([x, y, z])
    
    # Return coords as np.array
    return coords

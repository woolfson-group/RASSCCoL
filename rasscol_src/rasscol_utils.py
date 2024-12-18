# rasscol_utils.py

# local modules
from rasscol_src.general_utils import *

# buitlins
from itertools import product
from multiprocessing import Pool, Lock
from pathlib import Path
import subprocess, csv, logging, math, re, sys
from collections import Counter

# third party modules
import vina
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdmolfiles

def smiles2mol(smiles, addH:bool = False):
    
    mol = Chem.MolFromSmiles(smiles)
    
    if addH:
        mol = Chem.AddHs(mol)  # Add explicit hydrogens

    # Generate 3D coordinates
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    
    return mol

def minimize_mol(mol):
    # Perform UFF minimization
    result = AllChem.UFFOptimizeMolecule(mol)

    # Check if the minimization was successful
    if result == 0:
        print("Minimization successful!")
    else:
        print("Minimization failed.")
    
    return mol

def mol2pdb(three_letter_code:str, mol, output_dir:Path):
    
    # Set the residue name for each atom
    for atom in mol.GetAtoms():
        atom.SetProp("residueName", three_letter_code)
    
    # Save the minimized molecule to a PDB file
    pdb_path = output_dir / f'{three_letter_code}.pdb'
    rdmolfiles.MolToPDBFile(mol, pdb_path)
    
    return pdb_path
    
def pdb2pdbqt(ligand_pdb_path, obabel_path:Path = Path('/usr/bin/obabel')):
    
    ligand_pdbqt_path = ligand_pdb_path.with_suffix('.pdbqt')
    obabel_cmd = f'{obabel_path} -ipdb {ligand_pdb_path} -opdbqt -O {ligand_pdbqt_path}'
    subprocess.run(obabel_cmd.split())
    
    return ligand_pdbqt_path

def calc_bondi_vol_from_mol(mol:Chem.rdchem.Mol) -> float:
    """
    https://pubs.acs.org/doi/10.1021/jo034808o
    Rg = total rings
    RA = aromatic rings 
    RNA = non aromatic rings
    """

    bondi_vol = {'H':7.24, 'C':20.58, 'N':15.6, 'O':14.71, 'F': 13.31, 'Cl':22.54, 'Br':26.52, 'I':32.52, 'P':24.43, 'S': 24.43, 'As':26.52, 'B':40.48, 'Si':38.79, 'Se':28.73, 'Te':36.62}

    # get the counts of each element in the molecule
    element_counts = Counter(atom.GetSymbol() for atom in mol.GetAtoms())

    # total number of atoms
    num_atoms = len(mol.GetAtoms())

    # Get the total number of rings
    ring_info = mol.GetRingInfo()
    num_rings = ring_info.NumRings()

    # Count the number of aromatic rings
    num_aromatic_rings = 0
    for ring in ring_info.BondRings():
        if all(mol.GetBondWithIdx(bond_idx).GetIsAromatic() for bond_idx in ring):
            num_aromatic_rings += 1

    # total number of bonds
    vsa = sum(element_counts[a]*bondi_vol[a] for a in element_counts)
    num_bonds = num_atoms - 1 - num_rings
    num_non_aromatic_rings = num_rings - num_aromatic_rings
    bondi_volume = vsa - 5.92*num_bonds - 14.7*num_aromatic_rings - 3.8*num_non_aromatic_rings
    return round(bondi_volume, 2)

def tidy_ligand_pdbqt(ligand_pdbqt_path:Path, ligand_short_name:str):
    """Cleans up a ligand PDBQT file by replacing placeholders with actual ligand names."""
    
    # Open the ligand PDBQT file and read its content as a string
    with ligand_pdbqt_path.open() as f:
        pdbqt_str = f.read()

    # Check if 'UNL' exists in the content
    if 'UNL' in pdbqt_str:

        # Replace the 'UNL' placeholder in the file with the actual short name of the ligand
        pdbqt_str = pdbqt_str.replace('UNL', ligand_short_name)

        # Write the modified content back to the PDBQT file
        with ligand_pdbqt_path.open('w') as f:
            f.write(pdbqt_str)  # Save the updated file with the replaced values


class RASSCoL:
    
    aa_vol={'G': 0, 'A': 17, 'V': 52, 'L': 69, 'I': 69, 'M': 70, 'F': 102, 'Y': 111, 'W': 133, 'C': 21, 'S': 26, 'T': 43, 'N': 52, 'Q': 69, 'D': 50, 'E': 67, 'K': 80, 'R': 100, 'H': 85, 'P': 49}
    
    vina_version = vina.__version__
    
    def in_pocket(self, x):
        if x < 5:
            return 1
        else:
            return 0

    def layer_sequence_generator(self, layer_info: dict, target: float, tolerance: float):
        """
        Generates sequences whose total volume is within the specified target and tolerance.
        This is a generator function that yields one sequence at a time to minimize memory usage.

        Args:
            layer_info (dict): A dictionary where keys are positions and values are lists of amino acids.
            target (float): The target volume for the sequences.
            tolerance (float): The acceptable deviation from the target volume.

        Yields:
            tuple: A tuple containing the sequence (as a tuple of amino acids) and its total volume.
        """

        # Get the amino acids for each index in sorted order
        characters = [layer_info[i] for i in sorted(layer_info.keys())]

        # Generate all possible combinations (this is an iterator, so it doesn't consume memory)
        combinations = product(*characters)

        # Set volume thresholds once outside the loop
        lower_threshold = target - tolerance
        upper_threshold = target + tolerance

        for seq in combinations:
            # Calculate volume of the sequence
            volume = sum(self.aa_vol[aa] for aa in seq)

            # Filter sequences based on volume criteria
            if lower_threshold < volume < upper_threshold:
                if seq.count('G') <= 1:
                    yield ''.join(seq), volume  # Yield the sequence and its volume
                    
    def update_sequence(self, start_seq, positions, amino_acids):
        """Updates the starting sequence at specified positions with given amino acids."""
        seq_list = list(start_seq)
        for aa, pos in zip(amino_acids, positions):
            seq_list[pos - 1] = aa  # Adjust for 0-based index
        return ''.join(seq_list)

    def sequence_generator(self, starting_seq, seqs, design_idx):
        
        logging.info(f'Generating sequence library...')
        
        seq_dict = {}
        
        for seq_id, sequence_combination in enumerate(product(*seqs.values())):
            combined_seq = ''.join(seq_info[0] for seq_info in sequence_combination)
            volumes = [seq_info[1] for seq_info in sequence_combination]
            modified_seq = self.update_sequence(starting_seq, design_idx, combined_seq)
            seq_dict[seq_id] = {'pocket_seq': combined_seq, 'full_seq': modified_seq, 'volume':volumes}
            
        return seq_dict

    def repack_pdb(self,
        input_path: Path, 
        output_pdb_path: Path, 
        seq: str = None, 
        rm_seq_txt: bool = True,
        faspr_path: Path = Path('/opt/FASPR/FASPR'),
        verbose: int = 0
        ):

        # Construct the command for subprocess
        repack_cmd = [
            faspr_path,
            '-i', str(input_path),
            '-o', str(output_pdb_path)
        ]

        # Write the sequence to a file if provided and append path to command
        if seq != None:
            repack_seq_path = output_pdb_path.parent / f'{output_pdb_path.stem}_seq.txt'
            with repack_seq_path.open('w') as repack_seq_file:
                repack_seq_file.write(seq)
            repack_cmd.extend(['-s', str(repack_seq_path)])

        # Execute the command with varying verbosity
        if verbose == 0:
            subprocess.run(repack_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif verbose == 1:
            subprocess.run(repack_cmd, stdout=subprocess.DEVNULL)
        elif verbose == 2:
            subprocess.run(repack_cmd)

        # Delete sequence files if unwanted
        if rm_seq_txt and repack_seq_path.is_file():
            repack_seq_path.unlink()

    def receptor_pdb2pdbqt(self, receptor_pdb_path, receptor_pdbqt_path, obabel_path):
        obabel_cmd=f'{obabel_path} -ipdb {receptor_pdb_path} -opdbqt --addpolarh -xr -O {receptor_pdbqt_path}'
        subprocess.run(obabel_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)

    def quick_vina(self,
        receptor_pdbqt_path:Path, 
        ligand_pdbqt_path:Path, 
        output_pdbqt_path:Path,
        center_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
        box_size: list[float, float, float] = [20, 20, 20],
        do_quick: bool = True,
    ) -> float:

        # Initialise Vina with specified parameters
        v = vina.Vina(sf_name='vina', seed=42, verbosity=0, cpu=1)
        
        # Load in receptor and ligand, converting to strings if necessary
        v.set_receptor(str(receptor_pdbqt_path))
        v.set_ligand_from_file(str(ligand_pdbqt_path))

        # Generate 20 angstrom cube affinity map around the specified center
        if do_quick:
            v.compute_vina_maps(center=center_xyz, box_size=box_size, spacing=0.5)
            v.dock(exhaustiveness=1, n_poses=1, max_evals=5_000)

        else:
            v.compute_vina_maps(center=center_xyz, box_size=box_size)
            v.dock(exhaustiveness=16, n_poses=1, max_evals=10_000)

        # Retrieve the total score of the first pose
        vina_score = v.energies(n_poses=1)[0][0]
        
        # Save best pose
        v.write_poses(str(output_pdbqt_path), n_poses=1, overwrite=True)
            
        # Return the total score of the first pose, scaling to avoid very positive numbers due to clashed(e.g. 10e6, bad for ML)
        if vina_score<1:
            return vina_score
        else:
            return math.log10(vina_score)

    def pack_and_dock(self, seq_id, sequence, scaffold, ligand, pocket_centroid, output_dir, cube_side_length, config, save=False):
        
        packed_pdb_path = output_dir / f'{seq_id}.pdb'
        packed_pdbqt_path = packed_pdb_path.with_suffix('.pdbqt')
        docked_pdbqt_path = output_dir / f'{packed_pdb_path.stem}_{ligand.stem}.pdbqt'

        self.repack_pdb(scaffold,packed_pdb_path,sequence, faspr_path=config['run']['FASPR_path'])
        self.receptor_pdb2pdbqt(packed_pdb_path, packed_pdbqt_path, obabel_path=config['run']['obabel_path'])
        vina_score = self.quick_vina(packed_pdbqt_path, ligand, docked_pdbqt_path, pocket_centroid, cube_side_length)
        
        dock_coords = get_pdbqt_coords(docked_pdbqt_path)
        dock_centroid = get_centroid(dock_coords)
        dist_from_pocket = euclidean_distance(pocket_centroid, dock_centroid)

        packed_pdb_path.unlink()

        if not save:
            packed_pdbqt_path.unlink()
            docked_pdbqt_path.unlink()

        return vina_score, dist_from_pocket

    # Global lock to be inherited by worker processes
    csv_lock = None

    # Set up logging to a file only (no console output)
    def setup_logger(self, log_file):
        logger = logging.getLogger()  # Get the root logger
        logger.setLevel(logging.INFO)  # Set log level
        
        # Remove existing handlers to avoid duplicate logs
        if logger.hasHandlers():
            logger.handlers.clear()
        
        # Create a file handler for logging to a file
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create a console handler for logging to STDOUT
        console_handler = logging.StreamHandler(sys.stdout) 
        console_handler.setLevel(logging.INFO)
        
        # Define the logging format
        formatter = logging.Formatter('%(asctime)s - %(processName)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add both handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    # Thread-safe CSV writing function
    def write_to_csv(self, file_path, row):
        with csv_lock:  # Ensure only one process writes at a time
            with open(file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
    
    def vina_worker(self, seq_id, combined_seq, modified_seq, volumes, scaffold, ligand, pocket_centroid, num_lig_atoms, log_file, csv_file, output_dir, cube_side_length, config):
        """
        Worker function for docking and data collection.

        Parameters:
            seq_id (str): Identifier for the sequence.
            combined_seq (str): Sequence at the design positions.
            modified_seq (str): Full modified sequence.
            volumes (list): List of volumes associated with the sequence.
            scaffold (Path): Path to the scaffold file.
            ligand (Path): Path to the ligand file.
            pocket_centroid (tuple): Coordinates of the pocket centroid.
            num_lig_atoms (int): Number of atoms in the ligand.
            log_file (Path): Path to the log file.
            csv_file (Path): Path to the CSV file.
            output_dir (Path): Output directory.

        Returns:
            None
        """

        # Set up logging inside each worker
        self.setup_logger(log_file)

        try:
            # Perform docking and calculations
            vina_score, dist_from_pocket = self.pack_and_dock(seq_id, modified_seq, scaffold, ligand, pocket_centroid, output_dir, cube_side_length, config)
            vina_score_norm = round(vina_score / num_lig_atoms, 3)
            
            # Calculate side-chain volumes
            total_side_chain_vol = sum(volumes)

            # Calculate cavity volume and ligand fraction 
            cavity_vol = parent_volume - total_side_chain_vol
            cavity_lig_frac = cavity_vol / total_ligand_volume

            # Write to CSV file in a thread-safe manner
            self.write_to_csv(csv_file, [seq_id, modified_seq, combined_seq, vina_score, vina_score_norm, dist_from_pocket, total_side_chain_vol, cavity_vol, cavity_lig_frac])

            logging.info(f'Job {seq_id} ({combined_seq}) finished with vina_score (norm): {round(vina_score,3)} ({vina_score_norm})')

        except Exception as e:
            logging.error(f'Error in job {seq_id}: {e}')

    
    def run_parallel(self, starting_seq, design_idx, seq_gen, pocket_centroid, num_lig_atoms, output_dir, scaffold, ligand, config, design_config):
        
        global csv_lock  # Declare the global lock
        csv_lock = Lock()  # Initialize the lock once, before starting the pool
        
        global parent_volume  # Declare the starting pocket side chain volume
        parent_volume = sum(self.aa_vol[resname] for resnum, resname in enumerate(starting_seq, start=1) if resnum in design_idx)
        
        global total_ligand_volume # Declare the total ligand volume for all layers
        total_ligand_volume = sum(design_config[layer]['ligand_layer_vol'] for layer in design_config)
        
        # Initialize logging
        log_file = output_dir / 'RASSCoL_log.log'
        self.setup_logger(log_file)

        # Initialize CSV file with headers
        csv_file = output_dir / 'RASSCoL_results.csv'
        
        if not csv_file.exists():
            self.write_to_csv(csv_file, ['id', 'seq', 'pocket_seq', 'vina_score', 'vina_score_norm', 'distance_from_pocket', 'sc_vol', 'cavity_vol', 'cavity_ligand_frac'])

        # Prepare arguments for starmap
        args = [
            (
                seq_id,                                  # Sequence ID
                seq_gen[seq_id]['pocket_seq'],           # Short sequence (pocket sequence)
                seq_gen[seq_id]['full_seq'],             # Long sequence (full sequence)
                seq_gen[seq_id]['volume'],               # Volumes associated with the sequence
                scaffold,                                # Scaffold path
                ligand,                                  # Ligand path
                pocket_centroid,                         # Pocket centroid
                num_lig_atoms,                           # Number of ligand atoms
                log_file,                                # Log file path
                csv_file,                                # CSV file path
                output_dir,                              # Output directory
                config['run']['cube_side_length'],       # The cube side length of the docking grid. 
                config                                   # Configuration info
            )
            for seq_id in seq_gen
        ]

        # Multiprocessing execution
        with Pool(config['run']['num_cpus']) as pool:
            pool.starmap(self.vina_worker, args)
            
    def save_structures(self, config:dict, results_csv_path:Path):

        # Read CSV data and store it in a list of dictionaries
        with results_csv_path.open() as f:
            reader = csv.DictReader(f)
            results = [row for row in reader]

        # Sort the results based on the 'vina_score_norm' column and get the top N
        top_n = results[:config['run']['save_top_n']]

        # Iterate through the top N and call pack_and_dock
        for row in top_n:
            self.pack_and_dock(row['id'], row['seq'], Path(config['run']['receptor_path']), Path(config['run']['ligand_path']), config['run']['pocket_ca_centroid'], Path(config['run']['output_directory']), config['run']['cube_side_length'], config=config, save=True)
            print(f"Saved design {row['id']} at {config['run']['output_directory']}/{row['id']}_{Path(config['run']['ligand_path']).stem}.pdbqt")

import subprocess
from pathlib import Path
from itertools import product
from typing import Dict, Tuple
from collections import Counter

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolfiles

from rasscol_src.general_utils import get_centroid, get_mol_len, get_pdbqt_coords

# ------------------------------------- Design info ------------------------------------- #

def parse_design_info(design_info_text: Path) -> dict:
    """
    Parses a block-formatted design config into a structured dictionary.
    
    Expected format:
        <layer> <cavity> <tolerance> <lig_layer_vol>
        <resnum> <aaset>
        ...
    
    Blocks separated by empty lines.
    """
    
    raw_text = design_info_text.read_text()
    
    design_config = {}

    for block in raw_text.strip().split('\n\n'):
        lines = [line.strip() for line in block.strip().splitlines() if line.strip() and not line.startswith('#')]
        if not lines:
            continue

        # First line: metadata
        try:
            layer_str, cavity, tolerance, lig_vol = lines[0].split()
            layer = int(layer_str)
        except ValueError as e:
            raise ValueError(f"Invalid header line: {lines[0]}") from e

        # Remaining lines: residue info
        res_info = {}
        for line in lines[1:]:
            try:
                resnum_str, aaset = line.split()
                res_info[int(resnum_str)] = aaset
            except ValueError as e:
                raise ValueError(f"Invalid residue line: {line}") from e

        design_config[layer] = {
            'res_info': res_info,
            'cavity': float(cavity),
            'tolerance': float(tolerance),
            'target': float(lig_vol),
        }

    return design_config


# ------------------------------------- LigandBuilder ------------------------------------- #

class LigandBuilder:
    def __init__(
        self,
        output_dir: Path,
        bondi_volumes: dict,
        obabel_path: Path,
    ):
        self.output_dir = Path(output_dir)
        self.bondi_volumes = bondi_volumes
        self.obabel_path = Path(obabel_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def smiles_to_mol(self, smiles: str, add_hydrogens: bool = True) -> Chem.Mol:
        mol = Chem.MolFromSmiles(smiles)
        if add_hydrogens:
            mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        return mol

    def minimize(self, mol: Chem.Mol) -> Chem.Mol:
        result = AllChem.UFFOptimizeMolecule(mol)
        if result != 0:
            print("Warning: minimization failed.")
        return mol

    def mol_to_pdb(self, mol: Chem.Mol, three_letter_code: str, pdb_path:Path):
        for atom in mol.GetAtoms():
            atom.SetProp("residueName", three_letter_code)
        rdmolfiles.MolToPDBFile(mol, str(pdb_path))

    def pdb_to_pdbqt(self, pdb_path: Path, pdbqt_path:Path) -> Path:
        cmd = [str(self.obabel_path), "-ipdb", str(pdb_path), "-opdbqt", "-O", str(pdbqt_path)]
        subprocess.run(cmd, check=True)

    def calc_bondi_volume(self, mol: Chem.Mol) -> float:

        element_counts = Counter(atom.GetSymbol() for atom in mol.GetAtoms())
        num_atoms = len(mol.GetAtoms())
        ring_info = mol.GetRingInfo()
        num_rings = ring_info.NumRings()

        num_aromatic_rings = sum(
            all(mol.GetBondWithIdx(b).GetIsAromatic() for b in ring)
            for ring in ring_info.BondRings()
        )

        vsa = sum(element_counts[el] * self.bondi_volumes.get(el, 0) for el in element_counts)
        num_bonds = num_atoms - 1 - num_rings
        num_non_aromatic_rings = num_rings - num_aromatic_rings

        volume = vsa - 5.92 * num_bonds - 14.7 * num_aromatic_rings - 3.8 * num_non_aromatic_rings
        return round(volume, 2)

    def tidy_pdbqt(self, pdbqt_path: Path, ligand_name: str):
        content = pdbqt_path.read_text()
        if "UNL" in content:
            pdbqt_path.write_text(content.replace("UNL", ligand_name))

    def analyze_ligand(self, pdbqt_path: Path) -> dict:
        """Analyze a processed ligand by extracting spatial and structural metadata."""
        
        coords = get_pdbqt_coords(pdbqt_path)
        num_atoms = len(coords)
        centroid = get_centroid(coords)
        length = get_mol_len(coords)
        

        return {
            'num_atoms': num_atoms,
            'coords': coords,
            'centroid': centroid,
            'length': length
        }

# ------------------------------------- SequenceBuilder ------------------------------------- #

class SequenceBuilder:
    """A class to generate sequences of amino acids based on specified layer information and volume constraints."""
    
    def __init__(self, aa_encoder: Dict[str, int], aa_volumes: Dict[str, int]):
        self.aa_encoder = aa_encoder
        self.aa_volumes = aa_volumes

    def layer_sequence_generator(self, layer_info, target, tolerance):
        """
        Build all amino acid sequences that fit a target volume within a specified tolerance.
        
        This function explores all possible amino acid combinations across specified positions
        and filters sequences whose total volume falls within [target - tolerance, target + tolerance].
        It uses depth-first search (DFS) with early pruning to optimize performance by avoiding
        paths that already exceed the maximum allowed volume.

        Args:
            layer_info (dict): A dictionary mapping residue positions (ints) to lists of allowed
                amino acids (strs) at that position.
            target (float): The desired total volume for the sequence.
            tolerance (float): The allowed deviation from the target volume.

        Returns:
            list of tuple: Each tuple contains:
                - A string representing a full amino acid sequence.
                - The total volume of that sequence (float).
        """
        
        # Sort residue positions to ensure consistent order when building sequences
        aa_order = sorted(layer_info.keys())
        n_positions = len(aa_order)
        
        # Precompute volume bounds for pruning
        lower_bound = target - tolerance
        upper_bound = target + tolerance
        
        # List to collect valid sequences
        results = []
        
        def dfs(pos, current_seq, current_volume):
            """Recursive depth-first search to build sequences."""
            
            # Base case: all positions filled
            if pos == n_positions:
                # Check if final sequence volume falls within acceptable range
                if lower_bound <= current_volume <= upper_bound:
                    # Join the amino acids into a string and save the sequence and its volume
                    results.append((''.join(current_seq), current_volume))
                return
            
            # Recursive case: explore next amino acid choices at current position
            for aa in layer_info[aa_order[pos]]:
                aa_volume = self.aa_volumes[aa]
                new_volume = current_volume + aa_volume
                
                # Early pruning: skip paths that already exceed upper bound
                if new_volume > upper_bound:
                    continue
                
                # Continue building the sequence
                dfs(pos + 1, current_seq + [aa], new_volume)
        
        # Start DFS with empty sequence and zero volume
        dfs(0, [], 0)
        
        return results
    
    
    def generate_sequences_and_volumes(
        self,
        seqs: Dict[int, list],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate integer-encoded sequences and total volumes from layered sequence data.

        Args:
            seqs (dict): Dictionary mapping layer index to list of (full_sequence_string, volume) tuples.
            aa_encoder (dict): Dictionary mapping amino acid letters to integer codes.

        Returns:
            Tuple:
                - sequences_array (np.ndarray): Shape (num_sequences, total_sequence_length), dtype=int8
                - volumes_array (np.ndarray): Shape (num_sequences,), dtype=int16
        """
        layer_keys = list(seqs.keys())

        # Get the full sequence length after combining all layers
        full_seq_len = sum(len(seqs[layer_key][0][0]) for layer_key in layer_keys)

        aa_lists = []
        vol_lists = []

        for layer_key in layer_keys:
            aa_entries = []
            vol_entries = []
            
            # encode the sequences
            for full_seq_string, vol in seqs[layer_key]:
                aa_codes = [self.aa_encoder[aa] for aa in full_seq_string]
                aa_entries.append(aa_codes)
                vol_entries.append(vol)

            aa_lists.append(np.array(aa_entries, dtype=np.int8))
            vol_lists.append(np.array(vol_entries, dtype=np.int16))

        # Cartesian product of indices
        idx_options = [np.arange(len(aa_list)) for aa_list in aa_lists]
        idx_combinations = np.array(list(product(*idx_options)), dtype=np.int32)  # shape (num_sequences, num_layers)

        num_seqs = idx_combinations.shape[0]

        # Initialize outputs
        sequences_array = np.zeros((num_seqs, full_seq_len), dtype=np.int8)
        volumes_array = np.zeros(num_seqs, dtype=np.int16)

        # Fill outputs
        current_pos = 0
        for layer_idx, (aa_list, vol_list) in enumerate(zip(aa_lists, vol_lists)):
            selected_indices = idx_combinations[:, layer_idx]

            volumes_array += vol_list[selected_indices]

            aa_layer = aa_list[selected_indices]  # shape (num_seqs, num_positions_in_layer)

            num_positions = aa_layer.shape[1]
            sequences_array[:, current_pos:current_pos + num_positions] = aa_layer
            current_pos += num_positions
            
        ids_array = np.arange(num_seqs, dtype=np.int32)

        return ids_array, sequences_array, volumes_array, num_seqs
        

import sys
import math
import time
import subprocess
from pathlib import Path

from rasscol_src.data_utils import AA_DECODER
from rasscol_src.general_utils import euclidean_distance, get_pdbqt_coords, get_centroid

import vina

class VinaJob:
    def __init__(self, idx, pocket_encoded, vol, config):
        self.idx = idx
        self.pocket_encoded = pocket_encoded
        self.vol = vol
        self.config = config
        self.tmp_dir = Path(config["output_directory"]) / 'out'

    def empty(self):
        time.sleep(1)

    @staticmethod
    def update_sequence(start_seq, positions, amino_acids):
        """Updates the starting sequence at specified positions with given amino acids."""
        seq_list = list(start_seq)
        for aa, pos in zip(amino_acids, positions):
            seq_list[pos - 1] = aa  # Adjust for 0-based index
        return ''.join(seq_list)

    @staticmethod
    def repack_pdb(input_path: Path, output_pdb_path: Path, seq: str = None, \
            rm_seq_txt: bool = True, faspr_path: Path = Path('/opt/FASPR/FASPR')):

        # Construct the command for subprocess
        repack_cmd = [faspr_path, '-i', str(input_path), '-o', str(output_pdb_path)]
        
        # Write the sequence to a file if provided and append path to command
        if seq != None:
            repack_seq_path = output_pdb_path.with_suffix('.seq')
            repack_seq_path.write_text(seq)
            repack_cmd.extend(['-s', str(repack_seq_path)])
        
        # Execute the command
        subprocess.run(repack_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Delete sequence files if unwanted
        if rm_seq_txt and repack_seq_path.is_file():
            repack_seq_path.unlink()

    @staticmethod
    def receptor_pdb2pdbqt(receptor_pdb_path, receptor_pdbqt_path, obabel_path):
        
        obabel_cmd=f'{obabel_path} -ipdb {receptor_pdb_path} -opdbqt --addpolarh -xr -O {receptor_pdbqt_path}'
        subprocess.run(obabel_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)

    @staticmethod
    def quick_vina(receptor_pdbqt_path:Path,  ligand_pdbqt_path:Path, output_pdbqt_path:Path, \
            center_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0), box_size: list[float, float, float] = [20, 20, 20], \
            do_quick: bool = True) -> float:

        # Initialise Vina with specified parameters
        v = vina.Vina(sf_name='vina', seed=42, verbosity=0, cpu=1)
        
        # Load in receptor and ligand, converting to strings if necessary
        v.set_receptor(str(receptor_pdbqt_path))
        v.set_ligand_from_file(str(ligand_pdbqt_path))

        # Generate affinity map around the specified center
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
        if vina_score<0:
            return vina_score
        else:
            return math.log10(vina_score)

    @staticmethod
    def calc_lig_dist_from_pocket(docked_pdbqt_path:Path, pocket_centroid:list) -> float:
        dock_coords = get_pdbqt_coords(docked_pdbqt_path)
        dock_centroid = get_centroid(dock_coords)
        return euclidean_distance(pocket_centroid, dock_centroid)

    @staticmethod
    def calc_vol_metrics(total_side_chain_vol, parent_volume, total_ligand_volume):

        # Calculate cavity volume and ligand fraction 
        cavity_vol = parent_volume - total_side_chain_vol
        cavity_lig_frac = cavity_vol / total_ligand_volume
        
        return cavity_vol, cavity_lig_frac
    
    def run_single_job(self, cleanup=True):
        
        config = self.config

        pocket_decoded = ''.join([AA_DECODER[aa] for aa in self.pocket_encoded])
        seq = VinaJob.update_sequence(config["starting_seq"].lower(), config['design_idx'], pocket_decoded)

        packed_pdb_path = self.tmp_dir / f'{self.idx}.pdb'
        packed_pdbqt_path = packed_pdb_path.with_suffix('.pdbqt')
        docked_pdbqt_path = self.tmp_dir / f'{packed_pdb_path.stem}_{Path(config["ligand_path"]).stem}.pdbqt'

        VinaJob.repack_pdb(config['receptor_path'], packed_pdb_path, seq, faspr_path=Path(config['FASPR_path']))
        VinaJob.receptor_pdb2pdbqt(packed_pdb_path, packed_pdbqt_path, obabel_path=Path(config['obabel_path']))

        vina_score = VinaJob.quick_vina(
            receptor_pdbqt_path=packed_pdbqt_path,
            ligand_pdbqt_path=config['ligand_path'],
            output_pdbqt_path=docked_pdbqt_path,
            center_xyz=config['pocket_ca_centroid'],
            box_size=config['cube_side_length']
        )

        vina_score_norm = round(vina_score / config['num_lig_atoms'], 3)
        dist = VinaJob.calc_lig_dist_from_pocket(docked_pdbqt_path, config['pocket_ca_centroid'])
        cav_vol, cav_lig_frac = VinaJob.calc_vol_metrics(self.vol, config["parent_volume"], config["total_ligand_volume"])

        result = {
            'id': int(self.idx),
            'vina_score': vina_score,
            'vina_score_norm': vina_score_norm,
            'distance': dist,
            'sc_volume': int(self.vol),
            'cavity_volume': int(cav_vol),
            'cavity_ligand_frac': float(cav_lig_frac)
        }

        print(f'Job {self.idx} done with vina_score: {vina_score:.3f}')
        
        if cleanup:
            for f in [packed_pdb_path, packed_pdbqt_path, docked_pdbqt_path]:
                try:
                    f.unlink()
                except FileNotFoundError:
                    pass
        
        return result

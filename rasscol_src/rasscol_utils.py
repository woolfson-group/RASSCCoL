# rasscol_utils.py

# local modules
from rasscol_src.general_utils import *

# buitlins
from itertools import product, repeat
from multiprocessing import Pool, Lock, Manager
import copy
from pathlib import Path
import subprocess, csv, logging, math, re, sys
from collections import Counter
import random
import warnings
import time


# third party modules
import vina
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdmolfiles
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_decision_forests as tfdf

seed = 42
random.seed(seed)

def smiles2mol(smiles, addH:bool = False):
    
    mol = Chem.MolFromSmiles(smiles)
    
    if addH:
        mol = Chem.AddHs(mol)  # Add explicit hydrogens

    # Generate 3D coordinates
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    
    return mol

def sample_and_remove(numbers: list, number_to_sample: int):
    arr = np.array(numbers)
    idx = np.random.choice(len(arr), number_to_sample, replace=False)
    sampled = arr[idx]
    remaining = np.delete(arr, idx)
    return remaining.astype('int'), sampled.astype('int')

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

class VinaJob:
    def __init__(self, seq_id, seq_dict, config_run, log_file, csv_file, aa_vol, csv_lock):
        self.seq_id = seq_id
        self.seq_dict = seq_dict
        self.config = config_run
        self.log_file = log_file
        self.csv_file = csv_file
        self.aa_vol = aa_vol
        self.parent_volume = config_run['parent_volume']
        self.total_ligand_volume = config_run['total_ligand_volume']
        self.csv_lock = csv_lock

        self.setup_logger()

    def empty(self):
        time.sleep(1)

    @staticmethod
    def update_sequence(start_seq, positions, amino_acids):
        """Updates the starting sequence at specified positions with given amino acids."""
        seq_list = list(start_seq)
        for aa, pos in zip(amino_acids, positions):
            seq_list[pos - 1] = aa  # Adjust for 0-based index
        return ''.join(seq_list)

    def setup_logger(self):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        if logger.hasHandlers():
            logger.handlers.clear()

        file_handler = logging.FileHandler(self.log_file)
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(processName)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    def write_to_csv(self, row):
        with self.csv_lock:
            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)

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
    def calc_vol_metrics(volumes, parent_volume, total_ligand_volume):
        # Calculate side-chain volumes
        total_side_chain_vol = sum(volumes)

        # Calculate cavity volume and ligand fraction 
        cavity_vol = parent_volume - total_side_chain_vol
        cavity_lig_frac = cavity_vol / total_ligand_volume
        
        return total_side_chain_vol, cavity_vol, cavity_lig_frac

    def run(self):
        try:
            # Setup paths
            output_dir = Path(self.config['output_directory'])
            scaffold = Path(self.config['receptor_path'])
            ligand = Path(self.config['ligand_path'])
            faspr_path = Path(self.config['FASPR_path'])
            obabel_path = Path(self.config['obabel_path'])

            combined_seq = self.seq_dict['pocket_seq']
            modified_seq = VinaJob.update_sequence(self.config['starting_seq'], self.config['design_idx'], combined_seq)

            packed_pdb_path = output_dir / f'{self.seq_id}.pdb'
            packed_pdbqt_path = packed_pdb_path.with_suffix('.pdbqt')
            docked_pdbqt_path = output_dir / f'{packed_pdb_path.stem}_{ligand.stem}.pdbqt'

            VinaJob.repack_pdb(scaffold, packed_pdb_path, modified_seq, faspr_path=faspr_path)
            VinaJob.receptor_pdb2pdbqt(packed_pdb_path, packed_pdbqt_path, obabel_path=obabel_path)
            vina_score = VinaJob.quick_vina(packed_pdbqt_path, ligand, docked_pdbqt_path, self.config['pocket_ca_centroid'], self.config['cube_side_length'])

            vina_score_norm = round(vina_score / self.config['num_lig_atoms'], 3)
            dist = VinaJob.calc_lig_dist_from_pocket(docked_pdbqt_path, self.config['pocket_ca_centroid'])
            sc_vol, cav_vol, cav_lig_frac = VinaJob.calc_vol_metrics(self.seq_dict['volume'], self.parent_volume, self.total_ligand_volume)

            self.write_to_csv([self.seq_id, combined_seq, vina_score, vina_score_norm, dist, sc_vol, cav_vol, cav_lig_frac])
            logging.info(f'Job {self.seq_id} done with vina_score: {vina_score:.3f} norm: {vina_score_norm}')

        except Exception as e:
            logging.error(f'Error in job {self.seq_id}: {e}')
            #active sampling will crash if this does not write a result
            self.write_to_csv([self.seq_id, self.seq_dict['pocket_seq'], 0, 0, 0, 0, 0, 0])
        finally:
            # Clean up files
            for f in [packed_pdb_path, packed_pdbqt_path, docked_pdbqt_path]:
                try:
                    f.unlink()
                except FileNotFoundError:
                    pass


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


        aa_order = sorted(layer_info.keys())
        aa_lists = [layer_info[i] for i in aa_order]
        
        # Create full Cartesian product of all amino acids
        all_combos = list(product(*aa_lists))  # shape: (N, len(aa_lists))

        # Convert to NumPy for vectorization
        aa_array = np.array(all_combos)  # shape: (num_combos, seq_len)

        # Vectorized volume lookup
        vol_lookup = np.vectorize(self.aa_vol.get)
        vol_array = vol_lookup(aa_array)  # shape: (num_combos, seq_len)

        # Sum volumes
        volumes = vol_array.sum(axis=1)

        # Filter by target range
        lower = target
        upper = target + tolerance
        mask = (volumes > lower) & (volumes < upper)

        # Apply mask to filter sequences + volumes
        valid_seqs = aa_array[mask]
        valid_volumes = volumes[mask]
        print(layer_info)
        print(f'This layer has {len(valid_seqs)} sequences.')

        return [(''.join(seq), int(vol)) for seq, vol in zip(valid_seqs, valid_volumes)]
                    
    def sequence_generator(self, seqs):
        
        logging.info(f'Generating sequence library...')
        
        seq_dict = {}
        
        for seq_id, sequence_combination in enumerate(product(*seqs.values())):
            combined_seq = ''.join(seq_info[0] for seq_info in sequence_combination)
            volumes = [seq_info[1] for seq_info in sequence_combination]
            seq_dict[seq_id] = {'pocket_seq': combined_seq, 'volume':volumes}
            
        return seq_dict


    @staticmethod
    def run_job(seq_id, seq_dict, config_run, log_file, csv_file, aa_vol, csv_lock):
        """
        Worker function for docking and data collection.

        Parameters:
            seq_id (str): Identifier for the sequence.
            config (dict): Configuration dictionary for run.
            log_file (Path): Path to the log file.
            csv_file (Path): Path to the CSV file.
            aa_vol (dict): Amino acid volume data.
            parent_volume (float): Parent volume.
            total_ligand_volume (float): Total ligand volume.
            csv_lock (Lock): CSV lock for thread-safe writing.

        Returns:
            None
        """

        job = VinaJob(
                seq_id,
                seq_dict,
                config_run,
                log_file,
                csv_file,
                aa_vol,
                csv_lock
            )
        job.run()

    def run_parallel(self, config, slice=None):
        
        with Manager() as manager:
            csv_lock = manager.Lock()

            log_file = Path(config['run']['output_directory']) / 'RASSCoL_log.log'
            csv_file = Path(config['run']['output_directory']) / 'RASSCoL_results.csv'

            if not csv_file.exists():
                with csv_lock:
                    with open(csv_file, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['id', 'pocket_seq', 'vina_score', 'vina_score_norm', 'distance_from_pocket', 'sc_vol', 'cavity_vol', 'cavity_ligand_frac'])


            # Prepare arguments for starmap execution
            run_config = config['run']
            job_ids = slice if slice is not None else config['seqs'].keys()

            args = (
                (
                    seq_id,
                    config['seqs'][seq_id],
                    config['run'],
                    log_file,
                    csv_file,
                    self.aa_vol,
                    csv_lock
                )
                for seq_id in job_ids
            )

            # Using starmap to run jobs in parallel
            with Pool(config['run']['num_cpus']) as pool:
                pool.starmap(RASSCoL.run_job, args)

        print("All jobs completed")
            
            
    def save_structures(self, config:dict, results_csv_path:Path):

        # Read CSV data and store it in a list of dictionaries
        with results_csv_path.open() as f:
            reader = csv.DictReader(f)
            results = [row for row in reader]

        # Sort the results based on the 'vina_score_norm' column and get the top N
        top_n = results[:config['run']['save_top_n']]

        # Iterate through the top N and call pack_and_dock
        for row in top_n:
            
            # generate file path variables
            packed_pdb_path = Path(config['run']['output_directory']) / f"{row['id']}.pdb"
            packed_pdbqt_path = packed_pdb_path.with_suffix('.pdbqt')
            docked_pdbqt_path = Path(config['run']['output_directory']) / f"{packed_pdb_path.stem}_{Path(config['run']['ligand_path']).stem}.pdbqt"
            seq = VinaJob.update_sequence(config['run']['starting_seq'], config['run']['design_idx'], row['pocket_seq'])
            VinaJob.repack_pdb(Path(config['run']['receptor_path']), packed_pdb_path, seq, faspr_path=config['run']['FASPR_path'])
            VinaJob.receptor_pdb2pdbqt(packed_pdb_path, packed_pdbqt_path, obabel_path=config['run']['obabel_path'])
            _ = VinaJob.quick_vina(packed_pdbqt_path, Path(config['run']['ligand_path']), docked_pdbqt_path, config['run']['pocket_ca_centroid'], config['run']['cube_side_length'])
            
            # delete unwanted files
            packed_pdb_path.unlink() 
            print(f"Saved design {row['id']} at {config['run']['output_directory']}/{row['id']}_{Path(config['run']['ligand_path']).stem}.pdbqt")

    def run_active_sampling(self, config, stopping_patience = None, verbose = 0):

        csv_file = Path(config['run']['output_directory']) / 'RASSCoL_results.csv'


        final_num_designs = config['run']['gradient_boosted_top_sequences']
        steps = config['run']['gradient_boosted_steps']

        step_size = config['run']['gradient_boosted_step_size']

        num_models = 10  # Number of models to train in the ensemble, 10 was found to save training time but give good uncertainty values

        # make a dataframe from the config, easier to track sampling process this way
        #expand sequences into training variables
        tmp_array = np.array([list(entry['pocket_seq']) + list(map(int, entry['volume'])) for entry in config['seqs'].values()])

        num_residues = len(config['seqs'][0]['pocket_seq'])
        num_layers = len(config['seqs'][0]['volume'])
        columns = [f's{i}' for i in range(num_residues)] + [f'v{i}' for i in range(num_layers)]
        df_rf = pd.DataFrame(tmp_array, columns=columns)

        df_rf['Docking'] = 0.

        seq_ids = list(df_rf.index)

        #shuffle the dataset, select training and testing sets
        seq_ids, train_index = sample_and_remove(seq_ids, round(step_size))
        seq_ids, mse_index = sample_and_remove(seq_ids, round(step_size))

        print('Datasets ready, starting active sampling!')
        self.run_parallel(config,train_index)
        self.run_parallel(config,mse_index)

        results_df = pd.read_csv(csv_file, index_col='id')
        results_df.index = results_df.index.astype(int)

        # Update training data with new docking scores

        df_rf.loc[train_index, 'Docking'] = results_df.loc[train_index, 'vina_score']
        df_rf.loc[mse_index, 'Docking'] = results_df.loc[mse_index, 'vina_score']

        mse_list = []

        # Set up logging 
        sampling_log_file = Path(config['run']['output_directory']) / 'RASSCoL_sampling.log'
        open_sampling_log = open(sampling_log_file, 'w')

        if stopping_patience is None:
            warnings.warn('stopping_patience = None')

        # ------------------------- Main Loop -------------------------

        for i in range(steps):
            print(f'\nStarting Step {i + 1}/{steps}')

            # Prepare training and test datasets

            df_train = df_rf.loc[train_index]
            df_mse = df_rf.loc[mse_index]
            
            df_test = df_rf.drop(np.concatenate((train_index, mse_index)))
            
            # Drop the "Predictions" column if it exists
            features = [col for col in df_train.columns if col not in ['Docking', 'Predictions', 'Predictions_std']]

            # Convert data to TensorFlow datasets
            train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(df_train[features + ['Docking']], label='Docking', task=tfdf.keras.Task.REGRESSION)
            test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(df_test[features + ['Docking']], label='Docking', task=tfdf.keras.Task.REGRESSION)
            mse_ds = tfdf.keras.pd_dataframe_to_tf_dataset(df_mse[features + ['Docking']], label='Docking', task=tfdf.keras.Task.REGRESSION)

            # ------------------------- Model Training -------------------------

            print('Training Gradient Boosted Trees models...')
            tuner = tfdf.tuner.RandomSearch(num_trials=20, use_predefined_hps=True)
            models = []
            
            for k in range(num_models):
                model = tfdf.keras.GradientBoostedTreesModel(tuner=tuner, task=tfdf.keras.Task.REGRESSION, random_seed=k, verbose=verbose)
                model.fit(train_ds)
                models.append(model)
                print(f'Model {k + 1}/{num_models} trained.')

            print(f'{num_models} models trained successfully.')

            class CombinedModel(tf.keras.Model):
                """Ensemble model combining predictions from multiple Gradient Boosted Trees models."""
                def call(self, inputs):
                    predictions = tf.concat([submodel(inputs) for submodel in models], axis=1)
                    return tf.math.reduce_mean(predictions, axis=1), tf.math.reduce_std(predictions, axis=1)

            combined_model = CombinedModel()
            train_predictions, _ = combined_model.predict(train_ds)

            # ------------------------- Evaluate on MSE Dataset -------------------------

            mse_predictions_scaled, mse_predictions_std = combined_model.predict(mse_ds)
            mse = np.mean(np.square(mse_predictions_scaled - df_mse['Docking'].values))

            mse_list.append(mse)
            open_sampling_log.write(f'Step {i + 1} validation MSE: {mse:.4f}')

            # Check for early stopping
            if stopping_patience is not None:

                if i >= stopping_patience and all(x > mse for x in mse_list[-stopping_patience:]):
                    print(f'Stopping early after {i + 1} steps due to no improvement in MSE.')
                    break

            # ------------------------- Active Sampling -------------------------

            test_predictions_scaled, prediction_std = combined_model.predict(test_ds)

            # Save predictions
            df_test['Predictions'] = test_predictions_scaled
            df_test['Predictions_std'] = prediction_std

            # Select samples for the next round
            worst_index = df_test.nlargest(round(step_size * 0.25), 'Predictions_std').index  # High uncertainty
            best_index = df_test.nsmallest(round(step_size * 0.25), 'Predictions').index  # Best docking scores

            # Combine worst and best indices, ensuring no duplicates
            combined_index = worst_index.union(best_index)

            # Exclude indices already in combined_index to avoid duplicates
            allowed_sample_space = df_test.index.difference(combined_index)

            # Randomly sample from the allowed sample space, sometime best and least certain data points overlap - random fraction needs to be adjusted
            random_index = allowed_sample_space.to_series().sample(n=round(step_size-len(combined_index)), random_state=seed).index

            # Final combined index with worst, best, and random samples
            final_combined_index  = pd.Index(combined_index).union(random_index)
            #sanity check final_combined_index = new_test_index
            new_test_index = list(final_combined_index.difference(results_df.index))
            print(len(final_combined_index),len(new_test_index))

            # ------------------------- Dock New Samples -------------------------

            print(f'Docking {len(new_test_index)} new samples...')
            self.run_parallel(config,new_test_index)

            results_df = pd.read_csv(csv_file, index_col='id')
            results_df.index = results_df.index.astype(int)

            # Update training data with new docking scores
            df_rf.loc[new_test_index, 'Docking'] = results_df.loc[new_test_index, 'vina_score']
            train_index = np.unique(np.concatenate((train_index,new_test_index)))
    
    # ----------------------------- Dock Best predictions -----------------------

        #prevent redocking already sampled sequences
        best_index = df_test.nsmallest(final_num_designs, 'Predictions').index
        self.run_parallel(config,best_index)
        df_test.to_csv(Path(config['run']['output_directory']) / 'RASSCoL_DF_test.csv')

        df_mse['Predictions'] = mse_predictions_scaled
        df_mse['Predictions_std'] = mse_predictions_std
        df_mse.to_csv(Path(config['run']['output_directory']) / 'RASSCoL_DF_validate.csv')

        open_sampling_log.close()

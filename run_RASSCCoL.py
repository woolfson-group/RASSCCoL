import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append("rasscol_src")

from rasscol_src.rasscol_utils import RASSCCoL
from rasscol_src.data_utils import BONDI_VOLUMES, AA_ENCODER, AA_VOLUMES
from rasscol_src.prep_utils import parse_design_info, LigandBuilder, SequenceBuilder
from rasscol_src.general_utils import get_timestamp, pdb2seq, get_pdbqt_coords, get_centroid

def parse_args():
    parser = argparse.ArgumentParser(description="Run RASSCCoL with optional gradient boosting.")

    # Required
    parser.add_argument("--output_directory", type=Path, required=True, help="Output directory for results.")
    parser.add_argument("--receptor_pdb_path", type=Path, required=True, help="Path to the receptor PDB file.")
    parser.add_argument("--ligand_smiles", type=str, required=True, help="Ligand SMILES string.")
    parser.add_argument("--design_info_path", type=Path, required=True, help="Path to the design info file.")
    parser.add_argument("--faspr_path", type=Path, required=True, help="Path to FASPR binary.")
    parser.add_argument("--obabel_path", type=Path, required=True, help="Path to Open Babel binary.")

    # Optional
    parser.add_argument("--ligand_name", type=str, default="ligand", help="Name for the ligand (default: ligand).")
    parser.add_argument("--ligand_3letter", type=str, default="LIG", help="3-letter code for the ligand (default: LIG).")
    
    parser.add_argument("--use_gradient_boosted_trees", action="store_true", help="Use gradient boosted trees for scoring.")
    parser.add_argument("--gradient_boosted_top_sequences", type=int, default=100, help="Number of top sequences to keep.")
    parser.add_argument("--gradient_boosted_steps", type=int, default=2, help="Number of boosting steps.")
    parser.add_argument("--gradient_boosted_step_size", type=int, default=250, help="Number of sequences per boosting step.")

    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for processing.")
    parser.add_argument("--num_cpus", type=int, default=24, help="Number of CPU threads to use.")
    parser.add_argument("--save_top_n", type=int, default=3, help="Number of top-scoring sequences to save.")
    parser.add_argument("--calc_seqs_only", action="store_true", help="Only calculate sequences, skip docking.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing outputs.")

    return parser.parse_args()

def main():
    
    # parse command line arguments
    args = parse_args()
    
    # set gradient boosting parameters to 0 if not used
    if not args.use_gradient_boosted_trees:
        args.gradient_boosted_top_sequences = 0
        args.gradient_boosted_steps = 0
        args.gradient_boosted_step_size = 0

    datestamp, timestamp = get_timestamp()

    #  I/O -----------------------------------------------------------------------

    job_dir = args.output_directory
    input_dir = job_dir / 'input'
    input_dir.mkdir(parents=True, exist_ok=True)

    config_json_path = input_dir / 'config.json'
    sequence_npz_path = input_dir / 'sequences.npz'
    ligand_pdb_path = input_dir / f'{args.ligand_name}.pdb'
    ligand_pdbqt_path = input_dir / f'{args.ligand_name}.pdbqt'

    # Ligand prep ----------------------------------------------------------------
    
    print("Preparing ligand...")
    
    ligand_builder = LigandBuilder(output_dir=input_dir, bondi_volumes=BONDI_VOLUMES, obabel_path=args.obabel_path)
    
    if not ligand_pdbqt_path.exists():

        mol = ligand_builder.smiles_to_mol(args.ligand_smiles)
        mol = ligand_builder.minimize(mol)
        bondi_vol = ligand_builder.calc_bondi_volume(mol)
        ligand_builder.mol_to_pdb(mol, args.ligand_name, ligand_pdb_path)
        ligand_builder.pdb_to_pdbqt(ligand_pdb_path, ligand_pdbqt_path)
        ligand_pdb_path.unlink()
        ligand_builder.tidy_pdbqt(ligand_pdbqt_path, args.ligand_3letter)
        
    lig_info = ligand_builder.analyze_ligand(ligand_pdbqt_path)

    # Design info ----------------------------------------------------------------
    
    print("Parsing design info...")

    design_info = parse_design_info(args.design_info_path)

    # Sequence prep --------------------------------------------------------------

    if not sequence_npz_path.exists():
        
        print("Generating sequences...")
        
        # create sequence library
        sequence_builder = SequenceBuilder(aa_encoder=AA_ENCODER, aa_volumes=AA_VOLUMES)

        seqs = {
            layer: sequence_builder.layer_sequence_generator(
                design_info[layer]['res_info'],
                design_info[layer]['target'],
                design_info[layer]['tolerance'],
            ) for layer in design_info
        }

        full_ids, full_seqs, full_vols, num_seqs = sequence_builder.generate_sequences_and_volumes(seqs)
        np.savez(sequence_npz_path, ids=full_ids, sequences=full_seqs, volumes=full_vols)
        
        print(f"Generated {num_seqs} sequences.")

    # Config prep -----------------------------------------------------------------

    # get starting sequence
    starting_seq = pdb2seq(args.receptor_pdb_path)['A']

    # get a flat list of all design indices
    design_idx = [resnum for k in design_info for resnum in design_info[k]['res_info']]

    # Receptor CA centroid
    scaffold_ca_coords = get_pdbqt_coords(args.receptor_pdb_path, ca_only=True)
    pocket_ca_coords = [xyz for i, xyz in enumerate(scaffold_ca_coords, start=1) if i in design_idx]
    centroid = get_centroid(pocket_ca_coords)

    # Main config object
    config = {
        'receptor_path': str(args.receptor_pdb_path),
        'ligand_path': str(ligand_pdbqt_path),
        'sequences_path': str(sequence_npz_path),
        'output_directory': str(job_dir),
        'FASPR_path': str(args.faspr_path),
        'obabel_path': str(args.obabel_path),
        'num_cpus': args.num_cpus,
        'batch_size': args.batch_size,
        'use_gradient_boosted_trees': args.use_gradient_boosted_trees,
        'gradient_boosted_top_sequences': args.gradient_boosted_top_sequences,
        'gradient_boosted_steps': args.gradient_boosted_steps,
        'save_top_n': args.save_top_n,
        'datestamp': datestamp,
        'timestamp': timestamp,
        'calc_seqs_only': args.calc_seqs_only,
        'cube_side_length': [lig_info["length"] * 1.5] * 3,
        'design_idx': design_idx,
        'starting_seq': starting_seq.lower(),
        'gradient_boosted_step_size': args.gradient_boosted_step_size,
        'pocket_ca_centroid': centroid,
        'num_lig_atoms': lig_info["num_atoms"],
        'parent_volume': sum(AA_VOLUMES[aa] for i, aa in enumerate(starting_seq, start=1) if i in design_idx),
        'total_ligand_volume': sum(layer['target'] for layer in design_info.values()),
        'design': design_info
    }

    # save config file
    with config_json_path.open('w') as f:
        json.dump(config, f, indent=4)

    if config['calc_seqs_only']:
        
        return
    
    # run the main job
    
    docking_results_path = job_dir / 'docking_results.csv'
    
    if not docking_results_path.exists() or args.overwrite:
        
        docking_results_path.unlink(missing_ok=True)
        
        seqs = np.load(config["sequences_path"], mmap_mode='r')

        rassccol = RASSCCoL()

        if config['use_gradient_boosted_trees']:
            
            rassccol.run_active_sampling(seqs, config)
            
        else:
            
            rassccol.batched_jobs(seqs['ids'], seqs, config)
            
        # save the best results
        if config['save_top_n'] is not None and config['save_top_n'] > 0:

            final_df = pd.read_csv(job_dir / 'docking_results.csv').drop_duplicates(subset='id')
            final_df = final_df.sort_values(by='vina_score_norm')
            top_n = final_df.head(config['save_top_n'])["id"].values
            rassccol.batched_jobs(top_n, seqs, config, cleanup=False, save_to_csv=False)
            
    else:
        print(f'{docking_results_path} already exists! Pass `overwrite` true to overwrite.')
        
if __name__ == '__main__':
    main()
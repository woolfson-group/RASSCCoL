
# local modules
from rasscol_src.rasscol_utils import *
from rasscol_src.general_utils import *

# builtin modules
import multiprocessing
import json, sys

import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Run the RASSCoL pipeline for generating binding pocket sequences and initial evaluation with Vina.')
    
    # Adding arguments with flags and their default values
    parser.add_argument('-o', '--output_directory', type=Path, required=True,
                        help='Output directory for results (required)')
    
    parser.add_argument('-r', '--receptor_pdb_path', type=Path, required=True,
                        help='Path to the receptor PDB file (required)')
    
    parser.add_argument('-l', '--ligand_pdbqt_path', type=Path, required=True,
                        help='Path to the ligand PDBQT file (required)')
    
    parser.add_argument('-d', '--design_config_json_path', type=Path, required=True,
                        help='Path to the design configuration JSON file (required)')
    
    parser.add_argument('-g', '--use_gradient_boosted_trees', action='store_true', default=False,
                        help='Use gradient boosted trees (default: False)')
    
    parser.add_argument('-f', '--faspr_path', type=Path, default=Path('/opt/FASPR/FASPR'),
                        help='Path to the FASPR executable (default: /opt/FASPR/FASPR)')
    
    parser.add_argument('-b', '--obabel_path', type=Path, default=Path('/usr/bin/obabel'),
                        help='Path to the Open Babel executable (default: /usr/bin/obabel)')
    
    parser.add_argument('-n', '--num_cpus', type=int, default=8,
                        help='Number of CPUs to use (default: 8)')
    
    parser.add_argument('-s', '--save_top_n', type=int, default=3,
                        help='Save top N results (default: 3)')
    
    parser.add_argument('-c', '--calc_seqs_only', action='store_true', default=False,
                        help='Calculate sequences only (default: False)')
    
    parser.add_argument('-t', '--timeout', type=int, default=180,
                        help='Timeout duration in seconds (default: 180)')
    
    # Parse the arguments
    args = parser.parse_args()

    rasscol = RASSCoL()

    datestamp, timestamp = get_timestamp()

    job_dir = args.output_directory 
    job_dir.mkdir(exist_ok=True, parents=True)
    config_json_path = job_dir / 'config.json'

    with open(args.design_config_json_path, 'r') as f:
        design_config = json.load(f)
        
    starting_seq = pdb2seq(args.receptor_pdb_path)['A']

    # Store all the design residue numbers as a list[int]
    design_idx = [
        int(resnum)
        for layer in design_config
        for resnum in design_config[layer]['res_info'].keys()
    ]

    for _layer in design_config:
        
        # Extract integer residue numbers for the current layer
        _layer_resnums = [int(_resnum) for _resnum in design_config[_layer]['res_info']]
        
        # Get the starting sequence at those residue positions
        _starting_seq_layer = [starting_seq[_resnum - 1] for _resnum in _layer_resnums]  # Adjust for 0-based indexing
        design_config[_layer]['starting_seq'] = _starting_seq_layer
        
        # Calculate the volume of the amino acids in the starting sequence
        _volume = sum(rasscol.aa_vol[_resname] for _resname in _starting_seq_layer)
        design_config[_layer]['volume'] = _volume
        
        # Calculate the target value
        design_config[_layer]['target'] = (_volume + design_config[_layer]['cavity']) - design_config[_layer]['ligand_layer_vol']

    lig_coords = get_pdbqt_coords(args.ligand_pdbqt_path)
    lig_length = get_mol_len(lig_coords)

    config = {
        'run':{
            'receptor_path':str(args.receptor_pdb_path),
            'ligand_path':str(args.ligand_pdbqt_path),
            'output_directory':str(job_dir),
            'FASPR_path':str(args.faspr_path),
            'obabel_path':str(args.obabel_path),
            'num_cpus':args.num_cpus,
            'use_gradient_boosted_trees':args.use_gradient_boosted_trees,
            'save_top_n':args.save_top_n,
            'datestamp': datestamp,
            'timestamp': timestamp,
            'calc_seqs_only': args.calc_seqs_only,
            'cube_side_length': [lig_length*1.5]*3,
            'timeout':args.timeout
        },
        'versions':{
            'python':sys.version.split()[0],
            'vina':rasscol.vina_version 
        },
        'design':design_config
    }

    # write out config for logging
    with config_json_path.open('w') as f:
        json.dump(config, f, indent=4)

    scaffold_ca_coords = get_pdbqt_coords(args.receptor_pdb_path, ca_only=True)
    pocket_ca_coords = [xyz for i, xyz in enumerate(scaffold_ca_coords, start=1) if i in design_idx]
    config['run']['pocket_ca_centroid'] = get_centroid(pocket_ca_coords)
    num_lig_atoms = len(lig_coords)

    # Generate the sequence generators dictionary
    seqs = {
        layer: rasscol.layer_sequence_generator(
            design_config[layer]['res_info'],
            design_config[layer]['target'],
            design_config[layer]['tolerance']
        ) for layer in design_config
    }

    # Define a wrapper function to run the sequence generator
    def run_sequence_generator(starting_seq, seqs, design_idx, return_dict):
        return_dict["result"] = rasscol.sequence_generator(starting_seq, seqs, design_idx)

    # Create a manager to handle shared data
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    # Create the process
    process = multiprocessing.Process(target=run_sequence_generator, args=(starting_seq, seqs, design_idx, return_dict))

    # Start the process and set a timeout
    process.start()
    process.join(timeout=config['run']['timeout'])

    # Check if the process is still alive
    if process.is_alive():
        process.terminate()
        print("The function call timed out!")
    else:
        seq_dict = return_dict["result"]

    print(f'Sequences generated: {len(seq_dict)}')

    if config['run']['calc_seqs_only']:
        pass

    elif config['run']['use_gradient_boosted_trees']:
        pass

    else:
        rasscol.run_parallel(starting_seq, design_idx, seq_dict, config['run']['pocket_ca_centroid'], num_lig_atoms, job_dir, args.receptor_pdb_path, args.ligand_pdbqt_path, config, design_config)
        
        results_csv = job_dir / 'RASSCoL_results.csv'
        
        # Read CSV data and store it in a list of dictionaries
        with results_csv.open() as f:
            reader = csv.DictReader(f)
            results = [row for row in reader]

        # Sort the results based on the 'vina_score_norm' column and get the top N
        data = sorted(results, key=lambda x: float(x['vina_score_norm']))

        # Write the list of dictionaries to a CSV file
        with open(results_csv, mode='w', newline='') as csvfile:
            # Create a csv.DictWriter object
            writer = csv.DictWriter(csvfile, fieldnames=data[0].keys())
            
            # Write the header (field names)
            writer.writeheader()
            
            # Write the data
            writer.writerows(data)
            
        rasscol.save_structures(config, job_dir / 'RASSCoL_results.csv')

if __name__ == '__main__':
    main()
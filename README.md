# Code for running RASSCoL without Random Forest

*This README is intended for Tombstone2 users in the Woolfson group only.*

## Installation

This code requires a small number of external libraries (see `env/RASSCoL_no_RF_env.yml`) as was tested using Python 3.9.18.

To set up the environment, run:

```bash
conda env create -f env/RASSCoL_no_RF_env.yml
```

## Running

The minimum required options are:

1. Receptor PDB path (`-r, --receptor_pdb_path`)
2. Ligand PDBQT path (`-l, --ligand_pdbqt_path`)
3. Design config JSON path (`-d, --design_config_json_path`)
4. Directory to save the output (`-o, --output_directory`)

First I suggest you run the example to test installation (~ 5 minutes):

```bash
python run_RASSCoL.py \
    -r ./example/scapCC4_NRD/scapCC4.pdb \
    -l ./example/scapCC4_NRD/NileRed.pdbqt \
    -d ./example/scapCC4_NRD/design_config.json \ 
    -o ./example/scapCC4_NRD/
```

There are several options you can manipulate for RASSCoL, see [Running options](#running-options).

For help setting up the ligand and the design configuration, see `./01_RASSCoL_ligand_prep.ipynb` and `./02_RASSCoL_design_prep.ipynb` respectively.

### Running options

For see below for full list, or use `python run_RASSCoL.py - h`.

```text
usage: run_RASSCoL.py [-h] -o OUTPUT_DIRECTORY -r RECEPTOR_PDB_PATH -l LIGAND_PDBQT_PATH -d DESIGN_CONFIG_JSON_PATH [-g] [-f FASPR_PATH] [-b OBABEL_PATH] [-n NUM_CPUS] [-s SAVE_TOP_N] [-c]
                      [-t TIMEOUT]

Run the RASSCoL pipeline for generating binding pocket sequences and initial evaluation with Vina.

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT_DIRECTORY, --output_directory OUTPUT_DIRECTORY
                        Output directory for results (required)
  -r RECEPTOR_PDB_PATH, --receptor_pdb_path RECEPTOR_PDB_PATH
                        Path to the receptor PDB file (required)
  -l LIGAND_PDBQT_PATH, --ligand_pdbqt_path LIGAND_PDBQT_PATH
                        Path to the ligand PDBQT file (required)
  -d DESIGN_CONFIG_JSON_PATH, --design_config_json_path DESIGN_CONFIG_JSON_PATH
                        Path to the design configuration JSON file (required)
  -g, --use_gradient_boosted_trees
                        Use gradient boosted trees (default: False)
  -f FASPR_PATH, --faspr_path FASPR_PATH
                        Path to the FASPR executable (default: /opt/FASPR/FASPR)
  -b OBABEL_PATH, --obabel_path OBABEL_PATH
                        Path to the Open Babel executable (default: /usr/bin/obabel)
  -n NUM_CPUS, --num_cpus NUM_CPUS
                        Number of CPUs to use (default: 8)
  -s SAVE_TOP_N, --save_top_n SAVE_TOP_N
                        Save top N results (default: 3)
  -c, --calc_seqs_only  Calculate sequences only (default: False)
  -t TIMEOUT, --timeout TIMEOUT
                        Timeout duration in seconds (default: 180)
```

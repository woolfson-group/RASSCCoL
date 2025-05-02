# Code for running RASSCCoL

*This README is intended for Tombstone2 users in the Woolfson group only.*

Active Sampling using Gradient Boosted trees is an experimental feature, and is under active development.

## Installation

This code requires a small number of external libraries (see `env/RASSCCoL_env.yml`) and was tested using Python 3.9.18.

To set up the environment, run:

```bash
conda env create -f env/RASSCCoL_env.yml
conda activate RASSCCoL_env
```

* To prepare files for AutoDock Vina, Open Babel is needed. [See here for installation](https://openbabel.org/docs/Installation/install.html#install-binaries)

* For fast repacking of sequences, FASPR is needed. [See here for installation](https://github.com/tommyhuangthu/FASPR?tab=readme-ov-file#installation)

> Both of these executables will need to be specified in the CLI when running.

## Running

The minimum required options are:

1. Receptor PDB path (`--receptor_pdb_path`)
2. Ligand SMILES string (`--ligand_smiles`)
3. Design info TXT path (`--design_info_path`)
4. Directory to save the output (`--output_directory`)
5. FASPR binary path (`--faspr_path`)
6. OpenBabel binary path (`--obabel_path`)

First I suggest you run the examples to test installation, navigate to the RASSCCoL directory (where this README is located) and run the commands below.

> You will need to manually overide the `faspr_path` and `obabel_path` arguments.

No random forest active sampling test (~  2 minutes):

```bash
bash helper_scripts/run_no_RF_test.sh
```

Random forest active sampling test (~ 15 minutes):

```bash
bash helper_scripts/run_RF_test.sh
```
There are several options you can manipulate for RASSCCoL, see [Running options](#running-options).

## Design info

RASSCCoL splits proteins/ligands up into layers to match shape. In order to do this the designer needs to pick out layers in their protein. This information is passed as a text file. An example is shown below:

```text
    1 0 20 110
    19 AILV
    44 AILV
    111 AILV

    2 0 20 110
    16 AILV
    84 AILV
    114 AILV

    3 0 20 110
    12 AILV
    51 AILV
    118 AILV
```

The expected format is:

```text
    <layer> <cavity> <tolerance> <lig_layer_vol>
    <resnum> <aaset>
    ...
```

With layers separated by empty lines.

>**Note**: the resnum corresponds to the 1-based index of residues in the PDB file

### Bypassing the layer approach

Layers can be bypassed by simply putting all of the intended pocket residues into one layer. Whilst this does work, we found splitting ligands into layers reduced the sequence space and therefore reduced compute. An example of this can be seen below (note the ligand layer volume is now the total ligand volume):

```text
    1 0 20 330
    19 AILV
    44 AILV
    111 AILV
    16 AILV
    84 AILV
    114 AILV
    12 AILV
    51 AILV
    118 AILV
```

## Known limitations

- Currently the code only works on monomeric proteins.

### Running options

For see below for full list, or use `python run_RASSCCoL.py - h`.

```text
usage: run_RASSCCoL.py [-h] --output_directory OUTPUT_DIRECTORY --receptor_pdb_path RECEPTOR_PDB_PATH --ligand_smiles LIGAND_SMILES --design_info_path DESIGN_INFO_PATH [--ligand_name LIGAND_NAME] [--ligand_3letter LIGAND_3LETTER] [--use_gradient_boosted_trees]
                       [--gradient_boosted_top_sequences GRADIENT_BOOSTED_TOP_SEQUENCES] [--gradient_boosted_steps GRADIENT_BOOSTED_STEPS] [--gradient_boosted_step_size GRADIENT_BOOSTED_STEP_SIZE] [--faspr_path FASPR_PATH] [--obabel_path OBABEL_PATH] [--batch_size BATCH_SIZE] [--num_cpus NUM_CPUS]
                       [--save_top_n SAVE_TOP_N] [--calc_seqs_only] [--overwrite]

Run RASSCCoL with optional gradient boosting.

optional arguments:
  -h, --help            show this help message and exit
  --output_directory OUTPUT_DIRECTORY
                        Output directory for results.
  --receptor_pdb_path RECEPTOR_PDB_PATH
                        Path to the receptor PDB file.
  --ligand_smiles LIGAND_SMILES
                        Ligand SMILES string.
  --design_info_path DESIGN_INFO_PATH
                        Path to the design info file.
  --ligand_name LIGAND_NAME
                        Name for the ligand (default: ligand).
  --ligand_3letter LIGAND_3LETTER
                        3-letter code for the ligand (default: LIG).
  --use_gradient_boosted_trees
                        Use gradient boosted trees for scoring.
  --gradient_boosted_top_sequences GRADIENT_BOOSTED_TOP_SEQUENCES
                        Number of top sequences to keep.
  --gradient_boosted_steps GRADIENT_BOOSTED_STEPS
                        Number of boosting steps.
  --gradient_boosted_step_size GRADIENT_BOOSTED_STEP_SIZE
                        Number of sequences per boosting step.
  --faspr_path FASPR_PATH
                        Path to FASPR binary.
  --obabel_path OBABEL_PATH
                        Path to Open Babel binary.
  --batch_size BATCH_SIZE
                        Batch size for processing.
  --num_cpus NUM_CPUS   Number of CPU threads to use.
  --save_top_n SAVE_TOP_N
                        Number of top-scoring docking PBDQT files to save.
  --calc_seqs_only      Only calculate sequences, skip docking.
  --overwrite           Allow overwriting existing outputs.
```

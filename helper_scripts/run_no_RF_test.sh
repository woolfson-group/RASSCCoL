# this script is hardcoded to be run in the RASSCCoL directory
python ./run_RASSCCoL.py \
    --output_directory example/no_RF_test \
    --receptor_pdb_path example/resources/scapCC4.pdb \
    --ligand_smiles "CCN(CC)C1=CC2=C(C=C1)N=C3C4=CC=CC=C4C(=O)C=C3O2" \
    --design_info_path example/resources/no_RF_desing_info.txt \
    --faspr_path /opt/FASPR/FASPR \
    --obabel_path /usr/bin/obabel \
    --overwrite

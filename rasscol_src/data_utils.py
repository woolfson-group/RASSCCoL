
BONDI_VOLUMES = {
    'H':7.24, 'C':20.58, 'N':15.6, 'O':14.71, 'F': 13.31, 
    'Cl':22.54, 'Br':26.52, 'I':32.52, 'P':24.43, 'S': 24.43, 
    'As':26.52, 'B':40.48, 'Si':38.79, 'Se':28.73, 'Te':36.62
}

# AA letters mapped to integer codes
AA_ENCODER = {
    'G': 0, 'A': 1, 'V': 2, 'L': 3, 'I': 4, 'M': 5, 'F': 6,
    'Y': 7, 'W': 8, 'C': 9, 'S': 10, 'T': 11, 'N': 12, 'Q': 13,
    'D': 14, 'E': 15, 'K': 16, 'R': 17, 'H': 18, 'P': 19
}

# The reverse mapping of AA_ENCODER
AA_DECODER = {v:k for k, v in AA_ENCODER.items()}

# side chain volumes in A^3
AA_VOLUMES = {
    'G': 0, 'A': 17, 'V': 52, 'L': 69, 'I': 69, 'M': 70, 'F': 102,
    'Y': 111, 'W': 133, 'C': 21, 'S': 26, 'T': 43, 'N': 52, 'Q': 69, 
    'D': 50, 'E': 67, 'K': 80, 'R': 100, 'H': 85, 'P': 49
} 

AA_3_TO_1 = {
    'Ala': 'A', 'Val': 'V', 'Met': 'M',
    'Phe': 'F', 'Tyr': 'Y', 'Gln': 'Q',
    'Thr': 'T', 'Gly': 'G', 'Leu': 'L',
    'Ile': 'I', 'Pro': 'P', 'Ser': 'S',
    'Cys': 'C', 'Trp': 'W', 'Asp': 'D',
    'Asn': 'N', 'Glu': 'E', 'Lys': 'K',
    'Arg': 'R', 'His': 'H'
}

AA_1_TO_3 = {v:k for k, v in AA_3_TO_1.items()} 

AA_ALPHABET = list(AA_1_TO_3.keys())
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Produces a FASTA file for the top N designs for structure prediction.')
    
    # Adding arguments with flags and their default values
    parser.add_argument('-i', '--input_csv', type=Path, required=True,
                        help='Path to CSV containing RASSCoL results (required)')
    
    parser.add_argument('-o', '--output_csvpath', type=Path, default=Path('./RASSCoL_sequences.fa'),
                        help='File path for FASTA file (default: ./RASSCoL_sequences.fa)')
    
    parser.add_argument('-n', '--top_N', type=int, default=3,
                        help='Top ranks to extract for FASTA (default: 3)')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # get the CSV data as a string
    csv_text = args.input_csv.read_text()
    
    # split the strings into lines skipping the header line and the tailing row
    csv_lines = csv_text.split('\n')[1:-1]
    
    # check if number of sequence requested is suitable
    if len(csv_lines) < args.top_N:
        raise KeyError("Number of sequences requested exceeds sequences generated.")
    
    # initialise empty string to write fasta info
    fasta_str = ''
    
    # iterate over each line 
    for line in csv_lines[:args.top_N]:
        
        # split the lines into a list of the comma separate values
        values = line.split(',')
        
        # assign the sequence id
        id = values[0]
        
        # assign the sequence
        sequence = values[1]
        
        # append the new fasta information to fasta string
        fasta_str += f'>{id}\n{sequence}\n'
        
    # write out the fasta string to provided path
    args.output_csvpath.write_text(fasta_str)
    
if __name__ == '__main__':
    main()
import torch
import utils
import pandas as pd

def encode_proteins(proteins):
    amino_acids_dic = utils.get_amino_acids_dic()
    sequences = proteins[1].tolist()
    encoded_sequences = []
    #encoding all the sequences
    for seq in sequences:
        encoded_seq = []

        for char in seq:
            encoded_seq.append(amino_acids_dic[char])

        encoded_sequences.append(encoded_seq)

    return pd.dataframe({"Label" : proteins[0], "Sequence" : encoded_sequences, "TF" : proteins[2]})

if __name__ == '__main__':
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loading the data
    proteins = encode_proteins(utils.load_data())
    print(proteins)
import pandas as pd
import numpy as np
import torch
import csv


amino_acids = ['A', 'R', 'N', 'D', 'B', 'C', 'E', 'Q', 'Z', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
                   'Y', 'V', '*']

# -------------------------------------
# Load the Proteins from the data file
# Returns : an Array of TF and not TF
# -------------------------------------
def load_data():
    # Load Transformation factors
    TF = pd.read_csv("data/TF_seqs.csv", header=None)
    names = pd.DataFrame(TF.iloc[::2, :])
    names = names.reset_index(drop=True)
    mol = pd.DataFrame(TF.iloc[1::2, :])
    mol = mol.reset_index(drop=True)
    isTF = pd.DataFrame(np.ones((3230, 1), dtype=bool))
    TF = pd.concat([names, mol, isTF], axis=1, ignore_index=True)
    TF = TF[TF[0].notna()]
    TF.columns = ['id', 'seq', 'isTf']
    TF['id'] = TF['id'].str[1:]

    # Load other non-TF protein
    Not_TF = pd.read_csv("data/random_seqs.csv", header=None)
    names = pd.DataFrame(Not_TF.iloc[::2, :])
    names = names.reset_index(drop=True)
    mol = pd.DataFrame(Not_TF.iloc[1::2, :])
    mol = mol.reset_index(drop=True)
    isTF = pd.DataFrame(np.zeros((3230, 1), dtype=bool))
    Not_TF = pd.concat([names, mol, isTF], axis=1, ignore_index=True)
    Not_TF = Not_TF[Not_TF[0].notna()]
    Not_TF.columns = ['id', 'seq', 'isTf']
    Not_TF['id'] = Not_TF['id'].str[1:]

    # Match each id to family of TF
    family = pd.read_csv("data/families.txt", skiprows=1, sep="\t", names=['id', 'Families'], header=None)
    family = family.groupby(['Families'])['id'].apply(','.join).reset_index()

    Proteins = pd.concat([TF, Not_TF]).reset_index(drop=True)
    Proteins.to_csv(r'data/BinaryDataset.csv', index=False)

    TF['Family'] = 'Unknown'

    family = family.drop([65])

    # Putting ARID together
    family = family.replace(to_replace="ARID/BRIGHT; RFX", value="ARID/BRIGHT")

    # Putting C2H2 together
    family = family.replace(to_replace="C2H2 ZF; AT hook", value="C2H2 ZF")
    family = family.replace(to_replace="C2H2 ZF; BED ZF", value="C2H2 ZF")
    family = family.replace(to_replace="C2H2 ZF; Myb/SANT", value="C2H2 ZF")

    # Putting Homeodomain together
    family = family.replace(to_replace="Homeodomain; POU", value="Homeodomain")
    family = family.replace(to_replace="Homeodomain; Paired box", value="Homeodomain")
    family = family.replace(to_replace="CUT; Homeodomain", value="Homeodomain")
    family = family.replace(to_replace="C2H2 ZF; Homeodomain", value="Homeodomain")

    # Putting MBD together
    family = family.replace(to_replace="MBD; AT hook", value="MBD")
    family = family.replace(to_replace="MBD; CxxC ZF", value="MBD")

    # Putting Ets together
    family = family.replace(to_replace="Ets; AT hook", value="Ets")

    # Putting CxxC together
    family = family.replace(to_replace="CxxC; AT hook", value="CxxC")

    family['id'].value_counts()

    for i in range(Proteins.shape[0]):
        for row in range(family.shape[0]):
            if TF['id'][i] in family.iloc[row].id:
                TF['Family'][i] = family.iloc[row].Families


    counts = TF['Family'].value_counts()
    TF['Family'] = np.where(TF['Family'].isin(counts.index[counts < 10]), 'Other', TF['Family'])

    TF[TF['Family'].str.match('ARID/BRIGHT; RFX')].Family = 'ARID'

    TF.to_csv(r'data/FamilyDataset.csv', index=False)

    return Proteins


# -----------------------------------------------
# Convert each letter in sequence to ascii number
# Returns : A converted X training array and Y array
# -----------------------------------------------
def get_ascii(data):
    train_tensor = torch.zeros(len(data), data[1].str.len().max())
    for i in range(len(data)):
        for j in range(len(data.iloc[i, 1])):
            number = ord(data.iloc[i, 1][j])
            train_tensor[i][j] = number

    return train_tensor, torch.FloatTensor(data[2]).reshape(-1, 1)


# -------------------------------------------------
# Split the data of proteins to have TF and non TF
# Returns : a training set X
# -------------------------------------------------
def split_data(data, n):
    part = int(n / 2)
    train = pd.concat([data[:part], data[-part:]], ignore_index=True)
    return train

# -------------------------------------------------
# Binarize an int number on a specified (or not) amount of bits
# Return : The binarized version
# -------------------------------------------------
def binarize(v, nb=0):
    if nb == 0:
        return bin(v)[2:]
    else:
        return np.binary_repr(v, width=nb)

# -------------------------------------------------
# Throw the user a yes or no question
# Return True if yes, False if no
# -------------------------------------------------
def yes_or_no_question(question, default_no=True):
    choices = ' [y/N]: ' if default_no else ' [Y/n]: '
    default_answer = 'n' if default_no else 'y'
    reply = str(input(question + choices)).lower().strip() or default_answer
    if reply[0] == 'y':
        return True
    if reply[0] == 'n':
        return False
    else:
        return False if default_no else True

# -------------------------------------------------
# Get a dictionary of the encoded amino acids
# Return the dictionary
# -------------------------------------------------
def get_amino_acids_dic(padding=True):
    dic = {}
    aa_list = amino_acids

    if padding:
        aa_list.insert(0, ' ')

    for idx,aa in enumerate(aa_list):
        dic[aa] = idx
    return dic

# -------------------------------------------------
# Encode the sequences as a list of integers (based on the amino acids dict)
# Return the modified DataFrame
# -------------------------------------------------
def encode_proteins(proteins, padding=True):
    amino_acids_dic = get_amino_acids_dic(padding)
    sequences = proteins["seq"].tolist()

    maxlen = len(max(sequences, key=len))


    encoded_sequences = []
    # encoding all the sequences
    for seq in sequences:
        encoded_seq = []

        # Padding the sequences with white spaces
        if padding:
            while len(seq) < maxlen:
                seq += ' '

        for char in seq:

            # U has the same function as C
            if char == 'U':
                # We append lists because we will could change the encoding to a hot one code for instance
                encoded_seq.append([amino_acids_dic['C']])
            else:
                encoded_seq.append([amino_acids_dic[char]])

        encoded_sequences.append(encoded_seq)

    data = {"Label" : proteins["id"], "Sequence" : encoded_sequences, "TF" : proteins["isTf"]}
    df = pd.DataFrame(data)
    df.to_csv(path_or_buf="data/EncodedDataset.csv", index=False)
    return df

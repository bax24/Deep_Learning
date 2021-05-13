import pandas as pd
import numpy as np
import torch


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

    # Load other non-TF protein
    Not_TF = pd.read_csv("data/random_seqs.csv", header=None)
    names = pd.DataFrame(Not_TF.iloc[::2, :])
    names = names.reset_index(drop=True)
    mol = pd.DataFrame(Not_TF.iloc[1::2, :])
    mol = mol.reset_index(drop=True)
    isTF = pd.DataFrame(np.zeros((3230, 1), dtype=bool))
    Not_TF = pd.concat([names, mol, isTF], axis=1, ignore_index=True)
    Not_TF = Not_TF[Not_TF[0].notna()]

    return pd.concat([TF, Not_TF])


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
    part = int(n/2)
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


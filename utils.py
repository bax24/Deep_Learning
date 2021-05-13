import pandas as pd
import numpy as np
import torch
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

import numpy as np
import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


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
    train = pd.concat([data[:n / 2], data[-n / 2:]], ignore_index=True)
    return train


# -------------------------------------
# Generic class of Neural Net
# -------------------------------------
class NeuralNet(nn.Module):
    def __init__(self, in_size, hidden_units, out_size):
        super(NeuralNet, self).__init__()

        self.net = nn.Sequential(nn.Linear(in_size, hidden_units), nn.ReLU(),
                                 nn.Linear(hidden_units, hidden_units), nn.ReLU(),
                                 nn.Linear(hidden_units, out_size), nn.Sigmoid())

    # We have also to define what is the forward of this module:
    def forward(self, x):
        out = self.net(x)
        return out


# -------------------------------------
# Defines the Loss function
# -------------------------------------
def loss_func(y_hat, y):
    return nn.BCELoss()(y_hat, y)


# ------------------------------------------------------
# Compute the frequencies of each amino acid in sequence
# Returns : Array of N by 22 training samples and output array
# ------------------------------------------------------
def get_freq(df):
    df = split
    amino_acids = ['A', 'R', 'N', 'D', 'B', 'C', 'E', 'Q', 'Z', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
                   'Y', 'V', '*']
    amino_freqs = []
    for row in df.index:
        amino_freq = {amino: 0 for amino in amino_acids}
        for amino in df.iloc[row, 1]:
            amino_freq[amino] += 1
        amino_freq.popitem()
        amino_freqs.append((list(amino_freq.values())))

    return torch.Tensor(amino_freqs), torch.FloatTensor(df[2]).reshape(-1, 1)


if __name__ == '__main__':
    # Debug
    disp = False

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if disp:
        print('The device used is : ' + str(device))

    # Loading the data
    Proteins = load_data()
    if disp:
        print(Proteins.head())

    # How many samples to train model ?
    N = 200
    split = split_data(Proteins, N)

    # How to fromat the data ?
    X_train, Y_train = get_freq(split)

    # Hyper-parameters
    input_size = X_train.shape[1]
    output_size = 1
    hidden_size = 50
    learning_rate = 0.01
    epochs = 100

    # Longest sequence in sample
    if disp:
        print('longest sequence is :' + str(split[1].str.len().max()))

    # -------------------------------------------------
    # Second attempt neural net wit frequencies of aa
    # -------------------------------------------------

    net = NeuralNet(input_size, hidden_size, output_size)

    # Defining the optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    train_loss = []
    train_accuracy = []

    # TRAINING LOOP
    for epoch in range(epochs):
        X_train_t = torch.FloatTensor(X_train)

        # Forward pass
        y_hat = net(X_train_t)

        # Computing the loss function
        loss = loss_func(y_hat, Y_train)

        # Accumulate partial derivatives of loss wrt paramters
        loss.backward()

        # Step in opposite direction of the gradient
        optimizer.step()

        # Clean the gradients mmmh dirty gradients
        optimizer.zero_grad()

        # Assign 0 (not TF) or 1 (TF) to output prob
        y_hat_class = np.where(y_hat.detach().numpy() < 0.5, 0, 1)

        # Compute accuracy
        accuracy = np.sum(Y_train.detach().numpy() == y_hat_class) / len(Y_train)

        train_accuracy.append(accuracy)
        train_loss.append(loss.item())

    # PLOTTING THE LOSS AND THE ACCURACY
    plt.figure(figsize=(13, 5))
    plt.subplot(1, 2, 1)
    plt.title('Training Loss')
    plt.plot(train_loss)
    plt.xlabel('Iterations')
    plt.ylabel('Loss (Binary Cross Entropy)')

    plt.subplot(1, 2, 2)
    plt.title('Training Accuracy')
    plt.plot(train_accuracy)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.show()

    # --------------------------------
    # First attempt neural net
    # --------------------------------
    # X_train, Y_train = get_training(split)
    # y_train = y_train.type(torch.LongTensor)
    epochs = 10
    model = NeuralNet(X_train.shape[1], 1000, 1)

    # Define the optimize
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    # Define the loss
    loss = nn.BCELoss()

    net = nn.Sequential(nn.Linear(X_train.shape[1], 500), nn.ReLU(),
                        nn.Linear(500, 50), nn.ReLU(),
                        nn.Linear(50, 1), nn.Sigmoid())
    optimizer = optim.Adam(net.parameters(), lr=.1)

    train_loss = []
    train_accuracy = []

    train_loss = []  # where we keep track of the loss
    train_accuracy = []  # where we keep track of the accuracy of the model
    iters = 100  # number of training iterations

    # Y_train_t = torch.FloatTensor(Y_train).reshape(-1, 1)  # re-arrange the data to an appropriate tensor

    for i in range(iters):
        X_train_t = torch.FloatTensor(X_train)
        y_hat = net(X_train_t)  # forward pass

        loss = loss_func(y_hat, Y_train)  # compute the loss
        loss.backward()  # obtain the gradients with respect to the loss
        optimizer.step()  # perform one step of gradient descent
        optimizer.zero_grad()  # reset the gradients to 0

        y_hat_class = np.where(y_hat.detach().numpy() < 0.5, 0,
                               1)  # we assign an appropriate label based on the network's prediction
        accuracy = np.sum(Y_train.reshape(-1, 1) == y_hat_class) / len(Y_train)  # compute final accuracy

        train_accuracy.append(accuracy)
        train_loss.append(loss.item())

    for epoch in range(epochs):
        # Forward pass
        y_hat = model(X_train)

        # Computing the Loss
        l = loss(y_hat, y_train)

        # Cleaning the gradients (dirty gradients mmmh)
        optimizer.zero_grad()

        # Accumulate the partial derivatives of l wrt params
        l.backward()

        # Optimize
        optimizer.step()

        y_hat_class = np.where(y_hat.detach().numpy() < 0.5, 0,
                               1)  # we assign an appropriate label based on the network's prediction
        accuracy = np.sum(y_train.reshape(-1, 1) == y_hat_class) / len(y_train)  # compute final accuracy

        print(f'Epoch {epoch + 1}, train loss: {l.item()}')

    # Longest is 14508

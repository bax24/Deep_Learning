import numpy as np
import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import utils as f


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
    disp = True

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if disp:
        print('The device used is : ' + str(device))

    # Loading the data
    Proteins = f.load_data()
    if disp:
        print(Proteins.head())

    # How many samples to train model ?
    N = 200
    split = f.split_data(Proteins, N)

    # How to format the data ?
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

        # Display loss
        if disp:
            print(f'Epoch {epoch + 1}, train loss: {loss.item()}')

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



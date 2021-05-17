import torch
import torch.nn as nn
import torch.optim as optim
import utils
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# RNN model (pick between simple, LSTM or GRU)
class MyRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(MyRNN, self).__init__()

        # Number of hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.n_layers = n_layers

        # RNN layer
        # self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)

        # GRU layer
        # self.gru = nn.GRU(input_size, hidden_dim, n_layers, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        batch_size = x.size(0)

        # Initializing hidden state for first input
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)

        # RNN
        # out, hidden = self.rnn(x, hidden)

        # LSTM
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(device)
        out, hidden = self.lstm(x, (hidden,c0))

        # GRU
        # out, hidden = self.gru(x, hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out[:, -1, :]
        out = self.fc(out)

        # Sigmoid for binary classification (in order to have 0 to 1 values -> BCE)
        return torch.sigmoid(out)


# -------------------------------------
# Defines the Loss function
# -------------------------------------
def loss_func(y_hat, y):
    return nn.BCELoss()(y_hat, y)


if __name__ == '__main__':

    print("Encoding the proteins ...")
    start = time.time()
    # Loading the data
    proteins = utils.encode_proteins(pd.read_csv("data/LightDataset.csv"))
    end = time.time()
    print("Done in {} seconds !\n".format(int(end - start)))

    print("Formatting LS and TS ...")
    start = time.time()
    # Getting the LS and TS in the appropriate tensor format.
    train, test = train_test_split(proteins, test_size=0.3)

    x_train = torch.Tensor(train["Sequence"].tolist())
    y_train = torch.FloatTensor(train["TF"].tolist()).reshape(-1, 1)
    x_test = torch.Tensor(test["Sequence"].tolist())
    y_test = torch.FloatTensor(test["TF"].tolist()).reshape(-1, 1)

    end = time.time()
    print("LS shape :\n \tInput : {}\n \tOutput : {}".format(x_train.shape, y_train.shape))
    print("TS shape :\n \tInput : {}\n \tOutput : {}".format(x_test.shape, y_test.shape))
    print("Done in {} seconds !\n".format(int(end - start)))

    # Hyper-parameters
    input_size = 1
    output_size = 1
    hidden_size = 25
    learning_rate = 0.005
    epochs = 20

    # Instantiate the model with hyper-parameters
    my_rnn = MyRNN(input_size=input_size, output_size=output_size, hidden_dim=hidden_size, n_layers=1)
    # We'll also set the model to the device that we defined earlier (default is CPU)
    my_rnn.to(device)

    # Defining the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(my_rnn.parameters(), lr=learning_rate)

    # Training Run
    print("Training the model ...")
    start = time.time()
    x_train = x_train.to(device)
    for epoch in range(1, epochs + 1):

        # Not to get stuck in a minima
        # idx = torch.randperm(x_train.shape[0])
        # x_train = x_train[idx].view(x_train.size())

        output = my_rnn(x_train)
        output = output.to(device)

        y_train = y_train.to(device)
        loss = loss_func(output, y_train)

        # Clears existing gradients from previous epoch
        optimizer.zero_grad()
        # Does backpropagation and calculates gradients
        loss.backward()
        # Updates the weights accordingly
        optimizer.step()

        if epoch % 1 == 0:
            print('Epoch: {}/{}.............'.format(epoch, epochs), end=' ')
            print("Loss: {:.4f}".format(loss.item()))

    end = time.time()
    print("Done in {} seconds !\n".format(int(end - start)))

    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    print("Testing the trained model ...")
    start = time.time()
    with torch.no_grad():
        test_output = my_rnn(x_test)
        y_hat_test_class = np.where(test_output.detach().numpy() < 0.5, 0, 1)
        accuracy_test = np.sum(y_test.detach().numpy() == y_hat_test_class) / len(y_test)
        print("Accuracy = {}".format(accuracy_test))
    end = time.time()
    print("Done in {} seconds !\n".format(int(end - start)))

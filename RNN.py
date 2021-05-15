import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import utils
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split

class MyRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(MyRNN, self).__init__()

        # Number of hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.n_layers = n_layers

        #RNN layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        batch_size = x.size(0)

        # Initializing hidden state for first input
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out[:, -1, :]
        out = self.fc(out)

        return F.softmax(out, dim=0)


# -------------------------------------
# Defines the Loss function
# -------------------------------------
def loss_func(y_hat, y):
    return nn.BCELoss()(y_hat, y)

if __name__ == '__main__':
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Encoding the proteins ...")
    start = time.time()
    # Loading the data
    proteins = utils.encode_proteins(pd.read_csv("data/BinaryDataset.csv"))
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
    hidden_size = 10
    learning_rate = 0.01
    epochs = 50

    # Instantiate the model with hyperparameters
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
        # Clears existing gradients from previous epoch
        optimizer.zero_grad()

        output = my_rnn(x_train)
        output = output.to(device)

        y_train = y_train.to(device)
        # print("output shape : {}\n y_train shape : {}".format(output.shape, y_train.shape))

        # RuntimeError: all elements of input should be between 0 and 1
        # c'est chelou pcq on est pas sensé avoir de valeur pas entre 0 et 1
        try:
            loss = loss_func(output, y_train)
        except RuntimeError:
            print("EPOCH {} \nRuntimeError: all elements of input should be between 0 and 1".format(epoch))
            for i in range(int(output.shape[0])):
                op = output[i, 0]
                if not op >= 0 and op <= 1:
                    # print("Output[{}] = {}".format(i, op))
                    break

        # Does backpropagation and calculates gradients
        loss.backward()
        # Updates the weights accordingly
        optimizer.step()

        if epoch % 10 == 0:
            print('Epoch: {}/{}.............'.format(epoch, epochs), end=' ')
            print("Loss: {:.4f}".format(loss.item()))

    end = time.time()
    print("Done in {} seconds !\n".format(int(end - start)))

import torch
import torch.nn as nn
import torch.optim as optim
import utils
import pandas as pd
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
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)

        return out


# -------------------------------------
# Defines the Loss function
# -------------------------------------
def loss_func(y_hat, y):
    return nn.BCELoss()(y_hat, y)



if __name__ == '__main__':
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loading the data
    proteins, maxlen = utils.encode_proteins(utils.load_data())
    #print(proteins)

    # Getting the LS and TS in the appropriate tensor format.
    train, test = train_test_split(proteins, test_size=0.3)

    x_train = torch.Tensor(train["Sequence"].tolist())
    y_train = torch.FloatTensor(train["TF"].tolist()).reshape(-1, 1)
    x_test = torch.Tensor(test["Sequence"].tolist())
    y_test = torch.FloatTensor(test["TF"].tolist()).reshape(-1, 1)

    # Hyper-parameters
    input_size = maxlen
    # binary classification at first
    output_size = 1
    hidden_size = 10
    learning_rate = 0.01
    epochs = 1000

    # Instantiate the model with hyperparameters
    my_rnn = MyRNN(input_size=input_size, output_size=output_size, hidden_units=hidden_size, n_layers=1)
    # We'll also set the model to the device that we defined earlier (default is CPU)
    my_rnn.to(device)

    # Defining the optimizer
    optimizer = optim.Adam(my_rnn.parameters(), lr=learning_rate)
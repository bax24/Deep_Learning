import torch
import torch.nn as nn
import utils
import pandas as pd

class MyRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_units, n_layers):
        super(MyRNN, self).__init__()

        #RNN layer with a fully connected output layer
        self.rnn = nn.Sequential(nn.RNN(input_size, hidden_units, n_layers, nonlinearity='relu'), nn.ReLU(),
                                 nn.linear(hidden_units, output_size))

    def forward(self, x):
        out = self.net(x)
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
    proteins = utils.encode_proteins(utils.load_data())
    print(proteins)

    train, test = train_test_split(proteins, test_size=0.3)

    # Hyper-parameters
    input_size = X_train.shape[1]
    output_size = 1
    hidden_size = 50
    learning_rate = 0.01
    epochs = 10000
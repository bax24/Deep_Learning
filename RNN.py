import torch
import torch.nn as nn
import torch.optim as optim
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
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)

        return out


# -------------------------------------
# Defines the Loss function
# -------------------------------------
def loss_func(y_hat, y):
    return nn.BCELoss()(y_hat, y)


def one_hot_encode(sequence, dict_size, seq_len, batch_size):
    # Creating a multi-dimensional array of zeros with the desired output shape
    features = np.zeros((batch_size, seq_len, dict_size), dtype=np.float32)

    # Replacing the 0 at the relevant character index with a 1 to represent that character
    for i in range(batch_size):
        for u in range(seq_len):
            features[i, u, sequence[i][u]] = 1
    return features

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
    input_size = int(x_test.shape[1])
    output_size = 1
    hidden_size = 10
    learning_rate = 0.01
    epochs = 1000

    # Instantiate the model with hyperparameters
    my_rnn = MyRNN(input_size=input_size, output_size=output_size, hidden_dim=hidden_size, n_layers=1)
    # We'll also set the model to the device that we defined earlier (default is CPU)
    my_rnn.to(device)

    # Defining the optimizer
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
        loss = loss_func(output, y_train)
        # Does backpropagation and calculates gradients
        loss.backward()
        # Updates the weights accordingly
        optimizer.step()

        if epoch % 10 == 0:
            print('Epoch: {}/{}.............'.format(epoch, epochs), end=' ')
            print("Loss: {:.4f}".format(loss.item()))

    end = time.time()
    print("Done in {} seconds !\n".format(int(end - start)))

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    text = ['hey how are you', 'good i am fine', 'have a nice day']

    # Join all the sentences together and extract the unique characters from the combined sentences
    chars = set(''.join(text))

    # Creating a dictionary that maps integers to the characters
    int2char = dict(enumerate(chars))

    # Creating another dictionary that maps characters to integers
    char2int = {char: ind for ind, char in int2char.items()}

    maxlen = len(max(text, key=len))
    print("The longest string has {} characters".format(maxlen))

    # Padding

    # A simple loop that loops through the list of sentences and adds a ' ' whitespace until the length of the sentence matches
    # the length of the longest sentence
    for i in range(len(text)):
        while len(text[i]) < maxlen:
            text[i] += ' '

    # Creating lists that will hold our input and target sequences
    input_seq = []
    target_seq = []

    for i in range(len(text)):
        # Remove last character for input sequence
        input_seq.append(text[i][:-1])

        # Remove firsts character for target sequence
        target_seq.append(text[i][1:])
        print("Input Sequence: {}\nTarget Sequence: {}".format(input_seq[i], target_seq[i]))

    for i in range(len(text)):
        input_seq[i] = [char2int[character] for character in input_seq[i]]
        target_seq[i] = [char2int[character] for character in target_seq[i]]

    dict_size = len(char2int)
    seq_len = maxlen - 1
    batch_size = len(text)

    input_seq = one_hot_encode(input_seq, dict_size, seq_len, batch_size)
    print("Input shape: {} --> (Batch Size, Sequence Length, One-Hot Encoding Size)".format(input_seq.shape))

    input_seq = torch.from_numpy(input_seq)
    target_seq = torch.Tensor(target_seq)

    # Instantiate the model with hyperparameters
    model = MyRNN(input_size=dict_size, output_size=dict_size, hidden_dim=12, n_layers=1)
    # We'll also set the model to the device that we defined earlier (default is CPU)
    model = model.to(device)

    # Define hyperparameters
    n_epochs = 100
    lr = 0.01

    # Define Loss, Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training Run
    input_seq = input_seq.to(device)
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()  # Clears existing gradients from previous epoch
        # input_seq = input_seq.to(device)
        output = model(input_seq)
        output = output.to(device)
        target_seq = target_seq.to(device)
        loss = criterion(output, target_seq.view(-1).long())
        loss.backward()  # Does backpropagation and calculates gradients
        optimizer.step()  # Updates the weights accordingly

        if epoch % 10 == 0:
            print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
            print("Loss: {:.4f}".format(loss.item()))


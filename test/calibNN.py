import torch
import torch.nn as nn
from numpy import array, rot90

# Define the array
data = rot90(array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])).copy()

# Convert the array to a PyTorch tensor
seqs = torch.from_numpy(data).float()


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


"""
This function creates a recurrent neural network (RNN) model 
using the Gated Recurrent Unit (GRU) architecture. 
The GRU is a type of RNN that is capable of capturing sequential 
information and has shown good performance in various sequence modeling tasks.


- dim_seq_in: This specifies the input size or dimensionality of 
               each element in the input sequence. 
               In this case, it is set to 2, which means each element 
               in the input sequence has a feature vector of size 2.
- hidden_size: This determines the number of output features or hidden 
            units in the GRU layer. 
            In this case, it is set to 32, meaning the GRU layer will 
            produce a hidden state vector of size 32.
- dim_out: This specifies the desired output size of the GRU layer.
            In this case, it is set to 64, meaning the GRU layer's output will have a dimensionality of 64.

To summarize, hidden_size determines the size of the hidden state vector produced by the GRU layer, 
while dim_out determines the size of the output vector produced by the GRU layer.

- bidirectional: This is a boolean flag that determines whether the GRU layer should be bidirectional or not. 
            If set to True, the GRU layer will process the input sequence in both forward and backward directions. 
            If set to False, it will only process the sequence in the forward direction. 
            Bidirectional RNNs are often used to capture dependencies in both past and future contexts.

"""
dim_seq_in = 2
hidden_size = 32
dim_out = 64
bidirectional = True
n_layers = 3
dropout = 0.0
dim_metadata = 1

rnn = nn.GRU(
    input_size=dim_seq_in,
    hidden_size=hidden_size,
    bidirectional=bidirectional,
    num_layers=n_layers,
    dropout=dropout,
)


out_layer = [
    nn.Linear(in_features=hidden_size * 2, out_features=dim_out),
    nn.Tanh(),
    nn.Dropout(dropout),
]
out_layer = nn.Sequential(*out_layer)
out_layer.apply(init_weights)


"""
1. latent_seqs
    To understand the value at a specific location in latent_seqs, 
    such as [2, 35], you need to consider the structure and semantics of the tensor.

    latent_seqs is a tensor resulting from the output of the GRU layer (rnn) applied to the input sequence seqs. 
    It has a size of 10x64, meaning it has 10 rows (corresponding to the 10 time steps in the input sequence) 
    and 64 columns (representing the hidden state size of the GRU layer).

    The value at a specific location, such as [2, 35], represents the value of the tensor at the 2nd row and 35th column. 
    In this context, the 2nd row corresponds to the hidden state at the 2nd time step of the input sequence,
    and the 35th column corresponds to a specific element or feature in the hidden state vector.

    RNNs, including GRU layers, are designed to capture and model sequential dependencies in the input data. 
    The hidden state vectors in latent_seqs are representations that encode information 
    about the temporal context of the input sequence, taking into account the order and relationships between the elements.

2. encoder_hidden
    encoder_hidden is the output of the GRU layer (rnn) when applied to the input sequence seqs. 
    The GRU layer is used as an encoder, which processes the input sequence and produces 
    a final hidden state that summarizes the information in the entire sequence.

    The first dimension (6) corresponds to (num_layers * num_directions), which in this case would be 6 (num_layers x num_directions = 3 x 2).
    The second dimension (32) represents the hidden_size of the GRU layer.

"""
latent_seqs, encoder_hidden = rnn(seqs)
latent_seqs = latent_seqs.sum(0)
output = out_layer(latent_seqs)

import numpy

training_weeks = 3
time_seq = numpy.arange(1, training_weeks).repeat(output.shape[0], 1).unsqueeze(2)

h0 = encoder_hidden[3:]


embed_input = nn.Linear(dim_seq_in, hidden_size)

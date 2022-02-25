import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import collections

import torch
import torch.nn as nn

from torch.autograd import Variable as V
from torch.utils.data import DataLoader


#original model from 22_02:


class AutoEncoder_1(nn.Module):
    def __init__(self, length):
        super().__init__()
        self.lin1 = nn.Linear(length, 24)
        self.lin2_bn = nn.BatchNorm1d(24)
        self.lin2 = nn.Linear(24, 12)
        self.lin3_bn = nn.BatchNorm1d(12)
        self.lin3 = nn.Linear(12, 6)

        self.lin6 = nn.Linear(6, 12)
        self.lin7_bn = nn.BatchNorm1d(12)
        self.lin7 = nn.Linear(12, 24)
        self.lin8_bn = nn.BatchNorm1d(24)
        self.lin8 = nn.Linear(24, length)

        #self.drop2 = nn.Dropout(0.2)

        self.lin1.weight.data.uniform_(-2, 2)
        self.lin2.weight.data.uniform_(-2, 2)
        self.lin3.weight.data.uniform_(-2, 2)

        self.lin6.weight.data.uniform_(-2, 2)
        self.lin7.weight.data.uniform_(-2, 2)
        self.lin8.weight.data.uniform_(-2, 2)

    def forward(self, data):

        x = torch.tanh(self.lin1(data))
        #x = torch.sigmoid(self.lin2(x))
        #x = torch.tanh(self.drop2(self.lin2(x)))
        x = torch.tanh(self.lin2(self.lin2_bn(x)))
        x = torch.tanh(self.lin3(self.lin3_bn(x)))

        x = torch.tanh(self.lin6(x))
        x = torch.tanh(self.lin7(self.lin7_bn(x)))
        x = torch.tanh(self.lin8(self.lin8_bn(x)))
        #x = torch.tanh(self.lin8(x))
        #x = torch.sigmoid(self.drop2(self.lin8(x)))
        return x

        #x = torch.tanh(self.lin2(x))
        #x = self.drop2(torch.tanh(self.lin2(x)))
        #print(x.shape)
        #x = self.lin3_bn(x)
        #x = torch.tanh(self.lin3(x))
        #x = self.drop2(torch.tanh(self.lin3(x)))
        #x = self.drop2(torch.tanh(self.lin4(x)))
        #x = torch.tanh(self.lin6(x))
        #x = torch.tanh(self.lin7(x))
        #x = self.lin8(x)
        #return x


class AutoEncoder_2(nn.Module): #similar to one above, slight adjustment to neurons per layer
    def __init__(self, length):
        super().__init__()
        self.lin1 = nn.Linear(length,20)
        self.lin2_bn = nn.BatchNorm1d(20)
        self.lin2 = nn.Linear(20, 8)
        self.lin3_bn = nn.BatchNorm1d(8)
        self.lin3 = nn.Linear(8, 4)

        self.lin6 = nn.Linear(4, 8)
        self.lin7_bn = nn.BatchNorm1d(8)
        self.lin7 = nn.Linear(8, 20)
        self.lin8_bn = nn.BatchNorm1d(20)
        self.lin8 = nn.Linear(20, length)


        self.lin1.weight.data.uniform_(-2, 2)
        self.lin2.weight.data.uniform_(-2, 2)
        self.lin3.weight.data.uniform_(-2, 2)

        self.lin6.weight.data.uniform_(-2, 2)
        self.lin7.weight.data.uniform_(-2, 2)
        self.lin8.weight.data.uniform_(-2, 2)

    def forward(self, data):

        x = torch.tanh(self.lin1(data))
        x = torch.tanh(self.lin2(self.lin2_bn(x)))
        x = torch.tanh(self.lin3(self.lin3_bn(x)))

        x = torch.tanh(self.lin6(x))
        x = torch.tanh(self.lin7(self.lin7_bn(x)))
        x = torch.tanh(self.lin8(self.lin8_bn(x)))

        return x


class AutoEncoder_3(nn.Module): #at encoder: batch norm -> drop at inner layer -> decoder: drop inner layer, batch norm outer
    def __init__(self, length):
        super().__init__()
        self.lin1 = nn.Linear(length, 24)
        self.lin2_bn = nn.BatchNorm1d(24)
        self.lin2 = nn.Linear(24, 12)
        #self.lin3_bn = nn.BatchNorm1d(12)
        self.lin3 = nn.Linear(12, 6)

        self.lin6 = nn.Linear(6, 12)
        self.lin7_bn = nn.BatchNorm1d(12)
        self.lin7 = nn.Linear(12, 24)
        #self.lin8_bn = nn.BatchNorm1d(24)
        self.lin8 = nn.Linear(24, length)

        self.drop2 = nn.Dropout(0.2)

        self.lin1.weight.data.uniform_(-2, 2)
        self.lin2.weight.data.uniform_(-2, 2)
        self.lin3.weight.data.uniform_(-2, 2)

        self.lin6.weight.data.uniform_(-2, 2)
        self.lin7.weight.data.uniform_(-2, 2)
        self.lin8.weight.data.uniform_(-2, 2)

    def forward(self, data):

        x = torch.tanh(self.lin1(data))
        x = torch.tanh(self.lin2(self.lin2_bn(x)))
        x = self.drop2(torch.tanh(self.lin3(x)))
        #x = torch.tanh(self.lin3(x))

        #x = torch.tanh(self.lin6(x))
        x = self.drop2(torch.tanh(self.lin6(x)))
        x = torch.tanh(self.lin7(self.lin7_bn(x)))
        #x = torch.tanh(self.lin8(self.lin8_bn(x)))
        x = torch.tanh(self.lin8(x))

        return x


class AutoEncoder_4(nn.Module):
    def __init__(self, length):
        super().__init__()
        self.lin1 = nn.Linear(length, 24)
        #self.lin2_bn = nn.BatchNorm1d(24)
        self.lin2 = nn.Linear(24, 12)
        self.lin3_bn = nn.BatchNorm1d(12)
        self.lin3 = nn.Linear(12, 6)

        self.lin6 = nn.Linear(6, 12)
        self.lin7_bn = nn.BatchNorm1d(12)
        self.lin7 = nn.Linear(12, 24)
        self.lin8_bn = nn.BatchNorm1d(24)
        self.lin8 = nn.Linear(24, length)

        self.drop2 = nn.Dropout(0.2)

        self.lin1.weight.data.uniform_(-2, 2)
        self.lin2.weight.data.uniform_(-2, 2)
        self.lin3.weight.data.uniform_(-2, 2)

        self.lin6.weight.data.uniform_(-2, 2)
        self.lin7.weight.data.uniform_(-2, 2)
        self.lin8.weight.data.uniform_(-2, 2)

    def forward(self, data):
        x = torch.tanh(self.lin1(data))
        x = self.drop2(torch.tanh(self.lin2(x)))
        x = torch.tanh(self.lin3(self.lin3_bn(x)))

        x = torch.tanh(self.lin6(x)) #mulig til helvete pga denne? problem med dropout -> videre batch_norm og utvidelse, tror dropout bare ok på vei inn?
        x = torch.tanh(self.lin7(self.lin7_bn(x)))
        x = torch.tanh(self.lin8(self.lin8_bn(x)))


        return x

#LSTM attempt
class Encoder(nn.Module):

 def __init__(self, seq_len, n_features, embedding_dim=64):
   super(Encoder, self).__init__()

   self.seq_len, self.n_features = seq_len, n_features
   self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

   self.rnn1 = nn.LSTM(
        input_size=n_features,
        hidden_size=self.hidden_dim,
        num_layers=1,
        batch_first=True)
  
   self.rnn2 = nn.LSTM(
        input_size=self.hidden_dim,
        hidden_size=embedding_dim,
        num_layers=1,
        batch_first=True)

 def forward(self, x):
   x = x.reshape((1, self.seq_len, self.n_features))

   x, (_, _) = self.rnn1(x)
   x, (hidden_n, _) = self.rnn2(x)

   return hidden_n.reshape((self.n_features, self.embedding_dim))


class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, n_features=1):
        super(Decoder, self).__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features

        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True
            )   
        self.rnn2 = nn.LSTM(
                input_size=input_dim,
                hidden_size=self.hidden_dim,
                num_layers=1,
                batch_first=True
            )
        self.output_layer = nn.Linear(self.hidden_dim, n_features)


def forward(self, x):
    x = x.repeat(self.seq_len, self.n_features)
    x = x.reshape((self.n_features, self.seq_len, self.input_dim))

    x, (hidden_n, cell_n) = self.rnn1(x)
    x, (hidden_n, cell_n) = self.rnn2(x)
    x = x.reshape((self.seq_len, self.hidden_dim))

    return self.output_layer(x)

class RecurrentAutoencoder(nn.Module):
   def __init__(self, seq_len, n_features, embedding_dim=64):
        super(RecurrentAutoencoder, self).__init__()

        self.encoder = Encoder(seq_len, n_features, embedding_dim)
        self.decoder = Decoder(seq_len, embedding_dim, n_features)

def forward(self, x):
   z = self.encoder(x)
   o = self.decoder(z)
   return o
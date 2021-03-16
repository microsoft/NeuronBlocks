# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from block_zoo.BaseLayer import BaseLayer, BaseConf
from utils.DocInherit import DocInherit

class BiLSTMLastConf(BaseConf):
    """ Configuration of BiLSTMLast

    Args:
        hidden_dim (int): dimension of hidden state
        dropout (float): dropout rate
        num_layers (int): number of BiLSTM layers
    """
    def __init__(self, **kwargs):
        super(BiLSTMLastConf, self).__init__(**kwargs)

    @DocInherit
    def default(self):
        self.hidden_dim = 128
        self.dropout = 0.0
        self.num_layers = 1

    @DocInherit
    def declare(self):
        self.num_of_inputs = 1
        self.input_ranks = [3]

    @DocInherit
    def inference(self):
        self.output_dim = [-1]
        self.output_dim.append(2 * self.hidden_dim)

        super(BiLSTMLastConf, self).inference()  # PUT THIS LINE AT THE END OF inference()

    @DocInherit
    def verify(self):
        super(BiLSTMLastConf, self).verify()

        necessary_attrs_for_user = ['hidden_dim', 'dropout', 'num_layers']
        for attr in necessary_attrs_for_user:
            self.add_attr_exist_assertion_for_user(attr)


class BiLSTMLast(BaseLayer):
    """ get last hidden states of Bidrectional LSTM

    Args:
        layer_conf (BiLSTMConf): configuration of a layer
    """
    def __init__(self, layer_conf):
        super(BiLSTMLast, self).__init__(layer_conf)
        self.lstm = nn.LSTM(layer_conf.input_dims[0][-1], layer_conf.hidden_dim, layer_conf.num_layers, bidirectional=True,
            dropout=layer_conf.dropout, batch_first=True)

    def forward(self, string, string_len):
        """ process inputs

        Args:
            string (Tensor): [batch_size, seq_len, dim]
            string_len (Tensor): [batch_size]

        Returns:
            Tensor: [batch_size, 2 * hidden_dim]

        """
        str_len, idx_sort = (-string_len).sort()
        str_len = -str_len
        idx_unsort = idx_sort.sort()[1]

        string = string.index_select(0, idx_sort)

        # Handling padding in Recurrent Networks
        string_packed = nn.utils.rnn.pack_padded_sequence(string, str_len.cpu(), batch_first=True)
        self.lstm.flatten_parameters()
        string_output, (hn, cn) = self.lstm(string_packed)  # seqlen x batch x 2*nhid

        emb = torch.cat((hn[0], hn[1]), 1)  # batch x 2*nhid
        emb = emb.index_select(0, idx_unsort)

        return emb, string_len

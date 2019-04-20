# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from block_zoo.BaseLayer import BaseLayer, BaseConf
from utils.DocInherit import DocInherit
import copy

class BiLSTMConf(BaseConf):
    """ Configuration of BiLSTM

    Args:
        hidden_dim (int): dimension of hidden state
        dropout (float): dropout rate
        num_layers (int): number of BiLSTM layers
    """
    def __init__(self, **kwargs):
        super(BiLSTMConf, self).__init__(**kwargs)

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
        self.output_dim = copy.deepcopy(self.input_dims[0])
        self.output_dim[-1] = 2 * self.hidden_dim

        super(BiLSTMConf, self).inference()      # PUT THIS LINE AT THE END OF inference()

    @DocInherit
    def verify(self):
        super(BiLSTMConf, self).verify()

        necessary_attrs_for_user = ['hidden_dim', 'dropout', 'num_layers']
        for attr in necessary_attrs_for_user:
            self.add_attr_exist_assertion_for_user(attr)


class BiLSTM(BaseLayer):
    """ Bidrectional LSTM

    Args:
        layer_conf (BiLSTMConf): configuration of a layer
    """
    def __init__(self, layer_conf):
        super(BiLSTM, self).__init__(layer_conf)
        self.lstm = nn.LSTM(layer_conf.input_dims[0][-1], layer_conf.hidden_dim, layer_conf.num_layers, bidirectional=True,
            dropout=layer_conf.dropout, batch_first=True)

    def forward(self, string, string_len):
        """ process inputs

        Args:
            string (Tensor): [batch_size, seq_len, dim]
            string_len (Tensor): [batch_size]

        Returns:
            Tensor: [batch_size, seq_len, 2 * hidden_dim]

        """
        padded_seq_len = string.shape[1]

        # Sort by length (keep idx)
        str_len, idx_sort = (-string_len).sort()
        str_len = -str_len
        idx_unsort = idx_sort.sort()[1]

        string = string.index_select(0, idx_sort)

        # Handling padding in Recurrent Networks
        string_packed = nn.utils.rnn.pack_padded_sequence(string, str_len, batch_first=True)
        self.lstm.flatten_parameters()
        string_output = self.lstm(string_packed)[0]  # seqlen x batch x 2*nhid
        string_output = nn.utils.rnn.pad_packed_sequence(string_output, batch_first=True, total_length=padded_seq_len)[0]

        # Un-sort by length
        string_output = string_output.index_select(0, idx_unsort)

        return string_output, string_len

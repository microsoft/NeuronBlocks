# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np

from block_zoo.BaseLayer import BaseLayer, BaseConf
from utils.DocInherit import DocInherit


class BiLSTMAttConf(BaseConf):
    """ Configuration of BiLSTMAtt layer

    Args:
        hidden_dim (int): dimension of hidden state
        dropout (float): dropout rate
        num_layers (int): number of BiLSTM layers
    """
    def __init__(self, **kwargs):
        super(BiLSTMAttConf, self).__init__(**kwargs)

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

        self.attention_dim = 2 * self.hidden_dim

        super(BiLSTMAttConf, self).inference()  # PUT THIS LINE AT THE END OF inference()

    @DocInherit
    def verify_before_inference(self):
        super(BiLSTMAttConf, self).verify_before_inference()
        necessary_attrs_for_user = ['hidden_dim']
        for attr in necessary_attrs_for_user:
            self.add_attr_exist_assertion_for_user(attr)

    @DocInherit
    def verify(self):
        super(BiLSTMAttConf, self).verify()

        necessary_attrs_for_user = ['dropout', 'attention_dim']
        for attr in necessary_attrs_for_user:
            self.add_attr_exist_assertion_for_user(attr)


class BiLSTMAtt(BaseLayer):
    """ BiLSTM with self attention

    Args:
        layer_conf (BiLSTMAttConf): configuration of a layer
    """
    def __init__(self, layer_conf):
        super(BiLSTMAtt, self).__init__(layer_conf)
        self.lstm = nn.LSTM(layer_conf.input_dims[0][-1], layer_conf.hidden_dim, layer_conf.num_layers, bidirectional=True,
            dropout=layer_conf.dropout, batch_first=True)
        self.att = nn.Parameter(torch.randn(layer_conf.attention_dim, layer_conf.attention_dim), requires_grad=True)
        nn.init.uniform_(self.att, a=0, b=1)
        self.softmax = nn.Softmax()

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
        string_len_sorted, idx_sort = (-string_len).sort()
        string_len_sorted = -string_len_sorted
        idx_unsort = idx_sort.sort()[1]

        bsize = string.shape[0]

        string = string.index_select(0, idx_sort)

        # Handling padding in Recurrent Networks
        string_packed = nn.utils.rnn.pack_padded_sequence(string, string_len_sorted, batch_first=True)
        self.lstm.flatten_parameters()
        string_output = self.lstm(string_packed)[0]  # seqlen x batch x 2*nhid
        string_output = nn.utils.rnn.pad_packed_sequence(string_output, batch_first=True, total_length=padded_seq_len)[0]

        # Un-sort by length
        string_output = string_output.index_select(0, idx_unsort).contiguous()   # [batch, seqlen, 2*nhid]

        # Self Attention
        alphas = string_output.matmul(self.att).bmm(string_output.transpose(1, 2).contiguous())  # [batch, seqlen, seqlen]

        # Set probas of padding to zero in softmax
        alphas = alphas + ((alphas == 0).float() * -10000)

        # softmax
        alphas = self.softmax(alphas.view(-1, int(padded_seq_len)))  # [batch*seglen, seqlen]

        alphas = alphas.view(bsize, -1, int(padded_seq_len))  # [batch, seglen, seq_len]

        string_output = alphas.bmm(string_output)  # [batch, seglen, 2*nhid]

        return string_output, string_len

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from block_zoo.BaseLayer import BaseLayer, BaseConf
from utils.DocInherit import DocInherit
from utils.common_utils import transfer_to_gpu

class BiGRULastConf(BaseConf):
    """ Configuration of the layer BiGRULast

    Args:
        hidden_dim (int): dimension of hidden state
        dropout (float): dropout rate
    """
    def __init__(self, **kwargs):

        super(BiGRULastConf, self).__init__(**kwargs)

    @DocInherit
    def default(self):
        self.hidden_dim = 128
        self.dropout = 0.0

    @DocInherit
    def declare(self):
        self.num_of_inputs = 1
        self.input_ranks = [3]

    @DocInherit
    def inference(self):
        self.output_dim = [-1]
        self.output_dim.append(2 * self.hidden_dim)

        super(BiGRULastConf, self).inference()  # PUT THIS LINE AT THE END OF inference()

    @DocInherit
    def verify_before_inference(self):
        super(BiGRULastConf, self).verify_before_inference()
        necessary_attrs_for_user = ['hidden_dim']
        for attr in necessary_attrs_for_user:
            self.add_attr_exist_assertion_for_user(attr)

    @DocInherit
    def verify(self):
        super(BiGRULastConf, self).verify()
        necessary_attrs_for_user = ['hidden_dim', 'dropout']
        for attr in necessary_attrs_for_user:
            self.add_attr_exist_assertion_for_user(attr)


class BiGRULast(BaseLayer):
    """ Get the last hidden state of Bi GRU

    Args:
        layer_conf (BiGRULastConf): configuration of a layer
    """
    def __init__(self, layer_conf):
        super(BiGRULast, self).__init__(layer_conf)
        self.GRU = nn.GRU(layer_conf.input_dims[0][-1], layer_conf.hidden_dim, 1, bidirectional=True,
            dropout=layer_conf.dropout, batch_first=True)

    def forward(self, string, string_len):
        """ process inputs

        Args:
            string (Tensor): [batch_size, seq_len, dim]
            string_len (Tensor): [batch_size]

        Returns:
            Tensor: [batch_size, 2 * hidden_dim]
        """
        #string = string.permute([1, 0, 2])
        self.init_GRU = torch.FloatTensor(2, string.size(0), self.layer_conf.hidden_dim).zero_()
        if self.is_cuda():
            self.init_GRU = transfer_to_gpu(self.init_GRU)

        # Sort by length (keep idx)
        str_len, idx_sort = (-string_len).sort()
        str_len = -str_len
        idx_unsort = idx_sort.sort()[1]

        string = string.index_select(0, idx_sort)

        # Handling padding in Recurrent Networks
        string_packed = nn.utils.rnn.pack_padded_sequence(string, str_len, batch_first=True)
        self.GRU.flatten_parameters()
        string_output, hn = self.GRU(string_packed, self.init_GRU)  # seqlen x batch x 2*nhid

        emb = torch.cat((hn[0], hn[1]), 1)  # batch x 2*nhid

        emb = emb.index_select(0, idx_unsort)
        return emb, string_len

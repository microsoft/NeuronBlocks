# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from block_zoo.BaseLayer import BaseLayer, BaseConf
from utils.DocInherit import DocInherit
from utils.common_utils import transfer_to_gpu


class BiGRUConf(BaseConf):
    """Configuration of BiGRU

    Args:
        hidden_dim (int): dimension of hidden state
        dropout (float): dropout rate

    """
    def __init__(self, **kwargs):
        super(BiGRUConf, self).__init__(**kwargs)

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
        self.output_dim = copy.deepcopy(self.input_dims[0])
        self.output_dim[-1] = 2 * self.hidden_dim
        super(BiGRUConf, self).inference()  # PUT THIS LINE AT THE END OF inference()

    @DocInherit
    def verify(self):
        super(BiGRUConf, self).verify()
        assert hasattr(self, 'hidden_dim'), "Please define hidden_dim attribute of BiGRUConf in default() or the configuration file"
        assert hasattr(self, 'dropout'), "Please define dropout attribute of BiGRUConf in default() or the configuration file"


class BiGRU(BaseLayer):
    """Bidirectional GRU

    Args:
        layer_conf (BiGRUConf): configuration of a layer
    """
    def __init__(self, layer_conf):
        super(BiGRU, self).__init__(layer_conf)
        self.GRU = nn.GRU(layer_conf.input_dims[0][-1], layer_conf.hidden_dim, 1, bidirectional=True,
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
        self.init_GRU = torch.FloatTensor(2, string.size(0), self.layer_conf.hidden_dim).zero_()
        if self.is_cuda():
            self.init_GRU = transfer_to_gpu(self.init_GRU)

        # Sort by length (keep idx)
        str_len, idx_sort = (-string_len).sort()
        str_len = -str_len
        idx_unsort = idx_sort.sort()[1]

        string = string.index_select(0, idx_sort)

        # Handling padding in Recurrent Networks
        string_packed = nn.utils.rnn.pack_padded_sequence(string, str_len.cpu(), batch_first=True)
        self.GRU.flatten_parameters()
        string_output, hn = self.GRU(string_packed, self.init_GRU)  # seqlen x batch x 2*nhid
        string_output = nn.utils.rnn.pad_packed_sequence(string_output, batch_first=True, total_length=padded_seq_len)[0]

        # Un-sort by length
        string_output = string_output.index_select(0, idx_unsort)

        return string_output, string_len


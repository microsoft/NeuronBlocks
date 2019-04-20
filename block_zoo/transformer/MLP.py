# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import math

from block_zoo.BaseLayer import BaseLayer,BaseConf
from utils.DocInherit import DocInherit
import copy

class MLPConf(BaseConf):
    """ Configuration of MLP layer

    Args:
        dropout (float): the dropout of MLP layer

    """
    def __init__(self, **kwargs):
        super(MLPConf, self).__init__(**kwargs)

    @DocInherit
    def default(self):
        self.dropout = 0.1

    @DocInherit
    def declare(self):
        self.num_of_inputs = 1
        self.input_ranks = [3]

    @DocInherit
    def inference(self):
        self.output_dim = copy.deepcopy(self.input_dims[0])
        super(MLPConf, self).inference()

    @DocInherit
    def verify(self):
        super(MLPConf, self).verify()

class MLP(nn.Module):
    """ MLP layer

    Args:
        layer_conf (MLPConf): configuration of a layer

    """
    def __init__(self, layer_conf):
        super(MLP, self).__init__()
        self.layer_conf = layer_conf
        self.n_state = self.layer_conf.input_dims[0][-1]
        self.c_fc = nn.Linear(self.layer_conf.input_dims[0][-1], 4*self.n_state)
        self.c_proj = nn.Linear(4*self.n_state, self.layer_conf.input_dims[0][-1])

    def gelu(self,x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

    def forward(self, string, string_len):
        """ process input

        Args:
            string, string_len
            e.g. string (Tensor): [batch_size, seq_len, dim], string_len (Tensor): [batch_size]

        Returns:
            Tensor: [batch_size, seq_len, output_dim], [batch_size]

        """
        h = self.gelu(self.c_fc(string))
        h2 = self.c_proj(h)
        return nn.Dropout(self.layer_conf.dropout)(h2), string_len

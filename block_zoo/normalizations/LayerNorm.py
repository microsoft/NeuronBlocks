# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn

from block_zoo.BaseLayer import BaseLayer,BaseConf
from utils.DocInherit import DocInherit
import copy

class LayerNormConf(BaseConf):
    """ Configuration of LayerNorm Layer

    """
    def __init__(self,**kwargs):
        super(LayerNormConf, self).__init__(**kwargs)

    # @DocInherit
    # def default(self):

    @DocInherit
    def declare(self):
        self.num_of_inputs = 1
        self.input_ranks = [3]

    @DocInherit
    def inference(self):
        self.output_dim = copy.deepcopy(self.input_dims[0])
        super(LayerNormConf, self).inference()

    @DocInherit
    def verify(self):
        super(LayerNormConf, self).verify()

class LayerNorm(nn.Module):
    """ LayerNorm layer

    Args:
        layer_conf (LayerNormConf): configuration of a layer

    """
    def __init__(self,layer_conf):
        super(LayerNorm, self).__init__()
        self.layer_conf = layer_conf
        self.g = nn.Parameter(torch.ones(self.layer_conf.input_dims[0][-1]))
        self.b = nn.Parameter(torch.zeros(self.layer_conf.input_dims[0][-1]))
        self.e = 1e-5

    def forward(self, string, string_len):
        """ process input

        Args:
            string, string_len
            e.g. string (Tensor): [batch_size, seq_len, dim], string_len (Tensor): [batch_size]

        Returns:
            Tensor: [batch_size, seq_len, output_dim], [batch_size]
        """
        u = string.mean(-1,keepdim=True)
        s = (string - u).pow(2).mean(-1,keepdim=True)
        string = (string - u)/torch.sqrt(s+self.e)
        return self.g * string + self.b, string_len
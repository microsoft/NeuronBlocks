# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch.nn as nn
import logging

from block_zoo.BaseLayer import BaseConf
from utils.DocInherit import DocInherit
from utils.exceptions import ConfigurationError
import copy

class FlattenConf(BaseConf):
    """Configuration of Flatten layer

    """

    #init the args
    def __init__(self, **kwargs):
        super(FlattenConf, self).__init__(**kwargs)

    #set default params
    #@DocInherit
    #def default(self):

    @DocInherit
    def declare(self):
        self.num_of_inputs = 1
        self.input_ranks = [-1]

    @DocInherit
    def inference(self):
        self.output_dim = []
        flatted_length = 1
        for i in range(1, len(self.input_dims[0])):
            if self.input_dims[0][i] == -1:
                raise ConfigurationError("For Flatten layer, the sequence length should be fixed")
            else:
                flatted_length *= self.input_dims[0][i]
        
        self.output_dim = [self.input_dims[0][0], flatted_length]
            
        super(FlattenConf, self).inference()

    @DocInherit
    def verify(self):
        super(FlattenConf, self).verify()

class Flatten(nn.Module):
    """  Flatten layer to flatten the input from [bsatch_size, seq_len, dim] to [batch_size, seq_len*dim]

    Args:
        layer_conf(FlattenConf): configuration of a layer
    """

    def __init__(self, layer_conf):
        super(Flatten, self).__init__()
        self.layer_conf = layer_conf

    def forward(self, string, string_len):
        """ process input

        Args:
            *args: (Tensor): string,string_len
                e.g. string (Tensor): [batch_size, seq_len, dim], string_len (Tensor): [batch_size]
        Returns:
            Tensor: [batch_size, seq_len*dim], [batch_size]
        """
        flattened = string.view(string.shape[0], -1)
        string_len = flattened.size(1)

        return flattened, string_len


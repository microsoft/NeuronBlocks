# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import logging

from ..BaseLayer import BaseConf,BaseLayer
from utils.DocInherit import DocInherit
from utils.exceptions import ConfigurationError
import copy

class Concat2DConf(BaseConf):
    """ Configuration of Concat2D Layer

    Args:
        concat2D_axis(int): which axis to conduct concat2D, default is 1.
    """

    # init the args
    def __init__(self,**kwargs):
        super(Concat2DConf, self).__init__(**kwargs)

    # set default params
    @DocInherit
    def default(self):
        self.concat2D_axis = 1

    @DocInherit
    def declare(self):
        self.num_of_inputs = -1
        self.input_ranks = [2]

    @DocInherit
    def inference(self):
        self.output_dim = copy.deepcopy(self.input_dims[0])
        self.output_dim[-1] = 0
        self.output_dim[-1] += sum([input_dim[-1] for input_dim in self.input_dims])
        
        super(Concat2DConf, self).inference()

    @DocInherit
    def verify(self):
        super(Concat2DConf, self).verify()

        # to check if the ranks of all the inputs are equal
        rank_equal_flag = True
        for i in range(len(self.input_ranks)):
            if self.input_ranks[i] != self.input_ranks[0]:
                rank_equal_flag = False
                break
        if rank_equal_flag == False:
            raise ConfigurationError("For layer Concat2D, the ranks of each inputs should be equal!")

        # to check if the concat2D_axis is legal
        if self.concat2D_axis != 1:
            raise ConfigurationError("For layer Concat2D, the concat axis must be 1!")

class Concat2D(nn.Module):
    """ Concat2D layer to merge sum of sequences(2D representation)

    Args:
        layer_conf (Concat2DConf): configuration of a layer
    """
    def __init__(self, layer_conf):
        super(Concat2D, self).__init__()
        self.layer_conf = layer_conf

        logging.warning("The length Concat2D layer returns is the length of first input")

    def forward(self, *args):
        """ process inputs

        Args:
            *args: (Tensor): string, string_len, string2, string2_len, ...
                e.g. string (Tensor): [batch_size, dim], string_len (Tensor): [batch_size]

        Returns:
            Tensor: [batch_size, output_dim], [batch_size]

        """
        result = []
        for idx, input in enumerate(args):
            if idx % 2 == 0:
                result.append(input)
        return torch.cat(result,self.layer_conf.concat2D_axis), args[1]
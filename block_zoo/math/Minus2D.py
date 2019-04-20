# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.


import torch
import torch.nn as nn
import logging

from ..BaseLayer import BaseConf
from utils.DocInherit import DocInherit
from utils.exceptions import ConfigurationError
import copy

class Minus2DConf(BaseConf):
    """ Configuration of Minus2D layer

    Args:
        abs_flag: if the result of the Minus2D is abs, default is False

    """

    #init the args
    def __init__(self,**kwargs):
        super(Minus2DConf, self).__init__(**kwargs)

    #set default params
    @DocInherit
    def default(self):
        self.abs_flag = False

    @DocInherit
    def declare(self):
        self.num_of_inputs = 2
        self.input_ranks = [2,2]

    @DocInherit
    def inference(self):
        self.output_dim = copy.deepcopy(self.input_dims[0])
        if self.input_dims[0][1] != 1:
            self.output_dim[-1] = self.input_dims[0][1]
        else:
            self.output_dim[-1] = self.input_dims[1][1]

        super(Minus2DConf, self).inference()

    @DocInherit
    def verify(self):
        super(Minus2DConf, self).verify()

        # # to check if the ranks of all the inputs are equal
        # rank_equal_flag = True
        # for i in range(len(self.input_ranks)):
        #     if self.input_ranks[i] != self.input_ranks[0]:
        #         rank_equal_flag = False
        #         break
        # if rank_equal_flag == False:
        #     raise ConfigurationError("For layer Minus2D, the ranks of each inputs should be equal!")

        # to check if the dimensions of all the inputs are equal or is 1
        dim_flag = True
        input_dims = list(self.input_dims)
        for i in range(len(input_dims)):
            if input_dims[i][1] != input_dims[0][1] and input_dims[i][1] != 1 and input_dims[0][1] != 1:
                dim_flag = False
                break
        if dim_flag == False:
            raise ConfigurationError("For layer Minus2D, the dimensions of each inputs should be equal or 1")

class Minus2D(nn.Module):
    """Minus2D layer to get subtraction of two sequences(2D representation)

    Args:
        layer_conf (Minus2DConf): configuration of a layer

    """
    def __init__(self, layer_conf):
        super(Minus2D, self).__init__()
        self.layer_conf = layer_conf

        logging.warning("The length Minus2D layer returns is the length of first input")

    def forward(self, *args):
        """ process inputs

        Args:
            *args: (Tensor): string, string_len, string2, string2_len
                e.g. string (Tensor): [batch_size, dim], string_len (Tensor): [batch_size]

        Returns:
            Tensor: [batch_size, output_dim], [batch_size]

        """
        if self.layer_conf.abs_flag == False:
            return (args[0] - args[2]), args[1]
        if self.layer_conf.abs_flag == True:
            return torch.abs(args[0] - args[2]),args[1]



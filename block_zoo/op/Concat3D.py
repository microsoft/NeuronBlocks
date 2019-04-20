# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import logging

from ..BaseLayer import BaseConf,BaseLayer
from utils.DocInherit import DocInherit
from utils.exceptions import ConfigurationError
import copy

class Concat3DConf(BaseConf):
    """ Configuration of Concat3D layer

    Args:
        concat3D_axis(1 or 2): which axis to conduct Concat3D, default is 2.

    """

    # init the args
    def __init__(self,**kwargs):
        super(Concat3DConf, self).__init__(**kwargs)

    # set default params
    @DocInherit
    def default(self):
        self.concat3D_axis = 2

    @DocInherit
    def declare(self):
        self.num_of_inputs = -1
        self.input_ranks =[3]

    @DocInherit
    def inference(self):
        self.output_dim = copy.deepcopy(self.input_dims[0])
        self.output_dim[-1] = 0
        self.output_dim[1] = 0
        if self.concat3D_axis == 1:
            self.output_dim[-1] = self.input_dims[0][-1]
            self.output_dim[1] = sum([input_dim[1] for input_dim in self.input_dims])
        if self.concat3D_axis == 2:
            self.output_dim[-1] = sum([input_dim[-1] for input_dim in self.input_dims])
            self.output_dim[1] = self.input_dims[0][1]

        super(Concat3DConf, self).inference()

    @DocInherit
    def verify(self):
        super(Concat3DConf, self).verify()

        # to check if the ranks of all the inputs are equal
        rank_equal_flag = True
        for i in range(len(self.input_ranks)):
            if self.input_ranks[i] != self.input_ranks[0]:
                rank_equal_flag = False
                break
        if rank_equal_flag == False:
            raise ConfigurationError("For layer Concat3D, the ranks of each inputs should be equal!")

        if self.concat3D_axis == 1:
            # to check if the dimensions of all the inputs are equal
            input_dims = list(self.input_dims)
            dim_equal_flag = True
            for i in range(len(input_dims)):
                if input_dims[i][-1] != input_dims[0][-1]:
                    dim_equal_flag = False
                    break
            if dim_equal_flag == False:
                raise Exception("Concat3D with axis = 1 require that the input dimensions should be the same!")

        # to check if the concat3D_axis is legal
        if self.concat3D_axis not in [1, 2]:
            raise ConfigurationError("For layer Concat3D, the concat axis must be 1 or 2!")

class Concat3D(nn.Module):
    """ Concat3D layer to merge sum of sequences(3D representation)

    Args:
        layer_conf (Concat3DConf): configuration of a layer
    """
    def __init__(self,layer_conf):
        super(Concat3D, self).__init__()
        self.layer_conf = layer_conf

        logging.warning("The length Concat3D layer returns is the length of first input")

    def forward(self, *args):
        """ process inputs

        Args:
            *args: (Tensor): string, string_len, string2, string2_len, ...
                e.g. string (Tensor): [batch_size, seq_len, dim], string_len (Tensor): [batch_size]

        Returns:
            Tensor: [batch_size, seq_len, output_dim], [batch_size]

        """
        result = []
        if self.layer_conf.concat3D_axis == 1:
            for idx, input in enumerate(args):
                if idx % 2 == 0:
                    result.append(input)
        if self.layer_conf.concat3D_axis == 2:
            input_shape = args[0].shape[1]
            for idx, input in enumerate(args):
                if idx % 2 == 0 and input_shape == input.shape[1]:
                    result.append(input)
                else:
                    raise Exception("Concat3D with axis = 2 require that the input sequences length should be the same!")

        return torch.cat(result, self.layer_conf.concat3D_axis), args[1]

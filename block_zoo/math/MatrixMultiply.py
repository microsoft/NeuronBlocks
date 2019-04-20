# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import logging

from ..BaseLayer import BaseConf
from utils.DocInherit import DocInherit
from utils.exceptions import ConfigurationError
import copy

class MatrixMultiplyConf(BaseConf):
    """ Configuration of MatrixMultiply layer

    Args:
        operation(String):  a element of ['common', 'seq_based', 'dim_based'], default is 'dim_based'
                'common' means (batch_size, seq_len, dim)*(batch_size, seq_len, dim)
                'seq_based' means (batch_size, dim, seq_len)*(batch_size, seq_len, dim)
                'dim_based' means (batch_size, seq_len, dim)*(batch_size, dim, seq_len)

    """

    #init the args
    def __init__(self,**kwargs):
        super(MatrixMultiplyConf, self).__init__(**kwargs)

    #set default params
    @DocInherit
    def default(self):
        self.operation = 'dim_based'

    @DocInherit
    def declare(self):
        self.num_of_inputs = 2
        self.input_ranks = [3,3]

    @DocInherit
    def inference(self):
        self.output_dim = copy.deepcopy(self.input_dims[0])
        if self.operation == 'common':
            self.output_dim[-1] = self.input_dims[1][-1]
        if self.operation == 'seq_based':
            self.output_dim[-1] = self.input_dims[1][-1]
            self.output_dim[1] = self.input_dims[0][-1]
        if self.operation == 'dim_based':
            self.output_dim[-1] = self.input_dims[1][1]

        super(MatrixMultiplyConf, self).inference()

    @DocInherit
    def varify(self):
        super(MatrixMultiplyConf, self).varify()
        # # to check if the ranks of all the inputs are equal
        # rank_equal_flag = True
        # for i in range(len(self.input_ranks)):
        #     if self.input_ranks[i] != self.input_ranks[0]:
        #         rank_equal_flag = False
        #         break
        # if rank_equal_flag == False:
        #     raise ConfigurationError("For layer MatrixMultiply, the ranks of each inputs should be equal!")

        # to check if the value of operation is legal
        if self.operation not in ['common', 'seq_based', 'dim_based']:
            raise ConfigurationError("the operation must be one of the 'common', 'seq_based' and 'dim_based'")

class MatrixMultiply(nn.Module):
    """ MatrixMultiply layer to multiply two matrix

    Args:
        layer_conf (MatrixMultiplyConf): configuration of a layer
    """
    def __init__(self, layer_conf):
        super(MatrixMultiply, self).__init__()
        self.layer_conf = layer_conf

        logging.warning("The length MatrixMultiply layer returns is the length of first input")

    def forward(self, *args):
        """ process input

        Args:
            *args: (Tensor): string, string_len, string2, string2_len
                e.g. string (Tensor): [batch_size, seq_len, dim], string_len (Tensor): [batch_size]

        Returns:
            Tensor: [batch_size, seq_len, output_dim], [batch_size]

        """
        if self.layer_conf.operation == 'common':
            if args[0].shape[2] == args[2].shape[1]:
                return torch.matmul(args[0],args[2]),args[1]
            else:
                raise Exception("the dimensions of the two matrix for multiply is illegal")
        if self.layer_conf.operation == 'seq_based':
            if args[0].shape[1] == args[2].shape[1]:
                string = args[0].permute(0,2,1)
                return torch.matmul(string,args[2]),args[1]
            else:
                raise Exception("the dimensions of the two matrix for multiply is illegal")
        if self.layer_conf.operation == 'dim_based':
            if args[0].shape[2] == args[2].shape[2]:
                string = args[2].permute(0,2,1)
                return torch.matmul(args[0],string),args[1]
            else:
                raise Exception("the dimensions of the two matrix for multiply is illegal")

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import logging

from block_zoo.BaseLayer import BaseConf
from utils.DocInherit import DocInherit
from utils.exceptions import ConfigurationError
import copy

class Combination2DConf(BaseConf):
    """ Configuration for combination layer

    Args:
        operations (list):  a subset of ["cosine", "bilinear"].

    """
    def __init__(self, **kwargs):
        super(Combination2DConf, self).__init__(**kwargs)
        if "bilinear" in self.operations:

            weight_bilinear = Parameter(torch.Tensor(self.output_dim, self.output_dim))


    @DocInherit
    def default(self):
        self.operations = ["cosine", "bilinear"]

    @DocInherit
    def declare(self):
        self.num_of_inputs = -1
        self.input_ranks = [-1]

    @DocInherit
    def inference(self):
        self.output_dim = [self.input_dims[0][0], self.input_dims[0][1], self.input_dims[0][1]]

        super(Combination2DConf, self).inference()

    @DocInherit
    def verify(self):
        super(Combination2DConf, self).verify()

        # to check if the ranks of all the inputs are equal
        rank_equal_flag = True
        for i in range(len(self.input_ranks)):
            if self.input_ranks[i] != self.input_ranks[0]:
                rank_equal_flag = False
                break
        if rank_equal_flag == False:
            raise ConfigurationError("For layer Combination, the ranks of each inputs should be consistent!")

        if "difference" in self.operations:
            assert len(self.input_dims) == 2, "Difference operation requires that there should be two inputs"

        if "difference" in self.operations or "dot_multiply" in self.operations:
            input_dims = list(self.input_dims)
            dim_equal_flag = True
            for i in range(len(input_dims)):
                if input_dims[i] != input_dims[0]:
                    dim_equal_flag = False
                    break
            if dim_equal_flag == False:
                raise Exception("Difference and dot multiply require that the input dimensions should be the same")


class Combination2D(nn.Module):
    """ Combination layer to merge the representation of two sequence

    Args:
        layer_conf (CombinationConf): configuration of a layer
    """
    def __init__(self, layer_conf):
        super(Combination2D, self).__init__()
        self.layer_conf = layer_conf

        logging.warning("The length Combination layer returns is the length of first input")

    def forward(self, *args):
        """ process inputs

        Args:
            args (list): [string, string_len, string2, string2_len, ...]
                e.g. string (Variable): [batch_size, dim], string_len (ndarray): [batch_size]

        Returns:
            Variable: [batch_size, output_dim], None

        """

        result = []
        if "cosine" in self.layer_conf.operations:
            string1 = args[0]
            string2 = args[2]
            result_multiply = torch.matmul(string1, string2.transpose(1,2))
            result.append(torch.unsqueeze(result_multiply, 1))

        if "bilinear" in self.layer_conf.operations:
            string1 = args[0]
            string2 = args[2]
            string1 = linear_bi(string1)
            result_multiply = torch.matmul(string1, string2.transpose(1,2))
            result.append(torch.unsqueeze(result_multiply, 1))

        return torch.cat(result, 1), args[1]



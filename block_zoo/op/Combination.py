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

class CombinationConf(BaseConf):
    """ Configuration for combination layer

    Args:
        operations (list):  a subset of ["origin", "difference", "dot_multiply"].
                "origin" means to keep the original representations;\n
                "difference" means abs(sequence1 - sequence2);
                "dot_multiply" means element-wised product;

    """
    def __init__(self, **kwargs):
        super(CombinationConf, self).__init__(**kwargs)

    @DocInherit
    def default(self):
        # supported operations: "origin", "difference", "dot_multiply"
        self.operations = ["origin", "difference", "dot_multiply"]

    @DocInherit
    def declare(self):
        self.num_of_inputs = -1
        self.input_ranks = [-1]

    @DocInherit
    def inference(self):
        self.output_dim = copy.deepcopy(self.input_dims[0])
        self.output_dim[-1] = 0
        if "origin" in self.operations:
            self.output_dim[-1] += sum([input_dim[-1] for input_dim in self.input_dims])
        if "difference" in self.operations:
            self.output_dim[-1] += int(np.mean([input_dim[-1] for input_dim in self.input_dims]))     # difference operation requires dimension of all the inputs should be equal
        if "dot_multiply" in self.operations:
            self.output_dim[-1] += int(np.mean([input_dim[-1] for input_dim in self.input_dims]))     # dot_multiply operation requires dimension of all the inputs should be equal
        super(CombinationConf, self).inference()

    @DocInherit
    def verify(self):
        super(CombinationConf, self).verify()

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


class Combination(nn.Module):
    """ Combination layer to merge the representation of two sequence

    Args:
        layer_conf (CombinationConf): configuration of a layer
    """
    def __init__(self, layer_conf):
        super(Combination, self).__init__()
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
        if "origin" in self.layer_conf.operations:
            for idx, input in enumerate(args):
                if idx % 2 == 0:
                    result.append(input)

        if "difference" in self.layer_conf.operations:
            result.append(torch.abs(args[0] - args[2]))

        if "dot_multiply" in self.layer_conf.operations:
            result_multiply = None
            for idx, input in enumerate(args):
                if idx % 2 == 0:
                    if result_multiply is None:
                        result_multiply = input
                    else:
                        result_multiply = result_multiply * input
            result.append(result_multiply)

        last_dim = len(args[0].size()) - 1
        return torch.cat(result, last_dim), args[1]  #concat on the last dimension


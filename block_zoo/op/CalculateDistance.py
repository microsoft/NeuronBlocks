# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from ..BaseLayer import BaseConf, BaseLayer
from utils.DocInherit import DocInherit
from utils.exceptions import ConfigurationError
import copy


class CalculateDistanceConf(BaseConf):
    """ Configuration of CalculateDistance Layer

    Args:
        operations (list):  a subset of ["cos", "euclidean", "manhattan", "chebyshev"].
    """

    # init the args
    def __init__(self, **kwargs):
        super(CalculateDistanceConf, self).__init__(**kwargs)

    # set default params
    @DocInherit
    def default(self):
        self.operations = ["cos", "euclidean", "manhattan", "chebyshev"]

    @DocInherit
    def declare(self):
        self.num_of_inputs = 2
        self.input_ranks = [2]

    @DocInherit
    def inference(self):
        self.output_dim = copy.deepcopy(self.input_dims[0])
        self.output_dim[-1] = 1

        super(CalculateDistanceConf, self).inference()

    @DocInherit
    def verify(self):
        super(CalculateDistanceConf, self).verify()

        assert len(self.input_dims) == 2, "Operation requires that there should be two inputs"

        # to check if the ranks of all the inputs are equal
        rank_equal_flag = True
        for i in range(len(self.input_ranks)):
            if self.input_ranks[i] != self.input_ranks[0] or self.input_ranks[i] != 2:
                rank_equal_flag = False
                break
        if rank_equal_flag == False:
            raise ConfigurationError("For layer CalculateDistance, the ranks of each inputs should be equal and 2!")


class CalculateDistance(BaseLayer):
    """ CalculateDistance layer to calculate the distance of sequences(2D representation)

    Args:
        layer_conf (CalculateDistanceConf): configuration of a layer
    """

    def __init__(self, layer_conf):
        super(CalculateDistance, self).__init__(layer_conf)
        self.layer_conf = layer_conf


    def forward(self, x, x_len, y, y_len):
        """

        Args:
            x: [batch_size, dim]
            x_len: [batch_size]
            y: [batch_size, dim]
            y_len: [batch_size]
        Returns:
            Tensor: [batch_size, 1], None

        """

        batch_size = x.size()[0]
        if "cos" in self.layer_conf.operations:
            result = F.cosine_similarity(x , y)
        elif "euclidean" in self.layer_conf.operations:
            result = torch.sqrt(torch.sum((x-y)**2, dim=1))
        elif "manhattan" in self.layer_conf.operations:
            result = torch.sum(torch.abs((x - y)), dim=1)
        elif "chebyshev" in self.layer_conf.operations:
            result = torch.abs((x - y)).max(dim=1)
        else:
            raise ConfigurationError("This operation is not supported!")

        result = result.view(batch_size, 1)
        return result, None

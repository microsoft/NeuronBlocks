# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from block_zoo.BaseLayer import BaseLayer, BaseConf
from utils.DocInherit import DocInherit


class Pooling1DConf(BaseConf):
    """

    Args:
        pool_type (str): 'max' or 'mean', default is 'max'.
        stride (int): which axis to conduct pooling, default is 1.
        padding (int): implicit zero paddings on both sides of the input. Can be a single number or a tuple (padH, padW). Default: 0
        window_size (int): the size of the pooling

    """

    def __init__(self, **kwargs):
        super(Pooling1DConf, self).__init__(**kwargs)

    @DocInherit
    def default(self):
        self.pool_type = 'max'  # Supported: ['max', mean']
        self.stride = 1
        self.padding = 0
        self.window_size = 3

    @DocInherit
    def declare(self):
        self.num_of_inputs = 1
        self.input_ranks = [3]


    @DocInherit
    def inference(self):

        self.output_dim = [self.input_dims[0][0]]
        if self.input_dims[0][1] != -1:
            self.output_dim.append(
                (self.input_dims[0][1] + 2 * self.padding - self.window_size) // self.stride + 1)
        else:
            self.output_dim.append(-1)

        self.output_dim.append(self.input_dims[0][-1])
        # DON'T MODIFY THIS
        self.output_rank = len(self.output_dim)

    @DocInherit
    def verify(self):
        super(Pooling1DConf, self).verify()

        necessary_attrs_for_user = ['pool_type']
        for attr in necessary_attrs_for_user:
            self.add_attr_exist_assertion_for_user(attr)

        self.add_attr_value_assertion('pool_type', ['max', 'mean'])

        assert self.output_dim[
                   -1] != -1, "The shape of input is %s , and the input channel number of pooling should not be -1." % (
            str(self.input_dims[0]))


class Pooling1D(BaseLayer):
    """ Pooling layer

    Args:
        layer_conf (PoolingConf): configuration of a layer
    """

    def __init__(self, layer_conf):
        super(Pooling1D, self).__init__(layer_conf)
        self.pool = None
        if layer_conf.pool_type == "max":
            self.pool = nn.MaxPool1d(kernel_size=layer_conf.window_size, stride=layer_conf.stride,
                                     padding=layer_conf.padding)
        elif layer_conf.pool_type == "mean":
            self.pool = nn.AvgPool1d(kernel_size=layer_conf.window_size, stride=layer_conf.stride,
                                     padding=layer_conf.padding)

    def forward(self, string, string_len=None):
        """ process inputs

        Args:
            string (Tensor): tensor with shape: [batch_size, length, feature_dim]
            string_len (Tensor): [batch_size], default is None.

        Returns:
            Tensor: Pooling result of string

        """

        string = string.permute([0, 2, 1]).contiguous()
        string = self.pool(string)
        string = string.permute([0, 2, 1]).contiguous()
        return string, string_len



# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from block_zoo.BaseLayer import BaseLayer, BaseConf
from utils.DocInherit import DocInherit


class Pooling2DConf(BaseConf):
    """

    Args:
        pool_type (str): 'max' or 'mean', default is 'max'.
        stride (int): which axis to conduct pooling, default is 1.
        padding (int): implicit zero paddings on both sides of the input. Can be a single number or a tuple (padH, padW). Default: 0
        window_size (int): the size of the pooling
        activation (string): activation functions, e.g. ReLU

    """
    def __init__(self, **kwargs):
        super(Pooling2DConf, self).__init__(**kwargs)

    @DocInherit
    def default(self):
        self.pool_type = 'max'  # Supported: ['max', mean']
        self.stride = 1
        self.padding = 0
        self.window_size = 3
        
    @DocInherit
    def declare(self):
        self.num_of_inputs = 1
        self.input_ranks = [4]
    
    def check_size(self, value, attr):
        res = value
        if isinstance(value,int):
            res = [value, value]
        elif (isinstance(self.window_size, tuple) or isinstance(self.window_size, list)) and len(value)==2:
            res = list(value)
        else:
            raise AttributeError("The Atrribute `%s' should be given an integer or a list/tuple with length of 2, instead of %s." %(attr,str(value)))
        return res
            
    @DocInherit
    def inference(self):
        
        self.window_size = self.check_size(self.window_size, "window_size")
        self.stride = self.check_size(self.stride, "stride")
        self.padding = self.check_size(self.padding, "padding")
        
        self.output_dim = [self.input_dims[0][0]]
        if self.input_dims[0][1] != -1:
            self.output_dim.append((self.input_dims[0][1] + 2 * self.padding[0] - self.window_size[0]) // self.stride[0] + 1)
        else:
            self.output_dim.append(-1)
        if self.input_dims[0][2] != -1:
            self.output_dim.append((self.input_dims[0][2] + 2 * self.padding[1] - self.window_size[1]) // self.stride[1] + 1)
        else:
            self.output_dim.append(-1)
        # print("pool",self.output_dim)
        self.input_channel_num = self.input_dims[0][-1]

        self.output_dim.append(self.input_dims[0][-1])

        # DON'T MODIFY THIS
        self.output_rank = len(self.output_dim)

    @DocInherit
    def verify(self):
        super(Pooling2DConf, self).verify()

        necessary_attrs_for_user = ['pool_type']
        for attr in necessary_attrs_for_user:
            self.add_attr_exist_assertion_for_user(attr)

        self.add_attr_value_assertion('pool_type', ['max', 'mean'])

        assert all([input_rank >= 4 for input_rank in self.input_ranks]), "Cannot apply a pooling layer on a tensor of which the rank is less than 4. Usually, a tensor whose rank is at least 4, e.g. [batch size, length, width, feature]"

        assert self.output_dim[-1] != -1, "The shape of input is %s , and the input channel number of pooling should not be -1." % (str(self.input_dims[0]))

class Pooling2D(BaseLayer):
    """ Pooling layer

    Args:
        layer_conf (PoolingConf): configuration of a layer
    """
    def __init__(self, layer_conf):
        super(Pooling2D, self).__init__(layer_conf)
        self.pool = None
        if layer_conf.pool_type == "max":
            self.pool = nn.MaxPool2d(kernel_size=layer_conf.window_size,stride=layer_conf.stride,padding=layer_conf.padding)
        elif layer_conf.pool_type == "mean":
            self.pool = nn.AvgPool2d(kernel_size=layer_conf.window_size,stride=layer_conf.stride,padding=layer_conf.padding)

    def forward(self, string, string_len=None):
        """ process inputs

        Args:
            string (Tensor): tensor with shape: [batch_size, length, width, feature_dim]
            string_len (Tensor): [batch_size], default is None.

        Returns:
            Tensor: Pooling result of string

        """

        string = string.permute([0,3,1,2]).contiguous()

        string = self.pool(string)

        string = string.permute([0,2,3,1]).contiguous()

        return string, string_len



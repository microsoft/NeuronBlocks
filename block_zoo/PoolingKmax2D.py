# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from block_zoo.BaseLayer import BaseLayer, BaseConf
from utils.DocInherit import DocInherit


class PoolingKmax2DConf(BaseConf):
    """

    Args:
        pool_type (str): 'max' .
        stride (int): which axis to conduct pooling, default is 1.
        padding (int): implicit zero paddings on both sides of the input. Can be a single number or a tuple (padH, padW). Default: 0
        window_size (int): the size of the pooling
        activation (string): activation functions, e.g. ReLU

    """
    def __init__(self, **kwargs):
        super(PoolingKmax2DConf, self).__init__(**kwargs)

    @DocInherit
    def default(self):
        self.pool_type = 'max'  # Supported: ['max']
        self.stride = 1
        self.padding = 0
        self.window_size = 3
        self.k = 50
        
    @DocInherit
    def declare(self):
        self.num_of_inputs = 1
        #self.input_ranks = [4]
        self.input_ranks = [3]

    
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
        '''
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
        '''
        self.output_dim = [self.input_dims[0][0], -self.input_dims[0][1] * self.k]   #?怎么设定维度，input_dims都是-1


        # DON'T MODIFY THIS
        self.output_rank = len(self.output_dim)

    @DocInherit
    def verify(self):
        super(PoolingKmax2DConf, self).verify()

        necessary_attrs_for_user = ['pool_type']
        for attr in necessary_attrs_for_user:
            self.add_attr_exist_assertion_for_user(attr)

        self.add_attr_value_assertion('pool_type', ['max'])

        #assert all([input_rank >= 4 for input_rank in self.input_ranks]), "Cannot apply a pooling layer on a tensor of which the rank is less than 4. Usually, a tensor whose rank is at least 4, e.g. [batch size, length, width, feature]"

        assert self.output_dim[-1] != -1, "The shape of input is %s , and the input channel number of pooling should not be -1." % (str(self.input_dims[0]))

class PoolingKmax2D(BaseLayer):
    """ Pooling layer

    Args:
        layer_conf (PoolingConf): configuration of a layer
    """
    def __init__(self, layer_conf):
        super(PoolingKmax2D, self).__init__(layer_conf)
        self.k = layer_conf.k

    def forward(self, string, string_len=None):
        """ process inputs

        Args:
            string (Tensor): tensor with shape: [batch_size, length, width, feature_dim]
            string_len (Tensor): [batch_size], default is None.

        Returns:
            Tensor: Pooling result of string

        """
        string = string.view(string.size()[0], string.size()[1], -1)
        index = string.topk(self.k, dim=-1)[1].sort(dim=-1)[0]
        string = string.gather(-1, index)
        string = string.view(string.size()[0], -1)

        return string, string_len



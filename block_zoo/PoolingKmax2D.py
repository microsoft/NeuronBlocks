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
        pool_type (str): 'max', default is 'max'.
        k (int): how many element to reserve.
    """
    def __init__(self, **kwargs):
        super(PoolingKmax2DConf, self).__init__(**kwargs)

    @DocInherit
    def default(self):
        self.pool_type = 'max'  # Supported: ['max']
        self.k = 50
        
    @DocInherit
    def declare(self):
        self.num_of_inputs = 1
        self.input_ranks = [4]

            
    @DocInherit
    def inference(self):
        self.output_dim = [self.input_dims[0][0], self.input_dims[0][3] * self.k] 
        self.output_rank = len(self.output_dim)

    @DocInherit
    def verify(self):
        super(PoolingKmax2DConf, self).verify()
        necessary_attrs_for_user = ['pool_type']
        for attr in necessary_attrs_for_user:
            self.add_attr_exist_assertion_for_user(attr)
        self.add_attr_value_assertion('pool_type', ['max'])

        assert all([input_rank == 4 for input_rank in self.input_ranks]), "Cannot apply a pooling layer on a tensor of which the rank is not 4. Usually, a tensor whose rank is 4, e.g. [batch size, length, width, feature]"
        assert self.output_dim[-1] != -1, "The shape of input is %s , and the input channel number of pooling should not be -1." % (str(self.input_dims[0]))

class PoolingKmax2D(BaseLayer):
    """ Pooling layer
    Args:
        layer_conf (PoolingKmax2DConf): configuration of a layer
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
        string = string.permute(0, 3, 1, 2)
        string = string.view(string.size()[0], string.size()[1], -1)
        index = string.topk(self.k, dim=-1)[1].sort(dim=-1)[0]
        string = string.gather(-1, index)
        string = string.view(string.size()[0], -1)

        return string, string_len

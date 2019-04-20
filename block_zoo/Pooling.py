# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from block_zoo.BaseLayer import BaseLayer, BaseConf
from utils.DocInherit import DocInherit


class PoolingConf(BaseConf):
    """

    Args:
        pool_type (str): 'max' or 'mean', default is 'max'.
        pool_axis (int): which axis to conduct pooling, default is 1.
    """
    def __init__(self, **kwargs):
        super(PoolingConf, self).__init__(**kwargs)

    @DocInherit
    def default(self):
        #self.input_dim = 128
        self.pool_type = 'max'  # Supported: ['max', mean']
        self.pool_axis = 1

    @DocInherit
    def declare(self):
        self.num_of_inputs = 1
        self.input_ranks = [-1]

    @DocInherit
    def inference(self):
        self.output_dim = []
        for idx, dim in enumerate(self.input_dims[0]):
            if idx != self.pool_axis:
                self.output_dim.append(dim)

        # DON'T MODIFY THIS
        self.output_rank = len(self.output_dim)

    @DocInherit
    def verify(self):
        super(PoolingConf, self).verify()

        necessary_attrs_for_user = ['pool_type', 'pool_axis']
        for attr in necessary_attrs_for_user:
            self.add_attr_exist_assertion_for_user(attr)

        self.add_attr_value_assertion('pool_type', ['max', 'mean'])

        assert all([input_rank >= 3 for input_rank in self.input_ranks]), "Cannot apply a pooling layer on a tensor of which the rank is less than 3. Usually, a tensor whose rank is at least 3, e.g. [batch size, sequence length, feature]"

        assert self.output_dim[-1] != -1, "Pooling on the axis %d while the input shape is %s requires that the sequence lengths should be fixed! Please set it on conf/training_params/fixed_lengths" % (self.pool_axis, str(self.input_dims[0]))

class Pooling(BaseLayer):
    """ Pooling layer

    Args:
        layer_conf (PoolingConf): configuration of a layer
    """
    def __init__(self, layer_conf):
        super(Pooling, self).__init__(layer_conf)

    def forward(self, string, string_len=None):
        """ process inputs

        Args:
            string (Tensor): any shape.
            string_len (Tensor): [batch_size], default is None.

        Returns:
            Tensor: Pooling result of string

        """
        if self.layer_conf.pool_type == "mean":
            assert not string_len is None, "Parameter string_len should not be None!"
            string = torch.sum(string, self.layer_conf.pool_axis).squeeze(self.layer_conf.pool_axis)
            if not torch.is_tensor(string_len):
                string_len = torch.FloatTensor(string_len).unsqueeze(1)
            if self.is_cuda():
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                string_len = string_len.to(device)
            string_len = string_len.unsqueeze(1)
            output = string / string_len.expand_as(string).float()
        elif self.layer_conf.pool_type == "max":
            output = torch.max(string, self.layer_conf.pool_axis)[0]

        return output, string_len



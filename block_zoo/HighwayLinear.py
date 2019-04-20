# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import copy
from block_zoo.BaseLayer import BaseLayer, BaseConf
from utils.DocInherit import DocInherit


class HighwayLinearConf(BaseConf):
    """ Configuration of BiLSTM

    Args:
        hidden_dim (int): dimension of hidden state
        dropout (float): dropout rate
        num_layers (int): number of BiLSTM layers
    """
    def __init__(self, **kwargs):
        super(HighwayLinearConf, self).__init__(**kwargs)

    @DocInherit
    def default(self):
        self.num_layers = 1
        self.activation = 'PReLU'

    @DocInherit
    def declare(self):
        self.num_of_inputs = 1
        self.input_ranks = [-1]

    @DocInherit
    def inference(self):
        self.output_dim = copy.deepcopy(self.input_dims[0])
        super(HighwayLinearConf, self).inference()  # PUT THIS LINE AT THE END OF inference()

    @DocInherit
    def verify(self):
        super(HighwayLinearConf, self).verify()

        necessary_attrs_for_user = ['num_layers', 'activation']
        for attr in necessary_attrs_for_user:
            self.add_attr_exist_assertion_for_user(attr)


class HighwayLinear(BaseLayer):
    """ A `Highway layer <https://arxiv.org/abs/1505.00387>`_ does a gated combination of a linear
    transformation and a non-linear transformation of its input.  :math:`y = g * x + (1 - g) *
    f(A(x))`, where :math:`A` is a linear transformation, :math:`f` is an element-wise
    non-linearity, and :math:`g` is an element-wise gate, computed as :math:`sigmoid(B(x))`.
    This module will apply a fixed number of highway layers to its input, returning the final
    result.

    Args:
        layer_conf (HighwayLinearConf): configuration of a layer
    """
    def __init__(self, layer_conf):
        super(HighwayLinear, self).__init__(layer_conf)
        self.layer_conf = layer_conf
        self.layers = torch.nn.ModuleList([torch.nn.Linear(layer_conf.input_dims[0][-1], layer_conf.input_dims[0][-1] * 2) for _ in range(layer_conf.num_layers)])
        self.activation = eval("nn." + layer_conf.activation)()

    def forward(self, string, string_len):
        """ process inputs

        Args:
            string (Tensor): [batch_size, seq_len, dim]
            string_len (Tensor): [batch_size]

        Returns:
            Tensor: [batch_size, seq_len, 2 * hidden_dim]

        """
        current_input = string
        for layer in self.layers:
            projected_input = layer(current_input)
            linear_part = current_input
            # NOTE: if you modify this, think about whether you should modify the initialization above, too.
            nonlinear_part, gate = projected_input.chunk(2, dim=-1)
            nonlinear_part = self.activation(nonlinear_part)
            gate = torch.sigmoid(gate)
            current_input = gate * linear_part + (1 - gate) * nonlinear_part
        return current_input, string_len


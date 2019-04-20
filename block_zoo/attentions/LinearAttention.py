# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np

from block_zoo.BaseLayer import BaseLayer, BaseConf
from utils.DocInherit import DocInherit

class LinearAttentionConf(BaseConf):
    """Configuration for Linear attention layer

    Args:
        keep_dim (bool): Whether to sum up the sequence representation along the sequence axis.
                if False, the layer would return (batch_size, dim)
                if True, the layer would keep the same dimension as input, thus return (batch_size, sequence_length, dim)
    """
    def __init__(self, **kwargs):
        super(LinearAttentionConf, self).__init__(**kwargs)

    @DocInherit
    def default(self):
        self.keep_dim = False

    @DocInherit
    def declare(self):
        self.num_of_inputs = 1
        self.input_ranks = [-1]

    @DocInherit
    def inference(self):
        self.attention_weight_dim = self.input_dims[0][-1]

        if self.keep_dim:
            self.output_dim = copy.deepcopy(self.input_dims[0])
        else:
            self.output_dim = []
            for idx, dim in enumerate(self.input_dims[0]):
                if idx != len(self.input_dims[0]) - 2:
                    self.output_dim.append(dim)

        super(LinearAttentionConf, self).inference()  # PUT THIS LINE AT THE END OF inference()

    @DocInherit
    def verify_before_inference(self):
        super(LinearAttentionConf, self).verify_before_inference()
        necessary_attrs_for_user = ['keep_dim']
        for attr in necessary_attrs_for_user:
            self.add_attr_exist_assertion_for_user(attr)

    @DocInherit
    def verify(self):
        super(LinearAttentionConf, self).verify()
        necessary_attrs_for_user = ['attention_weight_dim', 'keep_dim']
        for attr in necessary_attrs_for_user:
            self.add_attr_exist_assertion_for_user(attr)

        type_checks = [('attention_weight_dim', int),
                       ('keep_dim', bool)]
        for attr, attr_type in type_checks:
            self.add_attr_type_assertion(attr, attr_type)

        assert self.input_ranks[0] in set([2, 3]) and not (self.input_ranks[0] == 2 and self.keep_dim == False)


class LinearAttention(BaseLayer):
    """  Linear attention.
    Combinate the original sequence along the sequence_length dimension.

    Args:
        layer_conf (LinearAttentionConf): configuration of a layer

    """
    def __init__(self, layer_conf):
        """

        Args:
            layer_conf (LinearAttentionConf): configuration of a layer
        """
        super(LinearAttention, self).__init__(layer_conf)
        self.attention_weight = nn.Parameter(torch.FloatTensor(torch.randn(self.layer_conf.attention_weight_dim, 1)))

    def forward(self, string, string_len=None):
        """ process inputs

        Args:
            string (Variable): (batch_size, sequence_length, dim)
            string_len (ndarray or None): [batch_size]

        Returns:
            Variable:
                if keep_dim == False:
                    Output dimention: (batch_size, dim)
                else:
                    just reweight along the sequence_length dimension: (batch_size, sequence_length, dim)

        """
        attention_weight = torch.mm(string.contiguous().view(string.shape[0] * string.shape[1], string.shape[2]), self.attention_weight)
        attention_weight = nn.functional.softmax(attention_weight.view(string.shape[0], string.shape[1]), dim=1)

        attention_tiled = attention_weight.unsqueeze(2).expand_as(string)
        string_reweighted = torch.mul(string, attention_tiled)
        if self.layer_conf.keep_dim is False:
            string_reweighted = torch.sum(string_reweighted, 1)

        return string_reweighted, string_len



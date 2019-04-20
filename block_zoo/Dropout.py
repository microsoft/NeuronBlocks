# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import copy

from block_zoo.BaseLayer import BaseLayer, BaseConf
from utils.DocInherit import DocInherit

class DropoutConf(BaseConf):
    """ Configuration for Dropout

    Args:
        dropout (float): dropout rate, probability of an element to be zeroed

    Returns:

    """
    def __int__(self, **kwargs):

        super(DropoutConf, self).__init__(**kwargs)

    @DocInherit
    def default(self):
        self.dropout = 0.5

    @DocInherit
    def declare(self):
        self.num_of_inputs = 1
        self.input_ranks = [-1]

    @DocInherit
    def inference(self):
        self.output_dim = copy.deepcopy(self.input_dims[0])

        super(DropoutConf, self).inference()  # PUT THIS LINE AT THE END OF inference()

    @DocInherit
    def verify(self):
        super(DropoutConf, self).verify()

        necessary_attrs_for_user = ['dropout']
        for attr in necessary_attrs_for_user:
            self.add_attr_exist_assertion_for_user(attr)

        range_checks = [('dropout', (0, 1), (True, True))]
        for attr, ranges, bound_legal in range_checks:
            self.add_attr_range_assertion(attr, ranges, bound_legal)


class Dropout(BaseLayer):
    """ Dropout

    Args:
        layer_conf (DropoutConf): configuration of a layer
    """
    def __init__(self, layer_conf):
        super(Dropout, self).__init__(layer_conf)
        self.dropout_layer = nn.Dropout(layer_conf.dropout)

    def forward(self, string, string_len=None):
        """ process inputs

        Args:
            string (Tensor): any shape.
            string_len (Tensor): [batch_size], default is None.

        Returns:
            Tensor: has the same shape as string.
        """
        string_out = self.dropout_layer(string)
        return string_out, string_len


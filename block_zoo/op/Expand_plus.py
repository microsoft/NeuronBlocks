# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

# Come from http://www.hangli-hl.com/uploads/3/1/6/8/3168008/hu-etal-nips2014.pdf [ARC-II]

import torch
import torch.nn as nn
import copy

from block_zoo.BaseLayer import BaseLayer, BaseConf
from utils.DocInherit import DocInherit
from utils.exceptions import ConfigurationError

class Expand_plusConf(BaseConf):
    """Configuration for Expand_plus layer

    """
    def __init__(self, **kwargs):
        super(Expand_plusConf, self).__init__(**kwargs)

    @DocInherit
    def default(self):
        self.operation = 'Plus'

    @DocInherit
    def declare(self):
        self.num_of_inputs = 2
        self.input_ranks = [3, 3]

    @DocInherit
    def inference(self):
        self.output_dim = copy.deepcopy(self.input_dims[0])
        if self.input_dims[0][1] == -1 or self.input_dims[1][1] == -1:
            raise ConfigurationError("For Expand_plus layer, the sequence length should be fixed")
        self.output_dim.insert(2, self.input_dims[1][1])   # y_len
        super(Expand_plusConf, self).inference()  # PUT THIS LINE AT THE END OF inference()

    @DocInherit
    def verify(self):
        super(Expand_plusConf, self).verify()


class Expand_plus(BaseLayer):
    """  Expand_plus layer
    Given sequences X and Y, put X and Y expand_dim, and then add.

    Args:
        layer_conf (Expand_plusConf): configuration of a layer

    """
    def __init__(self, layer_conf):

        super(Expand_plus, self).__init__(layer_conf)
        assert layer_conf.input_dims[0][-1] == layer_conf.input_dims[1][-1]


    def forward(self, x, x_len, y, y_len):
        """

        Args:
            x:      [batch_size, x_max_len, dim].
            x_len:  [batch_size], default is None.
            y:      [batch_size, y_max_len, dim].
            y_len:  [batch_size], default is None.

        Returns:
            output: batch_size, x_max_len, y_max_len, dim].

        """

        x_new = torch.stack([x]*y.size()[1], 2) # [batch_size, x_max_len, y_max_len, dim]
        y_new = torch.stack([y]*x.size()[1], 1) # [batch_size, x_max_len, y_max_len, dim]

        return x_new + y_new, None



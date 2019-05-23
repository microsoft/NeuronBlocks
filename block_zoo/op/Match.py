# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch.nn as nn
import copy

from block_zoo.BaseLayer import BaseLayer, BaseConf
from utils.DocInherit import DocInherit
from utils.exceptions import ConfigurationError

class MatchConf(BaseConf):
    """Configuration for MatchAttention layer

    """
    def __init__(self, **kwargs):
        super(MatchConf, self).__init__(**kwargs)

    @DocInherit
    def default(self):
        self.activation = 'PReLU'

    @DocInherit
    def declare(self):
        self.num_of_inputs = 2
        self.input_ranks = [3, 3]

    @DocInherit
    def inference(self):
        self.output_dim = copy.deepcopy(self.input_dims[0])
        if self.input_dims[0][1] == -1 or self.input_dims[1][1] == -1:
            raise ConfigurationError("For Match layer, the sequence length should be fixed")
        self.output_dim[-1] = self.input_dims[1][1]     # y_len
        super(MatchConf, self).inference()  # PUT THIS LINE AT THE END OF inference()

    @DocInherit
    def verify(self):
        super(MatchConf, self).verify()


class Match(BaseLayer):
    """  Match layer
    Given sequences X and Y, match sequence Y to each element in X.

    Args:
        layer_conf (MatchConf): configuration of a layer

    """
    def __init__(self, layer_conf):

        super(Match, self).__init__(layer_conf)
        assert layer_conf.input_dims[0][-1] == layer_conf.input_dims[1][-1]
        self.linear = nn.Linear(layer_conf.input_dims[0][-1], layer_conf.input_dims[0][-1])
        if layer_conf.activation:
            self.activation = eval("nn." + self.layer_conf.activation)()
        else:
            self.activation = None
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, x_len, y, y_len):
        """

        Args:
            x:      [batch_size, x_max_len, dim].
            x_len:  [batch_size], default is None.
            y:      [batch_size, y_max_len, dim].
            y_len:  [batch_size], default is None.

        Returns:
            output: has the same shape as x.

        """

        x_proj = self.linear(x)  # [batch_size, x_max_len, dim]
        y_proj = self.linear(y)  # [batch_size, y_max_len, dim]
        if self.activation:
            x_proj = self.activation(x_proj)
            y_proj = self.activation(y_proj)
        scores = x_proj.bmm(y_proj.transpose(2, 1))     # [batch_size, x_max_len, y_max_len]
        return scores, x_len



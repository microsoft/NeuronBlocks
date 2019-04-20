# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import numpy as np
import torch.nn as nn
import copy

from block_zoo.BaseLayer import BaseLayer, BaseConf
from utils.DocInherit import DocInherit


class MatchAttentionConf(BaseConf):
    """Configuration for MatchAttention layer

    """
    def __init__(self, **kwargs):
        super(MatchAttentionConf, self).__init__(**kwargs)

    @DocInherit
    def default(self):
        pass

    @DocInherit
    def declare(self):
        self.num_of_inputs = 2
        self.input_ranks = [3, 3]

    @DocInherit
    def inference(self):
        self.output_dim = copy.deepcopy(self.input_dims[0])
        super(MatchAttentionConf, self).inference()  # PUT THIS LINE AT THE END OF inference()

    @DocInherit
    def verify(self):
        super(MatchAttentionConf, self).verify()


class MatchAttention(BaseLayer):
    """  MatchAttention layer for DrQA
    [paper]  https://arxiv.org/abs/1704.00051

    Given sequences X and Y, match sequence Y to each element in X.

    Args:
        layer_conf (MatchAttentionConf): configuration of a layer

    """
    def __init__(self, layer_conf):

        super(MatchAttention, self).__init__(layer_conf)
        assert layer_conf.input_dims[0][-1] == layer_conf.input_dims[1][-1]
        self.linear = nn.Linear(layer_conf.input_dims[0][-1], layer_conf.input_dims[0][-1])
        self.relu = nn.ReLU()
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

        x_proj = self.relu(self.linear(x))  # [batch_size, x_max_len, dim]
        y_proj = self.relu(self.linear(y))  # [batch_size, y_max_len, dim]
        scores = x_proj.bmm(y_proj.transpose(2, 1))     # [batch_size, x_max_len, y_max_len]

        # batch_size, y_max_len, _ = y.size()
        # y_length = y_len.cpu().numpy()
        # y_mask = np.ones((batch_size, y_max_len))
        # for i, single_len in enumerate(y_length):
        #     y_mask[i][:single_len] = 0
        # y_mask = torch.from_numpy(y_mask).byte().to(scores.device)
        # y_mask = y_mask.unsqueeze(1).expand(scores.size())
        # scores.data.masked_fill_(y_mask.data, float('-inf'))

        alpha = self.softmax(scores)    # [batch_size, x_max_len, y_len]
        output = alpha.bmm(y)   # [batch_size, x_max_len, dim]

        return output, x_len



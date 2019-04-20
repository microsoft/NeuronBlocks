# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch.nn as nn
import copy

from block_zoo.BaseLayer import BaseLayer, BaseConf
from utils.DocInherit import DocInherit

class BilinearAttentionConf(BaseConf):
    """Configuration for Bilinear attention layer

    """
    def __init__(self, **kwargs):
        super(BilinearAttentionConf, self).__init__(**kwargs)

    @DocInherit
    def default(self):
        pass

    @DocInherit
    def declare(self):
        self.num_of_inputs = 2
        self.input_ranks = [3, 2]

    @DocInherit
    def inference(self):
        self.output_dim = copy.deepcopy(self.input_dims[0])
        self.output_dim[-1] = 1
        super(BilinearAttentionConf, self).inference()  # PUT THIS LINE AT THE END OF inference()

    @DocInherit
    def verify(self):
        super(BilinearAttentionConf, self).verify()


class BilinearAttention(BaseLayer):
    """  BilinearAttention layer for DrQA
    [paper]  https://arxiv.org/abs/1704.00051
    Args:
        layer_conf (BilinearAttentionConf): configuration of a layer

    """
    def __init__(self, layer_conf):
        super(BilinearAttention, self).__init__(layer_conf)
        self.linear = nn.Linear(layer_conf.input_dims[1][-1], layer_conf.input_dims[0][-1])

    def forward(self, x, x_len, y, y_len):
        """ process inputs

        Args:
            x (Tensor):      [batch_size, x_len, x_dim].
            x_len (Tensor):  [batch_size], default is None.
            y (Tensor):      [batch_size, y_dim].
            y_len (Tensor):  [batch_size], default is None.
        Returns:
            output: [batch_size, x_len, 1].
            x_len:

        """

        Wy = self.linear(y)     # [batch_size, x_dim]
        xWy = x.bmm(Wy.unsqueeze(2))    # [batch_size, x_len, 1]

        return xWy, x_len



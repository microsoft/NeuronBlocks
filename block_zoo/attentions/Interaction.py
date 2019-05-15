# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from block_zoo.BaseLayer import BaseLayer, BaseConf
from utils.DocInherit import DocInherit
from utils.common_utils import transfer_to_gpu


class InteractionConf(BaseConf):
    """Configuration of Interaction Layer

    Args:
        hidden_dim (int): dimension of hidden state
        dropout (float): dropout rate

    """
    def __init__(self, **kwargs):
        super(InteractionConf, self).__init__(**kwargs)

    @DocInherit
    def default(self):
        self.hidden_dim = 128
        self.dropout = 0.0
        self.matching_type = 'general'

    @DocInherit
    def declare(self):
        self.num_of_inputs = 2
        self.input_ranks = [3, 3]

    @DocInherit
    def inference(self):
        shape1 = self.input_dims[0]
        shape2 = self.input_dims[1]
        if shape1[1] == -1 or shape2[1] == -1:
            raise ConfigurationError("For Interaction layer, the sequence length should be fixed")
        # print(shape1,shape2)
        self.output_dim = None
        if self.matching_type in ['mul', 'plus', 'minus']:        
            self.output_dim = [shape1[0], shape1[1], shape2[1], shape1[2]] 
        elif self.matching_type in ['dot', 'general']:
            self.output_dim = [shape1[0], shape1[1], shape2[1], 1]
        elif self.matching_type == 'concat':
            self.output_dim = [shape1[0], shape1[1], shape2[1], shape1[2] + shape2[2]]
        else:
            raise ValueError(f"Invalid `matching_type`."
                             f"{self.matching_type} received."
                             f"Must be in `mul`, `general`, `plus`, `minus` "
                             f"`dot` and `concat`.")
        # print(self.output_dim)
        super(InteractionConf, self).inference()  # PUT THIS LINE AT THE END OF inference()

    @DocInherit
    def verify(self):
        super(InteractionConf, self).verify()
        assert hasattr(self, 'hidden_dim'), "Please define hidden_dim attribute of BiGRUConf in default() or the configuration file"
        assert hasattr(self, 'dropout'), "Please define dropout attribute of BiGRUConf in default() or the configuration file"
        assert hasattr(self, 'matching_type'), "Please define matching_type attribute of BiGRUConf in default() or the configuration file"
        assert self.matching_type in ['general', 'dot', 'mul', 'plus', 'minus', 'add'], "Invalid `matching_type`{self.matching_type} received. Must be in `mul`, `general`, `plus`, `minus`, `dot` and `concat`."


class Interaction(BaseLayer):
    """Bidirectional GRU

    Args:
        layer_conf (BiGRUConf): configuration of a layer
    """
    def __init__(self, layer_conf):
        super(Interaction, self).__init__(layer_conf)
        self.matching_type = layer_conf.matching_type
        if self.matching_type == 'general':
            self.linear_in = nn.Linear(layer_conf.hidden_dim, layer_conf.hidden_dim, bias=False)

    def forward(self, string1, string1_len, string2, string2_len):
        """ process inputs

        Args:
            string1 (Tensor): [batch_size, seq_len1, dim]
            string1_len (Tensor): [batch_size]
            string2 (Tensor): [batch_size, seq_len2, dim]
            string2_len (Tensor): [batch_size]

        Returns:
            Tensor: [batch_size, seq_len1, seq_len2]

        """
        padded_seq_len1 = string1.shape[1]
        padded_seq_len2 = string2.shape[1]
        x1 = string1
        x2 = string2
        result = None


        if self.matching_type == 'dot' or self.matching_type == 'general':
            # if self._normalize:
            #     x1 = K.l2_normalize(x1, axis=2)
            #     x2 = K.l2_normalize(x2, axis=2)
            if self.matching_type=='general':
                x1 = x1.view(-1, self.layer_conf.hidden_dim)
                x1 = self.linear_in(x1)
                x1 = x1.view(-1, padded_seq_len1, self.layer_conf.hidden_dim)
            result = torch.bmm(x1, x2.transpose(1, 2).contiguous())
            result = torch.unsqueeze(result, -1)
            # print("result", result.size())
        else:
            if self.matching_type == 'mul':
                def func(x, y):
                    return x * y
            elif self.matching_type == 'plus':
                def func(x, y):
                    return x + y
            elif self.matching_type == 'minus':
                def func(x, y):
                    return x - y
            elif self.matching_type == 'concat':
                def func(x, y):
                    return torch.concat([x, y], axis=-1)
            else:
                raise ValueError(f"Invalid matching type."
                                 f"{self.matching_type} received."
                                 f"Mut be in `dot`, `general`, `mul`, `plus`, "
                                 f"`minus` and `concat`.")
            x1_exp = torch.stack([x1] * padded_seq_len2, dim=2)
            x2_exp = torch.stack([x2] * padded_seq_len1, dim=1)
            result = func(x1_exp, x2_exp)

        return result, padded_seq_len1


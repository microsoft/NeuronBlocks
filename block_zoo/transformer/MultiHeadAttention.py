# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import math

from block_zoo.BaseLayer import BaseLayer,BaseConf
from utils.DocInherit import DocInherit
import copy

class MultiHeadAttentionConf(BaseConf):
    """ Configuration of MultiHeadAttention Layer

    Args:
        n_head (int): the head number of attention
        scale (bool): if need to scale
        attn_dropout (float): the dropout of attention layer
        resid_dropout (float): the dropout of last Linear
    """
    
    def __init__(self,**kwargs):

        super(MultiHeadAttentionConf, self).__init__(**kwargs)

    @DocInherit
    def default(self):
        self.n_head = 12
        self.scale = True
        self.attn_dropout = 0.1
        self.resid_dropout = 0.1

    @DocInherit
    def declare(self):
        self.num_of_inputs = 1
        self.input_ranks = [3]

    @DocInherit
    def inference(self):
        self.output_dim = copy.deepcopy(self.input_dims[0])
        super(MultiHeadAttentionConf, self).inference()

    @DocInherit
    def verify(self):
        super(MultiHeadAttentionConf, self).verify()

class MultiHeadAttention(nn.Module):
    """ MultiHeadAttention Layer

    Args:
        layer_conf (MultiHeadAttentionConf): configuration of a layer

    """
    def __init__(self, layer_conf):
        super(MultiHeadAttention, self).__init__()
        self.layer_conf = layer_conf
        self.split_size = self.layer_conf.input_dims[0][-1]
        self.n_state = self.layer_conf.input_dims[0][-1]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert self.n_state % self.layer_conf.n_head == 0

        self.c_attn = nn.Linear(self.layer_conf.input_dims[0][-1],self.n_state * 3)
        self.c_proj = nn.Linear(self.layer_conf.input_dims[0][-1],self.n_state)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k).to(self.device)
        if self.layer_conf.scale:
            w = w / math.sqrt(v.size(-1))
        w = w * self.b + -1e9 * (1 - self.b)
        w = nn.Softmax(dim=-1)(w)
        w = nn.Dropout(self.layer_conf.attn_dropout)(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.layer_conf.n_head, x.size(-1) // self.layer_conf.n_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, string, string_len):
        """ process input

        Args:
            string, string_len
            e.g. string (Tensor): [batch_size, seq_len, dim], string_len (Tensor): [batch_size]

        Returns:
            Tensor: [batch_size, seq_len, output_dim], [batch_size]
        """
        self.register_buffer('b', torch.tril(torch.ones(string.shape[1], string.shape[1]).to(self.device)).view(1, 1, string.shape[1], string.shape[1]))
        x = self.c_attn(string)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = nn.Dropout(self.layer_conf.resid_dropout)(a)
        return a, string_len
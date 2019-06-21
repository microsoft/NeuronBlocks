# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn

from block_zoo.BaseLayer import BaseLayer, BaseConf
from block_zoo.transformer.MLP import MLP, MLPConf
from block_zoo.transformer.MultiHeadAttention import MultiHeadAttention, MultiHeadAttentionConf
from block_zoo.normalizations.LayerNorm import LayerNorm, LayerNormConf
from utils.DocInherit import DocInherit
import copy

class TransformerConf(BaseConf):
    """ Configuration of Transformer

    Args:
        attention (str): attention name
        attention_conf (dict): configurations of attention
        layernorm_1 (str): layernorm1 name
        layernorm1_conf (dict): configurations of layernorm1
        mlp (str): mlp name
        mlp_conf (dict): configuration of mlp
        layernorm_2 (str): layernorm2 name
        layernorm2_conf (dict): configurations of layernorm2
        n_layer (int) layer num of transformer

    """
    def __init__(self, **kwargs):
        self.attention_conf_cls = eval(kwargs['attention'] + "Conf")(**kwargs['attention_conf'])
        self.layernorm1_conf_cls = eval(kwargs['layernorm_1'] + "Conf")(**kwargs['layernorm1_conf'])
        self.mlp_conf_cls = eval(kwargs['mlp'] + "Conf")(**kwargs['mlp_conf'])
        self.layernorm2_conf_cls = eval(kwargs['layernorm_2'] + "Conf")(**kwargs['layernorm2_conf'])

        super(TransformerConf, self).__init__(**kwargs)

    @DocInherit
    def default(self):
        self.attention_name = "MultiHeadAttention"
        self.layernorm1_name = "LayerNorm"
        self.mlp_name = "MLP"
        self.layernorm2_name = "LayerNorm"

        self.attention_conf = dict()
        self.layernorm1_conf = dict()
        self.mlp_conf = dict()
        self.layernorm2_conf = dict()

        self.n_layer = 12

    @DocInherit
    def declare(self):
        self.num_of_inputs = 1
        self.input_ranks = [3]

    @DocInherit
    def inference(self):
        self.attention_conf_cls.use_gpu = self.use_gpu
        self.layernorm1_conf_cls.use_gpu = self.use_gpu
        self.mlp_conf_cls.use_gpu = self.use_gpu
        self.layernorm2_conf_cls.use_gpu = self.use_gpu

        self.attention_conf_cls.input_dims = copy.deepcopy(self.input_dims)
        self.attention_conf_cls.inference()

        self.layernorm1_conf_cls.input_dims = [self.attention_conf_cls.output_dim]
        self.layernorm1_conf_cls.inference()

        self.mlp_conf_cls.input_dims = [self.layernorm1_conf_cls.output_dim]
        self.mlp_conf_cls.inference()

        self.layernorm2_conf_cls.input_dims = [self.mlp_conf_cls.output_dim]
        self.layernorm2_conf_cls.inference()

        self.output_dim = self.layernorm2_conf_cls.output_dim

        super(TransformerConf, self).inference()

    @DocInherit
    def verify(self):
        super(TransformerConf, self).verify()
        self.attention_conf_cls.verify()
        self.layernorm1_conf_cls.verify()
        self.layernorm2_conf_cls.verify()
        self.mlp_conf_cls.verify()

class Transformer(nn.Module):
    """ Transformer layer

    Args:
        layer_conf (TransformerConf): configuration of a layer
    """
    def __init__(self, layer_conf):
        super(Transformer, self).__init__()
        self.layer_conf = layer_conf

        self.transformer_layer = nn.ModuleList([copy.deepcopy(nn.ModuleList([eval(layer_conf.attention_name)(layer_conf.attention_conf_cls),
                                    eval(layer_conf.layernorm1_name)(layer_conf.layernorm1_conf_cls),
                                    eval(layer_conf.mlp_name)(layer_conf.mlp_conf_cls),
                                    eval(layer_conf.layernorm2_name)(layer_conf.layernorm2_conf_cls)])) for _ in range(self.layer_conf.n_layer)])
        # self.attention_layers = nn.ModuleList()
        # self.layernorm1_layers = nn.ModuleList()
        # self.mlp_layers = nn.ModuleList()
        # self.layernorm2_layers = nn.ModuleList()
        #
        # for i in range(self.layer_conf.n_layer):
        #     self.attention_layers.append(eval(layer_conf.attention_name)(layer_conf.attention_conf_cls))
        #     self.layernorm1_layers.append(eval(layer_conf.layernorm1_name)(layer_conf.layernorm1_conf_cls))
        #     self.mlp_layers.append(eval(layer_conf.mlp_name)(layer_conf.mlp_conf_cls))
        #     self.layernorm2_layers.append(eval(layer_conf.layernorm2_name)(layer_conf.layernorm2_conf_cls))

        # self.attention = eval(layer_conf.attention_name)(layer_conf.attention_conf_cls)
        # self.layernorm1 = eval(layer_conf.layernorm1_name)(layer_conf.layernorm1_conf_cls)
        # self.mlp = eval(layer_conf.mlp_name)(layer_conf.mlp_conf_cls)
        # self.layernorm2 = eval(layer_conf.layernorm2_name)(layer_conf.layernorm2_conf_cls)

    def forward(self, string, string_len):
        """ process input

        Args:
            string (Tensor): [batch_size, seq_len, dim]
            string_len (Tensor): [batch_size]
        Returns:
            Tensor : [batch_size, seq_len, output_dim], [batch_size]
        """
        h = string
        l = string_len
        # for i in range(self.layer_conf.n_layer):
        #     a, a_len = self.attention_layers[i](h,l)
        #     n, n_len = self.layernorm1_layers[i](a+h, a_len)
        #     m, m_len = self.mlp_layers[i](n, n_len)
        #     h, l = self.layernorm2_layers[i](m+n, m_len)
        for block in self.transformer_layer:
            a, a_len = block[0](h,1)
            n, n_len = block[1](a+h, a_len)
            m, m_len = block[2](n, n_len)
            h, l = block[3](m + n, m_len)
        return h, l




# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import copy
import numpy as np
from utils.DocInherit import DocInherit

from block_zoo.BaseLayer import BaseLayer, BaseConf
from utils.exceptions import ConfigurationError

class FullAttentionConf(BaseConf):
    def __init__(self, **kwargs):
        super(FullAttentionConf, self).__init__(**kwargs)

    @DocInherit
    def default(self):
        self.hidden_dim = 128
        self.activation = 'ReLU'

    @DocInherit
    def declare(self):
        self.num_of_inputs = 4
        self.input_ranks = [3]

    @DocInherit
    def inference(self):
        self.output_dim = copy.deepcopy(self.input_dims[1])     # e.g. use query to represent passage, there fore the output dim depends on query's dim
        super(FullAttentionConf, self).inference()  # PUT THIS LINE AT THE END OF inference()

    @DocInherit
    def verify_before_inference(self):
        super(FullAttentionConf, self).verify_before_inference()
        necessary_attrs_for_user = ['hidden_dim']
        for attr in necessary_attrs_for_user:
            self.add_attr_exist_assertion_for_user(attr)

    def verify(self):
        super(FullAttentionConf, self).verify()

        supported_activation_pytorch = [None, 'Sigmoid', 'Tanh', 'ReLU', 'PReLU', 'ReLU6', 'LeakyReLU', 'LogSigmoid',
                                        'ELU',
                                        'SELU', 'Threshold', 'Hardtanh', 'Softplus', 'Softshrink', 'Softsign',
                                        'Tanhshrink', 'Softmin',
                                        'Softmax', 'Softmax2d', 'LogSoftmax']
        value_checks = [('activation', supported_activation_pytorch)]
        for attr, legal_values in value_checks:
            self.add_attr_value_assertion(attr, legal_values)


class FullAttention(BaseLayer):
    """ Full-aware fusion of:
            Via, U., With, T., & To, P. (2018). Fusion Net: Fusing Via Fully-Aware Attention with Application to Machine Comprehension, 1–17.

    """
    def __init__(self, layer_conf):
        super(FullAttention, self).__init__(layer_conf)
        self.layer_conf.hidden_dim = layer_conf.hidden_dim
        self.linear = nn.Linear(layer_conf.input_dims[2][-1], layer_conf.hidden_dim, bias=False)        # this requires that input_dims[0][-1] == input_dims[1][-1]
        if layer_conf.input_dims[2][-1] == layer_conf.input_dims[3][-1]:
            self.linear2 = self.linear
        else:
            self.linear2 = nn.Linear(layer_conf.input_dims[3][-1], layer_conf.hidden_dim, bias=False)
        self.linear_final = Parameter(torch.ones(1, layer_conf.hidden_dim), requires_grad=True)

        self.activation = eval("nn." + layer_conf.activation)()

    def forward(self, string1, string1_len, string2, string2_len, string1_HoW, string1_How_len, string2_HoW, string2_HoW_len):
        """ To get representation of string1, we use string1 and string2 to obtain attention weights and use string2 to represent string1

        Note: actually, the semantic information of string1 is not used, we only need string1's seq_len information

        Args:
            string1: [batch size, seq_len, input_dim1]
            string1_len: [batch_size]
            string2: [batch size, seq_len, input_dim2]
            string2_len: [batch_size]
            string1_HoW: [batch size, seq_len, att_dim1]
            string1_HoW_len: [batch_size]
            string2_HoW: [batch size, seq_len, att_dim2]
            string2_HoW_len: [batch_size]

        Returns:
            string1's representation
            string1_len

        """
        string1_key = self.activation(self.linear(string1_HoW.contiguous().view(-1, string1_HoW.size()[2])))     #[bs * seq_len, atten_dim1] -> [bs * seq_len, hidden_dim]
        string2_key = self.activation(self.linear2(string2_HoW.contiguous().view(-1, string2_HoW.size()[2])))    #[bs * seq_len, atten_dim2] -> [bs * seq_len, hidden_dim]
        final_v = self.linear_final.expand_as(string2_key)
        string2_key = final_v * string2_key

        string1_rep = string1_key.view(-1, string1.size(1), 1, self.layer_conf.hidden_dim).transpose(1, 2).contiguous().view(-1, string1.size(1), self.layer_conf.hidden_dim)        # get [bs, seq_len, hidden_dim]
        string2_rep = string2_key.view(-1, string2.size(1), 1, self.layer_conf.hidden_dim).transpose(1, 2).contiguous().view(-1, string2.size(1), self.layer_conf.hidden_dim)       # get [bs, seq_len, hidden_dim]

        scores = string1_rep.bmm(string2_rep.transpose(1, 2)).view(-1, 1, string1.size(1), string2.size(1)) # [bs, 1, seq_len1, seq_len2]

        string2_len_np = string2_len.cpu().numpy()
        if torch.cuda.device_count() > 1:
            # otherwise, it will raise a Exception because the length inconsistence
            string2_max_len = string2.shape[1]
        else:
            string2_max_len = string2_len_np.max()
        string2_mask = np.array([[0] * num + [1] * (string2_max_len - num) for num in string2_len_np])
        string2_mask = torch.from_numpy(string2_mask).unsqueeze(1).unsqueeze(2).expand_as(scores)
        if self.is_cuda():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            string2_mask = string2_mask.to(device)
        scores.data.masked_fill_(string2_mask.data.byte(), -float('inf'))

        alpha_flat = F.softmax(scores.view(-1, string2.size(1)), dim=1)        # [bs * seq_len1, seq_len2]
        alpha = alpha_flat.view(-1, string1.size(1), string2.size(1))   # [bs, seq_len1, seq_len2]

        #size_per_level = self.layer_conf.hidden_dim // 1
        #string1_atten_seq = alpha.bmm(string2.contiguous().view(-1, string2.size(1), 1, size_per_level).transpose(1, 2).contiguous().view(-1, string2.size(1), size_per_level))
        string1_atten_seq = alpha.bmm(string2)

        #return string1_atten_seq.view(-1, 1, string1.size(1), size_per_level).transpose(1, 2).contiguous().view(-1, string1.size(1), self.layer_conf.hidden_dim), string1_len
        return string1_atten_seq, string1_len
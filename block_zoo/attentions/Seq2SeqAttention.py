# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import copy
from block_zoo.BaseLayer import BaseLayer, BaseConf
from utils.DocInherit import DocInherit


class Seq2SeqAttentionConf(BaseConf):
    """Configuration for Seq2SeqAttention layer

    """
    def __int__(self, **kwargs):
        super(Seq2SeqAttentionConf, self).__init__(**kwargs)

    @DocInherit
    def default(self):
        #self.input_dim = 128
        self.attention_dropout = 0

    @DocInherit
    def declare(self):
        self.num_of_inputs = 2
        self.input_ranks = [3]

    @DocInherit
    def inference(self):
        self.output_dim = copy.deepcopy(self.input_dims[0])
        self.output_dim[-1] = 2 * self.input_dims[0][-1]    # all the inputs have the same input dim
        super(Seq2SeqAttentionConf, self).inference()  # PUT THIS LINE AT THE END OF inference()

    @DocInherit
    def verify(self):
        super(Seq2SeqAttentionConf, self).verify()

        necessary_attrs_for_user = ['attention_dropout']
        for attr in necessary_attrs_for_user:
            self.add_attr_exist_assertion_for_user(attr)


class Seq2SeqAttention(BaseLayer):
    """ Linear layer

    Args:
        layer_conf (LinearConf): configuration of a layer
    """
    def __init__(self, layer_conf):

        super(Seq2SeqAttention, self).__init__(layer_conf)
        self.layer_conf = layer_conf
        self.W = nn.Linear(layer_conf.input_dims[0][-1] * 3, 1)
        self.attention_dropout = nn.Dropout(layer_conf.attention_dropout)

    def forward(self, string, string_len, string2, string2_len=None):
        """ utilize both string2 and string itself to generate attention weights to represent string.
            There are two steps:
                1. get a string2 to string attention to represent string.
                2. get a string to string attention to represent string it self.
                3. merge the two representation above.

        Args:
            string (Variable): [batch_size, string_seq_len, dim].
            string_len (ndarray or None): [batch_size], default is None.
            string2 (Variable): [batch_size, string2_seq_len, dim].
            string2_len (ndarray or None): [batch_size], default is None.

        Returns:
            Variable: has the same shape as string.
        """
        assert string.size()[2] == string2.size()[2], 'The dimension of axis 2 of string and string2 must be consistent! But now, string.size() is %s and string2.size() is %s' % (string.size(), string2.size())

        batch_size = string.size()[0]
        string_seq_len = string.size()[1]
        string2_seq_len = string2.size()[1]
        feature_dim = string.size()[2]
        string2_aug = string2.unsqueeze(1).expand(batch_size, string_seq_len, string2_seq_len, feature_dim)  # [batch_size, string2_len, dim] -> [batch_size, string_len, string2_len, dim]
        string_aug = string.unsqueeze(1).expand(batch_size, string2_seq_len, string_seq_len, feature_dim)  # [batch_size, string_len, dim] -> [batch_size, string2_len, string_len, dim]

        string2_aug = string2_aug.contiguous().view(batch_size * string_seq_len * string2.size()[1], feature_dim)
        string_aug = string_aug.contiguous().view(batch_size * string2_seq_len * string_seq_len, feature_dim)

        # string2_string_comb = torch.cat((string2_aug, string_aug, string2_aug * string2_aug), 1)  # [batch_size * string2_len * string_len, 3 * dim]
        string2_string_comb = torch.cat((string2_aug, string_aug, string_aug * string2_aug), 1)  # [batch_size * string2_len * string_len, 3 * dim]

        attention = self.W(string2_string_comb)     # [batch_size * string2_len * string_len, 1]
        attention = self.attention_dropout(attention)
        attention = attention.view(batch_size, string_seq_len, string2_seq_len)  # [batch_size, string_len, string2_len]

        string_to_string_att_weight = torch.unsqueeze(nn.Softmax(dim=1)(torch.max(attention, 2)[0]), 2)  # [batch_size, string_len, 1]
        string_to_string_attention = string_to_string_att_weight * string  # [batch_size, string1_seq_len, feature_dim]

        string2_to_string_att_weight = nn.Softmax(dim=2)(attention)  # [batch_size, string1_seq_len, string2_seq_len]

        string2_to_string_attention = torch.sum(string2.unsqueeze(dim=1) * string2_to_string_att_weight.unsqueeze(dim=3), dim=2)  # [batch_size, string1_seq_len, feature_dim]

        string_out = torch.cat((string_to_string_attention, string2_to_string_attention), 2)  # [batch_size, string1_seq_len, 2 * feature_dim]

        return string_out, string_len

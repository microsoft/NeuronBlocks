# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F
from block_zoo.BaseLayer import BaseConf, BaseLayer
from utils.DocInherit import DocInherit
import copy


class BiAttFlowConf(BaseConf):
    """Configuration for AttentionFlow layer

    Args:
        attention_dropout(float): dropout rate of attention matrix dropout operation
    """
    def __init__(self, **kwargs):
        super(BiAttFlowConf, self).__init__(**kwargs)

    @DocInherit
    def default(self):
        self.attention_dropout = 0

    @DocInherit
    def declare(self):
        self.num_of_inputs = 2
        self.input_ranks = [3]

    @DocInherit
    def inference(self):
        self.output_dim = copy.deepcopy(self.input_dims[0])
        self.output_dim[-1] = 4 * self.input_dims[0][-1]
        super(BiAttFlowConf, self).inference()

    @DocInherit
    def verify(self):
        super(BiAttFlowConf, self).verify()
        necessary_attrs_for_user = ['attention_dropout']
        for attr in necessary_attrs_for_user:
            self.add_attr_exist_assertion_for_user(attr)


class BiAttFlow(BaseLayer):
    """
    implement AttentionFlow layer for BiDAF
    [paper]: https://arxiv.org/pdf/1611.01603.pdf

    Args:
        layer_conf(AttentionFlowConf): configuration of the AttentionFlowConf
    """
    def __init__(self, layer_conf):
        super(BiAttFlow, self).__init__(layer_conf)
        self.layer_conf = layer_conf
        # self.W = nn.Linear(layer_conf.input_dims[0][-1]*3, 1)
        self.attention_weight_content = nn.Linear(layer_conf.input_dims[0][-1], 1)
        self.attention_weight_query = nn.Linear(layer_conf.input_dims[0][-1], 1)
        self.attention_weight_cq = nn.Linear(layer_conf.input_dims[0][-1], 1)
        self.attention_dropout = nn.Dropout(layer_conf.attention_dropout)

    def forward(self, content, content_len, query, query_len=None):
        """
        implement the attention flow layer of BiDAF model

        :param content (Tensor): [batch_size, content_seq_len, dim]
        :param content_len: [batch_size]
        :param query (Tensor): [batch_size, query_seq_len, dim]
        :param query_len: [batch_size]
        :return: the tensor has same shape as content
        """
        assert content.size()[2] == query.size()[2], 'The dimension of axis 2 of content and query must be consistent! But now, content.size() is %s and query.size() is %s' % (content.size(), query.size())
        batch_size = content.size()[0]
        content_seq_len = content.size()[1]
        query_seq_len = query.size()[1]
        feature_dim = content.size()[2]

        # content_aug = content.unsqueeze(1).expand(batch_size, query_seq_len, content_seq_len, feature_dim)   #[batch_size, string2_seq_len, string_seq_len, feature_dim]
        content_aug = content.unsqueeze(2).expand(batch_size, content_seq_len, query_seq_len, feature_dim)  # [batch_size, string2_seq_len, string_seq_len, feature_dim]
        query_aug = query.unsqueeze(1).expand(batch_size, content_seq_len, query_seq_len, feature_dim) #[batch_size, string_seq_len, string2_seq_len, feature_dim]
        content_aug = content_aug.contiguous().view(batch_size*content_seq_len*query_seq_len, feature_dim)
        query_aug = query_aug.contiguous().view(batch_size*content_seq_len*query_seq_len, feature_dim)

        content_query_comb = torch.cat((query_aug, content_aug, content_aug * query_aug), 1)
        attention = self.W(content_query_comb)
        attention = self.attention_dropout(attention)
        # [batch_size, string_seq_len, string2_seq_len]
        attention = attention.view(batch_size, content_seq_len, query_seq_len)
        attention_logits = F.softmax(attention, dim=2)
        # [batch_size*content_seq_len*query_seq_len] * [batch_size*query_seq_len*feature_dim] --> [batch_size*content_seq_len*feature_dim]
        content2query_att = torch.bmm(attention_logits, query)
        # [batch_size, 1, content_seq_len]
        b = F.softmax(torch.max(attention, dim=2)[0], dim=1).unsqueeze(1)
        # [batch_size, 1, content_seq_len]*[batch_size, content_seq_len, feature_dim] = [batch_size, feature_dim]
        query2content_att = torch.bmm(b, content).squeeze(1)
        # [batch_size, content_seq_len, feature_dim]
        query2content_att = query2content_att.unsqueeze(1).expand(-1, content_seq_len, -1)

        result = torch.cat([content, content2query_att, content*content2query_att, content*query2content_att], dim=-1)

        return result, content_len




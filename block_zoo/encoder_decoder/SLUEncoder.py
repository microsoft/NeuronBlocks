# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from block_zoo.BaseLayer import BaseLayer, BaseConf
#from layers.EncoderDecoder import EncoderDecoderConf
from utils.DocInherit import DocInherit
from utils.corpus_utils import get_seq_mask
import copy

class SLUEncoderConf(BaseConf):
    """ Configuration of Spoken Language Understanding Encoder

    References:
        Liu, B., & Lane, I. (2016). Attention-based recurrent neural network models for joint intent detection and slot filling. Proceedings of the Annual Conference of the International Speech Communication Association, INTERSPEECH, (1), 685–689. https://doi.org/10.21437/Interspeech.2016-1352

    Args:
        hidden_dim (int): dimension of hidden state
        dropout (float): dropout rate
        num_layers (int): number of BiLSTM layers
    """
    def __init__(self, **kwargs):
        super(SLUEncoderConf, self).__init__(**kwargs)

    @DocInherit
    def default(self):
        self.hidden_dim = 128
        self.dropout = 0.0
        self.num_layers = 1

    @DocInherit
    def declare(self):
        self.num_of_inputs = 1
        self.input_ranks = [3]

    @DocInherit
    def inference(self):
        self.output_dim = copy.deepcopy(self.input_dims[0])
        self.output_dim[-1] = 2 * self.hidden_dim   #no matter what the num_layers is
        self.output_context_dim = copy.deepcopy(self.input_dims[0])
        self.output_context_dim[-1] = 2 * self.hidden_dim   #no matter what the num_layers is

        super(SLUEncoderConf, self).inference()  # PUT THIS LINE AT THE END OF inference()

    @DocInherit
    def verify_before_inference(self):
        super(SLUEncoderConf, self).verify_before_inference()
        necessary_attrs_for_user = ['hidden_dim']
        for attr in necessary_attrs_for_user:
            self.add_attr_exist_assertion_for_user(attr)

    @DocInherit
    def verify(self):
        super(SLUEncoderConf, self).verify()
        necessary_attrs_for_user = ['dropout', 'num_layers']
        for attr in necessary_attrs_for_user:
            self.add_attr_exist_assertion_for_user(attr)


class SLUEncoder(BaseLayer):
    """ Spoken Language Understanding Encoder

    References:
        Liu, B., & Lane, I. (2016). Attention-based recurrent neural network models for joint intent detection and slot filling. Proceedings of the Annual Conference of the International Speech Communication Association, INTERSPEECH, (1), 685–689. https://doi.org/10.21437/Interspeech.2016-1352

    Args:
        layer_conf (SLUEncoderConf): configuration of a layer
    """
    def __init__(self, layer_conf):
        super(SLUEncoder, self).__init__(layer_conf)
        self.layer_conf = layer_conf
        self.lstm = nn.LSTM(layer_conf.input_dims[0][-1], layer_conf.hidden_dim, layer_conf.num_layers, batch_first=True,
            bidirectional=True, dropout=layer_conf.dropout)

    def forward(self, string, string_len):
        """ process inputs

        Args:
            string (Variable): [batch_size, seq_len, dim]
            string_len (ndarray): [batch_size]

        Returns:
            Variable: output of bi-lstm with shape [batch_size, seq_len, 2 * hidden_dim]
            ndarray: string_len with shape [batch_size]
            Variable: context with shape [batch_size, 1, 2 * hidden_dim]

        """
        if torch.cuda.device_count() > 1:
            # otherwise, it will raise a Exception because the length inconsistence
            string_mask = torch.ByteTensor(1 - get_seq_mask(string_len, max_seq_len=string.shape[1]))  # [batch_size, max_seq_len]
        else:
            string_mask = torch.ByteTensor(1 - get_seq_mask(string_len))  # [batch_size, max_seq_len]

        hidden_init = torch.zeros(self.layer_conf.num_layers * 2, string.size(0), self.layer_conf.hidden_dim)
        context_init = torch.zeros(self.layer_conf.num_layers * 2, string.size(0), self.layer_conf.hidden_dim)
        if self.is_cuda():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            string_mask = string_mask.to(device)
            hidden_init = hidden_init.to(device)
            context_init = context_init.to(device)

        hidden = (hidden_init, context_init)
        self.lstm.flatten_parameters()
        output, hidden = self.lstm(string, hidden)

        assert output.shape[1] == string_mask.shape[1]

        real_context = []
        for i, o in enumerate(output):
            real_length = string_mask[i].data.tolist().count(0)
            real_context.append(o[real_length - 1])

        return output, torch.cat(real_context).view(string.size(0), -1).unsqueeze(1)
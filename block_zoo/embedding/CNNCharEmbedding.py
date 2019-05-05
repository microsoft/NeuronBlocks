# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from block_zoo.BaseLayer import BaseLayer, BaseConf
from utils.DocInherit import DocInherit


class CNNCharEmbeddingConf(BaseConf):
    """ Configuration of CNNCharEmbedding

    Args:
        dim (int, optional): the dimension of character embedding after convolution. Default: 30
        embedding_matrix_dim(int, optional): the dimension of character initialized embedding. Default: 30
        stride(int, optional): Stride of the convolution. Default: 1
        padding(int, optional): Zero-padding added to both sides of the input. Default: 0
        window_size(int, optional): width of convolution kernel. Default: 3
        activation(Str, optional): activation after convolution operation, can set null. Default: 'ReLU'
    """
    def __init__(self, **kwargs):
        super(CNNCharEmbeddingConf, self).__init__(**kwargs)

    @DocInherit
    def default(self):
        self.dim = 30       # cnn's output channel dim
        self.embedding_matrix_dim = 30      #
        self.stride = 1
        self.padding = 0
        self.window_size = 3
        self.activation = 'ReLU'

    @DocInherit
    def declare(self):
        self.input_channel_num = 1
        self.num_of_inputs = 1
        self.input_ranks = [3]

    @DocInherit
    def inference(self):
        self.output_channel_num = self.dim
        self.output_rank = 3

    @DocInherit
    def verify(self):
        # super(CNNCharEmbeddingConf, self).verify()

        necessary_attrs_for_user = ['dim', 'embedding_matrix_dim', 'stride', 'window_size', 'activation', 'vocab_size']
        for attr in necessary_attrs_for_user:
            self.add_attr_exist_assertion_for_user(attr)


class CNNCharEmbedding(BaseLayer):
    """
    This layer implements the character embedding use CNN
    Args:
        layer_conf (CNNCharEmbeddingConf): configuration of CNNCharEmbedding
    """
    def __init__(self, layer_conf):
        super(CNNCharEmbedding, self).__init__(layer_conf)
        self.layer_conf = layer_conf

        self.char_embeddings = nn.Embedding(layer_conf.vocab_size, layer_conf.embedding_matrix_dim, padding_idx=self.layer_conf.padding)
        nn.init.uniform_(self.char_embeddings.weight, -0.001, 0.001)

        self.filters = Variable(torch.randn(layer_conf.output_channel_num, layer_conf.input_channel_num,
                                            layer_conf.window_size, layer_conf.embedding_matrix_dim).float(),
                                requires_grad=True)
        if layer_conf.activation:
            self.activation = eval("nn." + self.layer_conf.activation)()
        else:
            self.activation = None
        if self.is_cuda():
            self.filters = self.filters.cuda()
            if self.activation:
                self.activation.weight = torch.nn.Parameter(self.activation.weight.cuda())

    def forward(self, string):
        """
        Step1: [batch_size, seq_len, char num in words] -> [batch_size, seq_len * char num in words]
        Step2: lookup embedding matrix -> [batch_size, seq_len * char num in words, embedding_dim]
        reshape -> [batch_size * seq_len, char num in words, embedding_dim]
        Step3: after convolution operation, got [batch_size * seq_len, char num related, output_channel_num]
        Step4: max pooling on axis 1 and -reshape-> [batch_size * seq_len, output_channel_dim]
        Step5: reshape -> [batch_size, seq_len, output_channel_dim]

        Args:
            string (Variable): [[char ids of word1], [char ids of word2], [...], ...], shape: [batch_size, seq_len, char num in words]

        Returns:
            Variable: [batch_size, seq_len, output_dim]

        """
        string_reshaped = string.view(string.size()[0], -1)     #[batch_size, seq_len * char num in words]
        char_embs_lookup = self.char_embeddings(string_reshaped).float()    # [batch_size, seq_len * char num in words, embedding_dim]
        if self.is_cuda():
            if self.filters.device == torch.device('cpu'):
                self.filters = self.filters.cuda()
            char_embs_lookup = char_embs_lookup.cuda(device=self.filters.device)
        char_embs_lookup = char_embs_lookup.view(-1, string.size()[2], self.layer_conf.embedding_matrix_dim)    #[batch_size * seq_len, char num in words, embedding_dim]

        string_input = torch.unsqueeze(char_embs_lookup, 1)   # [batch_size * seq_len, input_channel_num=1, char num in words, embedding_dim]

        string_conv = F.conv2d(string_input, self.filters, stride=self.layer_conf.stride, padding=self.layer_conf.padding)    # [batch_size * seq_len, output_channel_num, char num in word related, 1]
        string_conv = torch.squeeze(string_conv, 3).permute(0, 2, 1)      # [batch_size * seq_len, char num in word related, output_channel_num]
        if self.activation:
            string_conv = self.activation(string_conv)

        string_maxpooling = torch.max(string_conv, 1)[0]
        string_out = string_maxpooling.view(string.size()[0], string.size()[1], -1)

        return string_out.cpu()


if __name__ == '__main__':
    conf = {
        'dim': 30,
        'output_channel_num': 30,
        'input_channel_num': 1,
        'window_size': 3,
        'activation': 'PReLU',

        # should be infered from the corpus
        'vocab_size': 10,
        'input_dims': [5],
        'input_ranks': [3],
        'use_gpu': True
    }
    layer_conf = CNNCharEmbeddingConf(**conf)

    # make a fake input: [bs, seq_len, char num in words]
    # assume in this batch, the padded sentence length is 3 and the each word has 5 chars, including padding 0.
    input_chars = np.array([
        [[3, 1, 2, 5, 4], [1, 2, 3, 4, 0], [0, 0, 0, 0, 0]],
        [[1, 1, 0, 0, 0], [2, 3, 1, 0, 0], [1, 2, 3, 4, 5]]
    ])

    char_emb_layer = CNNCharEmbedding(layer_conf)

    input_chars = torch.LongTensor(input_chars)
    output = char_emb_layer(input_chars)

    print(output)




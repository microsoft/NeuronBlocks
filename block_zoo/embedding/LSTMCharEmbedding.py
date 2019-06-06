# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from block_zoo.BaseLayer import BaseLayer, BaseConf
from utils.DocInherit import DocInherit


class LSTMCharEmbeddingConf(BaseConf):
    """ Configuration of LSTMCharEmbedding

    Args:
        dim (int, optional): the dimension of character embedding after lstm. Default: 50
        embedding_matrix_dim(int, optional): the dimension of character initialized embedding. Default: 30
        padding(int, optional): Zero-padding added to both sides of the input. Default: 0
        dropout(float, optional): dropout rate. Default: 0.2
        bidirect_flag(Bool, optional): Using BiLSTM or not. Default: True
    """
    def __init__(self, **kwargs):
        super(LSTMCharEmbeddingConf, self).__init__(**kwargs)

    @DocInherit
    def default(self):

        self.dim = 50  # lstm's output channel dim
        self.embedding_matrix_dim = 30
        self.padding = 0
        self.dropout = 0.2
        self.bidirect_flag = True

    @DocInherit
    def declare(self):
        #self.input_channel_num = 1
        self.num_of_inputs = 1
        self.input_ranks = [3]

    @DocInherit
    def inference(self):
        #self.output_channel_num = self.hidden_dim
        self.output_rank = 3

    @DocInherit
    def verify(self):
        # super(LSTMCharEmbeddingConf, self).verify()

        necessary_attrs_for_user = ['embedding_matrix_dim', 'dim', 'dropout', 'bidirect_flag', 'vocab_size']
        for attr in necessary_attrs_for_user:
            self.add_attr_exist_assertion_for_user(attr)


class LSTMCharEmbedding(BaseLayer):
    """
    This layer implements the character embedding use LSTM
    Args:
        layer_conf (LSTMCharEmbeddingConf): configuration of LSTMCharEmbedding
    """
    def __init__(self, layer_conf):
        super(LSTMCharEmbedding, self).__init__(layer_conf)
        self.layer_conf = layer_conf

        self.char_embeddings = nn.Embedding(layer_conf.vocab_size, layer_conf.embedding_matrix_dim, padding_idx=self.layer_conf.padding)
        nn.init.uniform_(self.char_embeddings.weight, -0.001, 0.001)

        if layer_conf.bidirect_flag:
            self.dim = layer_conf.dim // 2
        self.dropout = nn.Dropout(layer_conf.dropout)
        self.char_lstm = nn.LSTM(layer_conf.embedding_matrix_dim, self.dim, num_layers=1, batch_first=True, bidirectional=layer_conf.bidirect_flag)

        if self.is_cuda():
            self.char_embeddings = self.char_embeddings.cuda()
            self.dropout = self.dropout.cuda()
            self.char_lstm = self.char_lstm.cuda()

    def forward(self, string):
        """
        Step1: [batch_size, seq_len, char num in words] -> [batch_size*seq_len, char num in words]
        Step2: lookup embedding matrix -> [batch_size*seq_len, char num in words, embedding_dim]
        Step3: after lstm operation, got [num_layer* num_directions, batch_size * seq_len, dim]
        Step5: reshape -> [batch_size, seq_len, dim]

        Args:
            string (Variable): [[char ids of word1], [char ids of word2], [...], ...], shape: [batch_size, seq_len, char num in words]

        Returns:
            Variable: [batch_size, seq_len, output_dim]

        """
        #print ('string shape: ', string.size())
        string_reshaped = string.view(string.size()[0]*string.size()[1],  -1)     #[batch_size, seq_len * char num in words]

        char_embs_lookup = self.char_embeddings(string_reshaped).float()    # [batch_size, seq_len * char num in words, embedding_dim]
        char_embs_drop = self.dropout(char_embs_lookup)
        char_hidden = None
        char_rnn_out, char_hidden = self.char_lstm(char_embs_drop, char_hidden)
        #print('char_hidden shape: ', char_hidden[0].size())
        string_out = char_hidden[0].transpose(1,0).contiguous().view(string.size()[0], string.size()[1], -1)
        #print('string_out shape: ', string_out.size())
        return string_out


if __name__ == '__main__':
    conf = {
        'embedding_matrix_dim': 30,
        'dim': 30,  # lstm's output channel dim
        'padding': 0,
        'dropout': 0.2,
        'bidirect_flag': True,

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

    char_emb_layer = LSTMCharEmbedding(layer_conf)

    input_chars = torch.LongTensor(input_chars)
    output = char_emb_layer(input_chars)

    print(output)




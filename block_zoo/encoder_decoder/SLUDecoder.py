# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import copy
import numpy as np
from block_zoo.BaseLayer import BaseLayer, BaseConf
#from layers.EncoderDecoder import EncoderDecoderConf
from utils.DocInherit import DocInherit
from utils.corpus_utils import get_seq_mask

class SLUDecoderConf(BaseConf):
    """ Configuration of Spoken Language Understanding Encoder

    References:
        Liu, B., & Lane, I. (2016). Attention-based recurrent neural network models for joint intent detection and slot filling. Proceedings of the Annual Conference of the International Speech Communication Association, INTERSPEECH, (1), 685–689. https://doi.org/10.21437/Interspeech.2016-1352

    Args:
        hidden_dim (int): dimension of hidden state
        dropout (float): dropout rate
        num_layers (int): number of BiLSTM layers
        num_decoder_output (int):
    """
    def __init__(self, **kwargs):
        super(SLUDecoderConf, self).__init__(**kwargs)

    @DocInherit
    def default(self):
        self.hidden_dim = 128
        self.dropout = 0.0
        self.num_layers = 1
        self.decoder_emb_dim = 100

        # number of decoder's outputs. E.g., for slot tagging, num_decoder_output means the number of tags;
        # for machine translation, num_decoder_output means the number of words in the target language;
        self.decoder_vocab_size = 10000

        #input_dim and input_context_dim should be inferenced from encoder

    @DocInherit
    def declare(self):
        self.num_of_inputs = 1
        self.input_ranks = [3]

    @DocInherit
    def inference(self):
        self.output_dim = copy.deepcopy(self.input_dims[0])
        self.output_dim[-1] = self.decoder_vocab_size
        super(SLUDecoderConf, self).inference()  # PUT THIS LINE AT THE END OF inference()

    @DocInherit
    def verify(self):
        super(SLUDecoderConf, self).verify()

        necessary_attrs_for_user = ['hidden_dim', 'dropout', 'num_layers', 'decoder_emb_dim', 'decoder_vocab_size']
        for attr in necessary_attrs_for_user:
            self.add_attr_exist_assertion_for_user(attr)

        necessary_attrs_for_dev = ['input_dims', 'input_context_dims']
        for attr in necessary_attrs_for_dev:
            self.add_attr_exist_assertion_for_dev(attr)


class SLUDecoder(BaseLayer):
    """ Spoken Language Understanding Encoder

    References:
        Liu, B., & Lane, I. (2016). Attention-based recurrent neural network models for joint intent detection and slot filling. Proceedings of the Annual Conference of the International Speech Communication Association, INTERSPEECH, (1), 685–689. https://doi.org/10.21437/Interspeech.2016-1352

    Args:
        layer_conf (SLUEncoderConf): configuration of a layer
    """
    def __init__(self, layer_conf):
        super(SLUDecoder, self).__init__(layer_conf)
        self.layer_conf = layer_conf

        self.embedding = nn.Embedding(layer_conf.decoder_vocab_size, layer_conf.decoder_emb_dim)
        self.embedding.weight.data.uniform_(-0.1, 0.1)  # init
        #nn.init.uniform(self.embedding.weight, -0.1, 0.1)

        #self.dropout = nn.Dropout(self.dropout_p)
        #self.lstm = nn.LSTM(layer_conf.decoder_emb_dim + layer_conf.hidden_dim * 2, layer_conf.hidden_dim, layer_conf.num_layers, batch_first=True)
        self.lstm = nn.LSTM(layer_conf.decoder_emb_dim + layer_conf.input_dims[0][-1] + layer_conf.input_context_dims[0][-1],
            layer_conf.hidden_dim, layer_conf.num_layers, batch_first=True)     # CAUTION: single direction
        self.attn = nn.Linear(layer_conf.input_context_dims[0][-1], layer_conf.hidden_dim *layer_conf.num_layers) # Attention
        self.slot_out = nn.Linear(layer_conf.input_context_dims[0][-1] + layer_conf.hidden_dim * 1 *layer_conf.num_layers, layer_conf.decoder_vocab_size)

    def Attention(self, hidden, encoder_outputs, encoder_maskings):
        """

        Args:
            hidden : 1,B,D
            encoder_outputs : B,T,D
            encoder_maskings : B,T # ByteTensor
        """

        hidden = hidden.view(hidden.size()[1], -1).unsqueeze(2)

        batch_size = encoder_outputs.size(0)  # B
        max_len = encoder_outputs.size(1)  # T
        energies = self.attn(encoder_outputs.contiguous().view(batch_size * max_len, -1))  # B*T,D -> B*T,D
        energies = energies.view(batch_size, max_len, -1)  # B,T,D
        attn_energies = energies.bmm(hidden).transpose(1, 2)  # B,T,D * B,D,1 --> B,1,T
        attn_energies = attn_energies.squeeze(1).masked_fill(encoder_maskings, -1e12)  # PAD masking

        alpha = F.softmax(attn_energies)  # B,T
        alpha = alpha.unsqueeze(1)  # B,1,T
        context = alpha.bmm(encoder_outputs)  # B,1,T * B,T,D => B,1,D

        return context  # B,1,D

    def forward(self, string, string_len, context, encoder_outputs):
        """ process inputs

        Args:
            string (Variable): word ids, [batch_size, seq_len]
            string_len (ndarray): [batch_size]
            context (Variable): [batch_size, 1, input_dim]
            encoder_outputs (Variable): [batch_size, max_seq_len, input_dim]

        Returns:
            Variable : decode scores with shape [batch_size, seq_len, decoder_vocab_size]

        """
        batch_size = string.size(0)
        if torch.cuda.device_count() > 1:
            # otherwise, it will raise a Exception because the length inconsistence
            string_mask = torch.ByteTensor(1 - get_seq_mask(string_len, max_seq_len=string.shape[1]))  # [batch_size, max_seq_len]
        else:
            string_mask = torch.ByteTensor(1 - get_seq_mask(string_len))  # [batch_size, max_seq_len]

        decoded = torch.LongTensor([[1] * batch_size])
        hidden_init = torch.zeros(self.layer_conf.num_layers * 1, batch_size, self.layer_conf.hidden_dim)
        context_init = torch.zeros(self.layer_conf.num_layers*1, batch_size, self.layer_conf.hidden_dim)
        if self.is_cuda():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            string_mask = string_mask.to(device)
            # Note id of "<start>" is 1!   decoded  is a batch of '<start>' at first
            decoded = decoded.to(device)
            hidden_init = hidden_init.to(device)
            context_init = context_init.to(device)

        decoded = decoded.transpose(1, 0)      # [batch_size, 1]

        embedded = self.embedding(decoded)
        hidden = (hidden_init, context_init)

        decode = []
        aligns = encoder_outputs.transpose(0, 1)    #[seq_len, bs, input_dim]
        length = encoder_outputs.size(1)
        for i in range(length):
            aligned = aligns[i].unsqueeze(1)  # [bs, 1, input_dim]
            self.lstm.flatten_parameters()
            _, hidden = self.lstm(torch.cat((embedded, context, aligned), 2), hidden)

            concated = torch.cat((hidden[0].view(1, batch_size, -1), context.transpose(0, 1)), 2)
            score = self.slot_out(concated.squeeze(0))
            softmaxed = F.log_softmax(score)        # decoder_vocab_dim
            decode.append(softmaxed)
            _, decoded = torch.max(softmaxed, 1)
            embedded = self.embedding(decoded.unsqueeze(1))
            context = self.Attention(hidden[0], encoder_outputs, string_mask)
        slot_scores = torch.cat(decode, 1)
        return slot_scores.view(batch_size, length, -1)





# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from block_zoo.BaseLayer import BaseLayer, BaseConf
import numpy as np
from utils.DocInherit import DocInherit
from block_zoo.embedding import *
import copy
import logging

class EmbeddingConf(BaseConf):
    """ Configuration for Embedding

    Args:
        conf:a dictionary. The key is embedding type, such as word embedding, char embedding, Part-of-Speech embedding and so on.

    Example::

        "conf": {
          "word": {
            "cols": ["question_text", "answer_text"],
            "dim": 300,
            "fix_weight": true
          },
          "postag": {
            "cols": ["question_postag","answer_postag"],
            "dim": 20
          },
          "char": {
            "cols": ["question_char", "answer_char"],
            "type": "CNNCharEmbedding",
            "dropout": 0.2,
            "dim": 30,
            "embedding_matrix_dim": 8,
            "stride":1,
            "window_size": 5,
            "activation": null
          }
        }
    """
    def __init__(self, **kwargs):
        super(EmbeddingConf, self).__init__(**kwargs)

    @DocInherit
    def default(self):
        self.conf = {
            'word': {
                'vocab_size': 1000,
                'dim': 300,
                'init_weights': np.random.randn(1000, 300)      # you can give a initial weight here like this or assign it to None
            }
        }

    @DocInherit
    def declare(self):
        self.num_of_inputs = 1
        self.input_ranks = [2]         #[batch size, sequence length]

    @DocInherit
    def inference(self):
        self.output_dim = [-1, -1, 0]
        for emb_type in self.conf:
            if emb_type == 'position':
                continue
            self.output_dim[2] += self.conf[emb_type]['dim']

        super(EmbeddingConf, self).inference()

    @DocInherit
    def verify_before_inference(self):
        necessary_attrs_for_user = ['conf']
        for attr in necessary_attrs_for_user:
            self.add_attr_exist_assertion_for_user(attr)

        necessary_attrs_for_dev = ['num_of_inputs', 'input_ranks']
        for attr in necessary_attrs_for_dev:
            self.add_attr_exist_assertion_for_dev(attr)

        type_checks = [('conf', dict),
                       ('num_of_inputs', int),
                       ('input_ranks', list)]
        for attr, attr_type in type_checks:
            self.add_attr_type_assertion(attr, attr_type)


    @DocInherit
    def verify(self):
        #super(EmbeddingConf, self).verify()

        necessary_attrs_for_dev = ['output_dim', 'output_rank']
        for attr in necessary_attrs_for_dev:
            self.add_attr_exist_assertion_for_dev(attr)

        type_checks = [('output_dim', list),
                       ('output_rank', int)]
        for attr, attr_type in type_checks:
            self.add_attr_type_assertion(attr, attr_type)


class Embedding(BaseLayer):
    """ Embedding layer

    Args:
        layer_conf (EmbeddingConf): configuration of a layer
    """
    def __init__(self, layer_conf):

        super(Embedding, self).__init__(layer_conf)
        self.layer_conf = layer_conf

        # self.embeddings = dict()
        self.embeddings = nn.ModuleDict()
        for input_cluster in layer_conf.conf:
            if 'type' in layer_conf.conf[input_cluster]:
                # char embedding
                char_emb_conf_dict = copy.deepcopy(layer_conf.conf[input_cluster])
                # del char_emb_conf_dict['cols'], char_emb_conf_dict['type']
                char_emb_conf_dict['use_gpu'] = layer_conf.use_gpu
                char_emb_conf = eval(layer_conf.conf[input_cluster]['type'] + "Conf")(** char_emb_conf_dict)
                char_emb_conf.inference()
                char_emb_conf.verify()
                self.embeddings[input_cluster] = eval(layer_conf.conf[input_cluster]['type'])(char_emb_conf)
            else:
                # word embedding, postag embedding, and so on
                self.embeddings[input_cluster] = nn.Embedding(layer_conf.conf[input_cluster]['vocab_size'], layer_conf.conf[input_cluster]['dim'], padding_idx=0)
                if 'init_weights' in layer_conf.conf[input_cluster] and layer_conf.conf[input_cluster]['init_weights'] is not None:
                    self.embeddings[input_cluster].weight = nn.Parameter(torch.from_numpy(layer_conf.conf[input_cluster]['init_weights']))

                # judge if fix the embedding weight
                if layer_conf.conf[input_cluster]['fix_weight']:
                    self.embeddings[input_cluster].weight.requires_grad = False
                    logging.info("The Embedding[%s][fix_weight] is true, fix the embeddings[%s]'s weight" % (input_cluster, input_cluster))


    def forward(self, inputs, use_gpu=False):
        """ process inputs

        Args:
            inputs (dict): a dictionary to describe each transformer_model inputs. e.g.:\n
                        char_emb': [[char ids of word1], [char ids of word2], [...], ...], shape: [batch_size, seq_len, word character num]\n
                        'word': word ids (Variable), shape:[batch_size, seq_len],\n
                        'postag': postag ids (Variable), shape: [batch_size, seq_len],\n
                        ...
            use_gpu (bool): put embedding matrix on GPU (True) or not (False)

        Returns:
            Variable: the embedding representation with shape [batch_size, seq_len, emb_dim]

        """
        features = []

        for input_cluster in inputs:
            if 'extra' in input_cluster:
                continue
            input = inputs[input_cluster]
            # if 'type' in self.layer_conf.conf[input_cluster]:
            #     emb = self.embeddings[input_cluster](input, lengths[input]).float()
            # else:
            #     emb = self.embeddings[input_cluster](input).float()
            # emb = self.embeddings[input_cluster](input.cpu()).float()
            emb = self.embeddings[input_cluster](input).to(torch.float32)
            if use_gpu is True:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                emb = emb.to(device)
            features.append(emb)

        if len(features) > 1:
            return torch.cat(features, 2)
        else:
            return features[0]





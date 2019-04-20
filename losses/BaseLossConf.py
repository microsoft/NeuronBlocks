# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
from torch.nn import CrossEntropyLoss
import copy
import logging

class BaseLossConf(object):
    @staticmethod
    def get_conf(**kwargs):
        # judge loss num & setting
        kwargs['multiLoss'] = True if len(kwargs['losses']) > 1 else False
        # loss = copy.deepcopy(kwargs['losses'])
        if kwargs['multiLoss']:
            if kwargs.get('multi_loss_op', '') is None:
                kwargs['multi_loss_op'] = 'weighted_sum'
                logging.info('model has multi-loss but no multi_loss_op, we set default option {0}.'.format('weighted_sum'))
            if kwargs.get('weights', None) is None:
                kwargs['weights'] = [1] * len(kwargs['losses'])
                logging.warning("MultiLoss have no weights, set the weights to 1.")
            assert len(kwargs['weights']) == len(kwargs['losses']), "The number of loss is inconsistent with loss weights!"


        # IF NEEDED, TRANSFORM SOME INT OR FLOAT, OR NUMPY ARRAY TO TENSORS.
        for single_loss in kwargs['losses']:
            if 'inputs' not in single_loss:
                raise Exception("Each loss must have inputs")
            if not isinstance(single_loss['inputs'], list):
                raise Exception('The inputs of loss must be list')
            if len(single_loss['inputs']) != 2:
                raise Exception('The length of loss inputs must be 2')
            if 'weight' in single_loss['conf']:
                single_loss['conf']['weight'] = torch.FloatTensor(single_loss['conf']['weight'])

        return kwargs



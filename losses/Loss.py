# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import sys
import logging
sys.path.append('../')
from settings import LossOperationType
from torch.nn import CrossEntropyLoss, L1Loss, MSELoss, NLLLoss, PoissonNLLLoss, NLLLoss2d, KLDivLoss, BCELoss, BCEWithLogitsLoss, MarginRankingLoss, HingeEmbeddingLoss, MultiLabelMarginLoss, SmoothL1Loss, SoftMarginLoss, MultiLabelSoftMarginLoss, CosineEmbeddingLoss, MultiMarginLoss, TripletMarginLoss


class Loss(nn.Module):
    '''
    For support multi_task or multi_output, the loss type changes to list.
    Using class Loss for parsing and constructing the loss list.
    Args:
        loss_conf: the loss for multi_task or multi_output.
                   multi_loss_op: the operation for multi_loss
                   losses: list type. Each element is single loss.
                   eg: "loss": {
                            "multi_loss_op": "weighted_sum",
                            "losses": [
                              {
                                "type": "CrossEntropyLoss",
                                "conf": {
                                    "gamma": 0,
                                    "alpha": 0.5,
                                    "size_average": true
                                },
                                "inputs": ["start_output", "start_label"]
                              },
                              {
                                "type": "CrossEntropyLoss",
                                "conf": {
                                    "gamma": 0,
                                    "alpha": 0.5,
                                    "size_average": true
                                },
                                "inputs": ["end_output", "end_label"]
                              }
                            ],
                            "weights": [0.5, 0.5]
                        }
    '''
    def __init__(self, **kwargs):
        super(Loss, self).__init__()

        self.loss_fn = nn.ModuleList()
        self.loss_input = []
        self.weights = kwargs['weights'] if 'weights' in kwargs else None
        support_loss_op = set(LossOperationType.__members__.keys())
        if kwargs['multiLoss']:
            # check multi_loss_op
            if not kwargs['multi_loss_op'].lower() in support_loss_op:
                raise Exception("The multi_loss_op %s is not supported. Supported multi_loss_op are: %s"
                                % (kwargs['multi_loss_op'], support_loss_op))
            self.multi_loss_op = kwargs['multi_loss_op']
        # check single loss inputs
        for single_loss in kwargs['losses']:
            if (not single_loss['inputs'][0] in kwargs['output_layer_id']) or (not single_loss['inputs'][1] in kwargs['answer_column_name']):
                raise Exception("The loss inputs are excepted to be part of output_layer_id and targets!")
            self.loss_fn.append(eval(single_loss['type'])(**single_loss['conf']))
            self.loss_input.append(single_loss['inputs'])

    def forward(self, model_outputs, targets):
        '''
        compute multi_loss according to multi_loss_op
        :param model_outputs: the representation of model output layer
                              :type: dict {output_layer_id: output layer data}
        :param targets: the label of raw data
                        :type: dict {target: data}
        :return:
        '''
        all_losses = []
        result_loss = 0.0
        for index, single_loss_fn in enumerate(self.loss_fn):
            all_losses.append(single_loss_fn(model_outputs[self.loss_input[index][0]], targets[self.loss_input[index][1]]))
        if hasattr(self, 'multi_loss_op'):
            if LossOperationType[self.multi_loss_op.lower()] == LossOperationType.weighted_sum:
                for index, single_loss in enumerate(all_losses):
                    result_loss += (self.weights[index]*single_loss)
        else:
            result_loss = all_losses[0]

        return result_loss


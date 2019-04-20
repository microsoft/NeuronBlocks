# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """ Focal loss
        reference: Lin T Y, Goyal P, Girshick R, et al. Focal loss for dense object detection[J]. arXiv preprint arXiv:1708.02002, 2017.
    Args:
        gamma (float): gamma >= 0.
        alpha (float): 0 <= alpha <= 1
        size_average (bool, optional): By default, the losses are averaged over observations for each minibatch. However, if the field size_average is set to False, the losses are instead summed for each minibatch. Default is True

    """
    def __init__(self, **kwargs):
        super(FocalLoss, self).__init__()

        # default parameters
        self.gamma = 0
        self.alpha = 0.5
        self.size_average = True

        for key in kwargs:
            setattr(self, key, kwargs[key])

        # varification
        assert self.alpha <= 1 and self.alpha >= 0, "The parameter alpha in Focal Loss must be in range [0, 1]."
        if self.alpha is not None:
            self.alpha = torch.Tensor([self.alpha, 1 - self.alpha])

    def forward(self, input, target):
        """ Get focal loss

        Args:
            input (Variable):  the prediction with shape [batch_size, number of classes]
            target (Variable): the answer with shape [batch_size, number of classes]

        Returns:
            Variable (float): loss
        """
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1-pt)**self.gamma * logpt

        if self.size_average: 
            return loss.mean()
        else:
            return loss.sum()

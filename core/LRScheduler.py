# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import numpy as np

class LRScheduler():
    def __init__(self, optimizer, decay_rate=1, minimum_lr=0, epoch_start_decay=1):
        """

        Args:
            optimizer:
            decay_rate:
            minimum_lr: if lr < minimum_lr, stop lr decay
        """
        self.optimizer = optimizer
        self.decay_rate = decay_rate
        self.minimum_lr = minimum_lr
        self.epoch_cnt = 0
        self.epoch_start_decay = epoch_start_decay

    def step(self):
        """ adjust learning rate

        Args:
            optimizer:
            decay_rate:
            minimum_lr:

        Returns:
            None

        """
        self.epoch_cnt += 1

        if self.epoch_cnt >= self.epoch_start_decay:
            for param_group in self.optimizer.param_groups:
                if param_group['lr'] * self.decay_rate >= self.minimum_lr:
                    param_group['lr'] = param_group['lr'] * self.decay_rate
                else:
                    param_group['lr'] = self.minimum_lr


    def get_lr(self):
        """ get average learning rate of optimizer.param_groups

        Args:
            optimizer:

        Returns:

        """
        lr_total = []
        for param_group in self.optimizer.param_groups:
            lr_total.append(param_group['lr'])
        return np.mean(lr_total)

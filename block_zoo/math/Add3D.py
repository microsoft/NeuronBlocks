# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import logging

from ..BaseLayer import BaseConf
from utils.DocInherit import DocInherit
from utils.exceptions import ConfigurationError
import copy

class Add3DConf(BaseConf):
    """ Configuration of Add3D layer

    """

    #init the args
    def __init__(self, **kwargs):
        super(Add3DConf, self).__init__(**kwargs)

    #set default params
    #@DocInherit
    #def default(self):

    @DocInherit
    def declare(self):
        self.num_of_inputs = 2
        self.input_ranks = [3,3]

    @DocInherit
    def inference(self):
        self.output_dim = copy.deepcopy(self.input_dims[0])
        if self.input_dims[0][-1] != 1:
            self.output_dim[-1] = self.input_dims[0][-1]
        else:
            self.output_dim[-1] = self.input_dims[1][-1]
            
        super(Add3DConf, self).inference()

    @DocInherit
    def verify(self):
        super(Add3DConf, self).verify()

        # # to check if the ranks of all the inputs are equal
        # rank_equal_flag = True
        # for i in range(len(self.input_ranks)):
        #     if self.input_ranks[i] != self.input_ranks[0]:
        #         rank_equal_flag = False
        #         break
        # if rank_equal_flag == False:
        #     raise ConfigurationError("For layer Add3D, the ranks of each inputs should be equal!")

class Add3D(nn.Module):
    """ Add3D layer to get sum of two sequences(3D representation)

    Args:
        layer_conf (Add3DConf): configuration of a layer

    """
    def __init__(self, layer_conf):
        super(Add3D, self).__init__()
        self.layer_conf = layer_conf

        logging.warning("The length Add3D layer returns is the length of first input")

    def forward(self, *args):
        """ process input

        Args:
            *args: (Tensor): string, string_len, string2, string2_len
                e.g. string (Tensor): [batch_size, seq_len, dim], string_len (Tensor): [batch_size]

        Returns:
            Tensor: [batch_size, seq_len, output_dim], [batch_size]
        """
        dim_flag = True
        input_dims = list(self.layer_conf.input_dims)
        if (args[0].shape[1] * args[0].shape[2]) != (args[2].shape[1] * args[2].shape[2]):
            if args[0].shape[1] == args[2].shape[1] and (input_dims[1][-1] == 1 or input_dims[0][-1] == 1):
                dim_flag = True
            else:
                dim_flag = False

        if dim_flag == False:
            raise ConfigurationError("For layer Add3D, the dimensions of each inputs should be equal or 1 ,or the elements number of two inputs (expect for the first dimension) should be equal")


        return torch.add(args[0],args[2]),args[1]
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from block_zoo.BaseLayer import BaseLayer, BaseConf
from utils.DocInherit import DocInherit


class Conv2DConf(BaseConf):
    """ Configuration of Conv

    Args:
        stride (int): the stride of the convolving kernel. Can be a single number or a tuple (sH, sW). Default: 1
        padding (int): implicit zero paddings on both sides of the input. Can be a single number or a tuple (padH, padW). Default: 0
        window_size (int): actually, the window size is (window_sizeH, window_sizeW), because for NLP tasks, 1d convolution is more commonly used.
        output_channel_num (int): number of feature maps
        batch_norm (bool): If True, apply batch normalization before activation
        activation (string): activation functions, e.g. ReLU

    """
    def __int__(self, **kwargs):
        super(Conv2DConf, self).__init__(**kwargs)

    @DocInherit
    def default(self):
        self.stride = 1
        self.padding = 0
        self.window_size = 3
        # self.input_channel_num = 1      # for NLP tasks, input_channel_num would always be 1
        self.output_channel_num = 16
        self.batch_norm = True
        self.activation = 'ReLU'

    @DocInherit
    def declare(self):
        self.num_of_inputs = 1
        self.input_ranks = [4]
    
    def check_size(self, value, attr):
        res = value
        if isinstance(value,int):
            res = [value, value]
        elif (isinstance(self.window_size, tuple) or isinstance(self.window_size, list)) and len(value)==2:
            res = list(value)
        else:
            raise AttributeError('The Atrribute %s should be given an integer or a list/tuple with length of 2, instead of %s.' %(attr,str(attr)))
        return res

    @DocInherit
    def inference(self):
        self.window_size = self.check_size(self.window_size, "window_size")
        self.stride = self.check_size(self.stride, "stride")
        self.padding = self.check_size(self.padding, "padding")
        
        self.input_channel_num = self.input_dims[0][-1]

        self.output_dim = [self.input_dims[0][0]]
        if self.input_dims[0][1] != -1:
            self.output_dim.append((self.input_dims[0][1] + 2 * self.padding[0] - self.window_size[0]) // self.stride[0] + 1)
        else:
            self.output_dim.append(-1)
        if self.input_dims[0][2] != -1:
            self.output_dim.append((self.input_dims[0][2] + 2 * self.padding[1] - self.window_size[1]) // self.stride[1] + 1)
        else:
            self.output_dim.append(-1)
        self.output_dim.append(self.output_channel_num)

        super(Conv2DConf, self).inference()  # PUT THIS LINE AT THE END OF inference()


    @DocInherit
    def verify_before_inference(self):
        super(Conv2DConf, self).verify_before_inference()
        necessary_attrs_for_user = ['output_channel_num']
        for attr in necessary_attrs_for_user:
            self.add_attr_exist_assertion_for_user(attr)

    @DocInherit
    def verify(self):
        super(Conv2DConf, self).verify()

        necessary_attrs_for_user = ['stride', 'padding', 'window_size', 'input_channel_num', 'output_channel_num', 'activation']
        for attr in necessary_attrs_for_user:
            self.add_attr_exist_assertion_for_user(attr)


class Conv2D(BaseLayer):
    """ Convolution along just 1 direction

    Args:
        layer_conf (ConvConf): configuration of a layer
    """
    def __init__(self, layer_conf):
        super(Conv2D, self).__init__(layer_conf)
        self.layer_conf = layer_conf
        if layer_conf.activation:
            self.activation = eval("nn." + self.layer_conf.activation)()
        else:
            self.activation = None
        
        self.cnn = nn.Conv2d(in_channels=layer_conf.input_channel_num, out_channels=layer_conf.output_channel_num,kernel_size=layer_conf.window_size,stride=layer_conf.stride,padding=layer_conf.padding)

        if layer_conf.batch_norm:
            self.batch_norm = nn.BatchNorm2d(layer_conf.output_channel_num)    # the output_chanel of Conv is the input_channel of BN
        else:
            self.batch_norm = None

    def forward(self, string, string_len=None):
        """ process inputs

        Args:
            string (Tensor): tensor with shape: [batch_size, length, width, feature_dim]
            string_len (Tensor):  [batch_size]

        Returns:
            Tensor: shape: [batch_size, (length - conv_window_size) // stride + 1, (width - conv_window_size) // stride + 1, output_channel_num]

        """

        string = string.permute([0,3,1,2]).contiguous()
        string_out = self.cnn(string)
        if hasattr(self, 'batch_norms') and self.batch_norm:
            string_out = self.batch_norm(string_out)

        string_out = string_out.permute([0,2,3,1]).contiguous()

        if self.activation:
            string_out = self.activation(string_out)
        if string_len is not None:
            string_len_out = (string_len - self.layer_conf.window_size[0]) // self.layer_conf.stride[0] + 1
        else:
            string_len_out = None
        return string_out, string_len_out

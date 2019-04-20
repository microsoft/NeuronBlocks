# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from block_zoo.BaseLayer import BaseLayer, BaseConf
from utils.DocInherit import DocInherit


class ConvPoolingConf(BaseConf):
    """ Configuration of Conv + Pooling architecture

    Args:
        stride (int): the stride of the convolving kernel. Can be a single number or a tuple (sH, sW). Default: 1
        padding (int): implicit zero paddings on both sides of the input. Can be a single number or a tuple (padH, padW). Default: 0
        window_sizes (list): for each window_size, the actual window size is (window_size, feature_dim), because for NLP tasks, 1d convolution is more commonly used.
        input_channel_num (int): for NLP tasks, input_channel_num would always be 1
        output_channel_num (int): number of feature maps
        batch_norm (bool): If True, apply batch normalization before activation
        activation (string): activation functions, e.g. ReLU

    """
    def __int__(self, **kwargs):
        super(ConvPoolingConf, self).__init__(**kwargs)

    @DocInherit
    def default(self):
        self.stride = 1
        self.padding = 0
        self.window_sizes = [1, 2, 3]
        self.input_channel_num = 1      # for NLP tasks, input_channel_num would always be 1
        self.output_channel_num = 16
        self.batch_norm = True
        self.activation = 'ReLU'
        self.pool_type = 'max'  # Supported: ['max', mean']
        self.pool_axis = 1

    @DocInherit
    def declare(self):
        self.num_of_inputs = 1
        self.input_ranks = [3]

    @DocInherit
    def inference(self):
        self.output_dim = [-1]
        self.output_dim.append(self.output_channel_num * len(self.window_sizes))

        super(ConvPoolingConf, self).inference()  # PUT THIS LINE AT THE END OF inference()
    
    @DocInherit
    def verify_before_inference(self):
        super(ConvPoolingConf, self).verify_before_inference()
        necessary_attrs_for_user = ['output_channel_num']
        for attr in necessary_attrs_for_user:
            self.add_attr_exist_assertion_for_user(attr)

    @DocInherit
    def verify(self):
        super(ConvPoolingConf, self).verify()

        necessary_attrs_for_user = ['stride', 'padding', 'window_sizes', 'input_channel_num', 'output_channel_num', 'activation', 'pool_type', 'pool_axis']
        for attr in necessary_attrs_for_user:
            self.add_attr_exist_assertion_for_user(attr)


class ConvPooling(BaseLayer):
    """ Convolution along just 1 direction

    Args:
        layer_conf (ConvConf): configuration of a layer
    """
    def __init__(self, layer_conf):
        super(ConvPooling, self).__init__(layer_conf)
        self.layer_conf = layer_conf

        self.filters = nn.ParameterList()
        if layer_conf.batch_norm:
            self.batch_norms = nn.ModuleList()
        else:
            self.batch_norms = None

        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for i in range(len(layer_conf.window_sizes)):
            self.filters.append(nn.Parameter(torch.randn(layer_conf.output_channel_num, layer_conf.input_channel_num, layer_conf.window_sizes[i], layer_conf.input_dims[0][2], requires_grad=True).float()))
            if layer_conf.batch_norm:
                self.batch_norms.append(nn.BatchNorm2d(layer_conf.output_channel_num))

        if layer_conf.activation:
            self.activation = eval("nn." + self.layer_conf.activation)()
        else:
            self.activation = None

    def forward(self, string, string_len=None):
        """ process inputs

        Args:
            string (Tensor): tensor with shape: [batch_size, seq_len, feature_dim]
            string_len (Tensor):  [batch_size]

        Returns:
            Tensor: shape: [batch_size, (seq_len - conv_window_size) // stride + 1, output_channel_num]

        """
        if string_len is not None:
            string_len_val = string_len.cpu().data.numpy()
            masks = []
            for i in range(len(string_len)):
                masks.append(torch.cat([torch.ones(string_len_val[i]), torch.zeros(string.shape[1] - string_len_val[i])]))
            masks = torch.stack(masks).view(string.shape[0], string.shape[1], 1).expand_as(string)
            if self.is_cuda():
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                masks = masks.to(device)
            string = string * masks

        string = torch.unsqueeze(string, 1)     # [batch_size, input_channel_num=1, seq_len, feature_dim]

        outputs = []
        for idx, (filter, window_size) in enumerate(zip(self.filters, self.layer_conf.window_sizes)):
            string_out = F.conv2d(string, filter, stride=self.layer_conf.stride, padding=self.layer_conf.padding)
            if hasattr(self, 'batch_norms') and self.batch_norms:     #hasattr(self, 'batch_norms') enable NB to be compatible to models trained previously
                string_out = self.batch_norms[idx](string_out)
            string_out = torch.squeeze(string_out, 3).permute(0, 2, 1)
            if self.activation:
                string_out = self.activation(string_out)

            if string_len is not None:
                string_len_out = (string_len - window_size) // self.layer_conf.stride + 1
            else:
                string_len_out = None

            if self.layer_conf.pool_type == "mean":
                assert not string_len_out is None, "Parameter string_len should not be None!"
                string_out = torch.sum(string_out, self.layer_conf.pool_axis).squeeze(self.layer_conf.pool_axis)
                #if not isinstance(string_len_out, torch.Tensor):
                if not torch.is_tensor(string_len_out):
                    string_len_out = torch.FloatTensor(string_len_out)
                string_len_out = string_len_out.unsqueeze(1)
                if self.is_cuda():
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    string_len_out = string_len_out.to(device)
                output = string_out / string_len_out.expand_as(string_out)
            elif self.layer_conf.pool_type == "max":
                output = torch.max(string_out, self.layer_conf.pool_axis)[0]

            outputs.append(output)

        if len(outputs) > 1:
            string_output = torch.cat(outputs, 1)
        else:
            string_output = outputs[0]

        return string_output, None




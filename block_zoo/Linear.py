# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from block_zoo.BaseLayer import BaseLayer, BaseConf
from utils.DocInherit import DocInherit
import logging
import copy


class LinearConf(BaseConf):
    """Configuration for Linear layer

    Args:
        hidden_dim (int or list): if is int, it means there is one linear layer and the hidden_dim is the dimension of that layer.
                if is list of int, it means there is multiple linear layer and hidden_dim are the dimensions of these layers.
        activation (str): Name of activation function. All the non-linear activations in http://pytorch.org/docs/0.3.1/nn.html#non-linear-activations are supported, such as 'Tanh', 'ReLU', 'PReLU', 'ReLU6' and 'LeakyReLU'. Default is None.
        last_hidden_activation (bool): [Optional], whether to add nonlinearity to the last linear layer's output. Default is True.
        last_hidden_softmax (bool): [Optional], whether to add softmax to the last linear layer's output. Default is False.
    """
    def __int__(self, **kwargs):
        super(LinearConf, self).__init__(**kwargs)

    @DocInherit
    def default(self):
        self.hidden_dim = 128
        self.batch_norm = True     # currently, batch_norm for rank 3 inputs is disabled
        self.activation = 'PReLU'
        self.last_hidden_activation = True
        self.last_hidden_softmax = False

    @DocInherit
    def declare(self):
        self.num_of_inputs = 1
        self.input_ranks = [-1]

    @DocInherit
    def inference(self):
        if isinstance(self.hidden_dim, int):
            self.output_dim = copy.deepcopy(self.input_dims[0])
            self.output_dim[-1] = self.hidden_dim
        elif isinstance(self.hidden_dim, list):
            self.output_dim = copy.deepcopy(self.input_dims[0])
            self.output_dim[-1] = self.hidden_dim[-1]

        super(LinearConf, self).inference()  # PUT THIS LINE AT THE END OF inference()

    @DocInherit
    def verify_before_inference(self):
        super(LinearConf, self).verify_before_inference()
        necessary_attrs_for_user = ['hidden_dim']
        for attr in necessary_attrs_for_user:
            self.add_attr_exist_assertion_for_user(attr)

    @DocInherit
    def verify(self):
        super(LinearConf, self).verify()
        necessary_attrs_for_user = ['activation', 'last_hidden_activation', 'last_hidden_softmax']
        for attr in necessary_attrs_for_user:
            self.add_attr_exist_assertion_for_user(attr)

        type_checks = [('activation', [None, str]),
                       ('last_hidden_activation', bool),
                       ('last_hidden_softmax', bool)]
        for attr, attr_type in type_checks:
            self.add_attr_type_assertion(attr, attr_type)

        # supported activation of PyTorch now:
        supported_activation_pytorch = [None, 'Sigmoid', 'Tanh', 'ReLU', 'PReLU', 'ReLU6', 'LeakyReLU', 'LogSigmoid', 'ELU',
                'SELU', 'Threshold', 'Hardtanh', 'Softplus', 'Softshrink', 'Softsign', 'Tanhshrink', 'Softmin',
                'Softmax', 'Softmax2d', 'LogSoftmax']
        value_checks = [('activation', supported_activation_pytorch)]
        for attr, legal_values in value_checks:
            self.add_attr_value_assertion(attr, legal_values)


class Linear(BaseLayer):
    """ Linear layer

    Args:
        layer_conf (LinearConf): configuration of a layer
    """
    def __init__(self, layer_conf):

        super(Linear, self).__init__(layer_conf)

        if layer_conf.input_ranks[0] == 3 and layer_conf.batch_norm is True:
            layer_conf.batch_norm = False
            logging.warning('Batch normalization for dense layers of which the rank is 3 is not available now. Batch norm is set to False now.')

        if isinstance(layer_conf.hidden_dim, int):
            layer_conf.hidden_dim = [layer_conf.hidden_dim]

        layers = OrderedDict()
        former_dim = layer_conf.input_dims[0][-1]
        for i in range(len(layer_conf.hidden_dim)):
            #cur_layer_name = 'linear_%d' % len(layers)
            layers['linear_%d' % len(layers)] = nn.Linear(former_dim, layer_conf.hidden_dim[i])
            if layer_conf.activation is not None and \
                    (layer_conf.last_hidden_activation is True or (i != len(layer_conf.hidden_dim) - 1)):
                try:
                    if layer_conf.batch_norm:
                        layers['batch_norm_%d' % len(layers)] = nn.BatchNorm1d(layer_conf.hidden_dim[i])
                    layers['linear_activate_%d' % len(layers)] = eval("nn." + layer_conf.activation)()
                except NameError as e:
                    raise Exception("%s; Activation layer \"nn.%s\"" % (str(e), layer_conf.activation))

            if layer_conf.last_hidden_softmax is True and i == len(layer_conf.hidden_dim) - 1:
                layers['linear_softmax_%d' % len(layers)] = nn.Softmax(layer_conf.output_rank - 1)

            former_dim = layer_conf.hidden_dim[i]

        self.linear = nn.Sequential(layers)

    def forward(self, string, string_len=None):
        """ process inputs

        Args:
            string (Tensor): any shape.
            string_len (Tensor): [batch_size], default is None.

        Returns:
            Tensor: has the same shape as string.
        """
        if self.layer_conf.input_ranks[0] == 3 and string_len is not None:
            # need padding mask
            string_len_val = string_len.cpu().data.numpy()
            masks = []
            for i in range(len(string_len)):
                masks.append(
                    torch.cat([torch.ones(string_len_val[i]), torch.zeros(string.shape[1] - string_len_val[i])]))
            masks = torch.stack(masks).view(string.shape[0], string.shape[1], 1).expand_as(string)
            if self.is_cuda():
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                masks = masks.to(device)
            string = string * masks
        string_out = self.linear(string.float())
        return string_out, string_len



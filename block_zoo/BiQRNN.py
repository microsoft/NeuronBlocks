# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn

import numpy as np
from block_zoo.BaseLayer import BaseLayer, BaseConf
from utils.DocInherit import DocInherit
import copy


class ForgetMult(torch.nn.Module):
    """ForgetMult computes a simple recurrent equation:
    h_t = f_t * x_t + (1 - f_t) * h_{t-1}

    This equation is equivalent to dynamic weighted averaging.

    Inputs: X, hidden
        - X (seq_len, batch, input_size): tensor containing the features of the input sequence.
        - F (seq_len, batch, input_size): tensor containing the forget gate values, assumed in range [0, 1].
        - hidden_init (batch, input_size): tensor containing the initial hidden state for the recurrence (h_{t-1}).
    """

    def __init__(self):
        super(ForgetMult, self).__init__()

    def forward(self, f, x, hidden_init=None):
        result = []
        forgets = f.split(1, dim=0)
        prev_h = hidden_init
        for i, h in enumerate((f * x).split(1, dim=0)):
            if prev_h is not None: h = h + (1 - forgets[i]) * prev_h
            # h is (1, batch, hidden) when it needs to be (batch_hidden)
            # Calling squeeze will result in badness if batch size is 1
            h = h.view(h.size()[1:])
            result.append(h)
            prev_h = h
        return torch.stack(result)


class QRNNLayer(nn.Module):
    """Applies a single layer Quasi-Recurrent Neural Network (QRNN) to an input sequence.

    Args:
        input_size: The number of expected features in the input x.
        hidden_size: The number of features in the hidden state h. If not specified, the input size is used.
        save_prev_x: Whether to store previous inputs for use in future convolutional windows (i.e. for a continuing sequence such as in language modeling). If true, you must call reset to remove cached previous values of x. Default: False.
        window: Defines the size of the convolutional window (how many previous tokens to look when computing the QRNN values). Supports 1 and 2. Default: 1.
        zoneout: Whether to apply zoneout (i.e. failing to update elements in the hidden state) to the hidden state updates. Default: 0.
        output_gate: If True, performs QRNN-fo (applying an output gate to the output). If False, performs QRNN-f. Default: True.

    Inputs: X, hidden
        - X (seq_len, batch, input_size): tensor containing the features of the input sequence.
        - hidden (batch, hidden_size): tensor containing the initial hidden state for the QRNN.

    Outputs: output, h_n
        - output (seq_len, batch, hidden_size): tensor containing the output of the QRNN for each timestep.
        - h_n (1, batch, hidden_size): tensor containing the hidden state for t=seq_len
    """

    def __init__(self, input_size, hidden_size=None, save_prev_x=False, zoneout=0, window=1, output_gate=True):
        super(QRNNLayer, self).__init__()

        assert window in [1, 2], "This QRNN implementation currently only handles convolutional window of size 1 or size 2"
        self.window = window
        self.input_size = input_size
        self.hidden_size = hidden_size if hidden_size else input_size
        self.zoneout = zoneout
        self.save_prev_x = save_prev_x
        self.prevX = None
        self.output_gate = output_gate

        # One large matmul with concat is faster than N small matmuls and no concat
        self.linear = nn.Linear(self.window * self.input_size, 3 * self.hidden_size if self.output_gate else 2 * self.hidden_size)

    def reset(self):
        # If you are saving the previous value of x, you should call this when starting with a new state
        self.prevX = None

    def forward(self, X, hidden=None):
        seq_len, batch_size, _ = X.size()

        source = None
        if self.window == 1:
            source = X
        elif self.window == 2:
            # Construct the x_{t-1} tensor with optional x_{-1}, otherwise a zeroed out value for x_{-1}
            Xm1 = []
            Xm1.append(self.prevX if self.prevX is not None else X[:1, :, :] * 0)
            # Note: in case of len(X) == 1, X[:-1, :, :] results in slicing of empty tensor == bad
            if len(X) > 1:
                Xm1.append(X[:-1, :, :])
            Xm1 = torch.cat(Xm1, 0)
            # Convert two (seq_len, batch_size, hidden) tensors to (seq_len, batch_size, 2 * hidden)
            source = torch.cat([X, Xm1], 2)

        # Matrix multiplication for the three outputs: Z, F, O
        Y = self.linear(source)
        # Convert the tensor back to (batch, seq_len, len([Z, F, O]) * hidden_size)
        if self.output_gate:
            Y = Y.view(seq_len, batch_size, 3 * self.hidden_size)
            Z, F, O = Y.chunk(3, dim=2)
        else:
            Y = Y.view(seq_len, batch_size, 2 * self.hidden_size)
            Z, F = Y.chunk(2, dim=2)
        ###
        Z = torch.tanh(Z)
        F = torch.sigmoid(F)

        # If zoneout is specified, we perform dropout on the forget gates in F
        # If an element of F is zero, that means the corresponding neuron keeps the old value
        if self.zoneout:
            if self.training:
                # mask = Variable(F.data.new(*F.size()).bernoulli_(1 - self.zoneout), requires_grad=False)
                mask = F.new_empty(F.size(), requires_grad=False).bernoulli_(1 - self.zoneout)
                F = F * mask
            else:
                F *= 1 - self.zoneout

        # Forget Mult
        C = ForgetMult()(F, Z, hidden)

        # Apply (potentially optional) output gate
        if self.output_gate:
            H = torch.sigmoid(O) * C
        else:
            H = C

        # In an optimal world we may want to backprop to x_{t-1} but ...
        if self.window > 1 and self.save_prev_x:
            # self.prevX = Variable(X[-1:, :, :].data, requires_grad=False)
            self.prevX = X[-1:, :, :].detach()

        return H, C[-1:, :, :]


class QRNN(torch.nn.Module):
    """Applies a multiple layer Quasi-Recurrent Neural Network (QRNN) to an input sequence.

    Args:
        input_size: The number of expected features in the input x.
        hidden_size: The number of features in the hidden state h. If not specified, the input size is used.
        num_layers: The number of QRNN layers to produce.
        dropout: Whether to use dropout between QRNN layers. Default: 0.
        bidirectional: If True, becomes a bidirectional QRNN. Default: False.
        save_prev_x: Whether to store previous inputs for use in future convolutional windows (i.e. for a continuing sequence such as in language modeling). If true, you must call reset to remove cached previous values of x. Default: False.
        window: Defines the size of the convolutional window (how many previous tokens to look when computing the QRNN values). Supports 1 and 2. Default: 1.
        zoneout: Whether to apply zoneout (i.e. failing to update elements in the hidden state) to the hidden state updates. Default: 0.
        output_gate: If True, performs QRNN-fo (applying an output gate to the output). If False, performs QRNN-f. Default: True.

    Inputs: X, hidden
        - X (seq_len, batch, input_size): tensor containing the features of the input sequence.
        - hidden (num_layers * num_directions, batch, hidden_size): tensor containing the initial hidden state for the QRNN.

    Outputs: output, h_n
        - output (seq_len, batch, hidden_size * num_directions): tensor containing the output of the QRNN for each timestep.
        - h_n (num_layers * num_directions, batch, hidden_size): tensor containing the hidden state for t=seq_len
    """

    def __init__(self, input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0.0, bidirectional=False, **kwargs):
        # assert bidirectional == False, 'Bidirectional QRNN is not yet supported'
        assert batch_first == False, 'Batch first mode is not yet supported'
        assert bias == True, 'Removing underlying bias is not yet supported'

        super(QRNN, self).__init__()

        # self.layers = torch.nn.ModuleList(layers if layers else [QRNNLayer(input_size if l == 0 else hidden_size, hidden_size, **kwargs) for l in range(num_layers)])
        if bidirectional:
            self.layers = torch.nn.ModuleList(
                [QRNNLayer(input_size if l < 2 else hidden_size * 2, hidden_size, **kwargs) for l in
                 range(num_layers * 2)])
        else:
            self.layers = torch.nn.ModuleList(
                [QRNNLayer(input_size if l == 0 else hidden_size, hidden_size, **kwargs) for l in
                 range(num_layers)])

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        assert len(self.layers) == self.num_layers * self.num_directions

    def tensor_reverse(self, tensor):
        # idx = [i for i in range(tensor.size(0) - 1, -1, -1)]
        # idx = torch.LongTensor(idx)
        # inverted_tensor = tensor.index_select(0, idx)
        return tensor.flip(0)

    def reset(self):
        r'''If your convolutional window is greater than 1, you must reset at the beginning of each new sequence'''
        [layer.reset() for layer in self.layers]

    def forward(self, input, hidden=None):
        next_hidden = []
        for i in range(self.num_layers):
            all_output = []
            for j in range(self.num_directions):
                l = i * self.num_directions + j
                layer = self.layers[l]
                if j == 1:
                    input = self.tensor_reverse(input)  # reverse
                output, hn = layer(input, None if hidden is None else hidden[l])
                next_hidden.append(hn)
                if j == 1:
                    output = self.tensor_reverse(output)    # reverse
                all_output.append(output)

            input = torch.cat(all_output, input.dim() - 1)
            if self.dropout != 0 and i < self.num_layers - 1:
                input = torch.nn.functional.dropout(input, p=self.dropout, training=self.training, inplace=False)

        next_hidden = torch.cat(next_hidden, 0).view(self.num_layers * self.num_directions, *next_hidden[0].size()[-2:])

        # for i, layer in enumerate(self.layers):
        #     input, hn = layer(input, None if hidden is None else hidden[i])
        #     next_hidden.append(hn)
        #
        #     if self.dropout != 0 and i < len(self.layers) - 1:
        #         input = torch.nn.functional.dropout(input, p=self.dropout, training=self.training, inplace=False)
        #
        # next_hidden = torch.cat(next_hidden, 0).view(self.num_layers, *next_hidden[0].size()[-2:])

        return input, next_hidden


class BiQRNNConf(BaseConf):
    """ Configuration of BiQRNN

    Args:
        hidden_dim (int): dimension of hidden state
        window: the size of the convolutional window. Supports 1 and 2. Default: 1
        zoneout: Whether to apply zoneout (failing to update elements in the hidden state). Default: 0
        dropout (float): dropout rate bewteen BiQRNN layers
        num_layers (int): number of BiQRNN layers
    """
    def __init__(self, **kwargs):
        super(BiQRNNConf, self).__init__(**kwargs)

    @DocInherit
    def default(self):
        self.hidden_dim = 128
        self.window = 1
        self.zoneout = 0.0
        self.dropout = 0.0
        self.num_layers = 1

    @DocInherit
    def declare(self):
        self.num_of_inputs = 1
        self.input_ranks = [3]

    @DocInherit
    def inference(self):
        self.output_dim = copy.deepcopy(self.input_dims[0])
        self.output_dim[-1] = 2 * self.hidden_dim

        super(BiQRNNConf, self).inference()      # PUT THIS LINE AT THE END OF inference()

    @DocInherit
    def verify(self):
        super(BiQRNNConf, self).verify()

        necessary_attrs_for_user = ['hidden_dim', 'window', 'zoneout', 'dropout', 'num_layers']
        for attr in necessary_attrs_for_user:
            self.add_attr_exist_assertion_for_user(attr)


class BiQRNN(BaseLayer):
    """ Bidrectional QRNN

    Args:
        layer_conf (BiQRNNConf): configuration of a layer
    """
    def __init__(self, layer_conf):
        super(BiQRNN, self).__init__(layer_conf)
        self.qrnn = QRNN(layer_conf.input_dims[0][-1], layer_conf.hidden_dim, layer_conf.num_layers,
                         window=layer_conf.window, zoneout=layer_conf.zoneout, dropout=layer_conf.dropout,
                         bidirectional=True)

    def forward(self, string, string_len):
        """ process inputs

        Args:
            string (Tensor): [batch_size, seq_len, dim]
            string_len (Tensor): [batch_size]

        Returns:
            Tensor: [batch_size, seq_len, 2 * hidden_dim]

        """
        string = string.transpose(0, 1)
        string_output = self.qrnn(string)[0]
        string_output = string_output.transpose(0, 1)

        return string_output, string_len

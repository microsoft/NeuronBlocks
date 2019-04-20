# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import ujson as json
import codecs
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod

from utils.exceptions import LayerDefineError, ConfigurationError


class BaseConf(ABC):
    """Basic configuration

    Args:
        input_dim (int): the dimension of input.
        hidden_dim (int): the dimension of hidden state.
        dropout (float): dropout rate.
        (others)...
    """
    def __init__(self, **kwargs):
        self.default()

        for key in kwargs:
            setattr(self, key, kwargs[key])

        self.declare()

        self.verify_before_inference()

        # Moved to get_conf() in Model.py
        #self.inference()
        #self.verify()

    def default(self):
        """ Define the default hyper parameters here. You can define these hyper parameters in your configuration file as well.

        Returns:
            None

        """
        #self.input_dims = [xxx, xxx]  would be inferenced automatically
        self.hidden_dim = 10


    @abstractmethod
    def declare(self):
        """ Define things like "input_ranks" and "num_of_inputs", which are certain with regard to your layer

            num_of_input is N(N>0) means this layer accepts N inputs;

            num_of_input is -1 means this layer accepts any number of inputs;

            The rank here is not the same as matrix rank:

              For a scalar, its rank is 0;\n
              For a vector, its rank is 1;\n
              For a matrix, its rank is 2;\n
              For a cube of numbers, its rank is 3.\n
            ...
            For instance, the rank of (batch size, sequence length, hidden_dim) is 3.

            if num_of_input > 0:

              len(input_ranks) should be equal to num_of_input

            elif num_of_input == -1:

              input_ranks should be a list with only one element and the rank of all the inputs should be equal to that element.

            NOTE: when we build the model, if num_of_input is -1, we would replace it with the real number of inputs and replace input_ranks with a list of real input_ranks.

        Returns:
            None

        """
        self.num_of_inputs = 1
        self.input_ranks = [3]

    @abstractmethod
    def inference(self):
        """ Inference things like output_dim, which may relies on defined hyper parameter such as hidden dim and input_dim

        Returns:
            None

        """
        # inference the output dim and output rank from inputs. Here are some examples:
        #self.output_dim = copy.deepcopy(self.input_dims[0])
        #self.output_dim[-1] = sum([input_dim[-1] for input_dim in self.input_dims])

        self.output_rank = len(self.output_dim)  # DON'T MODIFY THIS

    def verify_before_inference(self):
        """ Some conditions must be fulfilled, otherwise there would be errors when calling inference()

        The difference between verify_before_inference() and verify() is that:
            verify_before_inference() is called before inference() while verify() is called after inference().

        Returns:
            None

        """
        necessary_attrs_for_dev = ['num_of_inputs', 'input_ranks']
        for attr in necessary_attrs_for_dev:
            self.add_attr_exist_assertion_for_dev(attr)

        type_checks = [('num_of_inputs', int),
                       ('input_ranks', list)]
        for attr, attr_type in type_checks:
            self.add_attr_type_assertion(attr, attr_type)


    def verify(self):
        """ Define some necessary varification for your layer when we define the model.

        If you define your own layer and rewrite this funciton, please add "super(YourLayerConf, self).verify()" at the beginning

        Returns:
            None

        """
        self.verify_before_inference()      # prevent to be neglected, verify again

        necessary_attrs_for_user = []
        for attr in necessary_attrs_for_user:
            self.add_attr_exist_assertion_for_user(attr)

        necessary_attrs_for_dev = ['input_dims', 'output_dim', 'output_rank', 'use_gpu']
        for attr in necessary_attrs_for_dev:
            self.add_attr_exist_assertion_for_dev(attr)

        type_checks = [('output_dim', list),
                       ('input_dims', list),
                       ('output_rank', int)]
        for attr, attr_type in type_checks:
            self.add_attr_type_assertion(attr, attr_type)

        range_checks = [('dropout', (0, 1), (True, True))]
        for attr, ranges, bound_legal in range_checks:
            self.add_attr_range_assertion(attr, ranges, bound_legal)

        # demonstration for value checks
        '''
        value_checks = [('some_attr', ['legal_value1', 'legal_value2', 'legal_value3'])]
        for attr, legal_values in value_checks:
            self.add_attr_value_assertion(attr, legal_values)
        '''

        # To check if deepcopy is applied
        assert id(self.output_dim) != id(self.input_dims[0]), 'Please use copy.deepcopy to copy the input_dim to output_dim'


    def add_attr_type_assertion(self, attr, specified_type):
        """ check if the types of attributes are legal

        Args:
            attr (str): the attribution name
            specified_type (None/str/list): one specified_type of a list of specified_type(including None)

        Returns:
            None

        """
        if not hasattr(self, attr):
            return

        if isinstance(specified_type, list):
            hit_flag = False
            for specified_type_ in specified_type:
                if specified_type_ is None:
                    if getattr(self, attr) is None:
                        hit_flag = True
                        break
                else:
                    if isinstance(getattr(self, attr), specified_type_):
                        hit_flag = True
                        break

            if hit_flag is False:
                raise Exception("For layer %s, the attribute %s should be one of [%s]!" % (
                    type(self).__name__, attr, ", ".join(specified_type_.__name__ if specified_type_ is not None else "None" for specified_type_ in specified_type)))
        else:
            if not (getattr(self, attr) is None and specified_type is None or isinstance(getattr(self, attr), specified_type)):
                raise LayerDefineError("For layer %s, the attribute %s should be a/an %s!" %
                            (type(self).__name__, attr, specified_type.__name__))

    def add_attr_range_assertion(self, attr, range, bounds_legal=(True, True)):
        """ check if attribute falls into the legal range

        Args:
            attr (str): the attribution name
            range (tuple): (num/float('-inf')/None, num/float('inf')/None), None means -inf or inf.
            bounds_legal (tuple): (bool, bool), if the left/right bound is legal

        Returns:
            None

        """
        if not hasattr(self, attr):
            return

        value = getattr(self, attr)
        range = list(range)
        bounds_legal = list(bounds_legal)
        if range[0] is None:
            range[0] = float('-inf')
        if range[1] is None:
            range[1] = float('inf')
        if range[0] == float('-inf'):
            bounds_legal[0] = False
        if range[1] == float('inf'):
            bounds_legal[1] = False

        left_bound_ch = '[' if bounds_legal[0] else '('
        right_bound_ch = ']' if bounds_legal[1] else ')'
        if not ((bounds_legal[0] and value >= range[0] or bounds_legal[0] is False and value > range[0]) and (
                bounds_legal[1] and value <= range[1] or bounds_legal[1] is False and value < range[1])):
            raise Exception("For layer %s, the legal range of attribute %s is %s%f, %f%s" % (
                type(self).__name__, attr, left_bound_ch, range[0], range[1], right_bound_ch))

    def add_attr_exist_assertion_for_dev(self, attr):
        """ check if there are some attributes being forgot by developers

        Args:
            attr (str): the attribution name

        Returns:
            None

        """
        if not hasattr(self, attr):
            raise LayerDefineError("For layer %s, please define %s attribute in declare() or inference()!" % (type(self).__name__, attr))

    def add_attr_exist_assertion_for_user(self, attr):
        """ check if there are some attributes being forgot by users

        Args:
            attr (str): the attribution name

        Returns:
            None

        """
        if not hasattr(self, attr):
            raise ConfigurationError("For layer %s, please configure %s attribute for %s in the configuration file!" % (type(self).__name__, attr, type(self).__name__))

    def add_attr_value_assertion(self, attr, legal_values):
        """ check if attr equals to one of the legal values

        Args:
            attr (str): the attribution name
            legal_values (list): include the legal value

        Returns:
            None

        """
        if not hasattr(self, attr):
            return
        hit_flag = False
        for legal_value in legal_values:
            if getattr(self, attr) == legal_value:
                hit_flag = True
                break
        if hit_flag is False:
            raise Exception("For layer %s, attribute %s should be one of [%s], but you give %s." % \
                (type(self).__name__, attr, ", ".join(str(legal_value) for legal_value in legal_values), str(getattr(self, attr))))


class BaseLayer(nn.Module):
    """The base class of layers

    Args:
        layer_conf (BaseConf): configuration of a layer
    """
    def __init__(self, layer_conf):
        super(BaseLayer, self).__init__()
        self.layer_conf = layer_conf

    def forward(self, *args):
        """

        Args:
            *args (list): a list of args in which arg should be a pair of (representation, length)

        Returns:
            None

        """
        pass

    def is_cuda(self):
        """ To judge if the layer is on CUDA
        if there are parameters in this layer, judge according to the parameters;
        else: judge according to the self.layer_conf.use_gpu

        Returns:
            bool: whether to use gpu

        """
        try:
            # In case someone forget to check use_gpu flag, this function would first detect if the parameters are on cuda
            # ret = next(self.parameters()).data.is_cuda
            ret = self.layer_conf.use_gpu
        except StopIteration as e:
            if not hasattr(self, 'layer_conf'):
                logging.error('Layer.layer_conf must be defined!')
            else:
                logging.error(e)

        return ret


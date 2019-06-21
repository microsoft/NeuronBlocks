# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn
from block_zoo import *
import copy
import logging
from utils.exceptions import ConfigurationError, LayerUndefinedError, LayerConfigUndefinedError
from queue import Queue
from utils.common_utils import transform_tensors2params, transfer_to_gpu

from block_zoo.Embedding import *

EMBED_LAYER_NAME = 'Embedding'
EMBED_LAYER_ID = 'embedding'


def get_conf(layer_id, layer_name, input_layer_ids, all_layer_configs, model_input_ids, use_gpu,
        conf_dict=None, shared_conf=None, succeed_embedding_flag=False, output_layer_flag=False,
        target_num=None, fixed_lengths=None, target_dict=None):
    """ get layer configuration

    Args
        layer_id: layer identifier
        layer_name: name of layer such as BiLSTM
        input_layer_ids (list): the inputs of current layer
        all_layer_configs (dict): records the conf class of each layer.
        model_input_ids (set): the inputs of the model, e.g. ['query', 'passage']
        use_gpu:
        conf_dict:
        shared_conf: if fixed_lengths is not None, the output_dim of shared_conf should be corrected!
        flag:
        output_layer_flag:
        target_num: used for inference the dimension of output space if someone declare a dimension of -1
        fixed_lengths
    Returns:
        configuration class coresponds to the layer

    """
    if shared_conf:
        conf = copy.deepcopy(shared_conf)
    else:
        try:
            conf_dict['use_gpu'] = use_gpu

            # for Embedding layer, add weight_on_gpu parameters
            if layer_id == EMBED_LAYER_ID:
                conf_dict['weight_on_gpu'] = conf_dict['conf']['weight_on_gpu']
                del conf_dict['conf']['weight_on_gpu']

            # for classification tasks, we usually add a Linear layer to project the output to dimension of number of classes. If we don't know the #classes, we can use '-1' instead and we would calculate the number of classes from the corpus.
            if layer_name == 'Linear':
                if isinstance(conf_dict['hidden_dim'], list):
                    if conf_dict['hidden_dim'][-1] == -1:
                        assert output_layer_flag is True, "Only in the last layer, hidden_dim == -1 is allowed!"
                        assert target_num is not None, "Number of targets should be given!"
                        conf_dict['hidden_dim'][-1] = target_num
                    elif conf_dict['hidden_dim'][-1] == '#target#':
                        logging.info('#target# position will be replace by target num: %d' % target_num)
                        conf_dict['hidden_dim'][-1] = target_num
                elif isinstance(conf_dict['hidden_dim'], int) and conf_dict['hidden_dim'] == -1:
                    assert output_layer_flag is True, "Only in the last layer, hidden_dim == -1 is allowed!"
                    assert target_num is not None, "Number of targets should be given!"
                    conf_dict['hidden_dim'] = target_num
                elif isinstance(conf_dict['hidden_dim'], str) and conf_dict['hidden_dim'] == '#target#':
                    logging.info('#target# position will be replace by target num: %d' % target_num)
                    conf_dict['hidden_dim'] = target_num
            # add some necessary attribute for CRF layer
            if layer_name == 'CRF':
                conf_dict['target_dict'] = target_dict

            conf = eval(layer_name + "Conf")(**conf_dict)
        except NameError as e:
            raise LayerConfigUndefinedError("\"%sConf\" has not been defined" % layer_name)

    # verify the rank consistence of joint layers
    if layer_name == EMBED_LAYER_NAME:
        # the embedding layer
        pass
    else:
        # make sure all the inputs to current layer exist
        for input_layer_id in input_layer_ids:
            if not (input_layer_id in all_layer_configs or input_layer_id in model_input_ids):
                raise ConfigurationError("The input %s of layer %s does not exist. Please define it before "
                    "defining layer %s!" % (input_layer_id, layer_id, layer_id))

        former_output_ranks = [all_layer_configs[input_layer_id].output_rank if input_layer_id in all_layer_configs else all_layer_configs[EMBED_LAYER_ID].output_rank for input_layer_id in input_layer_ids]
        # inference input_dim
        conf.input_dims = [all_layer_configs[input_layer_id].output_dim if input_layer_id in all_layer_configs else all_layer_configs[EMBED_LAYER_ID].output_dim for input_layer_id in input_layer_ids]

        # If the inputs come from embedding layer and fixed_lengths exist, set the length to input_dims
        if len(input_layer_ids) == 1 and input_layer_ids[0] in model_input_ids and fixed_lengths:
            conf.input_dims[0][1] = fixed_lengths[input_layer_ids[0]]

        # check and verify input ranks
        if conf.num_of_inputs > 0:
            if conf.num_of_inputs != len(input_layer_ids):
                raise ConfigurationError("%s only accept %d inputs but you feed %d inputs to it!" % \
                        (layer_name, conf.num_of_inputs, len(input_layer_ids)))
        elif conf.num_of_inputs == -1:
            conf.num_of_inputs = len(input_layer_ids)
            if isinstance(conf.input_ranks, list):
                conf.input_ranks = conf.input_ranks * conf.num_of_inputs
            else:
                logging.warning("[For developer of %s] The input_ranks attribute should be a list!" % (layer_name))
                [conf.input_ranks] * conf.num_of_inputs

        for input_rank, former_output_rank in zip(conf.input_ranks, former_output_ranks):
            if input_rank != -1 and input_rank != former_output_rank:
                raise ConfigurationError("Input ranks of %s are inconsistent with former layers" % layer_id)
        conf.input_ranks = copy.deepcopy(former_output_ranks)

    # inference and varification inside the layer
    conf.inference()        # update some attributes which relies on input dimension or something else
    conf.verify()           # verify if the configuration is legal
    former_conf = None if len(all_layer_configs) == 0 else list(all_layer_configs.values())[-1]
    conf.verify_former_block(former_conf)  # check if has special attribute rely on former layer

    logging.debug('Layer id: %s; name: %s; input_dims: %s; input_ranks: %s; output_dim: %s; output_rank: %s' % (layer_id, layer_name, conf.input_dims if layer_id != 'embedding' else 'None', conf.input_ranks, conf.output_dim, conf.output_rank))

    return conf


def get_layer(layer_name, conf):
    """

    Args:
        layer_name:
        conf:  configuration class

    Returns:
        specific layer

    """
    try:
        layer = eval(layer_name)(conf)
    except NameError as e:
        raise Exception("%s; Layer \"%s\" has not been defined" % (str(e), layer_name))
    return layer


class Model(nn.Module):
    def __init__(self, conf, problem, vocab_info, use_gpu):
        """

        Args:
            inputs: ['string1', 'string2']
            layer_archs:  The layers must produce tensors with similar shapes. The layers may be nested.
                [
                    {
                    'layer': Layer name,
                    'conf': {xxxx}
                    },
                    [
                        {
                        'layer': Layer name,
                        'conf': {},
                        },
                        {
                        'layer': Layer name,
                        'conf': {},
                        }
                    ]
                ]
            vocab_info:
                {
                    'word':  {
                        'vocab_size': xxx,
                        'init_weights': np matrix
                        }
                    'postag': {
                        'vocab_size': xxx,
                        'init_weights': None
                        }
                }
        """
        super(Model, self).__init__()

        inputs = conf.object_inputs_names
        layer_archs = conf.architecture
        target_num = problem.output_target_num()

        # correct the real fixed length if begin/end of sentence are added
        if conf.fixed_lengths:
            fixed_lengths_corrected = copy.deepcopy(conf.fixed_lengths)
            for seq in fixed_lengths_corrected:
                if problem.with_bos_eos:
                    fixed_lengths_corrected[seq] += 2
        else:
            fixed_lengths_corrected = None

        self.use_gpu = use_gpu

        all_layer_configs = dict()
        self.layers = nn.ModuleDict()
        self.layer_inputs = dict()
        self.layer_dependencies = dict()
        self.layer_dependencies[EMBED_LAYER_ID] = set()
        # change output_layer_id to list for support multi_output
        self.output_layer_id = []

        for layer_index, layer_arch in enumerate(layer_archs):
            output_layer_flag = True if 'output_layer_flag' in layer_arch and layer_arch['output_layer_flag'] is True else False
            succeed_embedding_flag = True if layer_index > 0 and 'inputs' in layer_arch and \
                    [input in inputs for input in layer_arch['inputs']].count(True) == len(layer_arch['inputs']) else False

            if output_layer_flag:
                self.output_layer_id.append(layer_arch['layer_id'])
                # if hasattr(self, 'output_layer_id'):
                #     raise ConfigurationError("There should be only one output!")
                # else:
                #     self.output_layer_id = layer_arch['layer_id']

            if layer_index == 0:
                # embedding layer
                emb_conf = copy.deepcopy(vocab_info)
                for input_cluster in emb_conf:
                    emb_conf[input_cluster]['dim'] = layer_arch['conf'][input_cluster]['dim']
                    emb_conf[input_cluster]['fix_weight'] = layer_arch['conf'][input_cluster].get('fix_weight', False)
                emb_conf['weight_on_gpu'] = layer_arch.get('weight_on_gpu', True)

                all_layer_configs[EMBED_LAYER_ID] = get_conf(EMBED_LAYER_ID, layer_arch['layer'],
                    None, all_layer_configs, inputs, self.use_gpu, conf_dict={'conf': emb_conf},
                    shared_conf=None, succeed_embedding_flag=False, output_layer_flag=output_layer_flag,
                    target_num=target_num, fixed_lengths=fixed_lengths_corrected, target_dict=problem.output_dict)
                self.add_layer(EMBED_LAYER_ID, get_layer(layer_arch['layer'], all_layer_configs[EMBED_LAYER_ID]))
            else:
                if layer_arch['layer'] in self.layers and not 'conf' in layer_arch:
                    # reuse formly defined layers (share the same parameters)
                    logging.debug("Layer id: %s; Sharing configuration with layer %s" % (layer_arch['layer_id'], layer_arch['layer']))
                    conf_dict = None
                    shared_conf = all_layer_configs[layer_arch['layer']]
                else:
                    conf_dict = layer_arch['conf']
                    shared_conf = None

                # if the layer is EncoderDecoder, inference the vocab size
                if layer_arch['layer'] == 'EncoderDecoder':
                        layer_arch['conf']['decoder_conf']['decoder_vocab_size'] = target_num
                all_layer_configs[layer_arch['layer_id']] = get_conf(layer_arch['layer_id'], layer_arch['layer'],
                    layer_arch['inputs'], all_layer_configs, inputs, self.use_gpu, conf_dict=conf_dict,
                    shared_conf=shared_conf, succeed_embedding_flag=succeed_embedding_flag,
                    output_layer_flag=output_layer_flag, target_num=target_num,
                    fixed_lengths=fixed_lengths_corrected, target_dict=problem.output_dict)

                if layer_arch['layer'] in self.layers and not 'conf' in layer_arch:
                    self.add_layer(layer_arch['layer_id'], self.layers[layer_arch['layer']])
                else:
                    self.add_layer(layer_arch['layer_id'], get_layer(layer_arch['layer'], all_layer_configs[layer_arch['layer_id']]))

                self.layer_inputs[layer_arch['layer_id']] = layer_arch['inputs']

                # register dependencies, except embeddings
                cur_layer_depend = set()
                for layer_depend_id in layer_arch['inputs']:
                    if not layer_depend_id in inputs:
                        cur_layer_depend.add(layer_depend_id)
                self.add_dependency(layer_arch['layer_id'], cur_layer_depend)

        logging.debug("Layer dependencies: %s" % repr(self.layer_dependencies))

        if not hasattr(self, 'output_layer_id'):
            raise ConfigurationError("Please define an output layer")

        self.layer_topological_sequence = self.get_topological_sequence()

    def add_layer(self, layer_id, layer):
        """ register a layer

        Args:
            layer_id:
            layer:

        Returns:

        """
        if layer_id in self.layers:
            raise ConfigurationError("The layer id %s is not unique!")
        else:
            self.layers[layer_id] = layer

    def add_dependency(self, layer_id, depend_layer_id):
        """ add the layers have to be proceed before layer_id

        Args:
            layer_id:
            depend_layer_id:

        Returns:

        """
        if not layer_id in self.layer_dependencies:
            self.layer_dependencies[layer_id] = set()

        if isinstance(depend_layer_id, int):
            self.layer_dependencies[layer_id].add(depend_layer_id)
        else:
            self.layer_dependencies[layer_id] |= set(depend_layer_id)

    def remove_dependency(self, depend_layer_id):
        """ remove dependencies on layer_id

        Args:
            layer_id:

        Returns:

        """
        for layer_id in self.layer_dependencies:
            self.layer_dependencies[layer_id].remove(depend_layer_id)

    def get_topological_sequence(self):
        """ get topological sequence of nodes in the model

        Returns:

        """
        total_layer_ids = Queue()
        for layer_id in self.layers.keys():
            if layer_id != EMBED_LAYER_ID:
                total_layer_ids.put(layer_id)

        topological_list = []
        circular_cnt = 0     # used for checking if there is at least one legal topological sorting
        while not total_layer_ids.empty():
            layer_id = total_layer_ids.get()
            if len(self.layer_dependencies[layer_id]) == 0:
                for layer_id2 in self.layer_dependencies:
                    if layer_id in self.layer_dependencies[layer_id2]:
                        self.layer_dependencies[layer_id2].remove(layer_id)
                circular_cnt = 0
                topological_list.append(layer_id)
            else:
                total_layer_ids.put(layer_id)
                circular_cnt += 1
                if circular_cnt >= total_layer_ids.qsize():
                    rest_layers = []
                    while not total_layer_ids.empty():
                        rest_layers.append(total_layer_ids.get())
                    raise ConfigurationError("The model architecture is illegal because there is a circular dependency "
                        "or there are some isolated layers. The layers can not be resolved: [%s]" % (", ".join(rest_layers)))

        logging.debug("Topological sequence of nodes: %s" % (",".join(topological_list)))
        return topological_list

    def forward(self, inputs_desc, length_desc, *param_list):
        """

        Args:
            with the help of transform_tensors2params(inputs_desc, length_desc, param_list), we can get the below inputs and lengths

            inputs: dict.
                {
                    "string1":{
                        'word': word ids, [batch size, seq len]
                        'postag': postag ids,[batch size, seq len]
                        ...
                    }
                    "string2":{
                        'word': word ids,[batch size, seq len]
                        'postag': postag ids,[batch size, seq len]
                        ...
                    }
                }
            lengths: dict.
                {
                    "string1": [...]
                    "string2": [...]
                }

        Returns:

        """
        inputs, lengths = transform_tensors2params(inputs_desc, length_desc, param_list)

        representation = dict()
        representation[EMBED_LAYER_ID] = dict()
        repre_lengths = dict()
        repre_lengths[EMBED_LAYER_ID] = dict()

        for input in inputs:
            representation[input] = self.layers[EMBED_LAYER_ID](inputs[input], lengths[input], use_gpu=self.is_cuda())
            if self.use_gpu:
                repre_lengths[input] = transfer_to_gpu(lengths[input]['sentence_length'])
            else:
                repre_lengths[input] = lengths[input]['sentence_length']

        for layer_id in self.layer_topological_sequence:
            #logging.debug("To proces layer %s" % layer_id)
            input_params = []
            for input_layer_id in self.layer_inputs[layer_id]:
                input_params.append(representation[input_layer_id])
                input_params.append(repre_lengths[input_layer_id])

            representation[layer_id], repre_lengths[layer_id] = self.layers[layer_id](*input_params)
            #logging.debug("Layer %s processed. output size: %s" % (layer_id, representation[layer_id].size()))

        # for support multi_output
        representation_output = dict()
        for single_output_layer_id in self.output_layer_id:
            representation_output[single_output_layer_id] = representation[single_output_layer_id]
        return representation_output

    def is_cuda(self):
        return list(self.parameters())[-1].data.is_cuda

    def update_use_gpu(self, new_use_gpu):
        self.use_gpu = new_use_gpu
        for layer_id in self.layers.keys():
            if isinstance(self.layers[layer_id], Embedding):
                for input_cluster in self.layers[layer_id].embeddings:
                    if isinstance(self.layers[layer_id].embeddings[input_cluster], CNNCharEmbedding):
                        self.layers[layer_id].embeddings[input_cluster].layer_conf.use_gpu = new_use_gpu
            elif isinstance(self.layers[layer_id], EncoderDecoder):
                self.layers[layer_id].encoder.layer_conf.use_gpu = new_use_gpu
                self.layers[layer_id].decoder.layer_conf.use_gpu = new_use_gpu
            else:
                self.layers[layer_id].layer_conf.use_gpu = new_use_gpu




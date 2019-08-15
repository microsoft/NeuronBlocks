# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import logging
import pickle as pkl
import json
import torch
import torch.nn as nn
import os
import shutil
import time
import tempfile
import subprocess
import hashlib
from .exceptions import ConfigurationError

def log_set(log_path, console_level='INFO', console_detailed=False, disable_log_file=False):
    """

    Args:
        log_path:
        console_level: 'INFO', 'DEBUG'

    Returns:

    """
    if not disable_log_file:
        logging.basicConfig(filename=log_path, filemode='w',
            format='%(asctime)s %(levelname)s %(filename)s %(funcName)s %(lineno)d: %(message)s',
            level=logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(getattr(logging, console_level.upper()))
    if console_detailed:
        console.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s %(filename)s %(funcName)s %(lineno)d: %(message)s'))
    else:
        console.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s %(message)s'))
    logging.getLogger().addHandler(console)


def load_from_pkl(pkl_path):
    with open(pkl_path, 'rb') as fin:
        obj = pkl.load(fin)
    logging.debug("%s loaded!" % pkl_path)
    return obj


def dump_to_pkl(obj, pkl_path):
    with open(pkl_path, 'wb') as fout:
        pkl.dump(obj, fout, protocol=pkl.HIGHEST_PROTOCOL)
    logging.debug("Obj dumped to %s!" % pkl_path)

def load_from_json(json_path):
    data = None
    with open(json_path, 'r', encoding='utf-8') as f:
        try:
            data = json.loads(f.read())
        except Exception as e:
            raise ConfigurationError("%s is not a legal JSON file, please check your JSON format!" % json_path)
    logging.debug("%s loaded!" % json_path)
    return data

def dump_to_json(obj, json_path):
    with open(json_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(obj))
    logging.debug("Obj dumped to %s!" % json_path)

def get_trainable_param_num(model):
    """ get the number of trainable parameters

    Args:
        model:

    Returns:

    """
    if isinstance(model, nn.DataParallel):
        if isinstance(model.module.layers['embedding'].embeddings, dict):
            model_param = list(model.parameters()) + list(model.module.layers['embedding'].get_parameters())
        else:
            model_param = list(model.parameters())
    else:
        if isinstance(model.layers['embedding'].embeddings, dict):
            model_param = list(model.parameters()) + list(model.layers['embedding'].get_parameters())
        else:
            model_param = list(model.parameters())

    return sum(p.numel() for p in model_param if p.requires_grad)


def get_param_num(model):
    """ get the number of parameters

    Args:
        model:

    Returns:

    """
    return sum(p.numel() for p in model.parameters())


def transfer_to_gpu(cpu_element):
    """

    Args:
        cpu_element: either a tensor or a module

    Returns:

    """
    return cpu_element.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


def transform_params2tensors(inputs, lengths):
    """ Because DataParallel only splits Tensor-like parameters, we have to transform dict parameter into tensors and keeps the information for forward().

    Args:
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
        param_list (list): records all the tensors in inputs and lengths
        inputs_desc (dict): the key records the information of inputs, the value indicate the index of a tensor in param_list
            e.g. {
                "string1_word": index in the param_list
                "string1_postag": index in the param_list
                ...
            }
        lengths_desc (dict): similar to inputs_desc
            e.g. {
                "string1": index in the param_list,
                "string2": index in the param_list
            }

    """
    param_list = []
    inputs_desc = {}
    cnt = 0
    for input in inputs:
        for input_type in inputs[input]:
            inputs_desc[input + '___' + input_type] = cnt
            param_list.append(inputs[input][input_type])
            cnt += 1

    length_desc = {}
    for length in lengths:
        if isinstance(lengths[length], dict):
            for length_type in lengths[length]:
                length_desc[length + '__' + length_type] = cnt
                param_list.append(lengths[length][length_type])
        else:
            length_desc[length] = cnt
            param_list.append(lengths[length])
        cnt += 1

    return param_list, inputs_desc, length_desc


def transform_tensors2params(inputs_desc, length_desc, param_list):
    """ Inverse function of transform_params2tensors

    Args:
        param_list:
        inputs_desc:
        length_desc:

    Returns:

    """
    inputs = {}
    for key in inputs_desc:
        input, input_type = key.split('___')
        if not input in inputs:
            inputs[input] = dict()

        inputs[input][input_type] = param_list[inputs_desc[key]]

    lengths = {}
    for key in length_desc:
        if '__' in key:
            input, input_type = key.split('__')
            if not input in lengths:
                lengths[input] = dict()
            lengths[input][input_type] = param_list[length_desc[key]]
        else:
            lengths[key] = param_list[length_desc[key]]

    return inputs, lengths


def prepare_dir(path, is_dir, allow_overwrite=False, clear_dir_if_exist=False, extra_info=None):
    """ to make dir if a dir or the parent dir of a file does not exist

    Args:
        path: can be a file path or a dir path.

    Returns:

    """
    if is_dir:
        if clear_dir_if_exist:
            allow_overwrite = True

        if not os.path.exists(path):
            os.makedirs(path)
        else:
            if not allow_overwrite:
                overwrite_option = input('The directory %s already exists, input "yes" to allow us to overwrite the directory contents and "no" to exit. (default:no): ' % path) \
                    if not extra_info else \
                    input('The directory %s already exists, %s, \ninput "yes" to allow us to operate and "no" to exit. (default:no): ' % (path, extra_info))
                if overwrite_option.lower() != 'yes':
                    exit(0)
            if (allow_overwrite or overwrite_option == 'yes') and clear_dir_if_exist:
                shutil.rmtree(path)
                logging.info('Clear dir %s...' % path)
                while os.path.exists(path):
                    time.sleep(0.3)
                os.makedirs(path)
    else:
        dir = os.path.dirname(path)
        if dir == '':       # when the path is only a file name, the dir would be empty and raise exception when making dir
            dir = '.'
        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            if os.path.exists(path) and allow_overwrite is False:
                overwrite_option = input('The file %s already exists, input "yes" to allow us to overwrite it or "no" to exit. (default:no): ' % path)
                if overwrite_option.lower() != 'yes':
                    exit(0)

def md5(file_paths, chunk_size=1024*1024*1024):
    """ Calculate a md5 of lists of files. 

    Args:
        file_paths:  an iterable object contains file paths. Files will be concatenated orderly if there are more than one file
        chunk_size:  unit is byte, default value is 1GB
    Returns:
        md5

    """
    md5 = hashlib.md5()
    for path in file_paths:
        with open(path, 'rb') as fin:
            while True:
                data = fin.read(chunk_size)
                if not data:
                    break
                md5.update(data)
    return md5.hexdigest()


def get_layer_class(model, layer_id):
    """get the layer class use layer_id

    Args:
        model: the model architecture, maybe nn.DataParallel type or model
        layer_id: layer id from configuration
    """
    if isinstance(model, nn.DataParallel):
        return model.module.layers[layer_id]
    else:
        return model.layers[layer_id]
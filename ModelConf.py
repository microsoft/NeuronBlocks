# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import codecs
import json
import os
import tempfile
import random
import string
import copy
import torch
import logging
import shutil

from losses.BaseLossConf import BaseLossConf
#import traceback
from settings import LanguageTypes, ProblemTypes, TaggingSchemes, SupportedMetrics, PredictionTypes, DefaultPredictionFields, ConstantStatic
from utils.common_utils import log_set, prepare_dir, md5, load_from_json, dump_to_json
from utils.exceptions import ConfigurationError
import numpy as np

class ConstantStaticItems(ConstantStatic):
    @staticmethod
    def concat_key_desc(key_prefix_desc, key):
        return key_prefix_desc + '.' + key

    @staticmethod
    def get_value_by_key(json, key, key_prefix='', use_default=False, default=None):
        """
        Args:
            json: a json object
            key: a key pointing to the value wanted to acquire
            use_default: if you really want to use default value when key can not be found in json object, set use_default=True
            default: if key is not found and default is None, we would raise an Exception, except that use_default is True
        Returns:
            value: 
        """
        try:
            value = json[key]
        except:
            if not use_default:
                raise ConfigurationError("key[%s] can not be found in configuration file" % (key_prefix + key))
            else:
                value = default
        return value

    @staticmethod
    def add_item(item_name, use_default=False, default=None):
        def add_item_loading_func(use_default, default, func_get_value_by_key):
            @classmethod
            def load_data(cls, obj, json, key_prefix_desc='', use_default=use_default, default=default, func_get_value_by_key=func_get_value_by_key):
                obj.__dict__[cls.__name__] = func_get_value_by_key(json, cls.__name__, key_prefix_desc, use_default, default)
                return obj
            return load_data
        return type(item_name, (ConstantStatic, ), dict(load_data=add_item_loading_func(use_default, default, __class__.get_value_by_key)))

    @classmethod
    def load_data(cls, obj, json, key_prefix_desc=''):
        if cls.__name__ in json.keys():
            json = json[cls.__name__]
        for key in cls.__dict__.keys():
            if not hasattr(cls.__dict__[key], 'load_data'):
                continue
            item = cls.__dict__[key]
            obj = item.load_data(obj, json, cls.concat_key_desc(key_prefix_desc, item.__name__))
        return obj

class ModelConf(object):
    def __init__(self, phase, conf_path, nb_version, params=None, mode='normal'):
        """ loading configuration from configuration file and argparse parameters

        Args:
            phase: train/test/predict/cache
                specially, 'cache' phase is used for verifying old cache
            conf_path:
            params:
            mode: 'normal', 'philly'
        """
        self.phase = phase
        assert self.phase in set(['train', 'test', 'predict', 'cache'])
        self.conf_path = conf_path
        self.params = params
        self.mode = mode.lower()
        assert self.mode in set(['normal', 'philly']), 'Your mode %s is illegal, supported modes are: normal and philly!'
        
        self.load_from_file(conf_path)

        self.check_version_compat(nb_version, self.tool_version)

        if phase != 'cache':
            self.check_conf()

        logging.debug('Print ModelConf below:')
        logging.debug('=' * 80)
        # print ModelConf
        for name, value in vars(self).items():
            if name.startswith("__") is False:
                logging.debug('%s: %s' % (str(name), str(value)))
        logging.debug('=' * 80)

    class Conf(ConstantStaticItems):
        license = ConstantStaticItems.add_item('license')
        tool_version = ConstantStaticItems.add_item('tool_version')
        model_description = ConstantStaticItems.add_item('model_description')
        language = ConstantStaticItems.add_item('language', use_default=True, default='english')

        class inputs(ConstantStaticItems):
            use_cache = ConstantStaticItems.add_item('use_cache', use_default=True, default=True)
            dataset_type =  ConstantStaticItems.add_item('dataset_type')
            tagging_scheme = ConstantStaticItems.add_item('tagging_scheme', use_default=True, default=None)

            class data_paths(ConstantStaticItems):
                train_data_path = ConstantStaticItems.add_item('train_data_path', use_default=True, default=None)
                valid_data_path = ConstantStaticItems.add_item('valid_data_path', use_default=True, default=None)
                test_data_path = ConstantStaticItems.add_item('test_data_path', use_default=True, default=None)
                predict_data_path = ConstantStaticItems.add_item('predict_data_path', use_default=True, default=None)
                pre_trained_emb = ConstantStaticItems.add_item('pre_trained_emb', use_default=True, default=None)
                pretrained_model_path = ConstantStaticItems.add_item('pretrained_model_path', use_default=True, default=None)

            file_with_col_header = ConstantStaticItems.add_item('file_with_col_header', use_default=True, default=False)
            pretrained_emb_type = ConstantStaticItems.add_item('pretrained_emb_type', use_default=True, default='glove')
            pretrained_emb_binary_or_text = ConstantStaticItems.add_item('pretrained_emb_binary_or_text', use_default=True, default='text')
            involve_all_words_in_pretrained_emb = ConstantStaticItems.add_item('involve_all_words_in_pretrained_emb', use_default=True, default=False)
            add_start_end_for_seq = ConstantStaticItems.add_item('add_start_end_for_seq', use_default=True, default=False)
            file_header = ConstantStaticItems.add_item('file_header', use_default=True, default=None)
            predict_file_header = ConstantStaticItems.add_item('predict_file_header', use_default=True, default=None)
            model_inputs = ConstantStaticItems.add_item('model_inputs')
            target = ConstantStaticItems.add_item('target', use_default=True, default=None)
            positive_label = ConstantStaticItems.add_item('positive_label', use_default=True, default=None)
        
        class outputs(ConstantStaticItems):
            save_base_dir = ConstantStaticItems.add_item('save_base_dir', use_default=True, default=None)
            model_name = ConstantStaticItems.add_item('model_name')
        
            train_log_name = ConstantStaticItems.add_item('train_log_name', use_default=True, default=None)
            test_log_name = ConstantStaticItems.add_item('test_log_name', use_default=True, default=None)
            predict_log_name = ConstantStaticItems.add_item('predict_log_name', use_default=True, default=None)
            predict_fields = ConstantStaticItems.add_item('predict_fields', use_default=True, default=None)
            predict_output_name = ConstantStaticItems.add_item('predict_output_name', use_default=True, default='predict.tsv')
            cache_dir = ConstantStaticItems.add_item('cache_dir', use_default=True, default=None)
        
        class training_params(ConstantStaticItems):
            class vocabulary(ConstantStaticItems):
                min_word_frequency = ConstantStaticItems.add_item('min_word_frequency', use_default=True, default=3)
                max_vocabulary = ConstantStaticItems.add_item('max_vocabulary', use_default=True, default=800 * 1000)
                max_building_lines = ConstantStaticItems.add_item('max_building_lines', use_default=True, default=1000 * 1000)
            
            optimizer = ConstantStaticItems.add_item('optimizer', use_default=True, default=None)
            clip_grad_norm_max_norm = ConstantStaticItems.add_item('clip_grad_norm_max_norm', use_default=True, default=-1)
            chunk_size = ConstantStaticItems.add_item('chunk_size', use_default=True, default=1000 * 1000)
            lr_decay = ConstantStaticItems.add_item('lr_decay', use_default=True, default=1)
            minimum_lr = ConstantStaticItems.add_item('minimum_lr', use_default=True, default=0)
            epoch_start_lr_decay = ConstantStaticItems.add_item('epoch_start_lr_decay', use_default=True, default=1)
            use_gpu = ConstantStaticItems.add_item('use_gpu', use_default=True, default=False)
            cpu_num_workers = ConstantStaticItems.add_item('cpu_num_workers', use_default=True, default=-1) #by default, use all workers cpu supports
            batch_size = ConstantStaticItems.add_item('batch_size', use_default=True, default=1)
            batch_num_to_show_results = ConstantStaticItems.add_item('batch_num_to_show_results', use_default=True, default=10)
            max_epoch = ConstantStaticItems.add_item('max_epoch', use_default=True, default=float('inf'))
            valid_times_per_epoch = ConstantStaticItems.add_item('valid_times_per_epoch', use_default=True, default=None)
            steps_per_validation = ConstantStaticItems.add_item('steps_per_validation', use_default=True, default=10)
            text_preprocessing = ConstantStaticItems.add_item('text_preprocessing', use_default=True, default=list())
            max_lengths = ConstantStaticItems.add_item('max_lengths', use_default=True, default=None)
            fixed_lengths = ConstantStaticItems.add_item('fixed_lengths', use_default=True, default=None)
            tokenizer = ConstantStaticItems.add_item('tokenizer', use_default=True, default=None)

        architecture = ConstantStaticItems.add_item('architecture')
        loss = ConstantStaticItems.add_item('loss', use_default=True, default=None)
        metrics = ConstantStaticItems.add_item('metrics', use_default=True, default=None)

    def raise_configuration_error(self, key):
        raise ConfigurationError(
            "The configuration file %s is illegal. the item [%s] is not found." % (self.conf_path,  key))

    def load_from_file(self, conf_path):
        # load file
        self.conf = load_from_json(conf_path, debug=False)
        self = self.Conf.load_data(self, {'Conf' : self.conf}, key_prefix_desc='Conf')
        self.language = self.language.lower()
        self.configurate_outputs()
        self.configurate_inputs()
        self.configurate_training_params()
        self.configurate_architecture()
        self.configurate_loss()
        self.configurate_cache()

    def configurate_outputs(self):
        def configurate_logger(self):
            if self.phase == 'cache':
                return

            # dir
            if hasattr(self.params, 'log_dir') and self.params.log_dir:
                self.log_dir = self.params.log_dir
                prepare_dir(self.log_dir, True, allow_overwrite=True)
            else:
                self.log_dir = self.save_base_dir
            
            # path
            self.train_log_path = os.path.join(self.log_dir, self.train_log_name)
            self.test_log_path = os.path.join(self.log_dir, self.test_log_name)
            self.predict_log_path = os.path.join(self.log_dir, self.predict_log_name)
            if self.phase == 'train':
                log_path = self.train_log_path
            elif self.phase == 'test':
                log_path = self.test_log_path
            elif self.phase == 'predict':
                log_path =  self.predict_log_path
            if log_path is None:
                self.raise_configuration_error(self.phase + '_log_name')

            # log level
            if self.mode == 'philly' or self.params.debug:
                log_set(log_path, console_level='DEBUG', console_detailed=True, disable_log_file=self.params.disable_log_file)
            else:
                log_set(log_path, disable_log_file=self.params.disable_log_file)

        # save base dir
        if hasattr(self.params, 'model_save_dir') and self.params.model_save_dir:
            self.save_base_dir = self.params.model_save_dir
        elif self.save_base_dir is None:
            self.raise_configuration_error('save_base_dir')

        # prepare save base dir 
        if self.phase != 'cache':
            prepare_dir(self.save_base_dir, True, allow_overwrite=self.params.force or self.mode == 'philly',
                        extra_info='will overwrite model file and train.log' if self.phase=='train' else 'will add %s.log and predict file'%self.phase)

        # logger
        configurate_logger(self)

        # predict output path
        if self.phase != 'cache':
            if self.params.predict_output_path:
                self.predict_output_path = self.params.predict_output_path
            else:
                self.predict_output_path = os.path.join(self.save_base_dir, self.predict_output_name)
            logging.debug('Prepare dir for: %s' % self.predict_output_path)
            prepare_dir(self.predict_output_path, False, allow_overwrite=self.params.force or self.mode == 'philly')

        if self.predict_fields is None:
            self.predict_fields = DefaultPredictionFields[ProblemTypes[self.problem_type]]

        self.model_save_path = os.path.join(self.save_base_dir, self.model_name)

    def configurate_inputs(self):

        def configurate_data_path(self):
            self.pretrained_emb_path =self.pre_trained_emb

            if self.mode != "normal":
                self.train_data_path = None
                self.valid_data_path = None
                self.test_data_path = None
                self.predict_data_path = None
                self.pretrained_emb_path = None

            if hasattr(self.params, 'train_data_path') and self.params.train_data_path:
                self.train_data_path = self.params.train_data_path
            if hasattr(self.params, 'valid_data_path') and self.params.valid_data_path:
                self.valid_data_path = self.params.valid_data_path
            if hasattr(self.params, 'test_data_path') and self.params.test_data_path:
                self.test_data_path = self.params.test_data_path
            if hasattr(self.params, 'predict_data_path') and self.params.predict_data_path:
                self.predict_data_path = self.params.predict_data_path
            if hasattr(self.params, 'pretrained_emb_path') and self.params.pretrained_emb_path:
                self.pretrained_emb_path = self.params.pretrained_emb_path

            if self.phase == 'train' or self.phase == 'cache':
                if self.valid_data_path is None and self.test_data_path is not None:
                    # We support test_data_path == None, if someone set valid_data_path to None while test_data_path is not None,
                    # swap the valid_data_path and test_data_path
                    self.valid_data_path = self.test_data_path
                    self.test_data_path = None
            elif self.phase == 'predict':
                if self.predict_data_path is None and self.test_data_path is not None:
                    self.predict_data_path = self.test_data_path
                    self.test_data_path = None
            
            return self

        def configurate_data_format(self):
            # file columns
            if self.phase == 'train' or self.phase == 'test' or self.phase == 'cache':
                self.file_columns = self.file_header
                if self.file_columns is None:
                    self.raise_configuration_error('file_columns')
            if self.phase == 'predict':
                if self.file_columns is None and self.predict_file_columns is None:
                    self.raise_configuration_error('predict_file_columns')
                if self.file_columns and self.predict_file_columns is None:
                    self.predict_file_columns = self.file_columns

            # target
            if self.phase != 'predict':
                self.answer_column_name = self.target
                if self.target is None and self.phase != 'cache':
                    self.raise_configuration_error('target')

            if ProblemTypes[self.problem_type] == ProblemTypes.sequence_tagging and self.add_start_end_for_seq is None:
                self.add_start_end_for_seq = True

            # pretrained embedding
            if 'word' in self.architecture[0]['conf'] and self.pretrained_emb_path:
                if hasattr(self.params, 'involve_all_words_in_pretrained_emb') and self.params.involve_all_words_in_pretrained_emb:
                    self.involve_all_words_in_pretrained_emb = self.params.involve_all_words_in_pretrained_emb
                if hasattr(self.params, 'pretrained_emb_type') and self.params.pretrained_emb_type:
                    self.pretrained_emb_type = self.params.pretrained_emb_type
                if hasattr(self.params, 'pretrained_emb_binary_or_text') and self.params.pretrained_emb_binary_or_text:
                    self.pretrained_emb_binary_or_text = self.params.pretrained_emb_binary_or_text
                self.pretrained_emb_dim = self.architecture[0]['conf']['word']['dim']
            else:
                self.pretrained_emb_path = None
                self.involve_all_words_in_pretrained_emb = None
                self.pretrained_emb_type = None
                self.pretrained_emb_binary_or_text = None
                self.pretrained_emb_dim = None
            
            return self

        def configurate_model_input(self):
            self.object_inputs = self.model_inputs
            self.object_inputs_names = [name for name in self.object_inputs]

            return self

        self.problem_type = self.dataset_type.lower()

        # previous model path
        if hasattr(self.params, 'previous_model_path') and self.params.previous_model_path:
            self.previous_model_path = self.params.previous_model_path
        else:
            self.previous_model_path = os.path.join(self.save_base_dir, self.model_name)

        # pretrained model path
        if hasattr(self.params, 'pretrained_model_path') and self.params.pretrained_model_path:
            self.pretrained_model_path = self.params.pretrained_model_path

        # saved problem path
        model_path = None
        if self.phase == 'train':
            model_path = self.pretrained_model_path
        elif self.phase == 'test' or self.phase == 'predict':
            model_path = self.previous_model_path
        if model_path:
            model_path_dir = os.path.dirname(model_path)
            self.saved_problem_path = os.path.join(model_path_dir, '.necessary_cache', 'problem.pkl')
            if not os.path.isfile(self.saved_problem_path):
                self.saved_problem_path = os.path.join(model_path_dir, 'necessary_cache', 'problem.pkl')
            if not (os.path.isfile(model_path) and os.path.isfile(self.saved_problem_path)):
                raise Exception('Previous trained model %s or its dictionaries %s does not exist!' % (model_path, self.saved_problem_path))

        configurate_data_path(self)
        configurate_data_format(self)
        configurate_model_input(self) 

    def configurate_training_params(self):
        # optimizer
        if self.phase == 'train':
            if self.optimizer is None:
                self.raise_configuration_error('training_params.optimizer')
            if 'name' not in self.optimizer.keys():
                self.raise_configuration_error('training_params.optimizer.name')
            self.optimizer_name = self.optimizer['name']
            if 'params' not in self.optimizer.keys():
                self.raise_configuration_error('training_params.optimizer.params')
            self.optimizer_params = self.optimizer['params']
            if hasattr(self.params, 'learning_rate') and self.params.learning_rate:
                self.optimizer_params['lr'] = self.params.learning_rate
        
        # batch size
        self.batch_size_each_gpu = self.batch_size # the batch_size in conf file is the batch_size on each GPU
        if hasattr(self.params, 'batch_size') and self.params.batch_size:
            self.batch_size_each_gpu = self.params.batch_size
        if self.batch_size_each_gpu is None:
            self.raise_configuration_error('training_params.batch_size')
        self.batch_size_total = self.batch_size_each_gpu
        if torch.cuda.device_count() > 1:
            self.batch_size_total = torch.cuda.device_count() * self.batch_size_each_gpu
            self.batch_num_to_show_results = self.batch_num_to_show_results // torch.cuda.device_count()

    
        if hasattr(self.params, 'max_epoch') and self.params.max_epoch:
            self.max_epoch = self.params.max_epoch
        
        if self.valid_times_per_epoch is not None:
            logging.info("configuration[training_params][valid_times_per_epoch] is deprecated, please use configuration[training_params][steps_per_validation] instead")
        
        # sequence length
        if self.fixed_lengths:
            self.max_lengths = None
        if ProblemTypes[self.problem_type] == ProblemTypes.sequence_tagging:
            self.fixed_lengths = None
            self.max_lengths = None

        # text preprocessing
        self.__text_preprocessing = self.text_preprocessing
        self.DBC2SBC = True if 'DBC2SBC' in self.__text_preprocessing else False
        self.unicode_fix = True if 'unicode_fix' in self.__text_preprocessing else False
        self.remove_stopwords = True if 'remove_stopwords' in self.__text_preprocessing else False

        # tokenzier
        if self.tokenizer is None:
            self.tokenizer = 'jieba' if self.language == 'chinese' else 'nltk'
        
        # GPU/CPU
        if self.phase != 'cache':
            if torch.cuda.is_available() and torch.cuda.device_count() > 0 and self.use_gpu:
                logging.info("Activating GPU mode, there are %d GPUs available" % torch.cuda.device_count())
            else:
                self.use_gpu = False
                logging.info("Activating CPU mode")

    def configurate_architecture(self):
        self.input_types = self.architecture[0]['conf']
        
        # extra feature
        feature_all = set([_.lower() for _ in self.input_types.keys()])
        formal_feature = set(['word', 'char'])
        extra_feature_num = feature_all - formal_feature
        self.extra_feature = len(extra_feature_num) != 0
        if self.extra_feature:
            if self.DBC2SBC:
                logging.warning("Detect the extra feature %s, set the DBC2sbc is False." % ''.join(list(extra_feature_num)))
            if self.unicode_fix:
                logging.warning("Detect the extra feature %s, set the unicode_fix is False." % ''.join(list(extra_feature_num)))
            if self.remove_stopwords:
                logging.warning("Detect the extra feature %s, set the remove_stopwords is False." % ''.join(list(extra_feature_num)))

        # output layer
        self.output_layer_id = []
        for single_layer in self.architecture:
            if 'output_layer_flag' in single_layer and single_layer['output_layer_flag']:
                self.output_layer_id.append(single_layer['layer_id'])

        # check CNN layer & change min sentence length
        cnn_rele_layers = ['Conv', 'ConvPooling']
        self.min_sentence_len = 0
        for layer_index, single_layer in enumerate(self.architecture):
            if layer_index == 0:
                continue
            if sum([_ == single_layer['layer'] for _ in cnn_rele_layers]):
                # get window_size conf: type maybe int or list
                for single_conf, single_conf_value in single_layer['conf'].items():
                    if 'window' in single_conf.lower():
                        self.min_sentence_len = max(self.min_sentence_len, np.max(np.array([single_conf_value])))
                        break

    def configurate_loss(self):
        if self.phase != 'train' and self.phase != 'test':
            return
        
        if self.loss is None or self.metrics is None:
            self.raise_configuration_error('loss/metrics')
        self.loss = BaseLossConf.get_conf(**self.loss)

        if 'auc' in self.metrics and ProblemTypes[self.problem_type] == ProblemTypes.classification:
            self.pos_label = self.positive_label

    def configurate_cache(self):
        # whether use cache
        if self.mode == 'philly':
            self.use_cache = True

        # cache dir
        if self.phase == 'train':
            if hasattr(self.params, 'cache_dir') and self.params.cache_dir:
                self.cache_dir = self.params.cache_dir
            else:
                if self.mode == 'normal':
                    if self.use_cache is False:
                        self.cache_dir = os.path.join(tempfile.gettempdir(), 'neuron_blocks', ''.join(random.sample(string.ascii_letters+string.digits, 16)))
                else:
                    # for philly mode, we can only save files in model_path or scratch_path
                    self.cache_dir = os.path.join(self.save_base_dir, 'cache')

            self.problem_path = os.path.join(self.cache_dir, 'problem.pkl')
            if self.pretrained_emb_path is not None:
                self.emb_pkl_path = os.path.join(self.cache_dir, 'emb.pkl')
            else:
                self.emb_pkl_path = None
        else:
            tmp_problem_path = os.path.join(self.save_base_dir, '.necessary_cache', 'problem.pkl')
            self.problem_path = tmp_problem_path if os.path.isfile(tmp_problem_path) else os.path.join(self.save_base_dir, 'necessary_cache', 'problem.pkl')
        
        # md5 of training data and problem
        self.train_data_md5 = None
        if self.phase == 'train' and self.train_data_path:
            logging.info("Calculating the md5 of traing data ...")
            self.train_data_md5 = md5([self.train_data_path])
            logging.info("the md5 of traing data is %s"%(self.train_data_md5))
        self.problem_md5 = None

        # encoding 
        self.encoding_cache_dir = None
        self.encoding_cache_index_file_path = None
        self.encoding_cache_index_file_md5_path = None
        self.encoding_file_index = None
        self.encoding_cache_legal_line_cnt = 0
        self.encoding_cache_illegal_line_cnt = 0
        self.load_encoding_cache_generator = None

    def check_conf(self):
        """ verify if the configuration is legal or not

        Returns:

        """
        # In philly mode, ensure the data and model etc. are not the local paths defined in configuration file.
        if self.mode == 'philly':
            assert not (hasattr(self.params, 'train_data_path') and self.params.train_data_path is None and hasattr(self, 'train_data_path') and self.train_data_path), 'In philly mode, but you define a local train_data_path:%s in your configuration file' % self.train_data_path
            assert not (hasattr(self.params, 'valid_data_path') and self.params.valid_data_path is None and hasattr(self, 'valid_data_path') and self.valid_data_path), 'In philly mode, but you define a local valid_data_path:%s in your configuration file' % self.valid_data_path
            assert not (hasattr(self.params, 'test_data_path') and self.params.test_data_path is None and hasattr(self, 'test_data_path') and self.test_data_path), 'In philly mode, but you define a local test_data_path:%s in your configuration file' % self.test_data_path
            if self.phase == 'train':
                assert hasattr(self.params, 'model_save_dir') and self.params.model_save_dir, 'In philly mode, you must define a model save dir through the training params'
                assert not (self.params.pretrained_model_path is None and self.pretrained_model_path), 'In philly mode, but you define a local pretrained model path:%s in your configuration file' % self.pretrained_model_path
                assert not (self.pretrained_model_path is None and self.params.pretrained_emb_path is None and self.pretrained_emb_path), 'In philly mode, but you define a local pretrained embedding:%s in your configuration file' % self.pretrained_emb_path
            elif self.phase == 'test' or self.phase == 'predict':
                assert not (self.params.previous_model_path is None and self.previous_model_path), 'In philly mode, but you define a local model trained previously %s in your configuration file' % self.previous_model_path

        # check inputs
        # it seems that os.path.isfile cannot detect hdfs files
        if self.phase == 'train':
            assert self.train_data_path is not None, "Please define train_data_path"
            assert os.path.isfile(self.train_data_path), "Training data %s does not exist!" % self.train_data_path
            assert self.valid_data_path is not None, "Please define valid_data_path"
            assert os.path.isfile(self.valid_data_path), "Training data %s does not exist!" % self.valid_data_path

            if hasattr(self, 'pretrained_emb_type') and self.pretrained_emb_type:
                assert self.pretrained_emb_type in set(['glove', 'word2vec', 'fasttext']), 'Embedding type %s is not supported! We support glove, word2vec, fasttext now.'

            if hasattr(self, 'pretrained_emb_binary_or_text') and self.pretrained_emb_binary_or_text:
                assert self.pretrained_emb_binary_or_text in set(['text', 'binary']), 'Embedding file type %s is not supported! We support text and binary.'


        elif self.phase == 'test':
            assert self.test_data_path is not None, "Please define test_data_path"
            assert os.path.isfile(self.test_data_path), "Training data %s does not exist!" % self.test_data_path
        elif self.phase == 'predict':
            assert self.predict_data_path is not None, "Please define predict_data_path"
            assert os.path.isfile(self.predict_data_path), "Training data %s does not exist!" % self.predict_data_path

        # check language types
        SUPPORTED_LANGUAGES = set(LanguageTypes._member_names_)
        assert self.language in SUPPORTED_LANGUAGES, "Language type %s is not supported now. Supported types: %s" % (self.language, ",".join(SUPPORTED_LANGUAGES))

        # check problem types
        SUPPORTED_PROBLEMS = set(ProblemTypes._member_names_)
        assert self.problem_type in SUPPORTED_PROBLEMS, "Data type %s is not supported now. Supported types: %s" % (self.problem_type, ",".join(SUPPORTED_PROBLEMS))

        if ProblemTypes[self.problem_type] == ProblemTypes.sequence_tagging:
            SUPPORTED_TAGGING_SCHEMES = set(TaggingSchemes._member_names_)
            assert self.tagging_scheme is not None, "For sequence tagging proble, tagging scheme must be defined at configuration[\'inputs\'][\'tagging_scheme\']!"
            assert self.tagging_scheme in SUPPORTED_TAGGING_SCHEMES, "Tagging scheme %s is not supported now. Supported schemes: %s" % (self.tagging_scheme, ",".join(SUPPORTED_TAGGING_SCHEMES))

            # the max_lengths of all the inputs and targets should be consistent
            if self.max_lengths:
                max_lengths = list(self.max_lengths.values())
                for i in range(len(max_lengths) - 1):
                    assert max_lengths[i] == max_lengths[i + 1], "For sequence tagging tasks, the max_lengths of all the inputs and targets should be consistent!"

        # check appliable metrics
        if self.phase == 'train' or self.phase == 'test':
            self.metrics_post_check = set() # saved to check later
            diff = set(self.metrics) - SupportedMetrics[ProblemTypes[self.problem_type]]
            illegal_metrics = []
            for diff_metric in diff:
                if diff_metric.find('@') != -1:
                    field, target = diff_metric.split('@')
                    #if not field in PredictionTypes[ProblemTypes[self.problem_type]]:
                    if field != 'auc':
                        illegal_metrics.append(diff_metric)
                    else:
                        if target != 'average':
                            self.metrics_post_check.add(diff_metric)
            if len(illegal_metrics) > 0:
                raise Exception("Metrics %s are not supported for %s tasks!" % (",".join(list(illegal_metrics)), self.problem_type))

        # check predict fields
        if self.phase == 'predict':
            self.predict_fields_post_check = set() # saved to check later
            diff = set(self.predict_fields) - PredictionTypes[ProblemTypes[self.problem_type]]
            illegal_fields = []
            for diff_field in diff:
                if diff_field.find('@') != -1 and diff_field.startswith('confidence'):
                    field, target = diff_field.split('@')
                    #if not field in PredictionTypes[ProblemTypes[self.problem_type]]:
                    if field != 'confidence':
                        illegal_fields.append(diff_field)
                    else:
                        # don't know if the target exists in the output dictionary, check after problem loaded
                        self.predict_fields_post_check.add(diff_field)
                else:
                    illegal_fields.append(diff_field)
            if len(illegal_fields) > 0:
                raise Exception("The prediction fields %s is/are not supported!" % ",".join(illegal_fields))

    def check_version_compat(self, nb_version, conf_version):
        """ check if the version of toolkit and configuration file is compatible

        Args:
            nb_version: x.y.z
            conf_version: x.y.z

        Returns:
            If the x field and y field are both the same, return True, else return False

        """
        nb_version_split = nb_version.split('.')
        conf_version_split = conf_version.split('.')
        if len(nb_version_split) != len(conf_version_split):
            raise ConfigurationError('The tool_version field of your configuration is illegal!')
        if not (nb_version_split[0] == conf_version_split[0] and nb_version_split[1] == conf_version_split[1]):
            raise ConfigurationError('The NeuronBlocks version is %s, but the configuration version is %s, please update your configuration to %s.%s.X' % (nb_version, conf_version, nb_version_split[0], nb_version_split[1]))

    def back_up(self, params):
        shutil.copy(params.conf_path, self.save_base_dir)
        logging.info('Configuration file is backed up to %s' % (self.save_base_dir))

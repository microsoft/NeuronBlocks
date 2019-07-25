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
from settings import LanguageTypes, ProblemTypes, TaggingSchemes, SupportedMetrics, PredictionTypes, DefaultPredictionFields
from utils.common_utils import log_set, prepare_dir, md5
from utils.exceptions import ConfigurationError
import numpy as np

class ModelConf(object):
    def __init__(self, phase, conf_path, nb_version, params=None, mode='normal', online_encoder=False):
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
        self.online_encoder = online_encoder
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

    def load_from_file(self, conf_path):
        with codecs.open(conf_path, 'r', encoding='utf-8') as fin:
            try:
                self.conf = json.load(fin)
            except Exception as e:
                raise ConfigurationError("%s is not a legal JSON file, please check your JSON format!" % conf_path)

        self.tool_version = self.get_item(['tool_version'])
        self.language = self.get_item(['language'], default='english').lower()
        self.problem_type = self.get_item(['inputs', 'dataset_type']).lower()
        #if ProblemTypes[self.problem_type] == ProblemTypes.sequence_tagging:
        self.tagging_scheme = self.get_item(['inputs', 'tagging_scheme'], default=None, use_default=True)

        if self.mode == 'normal':
            self.use_cache = self.get_item(['inputs', 'use_cache'], True)
        elif self.mode == 'philly':
            self.use_cache = True

        # OUTPUTS

        # for encoder setting
        self.encoder = self.get_item(['outputs', 'encoder'], default=None, use_default=True)

        if hasattr(self.params, 'model_save_dir') and self.params.model_save_dir:
            self.save_base_dir = self.params.model_save_dir
        else:
            self.save_base_dir = self.get_item(['outputs', 'save_base_dir'])

        if self.phase == 'train':
            # in train.py, it is called pretrained_model_path
            if hasattr(self.params, 'pretrained_model_path') and self.params.pretrained_model_path:
                self.pretrained_model_path = self.previous_model_path = self.params.pretrained_model_path
            else:
                self.pretrained_model_path = self.previous_model_path = self.get_item(['inputs', 'data_paths', 'pretrained_model_path'], default=None, use_default=True)
        elif self.phase == 'test' or self.phase == 'predict':
            # in test.py and predict.py, it is called pretrained_model_path
            if hasattr(self.params, 'previous_model_path') and self.params.previous_model_path:
                self.previous_model_path = self.pretrained_model_path = self.params.previous_model_path
            else:
                self.previous_model_path = self.pretrained_model_path = os.path.join(self.save_base_dir, self.get_item(['outputs', 'model_name'])) # namely, the model_save_path

        if hasattr(self, 'pretrained_model_path') and self.pretrained_model_path:  # namely self.previous_model_path
            tmp_saved_problem_path = os.path.join(os.path.dirname(self.pretrained_model_path), '.necessary_cache', 'problem.pkl')
            self.saved_problem_path = tmp_saved_problem_path if os.path.isfile(tmp_saved_problem_path) \
                else os.path.join(os.path.dirname(self.pretrained_model_path), 'necessary_cache', 'problem.pkl')
            if not (os.path.isfile(self.pretrained_model_path) and os.path.isfile(self.saved_problem_path)):
                raise Exception('Previous trained model %s or its dictionaries %s does not exist!' % (self.pretrained_model_path, self.saved_problem_path))

        if self.phase != 'cache':
            prepare_dir(self.save_base_dir, True, allow_overwrite=self.params.force or self.mode == 'philly',
                        extra_info='will overwrite model file and train.log' if self.phase=='train' else 'will add %s.log and predict file'%self.phase)

        if hasattr(self.params, 'log_dir') and self.params.log_dir:
            self.log_dir = self.params.log_dir
            if self.phase != 'cache':
                prepare_dir(self.log_dir, True, allow_overwrite=True)
        else:
            self.log_dir = self.save_base_dir

        if self.phase == 'train':
            self.train_log_path = os.path.join(self.log_dir, self.get_item(['outputs', 'train_log_name']))
            if self.mode == 'philly' or self.params.debug:
                log_set(self.train_log_path, console_level='DEBUG', console_detailed=True, disable_log_file=self.params.disable_log_file)
            else:
                log_set(self.train_log_path, disable_log_file=self.params.disable_log_file)
        elif self.phase == 'test':
            self.test_log_path = os.path.join(self.log_dir, self.get_item(['outputs', 'test_log_name']))
            if self.mode == 'philly' or self.params.debug:
                log_set(self.test_log_path, console_level='DEBUG', console_detailed=True, disable_log_file=self.params.disable_log_file)
            else:
                log_set(self.test_log_path, disable_log_file=self.params.disable_log_file)
        elif self.phase == 'predict':
            self.predict_log_path = os.path.join(self.log_dir, self.get_item(['outputs', 'predict_log_name']))
            if self.mode == 'philly' or self.params.debug:
                log_set(self.predict_log_path, console_level='DEBUG', console_detailed=True, disable_log_file=self.params.disable_log_file)
            else:
                log_set(self.predict_log_path, disable_log_file=self.params.disable_log_file)
        if self.phase != 'cache':
            self.predict_output_path = self.params.predict_output_path if self.params.predict_output_path else os.path.join(self.save_base_dir, self.get_item(['outputs', 'predict_output_name'], default='predict.tsv'))
            logging.debug('Prepare dir for: %s' % self.predict_output_path)
            prepare_dir(self.predict_output_path, False, allow_overwrite=self.params.force or self.mode == 'philly')
        self.predict_fields = self.get_item(['outputs', 'predict_fields'], default=DefaultPredictionFields[ProblemTypes[self.problem_type]])

        self.model_save_path = os.path.join(self.save_base_dir, self.get_item(['outputs', 'model_name']))

        # INPUTS
        if hasattr(self.params, 'train_data_path') and self.params.train_data_path:
            self.train_data_path = self.params.train_data_path
        else:
            if self.mode == 'normal':
                self.train_data_path = self.get_item(['inputs', 'data_paths', 'train_data_path'], default=None, use_default=True)
            else:
                self.train_data_path = None
        if hasattr(self.params, 'valid_data_path') and self.params.valid_data_path:
            self.valid_data_path = self.params.valid_data_path
        else:
            if self.mode == 'normal':
                self.valid_data_path = self.get_item(['inputs', 'data_paths', 'valid_data_path'], default=None, use_default=True)
            else:
                self.valid_data_path = None
        if hasattr(self.params, 'test_data_path') and self.params.test_data_path:
            self.test_data_path = self.params.test_data_path
        else:
            if self.mode == 'normal':
                self.test_data_path = self.get_item(['inputs', 'data_paths', 'test_data_path'], default=None, use_default=True)
            else:
                self.test_data_path = None

        if self.phase == 'predict':
            if self.params.predict_data_path:
                self.predict_data_path = self.params.predict_data_path
            else:
                if self.mode == 'normal':
                    self.predict_data_path = self.get_item(['inputs', 'data_paths', 'predict_data_path'], default=None, use_default=True)
                else:
                    self.predict_data_path = None

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

        if self.phase == 'train' or self.phase == 'test' or self.phase == 'cache':
            self.file_columns = self.get_item(['inputs', 'file_header'])
        else:
            self.file_columns = self.get_item(['inputs', 'file_header'], default=None, use_default=True)

        if self.phase == 'predict':
            if self.file_columns is None:
                self.predict_file_columns = self.get_item(['inputs', 'predict_file_header'])
            else:
                self.predict_file_columns = self.get_item(['inputs', 'predict_file_header'], default=None, use_default=True)
                if self.predict_file_columns is None:
                    self.predict_file_columns = self.file_columns

        if self.phase != 'predict':
            if self.phase == 'cache':
                self.answer_column_name = self.get_item(['inputs', 'target'], default=None, use_default=True)
            else:
                self.answer_column_name = self.get_item(['inputs', 'target'])
        self.input_types = self.get_item(['architecture', 0, 'conf'])
        # add extra feature
        feature_all = set([_.lower() for _ in self.input_types.keys()])
        formal_feature = set(['word', 'char'])
        self.extra_feature = len(feature_all - formal_feature) != 0

        # add char embedding config
        # char_emb_type = None
        # char_emb_type_cols = None
        # for single_type in self.input_types:
        #     if single_type.lower() == 'char':
        #         char_emb_type = single_type
        #         char_emb_type_cols = [single_col.lower() for single_col in self.input_types[single_type]['cols']]
        #         break
        self.object_inputs = self.get_item(['inputs', 'model_inputs'])
        # if char_emb_type and char_emb_type_cols:
        #     for single_input in self.object_inputs:
        #         for single_col in char_emb_type_cols:
        #             if single_input.lower() in single_col:
        #                 self.object_inputs[single_input].append(single_col)

        self.object_inputs_names = [name for name in self.object_inputs]

        # vocabulary setting
        self.max_vocabulary = self.get_item(['training_params', 'vocabulary', 'max_vocabulary'], default=800000, use_default=True)
        self.min_word_frequency = self.get_item(['training_params', 'vocabulary', 'min_word_frequency'], default=3, use_default=True)

        # file column header setting
        self.file_with_col_header = self.get_item(['inputs', 'file_with_col_header'], default=False, use_default=True)

        if ProblemTypes[self.problem_type] == ProblemTypes.sequence_tagging:
            self.add_start_end_for_seq = self.get_item(['inputs', 'add_start_end_for_seq'], default=True)
        else:
            self.add_start_end_for_seq = self.get_item(['inputs', 'add_start_end_for_seq'], default=False)

        if hasattr(self.params, 'pretrained_emb_path') and self.params.pretrained_emb_path:
            self.pretrained_emb_path = self.params.pretrained_emb_path
        else:
            if self.mode == 'normal':
                self.pretrained_emb_path = self.get_item(['inputs', 'data_paths', 'pre_trained_emb'], default=None, use_default=True)
            else:
                self.pretrained_emb_path = None

        if 'word' in self.get_item(['architecture', 0, 'conf']) and self.pretrained_emb_path:
            if hasattr(self.params, 'involve_all_words_in_pretrained_emb') and self.params.involve_all_words_in_pretrained_emb:
                self.involve_all_words_in_pretrained_emb = self.params.involve_all_words_in_pretrained_emb
            else:
                self.involve_all_words_in_pretrained_emb = self.get_item(['inputs', 'involve_all_words_in_pretrained_emb'], default=False)
            if hasattr(self.params, 'pretrained_emb_type') and self.params.pretrained_emb_type:
                self.pretrained_emb_type = self.params.pretrained_emb_type
            else:
                self.pretrained_emb_type = self.get_item(['inputs', 'pretrained_emb_type'], default='glove')
            if hasattr(self.params, 'pretrained_emb_binary_or_text') and self.params.pretrained_emb_binary_or_text:
                self.pretrained_emb_binary_or_text = self.params.pretrained_emb_binary_or_text
            else:
                self.pretrained_emb_binary_or_text = self.get_item(['inputs', 'pretrained_emb_binary_or_text'], default='text')
            self.pretrained_emb_dim = self.get_item(['architecture', 0, 'conf', 'word', 'dim'])
        else:
            self.pretrained_emb_path = None
            self.involve_all_words_in_pretrained_emb = None
            self.pretrained_emb_binary_or_text = None
            self.pretrained_emb_dim = None
            self.pretrained_emb_type = None

        if self.phase == 'train':
            if hasattr(self.params, 'cache_dir') and self.params.cache_dir:
                # for aether
                self.cache_dir = self.params.cache_dir
            else:
                if self.mode == 'normal':
                    if self.use_cache:
                        self.cache_dir = self.get_item(['outputs', 'cache_dir'])
                    else:
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

        # cache configuration
        self._load_cache_config_from_conf()

        # training params
        self.training_params = self.get_item(['training_params'])

        if self.phase == 'train':
            self.optimizer_name = self.get_item(['training_params', 'optimizer', 'name'])
            self.optimizer_params = self.get_item(['training_params', 'optimizer', 'params'])

            self.clip_grad_norm_max_norm = self.get_item(['training_params', 'clip_grad_norm_max_norm'], default=-1)

            if hasattr(self.params, 'learning_rate') and self.params.learning_rate:
                self.optimizer_params['lr'] = self.params.learning_rate

        if hasattr(self.params, 'batch_size') and self.params.batch_size:
            self.batch_size_each_gpu = self.params.batch_size
        else:
            self.batch_size_each_gpu = self.get_item(['training_params', 'batch_size'])     #the batch_size in conf file is the batch_size on each GPU
        self.lr_decay = self.get_item(['training_params', 'lr_decay'], default=1)   # by default, no decay
        self.minimum_lr = self.get_item(['training_params', 'minimum_lr'], default=0)
        self.epoch_start_lr_decay = self.get_item(['training_params', 'epoch_start_lr_decay'], default=1)
        if hasattr(self.params, 'max_epoch') and self.params.max_epoch:
            self.max_epoch = self.params.max_epoch
        else:
            self.max_epoch = self.get_item(['training_params', 'max_epoch'], default=float('inf'))
        if 'valid_times_per_epoch' in self.conf['training_params']:
            logging.info("configuration[training_params][valid_times_per_epoch] is deprecated, please use configuration[training_params][steps_per_validation] instead")
        self.steps_per_validation = self.get_item(['training_params', 'steps_per_validation'], default=10)
        self.batch_num_to_show_results = self.get_item(['training_params', 'batch_num_to_show_results'], default=10)
        self.max_lengths = self.get_item(['training_params', 'max_lengths'], default=None, use_default=True)
        self.fixed_lengths = self.get_item(['training_params', 'fixed_lengths'], default=None, use_default=True)
        if self.fixed_lengths:
            self.max_lengths = None
        if ProblemTypes[self.problem_type] == ProblemTypes.sequence_tagging:
            self.fixed_lengths = None
            self.max_lengths = None

        if torch.cuda.device_count() > 1:
            self.batch_size_total = torch.cuda.device_count() * self.training_params['batch_size']
            self.batch_num_to_show_results = self.batch_num_to_show_results // torch.cuda.device_count()
        else:
            self.batch_size_total = self.batch_size_each_gpu

        self.cpu_num_workers = self.get_item(['training_params', 'cpu_num_workers'], default=-1)  #by default, use all workers cpu supports

        # text preprocessing
        self.__text_preprocessing = self.get_item(['training_params', 'text_preprocessing'], default=list())
        self.DBC2SBC = True if 'DBC2SBC' in self.__text_preprocessing else False
        self.unicode_fix = True if 'unicode_fix' in self.__text_preprocessing else False
        self.remove_stopwords = True if 'remove_stopwords' in self.__text_preprocessing else False

        # tokenzier
        if self.language == 'chinese':
            self.tokenizer = self.get_item(['training_params', 'tokenizer'], default='jieba')
        else:
            self.tokenizer = self.get_item(['training_params', 'tokenizer'], default='nltk')

        if self.extra_feature:
            if self.DBC2SBC:
                logging.warning("Detect the extra feature %s, set the DBC2sbc is False." % ''.join(list(feature_all-formal_feature)))
            if self.unicode_fix:
                logging.warning("Detect the extra feature %s, set the unicode_fix is False." % ''.join(list(feature_all-formal_feature)))
            if self.remove_stopwords:
                logging.warning("Detect the extra feature %s, set the remove_stopwords is False." % ''.join(list(feature_all-formal_feature)))

        if ProblemTypes[self.problem_type] == ProblemTypes.sequence_tagging:
            if self.unicode_fix:
                logging.warning('For sequence tagging task, unicode_fix may change the number of words.')
            if self.remove_stopwords:
                self.remove_stopwords = True
                logging.warning('For sequence tagging task, remove stopwords is forbidden! It is disabled now.')

        if self.phase != 'cache':
            if torch.cuda.is_available() and torch.cuda.device_count() > 0 and self.training_params.get('use_gpu', True):
                self.use_gpu = True
                logging.info("Activating GPU mode, there are %d GPUs available" % torch.cuda.device_count())
            else:
                self.use_gpu = False
                logging.info("Activating CPU mode")

        self.architecture = self.get_item(['architecture'])
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

        if self.phase == 'train' or self.phase == 'test':
            self.loss = BaseLossConf.get_conf(**self.get_item(['loss']))
            self.metrics = self.get_item(['metrics'])
            if 'auc' in self.metrics and ProblemTypes[self.problem_type] == ProblemTypes.classification:
                self.pos_label = self.get_item(['inputs', 'positive_label'], default=None, use_default=True)

    def get_item(self, keys, default=None, use_default=False):
        """

        Args:
            keys:
            default: if some key is not found and default is None, we would raise an Exception, except that use_default is True
            use_default: if you really want to set default to None, set use_default=True

        Returns:

        """
        item = self.conf
        valid_keys = []
        try:
            for key in keys:
                item = item[key]
                valid_keys.append(key)
        except:
            error_keys = copy.deepcopy(valid_keys)
            error_keys.append(key)
            if default is None and use_default is False:
                raise ConfigurationError(
                    "The configuration file %s is illegal. There should be an item configuration[%s], "
                    "but the item %s is not found." % (self.conf_path, "][".join(error_keys), key))
            else:
                # print("configuration[%s] is not found in %s, use default value %s" %
                #               ("][".join(error_keys), self.conf_path, repr(default)))
                item = default

        return item

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
            if self.online_encoder:
                pass
            else:
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
        
    def _load_cache_config_from_conf(self):
        # training data
        self.train_data_md5 = None
        if self.phase == 'train' and self.train_data_path:
            logging.info("Calculating the md5 of traing data ...")
            self.train_data_md5 = md5([self.train_data_path])
            logging.info("the md5 of traing data is %s"%(self.train_data_md5))

        # problem
        self.problem_md5 = None
        
        # encoding 
        self.encoding_cache_dir = None
        self.encoding_cache_index_file_path = None
        self.encoding_cache_index_file_md5_path = None
        self.encoding_file_index = None
        self.encoding_cache_legal_line_cnt = 0
        self.encoding_cache_illegal_line_cnt = 0
        self.load_encoding_cache_generator = None
        

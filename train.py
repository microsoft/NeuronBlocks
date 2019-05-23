# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from settings import ProblemTypes, version

import os
import argparse
import logging
import shutil
import time
import numpy as np
import copy

import torch
from ModelConf import ModelConf
from problem import Problem
from utils.common_utils import dump_to_pkl, load_from_pkl, prepare_dir
from utils.philly_utils import HDFSDirectTransferer
from losses import *
from optimizers import *

from LearningMachine import LearningMachine

class Cache:
    def __init__(self):
        self.dictionary_invalid = True
        self.embedding_invalid = True
        self.encoding_invalid = True
    
    def _check_dictionary(self, conf, params):
        # init status
        self.dictionary_invalid = True
        self.embedding_invalid = True

        # cache_conf
        cache_conf = None
        cache_conf_path = os.path.join(conf.cache_dir, 'conf_cache.json')
        if os.path.isfile(cache_conf_path):
            params_cache = copy.deepcopy(params)
            try:
                cache_conf = ModelConf('cache', cache_conf_path, version, params_cache)
            except Exception as e:
                cache_conf = None
        if cache_conf is None or not self._verify_conf(cache_conf, conf):
            return False
        
        # problem
        if not os.path.isfile(conf.problem_path):
            return False

        # embedding
        if conf.emb_pkl_path:
            if not os.path.isfile(conf.emb_pkl_path):
                return False
            self.embedding_invalid = False
        
        self.dictionary_invalid = False
        return True
        
    def _check_encoding(self, conf):
        self.encoding_invalid = False
        return True

    def check(self, conf, params):
        # dictionary
        if not self._check_dictionary(conf, params):
            self._renew_cache(params, conf.cache_dir)
            return
        # encoding
        if not self._check_encoding(conf):
            self._renew_cache(params, conf.cache_dir)

    def load(self, conf, problem, emb_matrix):
        # load dictionary when (not finetune) and (cache valid)
        if not conf.pretrained_model_path and not self.dictionary_invalid:
            problem.load_problem(conf.problem_path)
            if not self.embedding_invalid:
                emb_matrix = np.array(load_from_pkl(conf.emb_pkl_path))
            logging.info('[Cache] loading dictionary successfully')
        
        if not self.encoding_invalid:
            pass  
        return problem, emb_matrix

    def save(self, conf, params, problem, emb_matrix):
        if not os.path.exists(conf.cache_dir):
            os.makedirs(conf.cache_dir)
        shutil.copy(params.conf_path, os.path.join(conf.cache_dir, 'conf_cache.json'))
        if self.dictionary_invalid:
            if conf.mode == 'philly' and conf.emb_pkl_path.startswith('/hdfs/'):
                with HDFSDirectTransferer(conf.problem_path, with_hdfs_command=True) as transferer:
                    transferer.pkl_dump(problem.export_problem(conf.problem_path, ret_without_save=True))
            else:
                problem.export_problem(conf.problem_path)
            logging.info("[Cache] problem is saved to %s" % conf.problem_path)
            if emb_matrix is not None and conf.emb_pkl_path is not None:
                if conf.mode == 'philly' and conf.emb_pkl_path.startswith('/hdfs/'):
                    with HDFSDirectTransferer(conf.emb_pkl_path, with_hdfs_command=True) as transferer:
                        transferer.pkl_dump(emb_matrix)
                else:
                    dump_to_pkl(emb_matrix, conf.emb_pkl_path)
            logging.info("Embedding matrix saved to %s" % conf.emb_pkl_path)
        
        if self.encoding_invalid:
            pass

    def back_up(self, conf, problem):
        cache_bakup_path = os.path.join(conf.save_base_dir, 'necessary_cache/')
        logging.debug('Prepare dir: %s' % cache_bakup_path)
        prepare_dir(cache_bakup_path, True, allow_overwrite=True, clear_dir_if_exist=True)

        problem.export_problem(cache_bakup_path+'problem.pkl')
        logging.debug("Problem %s is backed up to %s" % (conf.problem_path, cache_bakup_path))

    def _renew_cache(self, params, cache_path):
        if not os.path.exists(cache_path):
            return
        logging.info('Found cache that is ineffective')
        renew_option = 'yes'
        if params.mode != 'philly' and params.force is not True:
            renew_option = input('There exists ineffective cache %s for old models. Input "yes" to renew cache and "no" to exit. (default:no): ' % os.path.abspath(cache_path))
        if renew_option.lower() != 'yes':
            exit(0)
        else:
            shutil.rmtree(cache_path)
            time.sleep(2)  # sleep 2 seconds since the deleting is asynchronous
            logging.info('Old cache is deleted')

    def _verify_conf(self, cache_conf, cur_conf):
        """ To verify if the cache is appliable to current configuration

        Args:
            cache_conf (ModelConf):
            cur_conf (ModelConf):

        Returns:

        """
        if cache_conf.tool_version != cur_conf.tool_version:
            return False

        attribute_to_cmp = ['file_columns', 'object_inputs', 'answer_column_name', 'input_types', 'language']

        flag = True
        for attr in attribute_to_cmp:
            if not (hasattr(cache_conf, attr) and hasattr(cur_conf, attr) and getattr(cache_conf, attr) == getattr(cur_conf, attr)):
                logging.error('configuration %s is inconsistent with the old cache' % attr)
                flag = False
        return flag

def main(params):
    # init
    conf = ModelConf("train", params.conf_path, version, params, mode=params.mode)
    problem = Problem("train", conf.problem_type, conf.input_types, conf.answer_column_name,
        with_bos_eos=conf.add_start_end_for_seq, tagging_scheme=conf.tagging_scheme, tokenizer=conf.tokenizer,
        remove_stopwords=conf.remove_stopwords, DBC2SBC=conf.DBC2SBC, unicode_fix=conf.unicode_fix)
    if conf.pretrained_model_path:
        ### when finetuning, load previous saved problem
        problem.load_problem(conf.saved_problem_path)
   
    # cache verification
    emb_matrix = None
    cache = Cache()
    if conf.use_cache:
        ## check
        cache.check(conf, params)
        ## load
        problem, emb_matrix = cache.load(conf, problem, emb_matrix)

    # data preprocessing
    ## build dictionary when (not in finetune model) and (not use cache or cache invalid)
    if (not conf.pretrained_model_path) and ((conf.use_cache == False) or cache.dictionary_invalid):
        logging.info("Preprocessing... Depending on your corpus size, this step may take a while.")
        # modify train_data_path to [train_data_path, valid_data_path, test_data_path]
        # remember the test_data may be None
        data_path_list = [conf.train_data_path, conf.valid_data_path, conf.test_data_path]
        emb_matrix = problem.build(data_path_list, conf.file_columns, conf.input_types, conf.file_with_col_header,
                                    conf.answer_column_name, word2vec_path=conf.pretrained_emb_path,
                                    word_emb_dim=conf.pretrained_emb_dim, format=conf.pretrained_emb_type,
                                    file_type=conf.pretrained_emb_binary_or_text, involve_all_words=conf.involve_all_words_in_pretrained_emb,
                                    show_progress=True if params.mode == 'normal' else False, cpu_num_workers = conf.cpu_num_workers,
                                    max_vocabulary=conf.max_vocabulary, word_frequency=conf.min_word_frequency)

    ## encode rawdata when do not use cache
    if conf.use_cache == False:
        pass

    # environment preparing
    ## cache save
    if conf.use_cache:
        cache.save(conf, params, problem, emb_matrix)

    if params.make_cache_only:
        if conf.use_cache:
            logging.info("Finish building cache!")
        else:
            logging.info('Please set parameters "use_cache" is true')
        return

    ## back up the problem.pkl to save_base_dir/.necessary_cache. 
    ## During test phase, we would load cache from save_base_dir/.necessary_cache/problem.pkl
    conf.back_up(params) 
    cache.back_up(conf, problem)
    if problem.output_dict:
        logging.debug("Problem target cell dict: %s" % (problem.output_dict.cell_id_map))
    
    # train phase
    ## init 
    ### model
    vocab_info, initialize = None, False
    if not conf.pretrained_model_path:
        vocab_info, initialize = get_vocab_info(problem, emb_matrix), True
  
    lm = LearningMachine('train', conf, problem, vocab_info=vocab_info, initialize=initialize, use_gpu=conf.use_gpu)
    if conf.pretrained_model_path:
        logging.info('Loading the pretrained model: %s...' % conf.pretrained_model_path)
        lm.load_model(conf.pretrained_model_path)

    ### loss
    if len(conf.metrics_post_check) > 0:
        for metric_to_chk in conf.metrics_post_check:
            metric, target = metric_to_chk.split('@')
            if not problem.output_dict.has_cell(target):
                raise Exception("The target %s of %s does not exist in the training data." % (target, metric_to_chk))
    loss_conf = conf.loss
    loss_conf['output_layer_id'] = conf.output_layer_id
    loss_conf['answer_column_name'] = conf.answer_column_name
    # loss_fn = eval(loss_conf['type'])(**loss_conf['conf'])
    loss_fn = Loss(**loss_conf)
    if conf.use_gpu is True:
        loss_fn.cuda()

    ### optimizer
    optimizer = eval(conf.optimizer_name)(lm.model.parameters(), **conf.optimizer_params)

    ## train
    lm.train(optimizer, loss_fn)

    ## test the best model with the best model saved
    lm.load_model(conf.model_save_path)
    if conf.test_data_path is not None:
        test_path = conf.test_data_path
    elif conf.valid_data_path is not None:
        test_path = conf.valid_data_path
    logging.info('Testing the best model saved at %s, with %s' % (conf.model_save_path, test_path))
    if not test_path.endswith('pkl'):
        lm.test(loss_fn, test_path, predict_output_path=conf.predict_output_path)
    else:
        lm.test(loss_fn, test_path)

def get_vocab_info(problem, emb_matrix):
    vocab_info = dict() # include input_type's vocab_size & init_emd_matrix
    vocab_sizes = problem.get_vocab_sizes()
    for input_cluster in vocab_sizes:
        vocab_info[input_cluster] = dict()
        vocab_info[input_cluster]['vocab_size'] = vocab_sizes[input_cluster]
        # add extra info for char_emb
        if input_cluster.lower() == 'char':
            for key, value in conf.input_types[input_cluster].items():
                if key != 'cols':
                    vocab_info[input_cluster][key] = value
        if input_cluster == 'word' and emb_matrix is not None:
            vocab_info[input_cluster]['init_weights'] = emb_matrix
        else:
            vocab_info[input_cluster]['init_weights'] = None
    return vocab_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument("--conf_path", type=str, help="configuration path")
    parser.add_argument("--train_data_path", type=str)
    parser.add_argument("--valid_data_path", type=str)
    parser.add_argument("--test_data_path", type=str)
    parser.add_argument("--pretrained_emb_path", type=str)
    parser.add_argument("--pretrained_emb_type", type=str, default='glove', help='glove|word2vec|fasttext')
    parser.add_argument("--pretrained_emb_binary_or_text", type=str, default='text', help='text|binary')
    parser.add_argument("--involve_all_words_in_pretrained_emb", type=bool, default=False, help='By default, only words that show up in the training data are involved.')
    parser.add_argument("--pretrained_model_path", type=str, help='load pretrained model, and then finetune it.')
    parser.add_argument("--cache_dir", type=str, help='where stores the built problem.pkl where there are dictionaries like word2id, id2word. CAUTION: if there is a previous model, the dictionaries would be loaded from os.path.dir(previous_model_path)/.necessary_cache/problem.pkl')
    parser.add_argument("--model_save_dir", type=str, help='where to store models')
    parser.add_argument("--predict_output_path", type=str, help='specify another prediction output path, instead of conf[outputs][save_base_dir] + conf[outputs][predict_output_name] defined in configuration file')
    parser.add_argument("--log_dir", type=str, help='If not specified, logs would be stored in conf_bilstmlast.json/outputs/save_base_dir')
    parser.add_argument("--make_cache_only", type=bool, default=False, help='make cache without training')
    parser.add_argument("--max_epoch", type=int, help='maximum number of epochs')
    parser.add_argument("--batch_size", type=int, help='batch_size of each gpu')
    parser.add_argument("--learning_rate", type=float, help='learning rate')
    parser.add_argument("--mode", type=str, default='normal', help='normal|philly')
    parser.add_argument("--force", type=bool, default=False, help='Allow overwriting if some files or directories already exist.')
    parser.add_argument("--disable_log_file", type=bool, default=False, help='If True, disable log file')
    parser.add_argument("--debug", type=bool, default=False)

    params, _ = parser.parse_known_args()
    # use for debug, remember delete
    # params.conf_path = 'configs_example/conf_debug_charemb.json'

    assert params.conf_path, 'Please specify a configuration path via --conf_path'
    if params.pretrained_emb_path and not os.path.isabs(params.pretrained_emb_path):
        params.pretrained_emb_path = os.path.join(os.getcwd(), params.pretrained_emb_path)
    if params.debug is True:
        import debugger
    main(params)

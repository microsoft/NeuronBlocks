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


def verify_cache(cache_conf, cur_conf):
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
    conf = ModelConf("train", params.conf_path, version, params, mode=params.mode)

    shutil.copy(params.conf_path, conf.save_base_dir)
    logging.info('Configuration file is backed up to %s' % (conf.save_base_dir))

    if ProblemTypes[conf.problem_type] == ProblemTypes.sequence_tagging:
        problem = Problem(conf.problem_type, conf.input_types, conf.answer_column_name,
            source_with_start=True, source_with_end=True, source_with_unk=True, source_with_pad=True,
            target_with_start=True, target_with_end=True, target_with_unk=True, target_with_pad=True, same_length=True,
            with_bos_eos=conf.add_start_end_for_seq, tagging_scheme=conf.tagging_scheme, tokenizer=conf.tokenizer,
            remove_stopwords=conf.remove_stopwords, DBC2SBC=conf.DBC2SBC, unicode_fix=conf.unicode_fix)
    elif ProblemTypes[conf.problem_type] == ProblemTypes.classification \
            or ProblemTypes[conf.problem_type] == ProblemTypes.regression:
        problem = Problem(conf.problem_type, conf.input_types, conf.answer_column_name,
            source_with_start=True, source_with_end=True, source_with_unk=True, source_with_pad=True,
            target_with_start=False, target_with_end=False, target_with_unk=False, target_with_pad=False,
            same_length=False, with_bos_eos=conf.add_start_end_for_seq, tokenizer=conf.tokenizer,
                          remove_stopwords=conf.remove_stopwords, DBC2SBC=conf.DBC2SBC, unicode_fix=conf.unicode_fix)
    elif ProblemTypes[conf.problem_type] == ProblemTypes.mrc:
        problem = Problem(conf.problem_type, conf.input_types, conf.answer_column_name,
                          source_with_start=True, source_with_end=True, source_with_unk=True, source_with_pad=True,
                          target_with_start=False, target_with_end=False, target_with_unk=False, target_with_pad=False,
                          same_length=False, with_bos_eos=False, tokenizer=conf.tokenizer,
                          remove_stopwords=conf.remove_stopwords, DBC2SBC=conf.DBC2SBC, unicode_fix=conf.unicode_fix)

    cache_load_flag = False
    if not conf.pretrained_model_path:
        # first time training, load cache if appliable
        if conf.use_cache:
            cache_conf_path = os.path.join(conf.cache_dir, 'conf_cache.json')
            if os.path.isfile(cache_conf_path):
                params_cache = copy.deepcopy(params)
                '''
                for key in vars(params_cache):
                    setattr(params_cache, key, None)
                params_cache.mode = params.mode
                '''
                try:
                    cache_conf = ModelConf('cache', cache_conf_path, version, params_cache)
                except Exception as e:
                    cache_conf = None
                if cache_conf is None or verify_cache(cache_conf, conf) is not True:
                    logging.info('Found cache that is ineffective')
                    if params.mode == 'philly' or params.force is True:
                        renew_option = 'yes'
                    else:
                        renew_option = input('There exists ineffective cache %s for old models. Input "yes" to renew cache and "no" to exit. (default:no): ' % os.path.abspath(conf.cache_dir))
                    if renew_option.lower() != 'yes':
                        exit(0)
                    else:
                        shutil.rmtree(conf.cache_dir)
                        time.sleep(2)  # sleep 2 seconds since the deleting is asynchronous
                        logging.info('Old cache is deleted')
                else:
                    logging.info('Found cache that is appliable to current configuration...')

            elif os.path.isdir(conf.cache_dir):
                renew_option = input('There exists ineffective cache %s for old models. Input "yes" to renew cache and "no" to exit. (default:no): ' % os.path.abspath(conf.cache_dir))
                if renew_option.lower() != 'yes':
                    exit(0)
                else:
                    shutil.rmtree(conf.cache_dir)
                    time.sleep(2)  # Sleep 2 seconds since the deleting is asynchronous
                    logging.info('Old cache is deleted')

            if not os.path.exists(conf.cache_dir):
                os.makedirs(conf.cache_dir)
                shutil.copy(params.conf_path, os.path.join(conf.cache_dir, 'conf_cache.json'))

        # first time training, load problem from cache, and then backup the cache to model_save_dir/.necessary_cache/
        if conf.use_cache and os.path.isfile(conf.problem_path):
            problem.load_problem(conf.problem_path)
            if conf.emb_pkl_path is not None:
                if os.path.isfile(conf.emb_pkl_path):
                    emb_matrix = np.array(load_from_pkl(conf.emb_pkl_path))
                    cache_load_flag = True
                else:
                    if params.mode == 'normal':
                        renew_option = input('The cache is invalid because the embedding matrix does not exist in the cache directory. Input "yes" to renew cache and "no" to exit. (default:no): ')
                        if renew_option.lower() != 'yes':
                            exit(0)
                    else:
                        # by default, renew cache
                        renew_option = 'yes'
            else:
                emb_matrix = None
                cache_load_flag = True
            if cache_load_flag:
                logging.info("Cache loaded!")

        if cache_load_flag is False:
            logging.info("Preprocessing... Depending on your corpus size, this step may take a while.")
            if conf.pretrained_emb_path:
                emb_matrix = problem.build(conf.train_data_path, conf.file_columns, conf.input_types, conf.file_with_col_header,
                                           conf.answer_column_name, word2vec_path=conf.pretrained_emb_path,
                                           word_emb_dim=conf.pretrained_emb_dim, format=conf.pretrained_emb_type,
                                           file_type=conf.pretrained_emb_binary_or_text, involve_all_words=conf.involve_all_words_in_pretrained_emb,
                                           show_progress=True if params.mode == 'normal' else False, cpu_num_workers = conf.cpu_num_workers,
                                           max_vocabulary=conf.max_vocabulary, word_frequency=conf.min_word_frequency)
            else:
                emb_matrix = problem.build(conf.train_data_path, conf.file_columns, conf.input_types, conf.file_with_col_header,
                                           conf.answer_column_name, word2vec_path=None, word_emb_dim=None, format=None,
                                           file_type=None, involve_all_words=conf.involve_all_words_in_pretrained_emb,
                                           show_progress=True if params.mode == 'normal' else False,  cpu_num_workers = conf.cpu_num_workers,
                                           max_vocabulary=conf.max_vocabulary, word_frequency=conf.min_word_frequency)

            if conf.mode == 'philly' and conf.emb_pkl_path.startswith('/hdfs/'):
                with HDFSDirectTransferer(conf.problem_path, with_hdfs_command=True) as transferer:
                    transferer.pkl_dump(problem.export_problem(conf.problem_path, ret_without_save=True))
            else:
                problem.export_problem(conf.problem_path)
            if conf.use_cache:
                logging.info("Cache saved to %s" % conf.problem_path)
                if emb_matrix is not None and conf.emb_pkl_path is not None:
                    if conf.mode == 'philly' and conf.emb_pkl_path.startswith('/hdfs/'):
                        with HDFSDirectTransferer(conf.emb_pkl_path, with_hdfs_command=True) as transferer:
                            transferer.pkl_dump(emb_matrix)
                    else:
                        dump_to_pkl(emb_matrix, conf.emb_pkl_path)
                    logging.info("Embedding matrix saved to %s" % conf.emb_pkl_path)
            else:
                logging.debug("Cache saved to %s" % conf.problem_path)

        # Back up the problem.pkl to save_base_dir/.necessary_cache. During test phase, we would load cache from save_base_dir/.necessary_cache/problem.pkl
        cache_bakup_path = os.path.join(conf.save_base_dir, 'necessary_cache/')
        logging.debug('Prepare dir: %s' % cache_bakup_path)
        prepare_dir(cache_bakup_path, True, allow_overwrite=True, clear_dir_if_exist=True)

        shutil.copy(conf.problem_path, cache_bakup_path)
        logging.debug("Problem %s is backed up to %s" % (conf.problem_path, cache_bakup_path))
        if problem.output_dict:
            logging.debug("Problem target cell dict: %s" % (problem.output_dict.cell_id_map))

        if params.make_cache_only:
            logging.info("Finish building cache!")
            return

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

        lm = LearningMachine('train', conf, problem, vocab_info=vocab_info, initialize=True, use_gpu=conf.use_gpu)
    else:
        # when finetuning, load previous saved problem
        problem.load_problem(conf.saved_problem_path)
        lm = LearningMachine('train', conf, problem, vocab_info=None, initialize=False, use_gpu=conf.use_gpu)

    if len(conf.metrics_post_check) > 0:
        for metric_to_chk in conf.metrics_post_check:
            metric, target = metric_to_chk.split('@')
            if not problem.output_dict.has_cell(target):
                raise Exception("The target %s of %s does not exist in the training data." % (target, metric_to_chk))

    if conf.pretrained_model_path:
        logging.info('Loading the pretrained model: %s...' % conf.pretrained_model_path)
        lm.load_model(conf.pretrained_model_path)

    loss_conf = conf.loss
    loss_conf['output_layer_id'] = conf.output_layer_id
    loss_conf['answer_column_name'] = conf.answer_column_name
    # loss_fn = eval(loss_conf['type'])(**loss_conf['conf'])
    loss_fn = Loss(**loss_conf)
    if conf.use_gpu is True:
        loss_fn.cuda()

    optimizer = eval(conf.optimizer_name)(lm.model.parameters(), **conf.optimizer_params)

    lm.train(optimizer, loss_fn)

    # test the best model with the best model saved
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

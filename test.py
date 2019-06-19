# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from settings import ProblemTypes, version

import os
import argparse
import logging

from ModelConf import ModelConf
from problem import Problem
from losses import *

from LearningMachine import LearningMachine


def main(params):
    conf = ModelConf("test", params.conf_path, version, params, mode=params.mode)
    problem = Problem("test", conf.problem_type, conf.input_types, conf.answer_column_name,
        with_bos_eos=conf.add_start_end_for_seq, tagging_scheme=conf.tagging_scheme, tokenizer=conf.tokenizer,
        remove_stopwords=conf.remove_stopwords, DBC2SBC=conf.DBC2SBC, unicode_fix=conf.unicode_fix)

    if os.path.isfile(conf.saved_problem_path):
        problem.load_problem(conf.saved_problem_path)
        logging.info("Problem loaded!")
        logging.debug("Problem loaded from %s" % conf.saved_problem_path)
    else:
        raise Exception("Problem does not exist!")

    if len(conf.metrics_post_check) > 0:
        for metric_to_chk in conf.metrics_post_check:
            metric, target = metric_to_chk.split('@')
            if not problem.output_dict.has_cell(target):
                raise Exception("The target %s of %s does not exist in the training data." % (target, metric_to_chk))

    lm = LearningMachine('test', conf, problem, vocab_info=None, initialize=False, use_gpu=conf.use_gpu)
    lm.load_model(conf.previous_model_path)

    loss_conf = conf.loss
    # loss_fn = eval(loss_conf['type'])(**loss_conf['conf'])
    loss_conf['output_layer_id'] = conf.output_layer_id
    loss_conf['answer_column_name'] = conf.answer_column_name
    loss_fn = Loss(**loss_conf)
    if conf.use_gpu is True:
        loss_fn.cuda()

    test_path = params.test_data_path
    if conf.test_data_path is not None:
        test_path = conf.test_data_path
    elif conf.valid_data_path is not None:
        test_path = conf.valid_data_path

    logging.info('Testing the best model saved at %s, with %s' % (conf.previous_model_path, test_path))
    if not test_path.endswith('pkl'):
        lm.test(loss_fn, test_path, predict_output_path=conf.predict_output_path)
    else:
        lm.test(loss_fn, test_path)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    parser = argparse.ArgumentParser(description='testing')
    parser.add_argument("--conf_path", type=str, help="configuration path")
    parser.add_argument("--test_data_path", type=str, help='specify another test data path, instead of the one defined in configuration file')
    parser.add_argument("--previous_model_path", type=str, help='load model trained previously.')
    parser.add_argument("--predict_output_path", type=str, help='specify another prediction output path, instead of conf[outputs][save_base_dir] + conf[outputs][predict_output_name] defined in configuration file')
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--batch_size", type=int, help='batch_size of each gpu')
    parser.add_argument("--mode", type=str, default='normal', help='normal|philly')
    parser.add_argument("--force", type=bool, default=False, help='Allow overwriting if some files or directories already exist.')
    parser.add_argument("--disable_log_file", type=bool, default=False, help='If True, disable log file')
    parser.add_argument("--debug", type=bool, default=False)
    params, _ = parser.parse_known_args()

    assert params.conf_path, 'Please specify a configuration path via --conf_path'
    if params.debug is True:
        import debugger
    main(params)
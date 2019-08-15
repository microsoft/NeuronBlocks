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

    lm.model.eval()
    import torch
    import numpy as np
    export_onnx_model = True
    export_onnx_input_data = True
    if export_onnx_model:
        if not os.path.exists("./export"):
            os.mkdir("./export")
        input_ids_query = torch.zeros([30, 30], dtype=torch.long)
        input_ids_passage = torch.zeros([30, 100], dtype=torch.long)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        query_word = torch.from_numpy(np.zeros((30, 30), dtype=int)).to(device)
        answer_word = torch.from_numpy(np.zeros((30, 100), dtype=int)).to(device)
        question_len = torch.from_numpy(np.ones(30, dtype=int) * 30).to(device)
        answer_len = torch.from_numpy(np.ones(30, dtype=int) * 100).to(device)
        target_len = torch.from_numpy(np.ones(30, dtype=int)).to(device)

    torch.onnx.export(lm.model, (query_word, answer_word, question_len, answer_len, target_len), "export/from_pytorch_10.onnx", input_names=['query_word', 'answer_word', 'question_len', 'answer_len', 'target_len'], output_names=["logits"], aten=False,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX, verbose=True, _retain_param_name=True, opset_version=9, do_constant_folding=True)
    logging.info('DONE!')


if __name__ == "__main__":
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
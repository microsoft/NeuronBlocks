# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

# add the project root to python path
import os
from settings import ProblemTypes, version

import argparse
import logging

from ModelConf import ModelConf
from problem import Problem
from utils.common_utils import log_set, dump_to_pkl, load_from_pkl

def main(params, data_path, save_path):
    conf = ModelConf("cache", params.conf_path, version, params)

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
            target_with_start=False, target_with_end=False, target_with_unk=False, target_with_pad=False, same_length=True,
            with_bos_eos=conf.add_start_end_for_seq, tokenizer=conf.tokenizer, remove_stopwords=conf.remove_stopwords,
                          DBC2SBC=conf.DBC2SBC, unicode_fix=conf.unicode_fix)

    if os.path.isfile(conf.problem_path):
        problem.load_problem(conf.problem_path)
        logging.info("Cache loaded!")
        logging.debug("Cache loaded from %s" % conf.problem_path)
    else:
        raise Exception("Cache does not exist!")

    data, length, target = problem.encode(data_path, conf.file_columns, conf.input_types, conf.file_with_col_header,
                                          conf.object_inputs, conf.answer_column_name, conf.min_sentence_len,
                                          extra_feature=conf.extra_feature,max_lengths=conf.max_lengths, file_format='tsv',
                                          cpu_num_workers=conf.cpu_num_workers)
    if not os.path.isdir(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    dump_to_pkl({'data': data, 'length': length, 'target': target}, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data encoding')
    parser.add_argument("data_path", type=str)
    parser.add_argument("save_path", type=str)
    parser.add_argument("--conf_path", type=str, default='conf.json', help="configuration path")
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--force", type=bool, default=False)

    log_set('encoding_data.log')

    params, _ = parser.parse_known_args()

    if params.debug is True:
        import debugger
    main(params, params.data_path, params.save_path)
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import logging
import numpy as np
from core.CellDict import CellDict
from tqdm import tqdm
from utils.corpus_utils import load_embedding
import nltk
nltk.download('punkt', quiet=True)
from utils.BPEEncoder import BPEEncoder
import codecs
from multiprocessing import cpu_count
import os
import pickle as pkl
from utils.common_utils import load_from_pkl, dump_to_pkl

from settings import ProblemTypes
import multiprocessing
import math

from core.EnglishTokenizer import EnglishTokenizer
from core.EnglishTextPreprocessor import EnglishTextPreprocessor
from utils.exceptions import PreprocessError
import torch
import torch.nn as nn

class Problem():
    def __init__(self, problem_type, input_types, answer_column_name=None, lowercase=False,
            source_with_start=True, source_with_end=True, source_with_unk=True,
            source_with_pad=True, target_with_start=False, target_with_end=False,
            target_with_unk=True, target_with_pad=True, same_length=True, with_bos_eos=True,
            tagging_scheme=None, remove_stopwords=False, DBC2SBC=True, unicode_fix=True):
        """

        Args:
            input_types: {
                  "word": ["word1", "word1"],
                  "postag": ["postag_feature1", "postag_feature2"]
                }
            answer_column_name: "label" after v1.0.0 answer_column_name change to list
            source_with_start:
            source_with_end:
            source_with_unk:
            source_with_pad:
            target_with_start:
            target_with_end:
            target_with_unk:
            target_with_pad:
            same_length:
            with_bos_eos: whether to add bos and eos when encoding
        """
        self.lowercase = lowercase

        self.input_dicts = dict()
        self.problem_type = problem_type
        self.tagging_scheme = tagging_scheme
        self.with_bos_eos = with_bos_eos
        self.source_with_start = source_with_start
        self.source_with_end = source_with_end
        self.source_with_unk = source_with_unk
        self.source_with_pad = source_with_pad
        self.target_with_start = target_with_start
        self.target_with_end = target_with_end
        self.target_with_unk = target_with_unk
        self.target_with_pad = target_with_pad

        for input_type in input_types:
           self.input_dicts[input_type] = CellDict(with_unk=source_with_unk, with_pad=source_with_pad,
                                        with_start=source_with_start, with_end=source_with_end)
        if ProblemTypes[self.problem_type] == ProblemTypes.sequence_tagging or \
                ProblemTypes[self.problem_type] == ProblemTypes.classification :
            self.output_dict = CellDict(with_unk=target_with_unk, with_pad=target_with_pad,
                                    with_start=target_with_start, with_end=target_with_end)
        elif ProblemTypes[self.problem_type] == ProblemTypes.regression or \
                ProblemTypes[self.problem_type] == ProblemTypes.mrc:
            self.output_dict = None

        self.file_column_num = None

        self.tokenizer = EnglishTokenizer(tokenizer='nltk', remove_stopwords=remove_stopwords)
        self.text_preprocessor = EnglishTextPreprocessor(DBC2SBC=DBC2SBC, unicode_fix=unicode_fix)

    def input_word_num(self):
        return self.input_word_dict.cell_num()

    def output_target_num(self):
        if ProblemTypes[self.problem_type] == ProblemTypes.sequence_tagging or ProblemTypes[self.problem_type] == ProblemTypes.classification:
            return self.output_dict.cell_num()
        else:
            return None

    def get_data_list_from_file(self, fin, file_with_col_header):
        data_list = list()
        for index, line in enumerate(fin):
            if file_with_col_header and index == 0:
                continue
            line = line.rstrip()
            if not line:
                break
            data_list.append(line)
        return data_list

    def build_training_data_list(self, training_data_list, file_columns, input_types, answer_column_name, bpe_encoder=None):
        docs = dict()           # docs of each type of input
        col_index_types = dict()        # input type of each column, col_index_types[0] = 'word'/'postag'

        target_docs = {}  # after v1.0.0, the target_docs change to dict for support multi_label
        columns_to_target = {}
        for single_target in answer_column_name:
            target_docs[single_target] = []
            columns_to_target[file_columns[single_target]] = single_target

        for input_type in input_types:
            docs[input_type] = []
            # char is not in file_columns
            if input_type == 'char':
                continue
            for col in input_types[input_type]['cols']:
                col_index_types[file_columns[col]] = input_type

        cnt_legal = 0
        cnt_illegal = 0
        for line in training_data_list:
            # line_split = list(filter(lambda x: len(x) > 0, line.rstrip().split('\t')))
            line_split = line.rstrip().split('\t')
            if len(line_split) != len(file_columns):
                logging.warning("Current line is inconsistent with configuration/inputs/file_header. Ingore now. %s" % line)
                cnt_illegal += 1
                continue
            cnt_legal += 1

            for i in range(len(line_split)):
                if i in col_index_types:
                    if self.lowercase:
                        line_split[i] = line_split[i].lower()
                    line_split[i] = self.text_preprocessor.preprocess(line_split[i])

                    if col_index_types[i] == 'word':
                        token_list = self.tokenizer.tokenize(line_split[i])
                        docs[col_index_types[i]].append(token_list)
                        if 'char' in docs:
                            # add char
                            docs['char'].append([single_char for single_char in ''.join(token_list)])
                    elif col_index_types[i] == 'bpe':
                        bpe_tokens = []
                        for token in self.tokenizer.tokenize(line_split[i]):
                            bpe_tokens.extend(bpe_encoder.bpe(token))
                        docs[col_index_types[i]].append(bpe_tokens)
                    else:
                        docs[col_index_types[i]].append(line_split[i].split(" "))
                # target_docs change to dict
                elif i in columns_to_target.keys():
                    curr_target = columns_to_target[i]
                    if ProblemTypes[self.problem_type] == ProblemTypes.classification:
                        target_docs[curr_target].append(line_split[i])
                    elif ProblemTypes[self.problem_type] == ProblemTypes.sequence_tagging:
                        target_docs[curr_target].append(line_split[i].split(" "))
                    elif ProblemTypes[self.problem_type] == ProblemTypes.regression or \
                            ProblemTypes[self.problem_type] == ProblemTypes.mrc:
                        pass
        return docs, target_docs, cnt_legal, cnt_illegal

    def build_training_multi_processor(self, training_data_list, cpu_num_workers, file_columns, input_types, answer_column_name, bpe_encoder=None):
        res = []
        process_num = cpu_count()
        if cpu_num_workers > 0:
            process_num = cpu_num_workers
        # logging.info("multiprocess enabled, process num: %d" % (process_num))
        process_p = multiprocessing.Pool(process_num)
        for i in range(process_num):
            size = math.ceil(len(training_data_list)/ process_num)
            start = size * i
            end = (i + 1) * size if (i + 1) * size < len(training_data_list) else len(training_data_list)
            temp_data_list = training_data_list[start:end]
            res.append((i, process_p.apply_async(self.build_training_data_list,
                            args=(temp_data_list, file_columns, input_types, answer_column_name, bpe_encoder)
                                                 )
                        )
                       )

        process_p.close()
        process_p.join()

        docs = dict()           # docs of each type of input
        target_docs = []
        cnt_legal = 0
        cnt_illegal = 0
        sort_res = sorted(res, key=lambda x:x[0])
        for (index, j) in sort_res:
            #logging.info("collect proccesor %d result" % index)
            tmp_docs, tmp_target_docs, tmp_cnt_legal, tmp_cnt_illegal = j.get()
            if len(docs) == 0:
                docs = tmp_docs
            else:
                for key, value in tmp_docs.items():
                    docs[key].extend(value)
            if len(target_docs) == 0:
                target_docs = tmp_target_docs
            else:
                for single_type in tmp_target_docs:
                    target_docs[single_type].extend(tmp_target_docs[single_type])
            # target_docs.extend(tmp_target_docs)
            cnt_legal += tmp_cnt_legal
            cnt_illegal += tmp_cnt_illegal

        return docs, target_docs, cnt_legal, cnt_illegal

    def build(self, training_data_path, file_columns, input_types, file_with_col_header, answer_column_name, word2vec_path=None, word_emb_dim=None,
              format=None, file_type=None, involve_all_words=None, file_format="tsv", show_progress=True,
              cpu_num_workers=-1, max_vocabulary=800000, word_frequency=3):
        """

        Args:
            training_data_path:
            file_columns: {
                  "word1": 0,
                  "word2": 1,
                  "label":   2,
                  "postag_feature1": 3,
                  "postag_feature2": 4
                },
            input_types:
                e.g.
                {
                  "word": {
                    "cols": ["word1", "word2"],
                    "dim": 300
                  },
                  "postag": {
                    "cols": ["postag_feature1", "postag_feature2"],
                    "dim": 20
                  },
                }
                or
                {
                  "bpe": {
                    "cols": ["word1", "word2"],
                    "dim": 100
                    "bpe_path": "xxx.bpe"
                  }
                }

            word2vec_path:
            word_emb_dim:
            involve_all_word: involve all words that show up in the pretrained embedding
            file_format: "tsv", or "json". Note "json" means each sample is represented by a json string.

        Returns:

        """
        if 'bpe' in input_types:
            try:
                bpe_encoder = BPEEncoder(input_types['bpe']['bpe_path'])
            except KeyError:
                raise Exception('Please define a bpe path at the embedding layer.')
        else:
            bpe_encoder = None

        self.file_column_num = len(file_columns)
        with open(training_data_path, "r", encoding='utf-8') as f:
            progress = self.get_data_list_from_file(f, file_with_col_header)
            docs, target_docs, cnt_legal, cnt_illegal = self.build_training_multi_processor(progress, cpu_num_workers, file_columns, input_types, answer_column_name, bpe_encoder=bpe_encoder)

        logging.info("Corpus imported: %d legal lines, %d illegal lines." % (cnt_legal, cnt_illegal))

        if word2vec_path and involve_all_words is True:
            logging.info("Getting pre-trained embeddings...")
            word_emb_dict = load_embedding(word2vec_path, word_emb_dim, format, file_type, with_head=False, word_set=None)
            self.input_dicts['word'].build([list(word_emb_dict.keys())], max_vocabulary_num=len(word_emb_dict), threshold=0)
        for input_type in input_types:
            if input_type != 'word':
                self.input_dicts[input_type].build(docs[input_type], max_vocabulary_num=max_vocabulary, threshold=word_frequency)
            else:
                self.input_dicts[input_type].build(docs[input_type], max_vocabulary_num=max_vocabulary, threshold=word_frequency)
            logging.info("%d types in %s" % (self.input_dicts[input_type].cell_num(), input_type))
        if ProblemTypes[self.problem_type] == ProblemTypes.classification:
            self.output_dict.build(list(target_docs.values())[0], threshold=0)
        elif ProblemTypes[self.problem_type] == ProblemTypes.sequence_tagging:
            self.output_dict.build(list(target_docs.values())[0], threshold=0)
        elif ProblemTypes[self.problem_type] == ProblemTypes.regression or \
                ProblemTypes[self.problem_type] == ProblemTypes.mrc:
            pass

        if self.output_dict:
            logging.info("%d types in target" % (self.output_dict.cell_num()))

        logging.debug("Cell dict built")

        if word2vec_path:
            if not involve_all_words:
                logging.info("Getting pre-trained embeddings...")
                word_emb_dict = load_embedding(word2vec_path, word_emb_dim, format, file_type, with_head=False, word_set=self.input_dicts['word'].cell_id_map.keys())

            for word in word_emb_dict:
                loaded_emb_dim = len(word_emb_dict[word])
                break

            assert loaded_emb_dim == word_emb_dim, "The dimension of defined word embedding is inconsistent with the pretrained embedding provided!"

            if self.input_dicts['word'].with_unk:
                word_emb_dict['<unk>'] = np.random.random(size=word_emb_dim)
            if self.input_dicts['word'].with_pad:
                word_emb_dict['<pad>'] = np.random.random(size=word_emb_dim)

            word_emb_matrix = []
            unknown_word_count = 0
            for i in range(self.input_dicts['word'].cell_num()):
                if self.input_dicts['word'].id_cell_map[i] in word_emb_dict:
                    word_emb_matrix.append(word_emb_dict[self.input_dicts['word'].id_cell_map[i]])
                else:
                    word_emb_matrix.append(word_emb_dict['<unk>'])
                    unknown_word_count += 1
            word_emb_matrix = np.array(word_emb_matrix)
            logging.info("word embedding matrix shape:(%d, %d); unknown word count: %d;" %
                         (len(word_emb_matrix), len(word_emb_matrix[0]), unknown_word_count))
            logging.info("Word embedding loaded")
        else:
            word_emb_matrix = None
        return word_emb_matrix
    
    def encode_data_multi_processor(self, data_list, cpu_num_workers, file_columns, input_types, object_inputs,
                answer_column_name, min_sentence_len, extra_feature, max_lengths=None, fixed_lengths=None, file_format="tsv", bpe_encoder=None):
        def judge_dict(obj):
            return True if isinstance(obj, dict) else False
        res = []

        process_num = cpu_count()
        if cpu_num_workers > 0:
            process_num = cpu_num_workers
        #logging.info("multiprocess enabled, process num: %d" % (process_num))
        process_p = multiprocessing.Pool(process_num)
        for i in range(process_num):
            size = math.ceil(len(data_list)/ process_num)
            start = size * i
            end = (i + 1) * size if (i + 1) * size < len(data_list) else len(data_list)
            temp_data_list = data_list[start:end]
            res.append((i, process_p.apply_async(self.encode_data_list,
                                                 args=((temp_data_list, file_columns, input_types, object_inputs,
                    answer_column_name, min_sentence_len, extra_feature, max_lengths, fixed_lengths, file_format, bpe_encoder)
                                                 )
                                                 )
                        )
                       )

        process_p.close()
        process_p.join()

        data = dict()
        lengths = dict()
        target = dict()
        cnt_legal = 0
        cnt_illegal = 0

        sort_res = sorted(res, key=lambda x:x[0])
        for (index, j) in sort_res:
            # logging.info("collect proccesor %d result"%index)
            tmp_data, tmp_lengths, tmp_target, tmp_cnt_legal, tmp_cnt_illegal = j.get()

            if len(data) == 0:
                data = tmp_data
            else:
                for branch in tmp_data:
                    for input_type in data[branch]:
                        data[branch][input_type].extend(tmp_data[branch][input_type])
            if len(lengths) == 0:
                lengths = tmp_lengths
            else:
                for branch in tmp_lengths:
                    if judge_dict(tmp_lengths[branch]):
                        for type_branch in tmp_lengths[branch]:
                            lengths[branch][type_branch].extend(tmp_lengths[branch][type_branch])
                    else:
                        lengths[branch].extend(tmp_lengths[branch])
            if not tmp_target:
                target = None
            else:
                if len(target) == 0:
                    target = tmp_target
                else:
                    for single_type in tmp_target:
                        target[single_type].extend(tmp_target[single_type])
            cnt_legal += tmp_cnt_legal
            cnt_illegal += tmp_cnt_illegal

        return data, lengths, target, cnt_legal, cnt_illegal

    def encode_data_list(self, data_list, file_columns, input_types, object_inputs, answer_column_name, min_sentence_len,
                         extra_feature, max_lengths=None, fixed_lengths=None, file_format="tsv", bpe_encoder=None):
        data = dict()
        lengths = dict()
        char_emb = True if 'char' in [single_input_type.lower() for single_input_type in input_types] else False
        if answer_column_name is not None and len(answer_column_name)>0:
            target = {}
            lengths['target'] = {}
            columns_to_target = {}
            for single_target in answer_column_name:
                target[single_target] = []
                columns_to_target[file_columns[single_target]] = single_target
                lengths['target'][single_target] = []
        else:
            target = None

        col_index_types = dict()        # input type of each column, namely the inverse of file_columns, e.g. col_index_types[0] = 'query_index'
        type2cluster = dict()           # e.g. type2cluster['query_index'] = 'word'

        type_branches = dict()            # branch of input type, e.g. type_branches['query_index'] = 'query'

        for branch in object_inputs:
            data[branch] = dict()
            lengths[branch] = dict()
            lengths[branch]['sentence_length'] = []
            temp_branch_char = False
            for input_type in object_inputs[branch]:
                type_branches[input_type] = branch
                data[branch][input_type] = []
                if 'char' in input_type.lower():
                    temp_branch_char = True
            if char_emb and temp_branch_char:
                lengths[branch]['word_length'] = []
        # for extra_info for mrc task
        if ProblemTypes[self.problem_type] == ProblemTypes.mrc:
            extra_info_type = 'passage'
            if extra_info_type not in object_inputs:
                raise Exception('MRC task need passage for model_inputs, given: {0}'.format(';'.join(list(object_inputs.keys()))))
            data[extra_info_type]['extra_passage_text'] = []
            data[extra_info_type]['extra_passage_token_offsets'] = []

        for input_type in input_types:
            for col_name in input_types[input_type]['cols']:
                type2cluster[col_name] = input_type
                if col_name in file_columns:
                    col_index_types[file_columns[col_name]] = col_name


        cnt_legal = 0
        cnt_illegal = 0

        # cnt_length_unconsistent = 0
        cnt_all = 0

        for line in data_list:
            # line_split = list(filter(lambda x: len(x) > 0, line.rstrip().split('\t')))
            line_split = line.rstrip().split('\t')
            cnt_all += 1
            if len(line_split) != len(file_columns):
                # logging.warning("Current line is inconsistent with configuration/inputs/file_header. Ingore now. %s" % line)
                cnt_illegal += 1
                if cnt_illegal / cnt_all > 0.33:
                    raise PreprocessError('The illegal data is too much. Please check the number of data columns or text token version.')
                continue
            # cnt_legal += 1
            length_appended_set = set()  # to store branches whose length have been appended to lengths[branch]

            if ProblemTypes[self.problem_type] == ProblemTypes.mrc:
                passage_token_offsets = None

            for i in range(len(line_split)):
                line_split[i] = line_split[i].strip()
                if i in col_index_types:
                    # these are data
                    branch = type_branches[col_index_types[i]]
                    input_type = []
                    input_type.append(col_index_types[i])
                    if(type2cluster[col_index_types[i]] == 'word' and char_emb):
                        temp_col_char = col_index_types[i].split('_')[0] + '_' + 'char'
                        if temp_col_char in input_types['char']['cols']:
                            input_type.append(temp_col_char)
                    if type2cluster[col_index_types[i]] == 'word' or type2cluster[col_index_types[i]] == 'bpe':
                        if self.lowercase:
                            line_split[i] = line_split[i].lower()
                        line_split[i] = self.text_preprocessor.preprocess(line_split[i])
                    if type2cluster[col_index_types[i]] == 'word':
                        if ProblemTypes[self.problem_type] == ProblemTypes.mrc:
                            token_offsets = self.tokenizer.span_tokenize(line_split[i])
                            tokens = [line_split[i][span[0]:span[1]] for span in token_offsets]
                            if branch == 'passage':
                                passage_token_offsets = token_offsets
                                data[extra_info_type]['extra_passage_text'].append(line_split[i])
                                data[extra_info_type]['extra_passage_token_offsets'].append(passage_token_offsets)
                        else:
                            if extra_feature == False:
                                tokens = self.tokenizer.tokenize(line_split[i])
                            else:
                                tokens = line_split[i].split(' ')
                    elif type2cluster[col_index_types[i]] == 'bpe':
                        tokens = bpe_encoder.encode(line_split[i])
                    else:
                        tokens = line_split[i].split(' ')

                    if fixed_lengths and type_branches[input_type[0]] in fixed_lengths:
                        if len(tokens) >= fixed_lengths[type_branches[input_type[0]]]:
                            tokens = tokens[:fixed_lengths[type_branches[input_type[0]]]]
                        else:
                            tokens = tokens + ['<pad>'] * (fixed_lengths[type_branches[input_type[0]]] - len(tokens))
                    else:
                        if max_lengths and type_branches[input_type[0]] in max_lengths:  # cut sequences which are too long
                            tokens = tokens[:max_lengths[type_branches[input_type[0]]]]

                    if len(tokens) < min_sentence_len:
                        tokens = tokens + ['<pad>'] * (min_sentence_len - len(tokens))

                    if self.with_bos_eos is True:
                        tokens = ['<start>'] + tokens + ['<eos>']  # so that source_with_start && source_with_end should be True

                    if not branch in length_appended_set:
                        lengths[branch]['sentence_length'].append(len(tokens))
                        length_appended_set.add(branch)
                    else:
                        if len(tokens) != lengths[branch]['sentence_length'][-1]:
                            # logging.warning(
                            #     "The length of inputs are not consistent. Ingore now. %s" % line)
                            cnt_illegal += 1
                            if cnt_illegal / cnt_all > 0.33:
                                raise PreprocessError("The illegal data is too much. Please check the number of data columns or text token version.")
                            lengths[branch]['sentence_length'].pop()
                            true_len = len(lengths[branch]['sentence_length'])
                            # need delete the last example
                            check_list = ['data', 'lengths', 'target']
                            for single_check in check_list:
                                single_check = eval(single_check)
                                self.delete_example(single_check, true_len)
                            break

                    for single_input_type in input_type:
                        if 'char' in single_input_type:
                            temp_word_char = []
                            temp_word_length = []
                            for single_token in tokens:
                                temp_word_char.append(self.input_dicts[type2cluster[single_input_type]].lookup(single_token))
                                temp_word_length.append(len(single_token))
                            data[branch][single_input_type].append(temp_word_char)
                            lengths[branch]['word_length'].append(temp_word_length)
                        else:
                            data[branch][single_input_type].\
                                append(self.input_dicts[type2cluster[single_input_type]].lookup(tokens))

                else:
                    # judge target
                    if answer_column_name is not None and len(answer_column_name) > 0:
                        if i in columns_to_target.keys():
                            # this is target
                            curr_target = columns_to_target[i]
                            if ProblemTypes[self.problem_type] == ProblemTypes.mrc:
                                try:
                                    trans2int = int(line_split[i])
                                except(ValueError):
                                    target[curr_target].append(line_split[i])
                                else:
                                    target[curr_target].append(trans2int)
                                lengths['target'][curr_target].append(1)
                            if ProblemTypes[self.problem_type] == ProblemTypes.sequence_tagging:
                                target_tags = line_split[i].split(" ")
                                if fixed_lengths and "target" in fixed_lengths:
                                    if len(target_tags) >= fixed_lengths[type_branches[input_type[0]]]:
                                        target_tags = target_tags[:fixed_lengths[type_branches[input_type[0]]]]
                                    else:
                                        target_tags = target_tags + ['<pad>'] * (fixed_lengths[type_branches[input_type[0]]] - len(target_tags))
                                else:
                                    if max_lengths and "target" in max_lengths:  # cut sequences which are too long
                                        target_tags = target_tags[:max_lengths["target"]]

                                if self.with_bos_eos is True:
                                    target_tags = ['O'] + target_tags + ['O']
                                target[curr_target].append(self.output_dict.lookup(target_tags))
                                lengths['target'][curr_target].append(len(target_tags))
                            elif ProblemTypes[self.problem_type] == ProblemTypes.classification:
                                target[curr_target].append(self.output_dict.id(line_split[i]))
                                lengths['target'][curr_target].append(1)
                            elif ProblemTypes[self.problem_type] == ProblemTypes.regression:
                                target[curr_target].append(float(line_split[i]))
                                lengths['target'][curr_target].append(1)
                        else:
                            # these columns are useless in the configuration
                            pass

            cnt_legal += 1
            if ProblemTypes[self.problem_type] == ProblemTypes.mrc and target is not None:
                if passage_token_offsets:
                    if 'start_label' not in target or 'end_label' not in target:
                        raise Exception('MRC task need start_label and end_label.')
                    start_char_label = target['start_label'][-1]
                    end_char_label = target['end_label'][-1]
                    start_word_label = 0
                    end_word_label = len(passage_token_offsets) - 1
                    # for i in range(len(passage_token_offsets)):
                    #     token_s, token_e = passage_token_offsets[i]
                    #     if token_s > start_char_label:
                    #         break
                    #     start_word_label = i
                    # for i in range(len(passage_token_offsets)):
                    #     token_s, token_e = passage_token_offsets[i]
                    #     end_word_label = i
                    #     if token_e >= end_char_label:
                    #         break
                    for i in range(len(passage_token_offsets)):
                        token_s, token_e = passage_token_offsets[i]
                        if token_s <= start_char_label <= token_e:
                            start_word_label = i
                        if token_s <= end_char_label - 1 <= token_e:
                            end_word_label = i
                    target['start_label'][-1] = start_word_label
                    target['end_label'][-1] = end_word_label
                else:
                    raise Exception('MRC task need passage.')

        return data, lengths, target, cnt_legal, cnt_illegal

    def encode(self, data_path, file_columns, input_types, file_with_col_header, object_inputs, answer_column_name,
               min_sentence_len, extra_feature, max_lengths=None, fixed_lengths=None, file_format="tsv", show_progress=True,
               cpu_num_workers = -1):
        """

        Args:
            data_path:
            file_columns: {
                  "word1": 0,
                  "word2": 1,
                  "label":   2,
                  "postag_feature1": 3,
                  "postag_feature2": 4
                },
            input_types:
                {
                  "word": {
                    "cols": [
                      "word1",
                      "word2"
                    ],
                    "dim": 300
                  },
                  "postag": {
                    "cols": ["postag_feature1", "postag_feature2"],
                    "dim": 20
                  }
                }
                or
                {
                  "bpe": {
                    "cols": ["word1", "word2"],
                    "dim": 100
                    "bpe_path": "xxx.bpe"
                  }
                }
            object_inputs: {
              "string1": [
                "word1",
                "postag_feature1"
              ],
              "string2": [
                "word2",
                "postag_feature2"
              ]
            },
            answer_column_name: 'label' / None. None means there is no target and it is used for prediction only.
            max_lengths: if it is a dict, firstly cut the sequences if they exceed the max length. Then, pad all the sequences to the length of longest string.
                {
                    "string1": 25,
                    "string2": 100
                }
            fixed_lengths: if it is a dict, cut or pad the sequences to the fixed lengths.
                {
                    "string1": 25,
                    "string2": 100
                }
            file_format:

        Returns:
            data: indices, padded
                {
                'string1': {
                    'word1': [...],
                    'postage_feature1': [..]
                    }
                'string2': {
                    'word1': [...],
                    'postage_feature1': [..]
                }
            lengths: real length of data
                {
                'string1':   [...],
                'string2':   [...]
                }
            target: [...]

        """
        if 'bpe' in input_types:
            try:
                bpe_encoder = BPEEncoder(input_types['bpe']['bpe_path'])
            except KeyError:
                raise Exception('Please define a bpe path at the embedding layer.')
        else:
            bpe_encoder = None

        with open(data_path, 'r', encoding='utf-8') as fin:
            progress = self.get_data_list_from_file(fin, file_with_col_header)
            data, lengths, target, cnt_legal, cnt_illegal = self.encode_data_multi_processor(progress, cpu_num_workers,
                    file_columns, input_types, object_inputs, answer_column_name, min_sentence_len, extra_feature, max_lengths,
                    fixed_lengths, file_format, bpe_encoder=bpe_encoder)
        logging.info("%s: %d legal samples, %d illegal samples" % (data_path, cnt_legal, cnt_illegal))
        return data, lengths, target

    def decode(self, model_output, lengths=None, batch_data=None):
        """ decode the model output, either a batch of output or a single output

        Args:
            model_output: target indices.
                if is 1d array, it is an output of a sample;
                if is 2d array, it is outputs of a batch of samples;
            lengths: if not None, the shape of length should be consistent with model_output.

        Returns:
            the original output

        """
        if ProblemTypes[self.problem_type] == ProblemTypes.classification:
            if isinstance(model_output, int):       # output of a sample
                return self.output_dict.cell(model_output)
            else:   # output of a batch
                return self.output_dict.decode(model_output)
        elif ProblemTypes[self.problem_type] == ProblemTypes.sequence_tagging:
            if isinstance(model_output, dict):
                model_output = list(model_output.values())[0]
            if not isinstance(model_output, np.ndarray):
                model_output = np.array(model_output)
            if len(model_output.shape) == 1:        # output of a sample
                if lengths is None:
                    outputs = np.array(self.output_dict.decode(model_output))
                else:
                    outputs = np.array(self.output_dict.decode(model_output[:lengths]))
                if self.with_bos_eos:
                    outputs = outputs[1:-1]

            elif len(model_output.shape) == 2:      # output of a batch of sequence
                outputs = []
                if lengths is None:
                    for sample in model_output:
                        if self.with_bos_eos:
                            outputs.append(self.output_dict.decode(sample[1:-1]))
                        else:
                            outputs.append(self.output_dict.decode(sample))
                else:
                    for sample, length in zip(model_output, lengths):
                        if self.with_bos_eos:
                            outputs.append(self.output_dict.decode(sample[:length][1:-1]))
                        else:
                            outputs.append(self.output_dict.decode(sample[:length]))
            return outputs
        elif ProblemTypes[self.problem_type] == ProblemTypes.mrc:
            # for mrc, model_output is dict
            answers = []
            p1, p2 = list(model_output.values())[0], list(model_output.values())[1]
            batch_size, c_len = p1.size()
            passage_length = lengths.numpy()
            padding_mask = np.ones((batch_size, c_len))
            for i, single_len in enumerate(passage_length):
                padding_mask[i][:single_len] = 0
            device = p1.device
            padding_mask = torch.from_numpy(padding_mask).byte().to(device)
            p1.data.masked_fill_(padding_mask.data, float('-inf'))
            p2.data.masked_fill_(padding_mask.data, float('-inf'))
            ls = nn.LogSoftmax(dim=1)
            mask = (torch.ones(c_len, c_len) * float('-inf')).to(device).tril(-1).unsqueeze(0).expand(batch_size, -1, -1)
            score = (ls(p1).unsqueeze(2) + ls(p2).unsqueeze(1)) + mask
            score, s_idx = score.max(dim=1)
            score, e_idx = score.max(dim=1)
            s_idx = torch.gather(s_idx, 1, e_idx.view(-1, 1)).squeeze()
            # encode mrc answer text
            passage_text = 'extra_passage_text'
            passage_token_offsets = 'extra_passage_token_offsets'
            for i in range(batch_size):
                char_s_idx, _ = batch_data[passage_token_offsets][i][s_idx[i]]
                _, char_e_idx = batch_data[passage_token_offsets][i][e_idx[i]]
                answer = batch_data[passage_text][i][char_s_idx:char_e_idx]
                answers.append(answer)
            return answers

    def get_vocab_sizes(self):
        """ get size of vocabs: including word embedding, postagging ...

        Returns:
            {
                'word':  xxx,
                'postag': xxx,
            }

        """
        vocab_sizes = dict()
        for input in self.input_dicts:
            vocab_sizes[input] = self.input_dicts[input].cell_num()
        return vocab_sizes

    def export_problem(self, save_path, ret_without_save=False):
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        problem = dict()
        for name, value in vars(self).items():
            if name.startswith("__") is False:
                if isinstance(value, CellDict):
                    problem[name] = value.export_cell_dict()
                else:
                    problem[name] = value

        if ret_without_save is False:
            with open(save_path, 'wb') as fout:
                pkl.dump(problem, fout, protocol=pkl.HIGHEST_PROTOCOL)
            logging.debug("Problem saved to %s" % save_path)
            return None
        else:
            return problem

    def load_problem(self, problem_path):
        info_dict = load_from_pkl(problem_path)
        for name in info_dict:
            if isinstance(getattr(self, name), CellDict):
                getattr(self, name).load_cell_dict(info_dict[name])

            else:
                setattr(self, name, info_dict[name])
            # the type of input_dicts is dict
            # elif name == 'input_dicts' and isinstance(getattr(self, name), type(info_dict[name])):
            #     setattr(self, name, info_dict[name])
        logging.debug("Problem loaded")

    def delete_example(self, data, true_len):
        if isinstance(data, list):
            if len(data)>true_len:
                data.pop()
        else:
            # data is dict
            for single_value in data.values():
                self.delete_example(single_value, true_len)
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn

import os
import time
import numpy as np
from tqdm import tqdm
import random
import codecs
import pickle as pkl

from utils.common_utils import dump_to_pkl, load_from_pkl, get_param_num, get_trainable_param_num, \
    transfer_to_gpu, transform_params2tensors
from utils.philly_utils import HDFSDirectTransferer, open_and_move, convert_to_tmppath, \
    convert_to_hdfspath, move_from_local_to_hdfs
from Model import Model
import logging
from metrics.Evaluator import Evaluator
from utils.corpus_utils import get_batches
from core.StreamingRecorder import StreamingRecorder
from core.LRScheduler import LRScheduler
from settings import ProblemTypes


class LearningMachine(object):
    def __init__(self, phase, conf, problem, vocab_info=None, initialize=True, use_gpu=False, **kwargs):
        if initialize is True:
            assert vocab_info is not None
            self.model = Model(conf, problem, vocab_info, use_gpu)
            if use_gpu is True:
                self.model = nn.DataParallel(self.model)
                self.model = transfer_to_gpu(self.model)
            logging.info(self.model)
            #logging.info("Total parameters: %d; trainable parameters: %d" % (get_param_num(self.model), get_trainable_param_num(self.model)))
            logging.info("Total trainable parameters: %d" % (get_trainable_param_num(self.model)))
            logging.info("Model built!")
        else:
            self.model = None

        self.conf = conf
        self.problem = problem
        self.phase = phase
        self.use_gpu = use_gpu

        # if it is a 2-class classification problem, figure out the real positive label
        # CAUTION: multi-class classification
        if phase != 'predict':
            if 'auc' in conf.metrics:
                if not hasattr(self.conf, 'pos_label') or self.conf.pos_label is None:
                    if problem.output_dict.cell_num() == 2 and \
                        problem.output_dict.has_cell("0") and problem.output_dict.has_cell("1"):
                        self.conf.pos_label = problem.output_dict.id("1")
                        logging.debug("Postive label (target index): %d" % self.conf.pos_label)
                    else:
                        # default
                        raise Exception('Please configure the positive label for auc metric at inputs/positive_label in the configuration file')
                else:
                    self.conf.pos_label = problem.output_dict.id(self.conf.pos_label)
            else:
                self.conf.pos_label = 1  # whatever

            self.metrics = conf.metrics
            if ProblemTypes[self.problem.problem_type] == ProblemTypes.classification \
                or ProblemTypes[self.problem.problem_type] == ProblemTypes.sequence_tagging:
                self.evaluator = Evaluator(metrics=self.metrics, pos_label=self.conf.pos_label, tagging_scheme=problem.tagging_scheme, label_indices=self.problem.output_dict.cell_id_map)
            elif ProblemTypes[self.problem.problem_type] == ProblemTypes.regression:
                self.evaluator = Evaluator(metrics=self.metrics, pos_label=self.conf.pos_label, tagging_scheme=problem.tagging_scheme, label_indices=None)
            elif ProblemTypes[self.problem.problem_type] == ProblemTypes.mrc:
                curr_mrc_metric = []
                for single_mrc_metric in self.metrics:
                    if 'mrc' in single_mrc_metric.lower():
                        curr_mrc_metric.append(single_mrc_metric.lower())
                    else:
                        curr_mrc_metric.append('mrc_' + single_mrc_metric.lower())
                self.evaluator = Evaluator(metrics=curr_mrc_metric, pos_label=self.conf.pos_label, tagging_scheme=problem.tagging_scheme, label_indices=None)
        self.use_gpu = use_gpu

        self.best_test_result = "(No best test result yet)"

    def train(self, optimizer, loss_fn):
        self.model.train()
        if not self.conf.train_data_path.endswith('.pkl'):
            train_data, train_length, train_target = self.problem.encode(self.conf.train_data_path, self.conf.file_columns,
                self.conf.input_types, self.conf.file_with_col_header, self.conf.object_inputs, self.conf.answer_column_name, max_lengths=self.conf.max_lengths,
                min_sentence_len = self.conf.min_sentence_len, extra_feature=self.conf.extra_feature,fixed_lengths=self.conf.fixed_lengths, file_format='tsv',
                show_progress=True if self.conf.mode == 'normal' else False, cpu_thread_num=self.conf.cpu_thread_num)
        else:
            train_pkl_data = load_from_pkl(self.conf.train_data_path)
            train_data, train_length, train_target = train_pkl_data['data'], train_pkl_data['length'], train_pkl_data['target']

        if not self.conf.valid_data_path.endswith('.pkl'):
            valid_data, valid_length, valid_target = self.problem.encode(self.conf.valid_data_path, self.conf.file_columns,
                self.conf.input_types, self.conf.file_with_col_header, self.conf.object_inputs, self.conf.answer_column_name, max_lengths=self.conf.max_lengths,
                min_sentence_len = self.conf.min_sentence_len, extra_feature = self.conf.extra_feature,fixed_lengths=self.conf.fixed_lengths, file_format='tsv',
                show_progress=True if self.conf.mode == 'normal' else False, cpu_thread_num=self.conf.cpu_thread_num)
        else:
            valid_pkl_data = load_from_pkl(self.conf.valid_data_path)
            valid_data, valid_length, valid_target = valid_pkl_data['data'], valid_pkl_data['length'], valid_pkl_data['target']

        if self.conf.test_data_path is not None:
            if not self.conf.test_data_path.endswith('.pkl'):
                test_data, test_length, test_target = self.problem.encode(self.conf.test_data_path, self.conf.file_columns, self.conf.input_types,
                    self.conf.file_with_col_header, self.conf.object_inputs, self.conf.answer_column_name, max_lengths=self.conf.max_lengths,
                    min_sentence_len = self.conf.min_sentence_len, extra_feature = self.conf.extra_feature,fixed_lengths=self.conf.fixed_lengths,
                    file_format='tsv', show_progress=True if self.conf.mode == 'normal' else False, cpu_thread_num=self.conf.cpu_thread_num)
            else:
                test_pkl_data = load_from_pkl(self.conf.test_data_path)
                test_data, test_length, test_target = test_pkl_data['data'], test_pkl_data['length'], test_pkl_data['target']

        stop_training = False
        epoch = 1
        best_result = None
        show_result_cnt = 0
        lr_scheduler = LRScheduler(optimizer, self.conf.lr_decay, self.conf.minimum_lr, self.conf.epoch_start_lr_decay)

        if ProblemTypes[self.problem.problem_type] == ProblemTypes.classification:
            streaming_recoder = StreamingRecorder(['prediction', 'pred_scores', 'pred_scores_all', 'target'])
        elif ProblemTypes[self.problem.problem_type] == ProblemTypes.sequence_tagging:
            streaming_recoder = StreamingRecorder(['prediction', 'pred_scores', 'target'])
        elif ProblemTypes[self.problem.problem_type] == ProblemTypes.regression:
            streaming_recoder = StreamingRecorder(['prediction', 'target'])
        elif ProblemTypes[self.problem.problem_type] == ProblemTypes.mrc:
            streaming_recoder = StreamingRecorder(['prediction', 'answer_text'])

        while not stop_training and epoch <= self.conf.max_epoch:
            logging.info('Training: Epoch ' + str(epoch))

            data_batches, length_batches, target_batches = \
                get_batches(self.problem, train_data, train_length, train_target, self.conf.batch_size_total,
                    self.conf.input_types, None, permutate=True, transform_tensor=True)

            whole_batch_num = len(target_batches)
            valid_batch_num = max(len(target_batches) // self.conf.valid_times_per_epoch, 1)
            if torch.cuda.device_count() > 1:
                small_batch_num = whole_batch_num * torch.cuda.device_count()       # total batch num over all the gpus
                valid_batch_num_show = valid_batch_num * torch.cuda.device_count()      # total batch num over all the gpus to do validation
            else:
                small_batch_num = whole_batch_num
                valid_batch_num_show = valid_batch_num

            streaming_recoder.clear_records()
            all_costs = []

            logging.info('There are %d batches during an epoch; validation are conducted every %d batch' % (small_batch_num, valid_batch_num_show))

            if self.conf.mode == 'normal':
                progress = tqdm(range(len(target_batches)))
            elif self.conf.mode == 'philly':
                progress = range(len(target_batches))
            for i in progress:
                # the result shape: for classification: [batch_size, # of classes]; for sequence tagging: [batch_size, seq_len, # of tags]
                param_list, inputs_desc, length_desc = transform_params2tensors(data_batches[i], length_batches[i])
                logits_softmax = self.model(inputs_desc, length_desc, *param_list)

                # check the output
                if ProblemTypes[self.problem.problem_type] == ProblemTypes.classification:
                    logits_softmax = list(logits_softmax.values())[0]
                    assert len(logits_softmax.shape) == 2, 'The dimension of your output is %s, but we need [batch_size*GPUs, class num]' % (str(list(logits_softmax.shape)))
                    assert logits_softmax.shape[1] == self.problem.output_target_num(), 'The dimension of your output layer %d is inconsistent with your type number %d!' % (logits_softmax.shape[1], self.problem.output_target_num())
                    # for auc metric
                    prediction_scores = logits_softmax[:, self.conf.pos_label].cpu().data.numpy()
                    if self.evaluator.has_auc_type_specific:
                        prediction_scores_all = logits_softmax.cpu().data.numpy()
                    else:
                        prediction_scores_all = None
                elif ProblemTypes[self.problem.problem_type] == ProblemTypes.sequence_tagging:
                    logits_softmax = list(logits_softmax.values())[0]
                    assert len(logits_softmax.shape) == 3, 'The dimension of your output is %s, but we need [batch_size*GPUs, sequence length, representation dim]' % (str(list(logits_softmax.shape)), )
                    prediction_scores = None
                    prediction_scores_all = None
                elif ProblemTypes[self.problem.problem_type] == ProblemTypes.regression:
                    logits_softmax = list(logits_softmax.values())[0]
                    assert len(logits_softmax.shape) == 2 and logits_softmax.shape[1] == 1, 'The dimension of your output is %s, but we need [batch_size*GPUs, 1]' % (str(list(logits_softmax.shape)))
                    prediction_scores = None
                    prediction_scores_all = None
                elif ProblemTypes[self.problem.problem_type] == ProblemTypes.mrc:
                    for single_value in logits_softmax.values():
                        assert len(single_value.shape) == 3, 'The dimension of your output is %s, but we need [batch_size*GPUs, sequence_len, 1]' % (str(list(single_value.shape)))
                    prediction_scores = None
                    prediction_scores_all = None

                logits_softmax_flat = dict()
                if ProblemTypes[self.problem.problem_type] == ProblemTypes.sequence_tagging:
                    # Transform output shapes for metric evaluation
                    # for seq_tag_f1 metric
                    prediction_indices = logits_softmax.data.max(2)[1].cpu().numpy()    # [batch_size, seq_len]
                    streaming_recoder.record_one_row([self.problem.decode(prediction_indices, length_batches[i]['target'][self.conf.answer_column_name[0]].numpy()),
                                                      prediction_scores, self.problem.decode(target_batches[i][self.conf.answer_column_name[0]],
                                                                                             length_batches[i]['target'][self.conf.answer_column_name[0]].numpy())], keep_dim=False)

                    # pytorch's CrossEntropyLoss only support this
                    logits_softmax_flat[self.conf.output_layer_id[0]] = logits_softmax.view(-1, logits_softmax.size(2))    # [batch_size * seq_len, # of tags]
                    #target_batches[i] = target_batches[i].view(-1)                      # [batch_size * seq_len]
                    # [batch_size * seq_len]
                    target_batches[i][self.conf.answer_column_name[0]] = target_batches[i][self.conf.answer_column_name[0]].reshape(-1)

                elif ProblemTypes[self.problem.problem_type] == ProblemTypes.classification:
                    prediction_indices = logits_softmax.detach().max(1)[1].cpu().numpy()
                    # Should not decode!
                    streaming_recoder.record_one_row([prediction_indices, prediction_scores, prediction_scores_all, target_batches[i][self.conf.answer_column_name[0]].numpy()])
                    logits_softmax_flat[self.conf.output_layer_id[0]] = logits_softmax
                elif ProblemTypes[self.problem.problem_type] == ProblemTypes.regression:
                    temp_logits_softmax_flat = logits_softmax.squeeze(1)
                    prediction_scores = temp_logits_softmax_flat.detach().cpu().numpy()
                    streaming_recoder.record_one_row([prediction_scores, target_batches[i][self.conf.answer_column_name[0]].numpy()])
                    logits_softmax_flat[self.conf.output_layer_id[0]] = temp_logits_softmax_flat
                elif ProblemTypes[self.problem.problem_type] == ProblemTypes.mrc:
                    for key, value in logits_softmax.items():
                        logits_softmax[key] = value.squeeze()
                    passage_identify = None
                    for type_key in data_batches[i].keys():
                        if 'p' in type_key.lower():
                            passage_identify = type_key
                            break
                    if not passage_identify:
                        raise Exception('MRC task need passage information.')
                    prediction = self.problem.decode(logits_softmax, lengths=length_batches[i][passage_identify],
                                                     batch_data=data_batches[i][passage_identify])
                    logits_softmax_flat = logits_softmax
                    mrc_answer_target = None
                    for single_target in target_batches[i]:
                        if isinstance(target_batches[i][single_target][0], str):
                            mrc_answer_target = target_batches[i][single_target]
                    streaming_recoder.record_one_row([prediction, mrc_answer_target])

                if self.use_gpu:
                    for single_target in self.conf.answer_column_name:
                        if isinstance(target_batches[i][single_target], torch.Tensor):
                            target_batches[i][single_target] = transfer_to_gpu(target_batches[i][single_target])
                loss = loss_fn(logits_softmax_flat, target_batches[i])

                all_costs.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.conf.clip_grad_norm_max_norm)
                optimizer.step()

                del loss, logits_softmax, logits_softmax_flat
                del prediction_scores
                if ProblemTypes[self.problem.problem_type] == ProblemTypes.sequence_tagging \
                        or ProblemTypes[self.problem.problem_type] == ProblemTypes.classification:
                    del prediction_indices

                if show_result_cnt == self.conf.batch_num_to_show_results:
                    if ProblemTypes[self.problem.problem_type] == ProblemTypes.classification:
                        result = self.evaluator.evaluate(streaming_recoder.get('target'),
                            streaming_recoder.get('prediction'), y_pred_pos_score=streaming_recoder.get('pred_scores'),
                            y_pred_scores_all=streaming_recoder.get('pred_scores_all'), formatting=True)
                    elif ProblemTypes[self.problem.problem_type] == ProblemTypes.sequence_tagging:
                        result = self.evaluator.evaluate(streaming_recoder.get('target'),
                            streaming_recoder.get('prediction'), y_pred_pos_score=streaming_recoder.get('pred_scores'),
                            formatting=True)
                    elif ProblemTypes[self.problem.problem_type] == ProblemTypes.regression:
                        result = self.evaluator.evaluate(streaming_recoder.get('target'),
                            streaming_recoder.get('prediction'), y_pred_pos_score=None, y_pred_scores_all=None, formatting=True)
                    elif ProblemTypes[self.problem.problem_type] == ProblemTypes.mrc:
                        result = self.evaluator.evaluate(streaming_recoder.get('answer_text'), streaming_recoder.get('prediction'),
                                                         y_pred_pos_score=None, y_pred_scores_all=None, formatting=True)

                    if torch.cuda.device_count() > 1:
                        logging.info("Epoch %d batch idx: %d; lr: %f; since last log, loss=%f; %s" % \
                            (epoch, i * torch.cuda.device_count(), lr_scheduler.get_lr(), np.mean(all_costs), result))
                    else:
                        logging.info("Epoch %d batch idx: %d; lr: %f; since last log, loss=%f; %s" % \
                            (epoch, i, lr_scheduler.get_lr(), np.mean(all_costs), result))
                    show_result_cnt = 0
                    # The loss and other metrics printed during a training epoch are just the result of part of the training data.
                    all_costs = []
                    streaming_recoder.clear_records()

                if (i != 0 and i % valid_batch_num == 0) or i == len(target_batches) - 1:
                    torch.cuda.empty_cache()    # actually useless
                    logging.info('Valid & Test : Epoch ' + str(epoch))
                    new_result = self.evaluate(valid_data, valid_length, valid_target,
                        self.conf.input_types, self.evaluator, loss_fn, pad_ids=None, cur_best_result=best_result,
                        model_save_path=self.conf.model_save_path, phase="valid", epoch=epoch)
                    renew_flag = best_result != new_result
                    best_result = new_result

                    if renew_flag and self.conf.test_data_path is not None:
                        self.evaluate(test_data, test_length, test_target,
                            self.conf.input_types, self.evaluator, loss_fn, pad_ids=None, phase="test", epoch=epoch)
                    self.model.train()
                show_result_cnt += 1

            del data_batches, length_batches, target_batches
            lr_scheduler.step()
            epoch += 1

    def test(self, loss_fn, test_data_path=None, predict_output_path=None):
        if test_data_path is None:
            # test_data_path in the parameter is prior to self.conf.test_data_path
            test_data_path = self.conf.test_data_path

        if not test_data_path.endswith('.pkl'):
            test_data, test_length, test_target = self.problem.encode(test_data_path, self.conf.file_columns, self.conf.input_types,
                self.conf.file_with_col_header, self.conf.object_inputs, self.conf.answer_column_name, max_lengths=self.conf.max_lengths,
                min_sentence_len = self.conf.min_sentence_len, extra_feature = self.conf.extra_feature,fixed_lengths=self.conf.fixed_lengths, file_format='tsv',
                show_progress=True if self.conf.mode == 'normal' else False, cpu_thread_num=self.conf.cpu_thread_num)
        else:
            test_pkl_data = load_from_pkl(test_data_path)
            test_data, test_length, test_target = test_pkl_data['data'], test_pkl_data['length'], test_pkl_data['target']

        if not predict_output_path:
            self.evaluate(test_data, test_length, test_target,
                self.conf.input_types, self.evaluator, loss_fn, pad_ids=None, phase="test")
        else:
            self.evaluate(test_data, test_length, test_target,
                self.conf.input_types, self.evaluator, loss_fn, pad_ids=None, phase="test",
                origin_data_path=test_data_path, predict_output_path=predict_output_path)

    def evaluate(self, data, length, target, input_types, evaluator,
                 loss_fn, pad_ids=None, cur_best_result=None, model_save_path=None, phase="", epoch=None, origin_data_path=None, predict_output_path=None):
        """

        Args:
            qp_net:
            epoch:
            data:
                    {
                    'string1': {
                        'word1': [...],
                        'postage_feature1': [..]
                        }
                    'string2': {
                        'word1': [...],
                        'postage_feature1': [..]
                    }
            lengths:
                    {
                    'string1':   [...],
                    'string2':   [...]
                    }
            target:  [...]
            input_types:  {
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
            origin_data_path:
            predict_output_path: if predict_output_path exists, output the prediction result.

        Returns:

        """
        assert not (predict_output_path and not origin_data_path)
        if predict_output_path:
            to_predict = True
        else:
            to_predict = False

        logging.info("Starting %s ..." % phase)
        self.model.eval()
        with torch.no_grad():
            data_batches, length_batches, target_batches = \
                get_batches(self.problem, data, length, target, self.conf.batch_size_total, input_types, pad_ids, permutate=False, transform_tensor=True)

            if ProblemTypes[self.problem.problem_type] == ProblemTypes.classification:
                streaming_recoder = StreamingRecorder(['prediction', 'pred_scores', 'pred_scores_all', 'target'])
            elif ProblemTypes[self.problem.problem_type] == ProblemTypes.sequence_tagging:
                streaming_recoder = StreamingRecorder(['prediction', 'pred_scores', 'target'])
            elif ProblemTypes[self.problem.problem_type] == ProblemTypes.regression:
                streaming_recoder = StreamingRecorder(['prediction', 'target'])
            elif ProblemTypes[self.problem.problem_type] == ProblemTypes.mrc:
                streaming_recoder = StreamingRecorder(['prediction', 'answer_text'])

            if to_predict:
                predict_stream_recoder = StreamingRecorder(self.conf.predict_fields)
                fin = open(origin_data_path, 'r', encoding='utf-8')
                if predict_output_path.startswith('/hdfs/'):
                    direct_hdfs_path = convert_to_hdfspath(predict_output_path)
                    local_tmp_path = convert_to_tmppath(predict_output_path)
                    fout = open(local_tmp_path, 'w', encoding='utf-8')
                else:
                    direct_hdfs_path = None
                    fout = open(predict_output_path, 'w', encoding='utf-8')
                if self.conf.file_with_col_header:
                    title_line = fin.readline()
                    fout.write(title_line)

            temp_key_list = list(length_batches[0].keys())
            if 'target' in temp_key_list:
                temp_key_list.remove('target')
            key_random = random.choice(temp_key_list)
            loss_recoder = StreamingRecorder(['loss'])
            if self.conf.mode == 'normal':
                progress = tqdm(range(len(target_batches)))
            elif self.conf.mode == 'philly':
                progress = range(len(target_batches))
            for i in progress:
                # batch_size_actual = target_batches[i].size(0)

                param_list, inputs_desc, length_desc = transform_params2tensors(data_batches[i], length_batches[i])
                logits_softmax = self.model(inputs_desc, length_desc, *param_list)

                if ProblemTypes[self.problem.problem_type] == ProblemTypes.classification:
                    logits_softmax = list(logits_softmax.values())[0]
                    # for auc metric
                    prediction_pos_scores = logits_softmax[:, self.conf.pos_label].cpu().data.numpy()
                    if self.evaluator.has_auc_type_specific:
                        prediction_scores_all = logits_softmax.cpu().data.numpy()
                    else:
                        prediction_scores_all = None
                else:
                    prediction_pos_scores = None
                    prediction_scores_all = None

                logits_softmax_flat = {}
                if ProblemTypes[self.problem.problem_type] == ProblemTypes.sequence_tagging:
                    logits_softmax = list(logits_softmax.values())[0]
                    # Transform output shapes for metric evaluation
                    # for seq_tag_f1 metric
                    prediction_indices = logits_softmax.data.max(2)[1].cpu().numpy()  # [batch_size, seq_len]
                    streaming_recoder.record_one_row(
                        [self.problem.decode(prediction_indices, length_batches[i]['target'][self.conf.answer_column_name[0]].numpy()), prediction_pos_scores,
                         self.problem.decode(target_batches[i], length_batches[i]['target'][self.conf.answer_column_name[0]].numpy())], keep_dim=False)

                    # pytorch's CrossEntropyLoss only support this
                    logits_softmax_flat[self.conf.output_layer_id[0]] = logits_softmax.view(-1, logits_softmax.size(2))  # [batch_size * seq_len, # of tags]
                    #target_batches[i] = target_batches[i].view(-1)  # [batch_size * seq_len]
                    target_batches[i][self.conf.answer_column_name[0]] = target_batches[i][self.conf.answer_column_name[0]].reshape(-1)  # [batch_size * seq_len]

                    if to_predict:
                        prediction_batch = self.problem.decode(prediction_indices, length_batches[i][key_random].numpy())
                        for prediction_sample in prediction_batch:
                            predict_stream_recoder.record('prediction', " ".join(prediction_sample))

                elif ProblemTypes[self.problem.problem_type] == ProblemTypes.classification:
                    prediction_indices = logits_softmax.data.max(1)[1].cpu().numpy()
                    # Should not decode!
                    streaming_recoder.record_one_row([prediction_indices, prediction_pos_scores, prediction_scores_all, target_batches[i][self.conf.answer_column_name[0]].numpy()])
                    logits_softmax_flat[self.conf.output_layer_id[0]] = logits_softmax

                    if to_predict:
                        for field in self.conf.predict_fields:
                            if field == 'prediction':
                                predict_stream_recoder.record(field, self.problem.decode(prediction_indices, length_batches[i][key_random].numpy()))
                            elif field == 'confidence':
                                prediction_scores = logits_softmax.cpu().data.numpy()
                                for prediction_score, prediction_idx in zip(prediction_scores, prediction_indices):
                                    predict_stream_recoder.record(field, prediction_score[prediction_idx])
                            elif field.startswith('confidence') and field.find('@') != -1:
                                label_specified = field.split('@')[1]
                                label_specified_idx = self.problem.output_dict.id(label_specified)
                                confidence_specified = torch.index_select(logits_softmax.cpu(), 1, torch.tensor([label_specified_idx], dtype=torch.long)).squeeze(1)
                                predict_stream_recoder.record(field, confidence_specified.data.numpy())

                elif ProblemTypes[self.problem.problem_type] == ProblemTypes.regression:
                    logits_softmax = list(logits_softmax.values())[0]
                    temp_logits_softmax_flat = logits_softmax.squeeze(1)
                    prediction_scores = temp_logits_softmax_flat.detach().cpu().numpy()
                    streaming_recoder.record_one_row([prediction_scores, target_batches[i][self.conf.answer_column_name[0]].numpy()])
                    logits_softmax_flat[self.conf.output_layer_id[0]] = temp_logits_softmax_flat
                    if to_predict:
                        predict_stream_recoder.record_one_row([prediction_scores])

                elif ProblemTypes[self.problem.problem_type] == ProblemTypes.mrc:
                    for key, value in logits_softmax.items():
                        logits_softmax[key] = value.squeeze()
                    passage_identify = None
                    for type_key in data_batches[i].keys():
                        if 'p' in type_key.lower():
                            passage_identify = type_key
                            break
                    if not passage_identify:
                        raise Exception('MRC task need passage information.')
                    prediction = self.problem.decode(logits_softmax, lengths=length_batches[i][passage_identify],
                                                     batch_data=data_batches[i][passage_identify])
                    logits_softmax_flat = logits_softmax
                    mrc_answer_target = None
                    for single_target in target_batches[i]:
                        if isinstance(target_batches[i][single_target][0], str):
                            mrc_answer_target = target_batches[i][single_target]
                    streaming_recoder.record_one_row([prediction, mrc_answer_target])

                    if to_predict:
                        predict_stream_recoder.record_one_row([prediction])

                if to_predict:
                    logits_softmax_len = len(list(logits_softmax.values())[0]) \
                        if ProblemTypes[self.problem.problem_type] == ProblemTypes.mrc else len(logits_softmax)
                    for sample_idx in range(logits_softmax_len):
                        while True:
                            sample = fin.readline().rstrip()
                            line_split = list(filter(lambda x: len(x) > 0, sample.rstrip().split('\t')))
                            if self.problem.file_column_num is None or len(line_split) == self.problem.file_column_num:
                                break

                        fout.write("%s\t%s\n" % (sample,
                            "\t".join([str(predict_stream_recoder.get(field)[sample_idx]) for field in self.conf.predict_fields])))
                    predict_stream_recoder.clear_records()

                if self.use_gpu:
                    for single_target in self.conf.answer_column_name:
                        if isinstance(target_batches[i][single_target], torch.Tensor):
                            target_batches[i][single_target] = transfer_to_gpu(target_batches[i][single_target])
                loss = loss_fn(logits_softmax_flat, target_batches[i])
                loss_recoder.record('loss', loss.item())

                del loss, logits_softmax, logits_softmax_flat
                del prediction_pos_scores
                if ProblemTypes[self.problem.problem_type] == ProblemTypes.sequence_tagging or ProblemTypes[self.problem.problem_type] == ProblemTypes.classification:
                    del prediction_indices

            del data_batches, length_batches, target_batches

            if ProblemTypes[self.problem.problem_type] == ProblemTypes.classification:
                result = self.evaluator.evaluate(streaming_recoder.get('target'), streaming_recoder.get('prediction'),
                    y_pred_pos_score=streaming_recoder.get('pred_scores'),
                    y_pred_scores_all=streaming_recoder.get('pred_scores_all'), formatting=True)
            elif ProblemTypes[self.problem.problem_type] == ProblemTypes.sequence_tagging:
                result = self.evaluator.evaluate(streaming_recoder.get('target'), streaming_recoder.get('prediction'), y_pred_pos_score=streaming_recoder.get('pred_scores'), formatting=True)
            elif ProblemTypes[self.problem.problem_type] == ProblemTypes.regression:
                result = self.evaluator.evaluate(streaming_recoder.get('target'), streaming_recoder.get('prediction'), y_pred_pos_score=None, formatting=True)
            elif ProblemTypes[self.problem.problem_type] == ProblemTypes.mrc:
                result = self.evaluator.evaluate(streaming_recoder.get('answer_text'), streaming_recoder.get('prediction'),
                                                 y_pred_pos_score=None, y_pred_scores_all=None, formatting=True)

            if epoch:
                logging.info("Epoch %d, %s %s loss: %f" % (epoch, phase, result, loss_recoder.get('loss', 'mean')))
            else:
                logging.info("%s %s loss: %f" % (phase, result, loss_recoder.get('loss', 'mean')))

            if phase == 'valid':
                cur_result = evaluator.get_first_metric_result()
                if self.evaluator.compare(cur_result, cur_best_result) == 1:
                    logging.info(
                        'Cur result %f is better than previous best result %s, renew the best model now...' % (cur_result, "%f" % cur_best_result if cur_best_result else "None"))
                    if model_save_path is not None:
                        if self.conf.mode == 'philly' and model_save_path.startswith('/hdfs/'):
                            with HDFSDirectTransferer(model_save_path, with_hdfs_command=True) as transferer:
                                if isinstance(self.model, nn.DataParallel):
                                    transferer.torch_save(self.model.module)
                                else:
                                    transferer.torch_save(self.model)
                        else:
                            if not os.path.exists(os.path.dirname(model_save_path)):
                                os.makedirs(os.path.dirname(model_save_path))
                            if isinstance(self.model, nn.DataParallel):
                                torch.save(self.model.module, model_save_path, pickle_protocol=pkl.HIGHEST_PROTOCOL)
                            else:
                                torch.save(self.model, model_save_path, pickle_protocol=pkl.HIGHEST_PROTOCOL)
                        logging.info("Best model saved to %s" % model_save_path)
                    cur_best_result = cur_result
                else:
                    logging.info('Cur result %f is no better than previous best result %f' % (cur_result, cur_best_result))

        if to_predict:
            fin.close()
            fout.close()
            if direct_hdfs_path:
                move_from_local_to_hdfs(local_tmp_path, direct_hdfs_path)

        return cur_best_result

    def predict(self, predict_data_path, output_path, file_columns, predict_fields=['prediction']):
        """ prediction

        Args:
            predict_data_path:
            predict_fields: default: only prediction. For classification and regression tasks, prediction_confidence is also supported.

        Returns:

        """
        if predict_data_path is None:
            predict_data_path = self.conf.predict_data_path

        predict_data, predict_length, _ = self.problem.encode(predict_data_path, file_columns, self.conf.input_types,
            self.conf.file_with_col_header,self.conf.object_inputs, None, min_sentence_len=self.conf.min_sentence_len,
            extra_feature=self.conf.extra_feature,max_lengths=self.conf.max_lengths, fixed_lengths=self.conf.fixed_lengths,
            file_format='tsv', show_progress=True if self.conf.mode == 'normal' else False, 
            cpu_thread_num=self.conf.cpu_thread_num)

        logging.info("Starting predict ...")
        self.model.eval()
        with torch.no_grad():
            data_batches, length_batches, _ = \
                get_batches(self.problem, predict_data, predict_length, None, self.conf.batch_size_total,
                    self.conf.input_types, None, permutate=False, transform_tensor=True)

            streaming_recoder = StreamingRecorder(predict_fields)

            fin = open(predict_data_path, 'r', encoding='utf-8')
            with open_and_move(output_path) as fout:
                if self.conf.file_with_col_header:
                    title_line = fin.readline()
                    fout.write(title_line)
                key_random = random.choice(list(length_batches[0].keys()).remove('target') if 'target' in list(length_batches[0].keys()) else list(length_batches[0].keys()))
                if self.conf.mode == 'normal':
                    progress = tqdm(range(len(data_batches)))
                elif self.conf.mode == 'philly':
                    progress = range(len(data_batches))
                for i in progress:
                    # batch_size_actual = target_batches[i].size(0)
                    param_list, inputs_desc, length_desc = transform_params2tensors(data_batches[i], length_batches[i])
                    logits_softmax = self.model(inputs_desc, length_desc, *param_list)

                    if ProblemTypes[self.problem.problem_type] == ProblemTypes.sequence_tagging:
                        logits_softmax = list(logits_softmax.values())[0]
                        # Transform output shapes for metric evaluation
                        prediction_indices = logits_softmax.data.max(2)[1].cpu().numpy()  # [batch_size, seq_len]
                        prediction_batch = self.problem.decode(prediction_indices, length_batches[i][key_random].numpy())
                        for prediction_sample in prediction_batch:
                            streaming_recoder.record('prediction', " ".join(prediction_sample))
                    elif ProblemTypes[self.problem.problem_type] == ProblemTypes.classification:
                        logits_softmax = list(logits_softmax.values())[0]
                        prediction_indices = logits_softmax.data.max(1)[1].cpu().numpy()

                        for field in predict_fields:
                            if field == 'prediction':
                                streaming_recoder.record(field,
                                    self.problem.decode(prediction_indices, length_batches[i][key_random].numpy()))
                            elif field == 'confidence':
                                prediction_scores = logits_softmax.cpu().data.numpy()
                                for prediction_score, prediction_idx in zip(prediction_scores, prediction_indices):
                                    streaming_recoder.record(field, prediction_score[prediction_idx])
                            elif field.startswith('confidence') and field.find('@') != -1:
                                label_specified = field.split('@')[1]
                                label_specified_idx = self.problem.output_dict.id(label_specified)
                                confidence_specified = torch.index_select(logits_softmax.cpu(), 1,
                                        torch.tensor([label_specified_idx], dtype=torch.long)).squeeze(1)
                                streaming_recoder.record(field, confidence_specified.data.numpy())
                    elif ProblemTypes[self.problem.problem_type] == ProblemTypes.regression:
                        logits_softmax = list(logits_softmax.values())[0]
                        logits_softmax_flat = logits_softmax.squeeze(1)
                        prediction_scores = logits_softmax_flat.detach().cpu().numpy()
                        streaming_recoder.record_one_row([prediction_scores])
                    elif ProblemTypes[self.problem.problem_type] == ProblemTypes.mrc:
                        for key, value in logits_softmax.items():
                            logits_softmax[key] = value.squeeze()
                        passage_identify = None
                        for type_key in data_batches[i].keys():
                            if 'p' in type_key.lower():
                                passage_identify = type_key
                                break
                        if not passage_identify:
                            raise Exception('MRC task need passage information.')
                        prediction = self.problem.decode(logits_softmax, lengths=length_batches[i][passage_identify],
                                                         batch_data=data_batches[i][passage_identify])
                        streaming_recoder.record_one_row([prediction])

                    logits_softmax_len = len(list(logits_softmax.values())[0]) \
                        if ProblemTypes[self.problem.problem_type] == ProblemTypes.mrc else len(logits_softmax)
                    for sample_idx in range(logits_softmax_len):
                        sample = fin.readline().rstrip()
                        fout.write("%s\t%s\n" % (sample,
                            "\t".join([str(streaming_recoder.get(field)[sample_idx]) for field in predict_fields])))
                    streaming_recoder.clear_records()

                    del logits_softmax

        fin.close()

    def load_model(self, model_path):
        if self.use_gpu is True:
            self.model = torch.load(model_path)
            if isinstance(self.model, nn.DataParallel):
                self.model = self.model.module
            self.model = nn.DataParallel(self.model)
        else:
            self.model = torch.load(model_path, map_location='cpu')
            if isinstance(self.model, nn.DataParallel):
                self.model = self.model.module
            self.model.use_gpu = False

        logging.info("Model %s loaded!" % model_path)
        logging.info("Total trainable parameters: %d" % (get_trainable_param_num(self.model)))




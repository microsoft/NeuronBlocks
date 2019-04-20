# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from sklearn import metrics
from sklearn.metrics import mean_squared_error
from .conlleval import countChunks, evaluate, to_conll_format
from settings import TaggingSchemes
import numpy as np
import re
import string
from collections import Counter


class Evaluator(object):
    def __init__(self, metrics, pos_label=1, first_metric=None, tagging_scheme=None, label_indices=None):
        """

        Args:
            metrics:
            pos_label: the positive label for auc metric
            first_metric:
            tagging_scheme:
            label_indices: label to index dictionary, for auc@average or auc@some_type metric
        """
        self.__metrics = metrics
        self.__pos_label = pos_label
        if first_metric is None:
            self.__first_metric = metrics[0]
        else:
            self.__first_metric = first_metric
        self.__tagging_scheme = tagging_scheme
        self.__label_indices = label_indices

        self.has_auc_type_specific = False      # if True, the recorder needs to record the pred score of all types
        supported_metrics = self.get_supported_metrics()
        for metric in metrics:
            if not metric in supported_metrics:
                if metric.find('@') != -1:
                    field, target = metric.split('@')
                    if field != 'auc' or (self.__label_indices and (not target in self.__label_indices) and target != 'average'):
                        raise Exception("The metric %s is not supported. Supported metrics are: %s" % (metric, supported_metrics))
                    else:
                        self.has_auc_type_specific = True

    def evaluate(self, y_true, y_pred, y_pred_pos_score=None, y_pred_scores_all=None, formatting=False):
        """ evalution

        Args:
            y_true:
            y_pred:
            y_pred_pos_score:
            formatting:

        Returns:

        """
        result = dict()

        for metric in self.__metrics:
            if metric == 'auc':
                result[metric] = getattr(self, metric)(y_true, y_pred_pos_score)
            elif metric.startswith('auc@'):
                field, target = metric.split('@')
                if target == 'average':
                    results = []
                    for i in range(len(y_pred_scores_all[0])):
                        results.append(self.auc(y_true, np.array(y_pred_scores_all)[:, i]))
                    result[metric] = np.mean(results)
                else:
                    result[metric] = self.auc(y_true, np.array(y_pred_scores_all)[:, self.__label_indices[target]])
            else:
                result[metric] = getattr(self, metric)(y_true, y_pred)

        self.__last_result = result
        if formatting is True:
            ret = self.format_result(result)
        else:
            ret = result
        return ret

    def compare(self, current_result, previous_result, metric=None):
        """

        Args:
            current_result:
            previous_result:
            metric:

        Returns:
            current better than previous: 1
            current worse than previous: -1
            current equal to previous: 0

        """
        if previous_result is None:
            return 1

        if metric is None:
            metric = self.__first_metric

        # by default, metrics are the bigger, the better
        small_better_metrics = set(['MSE', 'RMSE'])

        if not metric in small_better_metrics:
            if current_result > previous_result:
                return 1
            elif current_result < previous_result:
                return -1
            else:
                return 0
        else:
            if current_result > previous_result:
                return -1
            elif current_result < previous_result:
                return 1
            else:
                return 0

    def get_first_metric_result(self):
        return self.__last_result[self.__first_metric]

    def get_supported_metrics(self):
        except_methods = ["evaluate", "format_result", "get_supported_metrics", "get_first_metric_result", "normalize_answer"]
        supported_metrics = []
        for name in dir(self):
            if callable(getattr(self, name)) and name.startswith("_") is False and not name in except_methods:
                supported_metrics.append(name)
        return supported_metrics

    def format_result(self, result):
        return "; ".join(["%s: %.6f" % (metric, result[metric]) for metric in self.__metrics])

    def auc(self, y_true, y_pred_pos_score):
        assert y_pred_pos_score is not None, "Prediction confidence of positive label should not be None for auc metric!"
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred_pos_score, pos_label=self.__pos_label)
        return metrics.auc(fpr, tpr)

    def accuracy(self, y_true, y_pred):
        return metrics.accuracy_score(y_true, y_pred)

    def seq_tag_f1(self, y_true, y_pred):
        """ For sequence tagging task, calculate F1-score(e.g. CONLL 2000)

        Args:
            y_true:
            y_pred:

        Returns:

        """
        assert self.__tagging_scheme is not None, "Please define tagging scheme!"
        if TaggingSchemes[self.__tagging_scheme] == TaggingSchemes.BIO:
            result_conll_format = to_conll_format(y_true, y_pred)
            correctChunk, foundGuessed, foundCorrect, correctTags, tokenCounter = countChunks(result_conll_format)
            overall_precision, overall_recall, overall_FB1 = evaluate(correctChunk, foundGuessed, foundCorrect, correctTags, tokenCounter)
        else:
            raise Exception("TO DO: SUPPORT MORE TAGGING SCHEMES")
        return overall_FB1

    def macro_f1(self, y_true, y_pred):
        """ For classification task, calculate f1-score for each label, and find their unweighted mean. This does not take label imbalance into account.

        Args:
            y_true:
            y_pred:

        Returns:

        """
        return metrics.f1_score(y_true, y_pred, average='macro')

    def macro_precision(self, y_true, y_pred):
        """ Calculate precision for each label, and find their unweighted mean. This does not take label imbalance into account.

        Args:
            y_true:
            y_pred:

        Returns:

        """
        return metrics.precision_score(y_true, y_pred, average='macro')

    def macro_recall(self, y_true, y_pred):
        """ Calculate recall for each label, and find their unweighted mean. This does not take label imbalance into account.

        Args:
            y_true:
            y_pred:

        Returns:

        """
        return metrics.recall_score(y_true, y_pred, average='macro')

    def micro_f1(self, y_true, y_pred):
        """ For classification task, calculate f1-score globally by counting the total true positives, false negatives and false positives.

        Args:
            y_true:
            y_pred:

        Returns:

        """
        return metrics.f1_score(y_true, y_pred, average='micro')

    def f1(self, y_true, y_pred):
        """ For classification task, calculate f1-score Only report results for the class specified by pos_label. This is applicable only if targets (y_{true,pred}) are binary..

        Args:
            y_true:
            y_pred:

        Returns:

        """
        return metrics.f1_score(y_true, y_pred)

    def micro_precision(self, y_true, y_pred):
        """ Calculate precision globally by counting the total true positives, false negatives and false positives.

        Args:
            y_true:
            y_pred:

        Returns:

        """
        return metrics.precision_score(y_true, y_pred, average='micro')

    def micro_recall(self, y_true, y_pred):
        """ Calculate recall globally by counting the total true positives, false negatives and false positives.

        Args:
            y_true:
            y_pred:

        Returns:

        """
        return metrics.recall_score(y_true, y_pred, average='micro')

    def weighted_f1(self, y_true, y_pred):
        """ Calculate metrics for each label, and find their average, weighted by support (the number of true instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.

        Args:
            y_true:
            y_pred:

        Returns:

        """
        return metrics.f1_score(y_true, y_pred, average='weighted')

    def weighted_precision(self, y_true, y_pred):
        """ Calculate metrics for each label, and find their average, weighted by support (the number of true instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.

        Args:
            y_true:
            y_pred:

        Returns:

        """
        return metrics.precision_score(y_true, y_pred, average='weighted')

    def weighted_recall(self, y_true, y_pred):
        """ Calculate metrics for each label, and find their average, weighted by support (the number of true instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.

        Args:
            y_true:
            y_pred:

        Returns:

        """
        return metrics.recall_score(y_true, y_pred, average='weighted')

    def MSE(self, y_true, y_pred):
        """ mean square error

        Args:
            y_true: true score
            y_pred: predict score

        Returns:

        """
        return mean_squared_error(y_true, y_pred)

    def RMSE(self, y_true, y_pred):
        """ root mean square error

        Args:
            y_true: true score
            y_pred: predict score

        Returns:

        """
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def normalize_answer(self, s):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def mrc_f1(self, y_true, y_pred):
        '''
        compute mrc task metric F1
        :param y_true: type list. ground thruth answer text
        :param y_pred: type list. length is same as y_true, model output answer text.
        :return: mrc task F1 score
        '''
        f1 = total = 0
        for single_true, single_pred in zip(y_true, y_pred):
            total += 1
            prediction_tokens = self.normalize_answer(single_pred).split()
            ground_truth_tokens = self.normalize_answer(single_true).split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 += (2*precision*recall) / (precision+recall)
        return 100.0 * f1 / total

    def mrc_em(self, y_true, y_pred):
        '''
        compute mrc task metric EM
        :param y_true:
        :param y_pred:
        :return: mrc task EM score
        '''
        em = total = 0
        for single_true, single_pred in zip(y_true, y_pred):
            total += 1
            em += (self.normalize_answer(single_true) == self.normalize_answer(single_pred))
        return 100.0 * em / total


if __name__ == '__main__':
    evaluator = Evaluator(['auc', 'accuracy'])
    print(evaluator.get_supported_metrics())


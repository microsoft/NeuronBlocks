# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

# add the project root to python path
import os
import sys
sys.path.append(sys.path[0])
from enum import Enum
import nltk


version = '1.1.0'


# Supported problems
ProblemTypes = Enum('ProblemTypes', ('sequence_tagging', 'classification', 'regression', 'mrc'))

# Supported sequence tagging scheme
TaggingSchemes = Enum('TaggingSchemes', ('BIO', 'BIOES'))

# supported metrics
SupportedMetrics = {
    ProblemTypes.sequence_tagging: set(['seq_tag_f1', 'seq_tag_accuracy']),
    ProblemTypes.classification: set(['auc', 'accuracy', 'f1', 'macro_f1', 'macro_precision', 'macro_recall', 'micro_f1', 'micro_precision', 'micro_recall', 'weighted_f1', 'weighted_precision', 'weighted_recall']),
    # In addition, for auc in multi-type classification,
    # if there is a type named 1, auc@1 means use 1 as the positive label
    # auc@average means enumerate all the types as the positive label and obtain the average auc.
    ProblemTypes.regression: set(['MSE', 'RMSE']),
    ProblemTypes.mrc :set(['f1', 'em']),
}

# Supported prediction types
PredictionTypes = {
    ProblemTypes.sequence_tagging: set(['prediction']),
    ProblemTypes.classification: set(['prediction', 'confidence']),     # In addition, if there is a type named positive, confidence@positive means the confidence of positive
    ProblemTypes.regression: set(['prediction']),
    ProblemTypes.mrc: set(['prediction']),
}

# Supported multi_loss operation
LossOperationType = Enum('LossOperationType', ('weighted_sum'))

# If prediction_field is not defined, use the default fields below
DefaultPredictionFields = {
    ProblemTypes.sequence_tagging: ['prediction'],
    ProblemTypes.classification: ['prediction', 'confidence'],
    ProblemTypes.regression: ['prediction'],
    ProblemTypes.mrc: ['prediction'],
}

# nltk's models
nltk.data.path.append(os.path.join(os.getcwd(), 'dataset', 'nltk_data'))


# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import argparse
from sklearn.metrics import roc_auc_score

def read_tsv(params):
    prediction, label = [], []
    predict_index, label_index = int(params.predict_index), int(params.label_index)
    min_column_num = max(predict_index, label_index) + 1
    with open(params.input_file, mode='r', encoding='utf-8') as f:
        for index, line in enumerate(f):
            if params.header and index == 0:
                continue
            line = line.rstrip()
            # skip empty line
            if not line:
                continue
            line = line.split('\t')
            if len(line) < min_column_num:
                print("at line:%s, %s"%(predict_index, line))
                raise Exception("the given index of predict or label is exceed the index of the column")
            prediction.append(float(line[predict_index]))
            label.append(int(line[label_index]))
    return prediction, label
            
def calculate_AUC(prediction, label):
    return roc_auc_score(label, prediction)

def main(params):
    prediction, label = read_tsv(params)
    auc = calculate_AUC(prediction, label)
    print("AUC is ", auc)
    return auc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AUC")
    parser.add_argument("--input_file", type=str, help="tsv file")
    parser.add_argument("--predict_index", type=str, help="the column index of prediction of model, start from 0")
    parser.add_argument("--label_index", type=str, help="the column index of label, start from 0")
    parser.add_argument("--header", action='store_true', default=False, help="whether contains header row or not, default is False")

    params, _ = parser.parse_known_args()

    assert params.input_file, 'Please specify a input file via --input_file'
    assert params.predict_index, 'Please specify the column index of prediction via --predict_index'
    assert params.label_index, 'Please specify the column index of label via --label_index'
    main(params)
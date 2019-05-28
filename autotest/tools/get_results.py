# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import re
from calculate_AUC import main

base_dir = './autotest/models'
task_dir = ['/20_newsgroup_bilstm_attn', '/chinese_text_matching', '/question_pairs_bilstm_attn']

results = {'english_text_matching': [0.96655], 'chinese_text_matching': [0.70001], 'quora_question_pairs': [0.72596], 'knowledge_distillation': [0.66329]}
for each_dir, key in zip(task_dir, results.keys()):
    target_dir = base_dir + each_dir
    with open(target_dir + '/train_autotest.log', 'r') as f_r:
        last_line = f_r.readlines()[-1].strip()
        score = ''.join(re.findall(r'(?<=accuracy:).*?(?=loss|;)', last_line))
        try:
            results[key].append(float(score))
        except:
            results[key].append('wrong')
            print ('GPU test. Wrong number in %s/train_autotest.log' %target_dir)

    with open(target_dir + '/test_autotest.log', 'r') as f_r:
        last_line = f_r.readlines()[-1].strip()
        score = ''.join(re.findall(r'(?<=accuracy:).*?(?=loss|;)', last_line))
        try:
            results[key].append(float(score))
        except:
            results[key].append('wrong')
            print ('CPU test. Wrong number in %s/test_autotest.log' %target_dir)


# for kdtm_match_linearAttn task, we use calculate_AUC.main()
params = {'input_file': './autotest/models/kdtm_match_linearAttn/predict.tsv', 'predict_index': '3', 'label_index': '2', 'header': False}
try:
    AUC = float(main(params))
    results['knowledge_distillation'].append(AUC)
except:
    results['knowledge_distillation'].append('wrong')

with open('./autotest/contrast_results.txt', 'w') as f_w:
    f_w.write('tasks' + '\t'*5 + 'GPU/CPU' + '\t' + 'old accuracy/AUC' + '\t' + 'new accuracy/AUC ' + '\n')
    for key, value in results.items():
        if key == 'knowledge_distillation':
            #f_w.write(key + '\t'*2 + 'GPU' + str(value[0]) + '\t'*3 + str(value[1]) + '\n')
            f_w.write(key + '\t'*2 + 'CPU' + '\t'*2 + str(value[0]) + '\t'*3 + str(value[1]) + '\n')
        else:
            f_w.write(key + '\t'*2 + 'GPU' + '\t'*2 + str(value[0]) + '\t'*3 + str(value[1]) + '\n')
            f_w.write(key + '\t'*2 + 'CPU' + '\t'*2 + str(value[0]) + '\t'*3 + str(value[2]) + '\n')

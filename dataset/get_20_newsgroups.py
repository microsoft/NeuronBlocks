# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.
import os
import shutil
from os import listdir
import tarfile
import argparse
from sys import version_info
from sklearn.model_selection import train_test_split
if version_info.major == 2:
    import urllib as urldownload
else:
    import urllib.request as urldownload


class NewsGroup(object):
    def __init__(self, params):
        self.params = params
        self.url = "http://archive.ics.uci.edu/ml/machine-learning-databases/20newsgroups-mld/20_newsgroups.tar.gz"
        self.file_name = '20_newsgroups.tar.gz'
        self.dirname = '20_newsgroups'

    def download_or_zip(self):
        if not os.path.exists(self.params.root_dir):
            os.mkdir(self.params.root_dir)
        path = os.path.join(self.params.root_dir, self.dirname)
        if not os.path.isdir(path):
            file_path = os.path.join(self.params.root_dir, self.file_name)
            if not os.path.isfile(file_path):
                print('DownLoading...')
                urldownload.urlretrieve(self.url, file_path)
            with tarfile.open(file_path, 'r', encoding='utf-8') as fin:
                print('Extracting...')
                fin.extractall(self.params.root_dir)
        return path

    def read_process_file(self, file_path):
        text_lines = []
        with open(file_path, 'rb') as fin:
            for single_line in fin:
                text_lines.append(str(single_line))
        return ''.join(text_lines).replace('\n', ' ').replace('\t', ' ')

    def data_combination(self):
        data_dir_path = self.download_or_zip()
        class_name_folders = listdir(data_dir_path)
        assert len(class_name_folders) == 20, 'The 20_newsgroups data has 20 classes and 20 sub folder accordingly, but we found %d' % len(class_name_folders)
        pathname_list = []
        label_list = []
        for sub_folder in class_name_folders:
            sub_folder_path = os.path.join(data_dir_path, sub_folder)
            for single_file in listdir(sub_folder_path):
                pathname_list.append(os.path.join(sub_folder_path, single_file))
                label_list.append(sub_folder)
        # prepare folder and write data
        if not os.path.exists(self.params.output_dir):
            os.mkdir(self.params.output_dir)
        data_all = []
        print('Preprocessing...')
        for single_file_path, singel_label in zip(pathname_list, label_list):
            text_line = '%s\t%s\n' % (singel_label, self.read_process_file(single_file_path))
            data_all.append(text_line)

        print('Write output file...')
        if self.params.isSplit:
            output_train_file_path = os.path.join(self.params.output_dir, 'train.tsv')
            output_test_file_path = os.path.join(self.params.output_dir, 'test.tsv')
            train_data, test_data = train_test_split(data_all, test_size=self.params.test_size, random_state=123)
            with open(output_train_file_path, 'w', encoding='utf-8') as fout:
                fout.writelines(train_data)
            with open(output_test_file_path, 'w', encoding='utf-8') as fout:
                fout.writelines(test_data)
        else:
            output_file_path = os.path.join(self.output_dir, 'output.tsv')
            with open(output_file_path, 'w', encoding='utf-8') as fout:
                fout.writelines(data_all)
        try:
            if os.path.exists(self.params.root_dir):
                shutil.rmtree(self.params.root_dir)
        except:
            pass


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='20_newsgroups data preprocess')
    parse.add_argument("--root_dir", type=str, default='./data', help='the folder path of saving download file and untar files')
    parse.add_argument("--output_dir", type=str, default='20_newsgroups', help='the folder path of saving tsv format files after preprocess')
    parse.add_argument("--isSplit", type=bool, default=True, help='appoint split data into train dataset and test dataset or not')
    parse.add_argument("--test_size", type=float, default=0.2)
    params, _ = parse.parse_known_args()
    newsgroup = NewsGroup(params)
    newsgroup.data_combination()

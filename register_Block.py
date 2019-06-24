# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import os
import argparse


def get_block_path(block_name, path='./block_zoo'):
    ''' find the block_name.py file in block_zoo
    Args:
         block_name: the name need to be registered. eg. BiLSTM/ CRF
    '''
    get_dir = os.listdir(path)
    for single in get_dir:
        sub_dir = os.path.join(path, single)
        if os.path.isdir(sub_dir):
            result = get_block_path(block_name, path=sub_dir)
            if result:
                return result
        else:
            if block_name + '.py' == single:
                return sub_dir
    return None


def write_file(new_block_path, file_path):
    init_path = os.path.join(file_path, '__init__.py')
    diff = new_block_path[len(file_path):].split('/')
    if diff[0] == '':
        diff.pop(0)
    # delete '.py' in the last str
    diff[-1] = diff[-1][:-3]
    line = 'from .' + diff[0] + ' import ' + diff[-1] + ', ' + diff[-1] + 'Conf'
    with open(init_path, 'a', encoding='utf-8') as fin:
        fin.write('\n' + line + '\n')


def register(block_name, new_block_path):
    ''' Add import code in the corresponding file. eg. block_zoo/__init__.py or block_zoo/subdir/__init__.py

    '''
    # check if block exist or not
    if new_block_path:
        block_path_split = new_block_path.split('/')
        for i in range(len(block_path_split)-1, 1, -1):
            # need_add_file.append(os.path.join('/'.join(block_path_split[:i])))
            write_file(new_block_path, os.path.join('/'.join(block_path_split[:i])))
        print('The block %s is registered successfully.' % block_name)
    else:
        raise Exception('The %s.py file does not exist! Please check your program or file name.' % block_name)


def main(params):
    new_block_path = get_block_path(params.block_name)
    register(params.block_name, new_block_path)


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='Register Block')
    parse.add_argument("--block_name", type=str, help="block name want to be registered")
    params, _ = parse.parse_known_args()
    assert params.block_name, 'Please specify a block_name via --block_name'
    main(params)

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import os
import tempfile
import subprocess
import shutil
import codecs
import torch
import pickle as pkl
import logging
from contextlib import contextmanager
import random
import string

def convert_to_tmppath(filepath):
    tmpfolder = tempfile.gettempdir()
    if filepath.startswith('/hdfs/'):
        filepath = filepath.replace('/hdfs/', '', 1)
    tmppath = os.path.join(tmpfolder, 'neuronblocks', filepath + '_' + ''.join(random.sample(string.ascii_letters+string.digits, 16)))
    tmpdir = os.path.dirname(tmppath)
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)
    logging.info('Obtain a local tmp path %s for %s' % (tmppath, filepath))
    return tmppath


def convert_to_hdfspath(filepath):
    if not filepath or not filepath.startswith("/hdfs/"):
        return filepath
    hdfs_direct_path = filepath.replace('/hdfs/', 'hdfs:///', 1)
    logging.info('Convert %s to hdfs direct path %s' % (filepath, hdfs_direct_path))
    return hdfs_direct_path


def move_from_local_to_hdfs(tmpfilepath, remotefilepath):
    '''
    ret = subprocess.check_call("/var/storage/shared/public/philly/sethadoop.sh && \
                    hdfs dfs -mkdir -p {2} && \
                    hdfs dfs -moveFromLocal -f {0} {1}".format(tmpfilepath, remotefilepath, os.path.dirname(remotefilepath)), shell=True)
    '''
    ret = subprocess.check_call("hdfs dfs -mkdir -p {2} && \
                    hdfs dfs -moveFromLocal -f {0} {1}".format(tmpfilepath, remotefilepath, os.path.dirname(remotefilepath)), shell=True)
    logging.info('With hdfs command, %s moved to hdfs path %s' % (tmpfilepath, remotefilepath))


class HDFSDirectTransferer(object):
    """ Help to access HDFS directly, for pickle file and torch.save
    For a hdfs path, save it to a local file firstly and then move it to the hdfs direct path
    For a local path, do save it at local path

    """
    def __init__(self, path, with_hdfs_command=True):
        """

        Args:
            path:
            with_hdfs_command: if True, transfer data with hdfs command; if False, with shutil.move
        """
        self.origin_path = path
        if path.startswith('/hdfs/'):
            self.activate = True
        else:
            self.activate = False

        self.with_hdfs_command = with_hdfs_command

    def __enter__(self):
        if self.activate:
            self.tmp_path = convert_to_tmppath(self.origin_path)
        else:
            self.tmp_path = self.origin_path
        logging.info('Convert %s to temp path %s' % (self.origin_path, self.tmp_path))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.activate:
            if self.with_hdfs_command:
                self.hdfs_direct_path = convert_to_hdfspath(self.origin_path)
                move_from_local_to_hdfs(self.tmp_path, self.hdfs_direct_path)
            else:
                shutil.move(self.tmp_path, self.origin_path)
                logging.info('With shutil.move command, %s moved to hdfs path %s' % (self.tmp_path, self.origin_path))

    def pkl_dump(self, obj):
        with open(self.tmp_path, 'wb') as fout:
            pkl.dump(obj, fout, protocol=pkl.HIGHEST_PROTOCOL)

    def torch_save(self, model):
        torch.save(model, self.tmp_path, pickle_protocol=pkl.HIGHEST_PROTOCOL)


@contextmanager
def open_and_move(path):
    """ write to a file directly or indirectly

    Args:
        path: the path to write to finally
        with_temp: if True, write to a local file firstly and move to another place,
                    if False, write to a path directly
    """

    if path.startswith('/hdfs/'):
        activate = True
    else:
        activate = False

    if activate:
        middle_path = convert_to_tmppath(path)
    else:
        middle_path = path

    fp = codecs.open(middle_path, 'w', encoding='utf-8')
    yield fp
    fp.close()

    if activate:
        hdfs_direct_path = convert_to_hdfspath(path)
        move_from_local_to_hdfs(middle_path, hdfs_direct_path)
        #shutil.move(middle_path, path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(filename)s %(funcName)s %(lineno)d: %(message)s')

    '''
    a = {1: 2, 3: 4}
    save_path = '/hdfs/a.pkl'
    with HDFSDirectTransferer(save_path, with_hdfs_command=True) as transferer:
        transferer.pkl_dump(a)
    '''

    with open_and_move('/hdfs/test/a.txt') as fout:
        fout.write('hello world\n')
        fout.write('hello world2\n')

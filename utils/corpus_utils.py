# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from __future__ import absolute_import

import os
import string
import sys
import numpy as np
import logging
import math
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.fasttext import FastText
import codecs
import random
import copy
from settings import ProblemTypes
import torch
import re


if sys.version_info < (3,):
    maketrans = string.maketrans
else:
    maketrans = str.maketrans


def load_embedding(embedding_path, embedding_dim, format, file_type, with_head=False, word_set=None):
    """
    Args:
        format: 'glove', 'word2vec', 'fasttext'
        file_type: 'text' or 'binary'
    """
    embedding_dict = dict()

    if format == 'word2vec' or format == 'fasttext':
        if file_type == 'text':
            vector_total = KeyedVectors.load_word2vec_format(embedding_path, binary=False, unicode_errors='ignore')
        else:
            if format == 'word2vec':
                vector_total = KeyedVectors.load_word2vec_format(embedding_path, binary=True, unicode_errors='ignore')
            elif format == 'fasttext':
                vector_total = FastText.load_fasttext_format(embedding_path, encoding='utf8')

        assert vector_total.vector_size == embedding_dim
        if word_set is None:
            embedding_dict = vector_total
        else:
            if not (format == 'fasttext' and file_type == 'binary'):
                word_total = vector_total.index2word    # actually, vector_total.index2word is the word list
            else:
                word_total = vector_total.wv.index2word
            for word in word_total:
                if word in word_set:
                    embedding_dict[word] = vector_total[word]
    elif format == 'glove':
        with codecs.open(embedding_path, 'r', encoding='utf-8') as fin:
            if with_head == True:
                _ = fin.readline()
            for idx, line in enumerate(fin):
                line = line.rstrip()
                if idx == 0 and len(line.split()) == 2:
                    continue
                if len(line) > 0:
                    word, vec = line.split(" ", 1)
                    if (word_set and word in word_set) or (word_set is None):
                        vector = [float(num) for num in vec.split(" ")]
                        assert len(vector) == embedding_dim
                        embedding_dict[word] = vector
    else:
        raise Exception('The format supported are glove, word2vec or fasttext, dost not support %s now.' % format)
    return embedding_dict


def split_array(arr, n, small_chunk_threshold=0):
    """split the array, each chunk has n elements (the last chunk might be different)
    Args:
        arr: the list to chunk, can be python list or numpy array
        n:   number of elements in a chunk
        small_chunk_threshold: chunks have less than {small_chunk_threshold} elements are forbiddened.
                if the last chunk has less than {small_chunk_threshold} elements, merge them into the former chunk.
    """
    result = [arr[i:i + n] for i in range(0, len(arr), n)] #namely, small_chunk_threshold = 0
    if len(result[-1]) < small_chunk_threshold:
        if isinstance(result[-2], np.ndarray) == True:
            result[-2] = result[-2].tolist()

        result[-2].extend(result[-1])   #result[-1] can be either python list or np.ndarray
        result[-2] = np.array(result[-2])
        logging.debug("The last chunk of size %d is smaller than the small_chunk_threshold %d, so merge it to chunk[-2]" % (len(result[-1]), small_chunk_threshold))
        logging.debug("Now the size of chunk[-2] is increase from %d to %d" % (n, len(result[-2])))
        del result[-1]
    return result


def split_array_averagely(arr, m):
    """ split the array to n small chunks with nearly the same sizes.

    Args:
        arr:
        m:

    Returns:

    """
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]


def cut_and_padding(seq, max_len, pad=0):
    """
    cut or pad the sequence to fixed size
    Args:
        seq:     sequence
        max_len:    the fixed size specified
        pad:    symbol to pad
    Returns:
        fixed size sequence
    """
    if len(seq) >= max_len:
        return seq[:max_len]
    else:
        return seq + [pad] * (max_len - len(seq))


def to_categorical(y, nb_classes=None):
    '''Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy.
    '''
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y



def base_filter():
    f = string.punctuation
    f = f.replace("'", '')
    f += '\t\n'
    return f


def text_to_word_sequence(text, filters=base_filter(), lower=True, split=" "):
    '''prune: sequence of characters to filter out
    '''
    if lower:
        text = text.lower()
    text = text.translate(maketrans(filters, split*len(filters)))
    seq = text.split(split)
    return [_f for _f in seq if _f]


def corpus_permutation(*corpora):
    """

    Args:
        *corpora:  different fields of a corpus

    Returns:

    """
    logging.info("Start permutation")
    perm = np.random.permutation(len(corpora[0]))

    corpora_perm = []
    for i in range(len(corpora)):
        corpora_perm.append(np.array(corpora[i])[perm])

    logging.info("Permutation end!")

    return corpora_perm


def get_batches(problem, data, length, target, batch_size, input_types, pad_ids=None, permutate=False, transform_tensor=True):
    """

    Args:
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
        permutate: shuffle data
        transform_tensor: if True the data returned would be Variables in CPU (except sentence length), otherwise the data would be numpy array

    Returns:
        data_batches: each element is a batch of data
            [
                {
                    "string1":{
                        'word': ndarray/Variable, shape:[batch_size, seq_len],
                        'postag': ndarray/Variable, postag ids, shape: [batch_size, seq_len],
                        ...
                    }
                    "string2":{
                        'word': ndarray/Variable, shape:[batch_size, seq_len],
                        'postag': ndarray/Variable, postag ids, shape: [batch_size, seq_len],
                        ...
                    }
                }
                ...
            ]
        length_batches: {
            'string1": ndarray, [number of batches, batch size]
            'string2": ndarray, [number of batches, batch size]
        }
        target_batches: ndarray/Variable shape: [number of batches, batch_size, targets]

    """
    logging.info("Start making batches")
    if permutate is True:
        #CAUTION! data and length would be revised
        data = copy.deepcopy(data)
        length = copy.deepcopy(length)
        if target is not None:
            target = copy.deepcopy(target)

        # shuffle the data
        permutation = np.random.permutation(len(list(target.values())[0]))
        for input_cluster in data:
            for input in data[input_cluster]:
                data[input_cluster][input] = np.array(data[input_cluster][input])[permutation]
            for single_type in length[input_cluster]:
                length[input_cluster][single_type] = np.array(length[input_cluster][single_type])[permutation]
        if target is not None:
            for single_target in target:
                length['target'][single_target] = np.array(length['target'][single_target])[permutation]
                target[single_target] = np.array(target[single_target])[permutation]
    else:
        for input_cluster in data:
            for input in data[input_cluster]:
                data[input_cluster][input] = np.array(data[input_cluster][input])
            for single_type in length[input_cluster]:
                length[input_cluster][single_type] = np.array(length[input_cluster][single_type])
        if target is not None:
            for single_target in target:
                length['target'][single_target] = np.array(length['target'][single_target])
                target[single_target] = np.array(target[single_target])

    # set up padding symbols for inputs and target
    if pad_ids is None:
        pad_ids = dict()
        for branch in input_types:
            pad_ids[branch] = problem.input_dicts[branch].id('<pad>')

        if ProblemTypes[problem.problem_type] == ProblemTypes.sequence_tagging:
            #pad_ids['target'] = problem.output_dict.id('O')
            if problem.target_with_pad:
                pad_ids['target'] = problem.output_dict.id('<pad>')
            else:
                pad_ids['target'] = 0       # CAUTION
        elif ProblemTypes[problem.problem_type] == ProblemTypes.classification:
            if problem.target_with_pad:
                pad_ids['target'] = problem.output_dict.id('<pad>')       # CAUTION
            else:
                pad_ids['target'] = 0       # CAUTION
        elif ProblemTypes[problem.problem_type] == ProblemTypes.regression:
            pad_ids['target'] = None
        elif ProblemTypes[problem.problem_type] == ProblemTypes.mrc:
            pass
    type2cluster = dict()       # type2cluster['word1'] = 'word'
    for input_type in input_types:
        for col_name in input_types[input_type]['cols']:
            type2cluster[col_name] = input_type

    # get the corpus size
    for input_cluster in data:
        for input_type in data[input_cluster]:
            corpus_size = len(data[input_cluster][input_type])
            break
        break

    data_batches = []
    if target is not None:
        target_batches = []
    else:
        target_batches = None
    length_batches = []
    for stidx in range(0, corpus_size, batch_size):
        data_batch = dict()
        length_batch = dict()

        for input_cluster in data:
            data_batch[input_cluster] = dict()
            length_batch[input_cluster] = dict()
            max_sen_len_cur_batch = None
            max_word_len_cur_batch = None
            if transform_tensor is True:
                # For nn.DataParallel, the length must be Variable as well, otherwise the length would not split for multiple GPUs
                #length_batch[input_cluster] = Variable(torch.LongTensor(length[input_cluster][stidx: stidx + batch_size]))
                for single_input_cluster in length[input_cluster]:
                    if not isinstance(length[input_cluster][single_input_cluster][0], list):
                        length_batch[input_cluster][single_input_cluster] = \
                            torch.LongTensor(np.array(length[input_cluster][single_input_cluster][stidx: stidx + batch_size]))
                    else:
                        length_batch[input_cluster][single_input_cluster] = []
                        for single_iterm in length[input_cluster][single_input_cluster][stidx: stidx + batch_size]:
                            length_batch[input_cluster][single_input_cluster].append(torch.LongTensor(np.array(single_iterm)))
            else:
                for single_input_cluster in length[input_cluster]:
                    length_batch[input_cluster][single_input_cluster] = \
                        np.array(length[input_cluster][single_input_cluster][stidx: stidx + batch_size])

            # max_len_cur_batch = np.sort(length[input_cluster][stidx: stidx + batch_size])[-1]
            for single_input_cluster in length[input_cluster]:
                if 'sentence' in single_input_cluster:
                    max_sen_len_cur_batch = np.sort(length[input_cluster][single_input_cluster][stidx: stidx + batch_size])[-1]
                elif 'word' in single_input_cluster:
                    max_word_len_cur_batch = np.sort([y for x in length[input_cluster][single_input_cluster][stidx: stidx + batch_size] for y in x])[-1]
            #logging.info("stidx: %d, max_len: %d" % (stidx, max_len_cur_batch))
            for input_type in data[input_cluster]:
                if input_type in type2cluster:
                    batch_with_pad = []
                    # process char data
                    if 'char' in input_type.lower():
                        for seq in data[input_cluster][input_type][stidx: stidx + batch_size]:
                            batch_char_pad = []
                            for seq_index in range(max_sen_len_cur_batch):
                                if seq_index < len(seq):
                                    batch_char_pad.append(cut_and_padding(seq[seq_index], max_word_len_cur_batch, pad_ids[type2cluster[input_type]]))
                                else:
                                    batch_char_pad.append(cut_and_padding([pad_ids[type2cluster[input_type]]], max_word_len_cur_batch, pad_ids[type2cluster[input_type]]))
                            batch_with_pad.append(batch_char_pad)
                    else:
                        for seq in data[input_cluster][input_type][stidx: stidx + batch_size]:
                        #batch_with_pad.append(cut_and_padding(seq, max_len_cur_batch, pad_ids[input_type]))
                            batch_with_pad.append(cut_and_padding(seq, max_sen_len_cur_batch, pad_ids[type2cluster[input_type]]))
                    if transform_tensor is True:
                        data_batch[input_cluster][type2cluster[input_type]] = torch.LongTensor(batch_with_pad)
                    else:
                        data_batch[input_cluster][type2cluster[input_type]] = np.array(batch_with_pad)
                else:
                    data_batch[input_cluster][input_type] = data[input_cluster][input_type][stidx: stidx + batch_size]
            # word_length is used for padding char sequence, now only save sentence_length
            length_batch[input_cluster] = length_batch[input_cluster]['sentence_length']

        data_batches.append(data_batch)
        length_batches.append(length_batch)

        if target is not None:
            target_batch = {}
            length_batch['target'] = {}
            for single_target in target:
                if transform_tensor is True:
                    length_batch['target'][single_target] = torch.LongTensor(np.array(length['target'][single_target][stidx: stidx + batch_size]))
                else:
                    length_batch['target'][single_target] = np.array(length['target'][single_target][stidx: stidx + batch_size])
                if not (isinstance(target[single_target][0], list) or isinstance(target[single_target][0], np.ndarray)):
                    target_batch[single_target] = target[single_target][stidx: stidx + batch_size]
                else:
                    # target is also a sequence, padding needed
                    temp_target_batch = []
                    for seq in target[single_target][stidx: stidx + batch_size]:
                        temp_target_batch.append(cut_and_padding(seq, max_sen_len_cur_batch, pad_ids['target']))
                    target_batch[single_target] = temp_target_batch
                if transform_tensor is True:
                    if ProblemTypes[problem.problem_type] == ProblemTypes.classification \
                            or ProblemTypes[problem.problem_type] == ProblemTypes.sequence_tagging:
                        target_batch[single_target] = torch.LongTensor(target_batch[single_target])
                    elif ProblemTypes[problem.problem_type] == ProblemTypes.regression:
                        target_batch[single_target] = torch.FloatTensor(target_batch[single_target])
                    elif ProblemTypes[problem.problem_type] == ProblemTypes.mrc:
                        if not isinstance(target_batch[single_target][0], str):
                            target_batch[single_target] = torch.LongTensor(target_batch[single_target])
                else:
                    target_batch[single_target] = np.array(target_batch[single_target])

            target_batches.append(target_batch)

    logging.info("Batches got!")
    return data_batches, length_batches, target_batches


def get_seq_mask(seq_len, max_seq_len=None):
    """

    Args:
        seq_len (ndarray): 1d numpy array/list

    Returns:
        ndarray : 2d array seq_len_mask. the mask symbol for a real token is 1 and for <pad> is 0.

    """
    if torch.is_tensor(seq_len):
        seq_len = seq_len.cpu().numpy()

    if max_seq_len is None:
        max_seq_len = seq_len.max()
    masks = np.array([[1]*seq_len[i] + [0] * (max_seq_len - seq_len[i]) for i in range(len(seq_len))])
    return masks



if __name__ == "__main__":
    '''
    y = [1, 0, 1, 0]
    y_convert = to_categorical(y, 2)
    print(y_convert)
    '''

    '''
    load_embedding(r'/data/t-wulin/data/embeddings/glove/glove.840B.300d.txt', 300, 'glove', 'text', word_set=None)
    print('glove text loaded')
    load_embedding(r'/data/t-wulin/data/embeddings/GoogleNews-vectors-negative300.bin', 300, 'word2vec', 'binary', word_set=None)
    print('word2vec bin loaded')
    load_embedding(r'/data/t-wulin/data/embeddings/fasttext.wiki.en/wiki.en.bin', 300, 'fasttext', 'binary', word_set=None)
    print('fasttext bin loaded')
    load_embedding(r'/data/t-wulin/data/embeddings/fasttext.wiki.en/wiki.en.vec', 300, 'fasttext', 'text', word_set=None)
    print('fasttext text loaded')
    '''
    load_embedding(r'/data/t-wulin/data/embeddings/fasttext.wiki.en/wiki.en.bin', 300, 'fasttext', 'binary', word_set=None)
    print('fasttext bin loaded')



# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import logging
from collections import Counter
import os
import pickle as pkl

class CellDict(object):
    def __init__(self, with_start=False, with_end=False, with_unk=False, with_pad=True):

        self.cell_id_map = dict()
        self.id_cell_map = dict()

        self.cell_doc_count = Counter()

        self.with_start = with_start
        self.with_end = with_end
        self.with_unk = with_unk
        self.with_pad = with_pad

        if with_pad:
            # 0 is reserved for padding
            self.cell_id_map["<pad>"] = 0
            self.id_cell_map[0] = "<pad>"

        if with_start:
            id = len(self.cell_id_map)
            self.cell_id_map["<start>"] = id
            self.id_cell_map[id] = "<start>"

        if with_end:
            id = len(self.cell_id_map)
            self.cell_id_map["<eos>"] = id
            self.id_cell_map[id] = "<eos>"

        if with_unk:
            id = len(self.cell_id_map)
            self.cell_id_map["<unk>"] = id
            self.id_cell_map[id] = "<unk>"

    def iter_ids(self):

        for id in self.id_cell_map:
            yield id

    def iter_cells(self):

        for cell in self.cell_id_map:
            yield cell

    def id(self, cell):
        return self.cell_id_map[cell]

    def cell(self, id):
        return self.id_cell_map[id]

    def cell_num(self):
        return len(self.id_cell_map)

    def has_id(self, id):

        return id in self.id_cell_map

    def has_cell(self, cell):

        return cell in self.cell_id_map

    def build(self, docs, threshold, max_vocabulary_num=800000):

        for doc in docs:
            # add type judge
            if isinstance(doc, list):
                self.cell_doc_count.update(set(doc))
            else:
                self.cell_doc_count.update([doc])

        # restrict the vocabulary size to prevent embedding dict weight size
        self.cell_doc_count = Counter(dict(self.cell_doc_count.most_common(max_vocabulary_num)))
        for cell in self.cell_doc_count:

            if cell not in self.cell_id_map and self.cell_doc_count[cell] >= threshold:
                id = len(self.cell_id_map)
                self.cell_id_map[cell] = id
                self.id_cell_map[id] = cell

    def id_unk(self, cell):
        """
        return <unk> in cells not in cell_id_map dict
        :param cell:
        :return:
        """
        if cell in self.cell_id_map:
            return self.cell_id_map[cell]
        else:
            return self.cell_id_map['<unk>']

    def load_from_exist_dict(self, word_dict,
                             start_str=None,
                             end_str=None,
                             unk_str=None,
                             pad_str=None):
        """
        build cells from exist dicts
        :param word_dict:
        :param start_str: start single in word_dict
        :param end_str: end single in word_dict
        :param unk_str:
        :param pad_str:
        :return:
        """
        id2word_dict = dict()
        if pad_str:
            # 0 is reserved for padding
            id = word_dict[pad_str]
            word_dict["<pad>"] = id
            del (word_dict[pad_str])
            self.with_pad = True

        if start_str:
            id = word_dict[start_str]
            word_dict["<start>"] = id
            del(word_dict[start_str])
            self.with_start = True

        if end_str:
            id = word_dict[end_str]
            word_dict["<eos>"] = id
            del(word_dict[end_str])
            self.with_end = True

        if unk_str:
            id = word_dict[unk_str]
            word_dict["<unk>"] = id
            del(word_dict[unk_str])
            self.with_unk = True

        for word, id in word_dict.iteritems():
            id2word_dict[id] = word
        self.cell_id_map = word_dict
        self.id_cell_map = id2word_dict

    def add_cells(self, cells):

        for cell in cells:
            if cell not in self.cell_id_map:
                id = len(self.cell_id_map)
                self.cell_id_map[cell] = id
                self.id_cell_map[id] = cell

    def lookup(self, cells):
        """ get id of cells. if cell in dict, return id, else update the dict and return id

        Args:
            cells:

        Returns:

        """
        cur_corpus = [0] * len(cells)

        for idx, cell in enumerate(cells):

            if cell not in self.cell_id_map:

                if self.with_unk:
                    id = self.id("<unk>")
                else:
                    logging.error(
                        u"Unknown cells {0} found. the repo "
                        u"should build with with_unk = True.".format(
                            cell).encode('utf8'))
                    raise Exception("Unknown cells found with with_unk=False")
            else:
                id = self.cell_id_map[cell]

            cur_corpus[idx] = id

        return cur_corpus

    def decode(self, ids):
        """ look up the cell for a list of ids

        Args:
            ids: list or 1d np array

        Returns:

        """
        return [self.cell(id) for id in ids]


    def export_cell_dict(self, save_path=None):
        if save_path is not None and not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        cell_dict = dict()
        for name, value in vars(self).items():
            if name.startswith("__") is False:
                cell_dict[name] = value

        if save_path != None:
            with open(save_path, 'wb') as fout:
                pkl.dump(cell_dict, fout, protocol=pkl.HIGHEST_PROTOCOL)
            logging.debug("Cell dict saved to %s" % save_path)
        return cell_dict

    def load_cell_dict(self, info_dict):
        for name in info_dict:
            setattr(self, name, info_dict[name])
        logging.debug("Cell dict loaded")




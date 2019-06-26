# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from block_zoo.BaseLayer import BaseLayer, BaseConf
from utils.DocInherit import DocInherit
import torch
import torch.nn as nn
from copy import deepcopy
import torch.autograd as autograd


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def log_sum_exp(vec, m_size):
    """
    calculate log of exp sum
    args:
        vec (batch_size, vanishing_dim, hidden_dim) : input tensor
        m_size : hidden_dim
    return:
        batch_size, hidden_dim
    """
    _, idx = torch.max(vec, 1)  # B * 1 * M
    max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)  # B * M
    return max_score.view(-1, m_size) + torch.log(torch.sum(torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1, m_size)  # B * M


class CRFConf(BaseConf):
    """
    Configuration of CRF layer

    Args:

    """
    def __init__(self, **kwargs):
        super(CRFConf, self).__init__(**kwargs)

    @DocInherit
    def default(self):
        self.START_TAG = "<start>"
        self.STOP_TAG = "<eos>"

    @DocInherit
    def declare(self):
        self.num_of_inputs = 1
        self.input_ranks = [3]

    @DocInherit
    def inference(self):
        self.output_dim = [1]
        # add target dict judgement start or end
        self.target_dict = deepcopy(self.target_dict.cell_id_map)
        if not self.target_dict.get(self.START_TAG):
            self.target_dict[self.START_TAG] = len(self.target_dict)
        if not self.target_dict.get(self.STOP_TAG):
            self.target_dict[self.STOP_TAG] = len(self.target_dict)

        super(CRFConf, self).inference()

    @DocInherit
    def verify(self):
        super(CRFConf, self).verify()


class CRF(BaseLayer):
    """ Conditional Random Field layer

    Args:
        layer_conf(CRFConf): configuration of CRF layer
    """
    def __init__(self, layer_conf):
        super(CRF, self).__init__(layer_conf)
        self.target_size = len(self.layer_conf.target_dict)

        init_transitions = torch.zeros(self.target_size, self.target_size)
        init_transitions[:, self.layer_conf.target_dict[self.layer_conf.START_TAG]] = -10000.0
        init_transitions[self.layer_conf.target_dict[self.layer_conf.STOP_TAG], :] = -10000.0
        init_transitions[:, 0] = -10000.0
        init_transitions[0, :] = -10000.0

        if self.layer_conf.use_gpu:
            init_transitions = init_transitions.cuda()
        self.transitions = nn.Parameter(init_transitions)

    def _calculate_forward(self, feats, mask):
        """
            input:
                feats: (batch, seq_len, self.tag_size)
                masks: (batch, seq_len)
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)

        mask = mask.transpose(1, 0).contiguous()
        ins_num = seq_len * batch_size
        # be careful the view shape, it is .view(ins_num, 1, tag_size) but not .view(ins_num, tag_size, 1)
        feats = feats.transpose(1, 0).contiguous().view(ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)
        # need to consider start
        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)
        # build iter
        seq_iter = enumerate(scores)
        _, inivalues = next(seq_iter)  # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        partition = inivalues[:, self.layer_conf.target_dict[self.layer_conf.START_TAG], :].clone().view(batch_size, tag_size, 1)  # bat_size * to_target_size

        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # partition: previous results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: bat_size * from_target * to_target

            cur_values = cur_values + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            cur_partition = log_sum_exp(cur_values, tag_size)

            # (bat_size * from_target * to_target) -> (bat_size * to_target)
            # partition = utils.switch(partition, cur_partition, mask[idx].view(bat_size, 1).expand(bat_size, self.tagset_size)).view(bat_size, -1)
            mask_idx = mask[idx, :].view(batch_size, 1).expand(batch_size, tag_size)

            # effective updated partition part, only keep the partition value of mask value = 1
            masked_cur_partition = cur_partition.masked_select(mask_idx)
            # let mask_idx broadcastable, to disable warning
            mask_idx = mask_idx.contiguous().view(batch_size, tag_size, 1)

            # replace the partition where the maskvalue=1, other partition value keeps the same
            partition.masked_scatter_(mask_idx, masked_cur_partition)
        # until the last state, add transition score for all partition (and do log_sum_exp) then select the value in STOP_TAG
        cur_values = self.transitions.view(1, tag_size, tag_size).expand(batch_size, tag_size, tag_size) + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
        cur_partition = log_sum_exp(cur_values, tag_size)
        final_partition = cur_partition[:, self.layer_conf.target_dict[self.layer_conf.STOP_TAG]]
        return final_partition.sum(), scores

    def _viterbi_decode(self, feats, mask):
        """
            input:
                feats: (batch, seq_len, self.tag_size)
                mask: (batch, seq_len)
            output:
                decode_idx: (batch, seq_len) decoded sequence
                path_score: (batch, 1) corresponding score for each sequence
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)

        # calculate sentence length for each sentence
        length_mask = torch.sum(mask.long(), dim=1).view(batch_size, 1).long()
        # mask to (seq_len, batch_size)
        mask = mask.transpose(1, 0).contiguous()
        ins_num = seq_len * batch_size
        # be careful the view shape, it is .view(ins_num, 1, tag_size) but not .view(ins_num, tag_size, 1)
        feats = feats.transpose(1, 0).contiguous().view(ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)
        # need to consider start
        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)

        # build iter
        seq_iter = enumerate(scores)
        # record the position of best score
        back_points = list()
        partition_history = list()
        #  reverse mask (bug for mask = 1- mask, use this as alternative choice)
        mask = (1 - mask.long()).byte()
        _, inivalues = next(seq_iter)  # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        partition = inivalues[:, self.layer_conf.target_dict[self.layer_conf.START_TAG], :].clone().view(batch_size, tag_size)  # bat_size * to_target_size
        # print "init part:",partition.size()
        partition_history.append(partition)
        # iter over last scores
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # partition: previous results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: batch_size * from_target * to_target
            cur_values = cur_values + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            partition, cur_bp = torch.max(cur_values, 1)
            partition_history.append(partition)
            # cur_bp: (batch_size, tag_size) max source score position in current tag
            # set padded label as 0, which will be filtered in post processing
            cur_bp.masked_fill_(mask[idx].view(batch_size, 1).expand(batch_size, tag_size), 0)
            back_points.append(cur_bp)
        # add score to final STOP_TAG
        partition_history = torch.cat(partition_history, 0).view(seq_len, batch_size, -1).transpose(1, 0).contiguous() # (batch_size, seq_len. tag_size)
        # get the last position for each setences, and select the last partitions using gather()
        last_position = length_mask.view(batch_size, 1, 1).expand(batch_size, 1, tag_size) - 1
        last_partition = torch.gather(partition_history, 1, last_position).view(batch_size,tag_size,1)
        # calculate the score from last partition to end state (and then select the STOP_TAG from it)
        last_values = last_partition.expand(batch_size, tag_size, tag_size) + self.transitions.view(1, tag_size, tag_size).expand(batch_size, tag_size, tag_size)
        _, last_bp = torch.max(last_values, 1)
        pad_zero = autograd.Variable(torch.zeros(batch_size, tag_size)).long()
        if self.layer_conf.use_gpu:
            pad_zero = pad_zero.cuda()
        back_points.append(pad_zero)
        back_points = torch.cat(back_points).view(seq_len, batch_size, tag_size)

        # select end ids in STOP_TAG
        pointer = last_bp[:, self.layer_conf.target_dict[self.layer_conf.STOP_TAG]]
        insert_last = pointer.contiguous().view(batch_size, 1, 1).expand(batch_size, 1, tag_size)
        back_points = back_points.transpose(1, 0).contiguous()
        # move the end ids(expand to tag_size) to the corresponding position of back_points to replace the 0 values
        back_points.scatter_(1, last_position, insert_last)
        back_points = back_points.transpose(1, 0).contiguous()
        # decode from the end, padded position ids are 0, which will be filtered if following evaluation
        decode_idx = autograd.Variable(torch.LongTensor(seq_len, batch_size))
        if self.layer_conf.use_gpu:
            decode_idx = decode_idx.cuda()
        decode_idx[-1] = pointer.detach()
        for idx in range(len(back_points)-2, -1, -1):
            pointer = torch.gather(back_points[idx], 1, pointer.contiguous().view(batch_size, 1))
            decode_idx[idx] = pointer.detach().view(batch_size)
        path_score = None
        decode_idx = decode_idx.transpose(1, 0)
        return path_score, decode_idx

    def forward(self, string, string_len):
        """
        CRF layer process: include use transition matrix compute score and  viterbi decode

        Args:
            string(Tensor): [batch_size, seq_len, target_num]
            string_len(Tensor): [batch_size]

        Returns:
            score: the score by CRF inference
            best_path: the best bath of viterbi decode
        """
        assert string_len is not None, "CRF layer need string length for mask."
        masks = []
        string_len_val = string_len.cpu().data.numpy()
        for i in range(len(string_len)):
            masks.append(
                torch.cat([torch.ones(string_len_val[i]), torch.zeros(string.shape[1] - string_len_val[i])]))
        masks = torch.stack(masks).view(string.shape[0], string.shape[1]).byte()
        if self.layer_conf.use_gpu:
            masks = masks.cuda()

        forward_score, scores = self._calculate_forward(string, masks)

        _, tag_seq = self._viterbi_decode(string, masks)

        return (forward_score, scores, masks, tag_seq, self.transitions, self.layer_conf), string_len

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.autograd as autograd


class CRFLoss(nn.Module):
    """CRFLoss
       use for crf output layer for sequence tagging task.
    """
    def __init__(self):
        super(CRFLoss, self).__init__()
        
    def _score_sentence(self, scores, mask, tags, transitions, crf_layer_conf):
        """
            input:
                scores: variable (seq_len, batch, tag_size, tag_size)
                mask: (batch, seq_len)
                tags: tensor  (batch, seq_len)
            output:
                score: sum of score for gold sequences within whole batch
        """
        # Gives the score of a provided tag sequence
        batch_size = scores.size(1)
        seq_len = scores.size(0)
        tag_size = scores.size(2)
        # convert tag value into a new format, recorded label bigram information to index
        new_tags = autograd.Variable(torch.LongTensor(batch_size, seq_len))
        if crf_layer_conf.use_gpu:
            new_tags = new_tags.cuda()
        for idx in range(seq_len):
            if idx == 0:
                # start -> first score
                new_tags[:, 0] = (tag_size-2)*tag_size + tags[:, 0]
            else:
                new_tags[:, idx] = tags[:, idx-1]*tag_size + tags[:, idx]

        # transition for label to STOP_TAG
        end_transition = transitions[:, crf_layer_conf.target_dict[crf_layer_conf.STOP_TAG]].contiguous().view(1, tag_size).expand(batch_size, tag_size)
        # length for batch,  last word position = length - 1
        length_mask = torch.sum(mask.long(), dim=1).view(batch_size, 1).long()
        # index the label id of last word
        end_ids = torch.gather(tags, 1, length_mask - 1)

        # index the transition score for end_id to STOP_TAG
        end_energy = torch.gather(end_transition, 1, end_ids)

        # convert tag as (seq_len, batch_size, 1)
        new_tags = new_tags.transpose(1, 0).contiguous().view(seq_len, batch_size, 1)
        # need convert tags id to search from positions of scores
        tg_energy = torch.gather(scores.view(seq_len, batch_size, -1), 2, new_tags).view(seq_len, batch_size)  # seq_len * batch_size
        # mask transpose to (seq_len, batch_size)
        tg_energy = tg_energy.masked_select(mask.transpose(1, 0))

        # add all score together
        gold_score = tg_energy.sum() + end_energy.sum()
        return gold_score
    
    def forward(self, forward_score, scores, masks, tags, transitions, crf_layer_conf):
        """
        
        :param forward_score: Tensor scale
        :param scores: Tensor [seq_len, batch_size, target_size, target_size]
        :param masks:  Tensor [batch_size, seq_len]
        :param tags:   Tensor [batch_size, seq_len]
        :return: goal_score - forward_score
        """
        gold_score = self._score_sentence(scores, masks, tags, transitions, crf_layer_conf)
        return forward_score - gold_score
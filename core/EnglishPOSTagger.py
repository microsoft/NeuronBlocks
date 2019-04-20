# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import os
'''
if not ("STANFORD_MODELS" in os.environ and "STANFORD_POSTAGGER_PATH" in os.environ \
        and "CLASSPATH" in os.environ  \
        and os.environ['CLASSPATH'].find('stanford-postagger.jar') != -1):
    raise Exception("To use Stanford POS tagger, please set the corresponding environment "
        "variables first")
from nltk.tag import StanfordPOSTagger
'''
#from nltk.tag import pos_tag, pos_tag_sents
from nltk.tag.perceptron import PerceptronTagger
import nltk
class EnglishPOSTagger(object):
    def __init__(self, model_type='english-bidirectional-distsim.tagger'):
        """
        Args:
            model:  model available in $STANFORD_MODELS:
                english-bidirectional-distsim.tagger
                english-caseless-left3words-distsim.tagger
                english-left3words-distsim.tagger
        """
        #self.eng_tagger = StanfordPOSTagger(model_type, java_options='-mx16000m')
        self.eng_tagger = PerceptronTagger()

    def postag(self, word_list):
        """
        Args:
            word_list:  word list
        Returns:
            pos tag list
        """
        #word_pos_pairs = self.eng_tagger.tag(word_list)
        
        #word_pos_pairs = pos_tag(word_list)
        word_pos_pairs = nltk.tag._pos_tag(word_list, None, self.eng_tagger)
        pos_list = [pos for (word, pos) in word_pos_pairs]
        return pos_list

    def postag_multi(self, multi_sentence):
        """ tag multiple sentences one time
        RECOMMAND! Because the efficiency of stanford pos tagger in NLTK is too slow.
        Args:
            multi_sentence: [[token1, token2], ..., [...]]
        Returns:
        """
        #word_pos_pairs_multi_sent = self.eng_tagger.tag_sents(multi_sentence)
        '''
        word_pos_pairs_multi_sent = pos_tag_sents(multi_sentence)
        pos_lists = []
        for word_pos_pairs in word_pos_pairs_multi_sent:
            pos_lists.append([pos for (word, pos) in word_pos_pairs])
        return pos_lists
        '''
        return [self.postag(sent) for sent in multi_sentence]


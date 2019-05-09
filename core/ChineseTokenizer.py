# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import jieba
import logging
jieba.setLogLevel(logging.INFO)
from nltk.tokenize.util import align_tokens
from .Stopwords import Stopwords

class ChineseTokenizer(object):
    def __init__(self, tokenizer='jieba', remove_stopwords=False):
        self.__tokenizer = tokenizer
        self.__remove_stopwords = remove_stopwords
        if self.__remove_stopwords:
            self.__stop_words = Stopwords.chinese_stopwords
        else:
            self.__stop_words = None

    def tokenize(self, string):
        if self.__tokenizer == 'jieba':
            tokens = list(jieba.cut(string))

        if self.__remove_stopwords:
            tokens = [word for word in tokens if word not in self.__stop_words]
        return tokens

    def span_tokenize(self, string):
        if self.__tokenizer == 'jieba':
            tokens = self.tokenize(string)
            spans = align_tokens(tokens, string)
        return spans


if __name__ == '__main__':
    import os
    # nltk.data.path.append(r"C:\Users\wutlin\OneDrive - Microsoft\workspace\DNNMatchingToolkit\dataset\nltk_data")
    tokenizer = ChineseTokenizer(tokenizer='jieba', remove_stopwords=True)
    print(tokenizer.tokenize("我爱北京天安门，天安门上太阳升。"))
    print(tokenizer.span_tokenize("我爱北京天安门，天安门上太阳升。"))
    print(tokenizer.tokenize("给每一条河每一座山取一个温暖的名字；陌生人，我也为你祝福；愿你有一个灿烂的前程；愿你有情人终成眷属；愿你在尘世获得幸福；我只愿面朝大海，春暖花开。"))

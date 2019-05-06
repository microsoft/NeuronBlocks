# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import nltk
import re
from nltk.tokenize.util import align_tokens
from .Stopwords import Stopwords

class EnglishTokenizer(object):
    def __init__(self, tokenizer='nltk', remove_stopwords=False):
        self.__tokenizer = tokenizer
        self.__remove_stopwords = remove_stopwords
        if self.__remove_stopwords:
            self.__stop_words = Stopwords.english_stopwords
        else:
            self.__stop_words = None

    def tokenize(self, string):
        if self.__tokenizer == 'nltk':
            tokens = nltk.word_tokenize(string)

        if self.__remove_stopwords:
            tokens = [word for word in tokens if word not in self.__stop_words]
        return tokens

    def span_tokenize(self, string):
        if self.__tokenizer == 'nltk':
            raw_tokens = nltk.word_tokenize(string)
            if ('"' in string) or ("''" in string):
                matched = [m.group() for m in re.finditer(r"``|'{2}|\"", string)]
                tokens = [matched.pop(0) if tok in ['"', "``", "''"] else tok for tok in raw_tokens]
            else:
                tokens = raw_tokens
            spans = align_tokens(tokens, string)
        return spans


if __name__ == '__main__':
    import os
    # nltk.data.path.append(r"C:\Users\wutlin\OneDrive - Microsoft\workspace\DNNMatchingToolkit\dataset\nltk_data")
    tokenizer = EnglishTokenizer(tokenizer='nltk', remove_stopwords=True)
    print(tokenizer.span_tokenize("""What singer did Beyonce record a song with for the movie, ''The Best Man"?"""))
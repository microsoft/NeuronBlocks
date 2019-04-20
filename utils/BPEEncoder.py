# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import nltk

def get_pairs(word):
    """ Return set of symbol pairs in a word.
    word is represented as tuple of symbols (symbols being variable-length strings)
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class BPEEncoder(object):
    """ Byte Pair Encoding
    """
    def __init__(self, bpe_path):
        merges = open(bpe_path, encoding='utf-8').read().split('\n')[1:-1]
        merges = [tuple(merge.split()) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = dict()

    def encode(self, sentence):
        tokens = nltk.word_tokenize(sentence)
        bpe_tokens = []
        for token in tokens:
            bpe_tokens.extend(self.bpe(token))

        return bpe_tokens

    def bpe(self, token):
        """

        Args:
            token (string): a word token

        Returns:
            list: byte pair encodings

        """
        word = tuple(token[:-1]) + (token[-1] + '</w>',)
        if token in self.cache:
            return self.cache[token]
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        if word == '\n  </w>':
            word = '\n</w>'
        self.cache[token] = word
        return word.split(' ')


if __name__ == '__main__':
    sentences = 'trip cost to beijing'
    import nltk
    tokens = nltk.word_tokenize(sentences)
    bpe_encoder = BPEEncoder('../dataset/bpe/vocab_40000.bpe')
    bpe_tokens = []
    for token in tokens:
        print(token)
        bpe_tokens.extend(bpe_encoder.bpe(token))
    print(bpe_tokens)

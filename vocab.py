#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
    vocab.py --train-src=<file> --train-tgt=<file> [options] VOCAB_FILE
Options:
    -h --help                  Show this screen.
    --train-src=<file>         File of training source sentences
    --train-tgt=<file>         File of training target sentences
    --size=<int>               vocab size [default: 50000]
    --freq-cutoff=<int>        frequency cutoff [default: 2]
"""

import json
from collections import Counter
from itertools import chain
from typing import List
import os

import torch
import utils
import numpy as np
import random
# from utils import pad_sents


class Vocab(object):
    """ Vocab, i.e. structure containing either
    src or tgt language terms.
    """

    def __init__(self, embedding, vocab):
        """ 
        Args:
            embedding: List<List<Float>>
                The embedding representations
            vocab: List<string>
                The vocab corresponding to the embedding
        """
        self.word2id = {}
        self.id2word = {}
        for token in vocab:
            self.add(token)
            
        self.embedding = embedding

        self.unk_id = self.word2id['<unk>']
        self.start_id = self.word2id['<s>']
        self.end_id = self.word2id['</s>']
        self.pad_id = self.word2id['<pad>']

    def __getitem__(self, word):
        """ Retrieve word's index. Return the index for the unk
        token if the word is out of vocabulary.
        @param word (str): word to look up.
        @returns index (int): index of word 
        """
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        """ Check if word is captured by VocabEntry.
        @param word (str): word to look up
        @returns contains (bool): whether word is contained    
        """
        return word in self.word2id

    def __setitem__(self, key, value):
        """ Raise error, if one tries to edit the VocabEntry.
        """
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        """ Compute number of words in VocabEntry.
        @returns len (int): number of words in VocabEntry
        """
        return len(self.word2id)

    def __repr__(self):
        """ Representation of VocabEntry to be used
        when printing the object.
        """
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, wid):
        """ Return mapping of index to word.
        @param wid (int): word index
        @returns word (str): word corresponding to index
        """
        return self.id2word[wid]

    def add(self, word):
        """ Add word to VocabEntry, if it is previously unseen.
        @param word (str): word to add to VocabEntry
        @return index (int): index that the word has been assigned
        """
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    def words2indices(self, sents):
        """ Convert list of words or list of sentences of words
        into list or list of list of indices.
        @param sents (list[str] or list[list[str]]): sentence(s) in words
        @return word_ids (list[int] or list[list[int]]): sentence(s) in indices
        """
        if type(sents[0]) == list:
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]

    def indices2words(self, word_ids):
        """ Convert list of indices into words.
        @param word_ids (list[int]): list of word ids
        @return sents (list[str]): list of words
        """
        return [self.id2word[w_id] for w_id in word_ids]

    def to_input_tensor(self, sents: List[List[str]], device: torch.device) -> torch.Tensor:
        """ Convert list of sentences (words) into tensor with necessary padding for 
        shorter sentences.
        @param sents (List[List[str]]): list of sentences (words)
        @param device: device on which to load the tensor, i.e. CPU or GPU
        @returns sents_var: tensor of (max_sentence_length, batch_size)
        """
        # TODO(ankur): consider adding the start and stop token here.
        lengths = [len(sent) for sent in sents]
        word_ids = self.words2indices(sents)
        word_ids = [torch.tensor(sent, device=device) for sent in word_ids]
        return word_ids
        # sents_padded = torch.nn.utils.rnn.pad_sequence(word_ids, batch_first=True, padding_value=self.pad_id)
        # sents_pack_padded = torch.nn.utils.rnn.pack_padded_sequence(sents_padded, lengths=lengths, batch_first=True, enforce_sorted=False)
        # return sents_pack_padded
 

def randvec(n=50, lower=-0.5, upper=0.5):
    """
    Returns a random vector of length `n`. `w` is ignored.
    """
    return np.array([random.uniform(lower, upper) for i in range(n)])

GLOVE_HOME = os.path.join('data', 'glove.6B')
def load_glove50():
    glove_src = os.path.join(GLOVE_HOME, 'glove.6B.100d.txt')
    # Creates a dict mapping strings (words) to GloVe vectors:
    GLOVE = utils.glove2dict(glove_src)
    print("The number of items in glove is %d" % len(GLOVE))
    return GLOVE
    
def create_pretrained_embedding(
    lookup, vocab, required_tokens=("<unk>", "<pad>", "<s>", "</s>")):
    """
    Create an embedding matrix from a lookup and a specified vocab.
    Words from `vocab` that are not in `lookup` are given random
    representations.
    Parameters
    ----------
    lookup : dict
        Must map words to their vector representations.
    vocab : list of str
        Words to create embeddings for.
    required_tokens : tuple of str
        Tokens that must have embeddings. If they are not available
        in the look-up, they will be given random representations.
    Returns
    -------
    np.array, list
        The np.array is an embedding for `vocab` and the `list` is
        the potentially expanded version of `vocab` that came in.
    """
    dim = len(next(iter(lookup.values())))
    embedding = np.array([lookup.get(w, randvec(dim)) for w in vocab])
    for tok in required_tokens:
        if tok not in vocab:
            vocab.append(tok)
            embedding = np.vstack((embedding, randvec(dim)))
    return embedding, vocab
    
def build_vocab():
    GLOVE = load_glove50()
    embedding, vocab = create_pretrained_embedding(GLOVE, list(GLOVE.keys()))
    return Vocab(embedding, vocab)
    

# if __name__ == '__main__':
#     args = docopt(__doc__)

#     print('read in source sentences: %s' % args['--train-src'])
#     print('read in target sentences: %s' % args['--train-tgt'])

#     src_sents = read_corpus(args['--train-src'], source='src')
#     tgt_sents = read_corpus(args['--train-tgt'], source='tgt')

#     vocab = Vocab.build(src_sents, tgt_sents, int(args['--size']), int(args['--freq-cutoff']))
#     print('generated vocabulary, source %d words, target %d words' % (len(vocab.src), len(vocab.tgt)))

#     vocab.save(args['VOCAB_FILE'])
#     print('vocabulary saved to %s' % args['VOCAB_FILE'])
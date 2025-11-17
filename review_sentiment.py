from argparse import Namespace
from collections import Counter
import json
import os
import re
import string

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm_notebook

class Vocabulary(object):
    """Class to process text and extract vocabulary for mapping"""
    def __init__(self, token_to_idx = None, add_unk = True, unk_token = "<UNK>"):
        """
        Args: 
            token_to_idx (dict): map of tokens to indices
            add_unk (bool): binary flag for whether to add the UNK token
            unk_token (str): the UNK token to add into the vocabulary
        """
        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx

        self._idx_to_token = {idx: token
                              for token, idx in self._token_to_idx.items()}
        
        self._add_unk = add_unk
        self._unk_token = unk_token

        self.unk_index = -1
        if add_unk:
            self.unk_token = self.add_token(unk_token)
    
    def to_serializable(self):
        """Returns a serializable dictionary"""
        return {'token_to_idx': self._token_to_idx,
                'add_unk': self._add_unk,
                'unk_token': self._unk_token
                }
    
    @classmethod
    def from_serializable(cls, contents):
        """Instantiates the Vocabulary from a serialized dictionary """
        return cls(**contents)
    
    def add_token(self, token):
        """
        Updates the mapping dict for a given token
        Args:
            token (str): the token to add to the vocabulary
        Returns:
            index (int): the index of the token in the dict
        """
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index
    
    def add_many(self, tokens):
        return [self.add_token(token) for token in tokens]
    
    def lookup_token(self, token):
        """
        Returns an index associated with the token
        Args:
            token (str): the token whose index the function is looking for
        Returns:
            index (int): the index of the token 
        """
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]
        
    def lookup_index(self, index):
        """
        Return the token associated with the index
        Args:
            index (int): the index to loop up
        Returns:
            token (str): the token associated with the index
        Raies:
            KeyError: If index is not in the Vocabulary
        """
        if index not in self._idx_to_token:
            raise KeyError("The index (%d) is not in the Vocabulary" % index)
        return self._idx_to_token[index]
    
    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)
    
    def __len__(self):
        return len(self._token_to_idx)

class ReviewVectorizer(object):
    """Text processing class and extracts vocabulary for mapping"""
    def __init__(self, review_vocab, rating_vocab):
        """
        Args: 
            review_vocab (Vocabulary): map words to integers
            rating_vocab (Vocabulary): map class labels to integers
        """
        self.review_vocab = review_vocab
        self.rating_vocab = rating_vocab

    def vectorize(self, review):
        """
        Creates a one hot vector representation of the review
        Args:
            review (str): the review
        Returns:
            one_hot (np.nd array): the collapsed one hot encoding
        """
        one_hot = np.zeroes(len(self.review_vocab), dtype = np.float32)

        for token in review.split(" "):
            if token not in string.punctuation:
                one_hot[self.review_vocab.lookup_token(token)] = 1
        
        return one_hot
    
    @classmethod
    def from_df(cls, review_df, cutoff=25):
        """
        Instantiate the vectorizer from a dataframe
        Args:
            review_df (pandas.Dataframe): dataframe of the reviews
            cutoff (int): the parameter for frequency based filtering
        Return:
            an instance of the vectorizer
        """
        review_vocab = Vocabulary(add_unk = True)
        rating_vocab = Vocabulary(add_unk = False)

        for rating in sorted(set(review_df.rating)):
            rating_vocab.add_token(rating)

        word_counts = Counter()
        for review in review_df.review:
            for word in review.split(" "):
                if word not in string.punctuation:
                    word_counts[word] += 1
                
        for word, count in word_counts.items():
            if count > cutoff:
                review_vocab.add_token(word)

        return cls(review_vocab, rating_vocab)
    
    @classmethod
    def from_serializable(cls, contents):
        """Instantiate a ReviewVectorizer from a serializable dictionary
        
        Args:
            contents (dict): the serializable dictionary
        Returns:
            an instance of the ReviewVectorizer class
        """
        review_vocab = Vocabulary.from_serializable(contents['review_vocab'])
        rating_vocab = Vocabulary.from_serializable(contents['rating_vocab'])

        return cls(review_vocab = review_vocab, rating_vocab = rating_vocab)
    
    def to_serializable(self):
        """Create the serializable dictionary for caching
        
        Returns:
            contents (dict): the serializable dictionary
        """
        return {'review_vocab': self.review_vocab.to_serializable(),
                'rating_vocab': self.rating_vocab.to_serializable()}
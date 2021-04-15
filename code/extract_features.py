# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 15:51:57 2021

@author: Christian Konstantinov
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from soundex import Soundex
from jellyfish import nysiis
from functools import lru_cache

@lru_cache(maxsize = 128)
def edit_distance(source, target, func=min):
    """Given a [source] and [target], return the [func] edit distance"""
    n = len(source)
    m = len(target)

    D = np.zeros((n+1, m+1))
    D[:, 0] = np.arange(n+1) # init for D(i, 0)
    D[0, :] = np.arange(m+1) # init for D(0, j)
    del_cost = 0.5 # cost of deletion
    ins_cost = 0.5 # cost of insertion
    sub_cost = 1.0 # cost of substitution

    for i in range(1, n+1):
        for j in range(1, m+1):
            distance = []
            distance.append(D[i-1, j] + del_cost)
            distance.append(D[i, j-1] + ins_cost)
            distance.append(D[i-1, j-1] + (0 if source[i-1] == target[j-1] else sub_cost))
            D[i, j] = func(distance) # final edit distance value
    return D[n, m]

@lru_cache(maxsize = 128)
def lcsr(word1, word2):
    """Given two words, return the least common subsequence ratio between them."""
    den = max(len(word1), len(word2))
    num =  int(den - edit_distance(word1, word2))
    return num/den

@lru_cache(maxsize = 128)
def PREFIX(word1, word2):
    """Given two words, get the longest common prefix coefficient"""
    den = max(len(word1), len(word2))
    num = 0
    for c1, c2 in zip(word1, word2):
        if(c1 == c2):
            num += 1
        else:
            break
    return num/den

@lru_cache(maxsize = 128)
def dice_coefficient(word1, word2):
    """Given two words, return their dice coefficent:
        number of shared character bigams / total number of bigrams in both words"""
    if len(word1) < 2 or len(word2) < 2:
        return 0
    den = len(word1) - 1 + len(word2) - 1
    bigrams1 = [word1[i:i+2] for i, _ in enumerate(word1) if i+1 < len(word1)]
    bigrams2 = [word2[i:i+2] for i, _ in enumerate(word2) if i+1 < len(word2)]
    num = len([b for b in bigrams1 if b in bigrams2])
    return num/den

def extract_features(word1, word2):
    features = {
        'lcsr': lcsr(word1, word2),
        'PREFIX': PREFIX(word1, word2),
        'dice_coefficient': dice_coefficient(word1, word2),
        'soundex': soundex.compare(word1, word2),
        'nysiis': lcsr(nysiis(word1), nysiis(word2))
    }
    return features

TRAIN_PATH = '../data/cognet_train.csv'
TEST_PATH = '../data/cognet_test.csv'
DEV_PATH = '../data/cognet_dev.csv'
DATA_PATH = '../data/extracted_features.npy'

#%% FEATURE EXTRACTION

v = DictVectorizer(sparse=False)
soundex = Soundex()

print('Reading training data...')
train_data = pd.read_csv(TRAIN_PATH)
print('Extracting features...')
x_train = v.fit_transform([extract_features(str(word1), str(word2)) for word1, word2 in zip(train_data['word 1'], train_data['word 2'])])
y_train = [y for y in train_data['class']]

print('Reading testing data...')
test_data = pd.concat([pd.read_csv(DEV_PATH), pd.read_csv(TEST_PATH)])
print('Extracting features...')
x_test = v.fit_transform([extract_features(str(word1), str(word2)) for word1, word2 in zip(test_data['word 1'], test_data['word 2'])])
y_test = [y for y in test_data['class']]

#%% SAVE EXTRACTED FEATURES

with open(DATA_PATH, 'wb+', encoding='utf-8') as f:
    np.save(f, x_train)
    np.save(f, y_train)
    np.save(f, x_test)
    np.save(f, y_test)
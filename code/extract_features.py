# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 15:51:57 2021

@author: Christian Konstantinov
@author: Moeez Sheikh
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from soundex import Soundex
from jellyfish import nysiis
from functools import lru_cache
import epitran
import pickle
from collections import defaultdict

@lru_cache(maxsize=128)
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

@lru_cache(maxsize=128)
def lcsr(word1, word2):
    """Given two words, return the least common subsequence ratio between them."""
    den = max(len(word1), len(word2))
    num =  int(den - edit_distance(word1, word2))
    return num/den

@lru_cache(maxsize=128)
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

@lru_cache(maxsize=128)
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

@lru_cache(maxsize=128)
def get_translit(lang, word):
    trans = epitran_dict[lang].transliterate(word, ligatures=True)
    if trans: return trans
    return word

@lru_cache(maxsize=128)
def manhattan_distance(vector1, vector2):
    """given two ndarrays of the same shape, return their manhattan distance."""
    return np.sum(np.abs(vector1 - vector2))

@lru_cache(maxsize=128)
def set_to_length(word, n):
    """set a word to length n by either underscore padding or clipping."""
    if len(word) > n:
        return clip(word, n)
    if len(word) < n:
        return pad_to_length(word, n)
    return word

@lru_cache(maxsize=128)
def clip(word, n):
    """clip a word to length n."""
    return word[:n]

@lru_cache(maxsize=128)
def pad_to_length(word, n):
    """Pad a word with n underscores."""
    return word + ''.join(['_']*(n-len(word)))

@lru_cache(maxsize=128)
def pad(word1, word2):
    """Pad the smaller word with underscores."""
    l1 = len(word1)
    l2 = len(word2)
    if l1 == l2:
        return word1, word2
    if l1 > l2:
        return word1, pad_to_length(word2, l1)
    return pad_to_length(word1, l2), word2

def get_phoneme_encodings(word_ipa, encodings):
    result = []
    for phoneme in word_ipa:
        result.append(encodings[phoneme])
    return np.array(result).T

def create_epitran_dict():
    """Return a dictionary of languages to Epitran Objects."""
    codes = pd.read_csv(SUPPORTED_LANGS_PATH, sep='\t', header=0, error_bad_lines=False)['Code']
    epitran_dict = {}
    for code in codes:
        if code[:3] in epitran_dict: continue
        try:
            epitran_dict[code[:3]] = epitran.Epitran(f'{code}')
        except OSError:
            continue
    return epitran_dict

def extract_features(lang1, word1, lang2, word2):
    features = {
        'lcsr': lcsr(word1, word2),
        'PREFIX': PREFIX(word1, word2),
        'dice_coefficient': dice_coefficient(word1, word2),
        'soundex': soundex.compare(word1, word2),
        'nysiis': lcsr(nysiis(word1), nysiis(word2)),
        'epitran': lcsr(get_translit(lang1, word1), get_translit(lang2, word2))
    }
    return features

def extract_features_phonetic_only(lang1, word1, lang2, word2):
    tl1 = get_translit(lang1, word1)
    tl2 = get_translit(lang2, word2)
    pad1, pad2 = pad(tl1, tl2)
    phonemes1 = np.array([ipa[c] for c in pad1])
    phonemes2 = np.array([ipa[c] for c in pad2])
    features = {
        'lcsr': lcsr(tl1, tl2),
        'PREFIX': PREFIX(tl1, tl2),
        'dice_coefficient': dice_coefficient(tl1, tl2),
        'manhattan_distance': manhattan_distance(phonemes1, phonemes2)
    }
    return features

def extract_phoneme_encodings(lang1, word1, lang2, word2, n=10):
    tl1 = get_translit(lang1, word1)
    tl2 = get_translit(lang2, word2)
    pad1 = set_to_length(tl1, n)
    pad2 = set_to_length(tl2, n)
    phonemes1 = np.array([ipa[c] for c in pad1])
    phonemes2 = np.array([ipa[c] for c in pad2])
    feature_vector = np.stack((phonemes1, phonemes2))
    return feature_vector

#%% Open some things

TRAIN_PATH = '../data/cognet_train.csv'
TEST_PATH = '../data/cognet_test.csv'
DEV_PATH = '../data/cognet_dev.csv'
DATA_PATH = '../data/extracted_features.npy'
SUPPORTED_LANGS_PATH = '../data/cognet_supported_langs.tsv'
IPA_ENCODING_PATH = '../data/ipa_encodings.pickle'

v = DictVectorizer(sparse=False)
soundex = Soundex()
epitran_dict = create_epitran_dict()
with open(IPA_ENCODING_PATH, 'rb') as f:
    ipa = pickle.load(f)
k = len(list(ipa.values())[0]) # always 24
ipa = defaultdict(lambda: np.array([0.]*k), ipa)

#%% FEATURE EXTRACTION

print('Reading training data...')
train_data = pd.read_csv(TRAIN_PATH)
print('Extracting features...')
x_train = v.fit_transform([
    extract_features(str(lang1), str(word1), str(lang2), str(word2))\
        for lang1, word1, lang2, word2 in\
            zip(train_data['lang 1'], train_data['translit 1'], train_data['lang 2'], train_data['translit 2'])
            ])
y_train = [y for y in train_data['class']]

print('Reading testing data...')
test_data = pd.concat([pd.read_csv(DEV_PATH), pd.read_csv(TEST_PATH)])
print('Extracting features...')
x_test = v.fit_transform([
    extract_features(str(lang1), str(word1), str(lang2), str(word2))\
        for lang1, word1, lang2, word2 in\
            zip(test_data['lang 1'], test_data['translit 1'], test_data['lang 2'], test_data['translit 2'])
            ])
y_test = [y for y in test_data['class']]

#%% PHONEMES

# phoneme vector shape = (|C|, 2, n, k)
# where |C| is the number of cognate pairs in the dataset
# and n is the number of characters for each word, which is set to 10
# and k is the length of the phoneme vector which is always 24

print('Extracting phonemes...')
x_train_phonemes = np.array([
    extract_phoneme_encodings(str(lang1), str(word1), str(lang2), str(word2))\
        for lang1, word1, lang2, word2 in\
            zip(train_data['lang 1'], train_data['word 1'], train_data['lang 2'], train_data['word 2'])
            ]).transpose((0,1,3,2))

x_test_phonemes = np.array([
    extract_phoneme_encodings(str(lang1), str(word1), str(lang2), str(word2))\
        for lang1, word1, lang2, word2 in\
            zip(test_data['lang 1'], test_data['word 1'], test_data['lang 2'], test_data['word 2'])
            ]).transpose((0,1,3,2))

print('done!')

#%% SAVE EXTRACTED FEATURES

with open(DATA_PATH, 'wb+') as f:
    np.save(f, x_train)
    np.save(f, x_train_phonemes)
    np.save(f, y_train)
    np.save(f, x_test)
    np.save(f, x_test_phonemes)
    np.save(f, y_test)

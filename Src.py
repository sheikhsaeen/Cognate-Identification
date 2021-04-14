# !pip install soundex
# !pip install jellyfish

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import editdistance
import os
import soundex
import jellyfish

def lcsr(word1, word2):
    """Given two words, return the least common subsequence ratio between them."""
    den = max(len(word1), len(word2))
    num =  int(den - editdistance.distance(word1, word2))
    return num/den

def prefix(strs):
  return len(os.path.commonprefix(strs))/max(len(strs[0]),len(strs[1]))

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
        'PREFIX': prefix([word1,word2]),
        'dice_coefficient': dice_coefficient(word1, word2),
        'edit_distance': editdistance.distance(word1,word2),
        'soundex': soundex.Soundex().compare(word1,word2),
        'nysiis': editdistance.distance(jellyfish.nysiis(word1),jellyfish.nysiis(word2))
    }
    return features

TRAIN_PATH = 'cognet_train.csv'
TEST_PATH = 'cognet_test.csv'

#%% DATASET CONSTRUCTION

v = DictVectorizer(sparse=False)

print('Reading training data...')
train_data = pd.read_csv(TRAIN_PATH)
print('Extracting features...')
x_train = v.fit_transform([extract_features(str(word1), str(word2)) for word1, word2 in zip(train_data['word 1'], train_data['word 2'])])
y_train = [y for y in train_data['class']]

print('Reading testing data...')
test_data = pd.read_csv(TEST_PATH)
print('Extracting features...')
x_test = v.fit_transform([extract_features(str(word1), str(word2)) for word1, word2 in zip(test_data['word 1'], test_data['word 2'])])
y_test = [y for y in test_data['class']]

#%% TRAINING

clf = MLPClassifier(hidden_layer_sizes=(48,48,24), max_iter=10, alpha=0.0001,
                    verbose=10,  random_state=21, tol=0.000000001,learning_rate='adaptive')

print('Started training...')
clf.fit(x_train, y_train)
print('Finished training...')

print('Making some predictions...')
y_pred = clf.predict(x_test)

print('-------------------------------------------------\nEvalutation\n\
        -------------------------------------------------\n')

print(f'Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%')

precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
print(f'F-score: {fscore*100:.2f}%')
print(f'precision: {precision*100:.2f}%')
print(f'recall: {recall*100:.2f}%')

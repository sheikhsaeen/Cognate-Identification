# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 21:21:15 2021

@author: Christian Konstantinov
"""
import numpy as np
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

DATA_PATH = '../data/extracted_features.npy'

#%% LOAD DATA

with open(DATA_PATH, 'rb') as f:
    x_train = np.load(f, allow_pickle=True)
    x_train_phonemes = np.load(f, allow_pickle=True)
    y_train = np.load(f, allow_pickle=True)
    x_test = np.load(f, allow_pickle=True)
    x_test_phonemes = np.load(f, allow_pickle=True)
    y_test = np.load(f, allow_pickle=True)

#%% SHUFFLE IT UP, HOMIE

x_train, y_train = shuffle(x_train, y_train)
x_test, y_test = shuffle(x_test, y_test)

#%% TRAINING AND EVALUATION

clf = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=50, early_stopping=True,
                    validation_fraction=0.5,verbose=True, solver='adam',
                    random_state=8, learning_rate='adaptive')

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

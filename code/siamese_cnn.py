# -*- coding: utf-8 -*-
"""
Created on Sat May  1 22:02:47 2021

@author: Christian Konstantinov
@author: Moeez Sheikh
"""
import torch
from torch import nn
import torch.optim as optim
import extract_features as ef
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

#%% Pleb Check

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% The model

class SiameseCNN(nn.Module):
    def __init__(self, h, m, k, n, hidden_size):
        super().__init__()
        self.h = h # filter length
        self.m = m # filter width
        self.k = k # phoneme vector size
        self.n = n # word length
        self.p = k - h + 1
        self.q = n - m + 1
        self.stride = 2

        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=self.p*self.q, kernel_size=(h, m))
        self.conv_2 = nn.Conv2d(in_channels=self.p*self.q, out_channels=self.p*self.q, kernel_size=(h, m))
        self.conv_3 = nn.Conv2d(in_channels=self.p*self.q, out_channels=self.p*self.q, kernel_size=(h, m))
        self.maxpool = nn.Maxpool2d((self.p, self.q), stride=self.stride)
        self.manhattan = ef.manhattan_distance

        self.input_layer = nn.Linear(7, hidden_size)
        self.hidden_1 = nn.Linear(hidden_size, hidden_size)
        self.hidden_2 = nn.Linear(hidden_size, hidden_size)
        self.hidden_3 = nn.Linear(hidden_size, hidden_size)

        self.output = nn.Linear(hidden_size, 1)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        feature, phoneme = x

        # Siamese CNN
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.maxpool(x)
        x = self.sigmoid(x)
        x = self.manhattan()

        # Fully Connected
        x = self.input_layer(x)
        x = self.sigmoid(x)
        x = self.hidden_1(x)
        x = self.sigmoid(x)
        x = self.hidden_2(x)
        x = self.sigmoid(x)
        x = self.hidden_3(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
        return x

#%% Load the data

IPA_ENCODING_PATH = '../data/ipa_encodings.pickle'
DATA_PATH = '../data/extracted_features.npy'

with open(IPA_ENCODING_PATH, 'rb') as f:
    ipa = pickle.load(f)

with open(DATA_PATH, 'rb') as f:
    x_train = np.load(f, allow_pickle=True)
    x_train_phonemes = np.load(f, allow_pickle=True)
    y_train = np.load(f, allow_pickle=True)
    x_test = np.load(f, allow_pickle=True)
    x_test_phonemes = np.load(f, allow_pickle=True)
    y_test = np.load(f, allow_pickle=True)

train_data = zip(zip(x_train, x_train_phonemes), y_train)
test_data = zip(zip(x_test, x_test_phonemes), y_test)

#%% Hyperparameters

n = 10
k = len(list(ipa.values())[0])
# h <= k
h = k
# m < n
m = 2

hidden_size = 3
lr = 0.001
epochs = 10

#%% Initialization

model = SiameseCNN(h, m, k, n).to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

#%% functions

def evaluate(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)*100
    print(f'Accuracy: {accuracy:.2f}%')

    precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    print(f'F-score: {fscore*100:.2f}%')
    print(f'precision: {precision*100:.2f}%')
    print(f'recall: {recall*100:.2f}%')
    return accuracy, fscore, precision, recall

def train(model, train_data, optimizer, loss_function):
    for epoch in range(epochs):
        print(f'Starting epcoh {epoch}...')
        for (feature_vec, phoneme_vec), target in train_data:
            model.zero_grad()
            x = (torch.FloatTensor(torch.tensor(feature_vec).float().to(device)), torch.FloatTensor(torch.tensor(phoneme_vec).float().to(device)))
            output = model(x)
            loss = loss_function(output, torch.FloatTensor(target.float()).to(device))
            loss.backward()
            optimizer.step()

def test(model, test_data):
    predictions = []
    targets = []
    with torch.no_grad():
        for (feature_vec, phoneme_vec), target in test_data:
            x = (torch.FloatTensor(torch.tensor(feature_vec).float().to(device)), torch.FloatTensor(torch.tensor(phoneme_vec).float().to(device)))
            output = model(x)
            label = round(output[0])
            predictions.append(label)
            targets.append(target)
    predictions = np.array(predictions)
    _ = evaluate(targets, predictions)
    return targets, predictions

#%% Run

train(model, train_data, optimizer, loss_function)
test(model, test_data)

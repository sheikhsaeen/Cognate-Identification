# -*- coding: utf-8 -*-
"""
Created on Sat May  1 22:02:47 2021

@author: Christian Konstantinov
@author: Moeez Sheikh
"""
import torch
from torch import nn
import torch.optim as optim
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

# TODO: Make custom dataset class
# TODO: Implement batching

#%% Pleb Check

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% The model

class SiameseCNN(nn.Module):
    def __init__(self, h, m, k, n):
        super().__init__()
        self.h = h # filter length
        self.m = m # filter width
        self.k = k # phoneme vector size
        self.n = n # word length
        self.p = k - h + 1
        self.q = n - m + 1
        self.stride = 2

        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=self.p*self.q, kernel_size=(h, m))
        self.conv_2 = nn.Conv2d(in_channels=self.p*self.q, out_channels=self.p*self.q, kernel_size=(1, m))
        self.maxpool = nn.MaxPool2d((self.p, self.q), stride=self.stride)
        self.relu = nn.ReLU()
        self.drop_1 = nn.Dropout(p=0.1)
        self.siamese = nn.Sequential(self.conv_1, self.relu, self.drop_1,
                                     self.conv_2, self.relu, self.drop_1)

        self.hidden_1 = nn.Linear(7, 64)
        self.hidden_2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        self.drop_2 = nn.Dropout(p=0.5)

        self.fully_connected = nn.Sequential(self.hidden_1, self.relu, self.drop_2,
                                             self.hidden_2, self.relu, self.drop_2,
                                             self.output, self.relu, self.drop_2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        distances, phonemes = x
        phonemes_1 = phonemes[0].unsqueeze(0).unsqueeze(0)
        phonemes_2 = phonemes[1].unsqueeze(0).unsqueeze(0)

        # Siamese CNN
        phonemes_1 = self.siamese(phonemes_1)
        phonemes_2 = self.siamese(phonemes_2)
        phonetic_distance = self.manhattan(phonemes_1, phonemes_2)

        x = torch.cat((distances, phonetic_distance.reshape(1)))

        # Fully Connected
        x = self.fully_connected(x)
        x = self.sigmoid(x)
        return x

    def manhattan(self, tensor_1, tensor_2):
        """Given 2 tensors, return their manhattan distance."""
        return torch.sum(torch.abs(tensor_1 - tensor_2))

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
k = len(list(ipa.values())[0]) # 24
# h <= k
h = k
# m < n
m = 2

hidden_size = 3
lr = 0.001
epochs = 10

#%% Initialization

model = SiameseCNN(h, m, k, n).to(device)
loss_function = nn.BCELoss()
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
        print(f'Starting epoch {epoch}...')
        for (feature_vec, phoneme_vec), target in train_data:
            model.zero_grad()
            x = (torch.FloatTensor(torch.tensor(feature_vec).float()).to(device), torch.FloatTensor(torch.tensor(phoneme_vec).float()).to(device))
            output = model(x)
            loss = loss_function(output, torch.FloatTensor(torch.tensor(target).reshape(1).float()).to(device))
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
if __name__ == '__main__':
    train(model, train_data, optimizer, loss_function)
    test(model, test_data)

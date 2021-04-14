# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 18:01:44 2021

@author: Christian Konstantinov
"""
import pandas as pd

DATASET_PATH = '../data/CogNet-v2.0.tsv'
SUPPORTED_LANGS_PATH = '../data/epitran_supported_langs.tsv'

# read in languages that epitran supports
langs = pd.read_csv(SUPPORTED_LANGS_PATH, sep='\t', header=0, error_bad_lines=False)
langs = [l[:3] for l in langs['Code']]

# read in the data from CogNet and format it into a dataframe.
with open(DATASET_PATH, encoding='utf-8') as data:
    next(data)
    header = next(data)
    f = lambda s: [x for x in s.strip('\n').split('\t')]
    d = []
    for line in data:
        row = f(line)
        if not (row[1] in langs and row[3] in langs):
            continue
        # replace word 1 with translit 1 if the latter exists.
        if row[6]:
                row[2] = row[6]
        # replace word 2 with translit 2 if the latter exists.
        if row[7]:
                row[4] = row[7]
        d.append(row)
    df = pd.DataFrame.from_records(d, columns=f(header))
    d = []
    df.drop('provenance', axis=1, inplace=True)
    df.drop('translit 1', axis=1, inplace=True)
    df.drop('translit 2', axis=1, inplace=True)
    df.drop('concept id', axis=1, inplace=True)

#%% false cognate generation

# df holds true cognates, now shuffle the data to generate false cognates.
seed = 10 # completely arbitrary; chosen to produce a (relatively) small number of false negatives.
false_cognates = df.copy()
false_cognates['lang 1'] = df['lang 1'].sample(frac=1, random_state=seed).values
false_cognates['word 1'] = df['word 1'].sample(frac=1, random_state=seed).values

true_pairs = []
false_pairs = {}

for i, (w1, w2) in enumerate(zip(df['word 1'], df['word 2'])):
    true_pair = [w1, w2]
    true_pair.sort()
    true_pairs.append(true_pair)
    false_pair = [false_cognates['word 1'][i], false_cognates['word 2'][i]]
    false_pair.sort()
    false_pairs[tuple(false_pair)] = 1

#%% false cognate validation

# validate that the false cognates are actually false,
# and remove any false negatives from the data.
to_remove = []
for i, tp in enumerate(true_pairs):
    try:
        false_pairs[tuple(tp)]
        to_remove.append(i)
    except KeyError:
        continue
false_cognates.drop(index=to_remove, inplace=True)
true_pairs = []
false_pairs = {}

#%% data splitting

# split the data 80/10/10
# start with the true cognates
train_true = df.sample(frac=0.8)
df.drop(train_true.index, inplace=True)
test_true = df.sample(frac=0.5)
dev_true = df.drop(test_true.index)

# now do the false cognates
train_false = false_cognates.sample(frac=0.8)
false_cognates.drop(false_cognates.index, inplace=True)
test_false = false_cognates.sample(frac=0.5)
dev_false = false_cognates.drop(test_false.index)

# add the classes (true or false)
train_true['class'] = 1
test_true['class'] = 1
dev_true['class'] = 1
train_false['class'] = 0
test_false['class'] = 0
dev_false['class'] = 0

# now concatenate them
train = pd.concat([train_true, train_false])
test = pd.concat([test_true, test_false])
dev = pd.concat([dev_true, dev_false])

#%% file writing

TRAIN_FILE = '../data/cognet_train.csv'
TEST_FILE = '../data/cognet_test.csv'
DEV_FILE = '../data/cognet_dev.csv'

with open(TRAIN_FILE, 'w+', encoding='utf-8', newline='') as f:
    train.to_csv(path_or_buf=f)

with open(TEST_FILE, 'w+', encoding='utf-8', newline='') as f:
    test.to_csv(path_or_buf=f)

with open(DEV_FILE, 'w+', encoding='utf-8', newline='') as f:
    dev.to_csv(path_or_buf=f)

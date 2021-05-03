# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 18:01:44 2021

@author: Christian Konstantinov
"""
import pandas as pd

DATASET_PATH = '../data/CogNet-v2.0.tsv'
SUPPORTED_LANGS_PATH = '../data/cognet_supported_langs.tsv'

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

        # replace translit 1 with word 1 if the former does not exist.
        if not row[6]:
            row[6] = row[2]
        # replace translit 2 with word 2 if the former does not exist.
        if not row[7]:
            row[7] = row[4]

        d.append(row)
    df = pd.DataFrame.from_records(d, columns=f(header))
    d = []
    df.drop('provenance', axis=1, inplace=True)
    df.drop('concept id', axis=1, inplace=True)

#%% false cognate generation

# df holds true cognates, now shuffle the data to generate false cognates.
seed = 10 # completely arbitrary; chosen to produce a (relatively) small number of false negatives.
false_cognates = df.copy()
false_cognates['lang 1'] = df['lang 1'].sample(frac=1, random_state=seed).values
false_cognates['word 1'] = df['word 1'].sample(frac=1, random_state=seed).values
false_cognates['translit 1'] = df['translit 1'].sample(frac=1, random_state=seed).values

true_pairs = []
false_pairs = {}

for i, (w1, w2) in enumerate(zip(df['word 1'], df['word 2'])):
    true_pairs.append((w1, w2))
    false_pair = (false_cognates['word 1'][i], false_cognates['word 2'][i])
    false_pairs[false_pair] = 1

#%% false cognate validation

# validate that the false cognates are actually false,
# and remove any false negatives from the data.
to_remove = []
for i, tp in enumerate(true_pairs):
    try:
        false_pairs[tp]
        to_remove.append(i)
    except KeyError:
        continue
false_cognates.drop(index=to_remove, inplace=True)

#%% data splitting

# split the data 80/10/10
# start with adding classes
df['class'] = 1
false_cognates['class'] = 0

# split the true cognates
train_true = df.sample(frac=0.8, random_state=seed)
df.drop(train_true.index, inplace=True)
test_true = df.sample(frac=0.5, random_state=seed)
df.drop(test_true.index, inplace=True)
dev_true = df

# now do the false cognates
train_false = false_cognates.sample(frac=0.8, random_state=seed)
false_cognates.drop(train_false.index, inplace=True)
test_false = false_cognates.sample(frac=0.5, random_state=seed)
false_cognates.drop(test_false.index, inplace=True)
dev_false = false_cognates

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

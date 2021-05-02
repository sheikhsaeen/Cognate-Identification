# -*- coding: utf-8 -*-
"""
Created on Sun May  2 01:12:43 2021

@author: Christian Konstantinov
"""

import numpy as np
import pandas as pd
import extract_features as ef
from extract_features import ipa

def main():
    TRAIN_PATH = '../data/cognet_train_no_translit.csv'
    TEST_PATH = '../data/cognet_test_no_translit.csv'
    DEV_PATH = '../data/cognet_dev_no_translit.csv'
    DATA_PATH = '../data/phoneme_encodings.npy'

    n = 10

    train_data = pd.read_csv(TRAIN_PATH)
    train_phoneme_encodings = np.array([\
    [ef.get_phoneme_encodings(ef.get_translit(str(lang1), ef.set_to_length(str(word1), n)), ipa),
     ef.get_phoneme_encodings(ef.get_translit(str(lang2), ef.set_to_length(str(word2), n)), ipa)]
            for lang1, word1, lang2, word2 in\
                zip(train_data['lang 1'], train_data['word 1'], train_data['lang 2'], train_data['word 2'])
                ])

    test_data = pd.concat([pd.read_csv(DEV_PATH), pd.read_csv(TEST_PATH)])

    test_phoneme_encodings = np.array([\
    [ef.get_phoneme_encodings(ef.get_translit(str(lang1), ef.set_to_length(str(word1)), n), ipa),
     ef.get_phoneme_encodings(ef.get_translit(str(lang2), ef.set_to_length(str(word2)), n), ipa)]
            for lang1, word1, lang2, word2 in\
                zip(test_data['lang 1'], test_data['word 1'], test_data['lang 2'], test_data['word 2'])
                ])

    with open(DATA_PATH, 'wb+') as f:
        np.save(f, train_phoneme_encodings)
        np.save(f, test_phoneme_encodings)

if __name__ == '__main__':
    main()
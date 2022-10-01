# -*- coding: utf-8 -*-
"""
Created on Sat May  1 17:55:22 2021

@author: Christian Konstantinov
"""
import pandas as pd
import pickle

def convert_value(x):
    """Convert the string into a float."""
    if x == '-':
        return -1.
    if x == '+':
        return 1.
    if x == '0':
        return 0.
    else:
        return x

def convert_data_frame(df):
    """Convert the data read in from panphon's ipa encodings to floating point."""
    return df.applymap(convert_value)

if __name__ == '__main__':
    IPA_DATA_PATH = '../data/ipa_all.csv'
    IPA_ENCODING_PATH = '../data/ipa_encodings.pickle'

    # Read in and convert the data to floating point numbers.
    df = pd.read_csv(IPA_DATA_PATH, header=0)
    df = convert_data_frame(df)

    # Create a dictionary mapping IPA characters to their vector representations.
    # First, get the IPA characters, and put them into a dict.
    ipa_df = df['ipa'].to_frame().to_dict()
    ipa_dict = {v: k for k, v in ipa_df['ipa'].items()}

    # Now get the vectors and store them as dict values.
    vector = df.drop('ipa', axis=1).to_numpy()
    for i, (k, _) in enumerate(ipa_dict.items()):
        ipa_dict[k] = vector[i].astype('float32')

    # Lastly, pickle the dict for later use.
    with open(IPA_ENCODING_PATH, 'wb+') as f:
        pickle.dump(ipa_dict, f)

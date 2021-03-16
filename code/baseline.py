# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 21:21:15 2021

@author: Christian Konstantinov
"""
import pandas as pd
import epitran

DATASET_PATH = '../data/CogNet-v2.0.tsv'
SUPPORTED_LANGS_PATH = '../data/epitran_supported_langs.tsv'

df = pd.read_csv(DATASET_PATH, sep='\t', header=1, error_bad_lines=False)
langs = pd.read_csv(SUPPORTED_LANGS_PATH, sep='\t', header=0, error_bad_lines=False)

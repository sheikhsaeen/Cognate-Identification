@author: Christian Konstantinov
@author: Moeez Sheikh

# Impporting libraries
import numpy as np
import pandas as pd
import sklearn_crfsuite
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import editdistance
import os
import soundex
import jellyfish
from collections import defaultdict
import epitran

def get_translit(lang, word):
    trans = epitran_dict[lang].transliterate(word, ligatures=True)
    if trans: return trans
    return word

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

# Feature Extraction
def extract_features(word1, word2,lang1,lang2):
    features = {
        'lcsr': lcsr(word1, word2),
        'PREFIX': prefix([word1,word2]),
        'dice_coefficient': dice_coefficient(word1, word2),
        'soundex': soundex.Soundex().compare(word1,word2),
        'nysiis': lcsr(jellyfish.nysiis(word1), jellyfish.nysiis(word2)),
        'epitran': lcsr(get_translit(lang1, word1), get_translit(lang2, word2))    }
    return features

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

def words_to_features(word_1,word_2,lang1,lang2):
	return [extract_features(word_1, word_2,lang1,lang2)]

def cogdentify(word1,word2,lang1,lang2,classifier):
  res = classifier.predict_single([extract_features(word1,word2,lang1,lang2)])
  print(f'Word: {word1.title()} from {lang1}\nTransliteration of {word1.title()}: {epitran_dict[lang1].transliterate(word1)}\nWord: {word2.title()} form {lang2}\nTransliteration of {word2.title()}: {epitran_dict[lang2].transliterate(word2)}')
  if res[0] == "T":
    print("These words are cognates.")
  else:
    print("These words are not cognates.")

# Data paths
TRAIN_PATH = '../data/cognet_train.csv'
TEST_PATH = '../data/cognet_test.csv'
SUPPORTED_LANGS_PATH = '../data/cognet_supported_langs.tsv'

epitran_dict = create_epitran_dict()

# Reading Data
print("Reading files...")

train_data = pd.read_csv(TRAIN_PATH)
test_data = pd.read_csv(TEST_PATH)
train_data = train_data.dropna()
test_data = test_data.dropna()

print("Finished reading the files")

# Train and Test set creation
print("Extracting festures and making train and test set...")

X_train = [words_to_features(word1,word2,lang1,lang2) for word1,word2,lang1,lang2 in train_data[['word 1','word 2','lang 1','lang 2']].values]
y_train =  ['T' if cls == 1 else 'F' for cls in train_data['class']] # Because CRF needs a character or a string in the target variable.

X_test = [words_to_features(word1,word2,lang1,lang2) for word1,word2,lang1,lang2 in test_data[['word 1','word 2','lang 1','lang 2']].values]
y_test = ['T' if cls == 1 else 'F' for cls in test_data['class']]

print("Extracting Completed...")

# Training
crf = sklearn_crfsuite.CRF(
	algorithm='lbfgs', 
	max_iterations=150, 
	all_possible_transitions=True
)
print("Starting training...")
crf.fit(X_train, y_train)
print("Finished training...")

# Evaluation
print("Starting Predictions on test...")
labels = list(crf.classes_)
y_pred = crf.predict(X_test)
print("Finished prediciting...")

print(f'Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%')

precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average='binary',pos_label='T')
print(f'F-score: {fscore*100:.2f}%')
print(f'precision: {precision*100:.2f}%')
print(f'recall: {recall*100:.2f}%')

# Wrong classifications produce by the model.
wrong_ones = []
right_ones = 0
for words,y in zip(test_data[['word 1','word 2','lang 1','lang 2']].values,y_test):
  right_ones += 1
  if crf.predict_single([extract_features(words[0],words[1],words[2],words[3])])[0] != y:
    wrong_ones.append((words[0],words[1],y))

# Loop for user input.
while True:
  word1 = input("Please enter the first word: ")
  lang1 = input("Please enter the language of the first word: ")
  word2 = input("Please enter the second word: ")
  lang2 = input("Please enter the language of the second word: ")
  cogdentify(word1,word2,lang1,lang2,crf)
  cont = input("Do you want to enter another pair? Press y for yes and n for no: ")
  if cont == 'y':
    continue
  break

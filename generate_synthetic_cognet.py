import csv
import random

# This script generates synthetic data files mimicking the output
# of create_dataset.py for quick testing.

LANGS = {
    'eng': ['house', 'cat', 'dog', 'tree', 'sun', 'water', 'fire', 'earth', 'air', 'stone'],
    'deu': ['haus', 'katze', 'hund', 'baum', 'sonne', 'wasser', 'feuer', 'erde', 'luft', 'stein'],
    'fra': ['maison', 'chat', 'chien', 'arbre', 'soleil', 'eau', 'feu', 'terre', 'air', 'pierre'],
    'spa': ['casa', 'gato', 'perro', 'arbol', 'sol', 'agua', 'fuego', 'tierra', 'aire', 'piedra'],
    'ita': ['casa', 'gatto', 'cane', 'albero', 'sole', 'acqua', 'fuoco', 'terra', 'aria', 'pietra'],
}

HEADER = [
    'lang 1', 'word 1', 'translit 1',
    'lang 2', 'word 2', 'translit 2',
    'class'
]

random.seed(42)

def sample_true_pair():
    """Return a tuple representing a cognate pair."""
    lang1, lang2 = random.sample(list(LANGS.keys()), 2)
    idx = random.randrange(len(LANGS[lang1]))
    word1 = LANGS[lang1][idx]
    word2 = LANGS[lang2][idx]
    return [lang1, word1, word1, lang2, word2, word2, 1]

def sample_false_pair():
    """Return a tuple representing a non-cognate pair."""
    lang1, lang2 = random.sample(list(LANGS.keys()), 2)
    word1 = random.choice(LANGS[lang1])
    word2 = random.choice(LANGS[lang2])
    return [lang1, word1, word1, lang2, word2, word2, 0]

def create_dataset(n_pairs=1000, train_ratio=0.8, test_ratio=0.1):
    """Generate train/dev/test CSV files with synthetic data."""
    n_true = n_pairs // 2
    n_false = n_pairs - n_true

    data = [sample_true_pair() for _ in range(n_true)]
    data += [sample_false_pair() for _ in range(n_false)]
    random.shuffle(data)

    n_train = int(len(data) * train_ratio)
    n_test = int(len(data) * test_ratio)

    train = data[:n_train]
    test = data[n_train:n_train + n_test]
    dev = data[n_train + n_test:]

    for fname, rows in [
        ('data/cognet_train.csv', train),
        ('data/cognet_test.csv', test),
        ('data/cognet_dev.csv', dev),
    ]:
        with open(fname, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)
            writer.writerows(rows)

if __name__ == '__main__':
    create_dataset()

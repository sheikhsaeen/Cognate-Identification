# Cognate Identification

This project explores various NLP approaches for identifying cognates across languages. The repository contains scripts for feature extraction, baseline models, and a Siamese CNN prototype.

## Goal
Our aim is to experiment with different methods for cognate detection and evaluate their performance. A reliable classifier can help linguists trace how languages evolved, discover word origins and preserve endangered languages.

## Synthetic Test Data
The repository does not ship the original CogNet dataset. To quickly test the code, you can generate synthetic data using `generate_synthetic_cognet.py`:

```bash
python generate_synthetic_cognet.py
```

This script creates `cognet_train.csv`, `cognet_test.csv` and `cognet_dev.csv` under the `data/` directory. The files contain randomly generated cognate and non-cognate pairs for a few languages so you can exercise the feature extraction and model scripts.

# Datasets

Training and evaluation used the following publicly available datasets. None are included in this repository — download before running experiments.

## C4 (Colossal Clean Crawled Corpus)
- HuggingFace: `allenai/c4`, English split
- Used for: NS-filtered fine-tuning ablations
- Download: `python dataset_download.py` (auto-downloads via HuggingFace datasets library)
- Size: ~300GB full; experiments used ~50K sampled examples

## TinyShakespeare
- Source: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
- Used for: small-scale ablation runs and sanity checks
- Download: `wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt` 

## Dependencies
```bash
pip install datasets transformers torch
```

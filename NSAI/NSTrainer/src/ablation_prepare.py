#!/usr/bin/env python3
"""Prepare C4 data for ablation: 10K samples, NS-scored by unigram surprisal, split into standard + NS-filtered."""
import json
import os
import argparse
import numpy as np
from collections import Counter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True, help="HF model path or name for tokenizer")
    parser.add_argument("--output-dir", default="/workspace/ablation_data")
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--retain-pct", type=float, default=0.13)
    args = parser.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    print(f"Loading {args.n_samples} C4 samples (streaming)...")
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    samples = []
    for i, ex in enumerate(ds):
        if i >= args.n_samples:
            break
        samples.append(ex["text"])
    print(f"Collected {len(samples)} samples")

    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    print("Tokenizing all samples...")
    tokenized = [tok.encode(s, add_special_tokens=False) for s in samples]

    # Compute unigram frequencies across all tokens
    print("Computing unigram statistics...")
    all_tokens = [t for tokens in tokenized for t in tokens]
    counts = Counter(all_tokens)
    total = len(all_tokens)
    log_counts = {t: np.log(c) for t, c in counts.items()}
    log_total = np.log(total)

    # Score each sample by average token surprisal
    print("Scoring samples by average unigram surprisal...")
    scores = []
    for i, tokens in enumerate(tokenized):
        if len(tokens) == 0:
            scores.append(0.0)
            continue
        surprisal = np.mean([log_total - log_counts[t] for t in tokens])
        scores.append(float(surprisal))

    scores = np.array(scores)

    # Sort by score descending, keep top retain_pct
    n_keep = int(len(samples) * args.retain_pct)
    sorted_idx = np.argsort(scores)[::-1]
    ns_indices = sorted_idx[:n_keep]

    # Save
    os.makedirs(args.output_dir, exist_ok=True)

    standard_path = os.path.join(args.output_dir, "standard_data.jsonl")
    with open(standard_path, "w") as f:
        for s in samples:
            f.write(json.dumps({"text": s}) + "\n")

    ns_path = os.path.join(args.output_dir, "ns_filtered_data.jsonl")
    with open(ns_path, "w") as f:
        for i in ns_indices:
            f.write(json.dumps({"text": samples[i], "score": float(scores[i])}) + "\n")

    print(f"\nStandard: {len(samples)} samples → {standard_path}")
    print(f"NS-filtered: {n_keep} samples ({args.retain_pct*100:.1f}%) → {ns_path}")
    print(f"Score range: {scores.min():.2f} - {scores.max():.2f}")
    print(f"Mean score: {scores.mean():.2f}")
    print(f"NS threshold (cutoff): {scores[sorted_idx[n_keep-1]]:.2f}")
    print(f"NS mean score: {scores[ns_indices].mean():.2f}")
    print(f"Standard mean score: {scores.mean():.2f}")

if __name__ == "__main__":
    main()

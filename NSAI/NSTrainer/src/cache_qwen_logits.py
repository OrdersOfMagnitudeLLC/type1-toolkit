#!/usr/bin/env python3
"""Universal teacher logit caching script.

Caches top-k logits from any GGUF model for NS bigram-filtered windows.
Uses the teacher's own tokenizer throughout - no external tokenizers.
"""

import os
import gc
import json
import argparse
import numpy as np
import torch
import ns_train


def main():
    parser = argparse.ArgumentParser(description="Cache teacher logits for NS-filtered windows")
    parser.add_argument("--model", required=True, help="Path to GGUF file")
    parser.add_argument("--model-name", required=True, help="Short tag for cache dir, e.g. 'qwen3b'")
    parser.add_argument("--data", default="data/c4_sample.txt", help="Path to text file")
    parser.add_argument("--n-windows", type=int, default=2945, help="Number of filtered windows to cache")
    parser.add_argument("--block-size", type=int, default=128, help="Tokens per window")
    parser.add_argument("--top-k", type=int, default=100, help="Logits to save per token position")
    parser.add_argument("--n-ctx", type=int, default=512, help="Context size for llama-cpp-python")
    parser.add_argument("--cache-dir", default=None, help="Output directory (default: teacher_logits_cache_{model_name}/)")
 parser.add_argument("--scorer", default="unigram", choices=["unigram", "bigram"], help="NS filter type (default: unigram: bigram is too sparse for large vocabs)")
    args = parser.parse_args()

    cache_dir = args.cache_dir or f"teacher_logits_cache_{args.model_name}"
    meta_path = os.path.join(cache_dir, "meta.json")

    # Step 1: Check if cache exists
    if os.path.exists(meta_path):
        print(f"Cache exists at {cache_dir}")
        with open(meta_path) as f:
            print(json.load(f))
        return

    # Step 2: Load GGUF model
    print(f"Loading {args.model}...")
    from llama_cpp import Llama

    llm = Llama(
        model_path=args.model,
        n_ctx=args.n_ctx,
        logits_all=True,
        n_gpu_layers=99,
        verbose=False,
    )
    vocab_size = llm.n_vocab()
    print(f"Loaded. vocab_size={vocab_size}")

    # Step 3: Tokenize data with teacher's own tokenizer
    print(f"Tokenizing {args.data}...")
    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()
    tokens = llm.tokenize(text.encode("utf-8"), add_bos=False, special=True)
    data = torch.tensor(tokens, dtype=torch.long)
    del text, tokens
    print(f"Total tokens: {len(data)}")

    # Step 4: Compute NS info and select top windows
    print(f"Computing NS {args.scorer} filter...")
    if args.scorer == "bigram":
        token_info = ns_train.compute_bigram_info(data, vocab_size, args.block_size)
    else:
        token_info = ns_train.compute_token_info(data, vocab_size)
    threshold = torch.quantile(token_info[data], 0.5)

    info_per_token = token_info[data]
    cs = torch.cat([torch.zeros(1), info_per_token.cumsum(dim=0)])
    window_sum = cs[args.block_size:] - cs[:-args.block_size]
    window_mean = window_sum / args.block_size
    valid = window_mean > threshold
    valid_idx = torch.nonzero(valid, as_tuple=True)[0]

    scores = window_mean[valid_idx]
    sorted_order = torch.argsort(scores, descending=True)
    n_take = min(args.n_windows, len(valid_idx))
    top_starts = valid_idx[sorted_order[:n_take]].tolist()

    print(f"NS filter: {len(valid_idx)} valid windows, taking top {n_take}")
    print(f"Score range: {scores[sorted_order[0]]:.4f} .. {scores[sorted_order[n_take-1]]:.4f}")

    n_tokens = n_take * args.block_size

    # Step 5: Prepare memmap output
    os.makedirs(cache_dir, exist_ok=True)
    top_indices = np.memmap(
        os.path.join(cache_dir, "top_indices.bin"),
        dtype=np.int32, mode="w+", shape=(n_tokens, args.top_k)
    )
    top_values = np.memmap(
        os.path.join(cache_dir, "top_values.bin"),
        dtype=np.float32, mode="w+", shape=(n_tokens, args.top_k)
    )
    window_tokens = np.memmap(
        os.path.join(cache_dir, "window_tokens.bin"),
        dtype=np.int32, mode="w+", shape=(n_take, args.block_size)
    )

    # Step 6: Process each window
    # Batch eval all tokens at once, then extract top-k from _scores.
    print(f"Processing {n_take} windows...")
    for wi, start in enumerate(top_starts):
        window_ids = data[start:start + args.block_size].tolist()

        llm.reset()
        llm.eval(window_ids)

        scores = llm._scores  # shape: (block_size, vocab_size)
        for ti in range(args.block_size):
            if ti >= scores.shape[0]:
                continue
            arr = scores[ti].astype(np.float32)

            k = min(args.top_k, len(arr))
            top_k_idx = np.argpartition(arr, -k)[-k:]
            top_k_vals = arr[top_k_idx]
            sort_order = np.argsort(-top_k_vals)
            top_k_idx = top_k_idx[sort_order]
            top_k_vals = top_k_vals[sort_order]

            row = wi * args.block_size + ti
            top_indices[row] = top_k_idx.astype(np.int32)
            top_values[row] = top_k_vals.astype(np.float32)

        window_tokens[wi] = np.array(window_ids, dtype=np.int32)

        del scores, top_k_idx, top_k_vals
        import gc; gc.collect()

        if (wi + 1) % 100 == 0 or wi == 0:
            print(f"  Window {wi+1}/{n_take}")

    # Step 7: Flush and save metadata
    top_indices.flush()
    top_values.flush()
    window_tokens.flush()
    del top_indices, top_values, window_tokens

    meta = {
        "model": args.model_name,
        "gguf": args.model,
        "n_windows": n_take,
        "block_size": args.block_size,
        "top_k": args.top_k,
        "vocab_size": vocab_size,
        "n_tokens_cached": n_tokens,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to {meta_path}")
    print(json.dumps(meta, indent=2))

    # Step 8: Cleanup
    del llm
    gc.collect()
    print("Done. Model unloaded.")


if __name__ == "__main__":
    main()

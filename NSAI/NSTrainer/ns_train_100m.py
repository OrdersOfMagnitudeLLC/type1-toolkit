#!/usr/bin/env python3
"""100M-param transformer with NS bigram filtering, bf16 autocast, gradient checkpointing."""

import os
import time
import resource
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import ns_train

# --- hyperparameters -------------------------------------------------

D_MODEL = 512
N_HEADS = 8
N_LAYERS = 8
D_FF = 2048
BLOCK_SIZE = 256
BATCH_SIZE = 4
LR = 1e-3
N_STEPS = 1000
SEED = 1337

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- model -----------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.n_heads = n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_model, bias=True)
        self.k = nn.Linear(d_model, d_model, bias=True)
        self.v = nn.Linear(d_model, d_model, bias=True)
        self.c = nn.Linear(d_model, d_model, bias=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.f1 = nn.Linear(d_model, d_ff, bias=True)
        self.f2 = nn.Linear(d_ff, d_model, bias=True)

    def attn(self, x):
        B, T, C = x.shape
        head_dim = C // self.n_heads
        Q = self.q(x).view(B, T, self.n_heads, head_dim).transpose(1, 2)
        K = self.k(x).view(B, T, self.n_heads, head_dim).transpose(1, 2)
        V = self.v(x).view(B, T, self.n_heads, head_dim).transpose(1, 2)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(head_dim)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool)).view(1, 1, T, T)
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = attn @ V
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.c(out)

    def ffn(self, x):
        return self.f2(F.gelu(self.f1(x)))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, block_size):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=True)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

    def forward(self, idx, use_checkpoint=False):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for block in self.blocks:
            if use_checkpoint:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)


def get_peak_rss_mb():
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_maxrss / 1024.0  # KB -> MB on Linux


def make_optimizer(params):
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=LR)
        print("Optimizer: 8-bit Adam (bitsandbytes)")
    except ImportError:
        opt = torch.optim.AdamW(params, lr=LR)
        print("Optimizer: AdamW (bitsandbytes not available)")
    return opt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorer", choices=["bigram", "none"], default="bigram")
    args = parser.parse_args()

    os.makedirs(ns_train.OUT_DIR, exist_ok=True)

    text = ns_train.load_text(
        os.path.join(ns_train.OUT_DIR, ns_train.DATA_PATH), ns_train.DATA_URL)
    _, stoi, _ = ns_train.build_tokenizer(text)
    data = ns_train.tokenize(text, stoi)
    vocab_size = len(stoi)

    print(f"Vocab size: {vocab_size}")
    print(f"Data length: {len(data)} tokens")

    # Build batches
    if args.scorer == "bigram":
        print("Computing bigram info...")
        token_info = ns_train.compute_bigram_info(data, vocab_size, BLOCK_SIZE)
        threshold = torch.quantile(token_info[data], 0.5)
        print(f"Bigram threshold (median): {threshold.item():.4f}")
        batches = ns_train.make_batches_ns(
            data, BLOCK_SIZE, BATCH_SIZE, N_STEPS, token_info, threshold, SEED)
    else:
        batches = ns_train.make_batches(data, BLOCK_SIZE, BATCH_SIZE, N_STEPS, SEED)

    # Build model
    print(f"\nBuilding 100M model: d_model={D_MODEL}, n_heads={N_HEADS}, "
          f"n_layers={N_LAYERS}, d_ff={D_FF}, block_size={BLOCK_SIZE}")
    model = Transformer(vocab_size, D_MODEL, N_HEADS, N_LAYERS, D_FF, BLOCK_SIZE)
    model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_params:,}")

    optimizer = make_optimizer(list(model.parameters()))

    # Training
    print(f"\nTraining {N_STEPS} steps (scorer={args.scorer})...")
    print(f"Initial RSS: {get_peak_rss_mb():.0f} MB")

    losses = {}
    t0 = time.time()
    t_block = t0

    autocast_dtype = torch.bfloat16
    print(f"Autocast: {autocast_dtype} on {DEVICE.type}")

    for step, (x, y) in enumerate(batches, 1):
        x, y = x.to(DEVICE), y.to(DEVICE)

        with torch.autocast(device_type=DEVICE.type, dtype=autocast_dtype):
            logits = model(x, use_checkpoint=True)

        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses[step] = loss.item()

        if step % 100 == 0:
            now = time.time()
            block_time = now - t_block
            rss = get_peak_rss_mb()
            print(f"  step {step}: loss {loss.item():.4f}  "
                  f"time/100: {block_time:.1f}s  RSS: {rss:.0f} MB")
            t_block = now

    total_time = time.time() - t0

    # Results
    lines = []
    lines.append("=== 100M NS-Train Results ===\n")
    lines.append(f"Architecture: d_model={D_MODEL}, n_heads={N_HEADS}, "
                 f"n_layers={N_LAYERS}, d_ff={D_FF}")
    lines.append(f"Block size: {BLOCK_SIZE}, Batch size: {BATCH_SIZE}")
    lines.append(f"Total params: {n_params:,}")
    lines.append(f"Scorer: {args.scorer}")
    lines.append(f"Steps: {N_STEPS}")
    lines.append(f"Total time: {total_time:.1f}s")
    lines.append(f"Peak RSS: {get_peak_rss_mb():.0f} MB")
    lines.append(f"Final loss: {losses[N_STEPS]:.4f}")
    lines.append("")
    lines.append("Step | Loss | Time/100 (s) | RSS (MB)")
    for step in range(100, N_STEPS + 1, 100):
        lines.append(f"{step:4d} | {losses[step]:.4f} | — | —")

    output = "\n".join(lines)
    print("\n" + output)

    out_path = os.path.join(ns_train.OUT_DIR, "ns_train_100m_results.txt")
    with open(out_path, "w") as f:
        f.write(output + "\n")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

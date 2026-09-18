#!/usr/bin/env python3
"""Train a student model against cached teacher logits (knowledge distillation).

Loads top-k logits cached by cache_qwen_logits.py and trains a GPT-style
student decoder. KD loss is KL(teacher || student) computed only over the
cached top-k vocab positions: never materializes the full vocab x seq tensor.
"""

import os
import gc
import json
import math
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GPTBlock(nn.Module):
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

    def forward(self, x):
        B, T, C = x.shape
        head_dim = C // self.n_heads
        h = self.ln1(x)
        Q = self.q(h).view(B, T, self.n_heads, head_dim).transpose(1, 2)
        K = self.k(h).view(B, T, self.n_heads, head_dim).transpose(1, 2)
        V = self.v(h).view(B, T, self.n_heads, head_dim).transpose(1, 2)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(head_dim)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool)).view(1, 1, T, T)
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = attn @ V
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c(out)
        x = x + out
        x = x + self.f2(F.gelu(self.f1(self.ln2(x))))
        return x


class StudentGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, block_size, tied=False):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.layers = nn.ModuleList([GPTBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        if tied:
            self.lm_head.weight = self.tok_emb.weight

    def forward(self, x, use_checkpoint=False):
        B, T = x.shape
        h = self.tok_emb(x) + self.pos_emb(torch.arange(T, device=x.device))
        for layer in self.layers:
            if use_checkpoint:
                h = checkpoint(layer, h, use_reentrant=False)
            else:
                h = layer(h)
        h = self.ln_f(h)
        return h  # hidden states, not logits


# ---------------------------------------------------------------------------
# KD loss: chunked, only over top-k positions
# ---------------------------------------------------------------------------

def kd_loss(hidden, lm_head_weight, teacher_indices, teacher_values, chunk_size=32):
    """KL(teacher || student) over top-k vocab positions only.

    hidden:          (B, T, D)   student hidden states
    lm_head_weight:  (V, D)      lm_head weight matrix
    teacher_indices: (B, T, K)   top-k vocab indices from teacher
    teacher_values:  (B, T, K)   teacher logit values at those indices
    """
    B, T, D = hidden.shape
    K = teacher_indices.size(-1)
    n = B * T

    h_flat = hidden.reshape(n, D)
    idx_flat = teacher_indices.reshape(n, K)
    val_flat = teacher_values.reshape(n, K)

    total = h_flat.new_tensor(0.0)
    for s in range(0, n, chunk_size):
        e = min(s + chunk_size, n)
        h_c = h_flat[s:e]                          # (chunk, D)
        idx_c = idx_flat[s:e]                      # (chunk, K)
        val_c = val_flat[s:e]                      # (chunk, K)

        w_c = lm_head_weight[idx_c]                # (chunk, K, D)
        logits_c = torch.einsum("cd,ckd->ck", h_c, w_c)  # (chunk, K)

        t_probs = F.softmax(val_c, dim=-1)         # (chunk, K)
        s_logp = F.log_softmax(logits_c, dim=-1)   # (chunk, K)

        total = total + F.kl_div(s_logp, t_probs, reduction="none").sum()

    return total / n


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def cosine_lr(step, total_steps, lr_max, lr_min=0.0):
    if step >= total_steps:
        return lr_min
    progress = step / total_steps
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train student against cached teacher logits")
    parser.add_argument("--cache-dir", required=True, help="Path to teacher_logits_cache_*/")
    parser.add_argument("--output", required=True, help="Path to save student checkpoint")
    parser.add_argument("--d-model", type=int, default=1024, help="Student hidden dim")
    parser.add_argument("--n-layers", type=int, default=20, help="Student transformer layers")
    parser.add_argument("--n-heads", type=int, default=16, help="Attention heads")
    parser.add_argument("--d-ff", type=int, default=4096, help="FFN inner dim")
    parser.add_argument("--steps", type=int, default=1000, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=8, help="Windows per step")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--block-size", type=int, default=128, help="Tokens per window (must match cache)")
    parser.add_argument("--tied", action="store_true", default=False, help="Tie lm_head to tok_emb")
    parser.add_argument("--grad-ckpt", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable/disable gradient checkpointing (default: enabled)")
    args = parser.parse_args()

    # --- Load cache metadata ---
    meta_path = os.path.join(args.cache_dir, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    vocab_size = meta["vocab_size"]
    top_k = meta["top_k"]
    n_windows = meta["n_windows"]
    block_size = meta["block_size"]
    assert block_size == args.block_size, f"block_size mismatch: cache={block_size}, args={args.block_size}"
    n_tokens = n_windows * block_size
    print(f"Cache: {n_windows} windows, {n_tokens} tokens, vocab={vocab_size}, top_k={top_k}")

    # --- Check for window_tokens.bin ---
    tokens_path = os.path.join(args.cache_dir, "window_tokens.bin")
    if not os.path.exists(tokens_path):
        print(f"ERROR: {tokens_path} not found.")
        print("       Update cache_qwen_logits.py to save window_tokens.bin, then re-run caching.")
        return

    # --- Load memmaps (read-only) ---
    top_indices = np.memmap(
        os.path.join(args.cache_dir, "top_indices.bin"),
        dtype=np.int32, mode="r", shape=(n_tokens, top_k))
    top_values = np.memmap(
        os.path.join(args.cache_dir, "top_values.bin"),
        dtype=np.float32, mode="r", shape=(n_tokens, top_k))
    window_tokens = np.memmap(
        tokens_path, dtype=np.int32, mode="r",
        shape=(n_windows, block_size))

    # --- Build model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StudentGPT(vocab_size, args.d_model, args.n_layers, args.n_heads,
                       args.d_ff, block_size, tied=args.tied).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Student: {n_params:,} params ({n_params/1e6:.1f}M), tied={args.tied}")
    model.train()

    # --- Optimizer: 8-bit AdamW if available, else standard ---
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr)
        print("Optimizer: 8-bit AdamW (bitsandbytes)")
    except ImportError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        print("Optimizer: standard AdamW (bitsandbytes not available)")

    # --- Training loop ---
    print(f"Training {args.steps} steps, batch_size={args.batch_size}, lr={args.lr}")
    for step in range(1, args.steps + 1):
        cur_lr = cosine_lr(step - 1, args.steps, args.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        # Sample random windows
        wi = np.random.randint(0, n_windows, size=args.batch_size)

        # Load input tokens
        x = torch.from_numpy(window_tokens[wi].copy()).long().to(device)  # (B, T)

        # Load teacher logits targets for those windows
        t_idx_chunks = []
        t_val_chunks = []
        for w in wi:
            s = int(w) * block_size
            t_idx_chunks.append(top_indices[s:s + block_size])
            t_val_chunks.append(top_values[s:s + block_size])
        t_indices = torch.from_numpy(np.stack(t_idx_chunks)).long().to(device)   # (B, T, K)
        t_values = torch.from_numpy(np.stack(t_val_chunks).astype(np.float32)).to(device)  # (B, T, K)

        # Forward → hidden states
        hidden = model(x, use_checkpoint=args.grad_ckpt)  # (B, T, D)

        # KD loss (chunked over tokens, only top-k positions)
        loss = kd_loss(hidden, model.lm_head.weight, t_indices, t_values)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 100 == 0 or step == 1:
            print(f"  Step {step}/{args.steps}  loss={loss.item():.4f}  lr={cur_lr:.2e}")

    # --- Save checkpoint ---
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "vocab_size": vocab_size,
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "d_ff": args.d_ff,
            "block_size": block_size,
            "tied": args.tied,
        },
        "meta": meta,
    }, args.output)
    print(f"Saved checkpoint to {args.output}")

    del model, optimizer
    gc.collect()
    print("Done.")


if __name__ == "__main__":
    main()

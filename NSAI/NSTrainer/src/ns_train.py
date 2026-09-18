#!/usr/bin/env python3
import os
import math
import requests
import argparse
import tiktoken
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- hyperparameters -------------------------------------------------

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "data/tinyshakespeare.txt"
C4_PATH = "data/c4_sample.txt"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

BLOCK_SIZE = 128
BATCH_SIZE = 4
D_MODEL = 256
N_HEADS = 4
N_LAYERS = 4
D_FF = 1024          # 4:1 FFN ratio, honest ~5M param transformer
LR = 1e-3
N_STEPS = 1000
SEED = 1337
P_SPARSE = 0.38

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- data ------------------------------------------------------------

def download_text(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(path, "w", encoding="utf-8") as f:
        f.write(r.text)

def load_text(path, url):
    if not os.path.exists(path):
        download_text(url, path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def build_tokenizer(text):
    chars = sorted(list(set(text)))
    stoi = {c: i for i, c in enumerate(chars)}
    return chars, stoi, {i: c for i, c in enumerate(chars)}

def tokenize(text, stoi):
    return torch.tensor([stoi[c] for c in text], dtype=torch.long)

def get_batch(data, block_size, batch_size):
    max_i = len(data) - block_size
    ix = torch.randint(max_i, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix.tolist()])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix.tolist()])
    return x, y

def make_batches(data, block_size, batch_size, n_steps, seed):
    torch.manual_seed(seed)
    return [get_batch(data, block_size, batch_size) for _ in range(n_steps)]


def compute_token_info(data, vocab_size):
    freq = torch.bincount(data, minlength=vocab_size).float()
    p = freq / freq.sum()
    return -torch.log(p.clamp(min=1e-9))


def compute_bigram_info(data, vocab_size, block_size):
    V = vocab_size
    bigrams = data[:-1].long() * V + data[1:].long()
    unique, counts = torch.unique(bigrams, sorted=True, return_counts=True)
    idx = torch.searchsorted(unique, bigrams)
    bigram_counts = counts[idx]
    total = bigram_counts.sum()
    p = bigram_counts / total
    scale = block_size / (block_size - 1)
    info = torch.zeros(len(data), device=data.device, dtype=torch.float32)
    info[:-1] = -torch.log(p.clamp(min=1e-9)) * scale
    return info


def compute_trigram_info(data, vocab_size, block_size):
    V = vocab_size
    trigrams = data[:-2].long() * V * V + data[1:-1].long() * V + data[2:].long()
    unique, counts = torch.unique(trigrams, sorted=True, return_counts=True)
    idx = torch.searchsorted(unique, trigrams)
    trigram_counts = counts[idx]
    total = trigram_counts.sum()
    p = trigram_counts / total
    scale = block_size / (block_size - 2)
    info = torch.zeros(len(data), device=data.device, dtype=torch.float32)
    info[:-2] = -torch.log(p.clamp(min=1e-9)) * scale
    return info


def get_batch_ns(data, block_size, batch_size, token_info, threshold):
    candidates = []
    while len(candidates) < batch_size:
        i = torch.randint(len(data) - block_size, (1,)).item()
        seq = data[i:i+block_size]
        if token_info[seq].mean() > threshold:
            candidates.append(i)
    x = torch.stack([data[i:i+block_size] for i in candidates])
    y = torch.stack([data[i+1:i+block_size+1] for i in candidates])
    return x, y


def make_batches_ns(data, block_size, batch_size, n_steps, token_info, threshold, seed):
    torch.manual_seed(seed)
    info_per_token = token_info[data]
    cs = torch.cat([torch.zeros(1, device=info_per_token.device), info_per_token.cumsum(dim=0)])
    window_sum = cs[block_size:] - cs[:-block_size]
    window_mean = window_sum / block_size
    valid = window_mean > threshold
    valid_idx = torch.nonzero(valid, as_tuple=True)[0]
    n_valid = len(valid_idx)
    print(f"NS-data: {n_valid} / {len(window_mean)} windows above threshold ({n_valid / len(window_mean) * 100:.1f}% retention)")
    ix = valid_idx[torch.randint(n_valid, (n_steps * batch_size,))]
    x = torch.stack([data[i:i+block_size] for i in ix.tolist()])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix.tolist()])
    return [(x[j:j+batch_size], y[j:j+batch_size]) for j in range(0, n_steps * batch_size, batch_size)]

# --- model -----------------------------------------------------------

def make_model(vocab_size, d_model, n_heads, n_layers, d_ff, block_size, seed):
    torch.manual_seed(seed)

    tok_emb = nn.Parameter(torch.empty(vocab_size, d_model))
    pos_emb = nn.Parameter(torch.empty(block_size, d_model))
    nn.init.normal_(tok_emb, mean=0.0, std=0.02)
    nn.init.normal_(pos_emb, mean=0.0, std=0.02)

    layers = []
    for _ in range(n_layers):
        ln1 = nn.LayerNorm(d_model)
        q = nn.Linear(d_model, d_model, bias=True)
        k = nn.Linear(d_model, d_model, bias=True)
        v = nn.Linear(d_model, d_model, bias=True)
        c = nn.Linear(d_model, d_model, bias=True)
        ln2 = nn.LayerNorm(d_model)
        f1 = nn.Linear(d_model, d_ff, bias=True)
        f2 = nn.Linear(d_ff, d_model, bias=True)
        layers.append((ln1, q, k, v, c, ln2, f1, f2))

    lm_head = nn.Linear(d_model, vocab_size, bias=True)

    params = [tok_emb, pos_emb]
    for l in layers:
        for m in l:
            params.extend(m.parameters())
    params.extend(lm_head.parameters())

    return (tok_emb, pos_emb, layers, lm_head), params

def move_model_to_device(model, device):
    tok_emb, pos_emb, layers, lm_head = model
    tok_emb.data = tok_emb.data.to(device)
    pos_emb.data = pos_emb.data.to(device)
    for l in layers:
        for m in l:
            m.to(device)
    lm_head.to(device)

def capture(m, x, captured):
    out = m(x)
    if captured is not None:
        captured[id(m)] = out.detach()
    return out

def multihead_attention(x, q, k, v, c, n_heads, captured):
    B, T, C = x.shape
    head_dim = C // n_heads

    Q = capture(q, x, captured).view(B, T, n_heads, head_dim).transpose(1, 2)
    K = capture(k, x, captured).view(B, T, n_heads, head_dim).transpose(1, 2)
    V = capture(v, x, captured).view(B, T, n_heads, head_dim).transpose(1, 2)

    scores = Q @ K.transpose(-2, -1) / math.sqrt(head_dim)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool)).view(1, 1, T, T)
    scores = scores.masked_fill(~mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)

    out = attn @ V
    out = out.transpose(1, 2).contiguous().view(B, T, C)
    return capture(c, out, captured)

def ffn(x, f1, f2, captured):
    pre = f1(x)
    h = F.gelu(pre)
    if captured is not None:
        captured[id(f1)] = h.detach()
    return capture(f2, h, captured)

def forward(x, tok_emb, pos_emb, layers, lm_head, n_heads, captured=None):
    B, T = x.shape
    x = tok_emb[x] + pos_emb[torch.arange(T, device=x.device)]

    for ln1, q, k, v, c, ln2, f1, f2 in layers:
        x = x + multihead_attention(ln1(x), q, k, v, c, n_heads, captured)
        x = x + ffn(ln2(x), f1, f2, captured)

    return capture(lm_head, x, captured)

# --- sparse masking --------------------------------------------------

def mask_gradients(model, captured, percentile):
    tok_emb, pos_emb, layers, lm_head = model
    with torch.no_grad():
        # embeddings: energy = L2 norm of each token / position vector
        for emb in (tok_emb, pos_emb):
            energy = torch.linalg.vector_norm(emb, dim=1)
            thr = torch.quantile(energy, percentile)
            keep = energy > thr
            emb.grad[~keep] = 0

        # linear layers: energy = L2 norm of each output neuron over batch + sequence
        for _, q, k, v, c, _, f1, f2 in layers:
            for m in (q, k, v, c, f1, f2):
                a = captured[id(m)]
                energy = torch.linalg.vector_norm(a, dim=tuple(range(a.ndim - 1)))
                thr = torch.quantile(energy, percentile)
                keep = energy > thr
                m.weight.grad[~keep] = 0
                if m.bias is not None:
                    m.bias.grad[~keep] = 0

        a = captured[id(lm_head)]
        energy = torch.linalg.vector_norm(a, dim=tuple(range(a.ndim - 1)))
        thr = torch.quantile(energy, percentile)
        keep = energy > thr
        lm_head.weight.grad[~keep] = 0
        if lm_head.bias is not None:
            lm_head.bias.grad[~keep] = 0

# --- training --------------------------------------------------------

def count_nonzero_grads(params):
    return sum(int(torch.count_nonzero(p.grad).item()) for p in params if p.grad is not None)

def train_run(model, params, batches, sparse=False):
    optimizer = torch.optim.AdamW(params, lr=LR)
    losses = {}
    total_updates = 0

    tok_emb, pos_emb, layers, lm_head = model
    for step, (x, y) in enumerate(batches, start=1):
        x, y = x.to(DEVICE), y.to(DEVICE)

        captured = {} if sparse else None
        logits = forward(x, tok_emb, pos_emb, layers, lm_head, N_HEADS, captured)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()

        if sparse:
            mask_gradients(model, captured, P_SPARSE)

        total_updates += count_nonzero_grads(params)
        optimizer.step()

        losses[step] = loss.item()

    return losses, total_updates


def train_run_curriculum(model, params, warmup_batches, data, n_warmup, n_total, seed):
    optimizer = torch.optim.AdamW(params, lr=LR)
    losses = {}
    total_updates = 0

    tok_emb, pos_emb, layers, lm_head = model

    # Phase 1: standard warmup on the provided warmup batches
    for step, (x, y) in enumerate(warmup_batches, start=1):
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = forward(x, tok_emb, pos_emb, layers, lm_head, N_HEADS, None)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        total_updates += count_nonzero_grads(params)
        optimizer.step()
        losses[step] = loss.item()

    # Phase 2: loss-based curriculum - sample 8, pick hardest, train
    for step in range(n_warmup + 1, n_total + 1):
        x_cands = []
        y_cands = []
        for _ in range(8):
            x, y = get_batch(data, BLOCK_SIZE, BATCH_SIZE)
            x_cands.append(x)
            y_cands.append(y)

        x_cands = torch.stack(x_cands).to(DEVICE)
        y_cands = torch.stack(y_cands).to(DEVICE)
        x_cat = x_cands.view(-1, x_cands.size(-1))

        with torch.no_grad():
            logits = forward(x_cat, tok_emb, pos_emb, layers, lm_head, N_HEADS, None)
            B_total, T, V = logits.shape
            B_per = B_total // 8
            logits_8 = logits.view(8, B_per, T, V)
            per_loss = F.cross_entropy(
                logits_8.view(8, -1, V).permute(0, 2, 1),
                y_cands.view(8, -1),
                reduction='none'
            ).mean(dim=1)
            best = int(per_loss.argmax())

        x, y = x_cands[best], y_cands[best]
        logits = forward(x, tok_emb, pos_emb, layers, lm_head, N_HEADS, None)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        total_updates += count_nonzero_grads(params)
        optimizer.step()
        losses[step] = loss.item()

    return losses, total_updates

# --- main ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["shakespeare", "c4"], default="shakespeare")
    parser.add_argument("--scorer", choices=["unigram", "bigram", "trigram"], default="unigram")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.dataset == "shakespeare":
        text = load_text(os.path.join(OUT_DIR, DATA_PATH), DATA_URL)
        _, stoi, _ = build_tokenizer(text)
        data = tokenize(text, stoi)
        vocab_size = len(stoi)
        block_size = BLOCK_SIZE
    else:
        with open(os.path.join(OUT_DIR, C4_PATH), "r", encoding="utf-8") as f:
            text = f.read()
        enc = tiktoken.get_encoding("p50k_base")
        tokens = enc.encode(text)
        data = torch.tensor(tokens, dtype=torch.long)
        vocab_size = enc.n_vocab
        block_size = 16

    if args.scorer == "unigram":
        token_info = compute_token_info(data, vocab_size)
    elif args.scorer == "bigram":
        token_info = compute_bigram_info(data, vocab_size, block_size)
    else:
        token_info = compute_trigram_info(data, vocab_size, block_size)
    threshold = torch.quantile(token_info[data], 0.5)

    batches = make_batches(data, block_size, BATCH_SIZE, N_STEPS, SEED)
    batches_c = make_batches_ns(data, block_size, BATCH_SIZE, N_STEPS, token_info, threshold, SEED)

    model_a, params_a = make_model(vocab_size, D_MODEL, N_HEADS, N_LAYERS, D_FF, block_size, SEED)
    model_c, params_c = make_model(vocab_size, D_MODEL, N_HEADS, N_LAYERS, D_FF, block_size, SEED)

    move_model_to_device(model_a, DEVICE)
    move_model_to_device(model_c, DEVICE)

    print(f"Run A: standard ({args.dataset}, full data)")
    losses_a, updates_a = train_run(model_a, params_a, batches, sparse=False)

    print(f"Run C: {args.scorer} NS-data ({args.dataset})")
    losses_c, updates_c = train_run(model_c, params_c, batches_c, sparse=False)

    print("\nResults:")
    print("Step | Standard (full) | NS-data (filtered)")
    for step in (100, 200, 500, 1000):
        print(f"{step:4d} | {losses_a[step]:.4f} | {losses_c[step]:.4f}")

 print(f"\nTotal gradient updates: Standard: {updates_a}, NS-data: {updates_c}")

    final_a = losses_a[1000]
    crossover = None
    for step in sorted(losses_c):
        if losses_c[step] < final_a:
            crossover = step
            break

    if crossover is not None:
        reduction = (1000 - crossover) / 1000 * 100
        print(f"NS-data matches standard quality at step {crossover} out of 1000 ({reduction:.1f}% compute reduction)")
    else:
        print("NS-data did not match standard final quality within 1000 steps")

    steps = sorted(set(losses_a) | set(losses_c))
    a_vals = [losses_a[s] for s in steps]
    c_vals = [losses_c[s] for s in steps]

    plt.figure(figsize=(8, 5))
    plt.plot(steps, a_vals, label="Standard (full data)", marker="o")
    plt.plot(steps, c_vals, label="NS-data (filtered)", marker="o")
    plt.axhline(y=final_a, color='gray', linestyle='--', label='Standard final loss')
    if crossover is not None:
        plt.scatter([crossover], [losses_c[crossover]], color='red', s=100, zorder=5, label=f'Crossover (step {crossover})')
    plt.xscale("log")
    plt.xlabel("Step")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"Standard vs NS-Filtered ({args.dataset})")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()

    out_plot = os.path.join(OUT_DIR, "loss_comparison.png")
    plt.savefig(out_plot, dpi=150)
    print(f"\nPlot saved: {out_plot}")


if __name__ == "__main__":
    main()

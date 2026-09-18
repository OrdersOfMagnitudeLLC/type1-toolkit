#!/usr/bin/env python3
"""Dynamic FFN experiment: replace layer-0 FFN with outer-product node library."""

import os
import argparse
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F

import ns_train


class DynamicFFN(nn.Module):
    """Rank-16 dynamic weight matrix from 512 outer-product nodes."""

    def __init__(self, d_model, n_nodes=512, topk=16, gelu_mode=None):
        super().__init__()
        self.d_model = d_model
        self.n_nodes = n_nodes
        self.topk = topk
        self.gelu_mode = gelu_mode  # None, "output", or "dot"
        self.node_a = nn.Embedding(n_nodes, d_model)
        self.node_b = nn.Embedding(n_nodes, d_model)
        self.router = nn.Linear(d_model, n_nodes)
        nn.init.normal_(self.node_a.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.node_b.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.router.bias)

    def forward(self, x):
        # x: (B, T, d_model)
        logits = self.router(x)                                    # (B, T, n_nodes)
        alphas = F.softmax(logits, dim=-1)                         # (B, T, n_nodes)
        topk_vals, topk_idx = alphas.topk(self.topk, dim=-1)       # (B, T, topk)
        topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)  # renormalize
        a = self.node_a(topk_idx)                                  # (B, T, topk, d_model)
        b = self.node_b(topk_idx)                                  # (B, T, topk, d_model)
        # decomposed: x @ outer(a_i, b_i) = (x . a_i) * b_i
        dot = (x.unsqueeze(-2) * a).sum(dim=-1)                    # (B, T, topk)
        if self.gelu_mode == "dot":
            dot = F.gelu(dot)
        out = (topk_vals * dot).unsqueeze(-1) * b                  # (B, T, topk, d_model)
        out = out.sum(dim=2)                                       # (B, T, d_model)
        if self.gelu_mode == "output":
            out = F.gelu(out)
        return out


def make_dynamic_model(vocab_size, d_model, n_heads, n_layers, d_ff, block_size, seed,
                       gelu_mode="dot", n_nodes=512, topk=32, dyn_layers=None):
    torch.manual_seed(seed)
    tok_emb = nn.Parameter(torch.empty(vocab_size, d_model))
    pos_emb = nn.Parameter(torch.empty(block_size, d_model))
    nn.init.normal_(tok_emb, mean=0.0, std=0.02)
    nn.init.normal_(pos_emb, mean=0.0, std=0.02)

    layers = []
    for i in range(n_layers):
        ln1 = nn.LayerNorm(d_model)
        q = nn.Linear(d_model, d_model, bias=True)
        k = nn.Linear(d_model, d_model, bias=True)
        v = nn.Linear(d_model, d_model, bias=True)
        c = nn.Linear(d_model, d_model, bias=True)
        ln2 = nn.LayerNorm(d_model)
        if dyn_layers is not None and i in dyn_layers:
            f1 = DynamicFFN(d_model, n_nodes=n_nodes, topk=topk, gelu_mode=gelu_mode)
            f2 = None
        else:
            f1 = nn.Linear(d_model, d_ff, bias=True)
            f2 = nn.Linear(d_ff, d_model, bias=True)
        layers.append((ln1, q, k, v, c, ln2, f1, f2))

    lm_head = nn.Linear(d_model, vocab_size, bias=True)

    params = [tok_emb, pos_emb]
    for l in layers:
        for m in l:
            if m is not None:
                params.extend(m.parameters())
    params.extend(lm_head.parameters())

    return (tok_emb, pos_emb, layers, lm_head), params


def move_dynamic_model_to_device(model, device):
    tok_emb, pos_emb, layers, lm_head = model
    tok_emb.data = tok_emb.data.to(device)
    pos_emb.data = pos_emb.data.to(device)
    for l in layers:
        for m in l:
            if m is not None:
                m.to(device)
    lm_head.to(device)


def forward_dynamic(x, tok_emb, pos_emb, layers, lm_head, n_heads):
    B, T = x.shape
    x = tok_emb[x] + pos_emb[torch.arange(T, device=x.device)]
    for i, (ln1, q, k, v, c, ln2, f1, f2) in enumerate(layers):
        x = x + ns_train.multihead_attention(ln1(x), q, k, v, c, n_heads, None)
        if f2 is None:
            x = x + f1(ln2(x))
        else:
            x = x + ns_train.ffn(ln2(x), f1, f2, None)
    return lm_head(x)


def count_params(params):
    return sum(p.numel() for p in params)


def train_dynamic(model, params, batches):
    optimizer = torch.optim.AdamW(params, lr=ns_train.LR)
    losses = {}
    tok_emb, pos_emb, layers, lm_head = model
    for step, (x, y) in enumerate(batches, 1):
        x, y = x.to(ns_train.DEVICE), y.to(ns_train.DEVICE)
        logits = forward_dynamic(x, tok_emb, pos_emb, layers, lm_head, ns_train.N_HEADS)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses[step] = loss.item()
        if step % 100 == 0:
            print(f"  step {step}: loss {loss.item():.4f}")
    return losses


def parse_layers(s):
    return set(int(x) for x in s.split(",") if x.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["shakespeare", "c4"], default="shakespeare")
    parser.add_argument("--nodes", type=int, default=512)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--layers", type=str, default="0",
                        help="Comma-separated layer indices for DynamicFFN")
    args = parser.parse_args()

    os.makedirs(ns_train.OUT_DIR, exist_ok=True)

    if args.dataset == "shakespeare":
        text = ns_train.load_text(
            os.path.join(ns_train.OUT_DIR, ns_train.DATA_PATH), ns_train.DATA_URL)
        _, stoi, _ = ns_train.build_tokenizer(text)
        data = ns_train.tokenize(text, stoi)
        vocab_size = len(stoi)
        block_size = ns_train.BLOCK_SIZE
    else:
        import tiktoken
        with open(os.path.join(ns_train.OUT_DIR, ns_train.C4_PATH), "r", encoding="utf-8") as f:
            text = f.read()
        enc = tiktoken.get_encoding("p50k_base")
        tokens = enc.encode(text)
        data = torch.tensor(tokens, dtype=torch.long)
        vocab_size = enc.n_vocab
        block_size = 16

    batches = ns_train.make_batches(
        data, block_size, ns_train.BATCH_SIZE, ns_train.N_STEPS, ns_train.SEED)

    # Standard model
    print("Building standard model...")
    model_std, params_std = ns_train.make_model(
        vocab_size, ns_train.D_MODEL, ns_train.N_HEADS, ns_train.N_LAYERS,
        ns_train.D_FF, block_size, ns_train.SEED)
    ns_train.move_model_to_device(model_std, ns_train.DEVICE)
    std_n = count_params(params_std)
    print(f"  Standard params: {std_n:,}")

    # Train standard
    print("\nTraining standard model (1000 steps)...")
    losses_std, _ = ns_train.train_run(model_std, params_std, batches, sparse=False)

    del model_std, params_std
    gc.collect()
    if ns_train.DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # Single experiment: layer 0 only, specified nodes/topk
    dyn_layers = parse_layers(args.layers)
    experiments = [
        (f"Dynamic FFN layer {args.layers}, {args.nodes} nodes, k={args.topk}",
         args.nodes, args.topk, dyn_layers),
    ]

    results = {}
    for label, n_nodes, topk, dyn_layers in experiments:
        print(f"\nBuilding dynamic FFN model ({label})...")
        model_dyn, params_dyn = make_dynamic_model(
            vocab_size, ns_train.D_MODEL, ns_train.N_HEADS, ns_train.N_LAYERS,
            ns_train.D_FF, block_size, ns_train.SEED,
            gelu_mode="dot", n_nodes=n_nodes, topk=topk, dyn_layers=dyn_layers)
        move_dynamic_model_to_device(model_dyn, ns_train.DEVICE)
        dyn_n = count_params(params_dyn)
        print(f"  Dynamic FFN params: {dyn_n:,}")

        print(f"\nTraining dynamic FFN model ({label}, 1000 steps)...")
        losses_dyn = train_dynamic(model_dyn, params_dyn, batches)
        results[label] = (dyn_n, losses_dyn)

        del model_dyn, params_dyn
        gc.collect()
        if ns_train.DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # Results
    lines = []
    lines.append(f"=== Dynamic FFN Results ({args.dataset}) ===\n")
    lines.append(f"Standard model params: {std_n:,}")
    for label, (dyn_n, _) in results.items():
        lines.append(f"Dynamic FFN params ({label}): {dyn_n:,} ({(dyn_n - std_n) / std_n * 100:.1f}%)")
    lines.append("")
    lines.append(f"Standard final loss (1000): {losses_std[1000]:.4f}")
    for label, (_, losses_dyn) in results.items():
        lines.append(f"{label} final loss (1000): {losses_dyn[1000]:.4f}  gap: {losses_dyn[1000] - losses_std[1000]:+.4f}")
    lines.append("")
    lines.append("Step | Standard | " + " | ".join(results.keys()))
    for step in (100, 200, 500, 1000):
        vals = [f"{results[k][1][step]:.4f}" for k in results]
        lines.append(f"{step:4d} | {losses_std[step]:.4f} | " + " | ".join(vals))

    output = "\n".join(lines)
    print("\n" + output)

    out_path = os.path.join(ns_train.OUT_DIR, f"dynamic_ffn_results_{args.dataset}.txt")
    with open(out_path, "w") as f:
        f.write(output + "\n")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

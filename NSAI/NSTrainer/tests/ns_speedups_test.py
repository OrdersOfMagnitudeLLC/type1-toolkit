#!/usr/bin/env python3
"""Speedup experiments: sparse gradients and layer freezing vs baseline."""

import os
import time
import torch
import torch.nn.functional as F
from collections import defaultdict

import ns_train


def make_batches(data, block_size, batch_size, n_steps, seed):
    torch.manual_seed(seed)
    ix = torch.randint(len(data) - block_size - 1, (n_steps * batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix.tolist()])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix.tolist()])
    return [(x[j:j + batch_size], y[j:j + batch_size])
            for j in range(0, n_steps * batch_size, batch_size)]


def run_baseline(model, params, batches, n_steps=500):
    tok_emb, pos_emb, layers, lm_head = model
    optimizer = torch.optim.AdamW(params, lr=ns_train.LR)

    t0 = time.time()
    losses = {}
    for step, (x, y) in enumerate(batches[:n_steps], 1):
        x, y = x.to(ns_train.DEVICE), y.to(ns_train.DEVICE)
        logits = ns_train.forward(x, tok_emb, pos_emb, layers, lm_head, ns_train.N_HEADS, None)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses[step] = loss.item()
        if step % 100 == 0:
            print(f"  step {step}: loss {loss.item():.4f}")
    elapsed = time.time() - t0
    return losses, elapsed


def run_sparse_grads(model, params, batches, n_steps=500, threshold=1e-4):
    tok_emb, pos_emb, layers, lm_head = model
    optimizer = torch.optim.AdamW(params, lr=ns_train.LR)

    t0 = time.time()
    losses = {}
    skipped_total = 0
    total_total = 0
    skipped_window = 0
    total_window = 0

    for step, (x, y) in enumerate(batches[:n_steps], 1):
        x, y = x.to(ns_train.DEVICE), y.to(ns_train.DEVICE)
        logits = ns_train.forward(x, tok_emb, pos_emb, layers, lm_head, ns_train.N_HEADS, None)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        optimizer.zero_grad()
        loss.backward()

        with torch.no_grad():
            for p in params:
                if p.grad is None:
                    continue
                n_elem = p.grad.numel()
                total_total += n_elem
                total_window += n_elem
                mean_abs = p.grad.abs().mean().item()
                if mean_abs < threshold:
                    p.grad.zero_()
                    skipped_total += n_elem
                    skipped_window += n_elem

        optimizer.step()
        losses[step] = loss.item()
        if step % 100 == 0:
            pct = skipped_window / max(total_window, 1) * 100
            print(f"  step {step}: loss {loss.item():.4f}  grads skipped: {pct:.1f}%")
            skipped_window = 0
            total_window = 0

    elapsed = time.time() - t0
    overall_pct = skipped_total / max(total_total, 1) * 100
    return losses, elapsed, overall_pct


def run_layer_freezing(model, params, batches, n_steps=500, threshold=1e-5, window=100):
    tok_emb, pos_emb, layers, lm_head = model
    optimizer = torch.optim.AdamW(params, lr=ns_train.LR)

    n_layers = len(layers)
    layer_param_ids = defaultdict(list)
    for i, l in enumerate(layers):
        for m in l:
            for p in m.parameters():
                layer_param_ids[i].append(id(p))

    all_param_ids = set()
    for p in params:
        all_param_ids.add(id(p))

    grad_magnitude_rolling = defaultdict(lambda: defaultdict(float))
    low_grad_count = defaultdict(int)
    frozen_layers = set()
    freeze_log = []

    t0 = time.time()
    losses = {}

    for step, (x, y) in enumerate(batches[:n_steps], 1):
        x, y = x.to(ns_train.DEVICE), y.to(ns_train.DEVICE)
        logits = ns_train.forward(x, tok_emb, pos_emb, layers, lm_head, ns_train.N_HEADS, None)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        optimizer.zero_grad()
        loss.backward()

        with torch.no_grad():
            for i in range(n_layers):
                if i in frozen_layers:
                    continue
                mag_sum = 0.0
                count = 0
                for m in layers[i]:
                    for p in m.parameters():
                        if p.grad is not None:
                            mag_sum += p.grad.abs().mean().item()
                            count += 1
                mean_mag = mag_sum / max(count, 1)
                grad_magnitude_rolling[i][step] = mean_mag

                recent_steps = range(max(1, step - window + 1), step + 1)
                recent_mags = [grad_magnitude_rolling[i][s] for s in recent_steps
                               if s in grad_magnitude_rolling[i]]
                if recent_mags and all(m < threshold for m in recent_mags) and len(recent_mags) >= window:
                    frozen_layers.add(i)
                    for m in layers[i]:
                        for p in m.parameters():
                            p.requires_grad = False
                    freeze_log.append((i, step))
                    print(f"  *** Layer {i} frozen at step {step} (rolling grad mag < {threshold} for {window} steps)")

        optimizer.step()
        losses[step] = loss.item()
        if step % 100 == 0:
            print(f"  step {step}: loss {loss.item():.4f}  frozen layers: {sorted(frozen_layers)}")

    elapsed = time.time() - t0
    return losses, elapsed, freeze_log


def main():
    os.makedirs(ns_train.OUT_DIR, exist_ok=True)

    text = ns_train.load_text(
        os.path.join(ns_train.OUT_DIR, ns_train.DATA_PATH), ns_train.DATA_URL)
    _, stoi, _ = ns_train.build_tokenizer(text)
    data = ns_train.tokenize(text, stoi)
    vocab_size = len(stoi)

    n_steps = 500
    batches = make_batches(data, ns_train.BLOCK_SIZE, ns_train.BATCH_SIZE,
                           n_steps, ns_train.SEED)

    lines = []
    lines.append("=== NS Speedups Test Results ===\n")

    # Exp A: Baseline
    print("Exp A: Baseline (standard training)")
    model_a, params_a = ns_train.make_model(
        vocab_size, ns_train.D_MODEL, ns_train.N_HEADS, ns_train.N_LAYERS,
        ns_train.D_FF, ns_train.BLOCK_SIZE, ns_train.SEED)
    ns_train.move_model_to_device(model_a, ns_train.DEVICE)
    losses_a, time_a = run_baseline(model_a, params_a, batches, n_steps)
    del model_a, params_a
    import gc; gc.collect()
    if ns_train.DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    print(f"  Time: {time_a:.1f}s  Final loss: {losses_a[n_steps]:.4f}\n")

    # Exp B: Sparse gradients
    print("Exp B: Sparse gradients (threshold=1e-4)")
    model_b, params_b = ns_train.make_model(
        vocab_size, ns_train.D_MODEL, ns_train.N_HEADS, ns_train.N_LAYERS,
        ns_train.D_FF, ns_train.BLOCK_SIZE, ns_train.SEED)
    ns_train.move_model_to_device(model_b, ns_train.DEVICE)
    losses_b, time_b, skip_pct = run_sparse_grads(model_b, params_b, batches, n_steps)
    del model_b, params_b
    gc.collect()
    if ns_train.DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    print(f"  Time: {time_b:.1f}s  Final loss: {losses_b[n_steps]:.4f}  Skip%: {skip_pct:.1f}\n")

    # Exp C: Layer freezing
    print("Exp C: Layer freezing (threshold=1e-5, window=100)")
    model_c, params_c = ns_train.make_model(
        vocab_size, ns_train.D_MODEL, ns_train.N_HEADS, ns_train.N_LAYERS,
        ns_train.D_FF, ns_train.BLOCK_SIZE, ns_train.SEED)
    ns_train.move_model_to_device(model_c, ns_train.DEVICE)
    losses_c, time_c, freeze_log = run_layer_freezing(model_c, params_c, batches, n_steps)
    del model_c, params_c
    gc.collect()
    if ns_train.DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    print(f"  Time: {time_c:.1f}s  Final loss: {losses_c[n_steps]:.4f}  Frozen: {freeze_log}\n")

    # Results
    lines.append(f"Steps: {n_steps}")
    lines.append("")
 lines.append("Exp A: Baseline")
    lines.append(f"  Wall-clock: {time_a:.1f}s")
    lines.append(f"  Final loss: {losses_a[n_steps]:.4f}")
    lines.append(f"  Speedup: 1.00x")
    lines.append("")
 lines.append("Exp B: Sparse gradients (threshold=1e-4)")
    lines.append(f"  Wall-clock: {time_b:.1f}s")
    lines.append(f"  Final loss: {losses_b[n_steps]:.4f}")
    lines.append(f"  Speedup: {time_a / time_b:.2f}x")
    lines.append(f"  Gradients skipped: {skip_pct:.1f}%")
    lines.append("")
 lines.append("Exp C: Layer freezing (threshold=1e-5, window=100)")
    lines.append(f"  Wall-clock: {time_c:.1f}s")
    lines.append(f"  Final loss: {losses_c[n_steps]:.4f}")
    lines.append(f"  Speedup: {time_a / time_c:.2f}x")
    if freeze_log:
        for layer, step in freeze_log:
            lines.append(f"  Layer {layer} frozen at step {step}")
    else:
        lines.append(f"  No layers frozen")
    lines.append("")
    lines.append("Step | Baseline | Sparse Grad | Layer Freeze")
    for step in (100, 200, 300, 400, 500):
        lines.append(f"{step:4d} | {losses_a[step]:.4f} | {losses_b[step]:.4f} | {losses_c[step]:.4f}")

    output = "\n".join(lines)
    print("\n" + output)

    out_path = os.path.join(ns_train.OUT_DIR, "ns_speedups_results.txt")
    with open(out_path, "w") as f:
        f.write(output + "\n")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

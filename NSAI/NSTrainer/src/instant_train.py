#!/usr/bin/env python3
"""Analytically reconstruct a trained Shakespeare transformer, layer by layer."""

import gc
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import ns_train


def build_one_hot(ids, n_classes):
    return F.one_hot(ids, num_classes=n_classes).float()


def solve_linear(A, B, has_bias=True):
    """Solve (A + reg I) W = B from Gram matrices. W has shape (in_features, out_features)."""
    reg = 1e-6
    A_reg = A + reg * torch.eye(A.size(0), device=A.device, dtype=A.dtype)
    W = torch.linalg.solve(A_reg, B)
    if has_bias:
        W_mat = W[:-1]  # (in_features, out_features)
        b = W[-1]       # (out_features,)
        return W_mat, b
    return W, None


def capture_pass(x, tok_emb, pos_emb, layers, lm_head, n_heads):
    """Forward pass that collects (X, Y) for every linear layer using teacher activations."""
    B, T = x.shape
    h = tok_emb[x] + pos_emb[torch.arange(T, device=x.device)]

    acts = {}
    acts["tok_emb"] = (build_one_hot(x, tok_emb.size(0)).view(-1, tok_emb.size(0)),
                       tok_emb[x].view(-1, tok_emb.size(1)))
    pos_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
    acts["pos_emb"] = (build_one_hot(pos_ids, pos_emb.size(0)).view(-1, pos_emb.size(0)),
                       pos_emb[pos_ids].view(-1, pos_emb.size(1)))

    for li, (ln1, q, k, v, c, ln2, f1, f2) in enumerate(layers):
        a_in = ln1(h)
        Q = q(a_in)
        K = k(a_in)
        V = v(a_in)
        acts[f"q_{li}"] = (a_in.view(-1, a_in.size(-1)), Q.view(-1, Q.size(-1)))
        acts[f"k_{li}"] = (a_in.view(-1, a_in.size(-1)), K.view(-1, K.size(-1)))
        acts[f"v_{li}"] = (a_in.view(-1, a_in.size(-1)), V.view(-1, V.size(-1)))

        C = Q.size(-1)
        head_dim = C // n_heads
        Qh = Q.view(B, T, n_heads, head_dim).transpose(1, 2)
        Kh = K.view(B, T, n_heads, head_dim).transpose(1, 2)
        Vh = V.view(B, T, n_heads, head_dim).transpose(1, 2)

        scores = Qh @ Kh.transpose(-2, -1) / math.sqrt(head_dim)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool)).view(1, 1, T, T)
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)

        out = attn @ Vh
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        c_in = out
        c_out = c(c_in)
        acts[f"c_{li}"] = (c_in.view(-1, c_in.size(-1)), c_out.view(-1, c_out.size(-1)))

        h = h + c_out

        f_in = ln2(h)
        pre = f1(f_in)
        f1_out = F.gelu(pre)
        acts[f"f1_{li}"] = (f_in.view(-1, f_in.size(-1)), pre.view(-1, pre.size(-1)))

        f2_in = f1_out
        f2_out = f2(f2_in)
        acts[f"f2_{li}"] = (f2_in.view(-1, f2_in.size(-1)), f2_out.view(-1, f2_out.size(-1)))

        h = h + f2_out

    lm_in = h
    logits = lm_head(lm_in)
    acts["lm_head"] = (lm_in.view(-1, lm_in.size(-1)), logits.view(-1, logits.size(-1)))
    return logits, acts


def save_teacher_activations(data, model, n_heads, n_seq=200, cache_dir=None):
    tok_emb, pos_emb, layers, lm_head = model
    if cache_dir is None:
        cache_dir = os.path.join(ns_train.OUT_DIR, "teacher_acts_cache")
    os.makedirs(cache_dir, exist_ok=True)

    N = n_seq * ns_train.BLOCK_SIZE
    memmaps = {}
    files = {}

    batches = ns_train.make_batches(data, ns_train.BLOCK_SIZE, 1, n_seq, ns_train.SEED + 42)
    for bi, (x, _) in enumerate(batches):
        x = x.to(ns_train.DEVICE)
        _, acts = capture_pass(x, tok_emb, pos_emb, layers, lm_head, n_heads)

        if bi == 0:
            for name, (X, Y) in acts.items():
                out_dim = Y.size(1)
                path = os.path.join(cache_dir, f"{name}.bin")
                arr = np.memmap(path, dtype=np.float32, mode="w+", shape=(N, out_dim))
                memmaps[name] = arr
                files[name] = (path, out_dim)

        for name, (X, Y) in acts.items():
            memmaps[name][bi * ns_train.BLOCK_SIZE:(bi + 1) * ns_train.BLOCK_SIZE] = Y.detach().cpu().numpy().astype(np.float32)

        del acts, x
        if ns_train.DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    for arr in memmaps.values():
        arr.flush()
    del memmaps
    return files


def save_layernorms(model):
    _, _, layers, _ = model
    saved = []
    for ln1, q, k, v, c, ln2, f1, f2 in layers:
        saved.append((ln1.weight.data.clone(), ln1.bias.data.clone(),
                      ln2.weight.data.clone(), ln2.bias.data.clone()))
    return saved


def restore_layernorms(model, saved):
    _, _, layers, _ = model
    for (ln1, _, _, _, _, ln2, _, _), (w1, b1, w2, b2) in zip(layers, saved):
        ln1.weight.data.copy_(w1)
        ln1.bias.data.copy_(b1)
        ln2.weight.data.copy_(w2)
        ln2.bias.data.copy_(b2)


def _layer_names(n_layers):
    names = ["tok_emb", "pos_emb"]
    for i in range(n_layers):
        names.extend([f"q_{i}", f"k_{i}", f"v_{i}", f"c_{i}", f"f1_{i}", f"f2_{i}"])
    names.append("lm_head")
    return names


def set_layer_by_name(model, name, W, b):
    tok_emb, pos_emb, layers, lm_head = model
    if name == "tok_emb":
        tok_emb.data = W
    elif name == "pos_emb":
        pos_emb.data = W
    elif name == "lm_head":
        lm_head.weight.data = W.T
        if b is not None:
            lm_head.bias.data = b
    else:
        kind, layer_idx = name.split("_")
        layer_idx = int(layer_idx)
        ln1, q, k, v, c, ln2, f1, f2 = layers[layer_idx]
        layer = {"q": q, "k": k, "v": v, "c": c, "f1": f1, "f2": f2}[kind]
        layer.weight.data = W.T
        if b is not None:
            layer.bias.data = b


def copy_layernorms(teacher, student):
    _, _, teacher_layers, _ = teacher
    _, _, student_layers, _ = student
    for (tl, tq, tk, tv, tc, tln2, tf1, tf2), (sl, sq, sk, sv, sc, sln2, sf1, sf2) in zip(teacher_layers, student_layers):
        sl.weight.data = tl.weight.data.clone()
        sl.bias.data = tl.bias.data.clone()
        sln2.weight.data = tln2.weight.data.clone()
        sln2.bias.data = tln2.bias.data.clone()


def evaluate(model, batches, n_heads):
    tok_emb, pos_emb, layers, lm_head = model
    total, tokens = 0.0, 0
    for x, y in batches:
        x, y = x.to(ns_train.DEVICE), y.to(ns_train.DEVICE)
        logits = ns_train.forward(x, tok_emb, pos_emb, layers, lm_head, n_heads)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum")
        total += loss.item()
        tokens += y.numel()
    return total / tokens


def main():
    n_calib = 200
    cache_dir = os.path.join(ns_train.OUT_DIR, "teacher_acts_cache")

    # ---- Step 1: Shakespeare char data ----
    text = ns_train.load_text(ns_train.OUT_DIR + "/" + ns_train.DATA_PATH, ns_train.DATA_URL)
    _, stoi, _ = ns_train.build_tokenizer(text)
    data = ns_train.tokenize(text, stoi)
    vocab_size = len(stoi)

    # ---- Step 2: teacher = standard 1000-step gradient descent ----
    teacher, teacher_params = ns_train.make_model(vocab_size, ns_train.D_MODEL,
                                                   ns_train.N_HEADS, ns_train.N_LAYERS,
                                                   ns_train.D_FF, ns_train.BLOCK_SIZE,
                                                   ns_train.SEED)
    ns_train.move_model_to_device(teacher, ns_train.DEVICE)

    teacher_batches = ns_train.make_batches(data, ns_train.BLOCK_SIZE, ns_train.BATCH_SIZE,
                                            ns_train.N_STEPS, ns_train.SEED)
    print("Training teacher (standard 1000-step run)...")
    teacher_losses, _ = ns_train.train_run(teacher, teacher_params, teacher_batches, sparse=False)
    print(f"Teacher final 1000-step loss: {teacher_losses[ns_train.N_STEPS]:.4f}")

    eval_batches = ns_train.make_batches(data, ns_train.BLOCK_SIZE, ns_train.BATCH_SIZE,
                                         100, ns_train.SEED + 123)
    standard_loss = evaluate(teacher, eval_batches, ns_train.N_HEADS)

 # ---- Step 3: teacher only: save target activations to disk ----
    print(f"Pass 1: saving teacher activations for {n_calib} sequences...")
    files = save_teacher_activations(data, teacher, ns_train.N_HEADS, n_calib, cache_dir)
    lns = save_layernorms(teacher)

    del teacher, teacher_params, teacher_batches, teacher_losses
    gc.collect()
    if ns_train.DEVICE.type == "cuda":
        torch.cuda.empty_cache()

 # ---- Step 4: student only: solve one layer at a time ----
    print("Pass 2: solving layers with student only...")
    student, _ = ns_train.make_model(vocab_size, ns_train.D_MODEL, ns_train.N_HEADS,
                                     ns_train.N_LAYERS, ns_train.D_FF,
                                     ns_train.BLOCK_SIZE, ns_train.SEED + 7)
    ns_train.move_model_to_device(student, ns_train.DEVICE)
    restore_layernorms(student, lns)

    t0 = time.time()
    tok_emb, pos_emb, layers, lm_head = student
    for name in _layer_names(ns_train.N_LAYERS):
        path, out_dim = files[name]
        Y_all = np.memmap(path, dtype=np.float32, mode="r", shape=(n_calib * ns_train.BLOCK_SIZE, out_dim))
        A, B = None, None
        batches = ns_train.make_batches(data, ns_train.BLOCK_SIZE, 1, n_calib, ns_train.SEED + 42)
        for bi, (x, _) in enumerate(batches):
            x = x.to(ns_train.DEVICE)
            _, acts = capture_pass(x, tok_emb, pos_emb, layers, lm_head, ns_train.N_HEADS)
            X, _ = acts[name]
            if name not in ("tok_emb", "pos_emb"):
                ones = torch.ones(X.size(0), 1, device=X.device, dtype=X.dtype)
                X = torch.cat([X, ones], dim=1)
            if A is None:
                A = torch.zeros(X.size(1), X.size(1), device=ns_train.DEVICE, dtype=X.dtype)
                B = torch.zeros(X.size(1), out_dim, device=ns_train.DEVICE, dtype=X.dtype)
            Y = torch.from_numpy(Y_all[bi * ns_train.BLOCK_SIZE:(bi + 1) * ns_train.BLOCK_SIZE].copy()).to(ns_train.DEVICE)
            A.addmm_(X.T, X, alpha=1.0, beta=1.0)
            B.addmm_(X.T, Y, alpha=1.0, beta=1.0)
            del X, Y, acts, x
        W, b = solve_linear(A, B, has_bias=(name not in ("tok_emb", "pos_emb")))
        set_layer_by_name(student, name, W, b)
        del A, B, Y_all
        if ns_train.DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    solve_ms = (time.time() - t0) * 1000

    # ---- Step 5: evaluate and compare ----
    random_model, _ = ns_train.make_model(vocab_size, ns_train.D_MODEL, ns_train.N_HEADS,
                                          ns_train.N_LAYERS, ns_train.D_FF,
                                          ns_train.BLOCK_SIZE, ns_train.SEED + 99)
    ns_train.move_model_to_device(random_model, ns_train.DEVICE)
    random_loss = evaluate(random_model, eval_batches, ns_train.N_HEADS)
    analytical_loss = evaluate(student, eval_batches, ns_train.N_HEADS)

    print("\n--- Results ---")
    print(f"Analytical solve time: {solve_ms:.0f} ms")
    print(f"Analytical loss: {analytical_loss:.4f}")
    print(f"Standard 1000-step loss: {standard_loss:.4f}")
    print(f"Random init loss: {random_loss:.4f} (log({vocab_size}) = {math.log(vocab_size):.4f})")
    print(f"Gap to standard: {analytical_loss - standard_loss:.4f}")


if __name__ == "__main__":
    main()

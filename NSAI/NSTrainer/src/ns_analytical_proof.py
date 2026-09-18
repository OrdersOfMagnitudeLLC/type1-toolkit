#!/usr/bin/env python3
"""Proof-of-concept: analytical init of 1M student from 500K teacher."""

import gc, math, os, time, resource, ctypes
import numpy as np
import torch
import torch.nn.functional as F
import ns_train

D_MODEL = 128
N_HEADS = 4
D_FF = 256
BLOCK_SIZE = 128
BATCH_SIZE = 4
LR = 1e-3
SEED = 1337
TEACHER_LAYERS = 2
STUDENT_LAYERS = 4
N_CALIB = 100
TEACHER_STEPS = 200
FT_STEPS = 50
RAM_LIMIT_MB = 1200

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def release_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def ram_guard(tag=""):
    rss = get_rss_mb()
    if rss > RAM_LIMIT_MB:
        raise RuntimeError(f"RAM LIMIT EXCEEDED at {tag}: {rss:.0f} MB > {RAM_LIMIT_MB} MB")
    return rss


def build_one_hot(ids, n):
    return F.one_hot(ids, num_classes=n).float()


def solve_linear(A, B, has_bias=True, lambda_val=1e-6):
    A_reg = A + lambda_val * torch.eye(A.size(0), device=A.device, dtype=A.dtype)
    W = torch.linalg.solve(A_reg, B)
    if has_bias:
        return W[:-1], W[-1]
    return W, None


def capture_pass(x, tok_emb, pos_emb, layers, lm_head, n_heads):
    B, T = x.shape
    h = tok_emb[x] + pos_emb[torch.arange(T, device=x.device)]
    acts = {}
    acts["tok_emb"] = (build_one_hot(x, tok_emb.size(0)).view(-1, tok_emb.size(0)), tok_emb[x].view(-1, tok_emb.size(1)))
    pos_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
    acts["pos_emb"] = (build_one_hot(pos_ids, pos_emb.size(0)).view(-1, pos_emb.size(0)), pos_emb[pos_ids].view(-1, pos_emb.size(1)))
    for li, (ln1, q, k, v, c, ln2, f1, f2) in enumerate(layers):
        a_in = ln1(h)
        Q, K, V = q(a_in), k(a_in), v(a_in)
        acts[f"q_{li}"] = (a_in.view(-1, a_in.size(-1)), Q.view(-1, Q.size(-1)))
        acts[f"k_{li}"] = (a_in.view(-1, a_in.size(-1)), K.view(-1, K.size(-1)))
        acts[f"v_{li}"] = (a_in.view(-1, a_in.size(-1)), V.view(-1, V.size(-1)))
        C = Q.size(-1); head_dim = C // n_heads
        Qh = Q.view(B, T, n_heads, head_dim).transpose(1, 2)
        Kh = K.view(B, T, n_heads, head_dim).transpose(1, 2)
        Vh = V.view(B, T, n_heads, head_dim).transpose(1, 2)
        scores = Qh @ Kh.transpose(-2, -1) / math.sqrt(head_dim)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool)).view(1, 1, T, T)
        scores = scores.masked_fill(~mask, float("-inf"))
        out = F.softmax(scores, dim=-1) @ Vh
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        c_out = c(out)
        acts[f"c_{li}"] = (out.view(-1, out.size(-1)), c_out.view(-1, c_out.size(-1)))
        h = h + c_out
        f_in = ln2(h)
        pre = f1(f_in); f1_out = F.gelu(pre)
        acts[f"f1_{li}"] = (f_in.view(-1, f_in.size(-1)), pre.view(-1, pre.size(-1)))
        f2_out = f2(f1_out)
        acts[f"f2_{li}"] = (f1_out.view(-1, f1_out.size(-1)), f2_out.view(-1, f2_out.size(-1)))
        h = h + f2_out
    acts["lm_head"] = (h.view(-1, h.size(-1)), lm_head(h).view(-1, lm_head(h).size(-1)))
    acts["final_hidden"] = (h.view(-1, h.size(-1)), h.view(-1, h.size(-1)))
    return acts


def save_teacher_acts(data, model, n_heads, n_seq, cache_dir):
    tok_emb, pos_emb, layers, lm_head = model
    os.makedirs(cache_dir, exist_ok=True)
    N = n_seq * BLOCK_SIZE
    memmaps, files = {}, {}
    batches = ns_train.make_batches(data, BLOCK_SIZE, 1, n_seq, SEED + 42)
    for bi, (x, _) in enumerate(batches):
        x = x.to(DEVICE)
        acts = capture_pass(x, tok_emb, pos_emb, layers, lm_head, n_heads)
        if bi == 0:
            for name, (X, Y) in acts.items():
                out_dim = Y.size(1)
                path = os.path.join(cache_dir, f"{name}.bin")
                memmaps[name] = np.memmap(path, dtype=np.float32, mode="w+", shape=(N, out_dim))
                files[name] = (path, out_dim)
        for name, (X, Y) in acts.items():
            memmaps[name][bi*BLOCK_SIZE:(bi+1)*BLOCK_SIZE] = Y.detach().cpu().numpy().astype(np.float32)
        del acts, x
    for arr in memmaps.values(): arr.flush()
    del memmaps
    return files


def save_lns(model):
    _, _, layers, _ = model
    return [(l[0].weight.data.clone(), l[0].bias.data.clone(),
             l[5].weight.data.clone(), l[5].bias.data.clone()) for l in layers]


def copy_lns(t_lns, student, t_layers, s_layers):
    _, _, s_l, _ = student
    for i in range(s_layers):
        src = min(i, t_layers - 1)
        s_l[i][0].weight.data.copy_(t_lns[src][0]); s_l[i][0].bias.data.copy_(t_lns[src][1])
        s_l[i][5].weight.data.copy_(t_lns[src][2]); s_l[i][5].bias.data.copy_(t_lns[src][3])


def set_layer(model, name, W, b):
    tok_emb, pos_emb, layers, lm_head = model
    if name == "tok_emb": tok_emb.data = W
    elif name == "pos_emb": pos_emb.data = W
    elif name == "lm_head":
        lm_head.weight.data = W.T
        if b is not None: lm_head.bias.data = b
    else:
        kind, idx = name.split("_"); idx = int(idx)
        layer = {"q":layers[idx][1],"k":layers[idx][2],"v":layers[idx][3],
                 "c":layers[idx][4],"f1":layers[idx][6],"f2":layers[idx][7]}[kind]
        layer.weight.data = W.T
        if b is not None: layer.bias.data = b


def evaluate(model, batches, n_heads):
    tok_emb, pos_emb, layers, lm_head = model
    total, tokens = 0.0, 0
    for x, y in batches:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = ns_train.forward(x, tok_emb, pos_emb, layers, lm_head, n_heads)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum")
        total += loss.item(); tokens += y.numel()
    return total / tokens


def train_steps(model, params, batches, n_steps, label=""):
    tok_emb, pos_emb, layers, lm_head = model
    opt = torch.optim.AdamW(params, lr=LR)
    losses = {}
    for step, (x, y) in enumerate(batches[:n_steps], 1):
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = ns_train.forward(x, tok_emb, pos_emb, layers, lm_head, N_HEADS, None)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        losses[step] = loss.item()
        if step % 50 == 0:
            print(f"  [{label}] step {step}: loss {loss.item():.4f}")
    return losses


def get_target_name(name, t_layers):
    if name in ("tok_emb", "pos_emb", "lm_head"):
        return name
    li = int(name.split("_")[1])
    if li < t_layers:
        return name
    # Extra layers: target last teacher layer's matching sub-layer
    kind = name.rsplit("_", 1)[0]
    return f"{kind}_{t_layers - 1}"


def main():
    os.makedirs(ns_train.OUT_DIR, exist_ok=True)
    cache_dir = os.path.join(ns_train.OUT_DIR, "teacher_acts_proof")

    text = ns_train.load_text(os.path.join(ns_train.OUT_DIR, ns_train.DATA_PATH), ns_train.DATA_URL)
    _, stoi, _ = ns_train.build_tokenizer(text)
    data = ns_train.tokenize(text, stoi)
    vocab_size = len(stoi)

    # NS bigram filter
    print("Computing bigram info...")
    token_info = ns_train.compute_bigram_info(data, vocab_size, BLOCK_SIZE)
    threshold = torch.quantile(token_info[data], 0.5)
    ns_batches = ns_train.make_batches_ns(data, BLOCK_SIZE, BATCH_SIZE, TEACHER_STEPS, token_info, threshold, SEED)

    # Step 1: Train 500K teacher
    print(f"\nStep 1: Training teacher ({TEACHER_LAYERS} layers, 200 steps)...")
    teacher, teacher_params = ns_train.make_model(vocab_size, D_MODEL, N_HEADS, TEACHER_LAYERS, D_FF, BLOCK_SIZE, SEED)
    ns_train.move_model_to_device(teacher, DEVICE)
    t_n = sum(p.numel() for p in teacher_params)
    print(f"  Teacher params: {t_n:,}  RSS: {get_rss_mb():.0f} MB")
    t0 = time.time()
    teacher_losses = train_steps(teacher, teacher_params, ns_batches, TEACHER_STEPS, "teacher")
    print(f"  Teacher time: {time.time()-t0:.1f}s  Final loss: {teacher_losses[TEACHER_STEPS]:.4f}")

    eval_batches = ns_train.make_batches(data, BLOCK_SIZE, BATCH_SIZE, 50, SEED + 123)
    teacher_loss = evaluate(teacher, eval_batches, N_HEADS)

    # Save teacher activations
    print(f"\nSaving teacher activations ({N_CALIB} sequences)...")
    files = save_teacher_acts(data, teacher, N_HEADS, N_CALIB, cache_dir)
    t_lns = save_lns(teacher)

    # CRITICAL: fully delete teacher before student
    del teacher, teacher_params, teacher_losses
    gc.collect()
    if DEVICE.type == "cuda": torch.cuda.empty_cache()
    print(f"  Teacher deleted. RSS: {get_rss_mb():.0f} MB")

    # Step 2: Build 1M student and solve analytically
    print(f"\nStep 2: Building student ({STUDENT_LAYERS} layers) and solving analytically...")
    student, student_params = ns_train.make_model(vocab_size, D_MODEL, N_HEADS, STUDENT_LAYERS, D_FF, BLOCK_SIZE, SEED + 7)
    ns_train.move_model_to_device(student, DEVICE)
    s_n = sum(p.numel() for p in student_params)
    print(f"  Student params: {s_n:,}  RSS: {get_rss_mb():.0f} MB")
    copy_lns(t_lns, student, TEACHER_LAYERS, STUDENT_LAYERS)

    lambda_log = []
    t0 = time.time()
    names = ["tok_emb", "pos_emb"]
    for i in range(STUDENT_LAYERS):
        names.extend([f"q_{i}", f"k_{i}", f"v_{i}", f"c_{i}", f"f1_{i}", f"f2_{i}"])
    names.append("lm_head")

    for name in names:
        target_name = get_target_name(name, TEACHER_LAYERS)
        path, out_dim = files[target_name]
        Y_all = np.memmap(path, dtype=np.float32, mode="r", shape=(N_CALIB * BLOCK_SIZE, out_dim))
        A, B = None, None
        n_samples = 0
        batches = ns_train.make_batches(data, BLOCK_SIZE, 1, N_CALIB, SEED + 42)
        for bi, (x, _) in enumerate(batches):
            x = x.to(DEVICE)
            acts = capture_pass(x, *student, N_HEADS)
            X, _ = acts[name]
            if name not in ("tok_emb", "pos_emb"):
                ones = torch.ones(X.size(0), 1, device=X.device, dtype=X.dtype)
                X = torch.cat([X, ones], dim=1)
            if A is None:
                A = torch.zeros(X.size(1), X.size(1), device=DEVICE, dtype=X.dtype)
                B = torch.zeros(X.size(1), out_dim, device=DEVICE, dtype=X.dtype)
            Y = torch.from_numpy(Y_all[bi*BLOCK_SIZE:(bi+1)*BLOCK_SIZE].copy()).to(DEVICE)
            A.addmm_(X.T, X); B.addmm_(X.T, Y)
            n_samples += X.size(0)
            del X, Y, acts, x
            release_memory()
        # Ledoit-Wolf shrinkage: lambda = trace(XtX) / (n_samples * d_model)
        d_in = A.size(0)
        lambda_val = (torch.trace(A) / (n_samples * d_in)).item()
        W, b = solve_linear(A, B, has_bias=(name not in ("tok_emb", "pos_emb")), lambda_val=lambda_val)
        lambda_log.append((name, lambda_val))
        set_layer(student, name, W, b)
        del A, B, Y_all
        release_memory()
        if DEVICE.type == "cuda": torch.cuda.empty_cache()
        rss = ram_guard(f"solve {name}")
        print(f"  Solved {name} -> {target_name}  lambda={lambda_val:.6f}  RSS: {rss:.0f} MB")
    solve_time = time.time() - t0
    print(f"  Analytical solve time: {solve_time:.1f}s")

    analytical_loss = evaluate(student, eval_batches, N_HEADS)
    print(f"  Analytical init loss: {analytical_loss:.4f}")

    # Step 3: Fine-tune 50 steps
    print(f"\nStep 3: Fine-tuning ({FT_STEPS} steps)...")
    ft_losses = train_steps(student, student_params, ns_batches, FT_STEPS, "finetune")
    ft_loss = evaluate(student, eval_batches, N_HEADS)
    print(f"  Post fine-tune loss: {ft_loss:.4f}")

    del student, student_params
    gc.collect()
    if DEVICE.type == "cuda": torch.cuda.empty_cache()

    # Step 4: Baseline - random init + 200 steps
    print(f"\nStep 4: Training random-init baseline ({TEACHER_STEPS} steps)...")
    baseline, baseline_params = ns_train.make_model(vocab_size, D_MODEL, N_HEADS, STUDENT_LAYERS, D_FF, BLOCK_SIZE, SEED + 99)
    ns_train.move_model_to_device(baseline, DEVICE)
    std_losses = train_steps(baseline, baseline_params, ns_batches, TEACHER_STEPS, "baseline")
    std_loss = evaluate(baseline, eval_batches, N_HEADS)
    print(f"  Standard {TEACHER_STEPS}-step loss: {std_loss:.4f}")

    random_model, _ = ns_train.make_model(vocab_size, D_MODEL, N_HEADS, STUDENT_LAYERS, D_FF, BLOCK_SIZE, SEED + 99)
    ns_train.move_model_to_device(random_model, DEVICE)
    random_loss = evaluate(random_model, eval_batches, N_HEADS)

    # Results
    lines = []
    lines.append("=== Analytical Proof-of-Concept Results (v2: Ledoit-Wolf) ===\n")
    lines.append(f"Teacher: {TEACHER_LAYERS} layers, {t_n:,} params, {TEACHER_STEPS} steps")
    lines.append(f"Student: {STUDENT_LAYERS} layers, {s_n:,} params")
    lines.append(f"Calibration sequences: {N_CALIB}")
    lines.append(f"Analytical solve time: {solve_time:.1f}s")
    lines.append(f"Peak RSS: {get_rss_mb():.0f} MB (limit: {RAM_LIMIT_MB} MB)")
    lines.append("")
    lines.append("Ledoit-Wolf lambda per layer:")
    for lname, lval in lambda_log:
        lines.append(f"  {lname:10s}  lambda={lval:.6f}")
    lines.append("")
    lines.append(f"Random init loss:           {random_loss:.4f}  (log({vocab_size})={math.log(vocab_size):.4f})")
    lines.append(f"Analytical init loss:       {analytical_loss:.4f}")
    lines.append(f"Analytical + {FT_STEPS} FT loss:    {ft_loss:.4f}")
    lines.append(f"Standard {TEACHER_STEPS}-step loss:  {std_loss:.4f}")
    lines.append(f"Teacher ({TEACHER_STEPS}-step) loss: {teacher_loss:.4f}")
    lines.append("")
    lines.append(f"Gap (analytical+FT vs std): {ft_loss - std_loss:+.4f}")
    lines.append(f"Speedup: {FT_STEPS} steps vs {TEACHER_STEPS} steps = {TEACHER_STEPS/FT_STEPS:.0f}x fewer gradient updates")

    output = "\n".join(lines)
    print("\n" + output)
    out_path = os.path.join(ns_train.OUT_DIR, "ns_analytical_proof_v2_results.txt")
    with open(out_path, "w") as f: f.write(output + "\n")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

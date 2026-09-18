#!/usr/bin/env python3
"""Analytical init of 100M student from 10M teacher, then 100-step fine-tune."""

import gc, math, os, time
import numpy as np
import torch
import torch.nn.functional as F
import ns_train

D_MODEL = 512
N_HEADS = 8
D_FF = 2048
BLOCK_SIZE = 128
BATCH_SIZE = 4
LR = 1e-3
SEED = 1337
TEACHER_LAYERS = 3
STUDENT_LAYERS = 8
N_CALIB = 200
FINE_STEPS = 100
STD_STEPS = 1000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_one_hot(ids, n):
    return F.one_hot(ids, num_classes=n).float()

def solve_linear(A, B, has_bias=True):
    reg = 1e-6
    A_reg = A + reg * torch.eye(A.size(0), device=A.device, dtype=A.dtype)
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
    # Also save final hidden state for extra student layers
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

def restore_lns(model, saved):
    _, _, layers, _ = model
    for l, (w1,b1,w2,b2) in zip(layers, saved):
        l[0].weight.data.copy_(w1); l[0].bias.data.copy_(b1)
        l[5].weight.data.copy_(w2); l[5].bias.data.copy_(b2)

def copy_lns_from_teacher(teacher, student, t_layers, s_layers):
    t_lns = save_lns(teacher)
    _, _, s_l, _ = student
    for i in range(min(t_layers, s_layers)):
        s_l[i][0].weight.data.copy_(t_lns[i][0]); s_l[i][0].bias.data.copy_(t_lns[i][1])
        s_l[i][5].weight.data.copy_(t_lns[i][2]); s_l[i][5].bias.data.copy_(t_lns[i][3])
    # Extra student layers: copy from last teacher layer
    for i in range(t_layers, s_layers):
        s_l[i][0].weight.data.copy_(t_lns[-1][0]); s_l[i][0].bias.data.copy_(t_lns[-1][1])
        s_l[i][5].weight.data.copy_(t_lns[-1][2]); s_l[i][5].bias.data.copy_(t_lns[-1][3])

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

def student_layer_names(s_layers, t_layers):
    names = ["tok_emb", "pos_emb"]
    for i in range(s_layers):
        if i < t_layers:
            names.extend([f"q_{i}",f"k_{i}",f"v_{i}",f"c_{i}",f"f1_{i}",f"f2_{i}"])
        else:
            # Extra layers target final_hidden (identity-like)
            names.extend([f"q_{i}",f"k_{i}",f"v_{i}",f"c_{i}",f"f1_{i}",f"f2_{i}"])
    names.append("lm_head")
    return names

def get_target_name(student_name, t_layers):
    kind, idx = student_name.split("_"), None
    if student_name in ("tok_emb", "pos_emb", "lm_head"):
        return student_name
    prefix = student_name.rsplit("_", 1)[0]
    li = int(student_name.split("_")[1])
    if li < t_layers:
        return student_name
    # Extra layers: target final_hidden as Y, but X from student's own activation
    return "final_hidden"

def evaluate(model, batches, n_heads):
    tok_emb, pos_emb, layers, lm_head = model
    total, tokens = 0.0, 0
    for x, y in batches:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = ns_train.forward(x, tok_emb, pos_emb, layers, lm_head, n_heads)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum")
        total += loss.item(); tokens += y.numel()
    return total / tokens

def train_steps(model, params, batches, n_steps):
    tok_emb, pos_emb, layers, lm_head = model
    opt = torch.optim.AdamW(params, lr=LR)
    losses = {}
    for step, (x, y) in enumerate(batches[:n_steps], 1):
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = ns_train.forward(x, tok_emb, pos_emb, layers, lm_head, ns_train.N_HEADS, None)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        losses[step] = loss.item()
        if step % 50 == 0:
            print(f"  step {step}: loss {loss.item():.4f}")
    return losses

def main():
    os.makedirs(ns_train.OUT_DIR, exist_ok=True)
    cache_dir = os.path.join(ns_train.OUT_DIR, "teacher_acts_100m")

    text = ns_train.load_text(os.path.join(ns_train.OUT_DIR, ns_train.DATA_PATH), ns_train.DATA_URL)
    _, stoi, _ = ns_train.build_tokenizer(text)
    data = ns_train.tokenize(text, stoi)
    vocab_size = len(stoi)

    # NS bigram filter
    print("Computing bigram info...")
    token_info = ns_train.compute_bigram_info(data, vocab_size, BLOCK_SIZE)
    threshold = torch.quantile(token_info[data], 0.5)
    ns_batches = ns_train.make_batches_ns(data, BLOCK_SIZE, BATCH_SIZE, STD_STEPS, token_info, threshold, SEED)

    # Step 1: Train 10M teacher
    print(f"\nStep 1: Training teacher ({TEACHER_LAYERS} layers, ~10M params, 500 steps)...")
    teacher, teacher_params = ns_train.make_model(vocab_size, D_MODEL, N_HEADS, TEACHER_LAYERS, D_FF, BLOCK_SIZE, SEED)
    ns_train.move_model_to_device(teacher, DEVICE)
    t_n = sum(p.numel() for p in teacher_params)
    print(f"  Teacher params: {t_n:,}")
    t0 = time.time()
    teacher_losses = train_steps(teacher, teacher_params, ns_batches, 500)
    print(f"  Teacher time: {time.time()-t0:.1f}s  Final loss: {teacher_losses[500]:.4f}")

    eval_batches = ns_train.make_batches(data, BLOCK_SIZE, BATCH_SIZE, 100, SEED + 123)
    teacher_loss = evaluate(teacher, eval_batches, N_HEADS)

    # Step 2: Save teacher activations
    print(f"\nStep 2: Saving teacher activations ({N_CALIB} sequences)...")
    files = save_teacher_acts(data, teacher, N_HEADS, N_CALIB, cache_dir)
    t_lns = save_lns(teacher)

    del teacher, teacher_params
    gc.collect()
    if DEVICE.type == "cuda": torch.cuda.empty_cache()

    # Step 3: Build 100M student and solve analytically
    print(f"\nStep 3: Building student ({STUDENT_LAYERS} layers) and solving analytically...")
    student, student_params = ns_train.make_model(vocab_size, D_MODEL, N_HEADS, STUDENT_LAYERS, D_FF, BLOCK_SIZE, SEED + 7)
    ns_train.move_model_to_device(student, DEVICE)
    s_n = sum(p.numel() for p in student_params)
    print(f"  Student params: {s_n:,}")

    # Copy layernorms from teacher
    _, _, t_layers_list, _ = (None, None, None, None)  # placeholder
    copy_lns_from_teacher_raw(t_lns, student, TEACHER_LAYERS, STUDENT_LAYERS)

    t0 = time.time()
    names = student_layer_names(STUDENT_LAYERS, TEACHER_LAYERS)
    for name in names:
        target_name = name if (name in ("tok_emb","pos_emb","lm_head") or int(name.split("_")[1]) < TEACHER_LAYERS) else "final_hidden"
        path, out_dim = files[target_name]
        Y_all = np.memmap(path, dtype=np.float32, mode="r", shape=(N_CALIB * BLOCK_SIZE, out_dim))
        A, B = None, None
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
            del X, Y, acts, x
        W, b = solve_linear(A, B, has_bias=(name not in ("tok_emb","pos_emb")))
        set_layer(student, name, W, b)
        del A, B, Y_all
        if DEVICE.type == "cuda": torch.cuda.empty_cache()
        print(f"  Solved {name} -> target {target_name}")
    solve_time = time.time() - t0
    print(f"  Analytical solve time: {solve_time:.1f}s")

    analytical_loss = evaluate(student, eval_batches, N_HEADS)
    print(f"  Analytical init loss: {analytical_loss:.4f}")

    # Step 4: Fine-tune 100 steps
    print(f"\nStep 4: Fine-tuning analytically initialized model ({FINE_STEPS} steps)...")
    ft_losses = train_steps(student, student_params, ns_batches, FINE_STEPS)
    ft_loss = evaluate(student, eval_batches, N_HEADS)
    print(f"  Post fine-tune loss: {ft_loss:.4f}")

    del student, student_params
    gc.collect()
    if DEVICE.type == "cuda": torch.cuda.empty_cache()

    # Step 5: Baseline - random init + 1000 steps
    print(f"\nStep 5: Training random-init baseline ({STD_STEPS} steps)...")
    baseline, baseline_params = ns_train.make_model(vocab_size, D_MODEL, N_HEADS, STUDENT_LAYERS, D_FF, BLOCK_SIZE, SEED + 99)
    ns_train.move_model_to_device(baseline, DEVICE)
    b_n = sum(p.numel() for p in baseline_params)
    std_losses = train_steps(baseline, baseline_params, ns_batches, STD_STEPS)
    std_loss = evaluate(baseline, eval_batches, N_HEADS)
    print(f"  Standard 1000-step loss: {std_loss:.4f}")

    random_model, _ = ns_train.make_model(vocab_size, D_MODEL, N_HEADS, STUDENT_LAYERS, D_FF, BLOCK_SIZE, SEED + 99)
    ns_train.move_model_to_device(random_model, DEVICE)
    random_loss = evaluate(random_model, eval_batches, N_HEADS)

    # Results
    lines = []
    lines.append("=== Analytical 100M Init Results ===\n")
    lines.append(f"Teacher: {TEACHER_LAYERS} layers, {t_n:,} params, 500 steps")
    lines.append(f"Student: {STUDENT_LAYERS} layers, {s_n:,} params")
    lines.append(f"Calibration sequences: {N_CALIB}")
    lines.append(f"Analytical solve time: {solve_time:.1f}s")
    lines.append("")
    lines.append(f"Random init loss:           {random_loss:.4f}")
    lines.append(f"Analytical init loss:       {analytical_loss:.4f}")
    lines.append(f"Analytical + 100 FT loss:   {ft_loss:.4f}")
    lines.append(f"Standard 1000-step loss:    {std_loss:.4f}")
    lines.append(f"Teacher (500-step) loss:    {teacher_loss:.4f}")
    lines.append("")
    lines.append(f"Gap (analytical+FT vs std): {ft_loss - std_loss:+.4f}")
    lines.append(f"Speedup: 100 steps vs 1000 steps = 10x fewer gradient updates")

    output = "\n".join(lines)
    print("\n" + output)
    out_path = os.path.join(ns_train.OUT_DIR, "ns_train_analytical_100m_results.txt")
    with open(out_path, "w") as f: f.write(output + "\n")
    print(f"\nResults saved to {out_path}")


def copy_lns_from_teacher_raw(t_lns, student, t_layers, s_layers):
    _, _, s_l, _ = student
    for i in range(s_layers):
        src = min(i, t_layers - 1)
        s_l[i][0].weight.data.copy_(t_lns[src][0]); s_l[i][0].bias.data.copy_(t_lns[src][1])
        s_l[i][5].weight.data.copy_(t_lns[src][2]); s_l[i][5].bias.data.copy_(t_lns[src][3])


if __name__ == "__main__":
    main()

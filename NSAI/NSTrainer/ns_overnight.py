#!/usr/bin/env python3
"""Overnight training: 50M, 100M, 300M analytical pipelines on C4."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import gc, math, time, resource, ctypes, traceback, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import tiktoken
import ns_train

NST_DIR = "/mnt/HIVE/NSTrainer"
MODELS_DIR = "/mnt/HIVE/models"
C4_PATH = os.path.join(NST_DIR, ns_train.C4_PATH)
BATCH_SIZE = 1
BLOCK_SIZE = 128
N_CALIB = 50
LR = 3e-4
SEED = 1337
DTYPE = torch.bfloat16
RAM_LIMIT_MB = 3500

DEVICE = torch.device("cpu")
ENC = tiktoken.get_encoding("gpt2")
VOCAB_SIZE = ENC.n_vocab

PIPELINES = [
    {
        "name": "50M",
        "d_model": 512, "n_heads": 8, "d_ff": 2048,
        "t_layers": 3, "t_steps": 400,
        "s_layers": 8, "ft_steps": 500,
        "teacher_ckpt": os.path.join(MODELS_DIR, "teacher_512.pt"),
        "acts_dir": os.path.join(MODELS_DIR, "teacher_acts_512"),
        "output": os.path.join(MODELS_DIR, "ns_50m.pt"),
    },
    {
        "name": "100M",
        "d_model": 768, "n_heads": 12, "d_ff": 3072,
        "t_layers": 3, "t_steps": 400,
        "s_layers": 12, "ft_steps": 500,
        "teacher_ckpt": os.path.join(MODELS_DIR, "teacher_768.pt"),
        "acts_dir": os.path.join(MODELS_DIR, "teacher_acts_768"),
        "output": os.path.join(MODELS_DIR, "ns_100m.pt"),
    },
    {
        "name": "300M",
        "d_model": 1024, "n_heads": 16, "d_ff": 4096,
        "t_layers": 3, "t_steps": 400,
        "s_layers": 16, "ft_steps": 300,
        "teacher_ckpt": os.path.join(MODELS_DIR, "teacher_1024.pt"),
        "acts_dir": os.path.join(MODELS_DIR, "teacher_acts_1024"),
        "output": os.path.join(MODELS_DIR, "ns_300m.pt"),
    },
]


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


def make_model(vocab_size, d_model, n_heads, n_layers, d_ff, block_size, seed, tied=True):
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
    lm_head = nn.Linear(d_model, vocab_size, bias=False)
    if tied:
        lm_head.weight = tok_emb
    params = [tok_emb, pos_emb]
    for l in layers:
        for m in l:
            params.extend(m.parameters())
    if not tied:
        params.extend(lm_head.parameters())
    return (tok_emb, pos_emb, layers, lm_head), params

def move_model(model, device, dtype=DTYPE):
    tok_emb, pos_emb, layers, lm_head = model
    tok_emb.data = tok_emb.data.to(device=device, dtype=dtype)
    pos_emb.data = pos_emb.data.to(device=device, dtype=dtype)
    for l in layers:
        for m in l:
            m.to(device=device, dtype=dtype)
    lm_head.to(device=device, dtype=dtype)

def _layer_forward(h, ln1, q, k, v, c, ln2, f1, f2, n_heads, T):
    B = h.size(0)
    a = ln1(h)
    head_dim = h.size(-1) // n_heads
    Q = q(a).view(B, T, n_heads, head_dim).transpose(1, 2)
    K = k(a).view(B, T, n_heads, head_dim).transpose(1, 2)
    V = v(a).view(B, T, n_heads, head_dim).transpose(1, 2)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(head_dim)
    mask = torch.tril(torch.ones(T, T, device=h.device, dtype=torch.bool)).view(1, 1, T, T)
    scores = scores.masked_fill(~mask, float("-inf"))
    out = F.softmax(scores, dim=-1) @ V
    out = out.transpose(1, 2).contiguous().view(B, T, -1)
    h = h + c(out)
    f = ln2(h)
    h = h + f2(F.gelu(f1(f)))
    return h

def forward(x, tok_emb, pos_emb, layers, lm_head, n_heads):
    B, T = x.shape
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        h = tok_emb[x] + pos_emb[torch.arange(T, device=x.device)]
        for ln1, q, k, v, c, ln2, f1, f2 in layers:
            h = checkpoint(_layer_forward, h, ln1, q, k, v, c, ln2, f1, f2, n_heads, T, use_reentrant=False)
        return lm_head(h)

def solve_linear(A, B, has_bias=True, lambda_val=1e-6):
    A_reg = A + lambda_val * torch.eye(A.size(0), device=A.device, dtype=A.dtype)
    W = torch.linalg.solve(A_reg, B)
    if has_bias:
        return W[:-1], W[-1]
    return W, None

def capture_pass(x, tok_emb, pos_emb, layers, lm_head, n_heads, skip_lm=True):
    B, T = x.shape
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        h = tok_emb[x] + pos_emb[torch.arange(T, device=x.device)]
        acts = {}
        for li, (ln1, q, k, v, c, ln2, f1, f2) in enumerate(layers):
            a_in = ln1(h)
            Q, K, V = q(a_in), k(a_in), v(a_in)
            acts[f"q_{li}"] = (a_in.view(-1, a_in.size(-1)).float(), Q.view(-1, Q.size(-1)).float())
            acts[f"k_{li}"] = (a_in.view(-1, a_in.size(-1)).float(), K.view(-1, K.size(-1)).float())
            acts[f"v_{li}"] = (a_in.view(-1, a_in.size(-1)).float(), V.view(-1, V.size(-1)).float())
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
            acts[f"c_{li}"] = (out.view(-1, out.size(-1)).float(), c_out.view(-1, c_out.size(-1)).float())
            h = h + c_out
            f_in = ln2(h)
            pre = f1(f_in); f1_out = F.gelu(pre)
            acts[f"f1_{li}"] = (f_in.view(-1, f_in.size(-1)).float(), pre.view(-1, pre.size(-1)).float())
            f2_out = f2(f1_out)
            acts[f"f2_{li}"] = (f1_out.view(-1, f1_out.size(-1)).float(), f2_out.view(-1, f2_out.size(-1)).float())
            h = h + f2_out
        if not skip_lm:
            acts["lm_head"] = (h.view(-1, h.size(-1)).float(), lm_head(h).view(-1, lm_head(h).size(-1)).float())
    return acts

def save_teacher_acts(data, model, n_heads, n_seq, cache_dir, block_size):
    tok_emb, pos_emb, layers, lm_head = model
    os.makedirs(cache_dir, exist_ok=True)
    N = n_seq * block_size
    memmaps, files = {}, {}
    skip_names = {"tok_emb", "pos_emb", "lm_head"}
    batches = ns_train.make_batches(data, block_size, 1, n_seq, SEED + 42)
    for bi, (x, _) in enumerate(batches):
        x = x.to(DEVICE)
        acts = capture_pass(x, tok_emb, pos_emb, layers, lm_head, n_heads, skip_lm=True)
        acts = {k: v for k, v in acts.items() if k not in skip_names}
        if bi == 0:
            for name, (X, Y) in acts.items():
                out_dim = Y.size(1)
                path = os.path.join(cache_dir, f"{name}.bin")
                memmaps[name] = np.memmap(path, dtype=np.float32, mode="w+", shape=(N, out_dim))
                files[name] = (path, out_dim)
        for name, (X, Y) in acts.items():
            memmaps[name][bi*block_size:(bi+1)*block_size] = Y.detach().cpu().numpy().astype(np.float32)
        del acts, x
        release_memory()
    for arr in memmaps.values(): arr.flush()
    del memmaps
    return files

def load_teacher_acts(cache_dir, n_seq, block_size, n_layers):
    files = {}
    names = []
    for i in range(n_layers):
        names.extend([f"q_{i}", f"k_{i}", f"v_{i}", f"c_{i}", f"f1_{i}", f"f2_{i}"])
    for name in names:
        path = os.path.join(cache_dir, f"{name}.bin")
        if not os.path.exists(path):
            return None
        arr = np.memmap(path, dtype=np.float32, mode="r", shape=(n_seq * block_size, -1))
        files[name] = (path, arr.shape[1])
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

def get_target_name(name, t_layers):
    if name in ("tok_emb", "pos_emb", "lm_head"):
        return name
    li = int(name.split("_")[1])
    if li < t_layers:
        return name
    kind = name.rsplit("_", 1)[0]
    return f"{kind}_{t_layers - 1}"

def save_model_checkpoint(model, path, cfg, extra=None):
    tok_emb, pos_emb, layers, lm_head = model
    state = {
        "tok_emb": tok_emb.data.cpu().float(),
        "pos_emb": pos_emb.data.cpu().float(),
        "layers": [[{k: v.cpu().float() for k, v in m.state_dict().items()} for m in l] for l in layers],
        "config": cfg,
    }
    if extra:
        state.update(extra)
    torch.save(state, path)

def load_checkpoint(path):
    state = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg = state["config"]
    model, _ = make_model(cfg["vocab_size"], cfg["d_model"], cfg["n_heads"],
                          cfg["n_layers"], cfg["d_ff"], BLOCK_SIZE, SEED)
    tok_emb, pos_emb, layers, lm_head = model
    tok_emb.data = state["tok_emb"].to(DEVICE, dtype=DTYPE)
    saved_pos = state["pos_emb"].to(DEVICE, dtype=DTYPE)
    pos_emb.data = saved_pos[:BLOCK_SIZE] if saved_pos.size(0) >= BLOCK_SIZE else saved_pos
    for l, sd_list in zip(layers, state["layers"]):
        for m, sd in zip(l, sd_list):
            sd_bf = {k: v.to(DEVICE, dtype=DTYPE) for k, v in sd.items()}
            m.load_state_dict(sd_bf)
    lm_head.to(DEVICE, dtype=DTYPE)
    cfg["block_size"] = BLOCK_SIZE
    return model, cfg

def make_optimizer(params):
    try:
        import bitsandbytes as bnb
        return bnb.optim.Adam8bit(params, lr=LR)
    except ImportError:
        return torch.optim.AdamW(params, lr=LR)

def train_steps(model, params, batches, n_steps, label, n_heads):
    tok_emb, pos_emb, layers, lm_head = model
    opt = make_optimizer(params)
    losses = {}
    t0 = time.time()
    for step, (x, y) in enumerate(batches[:n_steps], 1):
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = forward(x, tok_emb, pos_emb, layers, lm_head, n_heads)
        logits = logits.float()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        losses[step] = loss.item()
        if step % 100 == 0:
            elapsed = time.time() - t0
            rss = get_rss_mb()
            print(f"  [{label}] step {step}: loss {loss.item():.4f}  time {elapsed:.1f}s  RSS {rss:.0f} MB")
    return losses

def evaluate(model, batches, n_heads):
    tok_emb, pos_emb, layers, lm_head = model
    total, tokens = 0.0, 0
    for x, y in batches:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.no_grad():
            logits = forward(x, tok_emb, pos_emb, layers, lm_head, n_heads)
            logits = logits.float()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum")
            total += loss.item(); tokens += y.numel()
    return total / tokens


def run_pipeline(cfg, data, ns_batches, eval_batches):
    name = cfg["name"]
    d_model = cfg["d_model"]
    n_heads = cfg["n_heads"]
    d_ff = cfg["d_ff"]
    t_layers = cfg["t_layers"]
    t_steps = cfg["t_steps"]
    s_layers = cfg["s_layers"]
    ft_steps = cfg["ft_steps"]
    teacher_ckpt = cfg["teacher_ckpt"]
    acts_dir = cfg["acts_dir"]
    output_path = cfg["output"]

    print(f"\n{'='*60}")
    print(f"Pipeline {name}: d_model={d_model}, teacher={t_layers}L, student={s_layers}L, d_ff={d_ff}")
    print(f"{'='*60}")

    model_cfg = {"d_model": d_model, "n_heads": n_heads, "d_ff": d_ff,
                 "block_size": BLOCK_SIZE, "vocab_size": VOCAB_SIZE}

    # Step 1: Train or load teacher
    teacher_loss = None
    t_n = 0
    if os.path.exists(teacher_ckpt):
        print(f"\n  Loading teacher from {teacher_ckpt}...")
        teacher, _ = load_checkpoint(teacher_ckpt)
        tok_emb, pos_emb, layers, lm_head = teacher
        teacher_params = [tok_emb, pos_emb]
        for l in layers:
            for m in l:
                teacher_params.extend(m.parameters())
        t_n = sum(p.numel() for p in teacher_params)
        teacher_loss = evaluate(teacher, eval_batches, n_heads)
        print(f"  Teacher: {t_n:,} params, loss={teacher_loss:.4f}, RSS={get_rss_mb():.0f} MB")
    else:
        print(f"\n  Training teacher ({t_layers} layers, {t_steps} steps)...")
        teacher, teacher_params = make_model(VOCAB_SIZE, d_model, n_heads, t_layers, d_ff, BLOCK_SIZE, SEED)
        move_model(teacher, DEVICE)
        t_n = sum(p.numel() for p in teacher_params)
        print(f"  Teacher: {t_n:,} params, RSS={get_rss_mb():.0f} MB")
        t0 = time.time()
        teacher_losses = train_steps(teacher, teacher_params, ns_batches, t_steps, f"{name}-teacher", n_heads)
        print(f"  Time: {time.time()-t0:.1f}s, Final loss: {teacher_losses[t_steps]:.4f}")
        teacher_loss = evaluate(teacher, eval_batches, n_heads)
        print(f"  Eval loss: {teacher_loss:.4f}")
        t_cfg = dict(model_cfg); t_cfg["n_layers"] = t_layers
        save_model_checkpoint(teacher, teacher_ckpt, t_cfg, {"loss": teacher_loss})
        print(f"  Saved teacher to {teacher_ckpt}")
    ram_guard(f"{name}-teacher-loaded")

    # Step 2: Save or load teacher activations
    files = load_teacher_acts(acts_dir, N_CALIB, BLOCK_SIZE, t_layers)
    if files is not None:
        print(f"\n  Loading teacher activations from {acts_dir}...")
        t_lns = save_lns(teacher)
        t_tok_emb_data = teacher[0].data.clone()
        t_pos_emb_data = teacher[1].data.clone()
        del teacher, teacher_params
        release_memory()
        print(f"  Teacher deleted, RSS={get_rss_mb():.0f} MB")
    else:
        print(f"\n  Saving teacher activations ({N_CALIB} sequences)...")
        files = save_teacher_acts(data, teacher, n_heads, N_CALIB, acts_dir, BLOCK_SIZE)
        t_lns = save_lns(teacher)
        t_tok_emb_data = teacher[0].data.clone()
        t_pos_emb_data = teacher[1].data.clone()
        del teacher, teacher_params
        release_memory()
        print(f"  Teacher deleted, RSS={get_rss_mb():.0f} MB")
    ram_guard(f"{name}-acts-loaded")

    # Step 3: Build student and solve analytically
    print(f"\n  Building student ({s_layers} layers) and solving...")
    student, student_params = make_model(VOCAB_SIZE, d_model, n_heads, s_layers, d_ff, BLOCK_SIZE, SEED + 7)
    move_model(student, DEVICE)
    s_n = sum(p.numel() for p in student_params)
    print(f"  Student: {s_n:,} params, RSS={get_rss_mb():.0f} MB")
    copy_lns(t_lns, student, t_layers, s_layers)
    student[0].data.copy_(t_tok_emb_data)
    student[1].data.copy_(t_pos_emb_data)
    del t_tok_emb_data, t_pos_emb_data, t_lns
    release_memory()

    lambda_log = []
    t0 = time.time()
    names = []
    for i in range(s_layers):
        names.extend([f"q_{i}", f"k_{i}", f"v_{i}", f"c_{i}", f"f1_{i}", f"f2_{i}"])

    for lname in names:
        target_name = get_target_name(lname, t_layers)
        path, out_dim = files[target_name]
        Y_all = np.memmap(path, dtype=np.float32, mode="r", shape=(N_CALIB * BLOCK_SIZE, out_dim))
        A, B = None, None
        n_samples = 0
        batches = ns_train.make_batches(data, BLOCK_SIZE, 1, N_CALIB, SEED + 42)
        for bi, (x, _) in enumerate(batches):
            x = x.to(DEVICE)
            acts = capture_pass(x, *student, n_heads, skip_lm=True)
            X, _ = acts[lname]
            X = X.float()
            ones = torch.ones(X.size(0), 1, device=X.device, dtype=torch.float32)
            X = torch.cat([X, ones], dim=1)
            if A is None:
                A = torch.zeros(X.size(1), X.size(1), device=DEVICE, dtype=torch.float32)
                B = torch.zeros(X.size(1), out_dim, device=DEVICE, dtype=torch.float32)
            Y = torch.from_numpy(Y_all[bi*BLOCK_SIZE:(bi+1)*BLOCK_SIZE].copy()).to(DEVICE)
            A.addmm_(X.T, X); B.addmm_(X.T, Y)
            n_samples += X.size(0)
            del X, Y, acts, x
            release_memory()
        d_in = A.size(0)
        lambda_val = (torch.trace(A) / (n_samples * d_in)).item()
        W, b = solve_linear(A, B, has_bias=True, lambda_val=lambda_val)
        lambda_log.append((lname, lambda_val))
        W = W.to(DTYPE)
        b = b.to(DTYPE) if b is not None else None
        set_layer(student, lname, W, b)
        del A, B, Y_all
        release_memory()
        rss = ram_guard(f"{name}-solve-{lname}")
        print(f"    Solved {lname} -> {target_name}  lambda={lambda_val:.6f}  RSS={rss:.0f} MB")
    solve_time = time.time() - t0
    print(f"  Solve time: {solve_time:.1f}s")

    analytical_loss = evaluate(student, eval_batches, n_heads)
    print(f"  Analytical init loss: {analytical_loss:.4f}")

    # Save analytical init checkpoint (in case FT fails)
    s_cfg = dict(model_cfg); s_cfg["n_layers"] = s_layers
    analytical_ckpt = output_path.replace(".pt", "_analytical.pt")
    save_model_checkpoint(student, analytical_ckpt, s_cfg, {"loss": analytical_loss, "stage": "analytical"})
    print(f"  Saved analytical checkpoint to {analytical_ckpt}")

    # Step 4: Fine-tune
    print(f"\n  Fine-tuning ({ft_steps} steps)...")
    ft_losses = train_steps(student, student_params, ns_batches[t_steps:], ft_steps, f"{name}-ft", n_heads)
    ft_loss = evaluate(student, eval_batches, n_heads)
    print(f"  Post fine-tune loss: {ft_loss:.4f}")

    # Step 5: Save final checkpoint
    print(f"\n  Saving final checkpoint to {output_path}...")
    save_model_checkpoint(student, output_path, s_cfg, {"loss": ft_loss, "stage": "final"})
    print(f"  Saved. RSS={get_rss_mb():.0f} MB")

    # Cleanup
    del student, student_params
    release_memory()

    return {
        "name": name, "teacher_params": t_n, "student_params": s_n,
        "solve_time": solve_time, "analytical_loss": analytical_loss,
        "ft_loss": ft_loss, "teacher_loss": teacher_loss,
        "lambda_log": lambda_log,
    }


def main():
    global RAM_LIMIT_MB
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Run pipeline 1 only, 2.5GB RAM limit")
    parser.add_argument("--ram-limit", type=int, default=None, help="Override RAM limit in MB")
    args = parser.parse_args()

    if args.local:
        pipelines = [PIPELINES[0]]
        RAM_LIMIT_MB = 2500
        print("LOCAL MODE: pipeline 1 only, RAM limit 2.5GB")
    else:
        pipelines = PIPELINES
        if args.ram_limit:
            RAM_LIMIT_MB = args.ram_limit
        print(f"FULL MODE: all pipelines, RAM limit {RAM_LIMIT_MB}MB")

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(NST_DIR, exist_ok=True)

    print(f"Loading C4 data...")
    with open(C4_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    tokens = ENC.encode(text)
    del text
    data = torch.tensor(tokens, dtype=torch.long)
    del tokens
    print(f"C4: {len(data)} tokens, vocab={VOCAB_SIZE}")

    print("Computing unigram NS filter...")
    token_info = ns_train.compute_token_info(data, VOCAB_SIZE)
    threshold = torch.quantile(token_info[data], 0.5)
    max_steps = max(p["t_steps"] + p["ft_steps"] for p in pipelines)
    ns_batches = ns_train.make_batches_ns(data, BLOCK_SIZE, BATCH_SIZE, max_steps, token_info, threshold, SEED)
    del token_info
    release_memory()
    print(f"RSS after data prep: {get_rss_mb():.0f} MB")

    eval_batches = ns_train.make_batches(data, BLOCK_SIZE, BATCH_SIZE, 50, SEED + 123)

    results = []
    for cfg in pipelines:
        try:
            result = run_pipeline(cfg, data, ns_batches, eval_batches)
            results.append(result)
            print(f"\n  Pipeline {cfg['name']} COMPLETE: ft_loss={result['ft_loss']:.4f}")
        except Exception as e:
            print(f"\n  Pipeline {cfg['name']} FAILED: {e}")
            traceback.print_exc()
            release_memory()
            results.append({"name": cfg["name"], "error": str(e)})
            continue

    # Final summary
    print(f"\n{'='*60}")
    print("OVERNIGHT PIPELINE SUMMARY")
    print(f"{'='*60}")
    for r in results:
        if "error" in r:
            print(f"  {r['name']}: FAILED - {r['error']}")
        else:
            print(f"  {r['name']}: teacher_loss={r['teacher_loss']:.4f}  analytical={r['analytical_loss']:.4f}  ft={r['ft_loss']:.4f}  "
                  f"params={r['student_params']:,}  solve={r['solve_time']:.0f}s")

    summary_path = os.path.join(MODELS_DIR, "ns_overnight_summary.txt")
    with open(summary_path, "w") as f:
        f.write("NS Overnight Pipeline Summary\n\n")
        for r in results:
            if "error" in r:
                f.write(f"{r['name']}: FAILED - {r['error']}\n")
            else:
                f.write(f"{r['name']}:\n")
                f.write(f"  Teacher: {r['teacher_params']:,} params, loss={r['teacher_loss']:.4f}\n")
                f.write(f"  Student: {r['student_params']:,} params\n")
                f.write(f"  Solve time: {r['solve_time']:.1f}s\n")
                f.write(f"  Analytical init loss: {r['analytical_loss']:.4f}\n")
                f.write(f"  Fine-tune loss: {r['ft_loss']:.4f}\n")
                f.write(f"  Lambda per layer:\n")
                for lname, lval in r["lambda_log"]:
                    f.write(f"    {lname:10s}  lambda={lval:.6f}\n")
                f.write("\n")
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()

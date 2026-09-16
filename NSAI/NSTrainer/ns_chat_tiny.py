#!/usr/bin/env python3
"""Smallest chat model: analytical init from tiny teacher on C4 with 4K BPE."""

import gc, math, os, time, resource, ctypes, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import ns_train

D_MODEL = 256
N_HEADS = 4
D_FF = 512
BLOCK_SIZE = 128
BATCH_SIZE = 4
LR = 1e-3
SEED = 1337
TEACHER_LAYERS = 2
STUDENT_LAYERS = 4
N_CALIB = 25
TEACHER_STEPS = 300
FT_STEPS = 150
RAM_LIMIT_MB = 2000
VOCAB_SIZE = 50257

NST_DIR = "/mnt/HIVE/NSTrainer"
CKPT_PATH = os.path.join(NST_DIR, "ns_chat_v2.pt")
TEACHER_CKPT = os.path.join(NST_DIR, "teacher_v2.pt")
ACTS_DIR = os.path.join(NST_DIR, "teacher_acts_v2")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENC = tiktoken.get_encoding("gpt2")


def encode_text(text):
    return ENC.encode(text)

def decode_ids(ids):
    return ENC.decode(ids)

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


def move_model(model, device):
    tok_emb, pos_emb, layers, lm_head = model
    tok_emb.data = tok_emb.data.to(device)
    pos_emb.data = pos_emb.data.to(device)
    for l in layers:
        for m in l:
            m.to(device)
    lm_head.to(device)


def forward(x, tok_emb, pos_emb, layers, lm_head, n_heads):
    B, T = x.shape
    h = tok_emb[x] + pos_emb[torch.arange(T, device=x.device)]
    for ln1, q, k, v, c, ln2, f1, f2 in layers:
        a = ln1(h)
        head_dim = h.size(-1) // n_heads
        Q = q(a).view(B, T, n_heads, head_dim).transpose(1, 2)
        K = k(a).view(B, T, n_heads, head_dim).transpose(1, 2)
        V = v(a).view(B, T, n_heads, head_dim).transpose(1, 2)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(head_dim)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool)).view(1, 1, T, T)
        scores = scores.masked_fill(~mask, float("-inf"))
        out = F.softmax(scores, dim=-1) @ V
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        h = h + c(out)
        f = ln2(h)
        h = h + f2(F.gelu(f1(f)))
    return lm_head(h)


def build_one_hot(ids, n):
    return F.one_hot(ids, num_classes=n).float()

def solve_linear(A, B, has_bias=True, lambda_val=1e-6):
    A_reg = A + lambda_val * torch.eye(A.size(0), device=A.device, dtype=A.dtype)
    W = torch.linalg.solve(A_reg, B)
    if has_bias:
        return W[:-1], W[-1]
    return W, None

def capture_pass(x, tok_emb, pos_emb, layers, lm_head, n_heads, skip_emb=True, skip_lm=False):
    B, T = x.shape
    h = tok_emb[x] + pos_emb[torch.arange(T, device=x.device)]
    acts = {}
    if not skip_emb:
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
    if not skip_lm:
        acts["lm_head"] = (h.view(-1, h.size(-1)), lm_head(h).view(-1, lm_head(h).size(-1)))
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
        acts = capture_pass(x, tok_emb, pos_emb, layers, lm_head, n_heads, skip_emb=True)
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

def save_model_checkpoint(model, path, cfg_extra=None):
    tok_emb, pos_emb, layers, lm_head = model
    state = {
        "tok_emb": tok_emb.data.cpu(),
        "pos_emb": pos_emb.data.cpu(),
        "layers": [[m.state_dict() for m in l] for l in layers],
        "config": {"d_model": D_MODEL, "n_heads": N_HEADS, "n_layers": len(layers),
                   "d_ff": D_FF, "block_size": BLOCK_SIZE, "vocab_size": VOCAB_SIZE},
    }
    if cfg_extra:
        state.update(cfg_extra)
    torch.save(state, path)

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

def train_steps(model, params, batches, n_steps, label="", save_every=0, save_prefix=None):
    tok_emb, pos_emb, layers, lm_head = model
    opt = torch.optim.AdamW(params, lr=LR)
    losses = {}
    t0 = time.time()
    for step, (x, y) in enumerate(batches[:n_steps], 1):
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = forward(x, tok_emb, pos_emb, layers, lm_head, N_HEADS)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        losses[step] = loss.item()
        if step % 100 == 0:
            elapsed = time.time() - t0
            rss = get_rss_mb()
            print(f"  [{label}] step {step}: loss {loss.item():.4f}  time {elapsed:.1f}s  RSS {rss:.0f} MB")
        if save_every and save_prefix and step % save_every == 0:
            ckpt = os.path.join(NST_DIR, f"{save_prefix}_step{step}.pt")
            save_model_checkpoint(model, ckpt, {"loss": loss.item(), "step": step})
            print(f"  [checkpoint] saved {ckpt}")
    return losses

def evaluate(model, batches):
    tok_emb, pos_emb, layers, lm_head = model
    total, tokens = 0.0, 0
    for x, y in batches:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = forward(x, tok_emb, pos_emb, layers, lm_head, N_HEADS)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum")
        total += loss.item(); tokens += y.numel()
    return total / tokens


def run_training():
    os.makedirs(NST_DIR, exist_ok=True)

    with open(os.path.join(NST_DIR, ns_train.C4_PATH), "r", encoding="utf-8") as f:
        text = f.read()
    tokens = encode_text(text)
    del text
    data = torch.tensor(tokens, dtype=torch.long)
    del tokens
    print(f"C4 data: {len(data)} tokens, vocab={VOCAB_SIZE}")

    # NS unigram filter
    print("Computing unigram info...")
    token_info = ns_train.compute_token_info(data, VOCAB_SIZE)
    threshold = torch.quantile(token_info[data], 0.5)
    ns_batches = ns_train.make_batches_ns(data, BLOCK_SIZE, BATCH_SIZE, TEACHER_STEPS + FT_STEPS, token_info, threshold, SEED)
    del token_info
    release_memory()
    print(f"RSS: {get_rss_mb():.0f} MB")

    eval_batches = ns_train.make_batches(data, BLOCK_SIZE, BATCH_SIZE, 50, SEED + 123)

    # Step 1: Train or load teacher
    teacher_loss = None
    t_n = 0
    if os.path.exists(TEACHER_CKPT):
        print(f"\nStep 1: Loading teacher from {TEACHER_CKPT}...")
        teacher, teacher_params = load_checkpoint(TEACHER_CKPT)
        t_n = sum(p.numel() for p in teacher_params)
        teacher_loss = evaluate(teacher, eval_batches)
        print(f"  Teacher params: {t_n:,}  loss: {teacher_loss:.4f}  RSS: {get_rss_mb():.0f} MB")
    else:
        print(f"\nStep 1: Training teacher ({TEACHER_LAYERS} layers, {TEACHER_STEPS} steps)...")
        teacher, teacher_params = make_model(VOCAB_SIZE, D_MODEL, N_HEADS, TEACHER_LAYERS, D_FF, BLOCK_SIZE, SEED)
        move_model(teacher, DEVICE)
        t_n = sum(p.numel() for p in teacher_params)
        print(f"  Teacher params: {t_n:,}  RSS: {get_rss_mb():.0f} MB")
        t0 = time.time()
        teacher_losses = train_steps(teacher, teacher_params, ns_batches, TEACHER_STEPS, "teacher")
        print(f"  Time: {time.time()-t0:.1f}s  Final loss: {teacher_losses[TEACHER_STEPS]:.4f}")
        teacher_loss = evaluate(teacher, eval_batches)
        print(f"  Eval loss: {teacher_loss:.4f}")
        save_model_checkpoint(teacher, TEACHER_CKPT, {"loss": teacher_loss})
        print(f"  Saved teacher to {TEACHER_CKPT}")

    # Step 2: Save or load teacher activations
    files = load_teacher_acts(ACTS_DIR, N_CALIB, BLOCK_SIZE, TEACHER_LAYERS)
    if files is not None:
        print(f"\nStep 2: Loading teacher activations from {ACTS_DIR}...")
        t_lns = save_lns(teacher)
        t_tok_emb_data = teacher[0].data.clone()
        t_pos_emb_data = teacher[1].data.clone()
        del teacher, teacher_params
        release_memory()
        print(f"  Teacher deleted. RSS: {get_rss_mb():.0f} MB")
    else:
        print(f"\nStep 2: Saving teacher activations ({N_CALIB} sequences)...")
        files = save_teacher_acts(data, teacher, N_HEADS, N_CALIB, ACTS_DIR, BLOCK_SIZE)
        t_lns = save_lns(teacher)
        t_tok_emb_data = teacher[0].data.clone()
        t_pos_emb_data = teacher[1].data.clone()
        del teacher, teacher_params
        release_memory()
        print(f"  Teacher deleted. RSS: {get_rss_mb():.0f} MB")

    # Step 3: Build student and solve analytically
    print(f"\nStep 3: Building student ({STUDENT_LAYERS} layers) and solving...")
    student, student_params = make_model(VOCAB_SIZE, D_MODEL, N_HEADS, STUDENT_LAYERS, D_FF, BLOCK_SIZE, SEED + 7)
    move_model(student, DEVICE)
    s_n = sum(p.numel() for p in student_params)
    print(f"  Student params: {s_n:,}  RSS: {get_rss_mb():.0f} MB")
    copy_lns(t_lns, student, TEACHER_LAYERS, STUDENT_LAYERS)
    student[0].data.copy_(t_tok_emb_data)
    student[1].data.copy_(t_pos_emb_data)
    del t_tok_emb_data, t_pos_emb_data
    release_memory()

    lambda_log = []
    t0 = time.time()
    names = []
    for i in range(STUDENT_LAYERS):
        names.extend([f"q_{i}", f"k_{i}", f"v_{i}", f"c_{i}", f"f1_{i}", f"f2_{i}"])

    for name in names:
        target_name = get_target_name(name, TEACHER_LAYERS)
        path, out_dim = files[target_name]
        Y_all = np.memmap(path, dtype=np.float32, mode="r", shape=(N_CALIB * BLOCK_SIZE, out_dim))
        A, B = None, None
        n_samples = 0
        batches = ns_train.make_batches(data, BLOCK_SIZE, 1, N_CALIB, SEED + 42)
        for bi, (x, _) in enumerate(batches):
            x = x.to(DEVICE)
            acts = capture_pass(x, *student, N_HEADS, skip_lm=True)
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
        d_in = A.size(0)
        lambda_val = (torch.trace(A) / (n_samples * d_in)).item()
        W, b = solve_linear(A, B, has_bias=(name not in ("tok_emb", "pos_emb")), lambda_val=lambda_val)
        lambda_log.append((name, lambda_val))
        set_layer(student, name, W, b)
        del A, B, Y_all
        release_memory()
        rss = ram_guard(f"solve {name}")
        print(f"  Solved {name} -> {target_name}  lambda={lambda_val:.6f}  RSS: {rss:.0f} MB")
    solve_time = time.time() - t0
    print(f"  Solve time: {solve_time:.1f}s")

    analytical_loss = evaluate(student, eval_batches)
    print(f"  Analytical init loss: {analytical_loss:.4f}")

    # Step 4: Fine-tune with checkpoints every 50 steps
    print(f"\nStep 4: Fine-tuning ({FT_STEPS} steps, checkpoint every 50)...")
    ft_losses = train_steps(student, student_params, ns_batches[TEACHER_STEPS:], FT_STEPS, "finetune",
                            save_every=50, save_prefix="student_v2")
    ft_loss = evaluate(student, eval_batches)
    print(f"  Post fine-tune loss: {ft_loss:.4f}")

    # Step 5: Save final checkpoint
    print(f"\nStep 5: Saving final checkpoint to {CKPT_PATH}...")
    save_model_checkpoint(student, CKPT_PATH, {"loss": ft_loss})
    print(f"  Saved. RSS: {get_rss_mb():.0f} MB")

    # Results
    lines = []
    lines.append("=== NS Chat v2 Results ===\n")
    lines.append(f"Tokenizer: 4K BPE, vocab={VOCAB_SIZE}")
    lines.append(f"Teacher: {TEACHER_LAYERS} layers, {t_n:,} params, {TEACHER_STEPS} steps")
    lines.append(f"Student: {STUDENT_LAYERS} layers, {s_n:,} params")
    lines.append(f"Solve time: {solve_time:.1f}s")
    lines.append(f"Analytical init loss: {analytical_loss:.4f}")
    lines.append(f"Fine-tune loss ({FT_STEPS} steps): {ft_loss:.4f}")
    lines.append(f"Teacher loss: {teacher_loss:.4f}")
    lines.append("")
    lines.append("Ledoit-Wolf lambda per layer:")
    for lname, lval in lambda_log:
        lines.append(f"  {lname:10s}  lambda={lval:.6f}")
    output = "\n".join(lines)
    print("\n" + output)
    out_path = os.path.join(NST_DIR, "ns_chat_v2_results.txt")
    with open(out_path, "w") as f: f.write(output + "\n")
    print(f"Results saved to {out_path}")


def load_checkpoint(path):
    state = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg = state["config"]
    model, _ = make_model(cfg["vocab_size"], cfg["d_model"], cfg["n_heads"],
                          cfg["n_layers"], cfg["d_ff"], cfg["block_size"], SEED)
    tok_emb, pos_emb, layers, lm_head = model
    tok_emb.data = state["tok_emb"].to(DEVICE)
    pos_emb.data = state["pos_emb"].to(DEVICE)
    for l, sd_list in zip(layers, state["layers"]):
        for m, sd in zip(l, sd_list):
            m.load_state_dict(sd)
            m.to(DEVICE)
    lm_head.to(DEVICE)
    return model, cfg


def generate(model, prompt_ids, n_tokens=200, top_k=40, temperature=0.8):
    tok_emb, pos_emb, layers, lm_head = model
    ids = list(prompt_ids)
    for _ in range(n_tokens):
        x = torch.tensor([ids[-BLOCK_SIZE:]], device=DEVICE)
        with torch.no_grad():
            logits = forward(x, tok_emb, pos_emb, layers, lm_head, N_HEADS)
        logits = logits[0, -1] / temperature
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[-1]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        ids.append(next_id)
    return ids


def run_chat(ckpt_path):
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}. Run training first.")
        return
    print(f"Loading checkpoint from {ckpt_path}...")
    model, cfg = load_checkpoint(ckpt_path)
    print(f"Model: {cfg['n_layers']} layers, d_model={cfg['d_model']}, vocab={cfg['vocab_size']}")
    print()
    while True:
        try:
            prompt = input("NSChat > ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not prompt.strip():
            continue
        prompt_ids = encode_text(prompt)
        out_ids = generate(model, prompt_ids, n_tokens=200)
        response = decode_ids(out_ids[len(prompt_ids):])
        print(response)
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=CKPT_PATH)
    args = parser.parse_args()
    if args.chat:
        run_chat(args.checkpoint)
    else:
        run_training()


if __name__ == "__main__":
    main()

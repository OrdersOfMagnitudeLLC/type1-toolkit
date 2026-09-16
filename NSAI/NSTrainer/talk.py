#!/usr/bin/env python3
"""Quick inference test for ns_252m and ns_300m checkpoints."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

DEVICE = torch.device("cpu")
DTYPE = torch.bfloat16
BLOCK_SIZE = 128
ENC = tiktoken.get_encoding("gpt2")
VOCAB_SIZE = ENC.n_vocab

def make_model(vocab_size, d_model, n_heads, n_layers, d_ff, block_size):
    tok_emb = nn.Parameter(torch.empty(vocab_size, d_model))
    pos_emb = nn.Parameter(torch.empty(block_size, d_model))
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
    lm_head.weight = tok_emb
    return (tok_emb, pos_emb, layers, lm_head)

def load_checkpoint(path):
    state = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg = state["config"]
    model = make_model(cfg["vocab_size"], cfg["d_model"], cfg["n_heads"],
                       cfg["n_layers"], cfg["d_ff"], BLOCK_SIZE)
    tok_emb, pos_emb, layers, lm_head = model
    tok_emb.data = state["tok_emb"].to(DEVICE, dtype=DTYPE)
    saved_pos = state["pos_emb"].to(DEVICE, dtype=DTYPE)
    pos_emb.data = saved_pos[:BLOCK_SIZE] if saved_pos.size(0) >= BLOCK_SIZE else saved_pos
    for l, sd_list in zip(layers, state["layers"]):
        for m, sd in zip(l, sd_list):
            sd_bf = {k: v.to(DEVICE, dtype=DTYPE) for k, v in sd.items()}
            m.load_state_dict(sd_bf)
    return model, cfg

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

@torch.no_grad()
def generate(model, prompt_text, n_tokens=100, temperature=0.8, top_k=40):
    tok_emb, pos_emb, layers, lm_head = model
    n_heads = 16  # both 252M and 300M use d_model=1024, n_heads=16
    
    ids = ENC.encode(prompt_text)
    if len(ids) > BLOCK_SIZE:
        ids = ids[-BLOCK_SIZE:]
    
    generated = list(ids)
    
    for _ in range(n_tokens):
        x = torch.tensor([generated[-BLOCK_SIZE:]], device=DEVICE, dtype=torch.long)
        T = x.size(1)
        
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            h = tok_emb[x] + pos_emb[torch.arange(T, device=x.device)]
            for ln1, q, k, v, c, ln2, f1, f2 in layers:
                h = _layer_forward(h, ln1, q, k, v, c, ln2, f1, f2, n_heads, T)
            logits = lm_head(h[:, -1, :])
        
        logits = logits.float() / temperature
        if top_k > 0:
            v_top, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v_top[:, -1:]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_id)
    
    return ENC.decode(generated)

def main():
    prompts = [
        "The future of artificial intelligence is",
        "Once upon a time in a galaxy far away",
        "The most important thing in life is",
    ]
    
    _models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    for ckpt_path, name in [
        (os.path.join(_models_dir, "ns_252m.pt"), "252M"),
        (os.path.join(_models_dir, "ns_300m.pt"), "300M"),
    ]:
        print(f"\n{'='*60}")
        print(f"  Loading {name} from {ckpt_path}...")
        print(f"{'='*60}")
        
        if not os.path.exists(ckpt_path):
            print(f"  CHECKPOINT NOT FOUND: {ckpt_path}")
            continue
        
        model, cfg = load_checkpoint(ckpt_path)
        tok_emb, pos_emb, layers, lm_head = model
        params = sum(p.numel() for p in tok_emb.unsqueeze(0)) + pos_emb.numel()
        for l in layers:
            for m in l:
                params += sum(p.numel() for p in m.parameters())
        print(f"  Config: d_model={cfg['d_model']}, n_layers={cfg['n_layers']}, d_ff={cfg['d_ff']}")
        print(f"  Params: {params:,}")
        if "loss" in cfg or "loss" in (state := torch.load(ckpt_path, map_location=DEVICE, weights_only=False)):
            state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            if "loss" in state:
                print(f"  Loss: {state['loss']:.4f}")
        
        for prompt in prompts:
            print(f"\n  Prompt: \"{prompt}\"")
            output = generate(model, prompt, n_tokens=80, temperature=0.7, top_k=40)
            print(f"  Output: {output}")
            print()
        
        del model
        import gc; gc.collect()

if __name__ == "__main__":
    main()

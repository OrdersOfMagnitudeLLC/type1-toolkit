#!/usr/bin/env python3
"""Quick inference test for the 563M KD student checkpoint."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_student_kd import StudentGPT

CKPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "ns_500m_kd.pt")
DEVICE = torch.device("cpu")
DTYPE = torch.bfloat16


def load_model(path):
    state = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg = state["config"]
    model = StudentGPT(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        d_ff=cfg["d_ff"],
        block_size=cfg["block_size"],
        tied=cfg.get("tied", False),
    )
    model.load_state_dict(state["state_dict"])
    del state
    import gc; gc.collect()
    model = model.to(DEVICE, dtype=DTYPE)
    model.eval()
    return model, cfg


@torch.no_grad()
def generate(model, tokenizer, prompt_text, n_tokens=80, temperature=0.7, top_k=40):
    block_size = model.pos_emb.num_embeddings

    ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    if len(ids) > block_size:
        ids = ids[-block_size:]

    generated = list(ids)

    for _ in range(n_tokens):
        x = torch.tensor([generated[-block_size:]], device=DEVICE, dtype=torch.long)
        T = x.size(1)

        with torch.autocast(device_type="cpu", dtype=DTYPE):
            h = model(x)
            logits = model.lm_head(h[:, -1, :])

        logits = logits.float() / temperature
        if top_k > 0:
            v_top, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v_top[:, -1:]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_id)

    return tokenizer.decode(generated)


def main():
    from transformers import AutoTokenizer

    print("Loading Qwen tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

    print(f"Loading checkpoint from {CKPT_PATH}...")
    model, cfg = load_model(CKPT_PATH)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Config: d_model={cfg['d_model']}, n_layers={cfg['n_layers']}, "
          f"n_heads={cfg['n_heads']}, d_ff={cfg['d_ff']}, vocab={cfg['vocab_size']}")
    print(f"  Params: {n_params:,} ({n_params/1e6:.1f}M)")

    prompts = [
        "The future of artificial intelligence is",
        "Once upon a time in a galaxy far away",
        "The most important thing in life is",
        "In a world where technology advances rapidly",
    ]

    for prompt in prompts:
        print(f"\n{'='*60}")
        print(f"  Prompt: \"{prompt}\"")
        print(f"{'='*60}")
        output = generate(model, tokenizer, prompt, n_tokens=80, temperature=0.7, top_k=40)
        print(f"  Output: {output}")
        print()

    del model
    import gc; gc.collect()


if __name__ == "__main__":
    main()

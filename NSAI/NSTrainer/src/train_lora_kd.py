#!/usr/bin/env python3
"""Train LoRA adapters on Qwen2.5-0.5B-Instruct using cached 3B teacher logits.

Combines knowledge distillation (KL over top-k teacher logits) with standard
cross-entropy on input tokens to prevent gibberish. After training, merges
LoRA into base weights and saves the merged model.
"""

import os
import gc
import json
import math
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def cosine_lr(step, total_steps, lr_max, lr_min_ratio=0.1):
    if step >= total_steps:
        return lr_max * lr_min_ratio
    progress = step / total_steps
    return lr_max * lr_min_ratio + 0.5 * (lr_max - lr_max * lr_min_ratio) * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Chunked KD loss: only over top-k positions, never full vocab
# ---------------------------------------------------------------------------

def kd_loss_chunked(student_logits, teacher_indices, teacher_values, chunk_size=32):
    """KL(softmax(teacher_vals) || softmax(student_logits_at_indices)) over top-k.

    student_logits:  (B, T, V)   full student logits (we gather only top-k)
    teacher_indices: (B, T, K)   top-k vocab indices from teacher
    teacher_values:  (B, T, K)   teacher logit values at those indices
    """
    B, T, V = student_logits.shape
    K = teacher_indices.size(-1)
    n = B * T

    s_flat = student_logits.reshape(n, V)
    idx_flat = teacher_indices.reshape(n, K)
    val_flat = teacher_values.reshape(n, K)

    total = s_flat.new_tensor(0.0)
    for s in range(0, n, chunk_size):
        e = min(s + chunk_size, n)
        idx_c = idx_flat[s:e]                          # (chunk, K)
        val_c = val_flat[s:e]                          # (chunk, K)

        logits_c = s_flat[s:e].gather(1, idx_c)        # (chunk, K)

        t_probs = F.softmax(val_c, dim=-1)             # (chunk, K)
        s_logp = F.log_softmax(logits_c, dim=-1)       # (chunk, K)

        total = total + F.kl_div(s_logp, t_probs, reduction="none").sum()

    return total / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LoRA KD training on Qwen2.5-0.5B")
    parser.add_argument("--base-model", required=True, help="Path to HF model directory")
    parser.add_argument("--cache-dir", required=True, help="Path to teacher_logits_cache_*/")
    parser.add_argument("--output", required=True, help="Path to save merged model")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--steps", type=int, default=2000, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Windows per step")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--target-modules", type=str,
                        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                        help="Comma-separated LoRA target modules")
    args = parser.parse_args()

    # --- Load cache metadata ---
    meta_path = os.path.join(args.cache_dir, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    vocab_size = meta["vocab_size"]
    top_k = meta["top_k"]
    n_windows = meta["n_windows"]
    block_size = meta["block_size"]
    n_tokens = n_windows * block_size
    print(f"Cache: {n_windows} windows, {n_tokens} tokens, vocab={vocab_size}, top_k={top_k}")

    # --- Load memmaps (read-only) ---
    top_indices = np.memmap(
        os.path.join(args.cache_dir, "top_indices.bin"),
        dtype=np.int32, mode="r", shape=(n_tokens, top_k))
    top_values = np.memmap(
        os.path.join(args.cache_dir, "top_values.bin"),
        dtype=np.float32, mode="r", shape=(n_tokens, top_k))
    window_tokens = np.memmap(
        os.path.join(args.cache_dir, "window_tokens.bin"),
        dtype=np.int32, mode="r", shape=(n_windows, block_size))

    # --- Load base model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print(f"Loading base model from {args.base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, device_map=device
    )
    model.config.use_cache = False

    # --- Apply LoRA ---
    target_modules = [m.strip() for m in args.target_modules.split(",")]
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Optimizer: AdamW on LoRA params only ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    print(f"Optimizer: AdamW on {len(trainable_params)} param tensors, lr={args.lr}")

    # --- Training loop ---
    ce_weight = 0.3
    kd_weight = 0.7
    print(f"Training {args.steps} steps, batch_size={args.batch_size}, "
          f"lr={args.lr}, CE={ce_weight}, KD={kd_weight}")

    model.train()
    for step in range(1, args.steps + 1):
        cur_lr = cosine_lr(step - 1, args.steps, args.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        # Sample random windows
        wi = np.random.randint(0, n_windows, size=args.batch_size)

        # Load input tokens
        x = torch.from_numpy(window_tokens[wi].copy()).long().to(device)  # (B, T)

        # Load teacher logits targets
        t_idx_chunks = []
        t_val_chunks = []
        for w in wi:
            s = int(w) * block_size
            t_idx_chunks.append(top_indices[s:s + block_size])
            t_val_chunks.append(top_values[s:s + block_size])
        t_indices = torch.from_numpy(np.stack(t_idx_chunks)).long().to(device)
        t_values = torch.from_numpy(np.stack(t_val_chunks).astype(np.float32)).to(device)

 # Forward: full vocab logits (Qwen2.5-0.5B is small enough)
        outputs = model(input_ids=x)
        logits = outputs.logits  # (B, T, V)

        # Shift for next-token prediction: predict token t+1 from position t
        # Use positions 0..T-2 to predict tokens 1..T-1
        shift_logits = logits[:, :-1, :]            # (B, T-1, V)
        shift_targets = x[:, 1:]                     # (B, T-1)
 shift_t_indices = t_indices[:, 1:, :] # (B, T-1, K) - teacher logits for predicting token t+1
        shift_t_values = t_values[:, 1:, :]          # (B, T-1, K)

        # Cross-entropy loss
        ce_loss = F.cross_entropy(
            shift_logits.reshape(-1, vocab_size),
            shift_targets.reshape(-1),
        )

        # KD loss (chunked over top-k positions)
        kd_loss = kd_loss_chunked(shift_logits, shift_t_indices, shift_t_values)

        # Combined loss
        loss = ce_weight * ce_loss + kd_weight * kd_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()

        if step % 100 == 0 or step == 1:
            print(f"  Step {step}/{args.steps}  "
                  f"loss={loss.item():.4f}  "
                  f"CE={ce_loss.item():.4f}  "
                  f"KD={kd_loss.item():.4f}  "
                  f"lr={cur_lr:.2e}")

    # --- Merge LoRA and save ---
    print(f"\nMerging LoRA weights...")
    model = model.merge_and_unload()

    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    print(f"Saved merged model to {args.output}")

    # --- Save tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.save_pretrained(args.output)
    print(f"Saved tokenizer to {args.output}")

    # --- Sanity check generation ---
    print(f"\n--- Sanity check ---")
    model.eval()
    prompt = "Once upon a time"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=50,
            temperature=0.7,
            top_k=40,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Output: {generated}")

    del model, tokenizer
    gc.collect()
    print("\nDone.")


if __name__ == "__main__":
    main()

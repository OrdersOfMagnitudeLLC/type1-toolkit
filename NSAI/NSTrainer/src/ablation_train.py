#!/usr/bin/env python3
"""Full fine-tune Qwen 0.5B on C4 data (standard or NS-filtered) for ablation study."""
import json
import os
import math
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

class C4Dataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_len=512):
        self.samples = []
        with open(jsonl_path) as f:
            for line in f:
                self.samples.append(json.loads(line)["text"])
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.eos_id = tokenizer.eos_token_id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        ids = self.tokenizer.encode(text, add_special_tokens=True,
                                     truncation=True, max_length=self.max_len)
        # Pad to max_len
        attn = [1] * len(ids) + [0] * (self.max_len - len(ids))
        ids = ids + [self.eos_id] * (self.max_len - len(ids))
        return (
            torch.tensor(ids[:self.max_len], dtype=torch.long),
            torch.tensor(attn[:self.max_len], dtype=torch.long),
        )


def cosine_lr(step, total_steps, lr_max, lr_min_ratio=0.1):
    if step >= total_steps:
        return lr_max * lr_min_ratio
    progress = step / total_steps
    return lr_max * lr_min_ratio + 0.5 * (lr_max - lr_max * lr_min_ratio) * (1 + math.cos(math.pi * progress))


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen 0.5B on C4 data")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--data", required=True, help="Path to jsonl data file")
    parser.add_argument("--output", required=True, help="Output checkpoint directory")
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print(f"Loading tokenizer from {args.base_model}...")
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"Loading model from {args.base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, device_map=device
    )
    model.config.use_cache = False
    model.train()

    print(f"Loading dataset from {args.data}...")
    dataset = C4Dataset(args.data, tok, args.max_len)
    print(f"Dataset: {len(dataset)} samples")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        drop_last=True, num_workers=2, pin_memory=True)

    # 8-bit AdamW if available
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, weight_decay=0.01)
        print("Optimizer: 8-bit AdamW (bitsandbytes)")
    except ImportError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        print("Optimizer: standard AdamW")

    print(f"Training {args.steps} steps, batch_size={args.batch_size}, lr={args.lr}, max_len={args.max_len}")

    step = 0
    while step < args.steps:
        for input_ids, attention_mask in loader:
            if step >= args.steps:
                break
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            cur_lr = cosine_lr(step, args.steps, args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr
            optimizer.step()

            step += 1
            if step % 100 == 0 or step == 1:
                print(f"  Step {step}/{args.steps}  loss={loss.item():.4f}  lr={cur_lr:.2e}")

    print(f"\nSaving checkpoint to {args.output}...")
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    tok.save_pretrained(args.output)
    print(f"Saved. Done.")

if __name__ == "__main__":
    main()

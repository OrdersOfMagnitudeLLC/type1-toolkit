#!/usr/bin/env python3
# Orders of Magnitude LLC — OOM Commercial License v1.0
# https://ofmagnitude.com/license
# Copyright 2026 Orders of Magnitude LLC. All rights reserved.

import os
import struct
import torch
from transformers import AutoModelForCausalLM

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")

def main():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    print(f"Loading {MODEL_NAME} ...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    model.eval()

    layer = model.model.layers[0]
    tensors = {
        "gate_proj.bin": layer.mlp.gate_proj.weight,
        "up_proj.bin":   layer.mlp.up_proj.weight,
        "down_proj.bin": layer.mlp.down_proj.weight,
    }

    for fname, tensor in tensors.items():
        t = tensor.detach().cpu().to(torch.float16).contiguous()
        path = os.path.join(WEIGHTS_DIR, fname)
        with open(path, "wb") as f:
            f.write(t.numpy().tobytes())
        size = os.path.getsize(path)
        print(f"  {fname}: shape={tuple(t.shape)}  size={size} bytes ({size/1024/1024:.2f} MB)")

    print("Done.")

if __name__ == "__main__":
    main()

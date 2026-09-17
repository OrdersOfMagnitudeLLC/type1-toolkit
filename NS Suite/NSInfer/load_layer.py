# Copyright (c) 2026 Orders of Magnitude LLC
# Licensed under the OOM Commercial License v1.0
# See LICENSE.md in the repository root or ofmagnitude.com

import safetensors
import numpy as np
from pathlib import Path

WEIGHTS_DIR = Path(__file__).parent / "weights"

def load_layer():
    # Load the TinyLlama model from safetensors
    weights = safetensors.safe_open(WEIGHTS_DIR / 'model.safetensors', framework='pt', device='cpu')
    
    # Extract layer 0 MLP weights (TinyLlama uses SwiGLU)
    gate_proj = weights.get_tensor('model.layers.0.mlp.gate_proj.weight').float().numpy()  # [d_ff, d_model]
    up_proj = weights.get_tensor('model.layers.0.mlp.up_proj.weight').float().numpy()    # [d_ff, d_model]
    down_proj = weights.get_tensor('model.layers.0.mlp.down_proj.weight').float().numpy()  # [d_model, d_ff]
    
    # Transpose to match our expected layout [d_model x d_ff]
    gate_proj = gate_proj.T  # [d_ff, d_model] -> [d_model, d_ff]
    up_proj = up_proj.T      # [d_ff, d_model] -> [d_model, d_ff]
    down_proj = down_proj.T  # [d_model, d_ff] -> [d_ff, d_model]
    
    # Save as raw float32 binary files
    gate_proj.astype(np.float32).tofile(WEIGHTS_DIR / 'gate_proj.bin')
    up_proj.astype(np.float32).tofile(WEIGHTS_DIR / 'up_proj.bin')
    down_proj.astype(np.float32).tofile(WEIGHTS_DIR / 'down_proj.bin')
    
    print(f"gate_proj shape: {gate_proj.shape}")
    print(f"up_proj shape: {up_proj.shape}")
    print(f"down_proj shape: {down_proj.shape}")
    
    # Calculate memory size
    gate_size_mb = gate_proj.nbytes / (1024 * 1024)
    up_size_mb = up_proj.nbytes / (1024 * 1024)
    down_size_mb = down_proj.nbytes / (1024 * 1024)
    total_mb = gate_size_mb + up_size_mb + down_size_mb
    
    print(f"Memory sizes: gate={gate_size_mb:.2f}MB, up={up_size_mb:.2f}MB, down={down_size_mb:.2f}MB")
    print(f"Total: {total_mb:.2f}MB")
    print(f"Weights saved to {WEIGHTS_DIR}/")

if __name__ == '__main__':
    load_layer()

# Copyright (c) 2026 Orders of Magnitude LLC
# Licensed under the OOM Commercial License v1.0
# See LICENSE.md in the repository root or ofmagnitude.com

import numpy as np
from pathlib import Path

WEIGHTS_DIR = Path(__file__).parent / "weights"

def build_predictor():
    # Load gate_proj matrix [d_model x d_ff] = [2048 x 5632]
    gate_proj = np.fromfile(WEIGHTS_DIR / 'gate_proj.bin', dtype=np.float32)
    gate_proj = gate_proj.reshape(2048, 5632)
    
    print(f"gate_proj shape: {gate_proj.shape}")
    
    # Perform SVD once
    U, S, Vt = np.linalg.svd(gate_proj, full_matrices=False)
    
    # Sweep ranks
    ranks = [64, 128, 256, 512]
    
    for rank in ranks:
        U_trunc = U[:, :rank]  # [2048 x rank]
        S_trunc = S[:rank]    # [rank]
        Vt_trunc = Vt[:rank, :]  # [rank x 5632]
        
        # Build A = U_trunc @ diag(S_trunc)  [2048 x rank]
        A = U_trunc * S_trunc
        
        # B = Vt_trunc  [rank x 5632]
        B = Vt_trunc
        
        print(f"Rank {rank}: A shape {A.shape}, size {A.nbytes / (1024*1024):.2f} MB, "
              f"B shape {B.shape}, size {B.nbytes / (1024*1024):.2f} MB")
        
        # Save as float32 binary files
        A.astype(np.float32).tofile(WEIGHTS_DIR / f'A_{rank}.bin')
        B.astype(np.float32).tofile(WEIGHTS_DIR / f'B_{rank}.bin')
        
        # Verify reconstruction quality
        gate_reconstructed = A @ B
        reconstruction_error = np.linalg.norm(gate_proj - gate_reconstructed) / np.linalg.norm(gate_proj)
        print(f"  Relative reconstruction error: {reconstruction_error:.6f}")
    
    print(f"Predictor weights saved to {WEIGHTS_DIR}/")

if __name__ == '__main__':
    build_predictor()

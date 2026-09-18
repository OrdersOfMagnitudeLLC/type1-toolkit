# Copyright (c) 2026 Orders of Magnitude LLC
# Licensed under the OOM Commercial License v1.0
# See LICENSE.md in the repository root or ofmagnitude.com

import numpy as np
from pathlib import Path

WEIGHTS_DIR = Path(__file__).parent / "weights"

def build_clusters():
    # Load gate_proj matrix [d_model x d_ff] = [2048 x 5632]
    gate_proj = np.fromfile(WEIGHTS_DIR / 'gate_proj.bin', dtype=np.float32)
    gate_proj = gate_proj.reshape(2048, 5632)
    
    d_model = 2048
    d_ff = 5632
    n_samples = 2000
    n_train = 1800
    n_heldout = 200
    n_clusters = 256
    percentile = 0.7
    
    print(f"gate_proj shape: {gate_proj.shape}")
    print(f"Generating {n_samples} random inputs...")
    
    # Generate random inputs x ~ N(0, 0.1)
    np.random.seed(42)
    inputs = np.random.randn(n_samples, d_model).astype(np.float32) * 0.1
    
    # Split into training and held-out
    train_inputs = inputs[:n_train]
    heldout_inputs = inputs[n_train:]
    
    # Compute true gate masks for all inputs
    print("Computing gate masks...")
    masks = np.zeros((n_samples, d_ff), dtype=np.uint8)
    
    for i in range(n_samples):
        x = inputs[i]
        gate_out = x @ gate_proj
        gate_out = gate_out / (1 + np.exp(-gate_out))  # SiLU
        abs_gates = np.abs(gate_out)
        threshold = np.percentile(abs_gates, percentile * 100)
        masks[i] = (abs_gates > threshold).astype(np.uint8)
    
    # Simple k-means clustering from scratch (training only)
    print(f"Clustering {n_train} training inputs into {n_clusters} clusters...")
    
    # Initialize centroids randomly from training inputs
    np.random.seed(42)
    indices = np.random.choice(n_train, n_clusters, replace=False)
    centroids = train_inputs[indices].astype(np.float32)
    
    # K-means iterations
    max_iter = 100
    for it in range(max_iter):
        # Assign each training input to nearest centroid
        distances = np.zeros((n_train, n_clusters))
        for k in range(n_clusters):
            diff = train_inputs - centroids[k]
            distances[:, k] = np.sum(diff * diff, axis=1)
        train_labels = np.argmin(distances, axis=1)
        
        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for k in range(n_clusters):
            cluster_indices = np.where(train_labels == k)[0]
            if len(cluster_indices) > 0:
                new_centroids[k] = np.mean(train_inputs[cluster_indices], axis=0)
            else:
                new_centroids[k] = centroids[k]
        
        # Check convergence
        if np.allclose(centroids, new_centroids):
            print(f"Converged at iteration {it}")
            break
        centroids = new_centroids.astype(np.float32)
    
    print(f"Centroids shape: {centroids.shape}")
    
    # Majority-vote masks for each cluster (from training only)
    print("Computing cluster masks...")
    cluster_masks = np.zeros((n_clusters, d_ff), dtype=np.uint8)
    
    for k in range(n_clusters):
        cluster_indices = np.where(train_labels == k)[0]
        cluster_masks[k] = np.mean(masks[cluster_indices], axis=0) > 0.5
    
    # Save centroids
    centroids.tofile(WEIGHTS_DIR / 'cluster_centroids.bin')
    print(f"Saved cluster_centroids.bin: {centroids.nbytes / (1024*1024):.2f} MB")
    
    # Save masks as bit-packed (5632 bits = 704 bytes per cluster)
    n_bytes_per_mask = (d_ff + 7) // 8  # 704 bytes
    packed_masks = np.zeros((n_clusters, n_bytes_per_mask), dtype=np.uint8)
    
    for k in range(n_clusters):
        for byte_idx in range(n_bytes_per_mask):
            byte_val = 0
            for bit_idx in range(8):
                global_bit_idx = byte_idx * 8 + bit_idx
                if global_bit_idx < d_ff and cluster_masks[k, global_bit_idx]:
                    byte_val |= (1 << bit_idx)
            packed_masks[k, byte_idx] = byte_val
    
    packed_masks.tofile(WEIGHTS_DIR / 'cluster_masks.bin')
    print(f"Saved cluster_masks.bin: {packed_masks.nbytes / (1024*1024):.2f} MB")
    
    # Compute mask agreement on held-out set
    print("Computing held-out agreement...")
    total_agreement = 0
    for i in range(n_train, n_samples):
        # Find nearest centroid for held-out input
        distances = np.zeros(n_clusters)
        for k in range(n_clusters):
            diff = inputs[i] - centroids[k]
            distances[k] = np.sum(diff * diff)
        k = np.argmin(distances)
        agreement = np.mean(cluster_masks[k] == masks[i])
        total_agreement += agreement
    
    avg_agreement = total_agreement / n_heldout * 100
    print(f"Held-out agreement: {avg_agreement:.2f}%")
    
    print(f"Cluster data saved to {WEIGHTS_DIR}/")

if __name__ == '__main__':
    build_clusters()

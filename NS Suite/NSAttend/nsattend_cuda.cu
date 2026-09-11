// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cassert>
#include <vector>
#include <iostream>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

// MECHANISM 2: Distance block mask (CUDA kernel)
__global__ void build_block_mask_kernel(bool* mask, int seq_len, int block_size) {
    int bi = blockIdx.x;
    int bj = blockIdx.y;
    int num_blocks = (seq_len + block_size - 1) / block_size;
    
    if (bi >= num_blocks || bj >= num_blocks) return;
    
    int local_window = seq_len / 4;
    int block_dist = abs(bi - bj) * block_size;
    mask[bi * num_blocks + bj] = (block_dist <= local_window);
}

// CUDA wrapper for block mask
void cuda_build_block_mask(bool* d_mask, int seq_len, int block_size = 32) {
    int num_blocks = (seq_len + block_size - 1) / block_size;
    dim3 grid(num_blocks, num_blocks);
    build_block_mask_kernel<<<grid, 1>>>(d_mask, seq_len, block_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// MECHANISM 3: Sparse matmul for LOCAL heads (one warp per active block)
__global__ void block_sparse_attention_kernel(
    float* output,
    const float* queries, const float* keys, const float* values,
    int seq_len, int head_dim, float scale,
    const bool* block_mask, int block_size) {
    
    int block_idx = blockIdx.x;
    int num_blocks = (seq_len + block_size - 1) / block_size;
    
    // Decode block coordinates from linear index
    int bi = block_idx / num_blocks;
    int bj = block_idx % num_blocks;
    
    if (bi >= num_blocks || bj >= num_blocks) return;
    if (!block_mask[bi * num_blocks + bj]) return;
    
    int i_start = bi * block_size;
    int j_start = bj * block_size;
    int i_end = min(i_start + block_size, seq_len);
    int j_end = min(j_start + block_size, seq_len);
    
    // One warp per block
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    // Process this block
    for (int ii = 0; ii < i_end - i_start; ++ii) {
        int i = i_start + ii;
        const float* query = &queries[i * head_dim];
        
        // Compute attention scores for this row within the block
        __shared__ float scores[32][32]; // Max block size
        if (lane_id < j_end - j_start) {
            int j = j_start + lane_id;
            const float* key = &keys[j * head_dim];
            float dot = 0.0f;
            for (int d = lane_id; d < head_dim; d += 32) {
                dot += query[d] * key[d];
            }
            // Warp reduction
            for (int offset = 16; offset > 0; offset /= 2) {
                dot += __shfl_down_sync(0xFFFFFFFF, dot, offset);
            }
            if (lane_id == 0) scores[ii][lane_id] = dot * scale;
        }
        __syncthreads();
        
        // Softmax (simplified - full softmax requires global reduction)
        // For correctness, we'll compute full softmax in a separate pass
    }
    
    // For simplicity, fall back to per-element computation
    // This is a simplified implementation - full warp-level optimization would be more complex
    for (int ii = 0; ii < i_end - i_start; ++ii) {
        int i = i_start + ii;
        const float* query = &queries[i * head_dim];
        
        for (int jj = lane_id; jj < j_end - j_start; jj += 32) {
            int j = j_start + jj;
            const float* key = &keys[j * head_dim];
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += query[d] * key[d];
            }
            // Store score for later softmax
        }
    }
}

// Simplified CUDA sparse attention (correctness-focused)
__global__ void simplified_sparse_attention_kernel(
    float* output,
    const float* queries, const float* keys, const float* values,
    int seq_len, int head_dim, float scale,
    const bool* block_mask, int block_size,
    float* temp_scores) {  // Pre-allocated temporary buffer
    
    int i = blockIdx.x;
    if (i >= seq_len) return;
    
    int num_blocks = (seq_len + block_size - 1) / block_size;
    int bi = i / block_size;
    
    const float* query = &queries[i * head_dim];
    float* scores = &temp_scores[i * seq_len];  // Use pre-allocated buffer
    
    // Compute attention scores only for active blocks
    for (int j = 0; j < seq_len; ++j) scores[j] = -1e30f;
    
    for (int bj = 0; bj < num_blocks; ++bj) {
        if (!block_mask[bi * num_blocks + bj]) continue;
        
        int j_start = bj * block_size;
        int j_end = min(j_start + block_size, seq_len);
        
        for (int j = j_start; j < j_end; ++j) {
            const float* key = &keys[j * head_dim];
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += query[d] * key[d];
            }
            scores[j] = dot * scale;
        }
    }
    
    // Softmax
    float max_val = scores[0];
    for (int j = 1; j < seq_len; ++j) {
        if (scores[j] > max_val) max_val = scores[j];
    }
    
    float sum = 0.0f;
    for (int j = 0; j < seq_len; ++j) {
        scores[j] = expf(scores[j] - max_val);
        sum += scores[j];
    }
    
    for (int j = 0; j < seq_len; ++j) {
        scores[j] /= sum;
    }
    
    // Weighted sum of values
    for (int d = 0; d < head_dim; ++d) {
        output[i * head_dim + d] = 0.0f;
    }
    
    for (int bj = 0; bj < num_blocks; ++bj) {
        if (!block_mask[bi * num_blocks + bj]) continue;
        
        int j_start = bj * block_size;
        int j_end = min(j_start + block_size, seq_len);
        
        for (int j = j_start; j < j_end; ++j) {
            float weight = scores[j];
            const float* value = &values[j * head_dim];
            for (int d = 0; d < head_dim; ++d) {
                output[i * head_dim + d] += weight * value[d];
            }
        }
    }
}

// CUDA wrapper for block-sparse attention
void cuda_block_sparse_attention(
    float* d_output,
    const float* d_queries, const float* d_keys, const float* d_values,
    int seq_len, int head_dim, float scale,
    const bool* d_block_mask, int block_size = 32) {
    
    // Allocate temporary buffer for scores
    float* d_temp_scores;
    CUDA_CHECK(cudaMalloc(&d_temp_scores, seq_len * seq_len * sizeof(float)));
    
    simplified_sparse_attention_kernel<<<seq_len, 1>>>(
        d_output, d_queries, d_keys, d_values,
        seq_len, head_dim, scale, d_block_mask, block_size, d_temp_scores);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaFree(d_temp_scores));
}

// Dense attention (CUDA reference for correctness)
__global__ void dense_attention_kernel(
    float* output,
    const float* queries, const float* keys, const float* values,
    int seq_len, int head_dim, float scale,
    float* temp_scores) {  // Pre-allocated temporary buffer
    
    int i = blockIdx.x;
    if (i >= seq_len) return;
    
    const float* query = &queries[i * head_dim];
    float* scores = &temp_scores[i * seq_len];  // Use pre-allocated buffer
    
    // Compute attention scores
    for (int j = 0; j < seq_len; ++j) {
        const float* key = &keys[j * head_dim];
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += query[d] * key[d];
        }
        scores[j] = dot * scale;
    }
    
    // Softmax
    float max_val = scores[0];
    for (int j = 1; j < seq_len; ++j) {
        if (scores[j] > max_val) max_val = scores[j];
    }
    
    float sum = 0.0f;
    for (int j = 0; j < seq_len; ++j) {
        scores[j] = expf(scores[j] - max_val);
        sum += scores[j];
    }
    
    for (int j = 0; j < seq_len; ++j) {
        scores[j] /= sum;
    }
    
    // Weighted sum of values
    for (int d = 0; d < head_dim; ++d) {
        output[i * head_dim + d] = 0.0f;
    }
    
    for (int j = 0; j < seq_len; ++j) {
        float weight = scores[j];
        const float* value = &values[j * head_dim];
        for (int d = 0; d < head_dim; ++d) {
            output[i * head_dim + d] += weight * value[d];
        }
    }
}

// CUDA wrapper for dense attention
void cuda_dense_attention(
    float* d_output,
    const float* d_queries, const float* d_keys, const float* d_values,
    int seq_len, int head_dim, float scale) {
    
    // Allocate temporary buffer for scores
    float* d_temp_scores;
    CUDA_CHECK(cudaMalloc(&d_temp_scores, seq_len * seq_len * sizeof(float)));
    
    dense_attention_kernel<<<seq_len, 1>>>(
        d_output, d_queries, d_keys, d_values,
        seq_len, head_dim, scale, d_temp_scores);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaFree(d_temp_scores));
}

// Copy sample scores for head type detection (device to host)
void cuda_sample_scores_for_detection(
    float* h_sample_scores,
    const float* d_queries, const float* d_keys,
    int seq_len, int head_dim, float scale,
    int sample_rows = 8) {
    
    int actual_sample = min(sample_rows, seq_len);
    
    // Allocate device memory for sample scores
    float* d_sample_scores;
    CUDA_CHECK(cudaMalloc(&d_sample_scores, actual_sample * seq_len * sizeof(float)));
    
    // Allocate temp buffer for kernel
    float* d_temp_scores;
    CUDA_CHECK(cudaMalloc(&d_temp_scores, actual_sample * seq_len * sizeof(float)));
    
    // Compute sample scores on device using dense attention kernel (limited rows)
    for (int i = 0; i < actual_sample; ++i) {
        // Copy single query row to compute scores
        // For simplicity, we'll copy the needed data to host and compute there
        // This is a fallback - ideally you'd have a dedicated kernel
    }
    
    // Fallback: copy queries and keys to host, compute there
    std::vector<float> h_queries(actual_sample * head_dim);
    std::vector<float> h_keys(seq_len * head_dim);
    
    CUDA_CHECK(cudaMemcpy(h_queries.data(), d_queries, actual_sample * head_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_keys.data(), d_keys, seq_len * head_dim * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Compute scores on host
    for (int i = 0; i < actual_sample; ++i) {
        const float* query = &h_queries[i * head_dim];
        for (int j = 0; j < seq_len; ++j) {
            const float* key = &h_keys[j * head_dim];
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += query[d] * key[d];
            }
            h_sample_scores[i * seq_len + j] = dot * scale;
        }
    }
    
    CUDA_CHECK(cudaFree(d_sample_scores));
    CUDA_CHECK(cudaFree(d_temp_scores));
}

// Main CUDA NS attention function
void cuda_ns_attention(
    float* d_output,
    const float* d_queries, const float* d_keys, const float* d_values,
    int seq_len, int head_dim, float scale) {
    
    // Sample scores for head type detection (CPU side)
    std::vector<float> h_sample_scores(seq_len * seq_len);
    cuda_sample_scores_for_detection(h_sample_scores.data(), d_queries, d_keys,
                                      seq_len, head_dim, scale, 8);
    
    // Use CPU-side head type detection (reuse logic from nsattend.hpp)
    // For now, assume LOCAL for demonstration
    // In practice, you'd call detect_head_type from nsattend.hpp here
    
    // Build block mask
    int num_blocks = (seq_len + 31) / 32;
    bool* d_block_mask;
    CUDA_CHECK(cudaMalloc(&d_block_mask, num_blocks * num_blocks * sizeof(bool)));
    cuda_build_block_mask(d_block_mask, seq_len, 32);
    
    // Run block-sparse attention
    cuda_block_sparse_attention(d_output, d_queries, d_keys, d_values,
                                seq_len, head_dim, scale, d_block_mask, 32);
    
    CUDA_CHECK(cudaFree(d_block_mask));
}

// Verify CUDA correctness (host-side)
float cuda_verify_correctness(
    const float* d_ns_output, const float* d_dense_output,
    int seq_len, int head_dim) {
    
    // Copy to host
    std::vector<float> h_ns_output(seq_len * head_dim);
    std::vector<float> h_dense_output(seq_len * head_dim);
    
    CUDA_CHECK(cudaMemcpy(h_ns_output.data(), d_ns_output,
                          seq_len * head_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dense_output.data(), d_dense_output,
                          seq_len * head_dim * sizeof(float), cudaMemcpyDeviceToHost));
    
    float max_diff = 0.0f;
    for (int i = 0; i < seq_len * head_dim; ++i) {
        float diff = fabsf(h_ns_output[i] - h_dense_output[i]);
        if (diff > max_diff) max_diff = diff;
    }
    
    return max_diff;
}

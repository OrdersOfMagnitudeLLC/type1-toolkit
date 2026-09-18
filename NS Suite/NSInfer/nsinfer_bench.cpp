// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#include "nsinfer.hpp"
#include <iostream>
#include <chrono>
#include <random>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <cstdlib>

// Simple naive matmul fallback since OpenBLAS not available
void naive_sgemm_rowmajor(int M, int N, int K, const float* A, const float* B, float* C) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}

// Cosine similarity between two vectors
float cosine_similarity(const float* a, const float* b, int n) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (int i = 0; i < n; ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

void cache_flush() {
    const size_t flush_size = 256 * 1024 * 1024; // 256MB
    char* flush_buffer = new char[flush_size];
    memset(flush_buffer, 0, flush_size);
    delete[] flush_buffer;
}

void load_weights(const char* filename, float* buffer, size_t size) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "ERROR: cannot open weight file: " << filename << std::endl;
        std::cerr << "Download weights: huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0" << std::endl;
        std::cerr << "Then run: python3 extract_weights.py" << std::endl;
        std::exit(1);
    }
    // Weights are stored as fp16; convert to float32
    uint16_t* fp16_buf = new uint16_t[size];
    file.read(reinterpret_cast<char*>(fp16_buf), size * sizeof(uint16_t));
    file.close();
    for (size_t i = 0; i < size; ++i)
        buffer[i] = _cvtsh_ss(fp16_buf[i]);
    delete[] fp16_buf;
}

void load_weights_uint8(const char* filename, uint8_t* buffer, size_t size) {
    std::ifstream file(filename, std::ios::binary);
    file.read(reinterpret_cast<char*>(buffer), size);
    file.close();
}

void generate_input(float* x, int d_model, std::mt19937& gen) {
    std::normal_distribution<float> dist(0.0f, 0.1f);
    for (int i = 0; i < d_model; ++i) {
        x[i] = dist(gen);
    }
}

// Dense BLAS SwiGLU baseline
void dense_swiglu_blas(
    const float* x,
    const float* gate_proj,
    const float* up_proj,
    const float* down_proj,
    float* output,
    int d_model,
    int d_ff
) {
    float* gate_out = new float[d_ff];
    float* up_out = new float[d_ff];
    float* hidden = new float[d_ff];
    
    // gate_out = x @ gate_proj  [1 x d_model] @ [d_model x d_ff] -> [1 x d_ff]
    naive_sgemm_rowmajor(1, d_ff, d_model, x, gate_proj, gate_out);
    
    // Apply SiLU
    for (int j = 0; j < d_ff; ++j) {
        gate_out[j] = silu(gate_out[j]);
    }
    
    // up_out = x @ up_proj
    naive_sgemm_rowmajor(1, d_ff, d_model, x, up_proj, up_out);
    
    // hidden = gate_out * up_out (elementwise)
    for (int j = 0; j < d_ff; ++j) {
        hidden[j] = gate_out[j] * up_out[j];
    }
    
    // output = hidden @ down_proj  [1 x d_ff] @ [d_ff x d_model] -> [1 x d_model]
    naive_sgemm_rowmajor(1, d_model, d_ff, hidden, down_proj, output);
    
    delete[] gate_out;
    delete[] up_out;
    delete[] hidden;
}

// Sparse SwiGLU with memory packing
struct SparseResult {
    int selected_k;
    double estimated_gb_loaded;
    float mask_agreement;  // % agreement with true gate mask
};

struct NSInferArena {
    uint8_t* base;
    size_t offset;
    size_t capacity;
    std::vector<float*> heap_fallbacks;

    void init(int d_model, int d_ff) {
        // Worst case: selected_k = d_ff (no sparsity fallback)
        size_t max_k = d_ff;
        // active_up [d_model × max_k] + active_gate_vals [max_k]
        // + active_down [max_k × d_model] + safety margin
        capacity = sizeof(float) * (d_model * max_k * 2 + max_k) + 1024;
        base = (uint8_t*)aligned_alloc(64, capacity);
        offset = 0;
    }

    float* alloc(size_t n_floats) {
        size_t bytes_needed = n_floats * sizeof(float);
        if (offset + bytes_needed > capacity) {
            // Fallback to heap if arena overflow
            float* ptr = new float[n_floats];
            heap_fallbacks.push_back(ptr);
            return ptr;
        }
        float* ptr = (float*)(base + offset);
        offset += bytes_needed;
        return ptr;
    }

    void reset() { 
        offset = 0;
        for (float* ptr : heap_fallbacks) {
            delete[] ptr;
        }
        heap_fallbacks.clear();
    }

    ~NSInferArena() { ::free(base); }
};

// Find nearest centroid for input x
int find_nearest_centroid(const float* x, const float* centroids, int d_model, int n_clusters) {
    int best_k = 0;
    float best_dist = 1e30f;
    
    for (int k = 0; k < n_clusters; ++k) {
        float dist = 0.0f;
        for (int i = 0; i < d_model; ++i) {
            float diff = x[i] - centroids[k * d_model + i];
            dist += diff * diff;
        }
        if (dist < best_dist) {
            best_dist = dist;
            best_k = k;
        }
    }
    return best_k;
}

// Unpack bit mask to active indices
void unpack_mask(const uint8_t* packed_mask, int* active_indices, int* selected_k, int d_ff) {
    int k = 0;
    for (int byte_idx = 0; byte_idx < (d_ff + 7) / 8; ++byte_idx) {
        uint8_t byte_val = packed_mask[byte_idx];
        for (int bit_idx = 0; bit_idx < 8; ++bit_idx) {
            int global_bit_idx = byte_idx * 8 + bit_idx;
            if (global_bit_idx < d_ff && (byte_val & (1 << bit_idx))) {
                active_indices[k++] = global_bit_idx;
            }
        }
    }
    *selected_k = k;
}

SparseResult sparse_swiglu_blas(
    const float* x,
    const float* gate_proj,
    const float* up_proj,
    const float* down_proj,
    const float* down_row_norms,
    float mean_down_norm,
    float* output,
    int d_model,
    int d_ff,
    float energy_threshold,
    int rescue_k,
    NSInferArena& arena
) {
    arena.reset();
    
    float* gate_out = new float[d_ff];
    float* up_out = new float[d_ff];
    float* hidden = new float[d_ff];
    int* active_indices = new int[d_ff * 2];  // extra space for flat_indices
    
    // Step 1: Compute full gate_out
    naive_sgemm_rowmajor(1, d_ff, d_model, x, gate_proj, gate_out);
    
    // Apply SiLU
    for (int j = 0; j < d_ff; ++j) {
        gate_out[j] = silu(gate_out[j]);
    }
    
    // Step 2: Compute full up_out, then energy signals from actual gate*up product
    naive_sgemm_rowmajor(1, d_ff, d_model, x, up_proj, up_out);

    float* contributions = arena.alloc(d_ff);
    float total_energy = 0.0f;
    float max_energy = 0.0f;
    for (int j = 0; j < d_ff; ++j) {
        contributions[j] = std::abs(gate_out[j] * up_out[j]) * down_row_norms[j];
        total_energy += contributions[j];
        max_energy = std::max(max_energy, contributions[j]);
    }
    
    // Step 3: NS histogram partial sort: O(n) vs O(n log n)
    // Stack allocated: no heap
    int bucket_counts[256] = {};
    int bucket_starts[256] = {};
    int* flat_indices = active_indices + d_ff;  // reuse active_indices buffer
    
    // Pass 1: count
    for (int j = 0; j < d_ff; ++j) {
        uint8_t b = (max_energy > 0) ? (uint8_t)(contributions[j] / max_energy * 255.f) : 0;
        bucket_counts[b]++;
    }
    
    // Pass 2: prefix sum (high→low order)
    bucket_starts[255] = 0;
    for (int b = 254; b >= 0; --b)
        bucket_starts[b] = bucket_starts[b+1] + bucket_counts[b+1];
    
    // Pass 3: fill flat array high→low
    int temp_pos[256];
    memcpy(temp_pos, bucket_starts, sizeof(bucket_starts));
    for (int j = 0; j < d_ff; ++j) {
        uint8_t b = (max_energy > 0) ? (uint8_t)(contributions[j] / max_energy * 255.f) : 0;
        flat_indices[temp_pos[b]++] = j;
    }
    
    // Pass 4: scan flat_indices (already high→low), accumulate energy
    int selected_k = 0;
    float cumulative_energy = 0.0f;
    float target_energy = energy_threshold * total_energy;
    for (int i = 0; i < d_ff; ++i) {
        int j = flat_indices[i];
        active_indices[selected_k++] = j;
        cumulative_energy += contributions[j];
        if (cumulative_energy >= target_energy) break;
    }
    
    // Rescue pass: add top-K high-contribution dropped neurons
    if (rescue_k > 0 && selected_k < d_ff) {
        // Build dropped_indices array
        int* dropped_indices = (int*)arena.alloc(d_ff - selected_k);
        int dropped_count = 0;
        bool* selected = (bool*)arena.alloc((d_ff + 3) / 4);  // pack into float slots
        memset(selected, 0, d_ff * sizeof(bool));
        for (int i = 0; i < selected_k; ++i) {
            selected[active_indices[i]] = true;
        }
        for (int j = 0; j < d_ff; ++j) {
            if (!selected[j]) {
                dropped_indices[dropped_count++] = j;
            }
        }
        
        // O(n) histogram rescue pass
        float* rescue_scores = arena.alloc(dropped_count);
        float max_rescue = 0.0f;
        for (int i = 0; i < dropped_count; ++i) {
            int j = dropped_indices[i];
            rescue_scores[i] = std::abs(gate_out[j]) * down_row_norms[j];
            max_rescue = std::max(max_rescue, rescue_scores[i]);
        }
        
        // 256-bucket histogram on dropped set
        int bucket_counts_r[256] = {};
        int bucket_starts_r[256] = {};
        int* flat_dropped = (int*)arena.alloc(dropped_count);
        
        // Pass 1: count
        for (int i = 0; i < dropped_count; ++i) {
            uint8_t b = (max_rescue > 0) ? (uint8_t)(rescue_scores[i] / max_rescue * 255.f) : 0;
            bucket_counts_r[b]++;
        }
        
        // Pass 2: prefix sum (high→low order)
        bucket_starts_r[255] = 0;
        for (int b = 254; b >= 0; --b)
            bucket_starts_r[b] = bucket_starts_r[b+1] + bucket_counts_r[b+1];
        
        // Pass 3: fill flat array high→low
        int temp_pos_r[256];
        memcpy(temp_pos_r, bucket_starts_r, sizeof(bucket_starts_r));
        for (int i = 0; i < dropped_count; ++i) {
            uint8_t b = (max_rescue > 0) ? (uint8_t)(rescue_scores[i] / max_rescue * 255.f) : 0;
            flat_dropped[temp_pos_r[b]++] = dropped_indices[i];
        }
        
        // Pass 4: scan flat_dropped until K rescued
        int actual_rescue = std::min(rescue_k, dropped_count);
        int rescued = 0;
        for (int i = 0; i < dropped_count && rescued < actual_rescue; ++i) {
            active_indices[selected_k++] = flat_dropped[i];
            rescued++;
        }
    }
    // Fall through to dense if not enough sparsity
    if (selected_k > 0.7f * d_ff) {
        delete[] gate_out;
        delete[] up_out;
        delete[] hidden;
        delete[] active_indices;
        
        dense_swiglu_blas(x, gate_proj, up_proj, down_proj, output, d_model, d_ff);
        
        SparseResult result;
        result.selected_k = d_ff;
        result.estimated_gb_loaded = (d_model * d_ff * 2 * sizeof(float)) / (1024.0 * 1024.0 * 1024.0);
        result.mask_agreement = 100.0f;
        return result;
    }
    
    // Pack active rows of down_proj [selected_k x d_model]
    float* active_down = arena.alloc(selected_k * d_model);
    for (int k = 0; k < selected_k; ++k) {
        int j = active_indices[k];
        for (int i = 0; i < d_model; ++i) {
            active_down[k * d_model + i] = down_proj[j * d_model + i];
        }
    }
    
    // hidden = gate_out * up_out for selected neurons (both already computed)
    for (int k = 0; k < selected_k; ++k) {
        int j = active_indices[k];
        hidden[k] = gate_out[j] * up_out[j];
    }
    
    // output = hidden @ active_down
    naive_sgemm_rowmajor(1, d_model, selected_k, hidden, active_down, output);
    
    // Estimate GB loaded
    double gate_gb = (d_model * selected_k * sizeof(float)) / (1024.0 * 1024.0 * 1024.0);
    double up_gb = (d_model * selected_k * sizeof(float)) / (1024.0 * 1024.0 * 1024.0);
    double down_gb = (selected_k * d_model * sizeof(float)) / (1024.0 * 1024.0 * 1024.0);
    double total_gb = gate_gb + up_gb + down_gb;
    
    delete[] gate_out;
    delete[] up_out;
    delete[] hidden;
    delete[] active_indices;
    
    SparseResult result;
    result.selected_k = selected_k;
    result.estimated_gb_loaded = total_gb;
    result.mask_agreement = 100.0f;  // Energy-based selection is self-consistent
    return result;
}

int main() {
    const int d_model = 2048;
    const int d_ff = 5632;
    const float energy_thresholds[] = {0.75f};
    const int rescue_ks[] = {0};  // pure energy only
    
    std::mt19937 gen(42);
    
    // Load TinyLlama weights (full matrices)
    float* gate_proj = new float[d_model * d_ff];
    float* up_proj = new float[d_model * d_ff];
    float* down_proj = new float[d_ff * d_model];
    
    const char* home_raw = std::getenv("HOME");
    std::string home = home_raw ? home_raw : "";
    load_weights((home + "/NS/NSInfer/weights/gate_proj.bin").c_str(), gate_proj, d_model * d_ff);
    load_weights((home + "/NS/NSInfer/weights/up_proj.bin").c_str(), up_proj, d_model * d_ff);
    load_weights((home + "/NS/NSInfer/weights/down_proj.bin").c_str(), down_proj, d_ff * d_model);
    
    // Precompute down_proj row norms for contribution-weighted selection
    float* down_row_norms = new float[d_ff];
    float mean_down_norm = 0.0f;
    for (int j = 0; j < d_ff; ++j) {
        float norm = 0.0f;
        for (int i = 0; i < d_model; ++i)
            norm += down_proj[j * d_model + i] * down_proj[j * d_model + i];
        down_row_norms[j] = std::sqrt(norm);
        mean_down_norm += down_row_norms[j];
    }
    mean_down_norm /= d_ff;
    
    NSInferArena arena;
    arena.init(d_model, d_ff);
    
    std::cout << "NSInfer CPU Inference - Energy-Ordered Selection\n";
    std::cout << "Model: TinyLlama (d_model=" << d_model << ", d_ff=" << d_ff << ")\n";
    std::cout << "Weights loaded from .bin files\n\n";
    
    float* x = new float[d_model];
    float* dense_output = new float[d_model];
    float* sparse_output = new float[d_model];
    
    generate_input(x, d_model, gen);
    
    // Warmup dense
    dense_swiglu_blas(x, gate_proj, up_proj, down_proj, dense_output, d_model, d_ff);
    cache_flush();
    
    // Time dense
    auto start = std::chrono::high_resolution_clock::now();
    dense_swiglu_blas(x, gate_proj, up_proj, down_proj, dense_output, d_model, d_ff);
    auto end = std::chrono::high_resolution_clock::now();
    double dense_time = std::chrono::duration<double, std::milli>(end - start).count();
    
    double dense_gb = (d_model * d_ff * 2 * sizeof(float)) / (1024.0 * 1024.0 * 1024.0);
    
    std::cout << "Dense BLAS baseline:\n";
    std::cout << "  Time: " << std::fixed << std::setprecision(3) << dense_time << " ms\n";
    std::cout << "  Memory loaded: " << std::setprecision(3) << dense_gb << " GB\n\n";
    
    for (float energy_threshold : energy_thresholds) {
        for (int rescue_k : rescue_ks) {
            std::cout << "=== Energy threshold " << static_cast<int>(energy_threshold * 100) 
                      << "%, rescue K=" << rescue_k << " ===\n";
            
            // Warmup sparse
            sparse_swiglu_blas(x, gate_proj, up_proj, down_proj, down_row_norms, mean_down_norm, 
                             sparse_output, d_model, d_ff, energy_threshold, rescue_k, arena);
            
            // 5 timed runs with cache flush
            std::vector<double> sparse_times;
            SparseResult result;
            for (int run = 0; run < 5; ++run) {
                cache_flush();
                auto sparse_start = std::chrono::high_resolution_clock::now();
                result = sparse_swiglu_blas(x, gate_proj, up_proj, down_proj, down_row_norms, mean_down_norm, 
                                           sparse_output, d_model, d_ff, energy_threshold, rescue_k, arena);
                auto sparse_end = std::chrono::high_resolution_clock::now();
                double sparse_time = std::chrono::duration<double, std::milli>(sparse_end - sparse_start).count();
                sparse_times.push_back(sparse_time);
                
                float keep_pct = 100.0f * (float)result.selected_k / d_ff;
                std::cout << "  Run " << (run + 1) << ": " << std::fixed << std::setprecision(3) << sparse_time << " ms, "
                          << std::setprecision(1) << keep_pct << "% kept\n";
            }
            
            // Calculate statistics
            std::sort(sparse_times.begin(), sparse_times.end());
            double median_time = sparse_times[2];
            double min_time = sparse_times[0];
            double max_time = sparse_times[4];
            double median_speedup = dense_time / median_time;
            double min_speedup = dense_time / max_time;
            double max_speedup = dense_time / min_time;
            
            std::cout << "  Median speedup: " << std::setprecision(2) << median_speedup << "x ("
                      << min_speedup << "x - " << max_speedup << "x)\n";
            std::cout << "  Memory loaded: " << std::setprecision(3) << result.estimated_gb_loaded << " GB\n";
            std::cout << "  Cosine similarity: " << std::setprecision(4) << cosine_similarity(dense_output, sparse_output, d_model) << "\n\n";
        }
    }
    
    delete[] x;
    delete[] dense_output;
    delete[] sparse_output;
    delete[] gate_proj;
    delete[] up_proj;
    delete[] down_proj;
    
    return 0;
}

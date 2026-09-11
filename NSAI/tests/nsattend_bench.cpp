#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include "../include/nsattend.hpp"

const int HEAD_DIM = 64;
const int BLOCK_SIZE = 32;

// Generate LOCAL head data (exponential decay with distance)
void generate_local_data(float* queries, float* keys, float* values, int seq_len, int head_dim) {
    std::mt19937 gen(42);
    std::normal_distribution<float> noise(0.0f, 0.1f);
    
    for (int i = 0; i < seq_len; ++i) {
        for (int d = 0; d < head_dim; ++d) {
            queries[i * head_dim + d] = noise(gen);
            keys[i * head_dim + d] = noise(gen);
            values[i * head_dim + d] = noise(gen);
        }
        
        // Add distance-based structure to keys
        for (int j = 0; j < seq_len; ++j) {
            int distance = std::abs(i - j);
            float decay = std::exp(-distance / 64.0f);
            for (int d = 0; d < head_dim; ++d) {
                keys[j * head_dim + d] += decay;
            }
        }
    }
}

// Generate GLOBAL head data (8 dominant tokens, 60% weight)
void generate_global_data(float* queries, float* keys, float* values, int seq_len, int head_dim) {
    std::mt19937 gen(42);
    std::normal_distribution<float> noise(0.0f, 0.1f);
    
    for (int i = 0; i < seq_len; ++i) {
        for (int d = 0; d < head_dim; ++d) {
            queries[i * head_dim + d] = noise(gen);
            keys[i * head_dim + d] = noise(gen);
            values[i * head_dim + d] = noise(gen);
        }
    }
    
    // Boost 8 dominant tokens in keys to simulate global attention
    std::vector<int> dominant = {0, seq_len/8, seq_len/4, 3*seq_len/8, seq_len/2, 5*seq_len/8, 3*seq_len/4, 7*seq_len/8};
    for (int idx : dominant) {
        for (int d = 0; d < head_dim; ++d) {
            keys[idx * head_dim + d] += 2.0f;
        }
    }
}

// Generate UNIFORM head data (random noise)
void generate_uniform_data(float* queries, float* keys, float* values, int seq_len, int head_dim) {
    std::mt19937 gen(42);
    std::normal_distribution<float> noise(0.0f, 1.0f);
    
    for (int i = 0; i < seq_len; ++i) {
        for (int d = 0; d < head_dim; ++d) {
            queries[i * head_dim + d] = noise(gen);
            keys[i * head_dim + d] = noise(gen);
            values[i * head_dim + d] = noise(gen);
        }
    }
}

// Full dense attention (standard O(N²), no window restriction)
void full_dense_attention(float* output,
                         const float* queries, const float* keys, const float* values,
                         int seq_len, int head_dim, float scale) {
    assert(seq_len <= 8192 && "seq_len must be <= 8192");
    
    for (int i = 0; i < seq_len; ++i) {
        const float* query = &queries[i * head_dim];
        
        // Compute attention scores for ALL positions
        float scores[8192];
        for (int j = 0; j < seq_len; ++j) {
            const float* key = &keys[j * head_dim];
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += query[d] * key[d];
            }
            scores[j] = dot * scale;
        }
        
        // Softmax
        softmax_row(scores, seq_len);
        
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
}

// Local-window dense attention (honest baseline - same window as NSAttend)
void local_window_dense_attention(float* output,
                                 const float* queries, const float* keys, const float* values,
                                 int seq_len, int head_dim, float scale) {
    assert(seq_len <= 8192 && "seq_len must be <= 8192");
    int local_window = seq_len / 4;
    
    for (int i = 0; i < seq_len; ++i) {
        const float* query = &queries[i * head_dim];
        
        // Compute attention scores only within local window
        float scores[8192];
        std::fill(scores, scores + seq_len, -1e30f);
        
        int j_start = std::max(0, i - local_window);
        int j_end = std::min(seq_len, i + local_window + 1);
        
        for (int j = j_start; j < j_end; ++j) {
            const float* key = &keys[j * head_dim];
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += query[d] * key[d];
            }
            scores[j] = dot * scale;
        }
        
        // Softmax over all positions (inactive get 0 weight due to -inf)
        softmax_row(scores, seq_len);
        
        // Weighted sum of values (only active positions contribute)
        for (int d = 0; d < head_dim; ++d) {
            output[i * head_dim + d] = 0.0f;
        }
        
        for (int j = j_start; j < j_end; ++j) {
            float weight = scores[j];
            const float* value = &values[j * head_dim];
            for (int d = 0; d < head_dim; ++d) {
                output[i * head_dim + d] += weight * value[d];
            }
        }
    }
}

// Cache-friendly blocked dense attention (32x32 tiles)
void blocked_dense_attention(float* output,
                            const float* queries, const float* keys, const float* values,
                            int seq_len, int head_dim, float scale) {
    // Fallback to local-window dense for honest baseline
    local_window_dense_attention(output, queries, keys, values, seq_len, head_dim, scale);
}

// Verify correctness
float verify_correctness(const float* ns_output, const float* dense_output, int seq_len, int head_dim) {
    float max_diff = 0.0f;
    for (int i = 0; i < seq_len * head_dim; ++i) {
        float diff = std::abs(ns_output[i] - dense_output[i]);
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}

// Benchmark single scenario
void benchmark_scenario(const std::string& name,
                        void (*gen_func)(float*, float*, float*, int, int),
                        int seq_len, int head_dim) {
    // Skip UNIFORM scenario (fallback path, not a win condition)
    if (name == "UNIFORM") {
        std::cout << "=== UNIFORM (seq_len=" << seq_len << ") ===" << std::endl;
        std::cout << "UNIFORM: fallback path, not benchmarked" << std::endl;
        std::cout << std::endl;
        return;
    }
    
    std::cout << "=== " << name << " (seq_len=" << seq_len << ") ===" << std::endl;
    
    // Allocate data
    std::vector<float> queries(seq_len * head_dim);
    std::vector<float> keys(seq_len * head_dim);
    std::vector<float> values(seq_len * head_dim);
    std::vector<float> ns_output(seq_len * head_dim);
    std::vector<float> local_window_output(seq_len * head_dim);
    std::vector<float> full_dense_output(seq_len * head_dim);
    
    // Generate data
    gen_func(queries.data(), keys.data(), values.data(), seq_len, head_dim);
    
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    
    // Warmup iterations
    int warmup_iters = 1;
    for (int w = 0; w < warmup_iters; ++w) {
        ns_attention(ns_output.data(), queries.data(), keys.data(), values.data(),
                     seq_len, head_dim, scale);
        local_window_dense_attention(local_window_output.data(), queries.data(), keys.data(), values.data(),
                                     seq_len, head_dim, scale);
        full_dense_attention(full_dense_output.data(), queries.data(), keys.data(), values.data(),
                            seq_len, head_dim, scale);
    }
    
    // Timed iterations
    int timing_iters = 3;
    
    auto start_ns = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < timing_iters; ++t) {
        ns_attention(ns_output.data(), queries.data(), keys.data(), values.data(),
                     seq_len, head_dim, scale);
    }
    auto end_ns = std::chrono::high_resolution_clock::now();
    double ns_time = std::chrono::duration<double, std::milli>(end_ns - start_ns).count() / timing_iters;
    
    auto start_local = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < timing_iters; ++t) {
        local_window_dense_attention(local_window_output.data(), queries.data(), keys.data(), values.data(),
                                     seq_len, head_dim, scale);
    }
    auto end_local = std::chrono::high_resolution_clock::now();
    double local_time = std::chrono::duration<double, std::milli>(end_local - start_local).count() / timing_iters;
    
    auto start_full = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < timing_iters; ++t) {
        full_dense_attention(full_dense_output.data(), queries.data(), keys.data(), values.data(),
                            seq_len, head_dim, scale);
    }
    auto end_full = std::chrono::high_resolution_clock::now();
    double full_time = std::chrono::duration<double, std::milli>(end_full - start_full).count() / timing_iters;
    
    // Correctness check against local-window baseline
    float max_diff = verify_correctness(ns_output.data(), local_window_output.data(), seq_len, head_dim);
    
    std::cout << "NS time (ms): " << std::fixed << std::setprecision(6) << ns_time << std::endl;
    std::cout << "Local-window dense time (ms): " << std::fixed << std::setprecision(6) << local_time << std::endl;
    std::cout << "Full dense time (ms): " << std::fixed << std::setprecision(6) << full_time << std::endl;
    std::cout << "Max diff vs local-window: " << std::scientific << max_diff << std::endl;
    std::cout << "Correct: " << (max_diff < 1e-3 ? "YES" : "NO") << std::endl;
    std::cout << std::endl;
}

// Benchmark UNIFORM regression (fallback path overhead)
void benchmark_uniform_regression(int seq_len, int head_dim) {
    std::cout << "=== UNIFORM REGRESSION (seq_len=" << seq_len << ") ===" << std::endl;
    
    // Allocate data
    std::vector<float> queries(seq_len * head_dim);
    std::vector<float> keys(seq_len * head_dim);
    std::vector<float> values(seq_len * head_dim);
    std::vector<float> ns_output(seq_len * head_dim);
    std::vector<float> full_dense_output(seq_len * head_dim);
    
    // Generate uniform data
    generate_uniform_data(queries.data(), keys.data(), values.data(), seq_len, head_dim);
    
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    
    // Warmup iterations
    int warmup_iters = 1;
    for (int w = 0; w < warmup_iters; ++w) {
        ns_attention(ns_output.data(), queries.data(), keys.data(), values.data(),
                     seq_len, head_dim, scale);
        full_dense_attention(full_dense_output.data(), queries.data(), keys.data(), values.data(),
                            seq_len, head_dim, scale);
    }
    
    // Timed iterations
    int timing_iters = 3;
    
    auto start_ns = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < timing_iters; ++t) {
        ns_attention(ns_output.data(), queries.data(), keys.data(), values.data(),
                     seq_len, head_dim, scale);
    }
    auto end_ns = std::chrono::high_resolution_clock::now();
    double ns_time = std::chrono::duration<double, std::milli>(end_ns - start_ns).count() / timing_iters;
    
    auto start_full = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < timing_iters; ++t) {
        full_dense_attention(full_dense_output.data(), queries.data(), keys.data(), values.data(),
                            seq_len, head_dim, scale);
    }
    auto end_full = std::chrono::high_resolution_clock::now();
    double full_time = std::chrono::duration<double, std::milli>(end_full - start_full).count() / timing_iters;
    
    double overhead_pct = ((ns_time - full_time) / full_time) * 100.0;
    
    float max_diff = verify_correctness(ns_output.data(), full_dense_output.data(), seq_len, head_dim);
    
    std::cout << "NS time (ms): " << std::fixed << std::setprecision(6) << ns_time << std::endl;
    std::cout << "Full dense time (ms): " << std::fixed << std::setprecision(6) << full_time << std::endl;
    std::cout << "Overhead: " << std::fixed << std::setprecision(2) << overhead_pct << "%" << std::endl;
    std::cout << "Max diff vs full dense: " << std::scientific << max_diff << std::endl;
    std::cout << std::endl;
}

// Benchmark GLOBAL head scenario
void benchmark_scenario_global(int seq_len, int head_dim) {
    std::cout << "=== GLOBAL (seq_len=" << seq_len << ") ===" << std::endl;
    
    // Allocate data
    std::vector<float> queries(seq_len * head_dim);
    std::vector<float> keys(seq_len * head_dim);
    std::vector<float> values(seq_len * head_dim);
    std::vector<float> ns_output(seq_len * head_dim);
    std::vector<float> full_dense_output(seq_len * head_dim);
    
    // Generate global data
    generate_global_data(queries.data(), keys.data(), values.data(), seq_len, head_dim);
    
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    
    // Warmup iterations
    int warmup_iters = 1;
    for (int w = 0; w < warmup_iters; ++w) {
        ns_attention(ns_output.data(), queries.data(), keys.data(), values.data(),
                     seq_len, head_dim, scale);
        full_dense_attention(full_dense_output.data(), queries.data(), keys.data(), values.data(),
                            seq_len, head_dim, scale);
    }
    
    // Timed iterations
    int timing_iters = 3;
    
    auto start_ns = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < timing_iters; ++t) {
        ns_attention(ns_output.data(), queries.data(), keys.data(), values.data(),
                     seq_len, head_dim, scale);
    }
    auto end_ns = std::chrono::high_resolution_clock::now();
    double ns_time = std::chrono::duration<double, std::milli>(end_ns - start_ns).count() / timing_iters;
    
    auto start_full = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < timing_iters; ++t) {
        full_dense_attention(full_dense_output.data(), queries.data(), keys.data(), values.data(),
                            seq_len, head_dim, scale);
    }
    auto end_full = std::chrono::high_resolution_clock::now();
    double full_time = std::chrono::duration<double, std::milli>(end_full - start_full).count() / timing_iters;
    
    std::cout << "NS time (ms): " << std::fixed << std::setprecision(6) << ns_time << std::endl;
    std::cout << "Full dense time (ms): " << std::fixed << std::setprecision(6) << full_time << std::endl;
    std::cout << std::endl;
}

// Benchmark single scenario with fixed window
void benchmark_fixed_window(int seq_len, int head_dim, int fixed_window) {
    std::cout << "=== FIXED WINDOW (seq_len=" << seq_len << ", window=" << fixed_window << ") ===" << std::endl;
    
    // Allocate data
    std::vector<float> queries(seq_len * head_dim);
    std::vector<float> keys(seq_len * head_dim);
    std::vector<float> values(seq_len * head_dim);
    std::vector<float> ns_output(seq_len * head_dim);
    std::vector<float> full_dense_output(seq_len * head_dim);
    
    // Generate local data
    generate_local_data(queries.data(), keys.data(), values.data(), seq_len, head_dim);
    
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    
    // Warmup iterations
    int warmup_iters = 1;
    for (int w = 0; w < warmup_iters; ++w) {
        block_sparse_attention(ns_output.data(), queries.data(), keys.data(), values.data(),
                             seq_len, head_dim, scale, 32, fixed_window);
        full_dense_attention(full_dense_output.data(), queries.data(), keys.data(), values.data(),
                            seq_len, head_dim, scale);
    }
    
    // Timed iterations
    int timing_iters = 3;
    
    auto start_ns = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < timing_iters; ++t) {
        block_sparse_attention(ns_output.data(), queries.data(), keys.data(), values.data(),
                             seq_len, head_dim, scale, 32, fixed_window);
    }
    auto end_ns = std::chrono::high_resolution_clock::now();
    double ns_time = std::chrono::duration<double, std::milli>(end_ns - start_ns).count() / timing_iters;
    
    auto start_full = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < timing_iters; ++t) {
        full_dense_attention(full_dense_output.data(), queries.data(), keys.data(), values.data(),
                            seq_len, head_dim, scale);
    }
    auto end_full = std::chrono::high_resolution_clock::now();
    double full_time = std::chrono::duration<double, std::milli>(end_full - start_full).count() / timing_iters;
    
    std::cout << "NS time (ms): " << std::fixed << std::setprecision(6) << ns_time << std::endl;
    std::cout << "Full dense time (ms): " << std::fixed << std::setprecision(6) << full_time << std::endl;
    std::cout << std::endl;
}

// Benchmark mixed batch
void benchmark_mixed_batch(int seq_len, int head_dim, int batch_size, int fixed_window = -1) {
    std::cout << "=== MIXED BATCH (seq_len=" << seq_len << ", batch=" << batch_size;
    if (fixed_window > 0) std::cout << ", window=" << fixed_window;
    std::cout << ") ===" << std::endl;
    
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    
    // Preallocate all vectors outside timing loop
    std::vector<std::vector<float>> queries_batch(batch_size, std::vector<float>(seq_len * head_dim));
    std::vector<std::vector<float>> keys_batch(batch_size, std::vector<float>(seq_len * head_dim));
    std::vector<std::vector<float>> values_batch(batch_size, std::vector<float>(seq_len * head_dim));
    std::vector<std::vector<float>> ns_output_batch(batch_size, std::vector<float>(seq_len * head_dim));
    std::vector<std::vector<float>> dense_output_batch(batch_size, std::vector<float>(seq_len * head_dim));
    
    // Generate data once
    for (int b = 0; b < batch_size; ++b) {
        if (b % 2 == 0) {
            generate_local_data(queries_batch[b].data(), keys_batch[b].data(), values_batch[b].data(), seq_len, head_dim);
        } else {
            generate_uniform_data(queries_batch[b].data(), keys_batch[b].data(), values_batch[b].data(), seq_len, head_dim);
        }
    }
    
    // Warmup iterations
    int warmup_iters = 1;
    for (int w = 0; w < warmup_iters; ++w) {
        for (int b = 0; b < batch_size; ++b) {
            if (fixed_window > 0) {
                block_sparse_attention(ns_output_batch[b].data(), queries_batch[b].data(), keys_batch[b].data(), values_batch[b].data(),
                                     seq_len, head_dim, scale, 32, fixed_window);
            } else {
                ns_attention(ns_output_batch[b].data(), queries_batch[b].data(), keys_batch[b].data(), values_batch[b].data(),
                             seq_len, head_dim, scale);
            }
            blocked_dense_attention(dense_output_batch[b].data(), queries_batch[b].data(), keys_batch[b].data(), values_batch[b].data(),
                                    seq_len, head_dim, scale);
        }
    }
    
    // Timed iterations
    int timing_iters = 3;
    
    auto start_ns = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < timing_iters; ++t) {
#pragma omp parallel for schedule(dynamic, 1)
        for (int b = 0; b < batch_size; ++b) {
            if (fixed_window > 0) {
                block_sparse_attention(ns_output_batch[b].data(), queries_batch[b].data(), keys_batch[b].data(), values_batch[b].data(),
                                     seq_len, head_dim, scale, 32, fixed_window);
            } else {
                ns_attention(ns_output_batch[b].data(), queries_batch[b].data(), keys_batch[b].data(), values_batch[b].data(),
                             seq_len, head_dim, scale);
            }
        }
    }
    auto end_ns = std::chrono::high_resolution_clock::now();
    double ns_time = std::chrono::duration<double, std::milli>(end_ns - start_ns).count() / timing_iters;
    
    auto start_dense = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < timing_iters; ++t) {
#pragma omp parallel for schedule(dynamic, 1)
        for (int b = 0; b < batch_size; ++b) {
            blocked_dense_attention(dense_output_batch[b].data(), queries_batch[b].data(), keys_batch[b].data(), values_batch[b].data(),
                                    seq_len, head_dim, scale);
        }
    }
    auto end_dense = std::chrono::high_resolution_clock::now();
    double dense_time = std::chrono::duration<double, std::milli>(end_dense - start_dense).count() / timing_iters;
    
    std::cout << "NS time (ms): " << std::fixed << std::setprecision(6) << ns_time << std::endl;
    std::cout << "Dense time (ms): " << std::fixed << std::setprecision(6) << dense_time << std::endl;
    std::cout << std::endl;
}

int main() {
    std::vector<int> seq_lengths = {256, 512, 1024, 2048};
    std::vector<int> long_seq_lengths = {1024, 2048, 4096, 8192};
    
    std::cout << "=== NSATTEND BENCHMARK ===" << std::endl;
    std::cout << std::endl;
    
    // Scenario 1: LOCAL head
    std::cout << "--- SCENARIO 1: LOCAL HEAD ---" << std::endl;
    for (int seq_len : seq_lengths) {
        benchmark_scenario("LOCAL", generate_local_data, seq_len, HEAD_DIM);
    }
    
    // Scenario 2: GLOBAL head
    std::cout << "--- SCENARIO 2: GLOBAL HEAD ---" << std::endl;
    std::vector<int> global_seq_lengths = {1024, 2048, 4096};
    for (int seq_len : global_seq_lengths) {
        benchmark_scenario_global(seq_len, HEAD_DIM);
    }
    
    // Scenario 3: UNIFORM regression
    std::cout << "--- SCENARIO 3: UNIFORM REGRESSION ---" << std::endl;
    std::vector<int> uniform_seq_lengths = {1024, 2048, 4096};
    for (int seq_len : uniform_seq_lengths) {
        benchmark_uniform_regression(seq_len, HEAD_DIM);
    }
    
    // Scenario 4: MIXED BATCH
    std::cout << "--- SCENARIO 4: MIXED BATCH ---" << std::endl;
    for (int seq_len : seq_lengths) {
        benchmark_mixed_batch(seq_len, HEAD_DIM, 8);
    }
    
    // Scenario 4b: MIXED batch with fixed window
    std::cout << "--- SCENARIO 4b: MIXED BATCH (FIXED WINDOW 512) ---" << std::endl;
    std::vector<int> mixed_fixed_seq_lengths = {4096, 8192};
    for (int seq_len : mixed_fixed_seq_lengths) {
        benchmark_mixed_batch(seq_len, HEAD_DIM, 8, 512);
    }
    
    // Scenario 5: FIXED WINDOW
    std::cout << "--- SCENARIO 5: FIXED WINDOW (512) ---" << std::endl;
    std::vector<int> fixed_seq_lengths = {1024, 2048, 4096, 8192};
    for (int seq_len : fixed_seq_lengths) {
        benchmark_fixed_window(seq_len, HEAD_DIM, 512);
    }
    
    return 0;
}

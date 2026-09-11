// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#pragma once

#include <cmath>
#include <algorithm>
#include <cstring>
#include <cassert>
#include <vector>
#include <immintrin.h>

enum HeadType { LOCAL, GLOBAL, UNIFORM };

// Softmax for a single row (full array)
void softmax_row(float* logits, int n) {
    float max_val = logits[0];
    for (int i = 1; i < n; ++i) {
        if (logits[i] > max_val) max_val = logits[i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        logits[i] = std::exp(logits[i] - max_val);
        sum += logits[i];
    }
    
    for (int i = 0; i < n; ++i) {
        logits[i] /= sum;
    }
}

// Softmax for a subarray (compact, no dead position iteration)
void softmax_row_range(float* logits, int n) {
    float max_val = logits[0];
    for (int i = 1; i < n; ++i) {
        if (logits[i] > max_val) max_val = logits[i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        logits[i] = std::exp(logits[i] - max_val);
        sum += logits[i];
    }
    
    for (int i = 0; i < n; ++i) {
        logits[i] /= sum;
    }
}

// Compute Spearman correlation between distance rank and attention rank
float compute_spearman_correlation(const float* attention, int seq_len, int row_idx) {
    std::vector<std::pair<float, int>> distance_rank;
    std::vector<std::pair<float, int>> attention_rank;
    
    for (int j = 0; j < seq_len; ++j) {
        distance_rank.push_back({static_cast<float>(std::abs(row_idx - j)), j});
        attention_rank.push_back({attention[j], j});
    }
    
    std::sort(distance_rank.begin(), distance_rank.end());
    std::sort(attention_rank.begin(), attention_rank.end());
    
    std::vector<int> distance_pos(seq_len);
    std::vector<int> attention_pos(seq_len);
    
    for (int j = 0; j < seq_len; ++j) {
        distance_pos[distance_rank[j].second] = j;
        attention_pos[attention_rank[j].second] = j;
    }
    
    float mean_d = seq_len / 2.0f;
    float mean_a = seq_len / 2.0f;
    float numerator = 0.0f;
    float denom_d = 0.0f;
    float denom_a = 0.0f;
    
    for (int j = 0; j < seq_len; ++j) {
        float d_diff = distance_pos[j] - mean_d;
        float a_diff = attention_pos[j] - mean_a;
        numerator += d_diff * a_diff;
        denom_d += d_diff * d_diff;
        denom_a += a_diff * a_diff;
    }
    
    if (denom_d == 0.0f || denom_a == 0.0f) return 0.0f;
    return numerator / std::sqrt(denom_d * denom_a);
}

// MECHANISM 1: Head type detector
HeadType detect_head_type(const float* scores, int seq_len, int head_dim, int sample_rows = 8) {
    assert(seq_len <= 8192 && "seq_len must be <= 8192");
    int actual_sample = std::min(sample_rows, seq_len);
    
    float total_correlation = 0.0f;
    int global_signal_count = 0;
    
    for (int s = 0; s < actual_sample; ++s) {
        // Sample rows evenly across the sequence
        int i = std::min(s * seq_len / actual_sample, seq_len - 1);
        
        // Compute attention weights for this row
        float attention[8192];
        for (int j = 0; j < seq_len; ++j) {
            attention[j] = scores[s * seq_len + j];
        }
        softmax_row(attention, seq_len);
        
        // Compute Spearman correlation (using actual row position i)
        float correlation = compute_spearman_correlation(attention, seq_len, i);
        total_correlation += std::abs(correlation);
        
        // Check for global signal (top-1 token > 40% weight)
        float max_weight = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
            max_weight = std::max(max_weight, attention[j]);
        }
        if (max_weight > 0.4f) {
            global_signal_count++;
        }
        
        // Early-exit: after 2+ samples, if no spatial structure and no global
        // signal, skip remaining samples — this is UNIFORM
        if (s >= 1 && total_correlation < 0.2f && global_signal_count == 0) {
            return UNIFORM;
        }
    }
    
    float avg_correlation = total_correlation / actual_sample;
    
    // Decision logic
    if (avg_correlation > 0.3f) {
        return LOCAL;
    } else if (global_signal_count > actual_sample / 2) {
        return GLOBAL;
    } else {
        return UNIFORM;
    }
}

// MECHANISM 2: Distance block mask (LOCAL heads only)
void build_block_mask(uint8_t* mask, int seq_len, int block_size = 32) {
    int num_blocks = (seq_len + block_size - 1) / block_size;
    int local_window = seq_len / 4;
    
    for (int bi = 0; bi < num_blocks; ++bi) {
        for (int bj = 0; bj < num_blocks; ++bj) {
            int block_dist = std::abs(bi - bj) * block_size;
            mask[bi * num_blocks + bj] = (block_dist <= local_window) ? 1 : 0;
        }
    }
}

// MECHANISM 3: KV candidate prediction (GLOBAL heads)
void predict_kv_candidates(int* candidates, int& num_candidates,
                            const float* query, const float* keys,
                            int seq_len, int head_dim, float keep_fraction = 0.3f) {
    int stride = 4;
    int num_samples = (seq_len + stride - 1) / stride;
    
    // Compute cheap dot products against strided keys
    std::vector<std::pair<float, int>> sampled_scores;
    for (int i = 0; i < num_samples; ++i) {
        int key_idx = i * stride;
        if (key_idx >= seq_len) break;
        
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += query[d] * keys[key_idx * head_dim + d];
        }
        sampled_scores.push_back({dot, key_idx});
    }
    
    // Keep top candidates
    int keep_count = static_cast<int>(keep_fraction * seq_len);
    keep_count = std::min(keep_count, seq_len);
    
    std::sort(sampled_scores.rbegin(), sampled_scores.rend());
    
    num_candidates = 0;
    for (int i = 0; i < keep_count && i < static_cast<int>(sampled_scores.size()); ++i) {
        candidates[num_candidates++] = sampled_scores[i].second;
    }
}

// Dense attention (baseline)
void dense_attention(float* output,
                    const float* queries, const float* keys, const float* values,
                    int seq_len, int head_dim, float scale) {
    for (int i = 0; i < seq_len; ++i) {
        const float* query = &queries[i * head_dim];
        
        // Compute attention scores
        std::vector<float> scores(seq_len);
        for (int j = 0; j < seq_len; ++j) {
            const float* key = &keys[j * head_dim];
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += query[d] * key[d];
            }
            scores[j] = dot * scale;
        }
        
        // Softmax
        softmax_row(scores.data(), seq_len);
        
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

// Block-sparse attention for LOCAL heads
void block_sparse_attention(float* output,
                           const float* queries, const float* keys, const float* values,
                           int seq_len, int head_dim, float scale, int block_size = 32, int fixed_window = -1) {
    assert(seq_len <= 8192 && "seq_len must be <= 8192");
    int num_blocks = (seq_len + block_size - 1) / block_size;
    int local_window = (fixed_window > 0) ? fixed_window : seq_len / 4;
    
    for (int i = 0; i < seq_len; ++i) {
        const float* query = &queries[i * head_dim];
        
        // Initialize all scores to -inf (inactive positions never computed)
        static thread_local float scores[8192];
        std::fill(scores, scores + seq_len, -1e30f);
        
        // Compute active block range inline (no mask array)
        int bj_start = std::max(0, (i - local_window) / block_size);
        int bj_end = std::min(num_blocks, (i + local_window) / block_size + 1);
        
        // Compute compact active range for softmax
        int active_start = std::max(bj_start * block_size, i - local_window);
        int active_end = std::min((bj_end-1+1)*block_size, i + local_window + 1);
        
        // Compute scores ONLY for active blocks in this row
        for (int bj = bj_start; bj < bj_end; ++bj) {
            int j_start = std::max(bj * block_size, i - local_window);
            int j_end = std::min((bj+1)*block_size, std::min(seq_len, i + local_window + 1));
            
            for (int j = j_start; j < j_end; ++j) {
                const float* key = &keys[j * head_dim];
#ifdef __AVX2__
                __m256 sum_vec = _mm256_setzero_ps();
                int d = 0;
                for (; d + 8 <= head_dim; d += 8) {
                    sum_vec = _mm256_fmadd_ps(
                        _mm256_loadu_ps(&query[d]),
                        _mm256_loadu_ps(&key[d]),
                        sum_vec);
                }
                float buf[8]; _mm256_storeu_ps(buf, sum_vec);
                float dot = buf[0]+buf[1]+buf[2]+buf[3]+buf[4]+buf[5]+buf[6]+buf[7];
                for (; d < head_dim; ++d) dot += query[d] * key[d];
#else
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) dot += query[d] * key[d];
#endif
                scores[j] = dot * scale;
            }
        }
        
        // Compact softmax over active positions only
        softmax_row_range(scores + active_start, active_end - active_start);
        
        // Local accumulator for value accumulation (eliminates scattered write cache thrash)
        float acc[64] = {0.0f};
        
        for (int bj = bj_start; bj < bj_end; ++bj) {
            int j_start = std::max(bj * block_size, i - local_window);
            int j_end = std::min((bj+1)*block_size, std::min(seq_len, i + local_window + 1));
            
            for (int j = j_start; j < j_end; ++j) {
                float weight = scores[j];
                const float* value = &values[j * head_dim];
#ifdef __AVX2__
                __m256 w_vec = _mm256_set1_ps(weight);
                int d = 0;
                for (; d + 8 <= head_dim; d += 8) {
                    __m256 v_vec = _mm256_loadu_ps(&value[d]);
                    __m256 a_vec = _mm256_loadu_ps(&acc[d]);
                    a_vec = _mm256_fmadd_ps(w_vec, v_vec, a_vec);
                    _mm256_storeu_ps(&acc[d], a_vec);
                }
                for (; d < head_dim; ++d) {
                    acc[d] += weight * value[d];
                }
#else
                for (int d = 0; d < head_dim; ++d) {
                    acc[d] += weight * value[d];
                }
#endif
            }
        }
        
        // Single store from accumulator to output
        for (int d = 0; d < head_dim; ++d) {
            output[i * head_dim + d] = acc[d];
        }
    }
}

// Candidate-sparse attention for GLOBAL heads
void candidate_sparse_attention(float* output,
                                const float* queries, const float* keys, const float* values,
                                int seq_len, int head_dim, float scale,
                                const int* candidates, int num_candidates) {
    for (int i = 0; i < seq_len; ++i) {
        const float* query = &queries[i * head_dim];
        
        // Compute scores for ALL positions
        std::vector<float> scores(seq_len);
        for (int j = 0; j < seq_len; ++j) {
            const float* key = &keys[j * head_dim];
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += query[d] * key[d];
            }
            scores[j] = dot * scale;
        }
        
        // Set non-candidate positions to -inf
        std::vector<bool> is_candidate(seq_len, false);
        for (int c = 0; c < num_candidates; ++c) {
            is_candidate[candidates[c]] = true;
        }
        for (int j = 0; j < seq_len; ++j) {
            if (!is_candidate[j]) {
                scores[j] = -1e30f;
            }
        }
        
        // Softmax over ALL positions (non-candidates get 0 weight due to -inf)
        softmax_row(scores.data(), seq_len);
        
        // Weighted sum of values (only candidates contribute)
        for (int d = 0; d < head_dim; ++d) {
            output[i * head_dim + d] = 0.0f;
        }
        
        for (int c = 0; c < num_candidates; ++c) {
            int j = candidates[c];
            float weight = scores[j];
            const float* value = &values[j * head_dim];
            for (int d = 0; d < head_dim; ++d) {
                output[i * head_dim + d] += weight * value[d];
            }
        }
    }
}

// MECHANISM 4: Main NS attention function
void ns_attention(float* output,
                  const float* queries, const float* keys, const float* values,
                  int seq_len, int head_dim, float scale) {
    // Detect head type using sample rows (not full matrix to avoid O(N²) memory)
    int sample_rows = 8;
    std::vector<float> sample_scores(sample_rows * seq_len);
    for (int s = 0; s < sample_rows; ++s) {
        int i = std::min(s * seq_len / sample_rows, seq_len - 1);
        const float* query = &queries[i * head_dim];
        for (int j = 0; j < seq_len; ++j) {
            const float* key = &keys[j * head_dim];
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += query[d] * key[d];
            }
            sample_scores[s * seq_len + j] = dot * scale;
        }
    }
    
    HeadType head_type = detect_head_type(sample_scores.data(), seq_len, head_dim, sample_rows);
    
    if (head_type == LOCAL) {
        // Use block-sparse attention (inline block range, no mask)
        block_sparse_attention(output, queries, keys, values, seq_len, head_dim, scale, 32, -1);
    } else if (head_type == GLOBAL) {
        if (seq_len <= 1024) {
            // Small seq_len: candidate sparse path is strictly slower than dense
            dense_attention(output, queries, keys, values, seq_len, head_dim, scale);
        } else {
            // Predict KV candidates and use candidate-sparse attention
            std::vector<int> candidates(seq_len);
            int num_candidates = 0;
            predict_kv_candidates(candidates.data(), num_candidates, queries, keys,
                                  seq_len, head_dim, 0.3f);
            candidate_sparse_attention(output, queries, keys, values, seq_len, head_dim, scale,
                                      candidates.data(), num_candidates);
        }
    } else {
        // UNIFORM: dense fallback
        dense_attention(output, queries, keys, values, seq_len, head_dim, scale);
    }
}

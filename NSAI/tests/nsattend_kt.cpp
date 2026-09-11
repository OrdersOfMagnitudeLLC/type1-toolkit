#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>
#include <iomanip>

const int HEAD_DIM = 64;
const int BLOCK_SIZE = 32;

std::vector<double> softmax(const std::vector<double>& logits) {
    std::vector<double> result(logits.size());
    double max_val = *std::max_element(logits.begin(), logits.end());
    double sum = 0.0;
    
    for (size_t i = 0; i < logits.size(); i++) {
        result[i] = std::exp(logits[i] - max_val);
        sum += result[i];
    }
    
    for (size_t i = 0; i < result.size(); i++) {
        result[i] /= sum;
    }
    
    return result;
}

std::vector<std::vector<double>> generate_local_window_attention(int seq_len) {
    std::vector<std::vector<double>> attention(seq_len, std::vector<double>(seq_len));
    std::mt19937 gen(42);
    std::normal_distribution<double> noise(0.0, 0.1);
    
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            int distance = std::abs(i - j);
            double decay = std::exp(-distance / 64.0);
            attention[i][j] = decay + noise(gen);
        }
        attention[i] = softmax(attention[i]);
    }
    
    return attention;
}

std::vector<std::vector<double>> generate_diagonal_global_attention(int seq_len) {
    std::vector<std::vector<double>> attention(seq_len, std::vector<double>(seq_len));
    std::mt19937 gen(42);
    std::normal_distribution<double> noise(0.0, 0.05);
    
    int num_global = 8;
    std::vector<int> global_tokens;
    for (int i = 0; i < num_global; i++) {
        global_tokens.push_back(i * (seq_len / num_global));
    }
    
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            double score = 0.0;
            if (i == j) {
                score = 2.0;
            } else {
                for (int gt : global_tokens) {
                    if (j == gt) {
                        score = 1.5;
                        break;
                    }
                }
            }
            attention[i][j] = score + noise(gen);
        }
        attention[i] = softmax(attention[i]);
    }
    
    return attention;
}

std::vector<std::vector<double>> generate_uniform_random_attention(int seq_len) {
    std::vector<std::vector<double>> attention(seq_len, std::vector<double>(seq_len));
    std::mt19937 gen(42);
    std::normal_distribution<double> noise(0.0, 1.0);
    
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            attention[i][j] = noise(gen);
        }
        attention[i] = softmax(attention[i]);
    }
    
    return attention;
}

double measure_weight_sparsity(const std::vector<std::vector<double>>& attention, int seq_len) {
    double threshold = 5.0 / seq_len;
    int total = 0;
    int below_threshold = 0;
    
    for (const auto& row : attention) {
        for (double weight : row) {
            total++;
            if (weight < threshold) {
                below_threshold++;
            }
        }
    }
    
    return 100.0 * below_threshold / total;
}

double measure_spatial_predictability(const std::vector<std::vector<double>>& attention) {
    int seq_len = attention.size();
    double total_correlation = 0.0;
    int num_rows = 0;
    
    for (int i = 0; i < seq_len; i++) {
        std::vector<std::pair<double, int>> distance_rank;
        std::vector<std::pair<double, int>> attention_rank;
        
        for (int j = 0; j < seq_len; j++) {
            distance_rank.push_back({std::abs(i - j), j});
            attention_rank.push_back({attention[i][j], j});
        }
        
        std::sort(distance_rank.begin(), distance_rank.end());
        std::sort(attention_rank.begin(), attention_rank.end());
        
        std::vector<int> distance_pos(seq_len);
        std::vector<int> attention_pos(seq_len);
        
        for (int j = 0; j < seq_len; j++) {
            distance_pos[distance_rank[j].second] = j;
            attention_pos[attention_rank[j].second] = j;
        }
        
        double mean_d = seq_len / 2.0;
        double mean_a = seq_len / 2.0;
        double numerator = 0.0;
        double denom_d = 0.0;
        double denom_a = 0.0;
        
        for (int j = 0; j < seq_len; j++) {
            double d_diff = distance_pos[j] - mean_d;
            double a_diff = attention_pos[j] - mean_a;
            numerator += d_diff * a_diff;
            denom_d += d_diff * d_diff;
            denom_a += a_diff * a_diff;
        }
        
        double row_correlation = numerator / std::sqrt(denom_d * denom_a);
        total_correlation += row_correlation;
        num_rows++;
    }
    
    return total_correlation / num_rows;
}

double measure_top_k_concentration(const std::vector<std::vector<double>>& attention) {
    int seq_len = attention.size();
    int k = seq_len / 10;
    double total_concentration = 0.0;
    
    for (int i = 0; i < seq_len; i++) {
        std::vector<std::pair<double, int>> weighted;
        for (int j = 0; j < seq_len; j++) {
            weighted.push_back({attention[i][j], j});
        }
        std::sort(weighted.rbegin(), weighted.rend());
        
        double top_k_weight = 0.0;
        for (int t = 0; t < k; t++) {
            top_k_weight += weighted[t].first;
        }
        
        total_concentration += top_k_weight;
    }
    
    return 100.0 * total_concentration / seq_len;
}

double measure_block_sparsity(const std::vector<std::vector<double>>& attention, int seq_len) {
    int num_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int total_blocks = num_blocks * num_blocks;
    int sparse_blocks = 0;
    
    for (int bi = 0; bi < num_blocks; bi++) {
        for (int bj = 0; bj < num_blocks; bj++) {
            double block_weight = 0.0;
            int cells_in_block = 0;
            
            for (int i = bi * BLOCK_SIZE; i < std::min((bi + 1) * BLOCK_SIZE, seq_len); i++) {
                for (int j = bj * BLOCK_SIZE; j < std::min((bj + 1) * BLOCK_SIZE, seq_len); j++) {
                    block_weight += attention[i][j];
                    cells_in_block++;
                }
            }
            
            if (block_weight < (1.0 / seq_len) * cells_in_block) {
                sparse_blocks++;
            }
        }
    }
    
    return 100.0 * sparse_blocks / total_blocks;
}

std::vector<std::vector<bool>> compute_block_map(const std::vector<std::vector<double>>& attention) {
    int seq_len = attention.size();
    int num_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    std::vector<std::vector<bool>> block_map(num_blocks, std::vector<bool>(num_blocks, false));
    
    for (int bi = 0; bi < num_blocks; bi++) {
        for (int bj = 0; bj < num_blocks; bj++) {
            double block_weight = 0.0;
            int cells_in_block = 0;
            
            for (int i = bi * BLOCK_SIZE; i < std::min((bi + 1) * BLOCK_SIZE, seq_len); i++) {
                for (int j = bj * BLOCK_SIZE; j < std::min((bj + 1) * BLOCK_SIZE, seq_len); j++) {
                    block_weight += attention[i][j];
                    cells_in_block++;
                }
            }
            
            if (block_weight >= (1.0 / seq_len) * cells_in_block) {
                block_map[bi][bj] = true;
            }
        }
    }
    
    return block_map;
}

std::vector<std::vector<double>> dense_matmul(const std::vector<std::vector<double>>& A, 
                                               const std::vector<std::vector<double>>& B) {
    int m = A.size();
    int n = B[0].size();
    int k = A[0].size();
    
    std::vector<std::vector<double>> C(m, std::vector<double>(n, 0.0));
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int p = 0; p < k; p++) {
                C[i][j] += A[i][p] * B[p][j];
            }
        }
    }
    
    return C;
}

std::vector<std::vector<double>> sparse_matmul(const std::vector<std::vector<double>>& A,
                                                const std::vector<std::vector<double>>& B,
                                                const std::vector<std::vector<bool>>& block_map) {
    int m = A.size();
    int n = B[0].size();
    int k = A[0].size();
    int num_blocks = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    std::vector<std::vector<double>> C(m, std::vector<double>(n, 0.0));
    
    for (int bi = 0; bi < num_blocks; bi++) {
        for (int bj = 0; bj < num_blocks; bj++) {
            if (!block_map[bi][bj]) continue;
            
            int i_start = bi * BLOCK_SIZE;
            int j_start = bj * BLOCK_SIZE;
            int i_end = std::min(i_start + BLOCK_SIZE, k);
            int j_end = std::min(j_start + BLOCK_SIZE, n);
            
            for (int i = 0; i < m; i++) {
                for (int p = i_start; p < i_end; p++) {
                    for (int j = j_start; j < j_end; j++) {
                        C[i][j] += A[i][p] * B[p][j];
                    }
                }
            }
        }
    }
    
    return C;
}

int main() {
    std::vector<int> seq_lengths = {512, 1024, 2048, 4096};
    
    std::cout << std::fixed << std::setprecision(6);
    
    std::cout << "=== STEP 1 & 2: SPARSITY MEASUREMENT ===" << std::endl;
    std::cout << std::endl;
    
    for (int seq_len : seq_lengths) {
        std::cout << "Sequence Length: " << seq_len << std::endl;
        std::cout << std::endl;
        
        auto local = generate_local_window_attention(seq_len);
        auto diagonal = generate_diagonal_global_attention(seq_len);
        auto uniform = generate_uniform_random_attention(seq_len);
        
        std::cout << "Local Window Pattern:" << std::endl;
        double local_sparsity = measure_weight_sparsity(local, seq_len);
        double local_predictability = measure_spatial_predictability(local);
        double local_block_sparsity = measure_block_sparsity(local, seq_len);
        double local_topk = measure_top_k_concentration(local);
        std::cout << "Weight sparsity %: " << local_sparsity << std::endl;
        std::cout << "Spatial predictability (Spearman): " << local_predictability << std::endl;
        std::cout << "Block sparsity %: " << local_block_sparsity << std::endl;
        std::cout << "Top-K concentration %: " << local_topk << std::endl;
        std::cout << std::endl;
        
        std::cout << "Diagonal + Global Pattern:" << std::endl;
        double diag_sparsity = measure_weight_sparsity(diagonal, seq_len);
        double diag_predictability = measure_spatial_predictability(diagonal);
        double diag_block_sparsity = measure_block_sparsity(diagonal, seq_len);
        double diag_topk = measure_top_k_concentration(diagonal);
        std::cout << "Weight sparsity %: " << diag_sparsity << std::endl;
        std::cout << "Spatial predictability (Spearman): " << diag_predictability << std::endl;
        std::cout << "Block sparsity %: " << diag_block_sparsity << std::endl;
        std::cout << "Top-K concentration %: " << diag_topk << std::endl;
        std::cout << std::endl;
        
        std::cout << "Uniform Random Pattern:" << std::endl;
        double uniform_sparsity = measure_weight_sparsity(uniform, seq_len);
        double uniform_predictability = measure_spatial_predictability(uniform);
        double uniform_block_sparsity = measure_block_sparsity(uniform, seq_len);
        double uniform_topk = measure_top_k_concentration(uniform);
        std::cout << "Weight sparsity %: " << uniform_sparsity << std::endl;
        std::cout << "Spatial predictability (Spearman): " << uniform_predictability << std::endl;
        std::cout << "Block sparsity %: " << uniform_block_sparsity << std::endl;
        std::cout << "Top-K concentration %: " << uniform_topk << std::endl;
        std::cout << std::endl;
        std::cout << "---" << std::endl;
        std::cout << std::endl;
    }
    
    std::cout << "=== STEP 3: DENSE VS SPARSE MATMUL TIMING ===" << std::endl;
    std::cout << std::endl;
    
    for (int seq_len : seq_lengths) {
        std::cout << "Sequence Length: " << seq_len << std::endl;
        
        auto local = generate_local_window_attention(seq_len);
        auto block_map = compute_block_map(local);
        
        std::vector<std::vector<double>> value_matrix(seq_len, std::vector<double>(HEAD_DIM));
        std::mt19937 gen(42);
        std::normal_distribution<double> dist(0.0, 1.0);
        
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < HEAD_DIM; j++) {
                value_matrix[i][j] = dist(gen);
            }
        }
        
        int warmup_iters = 5;
        int timing_iters = 20;
        
        for (int w = 0; w < warmup_iters; w++) {
            dense_matmul(local, value_matrix);
        }
        
        auto start_dense = std::chrono::high_resolution_clock::now();
        for (int t = 0; t < timing_iters; t++) {
            dense_matmul(local, value_matrix);
        }
        auto end_dense = std::chrono::high_resolution_clock::now();
        double dense_time = std::chrono::duration<double, std::milli>(end_dense - start_dense).count() / timing_iters;
        
        for (int w = 0; w < warmup_iters; w++) {
            sparse_matmul(local, value_matrix, block_map);
        }
        
        auto start_sparse = std::chrono::high_resolution_clock::now();
        for (int t = 0; t < timing_iters; t++) {
            sparse_matmul(local, value_matrix, block_map);
        }
        auto end_sparse = std::chrono::high_resolution_clock::now();
        double sparse_time = std::chrono::duration<double, std::milli>(end_sparse - start_sparse).count() / timing_iters;
        
        std::cout << "Dense matmul time (ms): " << dense_time << std::endl;
        std::cout << "Sparse matmul time (ms): " << sparse_time << std::endl;
        std::cout << std::endl;
    }
    
    return 0;
}

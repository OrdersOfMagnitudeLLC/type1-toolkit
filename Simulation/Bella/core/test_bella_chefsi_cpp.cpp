// OOM Proprietary — All Rights Reserved, Orders of Magnitude LLC

#include "BellaCheFSI.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <tuple>

// CSR matrix structure
struct CSRMatrix {
    int n;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<double> values;
};

// Build 3D Laplacian in CSR format (n=16 per dimension, N=4096)
CSRMatrix build_3d_laplacian(int n) {
    int N = n * n * n;
    CSRMatrix H;
    H.n = N;
    H.row_ptr.resize(N + 1);
    
    // Count non-zeros per row (7-point stencil: center + 6 neighbors)
    std::vector<int> nnz_per_row(N, 7);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                int idx = i * n * n + j * n + k;
                int count = 1; // center
                if (i > 0) count++;
                if (i < n-1) count++;
                if (j > 0) count++;
                if (j < n-1) count++;
                if (k > 0) count++;
                if (k < n-1) count++;
                nnz_per_row[idx] = count;
            }
        }
    }
    
    // Build row_ptr
    H.row_ptr[0] = 0;
    for (int i = 0; i < N; ++i) {
        H.row_ptr[i + 1] = H.row_ptr[i] + nnz_per_row[i];
    }
    
    int total_nnz = H.row_ptr[N];
    H.col_idx.resize(total_nnz);
    H.values.resize(total_nnz);
    
    // Fill in CSR structure
    std::vector<int> current_pos(N, 0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                int idx = i * n * n + j * n + k;
                int pos = H.row_ptr[idx] + current_pos[idx];
                
                // Center (diagonal)
                H.col_idx[pos] = idx;
                H.values[pos] = -6.0;
                current_pos[idx]++;
                
                // Neighbors
                if (i > 0) {
                    int neighbor = (i-1) * n * n + j * n + k;
                    pos = H.row_ptr[idx] + current_pos[idx];
                    H.col_idx[pos] = neighbor;
                    H.values[pos] = 1.0;
                    current_pos[idx]++;
                }
                if (i < n-1) {
                    int neighbor = (i+1) * n * n + j * n + k;
                    pos = H.row_ptr[idx] + current_pos[idx];
                    H.col_idx[pos] = neighbor;
                    H.values[pos] = 1.0;
                    current_pos[idx]++;
                }
                if (j > 0) {
                    int neighbor = i * n * n + (j-1) * n + k;
                    pos = H.row_ptr[idx] + current_pos[idx];
                    H.col_idx[pos] = neighbor;
                    H.values[pos] = 1.0;
                    current_pos[idx]++;
                }
                if (j < n-1) {
                    int neighbor = i * n * n + (j+1) * n + k;
                    pos = H.row_ptr[idx] + current_pos[idx];
                    H.col_idx[pos] = neighbor;
                    H.values[pos] = 1.0;
                    current_pos[idx]++;
                }
                if (k > 0) {
                    int neighbor = i * n * n + j * n + (k-1);
                    pos = H.row_ptr[idx] + current_pos[idx];
                    H.col_idx[pos] = neighbor;
                    H.values[pos] = 1.0;
                    current_pos[idx]++;
                }
                if (k < n-1) {
                    int neighbor = i * n * n + j * n + (k+1);
                    pos = H.row_ptr[idx] + current_pos[idx];
                    H.col_idx[pos] = neighbor;
                    H.values[pos] = 1.0;
                    current_pos[idx]++;
                }
            }
        }
    }
    
    return H;
}

// Sparse matrix-vector multiplication: y = H * x
void sparse_matvec(const CSRMatrix& H, const double* x, double* y) {
    for (int i = 0; i < H.n; ++i) {
        y[i] = 0.0;
        for (int j = H.row_ptr[i]; j < H.row_ptr[i + 1]; ++j) {
            y[i] += H.values[j] * x[H.col_idx[j]];
        }
    }
}

// Analytical eigenvalues for 3D Laplacian (negative for -Laplacian)
// Correct formula: lambda = -(6 - 2*cos(pi*i/(n+1)) - 2*cos(pi*j/(n+1)) - 2*cos(pi*k/(n+1)))
// where i, j, k range from 1 to n
double analytical_eigenvalue(int i, int j, int k, int n) {
    double pi = M_PI;
    double term_i = 2.0 * (1.0 - std::cos(pi * i / (n + 1)));
    double term_j = 2.0 * (1.0 - std::cos(pi * j / (n + 1)));
    double term_k = 2.0 * (1.0 - std::cos(pi * k / (n + 1)));
    return -(term_i + term_j + term_k);  // Negative for -Laplacian
}

// ARPACK integration is complex and requires careful C interface handling
// For now, benchmarking Bella CheFSI alone
// ARPACK comparison can be added later with proper C bindings

// Benchmark function (returns time)
double benchmark(int n, int k, double arpack_py_time) {
    int N = n * n * n;
    
    // Build matrix
    auto H = build_3d_laplacian(n);
    
    // Wrap matvec as lambda
    auto matvec = [&](const double* x, double* y, int N_local) {
        sparse_matvec(H, x, y);
    };
    
    // Run Bella CheFSI
    auto start = std::chrono::high_resolution_clock::now();
    auto result = BellaCheFSI<double>::solve(matvec, N, k, 1e-10, 30, 60);
    auto end = std::chrono::high_resolution_clock::now();
    double bella_time = std::chrono::duration<double>(end - start).count();
    
    // Compute analytical eigenvalues for comparison
    // Generate ALL N eigenvalues (i, j, k from 1 to n)
    std::vector<double> analytical;
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= n; ++j) {
            for (int k_idx = 1; k_idx <= n; ++k_idx) {
                analytical.push_back(analytical_eigenvalue(i, j, k_idx, n));
            }
        }
    }
    std::sort(analytical.begin(), analytical.end());
    analytical.resize(k);
    
    // Compute max error for Bella (sort Bella eigenvalues ascending first)
    std::vector<double> bella_sorted = result.eigenvalues;
    std::sort(bella_sorted.begin(), bella_sorted.end());
    
    // Position-by-position comparison on sorted lists (handles degeneracy correctly)
    double max_error_bella = 0.0;
    for (int i = 0; i < k; ++i) {
        double error = std::abs(bella_sorted[i] - analytical[i]);
        max_error_bella = std::max(max_error_bella, error);
    }
    
    // Print summary (minimal format)
    std::cout << "N=" << N << "   k=" << k 
              << "  Bella=" << std::fixed << std::setprecision(3) << bella_time << "s  "
              << "ARPACK_py=" << (arpack_py_time > 0 ? std::to_string(arpack_py_time) + "s" : "N/A") << "  "
              << "Speedup=" << std::fixed << std::setprecision(2) 
              << (arpack_py_time > 0 ? arpack_py_time / bella_time : 0.0) << "x  "
              << "EigErr=" << std::scientific << max_error_bella << std::endl;
    
    return bella_time;
}

int main() {
    // ARPACK Python baseline times (from findings/sparse_fd_results.txt, same hardware)
    std::vector<std::tuple<int, int, double>> arpack_baseline = {
        {512, 20, 0.027},
        {1728, 50, 0.202},
        {4096, 100, 4.184},
        {8000, 200, 28.994},
        {13824, 300, -1.0}  // N/A (not tested)
    };
    
    // Run benchmark cases
    benchmark(8, 20, std::get<2>(arpack_baseline[0]));
    benchmark(12, 50, std::get<2>(arpack_baseline[1]));
    benchmark(16, 100, std::get<2>(arpack_baseline[2]));
    benchmark(20, 200, std::get<2>(arpack_baseline[3]));
    benchmark(24, 300, std::get<2>(arpack_baseline[4]));
    
    std::cout << "Note: ARPACK baseline = Python scipy.sparse.linalg.eigsh, Spectre i7, same matrix." << std::endl;
    
    return 0;
}

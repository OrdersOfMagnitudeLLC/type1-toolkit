// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#include "nsmatrix.hpp"
#include <iostream>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <algorithm>
#include <vector>

#ifdef USE_EIGEN
#include <Eigen/Dense>
#endif

#define WARMUP 10
#define ITERS 100

int main() {
    // typedef Eigen::Matrix<float, 16, 16, Eigen::RowMajor> Matrix16f;
    const size_t B = 1000;  // 1000 blocks
    const size_t k = 16;    // 16x16 blocks
    
    // Allocate NS block-diagonal matrices
    NSBlockDiagMatrix A_ns, B_ns, C_ns;
    if (!A_ns.allocate(B, k) || !B_ns.allocate(B, k) || !C_ns.allocate(B, k)) {
        std::cerr << "NS allocation failed" << std::endl;
        return 1;
    }
    
    // Initialize with random values
    for (size_t i = 0; i < B * k * k; ++i) {
        A_ns.blocks[i] = (float)rand() / RAND_MAX;
        B_ns.blocks[i] = (float)rand() / RAND_MAX;
    }
    
    // Set bottom row of first 5 blocks to [0,0,0,1] pattern
    for (size_t b = 0; b < 5; ++b) {
        size_t block_offset = b * k * k;
        // Bottom row is last row of block (row 15)
        size_t bottom_row_offset = block_offset + 15 * k;
        A_ns.blocks[bottom_row_offset + 12] = 0.0f;
        A_ns.blocks[bottom_row_offset + 13] = 0.0f;
        A_ns.blocks[bottom_row_offset + 14] = 0.0f;
        A_ns.blocks[bottom_row_offset + 15] = 1.0f;
        B_ns.blocks[bottom_row_offset + 12] = 0.0f;
        B_ns.blocks[bottom_row_offset + 13] = 0.0f;
        B_ns.blocks[bottom_row_offset + 14] = 0.0f;
        B_ns.blocks[bottom_row_offset + 15] = 1.0f;
    }
    
    // Assert: print bottom row values of first 5 blocks in A and B
    std::cout << "Bottom row values (first 5 blocks):" << std::endl;
    for (size_t b = 0; b < 5; ++b) {
        size_t block_offset = b * k * k;
        size_t bottom_row_offset = block_offset + 15 * k;
        std::cout << "Block " << b << " A: [" 
                  << A_ns.blocks[bottom_row_offset + 12] << ", "
                  << A_ns.blocks[bottom_row_offset + 13] << ", "
                  << A_ns.blocks[bottom_row_offset + 14] << ", "
                  << A_ns.blocks[bottom_row_offset + 15] << "]" << std::endl;
        std::cout << "Block " << b << " B: [" 
                  << B_ns.blocks[bottom_row_offset + 12] << ", "
                  << B_ns.blocks[bottom_row_offset + 13] << ", "
                  << B_ns.blocks[bottom_row_offset + 14] << ", "
                  << B_ns.blocks[bottom_row_offset + 15] << "]" << std::endl;
    }
    
#ifdef USE_EIGEN
    // Allocate reference matrices for correctness check
    NSBlockDiagMatrix C_ref;
    if (!C_ref.allocate(B, k)) {
        std::cerr << "Reference allocation failed" << std::endl;
        return 1;
    }
    
    // Compute reference using Eigen fixed-size per-block
    typedef Eigen::Matrix<float, 16, 16, Eigen::RowMajor> Matrix16f;
    for (size_t b = 0; b < B; ++b) {
        Eigen::Map<const Matrix16f> A_block(A_ns.blocks + b * k * k);
        Eigen::Map<const Matrix16f> B_block(B_ns.blocks + b * k * k);
        Eigen::Map<Matrix16f> C_block(C_ref.blocks + b * k * k);
        C_block = A_block * B_block;
    }
    
    // Compute NS result
    ns_blockdiag_mul(A_ns, B_ns, C_ns);
    
    // Correctness check: verify matches Eigen within 1e-4
    bool correct = true;
    for (size_t i = 0; i < k * k; ++i) {
        if (std::abs(C_ns.blocks[i] - C_ref.blocks[i]) > 1e-4f) {
            std::cerr << "Correctness check failed at index " << i 
                      << ": ns=" << C_ns.blocks[i] << ", ref=" << C_ref.blocks[i] << std::endl;
            correct = false;
            break;
        }
    }
    if (correct) {
        std::cout << "Correctness check: PASSED" << std::endl;
    }
    
    // Warmup + multi-iteration for Eigen
    for (int w = 0; w < WARMUP; ++w) {
        for (size_t b = 0; b < B; ++b) {
            Eigen::Map<const Matrix16f> A_block(A_ns.blocks + b * k * k);
            Eigen::Map<const Matrix16f> B_block(B_ns.blocks + b * k * k);
            Eigen::Map<Matrix16f> C_block(C_ns.blocks + b * k * k);
            C_block = A_block * B_block;
        }
    }
    double eigen_times[ITERS];
    for (int i = 0; i < ITERS; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (size_t b = 0; b < B; ++b) {
            Eigen::Map<const Matrix16f> A_block(A_ns.blocks + b * k * k);
            Eigen::Map<const Matrix16f> B_block(B_ns.blocks + b * k * k);
            Eigen::Map<Matrix16f> C_block(C_ns.blocks + b * k * k);
            C_block = A_block * B_block;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        eigen_times[i] = std::chrono::duration<double,std::nano>(t1-t0).count() / B;
    }
    volatile float sink_eigen = C_ns.blocks[0];
    (void)sink_eigen;
    
    // Warmup + multi-iteration for NS
    for (int w = 0; w < WARMUP; ++w) {
        ns_blockdiag_mul(A_ns, B_ns, C_ns);
    }
    double ns_times[ITERS];
    for (int i = 0; i < ITERS; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        ns_blockdiag_mul(A_ns, B_ns, C_ns);
        auto t1 = std::chrono::high_resolution_clock::now();
        ns_times[i] = std::chrono::duration<double,std::nano>(t1-t0).count() / B;
    }
    volatile float sink_ns = C_ns.blocks[0];
    (void)sink_ns;
    
    // Sort and compute median
    std::sort(ns_times, ns_times + ITERS);
    std::sort(eigen_times, eigen_times + ITERS);
    double ns_median = ns_times[ITERS / 2];
    double eigen_median = eigen_times[ITERS / 2];
    
    // Print results for B=1000
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "B=1000:" << std::endl;
    std::cout << "NSMatrix: " << ns_median << " ns/block" << std::endl;
    std::cout << "Eigen: " << eigen_median << " ns/block" << std::endl;
    std::cout << "Ratio (NS/Eigen): " << (ns_median / eigen_median) << std::endl;
    
    // Second benchmark with B=10000
    const size_t B2 = 10000;
    NSBlockDiagMatrix A_ns2, B_ns2, C_ns2;
    if (!A_ns2.allocate(B2, k) || !B_ns2.allocate(B2, k) || !C_ns2.allocate(B2, k)) {
        std::cerr << "NS allocation failed for B=10000" << std::endl;
        return 1;
    }
    
    // Initialize with random values
    for (size_t i = 0; i < B2 * k * k; ++i) {
        A_ns2.blocks[i] = (float)rand() / RAND_MAX;
        B_ns2.blocks[i] = (float)rand() / RAND_MAX;
    }
    
    // Set bottom row of first 5 blocks to [0,0,0,1] pattern
    for (size_t b = 0; b < 5; ++b) {
        size_t block_offset = b * k * k;
        size_t bottom_row_offset = block_offset + 15 * k;
        A_ns2.blocks[bottom_row_offset + 12] = 0.0f;
        A_ns2.blocks[bottom_row_offset + 13] = 0.0f;
        A_ns2.blocks[bottom_row_offset + 14] = 0.0f;
        A_ns2.blocks[bottom_row_offset + 15] = 1.0f;
        B_ns2.blocks[bottom_row_offset + 12] = 0.0f;
        B_ns2.blocks[bottom_row_offset + 13] = 0.0f;
        B_ns2.blocks[bottom_row_offset + 14] = 0.0f;
        B_ns2.blocks[bottom_row_offset + 15] = 1.0f;
    }
    
    // Warmup + multi-iteration for Eigen (B=10000)
    for (int w = 0; w < WARMUP; ++w) {
        for (size_t b = 0; b < B2; ++b) {
            Eigen::Map<const Matrix16f> A_block(A_ns2.blocks + b * k * k);
            Eigen::Map<const Matrix16f> B_block(B_ns2.blocks + b * k * k);
            Eigen::Map<Matrix16f> C_block(C_ns2.blocks + b * k * k);
            C_block = A_block * B_block;
        }
    }
    double eigen_times2[ITERS];
    for (int i = 0; i < ITERS; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (size_t b = 0; b < B2; ++b) {
            Eigen::Map<const Matrix16f> A_block(A_ns2.blocks + b * k * k);
            Eigen::Map<const Matrix16f> B_block(B_ns2.blocks + b * k * k);
            Eigen::Map<Matrix16f> C_block(C_ns2.blocks + b * k * k);
            C_block = A_block * B_block;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        eigen_times2[i] = std::chrono::duration<double,std::nano>(t1-t0).count() / B2;
    }
    volatile float sink_eigen2 = C_ns2.blocks[0];
    (void)sink_eigen2;
    
    // Warmup + multi-iteration for NS (B=10000)
    for (int w = 0; w < WARMUP; ++w) {
        ns_blockdiag_mul(A_ns2, B_ns2, C_ns2);
    }
    double ns_times2[ITERS];
    for (int i = 0; i < ITERS; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        ns_blockdiag_mul(A_ns2, B_ns2, C_ns2);
        auto t1 = std::chrono::high_resolution_clock::now();
        ns_times2[i] = std::chrono::duration<double,std::nano>(t1-t0).count() / B2;
    }
    volatile float sink_ns2 = C_ns2.blocks[0];
    (void)sink_ns2;
    
    // Sort and compute median for B=10000
    std::sort(ns_times2, ns_times2 + ITERS);
    std::sort(eigen_times2, eigen_times2 + ITERS);
    double ns_median2 = ns_times2[ITERS / 2];
    double eigen_median2 = eigen_times2[ITERS / 2];
    
    // Print results for B=10000
    std::cout << "B=10000:" << std::endl;
    std::cout << "NSMatrix: " << ns_median2 << " ns/block" << std::endl;
    std::cout << "Eigen: " << eigen_median2 << " ns/block" << std::endl;
    std::cout << "Ratio (NS/Eigen): " << (ns_median2 / eigen_median2) << std::endl;
    
    // Third benchmark with B=100000
    const size_t B3 = 100000;
    NSBlockDiagMatrix A_ns3, B_ns3, C_ns3;
    if (!A_ns3.allocate(B3, k) || !B_ns3.allocate(B3, k) || !C_ns3.allocate(B3, k)) {
        std::cerr << "NS allocation failed for B=100000" << std::endl;
        return 1;
    }
    
    // Initialize with random values
    for (size_t i = 0; i < B3 * k * k; ++i) {
        A_ns3.blocks[i] = (float)rand() / RAND_MAX;
        B_ns3.blocks[i] = (float)rand() / RAND_MAX;
    }
    
    // Set bottom row of first 5 blocks to [0,0,0,1] pattern
    for (size_t b = 0; b < 5; ++b) {
        size_t block_offset = b * k * k;
        size_t bottom_row_offset = block_offset + 15 * k;
        A_ns3.blocks[bottom_row_offset + 12] = 0.0f;
        A_ns3.blocks[bottom_row_offset + 13] = 0.0f;
        A_ns3.blocks[bottom_row_offset + 14] = 0.0f;
        A_ns3.blocks[bottom_row_offset + 15] = 1.0f;
        B_ns3.blocks[bottom_row_offset + 12] = 0.0f;
        B_ns3.blocks[bottom_row_offset + 13] = 0.0f;
        B_ns3.blocks[bottom_row_offset + 14] = 0.0f;
        B_ns3.blocks[bottom_row_offset + 15] = 1.0f;
    }
    
    // Warmup + multi-iteration for Eigen (B=100000)
    for (int w = 0; w < WARMUP; ++w) {
        for (size_t b = 0; b < B3; ++b) {
            Eigen::Map<const Matrix16f> A_block(A_ns3.blocks + b * k * k);
            Eigen::Map<const Matrix16f> B_block(B_ns3.blocks + b * k * k);
            Eigen::Map<Matrix16f> C_block(C_ns3.blocks + b * k * k);
            C_block = A_block * B_block;
        }
    }
    double eigen_times3[ITERS];
    for (int i = 0; i < ITERS; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (size_t b = 0; b < B3; ++b) {
            Eigen::Map<const Matrix16f> A_block(A_ns3.blocks + b * k * k);
            Eigen::Map<const Matrix16f> B_block(B_ns3.blocks + b * k * k);
            Eigen::Map<Matrix16f> C_block(C_ns3.blocks + b * k * k);
            C_block = A_block * B_block;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        eigen_times3[i] = std::chrono::duration<double,std::nano>(t1-t0).count() / B3;
    }
    volatile float sink_eigen3 = C_ns3.blocks[0];
    (void)sink_eigen3;
    
    // Warmup + multi-iteration for NS (B=100000)
    for (int w = 0; w < WARMUP; ++w) {
        ns_blockdiag_mul(A_ns3, B_ns3, C_ns3);
    }
    double ns_times3[ITERS];
    for (int i = 0; i < ITERS; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        ns_blockdiag_mul(A_ns3, B_ns3, C_ns3);
        auto t1 = std::chrono::high_resolution_clock::now();
        ns_times3[i] = std::chrono::duration<double,std::nano>(t1-t0).count() / B3;
    }
    volatile float sink_ns3 = C_ns3.blocks[0];
    (void)sink_ns3;
    
    // Sort and compute median for B=100000
    std::sort(ns_times3, ns_times3 + ITERS);
    std::sort(eigen_times3, eigen_times3 + ITERS);
    double ns_median3 = ns_times3[ITERS / 2];
    double eigen_median3 = eigen_times3[ITERS / 2];
    
    // Print results for B=100000
    std::cout << "B=100000:" << std::endl;
    std::cout << "NSMatrix: " << ns_median3 << " ns/block" << std::endl;
    std::cout << "Eigen: " << eigen_median3 << " ns/block" << std::endl;
    std::cout << "Ratio (NS/Eigen): " << (ns_median3 / eigen_median3) << std::endl;

    // ------------------------------------------------------------------
    // Toeplitz extension: FFT matvec O(n log n) vs Eigen dense O(n^2)
    // ------------------------------------------------------------------
    std::cout << "\nToeplitz matvec (FFT vs Eigen dense):" << std::endl;
    std::cout.unsetf(std::ios::floatfield);
    std::cout << std::setprecision(4);
    const size_t tsizes[] = {256, 1024, 4096, 16384};
    for (size_t n : tsizes) {
        // Dense row-major Toeplitz for detection + Eigen reference
        size_t mbytes = (n * n * sizeof(float) + 63) & ~(size_t)63;
        float* M = (float*)aligned_alloc(64, mbytes);
        std::vector<float> tc(n), tr(n);
        for (size_t i = 0; i < n; ++i) {
            tc[i] = (float)rand() / RAND_MAX;
            tr[i] = (float)rand() / RAND_MAX;
        }
        tr[0] = tc[0];
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                M[i * n + j] = (i >= j) ? tc[i - j] : tr[j - i];

        bool is_toeplitz = ns_toeplitz_detect(M, n, 1e-5f);

        NSToeplitzMatrix T;
        if (!T.allocate(n)) {
            std::cerr << "Toeplitz allocation failed at n=" << n << std::endl;
            free(M);
            return 1;
        }
        ns_toeplitz_from_dense(M, n, T);
        auto p0 = std::chrono::high_resolution_clock::now();
        ns_toeplitz_prepare(T);
        auto p1 = std::chrono::high_resolution_clock::now();
        double prep_ms = std::chrono::duration<double, std::milli>(p1 - p0).count();

        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                       Eigen::RowMajor>> Me(M, n, n);
        Eigen::VectorXf x(n), y_ref(n), y_ns(n);
        for (size_t i = 0; i < n; ++i) x(i) = (float)rand() / RAND_MAX;
        y_ref.noalias() = Me * x;

        ns_toeplitz_matvec(T, x.data(), y_ns.data());
        double max_rel = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double e = std::abs((double)y_ns(i) - (double)y_ref(i));
            max_rel = std::max(max_rel, e / (std::abs((double)y_ref(i)) + 1e-6));
        }

        int iters = (n <= 4096) ? 50 : 10;
        int warm  = (n <= 4096) ? 5 : 2;

        for (int w = 0; w < warm; ++w) y_ref.noalias() = Me * x;
        std::vector<double> et(iters);
        for (int i = 0; i < iters; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            y_ref.noalias() = Me * x;
            auto t1 = std::chrono::high_resolution_clock::now();
            et[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        volatile float sink_e = y_ref(0);
        (void)sink_e;

        for (int w = 0; w < warm; ++w) ns_toeplitz_matvec(T, x.data(), y_ns.data());
        std::vector<double> nt(iters);
        for (int i = 0; i < iters; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            ns_toeplitz_matvec(T, x.data(), y_ns.data());
            auto t1 = std::chrono::high_resolution_clock::now();
            nt[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        volatile float sink_n = y_ns(0);
        (void)sink_n;

        std::sort(et.begin(), et.end());
        std::sort(nt.begin(), nt.end());
        double em = et[iters / 2], nm = nt[iters / 2];

        std::cout << "N=" << std::setw(6) << n
                  << " | detect: " << (is_toeplitz ? "TOEPLITZ" : "no")
                  << " | NS " << nm << " ms"
                  << " | Eigen " << em << " ms"
                  << " | speedup " << (em / nm) << "x"
                  << " | prep " << prep_ms << " ms"
                  << " | max_rel_err " << max_rel
                  << std::endl;
        free(M);
    }

    // Random dense: detection must refuse the fast path (expected no-win)
    {
        const size_t n = 1024;
        size_t mbytes = (n * n * sizeof(float) + 63) & ~(size_t)63;
        float* M = (float*)aligned_alloc(64, mbytes);
        for (size_t i = 0; i < n * n; ++i) M[i] = (float)rand() / RAND_MAX;
        bool is_t = ns_toeplitz_detect(M, n, 1e-5f);
        std::cout << "Random dense N=1024 detect: "
                  << (is_t ? "TOEPLITZ (BUG)"
                           : "not Toeplitz - no fast path (expected)")
                  << std::endl;
        free(M);
    }

    // ------------------------------------------------------------------
    // Large N: NS only. The dense matrix cannot be materialized:
    //   65536^2   = 17.2 GB  > Spectre RAM (16 GB)
    //   262144^2  = 274.9 GB
    //   1048576^2 = 4.4 TB
    // No Eigen comparison is possible - there is no dense matrix to hand
    // it. Structure is caller-declared; correctness is spot-checked with
    // direct O(n) dot products on a few output entries.
    // ------------------------------------------------------------------
    std::cout << "\nToeplitz matvec, NS only (dense unrepresentable):"
              << std::endl;
    const size_t bigsizes[] = {65536, 262144, 1048576};
    for (size_t n : bigsizes) {
        NSToeplitzMatrix T;
        if (!T.allocate(n)) {
            std::cerr << "Toeplitz allocation failed at n=" << n << std::endl;
            return 1;
        }
        for (size_t i = 0; i < n; ++i) {
            T.col[i] = (float)rand() / RAND_MAX;
            T.row[i] = (float)rand() / RAND_MAX;
        }
        T.row[0] = T.col[0];

        auto p0 = std::chrono::high_resolution_clock::now();
        ns_toeplitz_prepare(T);
        auto p1 = std::chrono::high_resolution_clock::now();
        double prep_ms = std::chrono::duration<double, std::milli>(p1 - p0).count();

        std::vector<float> x(n), y(n);
        for (size_t i = 0; i < n; ++i) x[i] = (float)rand() / RAND_MAX;

        ns_toeplitz_matvec(T, x.data(), y.data());

        // Spot-check 8 entries against direct O(n) dot products
        srand(12345);
        double max_rel = 0.0;
        for (int s = 0; s < 8; ++s) {
            size_t i = (size_t)rand() % n;
            double ref = 0.0;
            for (size_t j = 0; j <= i; ++j)
                ref += (double)T.col[i - j] * x[j];
            for (size_t j = i + 1; j < n; ++j)
                ref += (double)T.row[j - i] * x[j];
            double e = std::abs(ref - (double)y[i]) / (std::abs(ref) + 1e-6);
            max_rel = std::max(max_rel, e);
        }

        int iters = (n <= 65536) ? 20 : (n <= 262144) ? 10 : 5;
        for (int w = 0; w < 2; ++w) ns_toeplitz_matvec(T, x.data(), y.data());
        std::vector<double> nt(iters);
        for (int i = 0; i < iters; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            ns_toeplitz_matvec(T, x.data(), y.data());
            auto t1 = std::chrono::high_resolution_clock::now();
            nt[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        volatile float sink_b = y[0];
        (void)sink_b;
        std::sort(nt.begin(), nt.end());
        double nm = nt[iters / 2];

        double dense_bytes = (double)n * (double)n * sizeof(float);
        double ns_mb = (32.0 * T.m + 8.0 * n) / 1e6;
        std::cout << "N=" << std::setw(8) << n
                  << " | NS " << nm << " ms"
                  << " | Eigen n/a (dense = "
                  << (dense_bytes >= 1e12 ? dense_bytes / 1e12 : dense_bytes / 1e9)
                  << (dense_bytes >= 1e12 ? " TB" : " GB") << ")"
                  << " | NS footprint " << ns_mb << " MB"
                  << " | prep " << prep_ms << " ms"
                  << " | max_rel_err " << max_rel
                  << std::endl;
    }
#else
    // Benchmark NS (AVX-512 per block)
    auto start_ns = std::chrono::high_resolution_clock::now();
    ns_blockdiag_mul(A_ns, B_ns, C_ns);
    auto end_ns = std::chrono::high_resolution_clock::now();
    double time_ns = std::chrono::duration<double, std::nano>(end_ns - start_ns).count();
    double ns_per_block_ns = time_ns / B;

    // Print results
    std::cout << std::fixed << std::setprecision(2);
    std::cout << ns_per_block_ns << std::endl;
#endif

    std::cout << "AVX-512 path: " << (__builtin_cpu_supports("avx512f") ? "YES" : "NO") << std::endl;

#ifdef __AVX512F__
    std::cout << "AVX-512: YES" << std::endl;
#else
    std::cout << "AVX-512: NO (scalar)" << std::endl;
#endif

    return 0;
}

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

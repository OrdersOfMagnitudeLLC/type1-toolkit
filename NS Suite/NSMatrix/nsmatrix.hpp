// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#ifndef NSMATRIX_HPP
#define NSMATRIX_HPP

#include <immintrin.h>
#include <cstddef>
#include <cstdlib>

// Block-diagonal sparse matrix structure
// Non-zero values only in square blocks along the diagonal
struct NSBlockDiagMatrix {
    float* blocks;      // contiguous: block0[k*k], block1[k*k], ...
    size_t num_blocks;  // B blocks
    size_t block_size;  // k (each block is k×k)
    
    NSBlockDiagMatrix() : blocks(nullptr), num_blocks(0), block_size(0) {}
    
    ~NSBlockDiagMatrix() {
        if (blocks) free(blocks);
    }
    
    bool allocate(size_t B, size_t k) {
        num_blocks = B;
        block_size = k;
        size_t total_size = B * k * k;
        blocks = (float*)aligned_alloc(64, total_size * sizeof(float));
        return blocks != nullptr;
    }
};

// AVX-512 block multiplication for 16x16 blocks
#ifdef __AVX512F__
// Row-major: for each row i of A, broadcast each element and multiply by row of B
static void block_mul_avx512(const float* A_block, const float* B_block,
                             float* C_block) {
    for (size_t i = 0; i < 16; ++i) {
        __m512 c_row = _mm512_setzero_ps();
        const float* A_row = A_block + i * 16;
        
        for (size_t p = 0; p < 16; ++p) {
            __m512 a_elem = _mm512_set1_ps(A_row[p]);
            __m512 b_row = _mm512_loadu_ps(B_block + p * 16);
            c_row = _mm512_fmadd_ps(a_elem, b_row, c_row);
        }
        
        _mm512_storeu_ps(C_block + i * 16, c_row);
    }
}
#endif

// Scalar block multiplication
static void block_mul_scalar(const float* A_block, const float* B_block,
                             float* C_block, size_t k) {
    for (size_t i = 0; i < k; ++i) {
        for (size_t j = 0; j < k; ++j) {
            float sum = 0.0f;
            for (size_t p = 0; p < k; ++p) {
                sum += A_block[i * k + p] * B_block[p * k + j];
            }
            C_block[i * k + j] = sum;
        }
    }
}

// Block-diagonal matrix multiplication
// Multiplies block i of A by block i of B → block i of C
// Skips all off-diagonal computation entirely
void ns_blockdiag_mul(const NSBlockDiagMatrix& A,
                      const NSBlockDiagMatrix& B,
                      NSBlockDiagMatrix& C) {
    size_t k = A.block_size;
    size_t B_count = A.num_blocks;

    for (size_t b = 0; b < B_count; ++b) {
        const float* A_block = A.blocks + b * k * k;
        const float* B_block = B.blocks + b * k * k;
        float* C_block = C.blocks + b * k * k;

#ifdef __AVX512F__
        if (k == 16) {
            block_mul_avx512(A_block, B_block, C_block);
        } else {
            block_mul_scalar(A_block, B_block, C_block, k);
        }
#else
        block_mul_scalar(A_block, B_block, C_block, k);
#endif
    }
}

#endif // NSMATRIX_HPP

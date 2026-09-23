// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#ifndef NSGEMM_HPP
#define NSGEMM_HPP

#include <hwy/highway.h>
#include <immintrin.h>
#include <cstddef>
#include <cstdint>
#include <cstring>

// ---------------------------------------------------------------------------
// NSGemm: structure-declared matrix multiply.
//
// Real-world matrices have structure the caller already knows — banded,
// symmetric, degenerate GEMV shapes. MKL/Eigen have no mechanism to declare
// it, so they pay dense cost. NSGemm takes the declaration as a parameter
// and unlocks the win:
//
//   banded     O(n*k) matvec vs O(n^2) dense
//   symmetric  reads n^2/2 of A vs n^2 — memory-bound ~2x
//   gemv       tall-skinny (transformer weight x vector): fused FMA rows,
//              no output matrix materialization
//
// Loss: random dense — documented, same as the rest of the suite.
// ---------------------------------------------------------------------------

// y = A * x, A row-major n x n, nonzero only where |i-j| <= k.
// Strip-mine each row's contiguous band with AVX-512; O(n*k).
// Rows processed in tiles of 4: the x windows of adjacent rows overlap
// almost entirely, so one shared window stays hot in L1 across the tile.
// For wide bands (k > 64) the A stream is the bottleneck — prefetch the
// next tile's bands ahead of use.
static HWY_ATTR void ns_gemv_banded(const float* A, const float* x, float* y,
                           size_t n, size_t k) {
    const size_t TILE = 4;
    size_t i = 0;
    for (; i + TILE <= n; i += TILE) {
        size_t j0 = (i > k) ? i - k : 0;
        size_t j1 = (i + TILE - 1 + k + 1 < n) ? i + TILE - 1 + k + 1 : n;
#ifdef __AVX512F__
        if (k > 64) {
            // One prefetch per row band start; the hardware prefetcher
            // covers the contiguous rest. Cheap — 4 lines per tile.
            for (size_t r = 0; r < TILE; ++r) {
                size_t ir = i + r;
                size_t p0 = (ir > k) ? ir - k : 0;
                _mm_prefetch((const char*)(A + (ir + TILE) * n + p0),
                             _MM_HINT_T0);
            }
        }
#endif
        for (size_t r = 0; r < TILE; ++r) {
            size_t i_r = i + r;
            size_t r0 = (i_r > k) ? i_r - k : 0;
            size_t r1 = (i_r + k + 1 < n) ? i_r + k + 1 : n;
            const float* arow = A + i_r * n;
            float acc = 0.0f;
            size_t j = r0;
#ifdef __AVX512F__
            __m512 vacc = _mm512_setzero_ps();
            for (; j + 16 <= r1; j += 16)
                vacc = _mm512_fmadd_ps(_mm512_loadu_ps(arow + j),
                                       _mm512_loadu_ps(x + j), vacc);
            size_t rem = r1 - j;
            if (rem) {
                __mmask16 m = (__mmask16)((1u << rem) - 1u);
                vacc = _mm512_fmadd_ps(
                    _mm512_maskz_loadu_ps(m, arow + j),
                    _mm512_maskz_loadu_ps(m, x + j), vacc);
            }
            acc = _mm512_reduce_add_ps(vacc);
#else
            for (; j < r1; ++j) acc += arow[j] * x[j];
#endif
            y[i_r] = acc;
        }
    }
    for (; i < n; ++i) {
        size_t j0 = (i > k) ? i - k : 0;
        size_t j1 = (i + k + 1 < n) ? i + k + 1 : n;
        const float* arow = A + i * n;
        float acc = 0.0f;
        for (size_t j = j0; j < j1; ++j) acc += arow[j] * x[j];
        y[i] = acc;
    }
}

// y = A * x, A symmetric row-major n x n. Tiled SSYMV: the upper
// triangle is processed in 16x16 tiles. Each off-diagonal tile is loaded
// once and feeds both contributions in the same pass — the row dot goes
// to y[i-block], the broadcast-FMA column accumulation goes to
// y[j-block]. Diagonal tiles do row dots only (no reflection needed).
// Half the A reads of dense, and the tile stays in registers.
static HWY_ATTR void ns_gemv_symmetric(const float* A, const float* x, float* y,
                              size_t n) {
    memset(y, 0, n * sizeof(float));
    size_t nf = n & ~(size_t)15;  // full-tile extent
    size_t i = 0;
#ifdef __AVX512F__
    for (; i + 16 <= nf; i += 16) {
        // Per-row accumulators persist across all j-tiles: reduce_add
        // runs once per row per i-block, not once per tile.
        __m512 acci[16];
        for (size_t r = 0; r < 16; ++r) acci[r] = _mm512_setzero_ps();
        // Diagonal tile: full row dots, both halves counted once.
        {
            __m512 xv = _mm512_loadu_ps(x + i);
            for (size_t r = 0; r < 16; ++r) {
                __m512 a = _mm512_loadu_ps(A + (i + r) * n + i);
                acci[r] = _mm512_fmadd_ps(a, xv, acci[r]);
            }
        }
        // Off-diagonal tiles: one load feeds y[i] dot and y[j] reflect.
        for (size_t j = i + 16; j + 16 <= nf; j += 16) {
            __m512 xj = _mm512_loadu_ps(x + j);
            __m512 yj = _mm512_loadu_ps(y + j);
            for (size_t r = 0; r < 16; ++r) {
                __m512 a = _mm512_loadu_ps(A + (i + r) * n + j);
                yj = _mm512_fmadd_ps(a, _mm512_set1_ps(x[i + r]), yj);
                acci[r] = _mm512_fmadd_ps(a, xj, acci[r]);
            }
            _mm512_storeu_ps(y + j, yj);
        }
        for (size_t r = 0; r < 16; ++r)
            y[i + r] += _mm512_reduce_add_ps(acci[r]);
        // Right-edge partial tiles (columns nf..n).
        for (size_t r = 0; r < 16; ++r) {
            size_t ir = i + r;
            for (size_t j = nf; j < n; ++j) {
                float a = A[ir * n + j];
                y[ir] += a * x[j];
                y[j]  += a * x[ir];
            }
        }
    }
#endif
    // Scalar tail rows / non-AVX512 fallback.
    for (; i < n; ++i) {
        const float* arow = A + i * n;
        float xi = x[i];
        float acc = arow[i] * xi;
        for (size_t j = i + 1; j < n; ++j) {
            float a = arow[j];
            acc += a * x[j];
            y[j] += a * xi;
        }
        y[i] += acc;
    }
}

// y = W * x, W row-major m x n (transformer weight shape: the "GEMM" is
// really a GEMV — B is n x 1). Fused load+FMA per row, 4-row unroll,
// no output matrix allocated. O(m*n) but at FMA throughput, not GEMM
// blocking overhead.
static HWY_ATTR void ns_gemv_weight(const float* W, const float* x, float* y,
                           size_t m, size_t n) {
    size_t i = 0;
#ifdef __AVX512F__
    for (; i + 4 <= m; i += 4) {
        const float* r0 = W + i * n;
        const float* r1 = r0 + n;
        const float* r2 = r1 + n;
        const float* r3 = r2 + n;
        __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
        __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
        size_t j = 0;
        for (; j + 16 <= n; j += 16) {
            __m512 xv = _mm512_loadu_ps(x + j);
            a0 = _mm512_fmadd_ps(_mm512_loadu_ps(r0 + j), xv, a0);
            a1 = _mm512_fmadd_ps(_mm512_loadu_ps(r1 + j), xv, a1);
            a2 = _mm512_fmadd_ps(_mm512_loadu_ps(r2 + j), xv, a2);
            a3 = _mm512_fmadd_ps(_mm512_loadu_ps(r3 + j), xv, a3);
        }
        if (j < n) {
            __mmask16 msk = (__mmask16)((1u << (n - j)) - 1u);
            __m512 xv = _mm512_maskz_loadu_ps(msk, x + j);
            a0 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(msk, r0 + j), xv, a0);
            a1 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(msk, r1 + j), xv, a1);
            a2 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(msk, r2 + j), xv, a2);
            a3 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(msk, r3 + j), xv, a3);
        }
        y[i]     = _mm512_reduce_add_ps(a0);
        y[i + 1] = _mm512_reduce_add_ps(a1);
        y[i + 2] = _mm512_reduce_add_ps(a2);
        y[i + 3] = _mm512_reduce_add_ps(a3);
    }
#endif
    for (; i < m; ++i) {
        const float* row = W + i * n;
        float acc = 0.0f;
        size_t j = 0;
#ifdef __AVX512F__
        __m512 vacc = _mm512_setzero_ps();
        for (; j + 16 <= n; j += 16)
            vacc = _mm512_fmadd_ps(_mm512_loadu_ps(row + j),
                                   _mm512_loadu_ps(x + j), vacc);
        if (j < n) {
            __mmask16 msk = (__mmask16)((1u << (n - j)) - 1u);
            vacc = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(msk, row + j),
                                   _mm512_maskz_loadu_ps(msk, x + j), vacc);
        }
        acc = _mm512_reduce_add_ps(vacc);
#else
        for (; j < n; ++j) acc += row[j] * x[j];
#endif
        y[i] = acc;
    }
}

// Y = W * X, W row-major m x n, X is B row-major n-vectors (batch of
// transformer inputs, B <= 8), Y is B row-major m-vectors.
// One W row load feeds all B accumulators — W traffic amortized Bx.
// This is where the declared shape wins: Eigen/MKL run a skinny GEMM and
// still stream W once per... they stream W once total but pay GEMM
// blocking overhead for B=4/8; we fuse directly.
static HWY_ATTR void ns_gemm_weight_batch(const float* W, const float* X,
                                 float* Y, size_t m, size_t n, size_t B) {
#ifdef __AVX512F__
    for (size_t i = 0; i < m; ++i) {
        const float* wrow = W + i * n;
        __m512 acc[8];
        for (size_t b = 0; b < B; ++b) acc[b] = _mm512_setzero_ps();
        size_t j = 0;
        for (; j + 16 <= n; j += 16) {
            __m512 w = _mm512_loadu_ps(wrow + j);
            for (size_t b = 0; b < B; ++b)
                acc[b] = _mm512_fmadd_ps(w, _mm512_loadu_ps(X + b * n + j),
                                         acc[b]);
        }
        if (j < n) {
            __mmask16 msk = (__mmask16)((1u << (n - j)) - 1u);
            __m512 w = _mm512_maskz_loadu_ps(msk, wrow + j);
            for (size_t b = 0; b < B; ++b)
                acc[b] = _mm512_fmadd_ps(
                    w, _mm512_maskz_loadu_ps(msk, X + b * n + j), acc[b]);
        }
        for (size_t b = 0; b < B; ++b)
            Y[b * m + i] = _mm512_reduce_add_ps(acc[b]);
    }
#else
    for (size_t i = 0; i < m; ++i) {
        const float* wrow = W + i * n;
        for (size_t b = 0; b < B; ++b) {
            float s = 0.0f;
            for (size_t j = 0; j < n; ++j) s += wrow[j] * X[b * n + j];
            Y[b * m + i] = s;
        }
    }
#endif
}

#endif // NSGEMM_HPP

// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#ifndef NSMATRIX_HPP
#define NSMATRIX_HPP

#include <hwy/highway.h>
#include <immintrin.h>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstring>

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
static HWY_ATTR void block_mul_avx512(const float* A_block, const float* B_block,
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

// ---------------------------------------------------------------------------
// Toeplitz extension: T[i][j] = t[i-j], constant along diagonals.
// Caller declares (or detects) Toeplitz structure -> matvec becomes a
// convolution computable in O(n log n) via FFT, vs O(n^2) dense matvec.
// Eigen has no Toeplitz type; this is the declare-structure-unlock-win pattern.
// T*X (matmul) is per-column matvec.
// ---------------------------------------------------------------------------

static inline size_t ns_next_pow2(size_t x) {
    size_t p = 1;
    while (p < x) p <<= 1;
    return p;
}

static inline float* ns_falloc(size_t count) {
    size_t bytes = (count * sizeof(float) + 63) & ~(size_t)63;
    return (float*)aligned_alloc(64, bytes);
}

// Toeplitz matrix stored as first column + first row (2n-1 dof).
// prepare() precomputes the FFT of the circulant embedding so matvec
// costs one forward FFT + pointwise multiply + one inverse FFT.
struct NSToeplitzMatrix {
    size_t n;        // matrix dimension
    size_t m;        // FFT size: smallest pow2 >= 2n-1
    float* col;      // first column  t[0..n-1]
    float* row;      // first row     t[0..-(n-1)]  (row[0] == col[0])
    float* spec_re;  // FFT of circulant embedding, m values
    float* spec_im;
    float* tw_re;    // forward twiddle table, m entries
    float* tw_im;
    size_t* rev;     // bit-reversal permutation, m entries
    float* buf_re;   // matvec workspace, m floats
    float* buf_im;

    NSToeplitzMatrix() : n(0), m(0), col(nullptr), row(nullptr),
        spec_re(nullptr), spec_im(nullptr), tw_re(nullptr), tw_im(nullptr),
        rev(nullptr), buf_re(nullptr), buf_im(nullptr) {}

    ~NSToeplitzMatrix() { release(); }

    NSToeplitzMatrix(const NSToeplitzMatrix&) = delete;
    NSToeplitzMatrix& operator=(const NSToeplitzMatrix&) = delete;

    void release() {
        free(col); free(row); free(spec_re); free(spec_im);
        free(tw_re); free(tw_im); free(rev); free(buf_re); free(buf_im);
        col = row = spec_re = spec_im = tw_re = tw_im = buf_re = buf_im = nullptr;
        rev = nullptr;
        n = m = 0;
    }

    bool allocate(size_t n_) {
        release();
        n = n_;
        m = ns_next_pow2(2 * n - 1);
        col     = ns_falloc(n);
        row     = ns_falloc(n);
        spec_re = ns_falloc(m);
        spec_im = ns_falloc(m);
        tw_re   = ns_falloc(m);
        tw_im   = ns_falloc(m);
        buf_re  = ns_falloc(m);
        buf_im  = ns_falloc(m);
        size_t rbytes = (m * sizeof(size_t) + 63) & ~(size_t)63;
        rev = (size_t*)aligned_alloc(64, rbytes);
        return col && row && spec_re && spec_im && tw_re && tw_im &&
               buf_re && buf_im && rev;
    }
};

// Detect Toeplitz structure in a row-major dense matrix: every diagonal
// constant within tol. O(n^2) one-time check that gates the fast path.
static bool ns_toeplitz_detect(const float* M, size_t n, float tol) {
    for (size_t i = 1; i < n; ++i)
        for (size_t j = 1; j < n; ++j)
            if (fabsf(M[i * n + j] - M[(i - 1) * n + (j - 1)]) > tol)
                return false;
    return true;
}

// Extract first column / first row from a row-major dense matrix.
static void ns_toeplitz_from_dense(const float* M, size_t n,
                                   NSToeplitzMatrix& T) {
    for (size_t i = 0; i < n; ++i) T.col[i] = M[i * n];
    for (size_t j = 0; j < n; ++j) T.row[j] = M[j];
}

// Iterative radix-2 FFT on split re/im arrays (SoA).
// Twiddles for stage len (half = len/2) live at offset half-1:
// tw[half-1+j] = exp(-2*pi*i*j/len). AVX-512 butterflies when half >= 16.
static HWY_ATTR void ns_fft(float* re, float* im, size_t n,
                   const size_t* rev, const float* tw_re, const float* tw_im,
                   bool inverse) {
    for (size_t i = 0; i < n; ++i) {
        size_t r = rev[i];
        if (r > i) {
            float t;
            t = re[i]; re[i] = re[r]; re[r] = t;
            t = im[i]; im[i] = im[r]; im[r] = t;
        }
    }
    const float sign = inverse ? -1.0f : 1.0f;
    for (size_t len = 2; len <= n; len <<= 1) {
        const size_t half = len >> 1;
        const float* wre = tw_re + (half - 1);
        const float* wim = tw_im + (half - 1);
        for (size_t i = 0; i < n; i += len) {
            size_t j = 0;
#ifdef __AVX512F__
            if (half >= 16) {
                const __m512 vs = _mm512_set1_ps(sign);
                for (; j + 16 <= half; j += 16) {
                    __m512 wr = _mm512_loadu_ps(wre + j);
                    __m512 wi = _mm512_mul_ps(_mm512_loadu_ps(wim + j), vs);
                    __m512 br = _mm512_loadu_ps(re + i + j + half);
                    __m512 bi = _mm512_loadu_ps(im + i + j + half);
                    __m512 vr = _mm512_fmsub_ps(br, wr, _mm512_mul_ps(bi, wi));
                    __m512 vi = _mm512_fmadd_ps(br, wi, _mm512_mul_ps(bi, wr));
                    __m512 ur = _mm512_loadu_ps(re + i + j);
                    __m512 ui = _mm512_loadu_ps(im + i + j);
                    _mm512_storeu_ps(re + i + j, _mm512_add_ps(ur, vr));
                    _mm512_storeu_ps(im + i + j, _mm512_add_ps(ui, vi));
                    _mm512_storeu_ps(re + i + j + half, _mm512_sub_ps(ur, vr));
                    _mm512_storeu_ps(im + i + j + half, _mm512_sub_ps(ui, vi));
                }
            }
#endif
            for (; j < half; ++j) {
                float wr = wre[j], wi = wim[j] * sign;
                float br = re[i + j + half], bi = im[i + j + half];
                float vr = br * wr - bi * wi;
                float vi = br * wi + bi * wr;
                float ur = re[i + j], ui = im[i + j];
                re[i + j]         = ur + vr;
                im[i + j]         = ui + vi;
                re[i + j + half]  = ur - vr;
                im[i + j + half]  = ui - vi;
            }
        }
    }
}

// Build bit-reversal table, twiddles, and the FFT of the circulant
// embedding. One-time cost O(m log m); amortized across matvecs.
static void ns_toeplitz_prepare(NSToeplitzMatrix& T) {
    const size_t m = T.m, n = T.n;
    const double PI = 3.14159265358979323846;

    unsigned lg = 0;
    while (((size_t)1 << lg) < m) ++lg;
    T.rev[0] = 0;
    for (size_t i = 1; i < m; ++i)
        T.rev[i] = (T.rev[i >> 1] >> 1) | ((i & 1) << (lg - 1));

    for (size_t len = 2; len <= m; len <<= 1) {
        size_t half = len >> 1;
        for (size_t j = 0; j < half; ++j) {
            double ang = -2.0 * PI * (double)j / (double)len;
            T.tw_re[half - 1 + j] = (float)cos(ang);
            T.tw_im[half - 1 + j] = (float)sin(ang);
        }
    }

    // Circulant embedding: c[k] = t[k] for k < n, c[m-k] = t[-k] for
    // k = 1..n-1, zeros in the gap. Then C*x = IFFT(FFT(c) .* FFT(x)).
    memset(T.spec_re, 0, m * sizeof(float));
    memset(T.spec_im, 0, m * sizeof(float));
    T.spec_re[0] = T.col[0];
    for (size_t k = 1; k < n; ++k) T.spec_re[k] = T.col[k];
    for (size_t k = 1; k < n; ++k) T.spec_re[m - k] = T.row[k];
    ns_fft(T.spec_re, T.spec_im, m, T.rev, T.tw_re, T.tw_im, false);
}

// y = T * x in O(m log m). T must be prepared. Reuses T's workspace,
// so T is non-const; not thread-safe on a shared T.
static HWY_ATTR void ns_toeplitz_matvec(NSToeplitzMatrix& T, const float* x, float* y) {
    const size_t m = T.m, n = T.n;
    float* re = T.buf_re;
    float* im = T.buf_im;

    memset(re, 0, m * sizeof(float));
    memset(im, 0, m * sizeof(float));
    memcpy(re, x, n * sizeof(float));

    ns_fft(re, im, m, T.rev, T.tw_re, T.tw_im, false);

    // Pointwise complex multiply by the circulant spectrum.
    size_t i = 0;
#ifdef __AVX512F__
    for (; i + 16 <= m; i += 16) {
        __m512 ar = _mm512_loadu_ps(re + i);
        __m512 ai = _mm512_loadu_ps(im + i);
        __m512 br = _mm512_loadu_ps(T.spec_re + i);
        __m512 bi = _mm512_loadu_ps(T.spec_im + i);
        _mm512_storeu_ps(re + i, _mm512_fmsub_ps(ar, br, _mm512_mul_ps(ai, bi)));
        _mm512_storeu_ps(im + i, _mm512_fmadd_ps(ar, bi, _mm512_mul_ps(ai, br)));
    }
#endif
    for (; i < m; ++i) {
        float ar = re[i], ai = im[i];
        re[i] = ar * T.spec_re[i] - ai * T.spec_im[i];
        im[i] = ar * T.spec_im[i] + ai * T.spec_re[i];
    }

    ns_fft(re, im, m, T.rev, T.tw_re, T.tw_im, true);

    const float inv_m = 1.0f / (float)m;
    size_t k = 0;
#ifdef __AVX512F__
    __m512 vinv = _mm512_set1_ps(inv_m);
    for (; k + 16 <= n; k += 16)
        _mm512_storeu_ps(y + k, _mm512_mul_ps(_mm512_loadu_ps(re + k), vinv));
#endif
    for (; k < n; ++k) y[k] = re[k] * inv_m;
}

// Y = T * X for row-major X (n x ncols): per-column matvec.
static void ns_toeplitz_matmul(NSToeplitzMatrix& T, const float* X,
                               float* Y, size_t ncols) {
    for (size_t c = 0; c < ncols; ++c)
        ns_toeplitz_matvec(T, X + c * T.n, Y + c * T.n);
}

#endif // NSMATRIX_HPP

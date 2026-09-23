// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

// NSGemm benchmark: structure-declared matvec vs dense kings.
//   1. banded     — O(n*k) vs Eigen dense matvec O(n^2)
//   2. symmetric  — half the matrix reads vs Eigen dense matvec
//   3. weight     — transformer GEMV (m x n @ n) vs Eigen / MKL SGEMM
// Loss case: random dense — documented.

#include "nsgemm.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <random>

#ifdef USE_EIGEN
#include <Eigen/Dense>
#endif
#ifdef USE_MKL
#include <mkl_cblas.h>
#endif

static float* falloc(size_t n) {
    return (float*)aligned_alloc(64, (n * sizeof(float) + 63) & ~(size_t)63);
}

static double now_s() {
    return std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

static float max_rel_err(const float* a, const float* b, size_t n) {
    float worst = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float d = fabsf(a[i] - b[i]);
        float r = d / (fabsf(b[i]) + 1e-6f);
        if (r > worst) worst = r;
    }
    return worst;
}

int main() {
    printf("NSGemm benchmark — declared structure vs dense kings\n\n");
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> uf(-1.0f, 1.0f);

    // ---- 1. Banded matvec ------------------------------------------------
    printf("== Banded matvec (nonzero within k diagonals) ==\n");
    {
        const size_t sizes[] = {1024, 4096, 16384};
        const size_t ks[]    = {8, 32, 128};
        for (size_t n : sizes) for (size_t k : ks) {
            float* A = falloc(n * n);
            float* x = falloc(n);
            float* y = falloc(n);
            float* yr = falloc(n);
            memset(A, 0, n * n * sizeof(float));
            for (size_t i = 0; i < n; ++i) {
                size_t j0 = i > k ? i - k : 0;
                size_t j1 = i + k + 1 < n ? i + k + 1 : n;
                for (size_t j = j0; j < j1; ++j) A[i * n + j] = uf(rng);
                x[i] = uf(rng);
            }

            // reference: dense scalar matvec
            for (size_t i = 0; i < n; ++i) {
                float s = 0;
                for (size_t j = 0; j < n; ++j) s += A[i * n + j] * x[j];
                yr[i] = s;
            }
            ns_gemv_banded(A, x, y, n, k);
            float err = max_rel_err(y, yr, n);

            size_t iters = (size_t)(2e9 / (double)(n * (2 * k + 1))) + 1;
            double t0 = now_s();
            for (size_t it = 0; it < iters; ++it)
                ns_gemv_banded(A, x, y, n, k);
            double ns_ms = (now_s() - t0) / iters * 1e3;

            double kg_ms = -1.0;
#ifdef USE_EIGEN
            {
                Eigen::Map<Eigen::MatrixXf> EA(A, n, n);
                Eigen::Map<Eigen::VectorXf> Ex(x, n), Ey(yr, n);
                size_t eit = (size_t)(2e9 / (double)(n * n)) + 1;
                t0 = now_s();
                for (size_t it = 0; it < eit; ++it) Ey = EA * Ex;
                kg_ms = (now_s() - t0) / eit * 1e3;
            }
#endif
            printf("n=%5zu k=%3zu  ns=%9.3f ms  eigen=%9.3f ms  "
                   "speedup=%7.1fx  err=%.2e\n",
                   n, k, ns_ms, kg_ms,
                   kg_ms > 0 ? kg_ms / ns_ms : 0.0, err);
            free(A); free(x); free(y); free(yr);
        }
    }

    // ---- 2. Symmetric matvec ----------------------------------------------
    printf("\n== Symmetric matvec (upper triangle only) ==\n");
    {
        const size_t sizes[] = {1024, 4096, 16384};
        for (size_t n : sizes) {
            float* A = falloc(n * n);
            float* x = falloc(n);
            float* y = falloc(n);
            float* yr = falloc(n);
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = i; j < n; ++j) {
                    float v = uf(rng);
                    A[i * n + j] = v;
                    A[j * n + i] = v;
                }
                x[i] = uf(rng);
            }

            for (size_t i = 0; i < n; ++i) {
                float s = 0;
                for (size_t j = 0; j < n; ++j) s += A[i * n + j] * x[j];
                yr[i] = s;
            }
            ns_gemv_symmetric(A, x, y, n);
            float err = max_rel_err(y, yr, n);

            size_t iters = (size_t)(2e9 / (double)(n * n)) + 1;
            double t0 = now_s();
            for (size_t it = 0; it < iters; ++it)
                ns_gemv_symmetric(A, x, y, n);
            double ns_ms = (now_s() - t0) / iters * 1e3;

            double kg_ms = -1.0;
#ifdef USE_EIGEN
            {
                Eigen::Map<Eigen::MatrixXf> EA(A, n, n);
                Eigen::Map<Eigen::VectorXf> Ex(x, n), Ey(yr, n);
                t0 = now_s();
                for (size_t it = 0; it < iters; ++it) Ey = EA * Ex;
                kg_ms = (now_s() - t0) / iters * 1e3;
            }
#endif
            printf("n=%5zu  ns=%9.3f ms  eigen=%9.3f ms  "
                   "speedup=%6.2fx  err=%.2e\n",
                   n, ns_ms, kg_ms,
                   kg_ms > 0 ? kg_ms / ns_ms : 0.0, err);
            free(A); free(x); free(y); free(yr);
        }
    }

    // ---- 3. Transformer weight GEMV / skinny GEMM --------------------------
    printf("\n== Weight GEMV/GEMM (m x n @ n x B, transformer inference) ==\n");
    {
        const size_t shapes[][2] = {{4096, 4096}, {11008, 4096}, {4096, 11008}};
        const size_t batches[] = {1, 4, 8};
        for (auto& s : shapes) for (size_t B : batches) {
            size_t m = s[0], n = s[1];
            float* W = falloc(m * n);
            float* X = falloc(B * n);
            float* Y = falloc(B * m);
            float* Yr = falloc(B * m);
            for (size_t i = 0; i < m * n; ++i) W[i] = uf(rng);
            for (size_t i = 0; i < B * n; ++i) X[i] = uf(rng);

            for (size_t b = 0; b < B; ++b)
                for (size_t i = 0; i < m; ++i) {
                    float acc = 0;
                    for (size_t j = 0; j < n; ++j)
                        acc += W[i * n + j] * X[b * n + j];
                    Yr[b * m + i] = acc;
                }
            if (B == 1) ns_gemv_weight(W, X, Y, m, n);
            else        ns_gemm_weight_batch(W, X, Y, m, n, B);
            float err = max_rel_err(Y, Yr, B * m);

            size_t iters = (size_t)(4e9 / (double)(m * n * B)) + 1;
            double t0 = now_s();
            for (size_t it = 0; it < iters; ++it) {
                if (B == 1) ns_gemv_weight(W, X, Y, m, n);
                else        ns_gemm_weight_batch(W, X, Y, m, n, B);
            }
            double ns_ms = (now_s() - t0) / iters * 1e3;

            double kg_ms = -1.0;
            const char* king = "none";
#ifdef USE_MKL
            king = "mkl";
            {
                t0 = now_s();
                for (size_t it = 0; it < iters; ++it)
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                m, B, n, 1.0f, W, n, X, B, 0.0f, Yr, B);
                kg_ms = (now_s() - t0) / iters * 1e3;
            }
#elif defined(USE_EIGEN)
            king = "eigen";
            {
                Eigen::Map<Eigen::MatrixXf> EW(W, m, n);
                // Row-major B x n X is column-major n x B — direct map.
                Eigen::Map<Eigen::MatrixXf> EX(X, n, B);
                Eigen::MatrixXf EY(m, B);
                t0 = now_s();
                for (size_t it = 0; it < iters; ++it) EY = EW * EX;
                kg_ms = (now_s() - t0) / iters * 1e3;
            }
#endif
            printf("m=%5zu n=%5zu B=%zu  ns=%8.3f ms  %s=%8.3f ms  "
                   "speedup=%6.2fx  err=%.2e\n",
                   m, n, B, ns_ms, king, kg_ms,
                   kg_ms > 0 ? kg_ms / ns_ms : 0.0, err);
            free(W); free(X); free(Y); free(Yr);
        }
    }

    printf("\nLoss case: random dense n x n matmul — no declared structure,\n");
    printf("no win. Use MKL/Eigen. Same pattern as the rest of the suite.\n");
    return 0;
}

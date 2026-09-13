#pragma once

#include <stdint.h>
#include <string.h>
#include <math.h>

#include "../vendor/ns_dequant.h"

#include <immintrin.h>

// (a) block_q8_K -- now defined in vendor/ns_dequant.h (shared with
// ns_llama_gemv.cpp so both AVX-512 and AVX2-only TUs use the same type).

#if defined(__AVX512F__) && defined(__AVX512BW__)
// (b) quantize_row_q8k -- AVX-512 quantization of one activation vector.
// Mirrors quantize_row_q8_K_ref() semantics (ggml-quants.c:2768-2805):
// signed max (not just amax) determines sign of iscale, per-16 bsums.
static inline void quantize_row_q8k(const float* src, block_q8_K* dst, int nb) {
    for (int i = 0; i < nb; i++) {
        const float* x = src + (size_t)i * QK_K;
        block_q8_K* y = dst + i;

        __m512 vamax = _mm512_setzero_ps();
        for (int j = 0; j < QK_K; j += 16) {
            __m512 v = _mm512_loadu_ps(x + j);
            vamax = _mm512_max_ps(vamax, _mm512_abs_ps(v));
        }
        float amax = _mm512_reduce_max_ps(vamax);

        if (amax == 0.0f) {
            y->d = 0.0f;
            memset(y->qs, 0, QK_K);
            memset(y->bsums, 0, sizeof(y->bsums));
            continue;
        }

        const float iscale = 127.0f / amax;
        const __m512 viscale = _mm512_set1_ps(iscale);

        for (int j = 0; j < QK_K; j += 16) {
            __m512 v = _mm512_loadu_ps(x + j);
            __m512 scaled = _mm512_mul_ps(v, viscale);
            __m512i vi32 = _mm512_cvtps_epi32(scaled);
            __m128i vi8  = _mm512_cvtsepi32_epi8(vi32);
            _mm_storeu_si128((__m128i*)(y->qs + j), vi8);
        }

        for (int g = 0; g < QK_K/16; g++) {
            int32_t sum = 0;
            for (int l = 0; l < 16; l++) sum += y->qs[g*16 + l];
            y->bsums[g] = (int16_t)sum;
        }

        y->d = amax / 127.0f;
    }
}

// AVX-512 upgrade: one call now covers TWO of the original 32-byte scale
// groups (g and g+1) in a single 64-byte load, using _mm512_maddubs_epi16
// over the combined 64-byte q4 span. Since group g and group g+1 carry
// different scales, the signature is widened to 4 scales (sc0/sc1 for
// group g's lo/hi nibble, sc0b/sc1b for group g+1's lo/hi nibble). The
// 512-bit maddubs result is split back into its two 256-bit halves
// (low = group g, high = group g+1) before applying the per-group scale,
// since maddubs/madd are lane-local and never mix across the two halves.
static inline void dot64_both_nibbles(
    const uint8_t* qb,
    const int8_t*  q8_0,
    const int8_t*  q8_1,
    __m256*        acc,
    uint8_t        sc0,
    uint8_t        sc1,
    uint8_t        sc0b,
    uint8_t        sc1b)
{
    const __m512i m4 = _mm512_set1_epi8(0x0F);

    // 64 q4 bytes: bytes[0..31] = group g, bytes[32..63] = group g+1
    __m512i bytes = _mm512_loadu_si512((const void*)qb);
    __m512i lo4   = _mm512_and_si512(bytes, m4);
    __m512i hi4   = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), m4);

    // q8_0/q8_1 point at group g's lo/hi; group g+1's lo/hi sit 64 bytes
    // further into the same activation buffer (see call-site addressing).
    __m256i q8v0_g  = _mm256_loadu_si256((const __m256i*)q8_0);
    __m256i q8v0_g1 = _mm256_loadu_si256((const __m256i*)(q8_0 + 64));
    __m256i q8v1_g  = _mm256_loadu_si256((const __m256i*)q8_1);
    __m256i q8v1_g1 = _mm256_loadu_si256((const __m256i*)(q8_1 + 64));

    __m512i q8v0 = _mm512_inserti64x4(_mm512_castsi256_si512(q8v0_g), q8v0_g1, 1);
    __m512i q8v1 = _mm512_inserti64x4(_mm512_castsi256_si512(q8v1_g), q8v1_g1, 1);

    __m512i p0 = _mm512_maddubs_epi16(lo4, q8v0);
    __m512i p1 = _mm512_maddubs_epi16(hi4, q8v1);

    const __m512i ones16 = _mm512_set1_epi16(1);
    __m512i s0 = _mm512_madd_epi16(p0, ones16);
    __m512i s1 = _mm512_madd_epi16(p1, ones16);

    __m256i s0_lo = _mm512_castsi512_si256(s0);          // group g,  lo
    __m256i s0_hi = _mm512_extracti64x4_epi64(s0, 1);    // group g+1, lo
    __m256i s1_lo = _mm512_castsi512_si256(s1);          // group g,  hi
    __m256i s1_hi = _mm512_extracti64x4_epi64(s1, 1);    // group g+1, hi

    *acc = _mm256_add_ps(*acc, _mm256_mul_ps(_mm256_cvtepi32_ps(s0_lo), _mm256_set1_ps((float)sc0)));
    *acc = _mm256_add_ps(*acc, _mm256_mul_ps(_mm256_cvtepi32_ps(s1_lo), _mm256_set1_ps((float)sc1)));
    *acc = _mm256_add_ps(*acc, _mm256_mul_ps(_mm256_cvtepi32_ps(s0_hi), _mm256_set1_ps((float)sc0b)));
    *acc = _mm256_add_ps(*acc, _mm256_mul_ps(_mm256_cvtepi32_ps(s1_hi), _mm256_set1_ps((float)sc1b)));
}

// (c) vec_dot_q4k_q8k -- fused integer dot, one block_q4_K row-block against
// one block_q8_K activation-block at a time. Scale/min unpack via
// get_scale_min_k4() (verbatim, ns_dequant.h), applied per 32-element
// sub-block (is=0..7), exactly matching dequantize_row_q4_K's byte layout:
// group g in [0,4), q4 bytes [g*32, g*32+32) -> lower nibble = elements
// [g*64, g*64+32), upper nibble = elements [g*64+32, g*64+64).
static inline float vec_dot_q4k_q8k(const block_q4_K* vx, const block_q8_K* vy, int nb) {
    float sumf = 0.0f;
    __m256 acc[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(),
                      _mm256_setzero_ps(), _mm256_setzero_ps() };
    float minsum[4] = { 0.f, 0.f, 0.f, 0.f };
    float ds[4], dmin[4], dy[4];

    auto hsum_float_8 = [](__m256 v) -> float {
        __m128 t = _mm_add_ps(_mm256_castps256_ps128(v),
                              _mm256_extractf128_ps(v, 1));
        t = _mm_add_ps(t, _mm_movehl_ps(t, t));
        t = _mm_add_ps(t, _mm_movehdup_ps(t));
        return _mm_cvtss_f32(t);
    };

    int i = 0;
    for (; i + 4 <= nb; i += 4) {
        __builtin_prefetch(vx + i + 1, 0, 1);
        __builtin_prefetch(vy + i + 1, 0, 1);
        __builtin_prefetch(vx + i + 2, 0, 1);
        __builtin_prefetch(vy + i + 2, 0, 1);
        __builtin_prefetch(vx + i + 3, 0, 1);
        __builtin_prefetch(vy + i + 3, 0, 1);

        for (int j = 0; j < 4; ++j) {
            acc[j]    = _mm256_setzero_ps();
            minsum[j] = 0.f;

            const block_q4_K* x = vx + i + j;
            const block_q8_K* y = vy + i + j;
            ds[j]    = GGML_FP16_TO_FP32(x->d);
            dmin[j]  = GGML_FP16_TO_FP32(x->dmin);
            dy[j]    = y->d;

            const uint8_t* q4 = x->qs;
            const int8_t*  q8 = y->qs;
            const int16_t* bsums_y = y->bsums;

            int is = 0;
            for (int g = 0; g < QK_K/64; g += 2) {
                uint8_t sc0, m0, sc1, m1, sc0b, m0b, sc1b, m1b;
                get_scale_min_k4(is,     x->scales, &sc0,  &m0);
                get_scale_min_k4(is + 1, x->scales, &sc1,  &m1);
                get_scale_min_k4(is + 2, x->scales, &sc0b, &m0b);
                get_scale_min_k4(is + 3, x->scales, &sc1b, &m1b);
                is += 4;

                dot64_both_nibbles(
                    q4 + g*32,
                    q8 + g*64,
                    q8 + g*64 + 32,
                    &acc[j],
                    sc0, sc1, sc0b, sc1b);

                minsum[j] += dmin[j] * ((float)m0  * (float)(bsums_y[g*4+0]     + bsums_y[g*4+1]) +
                                        (float)m1  * (float)(bsums_y[g*4+2]     + bsums_y[g*4+3]) +
                                        (float)m0b * (float)(bsums_y[(g+1)*4+0] + bsums_y[(g+1)*4+1]) +
                                        (float)m1b * (float)(bsums_y[(g+1)*4+2] + bsums_y[(g+1)*4+3]));
            }
        }

        for (int j = 0; j < 4; ++j) {
            float block_result = hsum_float_8(acc[j]);
            sumf += dy[j] * (ds[j] * block_result - minsum[j]);
        }
    }

    // Scalar tail for the remaining 1-3 blocks
    for (; i < nb; ++i) {
        __builtin_prefetch(vx + i + 1, 0, 1);
        const block_q4_K* x = vx + i;
        const block_q8_K* y = vy + i;

        const float d    = GGML_FP16_TO_FP32(x->d);
        const float dm   = GGML_FP16_TO_FP32(x->dmin);
        const float dys  = y->d;

        const uint8_t* q4 = x->qs;
        const int8_t*  q8 = y->qs;
        const int16_t* bsums_y = y->bsums;

        __m256 fp32_acc = _mm256_setzero_ps();
        float  minsum_f = 0.f;
        int is = 0;

        for (int g = 0; g < QK_K/64; g += 2) {
            uint8_t sc0, m0, sc1, m1, sc0b, m0b, sc1b, m1b;
            get_scale_min_k4(is,     x->scales, &sc0,  &m0);
            get_scale_min_k4(is + 1, x->scales, &sc1,  &m1);
            get_scale_min_k4(is + 2, x->scales, &sc0b, &m0b);
            get_scale_min_k4(is + 3, x->scales, &sc1b, &m1b);
            is += 4;

            dot64_both_nibbles(
                q4 + g*32,
                q8 + g*64,
                q8 + g*64 + 32,
                &fp32_acc,
                sc0, sc1, sc0b, sc1b);

            minsum_f += dm * ((float)m0  * (float)(bsums_y[g*4+0]     + bsums_y[g*4+1]) +
                              (float)m1  * (float)(bsums_y[g*4+2]     + bsums_y[g*4+3]) +
                              (float)m0b * (float)(bsums_y[(g+1)*4+0] + bsums_y[(g+1)*4+1]) +
                              (float)m1b * (float)(bsums_y[(g+1)*4+2] + bsums_y[(g+1)*4+3]));
        }

        sumf += dys * (d * hsum_float_8(fp32_acc) - minsum_f);
    }

    return sumf;
}

#else  // non-AVX-512 fallback: Highway widened int16 path

#include "hwy/highway.h"

namespace hwy {
namespace HWY_NAMESPACE {

static inline float vec_dot_q4k_q8k_highway(const block_q4_K* vx,
                                            const block_q8_K* vy,
                                            int nb) {
    namespace hn = hwy::HWY_NAMESPACE;

    const auto di32 = hn::ScalableTag<int32_t>();
    const auto di16 = hn::RepartitionToNarrow<decltype(di32)>();
    const auto di8  = hn::RepartitionToNarrow<decltype(di16)>();
    const auto du8  = hn::RebindToUnsigned<decltype(di8)>();

    auto Dot32 = [&](const uint8_t* q4p, const int8_t* q8p) -> int32_t {
        auto acc = hn::Zero(di32);
        for (int half = 0; half < 2; ++half) {
            const auto q4n = hn::LoadU(du8, q4p + half * 16);
            const auto q8v = hn::LoadU(di8, q8p + half * 16);
            const auto q4_l = hn::PromoteLowerTo(di16, hn::BitCast(di8, q4n));
            const auto q4_u = hn::PromoteUpperTo(di16, hn::BitCast(di8, q4n));
            const auto q8_l = hn::PromoteLowerTo(di16, q8v);
            const auto q8_u = hn::PromoteUpperTo(di16, q8v);
            auto prod = hn::Mul(q4_l, q8_l);
            acc = hn::Add(acc, hn::PromoteLowerTo(di32, prod));
            acc = hn::Add(acc, hn::PromoteUpperTo(di32, prod));
            prod = hn::Mul(q4_u, q8_u);
            acc = hn::Add(acc, hn::PromoteLowerTo(di32, prod));
            acc = hn::Add(acc, hn::PromoteUpperTo(di32, prod));
        }
        return hn::ReduceSum(di32, acc);
    };

    float sumf = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const block_q4_K* x = vx + i;
        const block_q8_K* y = vy + i;

        const float d   = GGML_FP16_TO_FP32(x->d);
        const float dm  = GGML_FP16_TO_FP32(x->dmin);
        const float dy  = y->d;

        const uint8_t* q4 = x->qs;
        const int8_t*  q8 = y->qs;
        const int16_t* bsums_y = y->bsums;

        float acc = 0.0f;
        float minsum = 0.0f;
        int is = 0;

        for (int g = 0; g < QK_K / 64; g += 2) {
            uint8_t sc0, m0, sc1, m1, sc0b, m0b, sc1b, m1b;
            get_scale_min_k4(is,     x->scales, &sc0,  &m0);
            get_scale_min_k4(is + 1, x->scales, &sc1,  &m1);
            get_scale_min_k4(is + 2, x->scales, &sc0b, &m0b);
            get_scale_min_k4(is + 3, x->scales, &sc1b, &m1b);
            is += 4;

            alignas(64) uint8_t q4lo[32], q4hi[32], q4lo1[32], q4hi1[32];
            {
                const auto mask4 = hn::Set(du8, 0x0F);
                const auto qb  = hn::LoadU(du8, q4 + g * 32);
                const auto qb1 = hn::LoadU(du8, q4 + (g + 1) * 32);
                hn::StoreU(hn::And(qb,  mask4),                    du8, q4lo);
                hn::StoreU(hn::And(hn::ShiftRight<4>(qb),  mask4), du8, q4hi);
                hn::StoreU(hn::And(qb1, mask4),                    du8, q4lo1);
                hn::StoreU(hn::And(hn::ShiftRight<4>(qb1), mask4), du8, q4hi1);
            }

            const int32_t dot_g_lo  = Dot32(q4lo,  q8 + g * 64);
            const int32_t dot_g_hi  = Dot32(q4hi,  q8 + g * 64 + 32);
            const int32_t dot_g1_lo = Dot32(q4lo1, q8 + (g + 1) * 64);
            const int32_t dot_g1_hi = Dot32(q4hi1, q8 + (g + 1) * 64 + 32);

            acc += (float)sc0  * (float)dot_g_lo
                 + (float)sc1  * (float)dot_g_hi
                 + (float)sc0b * (float)dot_g1_lo
                 + (float)sc1b * (float)dot_g1_hi;

            minsum += dm * (
                (float)m0  * (float)(bsums_y[g*4+0]     + bsums_y[g*4+1]) +
                (float)m1  * (float)(bsums_y[g*4+2]     + bsums_y[g*4+3]) +
                (float)m0b * (float)(bsums_y[(g+1)*4+0] + bsums_y[(g+1)*4+1]) +
                (float)m1b * (float)(bsums_y[(g+1)*4+2] + bsums_y[(g+1)*4+3]));
        }

        sumf += dy * (d * acc - minsum);
    }

    return sumf;
}

} // namespace HWY_NAMESPACE
} // namespace hwy

static float vec_dot_q4k_q8k(const block_q4_K* vx, const block_q8_K* vy, int nb) {
    return hwy::HWY_NAMESPACE::vec_dot_q4k_q8k_highway(vx, vy, nb);
}

#endif  // __AVX512F__ && __AVX512BW__

// ---------------------------------------------------------------------------
// Q4_Kx8 panel layout and production GEMV kernel.
// Uses SSE4.1 + AVX-FMA only: no sustained ZMM to avoid AVX-512 downclocking.
// ---------------------------------------------------------------------------
#include "ns_repack.h"

// Pure-AVX2/FMA GEMV kernel ported from llama.cpp (ggml_gemv_q4_K_8x8_q8_K),
// defined and compiled with -mavx2 -mfma (no -mavx512f) in ns_llama_gemv.cpp.
// repacked_weights: ns_q4_Kx8 panels (n_rows/8 panels of n_blocks entries each).
// n_rows must be a multiple of 8; out must hold n_rows floats.
void ns_gemv_q4k(const void* repacked_weights,
                  const block_q8_K* activation,
                  int n_rows, int n_blocks, float* out);

// Pure-AVX2/FMA per-row Q6_K dot product ported from llama.cpp
// (ggml_vec_dot_q6_K_q8_K, ggml-cpu/arch/x86/quants.c:2426-2508), defined
// in ns_llama_gemv.cpp. Avoids the sustained-ZMM usage of vec_dot_q6k_q8k
// above, which otherwise keeps the CPU downclocked for the whole forward
// pass even after the Q4_K path is fixed (29/198 quantized tensors in this
// model are Q6_K, incl. token_embd/output and some attn_v/ffn_down layers).
float ns_vec_dot_q6k_q8k(const block_q6_K* vx, const block_q8_K* vy, int nb);

// Scalar Q6_K panel GEMM kernel, defined in ns_llama_gemv.cpp.
// panel_data: ns_q6_Kx8 panels (n_rows/8 panels of n_blocks entries each).
// n_rows must be a multiple of 8; out_row_stride is the caller's full
// n_rows (for tail-row layout), out must hold batch_size*out_row_stride
// floats (only the first n_rows of each row-group are written here).
void ns_gemm_q6k(int n_cols, float* out, size_t out_row_stride,
                  const uint8_t* panel_data, const block_q8_K* act,
                  int batch_size, int n_rows);

// Panel GEMV kernel: 8 rows of ns_q4_Kx8 × one block_q8_K activation vector.
// Produces n_active float outputs in out[0..n_active-1].
// Uses SSE4.1 + AVX-FMA only: no sustained ZMM to avoid AVX-512 downclocking.
static inline void vec_dot_q4k_q8k_Rx1(
    const ns_q4_Kx8* __restrict__ vx,
    const block_q8_K* __restrict__ vy,
    int nb, float* __restrict__ out, int n_active = 8)
{
    float sumf[8] = {};
    const __m128i mask4  = _mm_set1_epi8(0x0F);
    const __m128i ones16 = _mm_set1_epi16(1);

    for (int b = 0; b < nb; ++b) {
        const ns_q4_Kx8& x = vx[b];
        const block_q8_K& y = vy[b];
        const float dy = y.d;

        float ds[8], dm[8];
        for (int r = 0; r < 8; ++r) {
            ds[r] = GGML_FP16_TO_FP32(x.d[r]);
            dm[r] = GGML_FP16_TO_FP32(x.dmin[r]);
        }

        float row_dot[8] = {};
        float row_min[8] = {};

        for (int sb = 0; sb < 8; ++sb) {
            // Scale plane: sub-blocks 0-3 at sb*12, sub-blocks 4-7 at (sb-4)*12+48
            const uint8_t* sq = (sb < 4) ? x.scales + sb*12
                                          : x.scales + (sb-4)*12 + 48;
            uint8_t su[8], mu[8];
            for (int r = 0; r < 8; ++r)
                get_scale_min_k4x8(r, sq, &su[r], &mu[r]);

            // bsum = sum of q8 values in this sub-block (2 groups of 16)
            int32_t bsum = (int32_t)y.bsums[sb*2] + (int32_t)y.bsums[sb*2+1];

            // Raw integer dot products for 8 rows, accumulated over 2 chunks
            int32_t raw_dot[8] = {};

            for (int cl = 0; cl < 2; ++cl) {
                int chunk = sb * 2 + cl;
                // 16 signed q8 values for this chunk
                __m128i q8v = _mm_loadu_si128(
                    (const __m128i*)(y.qs + sb*32 + cl*16));

                for (int r = 0; r < 8; ++r) {
                    // 8 packed q4 bytes for row r in this chunk
                    __m128i q4b = _mm_loadl_epi64(
                        (const __m128i*)(x.qs + (size_t)chunk*64 + r*8));
                    // Extract nibbles per byte
                    __m128i lo      = _mm_and_si128(q4b, mask4);
                    __m128i hi      = _mm_and_si128(_mm_srli_epi16(q4b, 4), mask4);
                    // Interleave: [lo0,hi0, lo1,hi1, ..., lo7,hi7]
                    __m128i nibbles = _mm_unpacklo_epi8(lo, hi);
                    // Dot: unsigned nibbles × signed q8 → int16 pairs → int32
                    __m128i prod16  = _mm_maddubs_epi16(nibbles, q8v);
                    __m128i prod32  = _mm_madd_epi16(prod16, ones16);
                    // Horizontal sum of 4 int32s
                    prod32 = _mm_add_epi32(prod32, _mm_srli_si128(prod32, 8));
                    prod32 = _mm_add_epi32(prod32, _mm_srli_si128(prod32, 4));
                    raw_dot[r] += _mm_cvtsi128_si32(prod32);
                }
            }

            for (int r = 0; r < 8; ++r) {
                row_dot[r] += (float)su[r] * (float)raw_dot[r];
                row_min[r] += (float)mu[r] * (float)bsum;
            }
        }

        for (int r = 0; r < n_active; ++r)
            sumf[r] += dy * (ds[r] * row_dot[r] - dm[r] * row_min[r]);
    }

    for (int r = 0; r < n_active; ++r) out[r] = sumf[r];
}

// (d) get_scale_shuffle -- copied verbatim from llama.cpp
// ggml-cpu/arch/x86/quants.c:540-552
static inline __m128i get_scale_shuffle(int i) {
    static const uint8_t k_shuffle[128] = {
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
         2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
         4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
         6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
         8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
        10,10,10,10,10,10,10,10, 11,11,11,11,11,11,11,11,
        12,12,12,12,12,12,12,12, 13,13,13,13,13,13,13,13,
        14,14,14,14,14,14,14,14, 15,15,15,15,15,15,15,15
    };
    return _mm_loadu_si128((const __m128i*)k_shuffle + i);
}

// (e) vec_dot_q6k_q8k -- fused integer dot: one block_q6_K weight row
// (read directly from the mmap'd GGUF, already pre-quantized: same as
// block_q4_K) against one block_q8_K activation block. Ports
// ggml_vec_dot_q6_K_q8_K's AVX2 logic (llama.cpp
// ggml-cpu/arch/x86/quants.c:2426-2508) to AVX-512: the two pairs of
// 32-byte maddubs (q4_0/q4_1 and q4_2/q4_3) are each combined into one
// 64-byte _mm512_maddubs_epi16, then split back into 256-bit halves
// before per-half scale application: same widening pattern as
// dot64_both_nibbles (Change 1). The -32 offset correction via bsums
// and q8sclsub is unchanged from the AVX2 reference.
static inline float vec_dot_q6k_q8k(const block_q6_K* vx, const block_q8_K* vy, int nb) {
    const __m256i m15 = _mm256_set1_epi8(15);
    const __m256i m3  = _mm256_set1_epi8(3);

    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; i++) {
        if (i + 1 < nb) __builtin_prefetch(vx + i + 1, 0, 1);
        const block_q6_K* x = vx + i;
        const block_q8_K* y = vy + i;

        const float d = y->d * GGML_FP16_TO_FP32(x->d);

        const uint8_t* q4 = x->ql;
        const uint8_t* qh = x->qh;
        const int8_t*  q8 = y->qs;

        const __m256i q8sums    = _mm256_loadu_si256((const __m256i*)y->bsums);
        const __m128i scales    = _mm_loadu_si128((const __m128i*)x->scales);
        const __m256i scales_16 = _mm256_cvtepi8_epi16(scales);
        const __m256i q8sclsub  = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales_16), 5);

        __m256i sumi = _mm256_setzero_si256();
        int is = 0;

        for (int j = 0; j < QK_K/128; j++) {
            const __m512i q4bits12 = _mm512_loadu_si512((const void*)q4); q4 += 64;
            const __m256i q4bitsH  = _mm256_loadu_si256((const __m256i*)qh); qh += 32;

            const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m3), 4);
            const __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(12)), 2);
            const __m256i q4h_2 = _mm256_and_si256(q4bitsH, _mm256_set1_epi8(48));
            const __m256i q4h_3 = _mm256_srli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(-64)), 2);

            const __m256i q4bits1 = _mm512_castsi512_si256(q4bits12);
            const __m256i q4bits2 = _mm512_extracti64x4_epi64(q4bits12, 1);

            const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
            const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
            const __m256i q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), q4h_2);
            const __m256i q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), q4h_3);

            const __m512i q4_01 = _mm512_inserti64x4(_mm512_castsi256_si512(q4_0), q4_1, 1);
            const __m512i q4_23 = _mm512_inserti64x4(_mm512_castsi256_si512(q4_2), q4_3, 1);

            const __m512i q8_01 = _mm512_loadu_si512((const void*)q8); q8 += 64;
            const __m512i q8_23 = _mm512_loadu_si512((const void*)q8); q8 += 64;

            const __m512i p16_01 = _mm512_maddubs_epi16(q4_01, q8_01);
            const __m512i p16_23 = _mm512_maddubs_epi16(q4_23, q8_23);

            __m256i p16_0 = _mm512_castsi512_si256(p16_01);
            __m256i p16_1 = _mm512_extracti64x4_epi64(p16_01, 1);
            __m256i p16_2 = _mm512_castsi512_si256(p16_23);
            __m256i p16_3 = _mm512_extracti64x4_epi64(p16_23, 1);

            const __m128i scale_0 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 0));
            const __m128i scale_1 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 1));
            const __m128i scale_2 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 2));
            const __m128i scale_3 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 3));
            is += 4;

            p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
            p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);
            p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_2);
            p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_3);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));
        }

        sumi = _mm256_sub_epi32(sumi, q8sclsub);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }

    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

// ---------------------------------------------------------------------------
// Q4_Kx8 panel layout and scalar reference dot product.
// This is NOT the production AVX-512 kernel; it is only used by
// Transformer::validate_repack_q4k() to prove the repacked layout is correct.
// ---------------------------------------------------------------------------

static inline void vec_dot_q4k_q8k_Rx1_ref(const ns_q4_Kx8* b_ptr,
                                           const block_q8_K* a_ptr,
                                           int n, float out[8], int n_active) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    (void)n_active;

    for (int j = 0; j < 8; j++) out[j] = 0.0f;

    for (int l = 0; l < nb; l++) {
        float sumf[8];
        float sum_minf[8];
        uint32_t utmp[32];

        for (int j = 0; j < 8; j++) {
            sumf[j] = 0.0f;
            sum_minf[j] = 0.0f;
        }

        // Unpack the 96 packed scale/min bytes into 128 bytes:
        //   8 sub-blocks * (8 scales + 8 mins) = 128 bytes.
        for (int sb = 0; sb < 8; sb++) {
            memcpy(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
            utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
            const uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
            utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
            utmp[sb * 4 + 2] = uaux_0;
            utmp[sb * 4 + 0] &= kmask1;
        }

        // Dot product over the 16 natural q4 chunks.
        // Chunk k belongs to sub-block sb = k/2; each q4 byte holds two
        // adjacent q8 values, both in that sub-block.
        for (int k = 0; k < (qk / (2 * blocklen)); k++) {
            int sb = k / 2;
            const uint8_t* scales = (const uint8_t*)utmp + sb * 16;
            for (int j = 0; j < ncols_interleaved; j++) {
                int sumi = 0;
                for (int i = 0; i < blocklen; ++i) {
                    const uint8_t q4b = b_ptr[l].qs[k * 64 + j * 8 + i];
                    const int v0 = (int)(q4b & 0xF);
                    const int v1 = (int)(q4b >> 4);

                    const int q0 = a_ptr[l].qs[k * 16 + 2 * i];
                    const int q1 = a_ptr[l].qs[k * 16 + 2 * i + 1];

                    sumi += (v0 * q0 + v1 * q1) * (int)scales[j];
                }
                sumf[j] += (float)sumi * GGML_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
            }
        }

        // Min correction term, one per sub-block.
        for (int sb = 0; sb < 8; sb++) {
            const uint8_t* mins = (const uint8_t*)utmp + 8 + sb * 16;
            for (int j = 0; j < ncols_interleaved; j++) {
                sum_minf[j] += (float)mins[j]
                             * (float)(a_ptr[l].bsums[sb * 2] + a_ptr[l].bsums[sb * 2 + 1])
                             * GGML_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
            }
        }

        for (int j = 0; j < ncols_interleaved; j++) {
            if (j < n_active) out[j] += sumf[j] - sum_minf[j];
        }
    }
}

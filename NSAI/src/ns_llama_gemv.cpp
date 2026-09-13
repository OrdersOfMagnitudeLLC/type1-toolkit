// ns_llama_gemv.cpp
//
// Ported directly from llama.cpp (MIT License):
//   ggml/src/ggml-cpu/arch/x86/repack.cpp
//     - ggml_gemv_q4_K_8x8_q8_K()  (lines 1464-1685)
//   Copyright (c) 2023-2024 The ggml authors
//   MIT License: https://github.com/ggml-org/llama.cpp/blob/master/LICENSE
//
// This file intentionally compiles with -mavx2 -mfma ONLY (no -mavx512f).
// llama.cpp's per-token GEMV hot path uses pure AVX2/FMA and never issues
// a sustained ZMM instruction, which avoids the AVX-512 downclocking that
// hurts NSRun's original vec_dot_q4k_q8k on the i7-1165G7.
//
// #define HWY_DISABLED_TARGETS (HWY_AVX3 | HWY_AVX3_DL | HWY_AVX3_ZMM)
// (Highway is not used in this file directly; the disabled-targets define is
// kept here for documentation/consistency with the rest of the codebase's
// AVX-512-avoidance policy for this kernel family.)

#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <cassert>

#include "../vendor/ns_dequant.h"  // QK_K, block_q6_K, block_q8_K
#include "ns_repack.h"             // ns_q4_Kx8 (byte-compatible with llama's block_q4_Kx8)

#define UNUSED(x) (void)(x)

// ---------------------------------------------------------------------------
// FP16 -> FP32 vector load helpers (non-F16C path from repack.cpp, using the
// scalar GGML_FP16_TO_FP32 conversion already available in ns_dequant.h).
// ---------------------------------------------------------------------------
static inline __m256 ns_f32cx8_load(const uint16_t* x) {
    float tmp[8];
    for (int i = 0; i < 8; i++) tmp[i] = GGML_FP16_TO_FP32(x[i]);
    return _mm256_loadu_ps(tmp);
}

static inline __m256 ns_f32cx8_rearranged_load(const uint16_t* x, __m128i arrangeMask) {
    uint16_t tmphalf[8];
    float tmp[8];
    _mm_storeu_si128((__m128i*)tmphalf, _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)x), arrangeMask));
    for (int i = 0; i < 8; i++) tmp[i] = GGML_FP16_TO_FP32(tmphalf[i]);
    return _mm256_loadu_ps(tmp);
}

// ---------------------------------------------------------------------------
// Exact port of ggml_gemv_q4_K_8x8_q8_K (AVX2 branch only).
// vx: array of ns_q4_Kx8 panels, nc/8 panels * nb blocks each, panel-major.
// vy: array of block_q8_K, nr rows * nb blocks each, row-major.
// s:  output, nr*nc floats; s[y*nr + x*8 .. +7] holds panel x's 8 outputs
//     for activation row y (bs is unused, matches upstream signature).
// ---------------------------------------------------------------------------
static void ns_ggml_gemv_q4_K_8x8_q8_K(int n, float* s, size_t bs,
                                        const void* vx, const void* vy,
                                        int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    assert(n % qk == 0);
    assert(nc % 8 == 0);
    UNUSED(bs);

    // Lookup table to convert signed nibbles to signed bytes
    __m256i signextendlut = _mm256_castsi128_si256(_mm_set_epi8(-1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0));
    signextendlut = _mm256_permute2f128_si256(signextendlut, signextendlut, 0);
    UNUSED(signextendlut); // unused on this path (kept for fidelity with upstream)

    __m128i deltamask = _mm_set_epi8(15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0);
    __m128i scalemask = _mm_set_epi8(7, 7, 3, 3, 6, 6, 2, 2, 5, 5, 1, 1, 4, 4, 0, 0);
    __m256i finalpermutemask = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);

    const __m256i m4b = _mm256_set1_epi8(0x0F);

    int64_t b_nb = n / QK_K;

    const ns_q4_Kx8*  b_ptr_start = (const ns_q4_Kx8*)vx;
    const block_q8_K*  a_ptr_start = (const block_q8_K*)vy;

    for (int64_t y = 0; y < nr; y++) {
        const block_q8_K* a_ptr = a_ptr_start + (y * nb);

        // q8s depends only on the activation row (y) and block (b), not on
        // the weight panel (x). Precompute it once per (y, b) and cache it
        // here, eliminating one hadd/permute per panel iteration on wide rows.
        assert(nb <= 128);
        __m256i q8s_init[128];
        for (int64_t b = 0; b < nb; b++) {
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*)(a_ptr[b].bsums));
            __m256i q8s = _mm256_castsi128_si256(_mm_hadd_epi16(_mm256_castsi256_si128(q8sums), _mm256_extracti128_si256(q8sums, 1)));
            q8s = _mm256_permute2f128_si256(q8s, q8s, 0);
            q8s_init[b] = q8s;
        }

        for (int64_t x = 0; x < nc / 8; x++) {
            const ns_q4_Kx8* b_ptr = b_ptr_start + (x * b_nb);
            const uint8_t* panel_base = (const uint8_t*)b_ptr;
            const size_t panel_stride = sizeof(ns_q4_Kx8);

            __m256 acc_row = _mm256_setzero_ps();
            __m256 acc_min_rows = _mm256_setzero_ps();

            for (int64_t b = 0; b < nb; b++) {
                if (b + 2 < nb)
                    __builtin_prefetch(panel_base + panel_stride * (b + 2), 0, 0);

                const __m256 row_scale_f32 = _mm256_set1_ps((a_ptr[b].d));
                const __m256 col_scale_f32 = ns_f32cx8_rearranged_load(b_ptr[b].d, deltamask);
                const __m256 col_dmin_f32 = ns_f32cx8_load(b_ptr[b].dmin);

                __m256i iacc_b = _mm256_setzero_si256();
                __m256i iacc_min_b = _mm256_setzero_si256();

                __m256i q8s = q8s_init[b];

                for (int sb = 0; sb < QK_K / 64; sb++) {
                    const __m256i rhs_raw_vec_0123_0 = _mm256_loadu_si256((const __m256i*)(b_ptr[b].qs + sb * 256));
                    const __m256i rhs_raw_vec_4567_0 = _mm256_loadu_si256((const __m256i*)(b_ptr[b].qs + 32 + sb * 256));
                    const __m256i rhs_raw_vec_0123_1 = _mm256_loadu_si256((const __m256i*)(b_ptr[b].qs + 64 + sb * 256));
                    const __m256i rhs_raw_vec_4567_1 = _mm256_loadu_si256((const __m256i*)(b_ptr[b].qs + 96 + sb * 256));
                    const __m256i rhs_raw_vec_0123_2 = _mm256_loadu_si256((const __m256i*)(b_ptr[b].qs + 128 + sb * 256));
                    const __m256i rhs_raw_vec_4567_2 = _mm256_loadu_si256((const __m256i*)(b_ptr[b].qs + 160 + sb * 256));
                    const __m256i rhs_raw_vec_0123_3 = _mm256_loadu_si256((const __m256i*)(b_ptr[b].qs + 192 + sb * 256));
                    const __m256i rhs_raw_vec_4567_3 = _mm256_loadu_si256((const __m256i*)(b_ptr[b].qs + 224 + sb * 256));

                    const __m256i rhs_vec_0123_00 = _mm256_and_si256(rhs_raw_vec_0123_0, m4b);
                    const __m256i rhs_vec_4567_00 = _mm256_and_si256(rhs_raw_vec_4567_0, m4b);
                    const __m256i rhs_vec_0123_01 = _mm256_and_si256(rhs_raw_vec_0123_1, m4b);
                    const __m256i rhs_vec_4567_01 = _mm256_and_si256(rhs_raw_vec_4567_1, m4b);
                    const __m256i rhs_vec_0123_02 = _mm256_and_si256(rhs_raw_vec_0123_2, m4b);
                    const __m256i rhs_vec_4567_02 = _mm256_and_si256(rhs_raw_vec_4567_2, m4b);
                    const __m256i rhs_vec_0123_03 = _mm256_and_si256(rhs_raw_vec_0123_3, m4b);
                    const __m256i rhs_vec_4567_03 = _mm256_and_si256(rhs_raw_vec_4567_3, m4b);

                    const __m256i rhs_vec_0123_10 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_0, 4), m4b);
                    const __m256i rhs_vec_4567_10 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_0, 4), m4b);
                    const __m256i rhs_vec_0123_11 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_1, 4), m4b);
                    const __m256i rhs_vec_4567_11 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_1, 4), m4b);
                    const __m256i rhs_vec_0123_12 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_2, 4), m4b);
                    const __m256i rhs_vec_4567_12 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_2, 4), m4b);
                    const __m256i rhs_vec_0123_13 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_3, 4), m4b);
                    const __m256i rhs_vec_4567_13 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_3, 4), m4b);

                    uint32_t utmp_0[4], utmp_1[4];

                    memcpy(utmp_0, b_ptr[b].scales + 24 * sb, 12);
                    utmp_0[3] = ((utmp_0[2] >> 4) & kmask2) | (((utmp_0[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_0 = utmp_0[1] & kmask1;
                    utmp_0[1] = (utmp_0[2] & kmask2) | (((utmp_0[0] >> 6) & kmask3) << 4);
                    utmp_0[2] = uaux_0;
                    utmp_0[0] &= kmask1;

                    memcpy(utmp_1, b_ptr[b].scales + 12 + sb * 24, 12);
                    utmp_1[3] = ((utmp_1[2] >> 4) & kmask2) | (((utmp_1[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_1 = utmp_1[1] & kmask1;
                    utmp_1[1] = (utmp_1[2] & kmask2) | (((utmp_1[0] >> 6) & kmask3) << 4);
                    utmp_1[2] = uaux_1;
                    utmp_1[0] &= kmask1;

                    const __m128i mins_and_scales_0 = _mm_set_epi32(utmp_0[3], utmp_0[2], utmp_0[1], utmp_0[0]);
                    __m128i scales_rearrange_0 = _mm_shuffle_epi8(mins_and_scales_0, scalemask);
                    __m256i scales_0 = _mm256_cvtepu8_epi16(scales_rearrange_0);

                    __m128i mins_and_scales_1 = _mm_set_epi32(utmp_1[3], utmp_1[2], utmp_1[1], utmp_1[0]);
                    __m128i scales_rearrange_1 = _mm_shuffle_epi8(mins_and_scales_1, scalemask);
                    __m256i scales_1 = _mm256_cvtepu8_epi16(scales_rearrange_1);

                    __m256i mins_01 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(_mm_shuffle_epi32(mins_and_scales_0, 78), _mm_shuffle_epi32(mins_and_scales_1, 78)));

                    __m256i lhs_vec_00 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*)(a_ptr[b].qs + sb * 64)));
                    __m256i lhs_vec_01 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*)(a_ptr[b].qs + 16 + sb * 64)));
                    __m256i lhs_vec_10 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*)(a_ptr[b].qs + 32 + sb * 64)));
                    __m256i lhs_vec_11 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*)(a_ptr[b].qs + 48 + sb * 64)));

                    lhs_vec_00 = _mm256_permute2f128_si256(lhs_vec_00, lhs_vec_00, 0);
                    lhs_vec_01 = _mm256_permute2f128_si256(lhs_vec_01, lhs_vec_01, 0);
                    lhs_vec_10 = _mm256_permute2f128_si256(lhs_vec_10, lhs_vec_10, 0);
                    lhs_vec_11 = _mm256_permute2f128_si256(lhs_vec_11, lhs_vec_11, 0);

                    __m256i iacc_0 = _mm256_setzero_si256();
                    __m256i iacc_1 = _mm256_setzero_si256();

                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_00, _mm256_shuffle_epi32(rhs_vec_4567_00, 177), 170), _mm256_shuffle_epi32(lhs_vec_00, 0)));
                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_00, 177), rhs_vec_4567_00, 170), _mm256_shuffle_epi32(lhs_vec_00, 85)));

                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_01, _mm256_shuffle_epi32(rhs_vec_4567_01, 177), 170), _mm256_shuffle_epi32(lhs_vec_00, 170)));
                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_01, 177), rhs_vec_4567_01, 170), _mm256_shuffle_epi32(lhs_vec_00, 255)));

                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_02, _mm256_shuffle_epi32(rhs_vec_4567_02, 177), 170), _mm256_shuffle_epi32(lhs_vec_01, 0)));
                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_02, 177), rhs_vec_4567_02, 170), _mm256_shuffle_epi32(lhs_vec_01, 85)));

                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_03, _mm256_shuffle_epi32(rhs_vec_4567_03, 177), 170), _mm256_shuffle_epi32(lhs_vec_01, 170)));
                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_03, 177), rhs_vec_4567_03, 170), _mm256_shuffle_epi32(lhs_vec_01, 255)));

                    iacc_0 = _mm256_madd_epi16(iacc_0, scales_0);

                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_10, _mm256_shuffle_epi32(rhs_vec_4567_10, 177), 170), _mm256_shuffle_epi32(lhs_vec_10, 0)));
                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_10, 177), rhs_vec_4567_10, 170), _mm256_shuffle_epi32(lhs_vec_10, 85)));

                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_11, _mm256_shuffle_epi32(rhs_vec_4567_11, 177), 170), _mm256_shuffle_epi32(lhs_vec_10, 170)));
                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_11, 177), rhs_vec_4567_11, 170), _mm256_shuffle_epi32(lhs_vec_10, 255)));

                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_12, _mm256_shuffle_epi32(rhs_vec_4567_12, 177), 170), _mm256_shuffle_epi32(lhs_vec_11, 0)));
                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_12, 177), rhs_vec_4567_12, 170), _mm256_shuffle_epi32(lhs_vec_11, 85)));

                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_13, _mm256_shuffle_epi32(rhs_vec_4567_13, 177), 170), _mm256_shuffle_epi32(lhs_vec_11, 170)));
                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_13, 177), rhs_vec_4567_13, 170), _mm256_shuffle_epi32(lhs_vec_11, 255)));

                    iacc_1 = _mm256_madd_epi16(iacc_1, scales_1);

                    __m256i iacc_sb = _mm256_add_epi32(iacc_0, iacc_1);

                    __m256i q8s_sb = _mm256_shuffle_epi32(q8s, 0);
                    __m256i iacc_min_sb = _mm256_madd_epi16(q8s_sb, mins_01);
                    q8s = _mm256_bsrli_epi128(q8s, 4);

                    iacc_b = _mm256_add_epi32(iacc_b, iacc_sb);
                    iacc_min_b = _mm256_add_epi32(iacc_min_b, iacc_min_sb);
                }

                acc_row = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_b), _mm256_mul_ps(col_scale_f32, row_scale_f32), acc_row);
                acc_min_rows = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_min_b), _mm256_mul_ps(col_dmin_f32, row_scale_f32), acc_min_rows);
            }

            acc_row = _mm256_permutevar8x32_ps(acc_row, finalpermutemask);
            _mm256_storeu_ps(s + (y * nr + x * 8), _mm256_sub_ps(acc_row, acc_min_rows));
        }
    }
}

// ---------------------------------------------------------------------------
// Thin C++ wrapper.
// ---------------------------------------------------------------------------
void ns_gemv_q4k(const void* repacked_weights,
                  const block_q8_K* activation,
                  int n_rows, int n_blocks, float* out) {
    // n_rows must be a multiple of 8 (panel width); callers handle tail rows.
    const int n = n_blocks * QK_K;
    ns_ggml_gemv_q4_K_8x8_q8_K(n, out, /*bs=*/(size_t)n_rows, repacked_weights, activation,
                                /*nr=*/1, /*nc=*/n_rows);
}

// ---------------------------------------------------------------------------
// Exact port of ggml_vec_dot_q6_K_q8_K (AVX2 branch only, from
// ggml-cpu/arch/x86/quants.c:2426-2508). One block_q6_K row against one
// block_q8_K activation vector, pure AVX2/FMA, no ZMM.
// ---------------------------------------------------------------------------
static inline __m128i ns_get_scale_shuffle(int i) {
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

static inline float ns_hsum_float_8(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

float ns_vec_dot_q6k_q8k(const block_q6_K* vx, const block_q8_K* vy, int nb) {
    const __m256i m3  = _mm256_set1_epi8(3);
    const __m256i m15 = _mm256_set1_epi8(15);

    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = vy[i].d * GGML_FP16_TO_FP32(vx[i].d);

        const uint8_t* q4 = vx[i].ql;
        const uint8_t* qh = vx[i].qh;
        const int8_t*  q8 = vy[i].qs;

        const __m256i q8sums    = _mm256_loadu_si256((const __m256i*)vy[i].bsums);
        const __m128i scales    = _mm_loadu_si128((const __m128i*)vx[i].scales);
        const __m256i scales_16 = _mm256_cvtepi8_epi16(scales);
        const __m256i q8sclsub  = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales_16), 5);

        __m256i sumi = _mm256_setzero_si256();
        int is = 0;

        for (int j = 0; j < QK_K / 128; ++j) {
            const __m256i q4bits1 = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
            const __m256i q4bits2 = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
            const __m256i q4bitsH = _mm256_loadu_si256((const __m256i*)qh); qh += 32;

            const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m3), 4);
            const __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(12)), 2);
            const __m256i q4h_2 = _mm256_and_si256(q4bitsH, _mm256_set1_epi8(48));
            const __m256i q4h_3 = _mm256_srli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(-64)), 2);

            const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
            const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
            const __m256i q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), q4h_2);
            const __m256i q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), q4h_3);

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

            __m256i p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
            __m256i p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
            __m256i p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
            __m256i p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);

            const __m128i scale_0 = _mm_shuffle_epi8(scales, ns_get_scale_shuffle(is + 0));
            const __m128i scale_1 = _mm_shuffle_epi8(scales, ns_get_scale_shuffle(is + 1));
            const __m128i scale_2 = _mm_shuffle_epi8(scales, ns_get_scale_shuffle(is + 2));
            const __m128i scale_3 = _mm_shuffle_epi8(scales, ns_get_scale_shuffle(is + 3));
            is += 4;

            p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
            p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);
            p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_2);
            p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_3);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));
        }

        sumi = _mm256_sub_epi32(sumi, q8sclsub);
        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
    }

    return ns_hsum_float_8(acc);
}

// ---------------------------------------------------------------------------
// Scalar Q6_K panel GEMM kernel.
//
// panel_data: array of ns_q6_Kx8 panels, (n_rows/8) panels * nb blocks each,
// panel-major (matches ns_gemv_q4k's layout convention). n_rows MUST be a
// multiple of 8: the caller is responsible for routing any tail rows
// (n_rows % 8 != 0) through the per-row ns_vec_dot_q6k_q8k/vec_dot_q6k_q8k
// path instead.
//
// Int32 accumulation only; no fp32 weight materialization. The -32
// zero-point is folded via the q8_K bsums exactly as in ns_vec_dot_q6k_q8k
// (see lines above): raw 6-bit values in [0,63] are accumulated directly
// against q8 activations, then corrected once per block via
// sumi -= 32 * sum_g(scale[g] * bsums[g]).
//
// Panel byte layout (see repack_q6_K_row_panel in ns_repack.h): for a
// row-local byte position p (0..127 for ql, 0..63 for qh) of row r, the
// panel offset is chunk*64 + r*8 + within, where chunk = p/8, within = p%8.
// ---------------------------------------------------------------------------
void ns_gemm_q6k(int n_cols, float* out, size_t out_row_stride,
                  const uint8_t* panel_data, const block_q8_K* act,
                  int batch_size, int n_rows) {
    const int nb = n_cols / QK_K;
    const int n_panels = n_rows / 8;
    const ns_q6_Kx8* panels = (const ns_q6_Kx8*)panel_data;

    for (int p = 0; p < n_panels; p++) {
        const ns_q6_Kx8* panel_blocks = panels + (size_t)p * nb;
        const int r0 = p * 8;

        for (int b = 0; b < batch_size; b++) {
            float sumf[8] = {0, 0, 0, 0, 0, 0, 0, 0};

            for (int l = 0; l < nb; l++) {
                const ns_q6_Kx8& blk = panel_blocks[l];
                const block_q8_K& a = act[(size_t)b * nb + l];

                float dr[8];
                for (int r = 0; r < 8; r++) dr[r] = GGML_FP16_TO_FP32(blk.d[r]);

                int32_t rawsum[8][16];
                memset(rawsum, 0, sizeof(rawsum));

                for (int half = 0; half < 2; half++) {
                    for (int is = 0; is < 2; is++) {
                        for (int vi = 0; vi < 4; vi++) {
                            const int variant = vi * 2; // 0,2,4,6 -> qh shift amount
                            const int group   = half * 8 + is + variant;
                            const int q8_base = half * 128 + variant * 16;
                            const int ql_base = half * 64 + ((variant == 2 || variant == 6) ? 32 : 0);
                            const int qh_base = half * 32;
                            const int nibble_shift = (variant >= 4) ? 4 : 0;

                            for (int ll = 0; ll < 16; ll++) {
                                const int l_idx  = is * 16 + ll;
                                const int ql_pos = ql_base + l_idx;
                                const int qh_pos = qh_base + l_idx;
                                const int8_t qa  = a.qs[q8_base + l_idx];

                                const int ql_chunk  = ql_pos / 8, ql_within = ql_pos % 8;
                                const int ql_off    = ql_chunk * 64 + ql_within;
                                const int qh_chunk  = qh_pos / 8, qh_within = qh_pos % 8;
                                const int qh_off    = qh_chunk * 64 + qh_within;

                                for (int r = 0; r < 8; r++) {
                                    uint8_t qlb = blk.ql[ql_off + r * 8];
                                    uint8_t qhb = blk.qh[qh_off + r * 8];
                                    int val = ((qlb >> nibble_shift) & 0xF) | (((qhb >> variant) & 0x3) << 4);
                                    rawsum[r][group] += val * (int)qa;
                                }
                            }
                        }
                    }
                }

                for (int r = 0; r < 8; r++) {
                    int32_t sumi = 0;
                    int32_t sclsub = 0;
                    for (int g = 0; g < 16; g++) {
                        int8_t sc = blk.scales[g * 8 + r];
                        sumi += rawsum[r][g] * sc;
                        sclsub += (int32_t)sc * a.bsums[g];
                    }
                    sumi -= sclsub << 5;
                    sumf[r] += dr[r] * a.d * (float)sumi;
                }
            }

            for (int r = 0; r < 8; r++) {
                out[(size_t)b * out_row_stride + r0 + r] = sumf[r];
            }
        }
    }
}

#pragma once
#include <immintrin.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <math.h>

typedef uint16_t ggml_half;
typedef uint16_t ggml_fp16_t;
#define GGML_RESTRICT __restrict__
#define GGML_UNUSED(x) (void)(x)
#define UNUSED(x) GGML_UNUSED(x)
#define QK_K 256

#define GGML_F32Cx8x2_LOAD(a, b) \
    _mm512_cvtph_ps(_mm256_set_m128i( \
        _mm_loadu_si128((const __m128i*)(b)), \
        _mm_loadu_si128((const __m128i*)(a))))

inline float ggml_cpu_fp16_to_fp32(uint16_t h);
#define GGML_CPU_FP16_TO_FP32(x) ggml_cpu_fp16_to_fp32(x)
#define GGML_CPU_E8M0_TO_FP32_HALF(x) \
    ((x)==0xFF ? NAN : powf(2.0f,(float)(uint8_t)(x)-127.0f))

inline int nearest_int(float x);

// Helper functions
inline __m256 __avx_f32cx8_load(const ggml_fp16_t *x);
inline __m256 __avx_repeat_f32cx8_load(const ggml_fp16_t *x);
inline __m512 __avx512_f32cx8x2_load(const ggml_fp16_t *x, const ggml_fp16_t *y);

#define GGML_F32Cx8_LOAD(x)     __avx_f32cx8_load(x)
#define GGML_F32Cx8_REPEAT_LOAD(x, loadMask)     __avx_repeat_f32cx8_load(x)

// Block structures
struct block_q8_Kx4 {
    float d[4];              // delta
    int8_t qs[QK_K * 4];     // quants
    int16_t bsums[QK_K / 4]; // sum of quants in groups of 16
};

struct block_q4_Kx8 {
    ggml_half d[8];      // super-block scale for quantized scales
    ggml_half dmin[8];   // super-block scale for quantized mins
    uint8_t scales[96];  // scales and mins, quantized with 6 bits
    uint8_t qs[1024];    // 4--bit quants
};

// Function declarations
void ggml_quantize_mat_q8_K_4x8_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
void ggml_quantize_mat_q8_K_4x8(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
void ggml_gemm_q4_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q4_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);

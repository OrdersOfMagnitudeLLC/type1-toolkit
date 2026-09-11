// ---------------------------------------------------------------------------
// AVX2-pinned panel GEMV. Ref: ggml_gemv_q4_K_8x8_q8_K
// (llama.cpp/ggml/src/ggml-cpu/arch/x86/repack.cpp:1464-1685)
//
// NOTE: ns_q4_Kx8.qs is NOT byte-identical to llama's block_q4_Kx8 (see
// docs/highway_rx1_kernel_handoff.md) so this is a from-first-principles
// AVX2 vectorization of NSRun's own reference decode
// (vec_dot_q4k_q8k_Rx1_ref, below), not a transcription of llama's exact
// blend constants. It copies llama's *structural* wins:
//   1) multiple rows decoded per SIMD op (here: 4 rows/load, via
//      unpacklo/hi_epi8 instead of llama's column blend),
//   2) scale folded in via integer madd before any float conversion,
//   3) one cvt-to-float per super-block per row-group, not per 128 elements.
//
// Pinned to AVX2: no ZMM, no Tiger Lake downclock, matching why llama's
// GEMV wins (docs/kernel_gap_analysis.md, Q2).
//
// COMPILED WITH -mavx2 -mfma ONLY (no -mavx512f/-mavx512bw) to ensure
// no AVX-512 instructions are generated.
// ---------------------------------------------------------------------------

#include "ns_quant_fused.h"
#include <immintrin.h>
#include <string.h>

// Decodes one 32-byte raw weight load (4 rows x 8 bytes) into two
// 256-bit "row-pair" vectors ready for maddubs against a tiled q8 window.
// Mirrors the nibble split in vec_dot_q4k_q8k_Rx1_ref's get_q4_value()-equivalent
// packing (ns_repack.h:173-179), vectorized.
//
// Input: raw32 = 32 bytes = rows {r0,r1,r2,r3} x 8 bytes each, where row
// layout is [lo0,lo1,...,lo7 as-packed-byte] per ns_repack.h:193
// (byte j = (v0 & 0xF) | ((v1 & 0xF) << 4), v0/v1 adjacent elements).
//
// Output:
//   pairA: lane0 (bytes 0-15)  = row r0's 16-byte [lo0,hi0,...,lo7,hi7]
//          lane1 (bytes 16-31) = row r2's 16-byte pattern
//   pairB: lane0 = row r1's pattern, lane1 = row r3's pattern
static inline void DecodeRowQuartet(
    const uint8_t* raw32,
    __m256i& pairA,
    __m256i& pairB) {
  const __m256i raw = _mm256_loadu_si256((const __m256i*)raw32);
  const __m256i mask0F = _mm256_set1_epi8(0x0F);

  const __m256i lo = _mm256_and_si256(raw, mask0F);
  // Standard ggml/llama nibble-extract trick: shift as u16 lanes, then
  // mask each byte to 0x0F — the top-nibble contamination from the
  // neighboring byte lands exactly in the bits the mask discards.
  const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(raw, 4), mask0F);

  pairA = _mm256_unpacklo_epi8(lo, hi);  // rows {r0, r2}
  pairB = _mm256_unpackhi_epi8(lo, hi);  // rows {r1, r3}
}

// Horizontal-sums a __m128i (4 int32 lanes) to a scalar.
static inline int32_t HSum4(__m128i v) {
  alignas(16) int32_t tmp[4];
  _mm_store_si128((__m128i*)tmp, v);
  return tmp[0] + tmp[1] + tmp[2] + tmp[3];
}

// Processes rows {g*4 .. g*4+3} for one nb super-block, accumulating into
// sumf[g*4..g*4+3]. su/mu are the already-decoded per-row scale/min for
// the CURRENT sub-block sb (get_scale_min_k4x8 output).
struct GroupAcc {
  __m256i accA;  // rows {r0, r2} -> 8 i32 lanes (4+4)
  __m256i accB;  // rows {r1, r3}
  GroupAcc() : accA(_mm256_setzero_si256()), accB(_mm256_setzero_si256()) {}
};

static inline void AccumulateChunkPair(
    GroupAcc& g,
    const uint8_t* qs_chunk0_ptr, const uint8_t* qs_chunk1_ptr,
    const int8_t*  q8_chunk0_ptr, const int8_t*  q8_chunk1_ptr,
    const uint8_t su_r0, const uint8_t su_r1,
    const uint8_t su_r2, const uint8_t su_r3) {
  __m256i pairA0, pairB0, pairA1, pairB1;
  DecodeRowQuartet(qs_chunk0_ptr, pairA0, pairB0);
  DecodeRowQuartet(qs_chunk1_ptr, pairA1, pairB1);

  // Load 16-byte q8 window and broadcast to both 128-bit lanes
  __m256i q8t0 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*)q8_chunk0_ptr));
  q8t0 = _mm256_inserti128_si256(q8t0, _mm_loadu_si128((const __m128i*)q8_chunk0_ptr), 1);
  __m256i q8t1 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*)q8_chunk1_ptr));
  q8t1 = _mm256_inserti128_si256(q8t1, _mm_loadu_si128((const __m128i*)q8_chunk1_ptr), 1);

  // maddubs: u8 nibble-pairs x i8 activation -> i16 pair-sums.
  const __m256i sA0 = _mm256_maddubs_epi16(pairA0, q8t0);
  const __m256i sA1 = _mm256_maddubs_epi16(pairA1, q8t1);
  const __m256i sB0 = _mm256_maddubs_epi16(pairB0, q8t0);
  const __m256i sB1 = _mm256_maddubs_epi16(pairB1, q8t1);

  // Sum both chunks (cl=0,1) of this sub-block before folding in scale
  const __m256i combinedA = _mm256_add_epi16(sA0, sA1);  // 16 i16 lanes: [r0 x8][r2 x8]
  const __m256i combinedB = _mm256_add_epi16(sB0, sB1);  // [r1 x8][r3 x8]

  // Per-row scale, broadcast to match combinedA/B's lane grouping
  alignas(32) int16_t scaleA[16];
  alignas(32) int16_t scaleB[16];
  for (int k = 0; k < 8; ++k) {
    scaleA[k]     = static_cast<int16_t>(su_r0);
    scaleA[k + 8] = static_cast<int16_t>(su_r2);
    scaleB[k]     = static_cast<int16_t>(su_r1);
    scaleB[k + 8] = static_cast<int16_t>(su_r3);
  }
  const __m256i scaleAv = _mm256_loadu_si256((const __m256i*)scaleA);
  const __m256i scaleBv = _mm256_loadu_si256((const __m256i*)scaleB);

  // madd: folds scale in AND reduces pairs to i32 in the same instruction
  const __m256i reducedA = _mm256_madd_epi16(combinedA, scaleAv);  // 8 i32: [r0 x4][r2 x4]
  const __m256i reducedB = _mm256_madd_epi16(combinedB, scaleBv);  // [r1 x4][r3 x4]

  g.accA = _mm256_add_epi32(g.accA, reducedA);
  g.accB = _mm256_add_epi32(g.accB, reducedB);
}

// Panel GEMV kernel: 8 rows of ns_q4_Kx8 x one block_q8_K activation vector.
// Produces n_active float outputs in out[0..n_active-1].
// AVX2-pinned via compilation flags — no sustained ZMM.
void vec_dot_q4k_q8k_Rx1_highway(
    const ns_q4_Kx8* __restrict__ vx,
    const block_q8_K* __restrict__ vy,
    int nb, float* __restrict__ out, int n_active)
{
  float sumf[8] = {};

  for (int b = 0; b < nb; ++b) {
    const ns_q4_Kx8& x = vx[b];
    const block_q8_K& y = vy[b];
    const float dy = y.d;

    float ds[8], dm[8];
    for (int r = 0; r < 8; ++r) {
      ds[r] = GGML_FP16_TO_FP32(x.d[r]);
      dm[r] = GGML_FP16_TO_FP32(x.dmin[r]);
    }

    // Two row-groups of 4: g=0 -> rows 0-3, g=1 -> rows 4-7.
    GroupAcc gacc[2];
    float row_min[8] = {};

    for (int sb = 0; sb < 8; ++sb) {
      const uint8_t* sq = (sb < 4) ? x.scales + sb * 12
                                    : x.scales + (sb - 4) * 12 + 48;
      uint8_t su[8], mu[8];
      for (int r = 0; r < 8; ++r) get_scale_min_k4x8(r, sq, &su[r], &mu[r]);

      const int32_t bsum = (int32_t)y.bsums[sb * 2] + (int32_t)y.bsums[sb * 2 + 1];
      for (int r = 0; r < 8; ++r) row_min[r] += (float)mu[r] * (float)bsum;

      const int chunk0 = sb * 2;
      const int chunk1 = sb * 2 + 1;
      const int8_t* q8c0 = y.qs + sb * 32;
      const int8_t* q8c1 = y.qs + sb * 32 + 16;

      for (int g = 0; g < 2; ++g) {
        const uint8_t* qc0 = x.qs + (size_t)chunk0 * 64 + g * 32;
        const uint8_t* qc1 = x.qs + (size_t)chunk1 * 64 + g * 32;
        AccumulateChunkPair(gacc[g], qc0, qc1, q8c0, q8c1,
                             su[g * 4 + 0], su[g * 4 + 1],
                             su[g * 4 + 2], su[g * 4 + 3]);
      }
    }

    // Finalize: one horizontal reduce + one cvt-to-float + one fma per
    // row, per super-block (matches llama's conversion cadence, Q3 fix).
    for (int g = 0; g < 2; ++g) {
      const int32_t r0_sum = HSum4(_mm256_castsi256_si128(gacc[g].accA));
      const int32_t r2_sum = HSum4(_mm256_extracti128_si256(gacc[g].accA, 1));
      const int32_t r1_sum = HSum4(_mm256_castsi256_si128(gacc[g].accB));
      const int32_t r3_sum = HSum4(_mm256_extracti128_si256(gacc[g].accB, 1));

      const int r0 = g * 4 + 0, r1 = g * 4 + 1, r2 = g * 4 + 2, r3 = g * 4 + 3;
      sumf[r0] += dy * (ds[r0] * (float)r0_sum - dm[r0] * row_min[r0]);
      sumf[r1] += dy * (ds[r1] * (float)r1_sum - dm[r1] * row_min[r1]);
      sumf[r2] += dy * (ds[r2] * (float)r2_sum - dm[r2] * row_min[r2]);
      sumf[r3] += dy * (ds[r3] * (float)r3_sum - dm[r3] * row_min[r3]);
    }
  }

  for (int r = 0; r < n_active; ++r) out[r] = sumf[r];
}

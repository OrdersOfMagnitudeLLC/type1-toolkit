// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

// nspack_hwy.hpp - declarations for the Highway SIMD kernels in
// nspack_hwy.cpp. No Highway headers here; callers guard use with
// #if defined(NSPACK_HIGHWAY) and keep the scalar loop as fallback.

#pragma once

#include <cstddef>
#include <cstdint>

namespace nspack_hwy {

// Morton: spread 21-bit axes into interleaved 64-bit codes (same bit
// pattern as scalar split3_64/morton3_64).
void morton3_64_batch(const uint64_t* x, const uint64_t* y, const uint64_t* z,
                      uint64_t* out, size_t n);

// Delta encode/decode over int16 and int32 arrays, in place.
void delta_encode_i16(int16_t* d, size_t n);
void delta_decode_i16(int16_t* d, size_t n);
void delta_encode_i32(int32_t* d, size_t n);
void delta_decode_i32(int32_t* d, size_t n);
void delta_encode_u32(uint32_t* d, size_t n);
void delta_decode_u32(uint32_t* d, size_t n);
void delta_encode_f32(float* d, size_t n);

// Max |d[i] - d[i-1]| as int32 (for delta_encode_i16_safe overflow check).
int32_t delta_max_abs_i16(const int16_t* d, size_t n);

// 2D delta (predictor: left + up - up_left), mod 256.
void delta2d_encode(const uint8_t* src, uint8_t* dst, int w, int h);
void delta2d_decode(const uint8_t* src, uint8_t* dst, int w, int h);

// Temporal delta between two frames, in place on curr.
void temporal_delta_encode(uint8_t* curr, const uint8_t* prev, size_t n);
void temporal_delta_decode(uint8_t* curr, const uint8_t* prev, size_t n);

// Channel separation: dst[i] = src[i*ch + c] and the inverse scatter.
void extract_channel_u8(const uint8_t* src, uint8_t* dst, size_t npix, int ch, int c);
void insert_channel_u8(uint8_t* dst, const uint8_t* src, size_t npix, int ch, int c);

// WAV: dst[i] = (int32_t)src[i*ch + c] and the inverse (truncate to int16).
void deinterleave_i16_i32(const int16_t* src, int32_t* dst, size_t n, int ch, int c);
void interleave_i32_i16(const int32_t* src, int16_t* dst, size_t n, int ch, int c);

} // namespace nspack_hwy

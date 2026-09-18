// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

// nspack_hwy.cpp - Highway SIMD kernels for NSPack hot loops.
// Compiled once per target via foreach_target.h; HWY_DYNAMIC_DISPATCH
// selects the best ISA at runtime. The scalar loops in nspack.cpp and
// nspack_extra.hpp remain the fallback when NSPACK_HIGHWAY is not set.

#include "nspack_hwy.hpp"

#include <vector>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "nspack_hwy.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();

namespace nspack_hwy {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// Scalar helpers for tails (identical ops to nspack_extra.hpp).
static uint64_t split3_64_s(uint64_t x) {
    x = (x | (x << 32)) & 0x1f00000000ffffULL;
    x = (x | (x << 16)) & 0x1f0000ff0000ffULL;
    x = (x | (x << 8))  & 0x100f00f00f00f00fULL;
    x = (x | (x << 4))  & 0x10c30c30c30c30c3ULL;
    x = (x | (x << 2))  & 0x1249249249249249ULL;
    return x;
}

// ===== Morton =====

void morton3_64_batch_impl(const uint64_t* x, const uint64_t* y, const uint64_t* z,
                           uint64_t* out, size_t n) {
    const hn::ScalableTag<uint64_t> d;
    const size_t N = hn::Lanes(d);
    const auto spread = [&](hn::Vec<decltype(d)> v) {
        v = hn::And(hn::Or(v, hn::ShiftLeft<32>(v)), hn::Set(d, 0x1f00000000ffffULL));
        v = hn::And(hn::Or(v, hn::ShiftLeft<16>(v)), hn::Set(d, 0x1f0000ff0000ffULL));
        v = hn::And(hn::Or(v, hn::ShiftLeft<8>(v)),  hn::Set(d, 0x100f00f00f00f00fULL));
        v = hn::And(hn::Or(v, hn::ShiftLeft<4>(v)),  hn::Set(d, 0x10c30c30c30c30c3ULL));
        v = hn::And(hn::Or(v, hn::ShiftLeft<2>(v)),  hn::Set(d, 0x1249249249249249ULL));
        return v;
    };
    size_t i = 0;
    for (; i + N <= n; i += N) {
        const auto vx = spread(hn::LoadU(d, x + i));
        const auto vy = spread(hn::LoadU(d, y + i));
        const auto vz = spread(hn::LoadU(d, z + i));
        hn::StoreU(hn::Or(hn::Or(vx, hn::ShiftLeft<1>(vy)), hn::ShiftLeft<2>(vz)),
                   d, out + i);
    }
    for (; i < n; i++) {
        out[i] = split3_64_s(x[i]) | (split3_64_s(y[i]) << 1) | (split3_64_s(z[i]) << 2);
    }
}

// ===== Delta encode/decode =====
// Encode writes d[i] = d[i] - d[i-1] in place; blocks run backward so a
// store never clobbers a not-yet-read predecessor. Decode is an inclusive
// prefix sum: Hillis-Steele scan inside each vector plus a carry lane.

template <typename T>
void delta_encode_impl_t(T* d_, size_t n) {
    if (n < 2) return;
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    size_t i = n - 1;
    while (i >= N) {
        const auto cur = hn::LoadU(d, d_ + i - N + 1);
        const auto prev = hn::LoadU(d, d_ + i - N);
        hn::StoreU(hn::Sub(cur, prev), d, d_ + i - N + 1);
        i -= N;
    }
    for (; i >= 1; i--) d_[i] = (T)(d_[i] - d_[i - 1]);
}

template <typename T>
void delta_decode_impl_t(T* d_, size_t n) {
    if (n < 2) return;
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    T carry = 0;
    size_t i = 0;
    for (; i + N <= n; i += N) {
        auto v = hn::LoadU(d, d_ + i);
        for (size_t k = 1; k < N; k <<= 1)
            v = hn::Add(v, hn::SlideUpLanes(d, v, k));
        v = hn::Add(v, hn::Set(d, carry));
        hn::StoreU(v, d, d_ + i);
        T last[64]; // >= max lanes for i16/i32 on any target
        hn::StoreU(v, d, last);
        carry = last[N - 1];
    }
    for (; i < n; i++) { carry = (T)(carry + d_[i]); d_[i] = carry; }
}

void delta_encode_i16_impl(int16_t* d, size_t n) { delta_encode_impl_t(d, n); }
void delta_decode_i16_impl(int16_t* d, size_t n) { delta_decode_impl_t(d, n); }
void delta_encode_i32_impl(int32_t* d, size_t n) { delta_encode_impl_t(d, n); }
void delta_decode_i32_impl(int32_t* d, size_t n) { delta_decode_impl_t(d, n); }
void delta_encode_u32_impl(uint32_t* d, size_t n) { delta_encode_impl_t(d, n); }
void delta_decode_u32_impl(uint32_t* d, size_t n) { delta_decode_impl_t(d, n); }
void delta_encode_f32_impl(float* d, size_t n) { delta_encode_impl_t(d, n); }

int32_t delta_max_abs_i16_impl(const int16_t* d_, size_t n) {
    if (n < 2) return 0;
    const hn::ScalableTag<int32_t> d32;
    const hn::Rebind<int16_t, decltype(d32)> d16;
    const size_t N = hn::Lanes(d32);
    auto vmax = hn::Zero(d32);
    size_t i = 1;
    for (; i + N <= n; i += N) {
        const auto cur = hn::PromoteTo(d32, hn::LoadU(d16, d_ + i));
        const auto prev = hn::PromoteTo(d32, hn::LoadU(d16, d_ + i - 1));
        vmax = hn::Max(vmax, hn::Abs(hn::Sub(cur, prev)));
    }
    int32_t m = hn::ReduceMax(d32, vmax);
    for (; i < n; i++) {
        int32_t diff = (int32_t)d_[i] - (int32_t)d_[i - 1];
        if (diff < 0) diff = -diff;
        if (diff > m) m = diff;
    }
    return m;
}

// ===== 2D delta =====
// All arithmetic is mod 256, so u8 lane ops match the scalar int math.

void delta2d_encode_impl(const uint8_t* src, uint8_t* dst, int w, int h) {
    const hn::ScalableTag<uint8_t> d;
    const size_t N = hn::Lanes(d);
    for (int y = 0; y < h; y++) {
        const uint8_t* srow = src + (size_t)y * (size_t)w;
        uint8_t* drow = dst + (size_t)y * (size_t)w;
        const uint8_t* urow = (y > 0) ? src + (size_t)(y - 1) * (size_t)w : nullptr;
        // x = 0: left = ul = 0, pred = up.
        drow[0] = (uint8_t)(srow[0] - (urow ? urow[0] : 0));
        int x = 1;
        for (; x + (int)N <= w; x += (int)N) {
            const auto vs = hn::LoadU(d, srow + x);
            const auto vl = hn::LoadU(d, srow + x - 1);
            const auto vu = urow ? hn::LoadU(d, urow + x) : hn::Zero(d);
            const auto vul = urow ? hn::LoadU(d, urow + x - 1) : hn::Zero(d);
            const auto pred = hn::Sub(hn::Add(vl, vu), vul);
            hn::StoreU(hn::Sub(vs, pred), d, drow + x);
        }
        for (; x < w; x++) {
            int left = srow[x - 1];
            int up = urow ? urow[x] : 0;
            int ul = urow ? urow[x - 1] : 0;
            drow[x] = (uint8_t)(srow[x] - (left + up - ul));
        }
    }
}

void delta2d_decode_impl(const uint8_t* src, uint8_t* dst, int w, int h) {
    const hn::ScalableTag<uint8_t> d;
    const size_t N = hn::Lanes(d);
    std::vector<uint8_t> t((size_t)w);
    for (int y = 0; y < h; y++) {
        const uint8_t* srow = src + (size_t)y * (size_t)w;
        uint8_t* drow = dst + (size_t)y * (size_t)w;
        const uint8_t* urow = (y > 0) ? dst + (size_t)(y - 1) * (size_t)w : nullptr;
        // t[x] = src[x] + up[x] - ul[x]; then dst[x] = dst[x-1] + t[x].
        int x = 0;
        for (; x + (int)N <= w; x += (int)N) {
            const auto vs = hn::LoadU(d, srow + x);
            const auto vu = urow ? hn::LoadU(d, urow + x) : hn::Zero(d);
            // ul[x] = up[x-1]; lane 0 of the row is 0.
            auto vul = hn::Zero(d);
            if (urow) {
                vul = (x == 0) ? hn::SlideUpLanes(d, vu, 1)
                               : hn::LoadU(d, urow + x - 1);
            }
            hn::StoreU(hn::Sub(hn::Add(vs, vu), vul), d, t.data() + x);
        }
        for (; x < w; x++) {
            int up = urow ? urow[x] : 0;
            int ul = (urow && x > 0) ? urow[x - 1] : 0;
            t[x] = (uint8_t)(srow[x] + up - ul);
        }
        uint8_t acc = 0;
        for (x = 0; x < w; x++) {
            acc = (uint8_t)(acc + t[x]);
            drow[x] = acc;
        }
    }
}

// ===== Temporal delta =====

void temporal_delta_encode_impl(uint8_t* curr, const uint8_t* prev, size_t n) {
    const hn::ScalableTag<uint8_t> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= n; i += N)
        hn::StoreU(hn::Sub(hn::LoadU(d, curr + i), hn::LoadU(d, prev + i)), d, curr + i);
    for (; i < n; i++) curr[i] = (uint8_t)(curr[i] - prev[i]);
}

void temporal_delta_decode_impl(uint8_t* curr, const uint8_t* prev, size_t n) {
    const hn::ScalableTag<uint8_t> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= n; i += N)
        hn::StoreU(hn::Add(hn::LoadU(d, curr + i), hn::LoadU(d, prev + i)), d, curr + i);
    for (; i < n; i++) curr[i] = (uint8_t)(curr[i] + prev[i]);
}

// ===== Channel separation (u8) =====

void extract_channel_u8_impl(const uint8_t* src, uint8_t* dst, size_t npix,
                             int ch, int c) {
    const hn::ScalableTag<uint8_t> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    if (ch == 2) {
        for (; i + N <= npix; i += N) {
            auto v0 = hn::Undefined(d), v1 = hn::Undefined(d);
            hn::LoadInterleaved2(d, src + i * 2, v0, v1);
            hn::StoreU(c == 0 ? v0 : v1, d, dst + i);
        }
    } else if (ch == 3) {
        for (; i + N <= npix; i += N) {
            auto v0 = hn::Undefined(d), v1 = hn::Undefined(d), v2 = hn::Undefined(d);
            hn::LoadInterleaved3(d, src + i * 3, v0, v1, v2);
            hn::StoreU(c == 0 ? v0 : (c == 1 ? v1 : v2), d, dst + i);
        }
    } else if (ch == 4) {
        for (; i + N <= npix; i += N) {
            auto v0 = hn::Undefined(d), v1 = hn::Undefined(d);
            auto v2 = hn::Undefined(d), v3 = hn::Undefined(d);
            hn::LoadInterleaved4(d, src + i * 4, v0, v1, v2, v3);
            hn::StoreU(c == 0 ? v0 : (c == 1 ? v1 : (c == 2 ? v2 : v3)), d, dst + i);
        }
    }
    for (; i < npix; i++) dst[i] = src[i * (size_t)ch + (size_t)c];
}

void insert_channel_u8_impl(uint8_t* dst, const uint8_t* src, size_t npix,
                            int ch, int c) {
    const hn::ScalableTag<uint8_t> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    // StoreInterleaved writes every channel, so read-modify-write the
    // interleaved buffer and substitute only channel c.
    if (ch == 2) {
        for (; i + N <= npix; i += N) {
            auto v0 = hn::Undefined(d), v1 = hn::Undefined(d);
            hn::LoadInterleaved2(d, dst + i * 2, v0, v1);
            const auto vc = hn::LoadU(d, src + i);
            if (c == 0) v0 = vc; else v1 = vc;
            hn::StoreInterleaved2(v0, v1, d, dst + i * 2);
        }
    } else if (ch == 3) {
        for (; i + N <= npix; i += N) {
            auto v0 = hn::Undefined(d), v1 = hn::Undefined(d), v2 = hn::Undefined(d);
            hn::LoadInterleaved3(d, dst + i * 3, v0, v1, v2);
            const auto vc = hn::LoadU(d, src + i);
            if (c == 0) v0 = vc; else if (c == 1) v1 = vc; else v2 = vc;
            hn::StoreInterleaved3(v0, v1, v2, d, dst + i * 3);
        }
    } else if (ch == 4) {
        for (; i + N <= npix; i += N) {
            auto v0 = hn::Undefined(d), v1 = hn::Undefined(d);
            auto v2 = hn::Undefined(d), v3 = hn::Undefined(d);
            hn::LoadInterleaved4(d, dst + i * 4, v0, v1, v2, v3);
            const auto vc = hn::LoadU(d, src + i);
            if (c == 0) v0 = vc; else if (c == 1) v1 = vc;
            else if (c == 2) v2 = vc; else v3 = vc;
            hn::StoreInterleaved4(v0, v1, v2, v3, d, dst + i * 4);
        }
    }
    for (; i < npix; i++) dst[i * (size_t)ch + (size_t)c] = src[i];
}

// ===== WAV channel separation (i16 <-> i32) =====

void deinterleave_i16_i32_impl(const int16_t* src, int32_t* dst, size_t n,
                               int ch, int c) {
    size_t i = 0;
    if (ch == 2) {
        const hn::ScalableTag<int32_t> d32;
        const hn::Rebind<int16_t, decltype(d32)> d16;
        const size_t N = hn::Lanes(d32);
        for (; i + N <= n; i += N) {
            auto v0 = hn::Undefined(d16), v1 = hn::Undefined(d16);
            hn::LoadInterleaved2(d16, src + i * 2, v0, v1);
            hn::StoreU(hn::PromoteTo(d32, c == 0 ? v0 : v1), d32, dst + i);
        }
    }
    for (; i < n; i++) dst[i] = (int32_t)src[i * (size_t)ch + (size_t)c];
}

void interleave_i32_i16_impl(const int32_t* src, int16_t* dst, size_t n,
                             int ch, int c) {
    size_t i = 0;
    if (ch == 2) {
        const hn::ScalableTag<int32_t> d32;
        const hn::Rebind<int16_t, decltype(d32)> d16;
        const hn::RebindToUnsigned<decltype(d32)> du32;
        const hn::RebindToUnsigned<decltype(d16)> du16;
        const size_t N = hn::Lanes(d32);
        for (; i + N <= n; i += N) {
            auto v0 = hn::Undefined(d16), v1 = hn::Undefined(d16);
            hn::LoadInterleaved2(d16, dst + i * 2, v0, v1);
            // Truncate low 16 bits, matching C++ int32 -> int16 conversion.
            const auto vc = hn::BitCast(d16,
                hn::TruncateTo(du16, hn::BitCast(du32, hn::LoadU(d32, src + i))));
            if (c == 0) v0 = vc; else v1 = vc;
            hn::StoreInterleaved2(v0, v1, d16, dst + i * 2);
        }
    }
    for (; i < n; i++) dst[i * (size_t)ch + (size_t)c] = (int16_t)src[i];
}

} // namespace HWY_NAMESPACE
} // namespace nspack_hwy

HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace nspack_hwy {

HWY_EXPORT(morton3_64_batch_impl);
HWY_EXPORT(delta_encode_i16_impl);
HWY_EXPORT(delta_decode_i16_impl);
HWY_EXPORT(delta_encode_i32_impl);
HWY_EXPORT(delta_decode_i32_impl);
HWY_EXPORT(delta_encode_u32_impl);
HWY_EXPORT(delta_decode_u32_impl);
HWY_EXPORT(delta_encode_f32_impl);
HWY_EXPORT(delta_max_abs_i16_impl);
HWY_EXPORT(delta2d_encode_impl);
HWY_EXPORT(delta2d_decode_impl);
HWY_EXPORT(temporal_delta_encode_impl);
HWY_EXPORT(temporal_delta_decode_impl);
HWY_EXPORT(extract_channel_u8_impl);
HWY_EXPORT(insert_channel_u8_impl);
HWY_EXPORT(deinterleave_i16_i32_impl);
HWY_EXPORT(interleave_i32_i16_impl);

void morton3_64_batch(const uint64_t* x, const uint64_t* y, const uint64_t* z,
                      uint64_t* out, size_t n) {
    HWY_DYNAMIC_DISPATCH(morton3_64_batch_impl)(x, y, z, out, n);
}
void delta_encode_i16(int16_t* d, size_t n) {
    HWY_DYNAMIC_DISPATCH(delta_encode_i16_impl)(d, n);
}
void delta_decode_i16(int16_t* d, size_t n) {
    HWY_DYNAMIC_DISPATCH(delta_decode_i16_impl)(d, n);
}
void delta_encode_i32(int32_t* d, size_t n) {
    HWY_DYNAMIC_DISPATCH(delta_encode_i32_impl)(d, n);
}
void delta_decode_i32(int32_t* d, size_t n) {
    HWY_DYNAMIC_DISPATCH(delta_decode_i32_impl)(d, n);
}
void delta_encode_u32(uint32_t* d, size_t n) {
    HWY_DYNAMIC_DISPATCH(delta_encode_u32_impl)(d, n);
}
void delta_decode_u32(uint32_t* d, size_t n) {
    HWY_DYNAMIC_DISPATCH(delta_decode_u32_impl)(d, n);
}
void delta_encode_f32(float* d, size_t n) {
    HWY_DYNAMIC_DISPATCH(delta_encode_f32_impl)(d, n);
}
int32_t delta_max_abs_i16(const int16_t* d, size_t n) {
    return HWY_DYNAMIC_DISPATCH(delta_max_abs_i16_impl)(d, n);
}
void delta2d_encode(const uint8_t* src, uint8_t* dst, int w, int h) {
    HWY_DYNAMIC_DISPATCH(delta2d_encode_impl)(src, dst, w, h);
}
void delta2d_decode(const uint8_t* src, uint8_t* dst, int w, int h) {
    HWY_DYNAMIC_DISPATCH(delta2d_decode_impl)(src, dst, w, h);
}
void temporal_delta_encode(uint8_t* curr, const uint8_t* prev, size_t n) {
    HWY_DYNAMIC_DISPATCH(temporal_delta_encode_impl)(curr, prev, n);
}
void temporal_delta_decode(uint8_t* curr, const uint8_t* prev, size_t n) {
    HWY_DYNAMIC_DISPATCH(temporal_delta_decode_impl)(curr, prev, n);
}
void extract_channel_u8(const uint8_t* src, uint8_t* dst, size_t npix, int ch, int c) {
    HWY_DYNAMIC_DISPATCH(extract_channel_u8_impl)(src, dst, npix, ch, c);
}
void insert_channel_u8(uint8_t* dst, const uint8_t* src, size_t npix, int ch, int c) {
    HWY_DYNAMIC_DISPATCH(insert_channel_u8_impl)(dst, src, npix, ch, c);
}
void deinterleave_i16_i32(const int16_t* src, int32_t* dst, size_t n, int ch, int c) {
    HWY_DYNAMIC_DISPATCH(deinterleave_i16_i32_impl)(src, dst, n, ch, c);
}
void interleave_i32_i16(const int32_t* src, int16_t* dst, size_t n, int ch, int c) {
    HWY_DYNAMIC_DISPATCH(interleave_i32_i16_impl)(src, dst, n, ch, c);
}

} // namespace nspack_hwy

#endif // HWY_ONCE

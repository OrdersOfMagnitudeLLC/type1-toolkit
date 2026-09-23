// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#ifndef NSNET_HPP
#define NSNET_HPP

#include <hwy/highway.h>
#include <immintrin.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

// ---------------------------------------------------------------------------
// NSNet: compile-time packet schema extraction.
//
// Ethernet/IPv4/TCP/UDP headers are fixed-format: every field lives at a
// known byte offset. DPDK et al. treat packets as opaque bytes and parse at
// runtime; NSNet encodes the schema as constexpr offsets and extracts fields
// from 16 packets at once with AVX-512 gathers.
//
// Fast path: Ethernet + IPv4 (no options) + TCP or UDP, fixed stride.
// UNKNOWN / IPv6 / extension headers / TCP options: no fast path (documented).
// ---------------------------------------------------------------------------

// Compile-time schemas: every field is a constexpr byte offset.
struct NSEthernetSchema {
    static constexpr size_t DST_MAC   = 0;
    static constexpr size_t SRC_MAC   = 6;
    static constexpr size_t ETHERTYPE = 12;  // 2 bytes, network order
    static constexpr size_t HEADER    = 14;
};

struct NSIPv4Schema {
    static constexpr size_t BASE      = NSEthernetSchema::HEADER; // 14
    static constexpr size_t VER_IHL   = BASE + 0;
    static constexpr size_t TOTAL_LEN = BASE + 2;   // 16
    static constexpr size_t TTL       = BASE + 8;   // 22
    static constexpr size_t PROTOCOL  = BASE + 9;   // 23
    static constexpr size_t SRC_IP    = BASE + 12;  // 26
    static constexpr size_t DST_IP    = BASE + 16;  // 30
    static constexpr size_t HEADER    = BASE + 20;  // 34 (IHL=5, no options)
};

struct NSTCPSchema {
    static constexpr size_t BASE      = NSIPv4Schema::HEADER;     // 34
    static constexpr size_t SRC_PORT  = BASE + 0;
    static constexpr size_t DST_PORT  = BASE + 2;
    static constexpr size_t PORTS     = BASE + 0;   // 4-byte span: src|dst
    static constexpr size_t SEQ       = BASE + 4;
    static constexpr size_t ACK       = BASE + 8;
    static constexpr size_t FLAGS     = BASE + 12;
};

struct NSUDPSchema {
    static constexpr size_t BASE      = NSIPv4Schema::HEADER;     // 34
    static constexpr size_t SRC_PORT  = BASE + 0;
    static constexpr size_t DST_PORT  = BASE + 2;
    static constexpr size_t PORTS     = BASE + 0;
    static constexpr size_t LENGTH    = BASE + 4;
};

enum NSNetSchemaType {
    NSNET_UNKNOWN = 0,
    NSNET_ETH_IP_TCP,
    NSNET_ETH_IP_UDP
};

// Schema detection from fixed offsets: ethertype 0x0800 (IPv4), IHL=5,
// protocol 6 (TCP) / 17 (UDP). Anything else gates off the fast path.
static inline NSNetSchemaType ns_net_detect(const uint8_t* pkt) {
    if (pkt[NSEthernetSchema::ETHERTYPE] != 0x08 ||
        pkt[NSEthernetSchema::ETHERTYPE + 1] != 0x00)
        return NSNET_UNKNOWN;
    if ((pkt[NSIPv4Schema::VER_IHL] & 0x0F) != 5)  // IHL != 5: options present
        return NSNET_UNKNOWN;
    uint8_t proto = pkt[NSIPv4Schema::PROTOCOL];
    if (proto == 6)  return NSNET_ETH_IP_TCP;
    if (proto == 17) return NSNET_ETH_IP_UDP;
    return NSNET_UNKNOWN;
}

// Extracted fields, struct-of-arrays. One slot per packet.
struct NSNetFields {
    uint32_t* src_ip;    // host order
    uint32_t* dst_ip;    // host order
    uint16_t* src_port;  // host order
    uint16_t* dst_port;  // host order
    uint8_t*  protocol;
    uint16_t* length;    // IPv4 total length, host order
    size_t count;

    NSNetFields() : src_ip(nullptr), dst_ip(nullptr), src_port(nullptr),
        dst_port(nullptr), protocol(nullptr), length(nullptr), count(0) {}
    ~NSNetFields() { release(); }
    NSNetFields(const NSNetFields&) = delete;
    NSNetFields& operator=(const NSNetFields&) = delete;

    void release() {
        free(src_ip); free(dst_ip); free(src_port); free(dst_port);
        free(protocol); free(length);
        src_ip = dst_ip = nullptr; src_port = dst_port = nullptr;
        protocol = nullptr; length = nullptr; count = 0;
    }

    bool allocate(size_t n) {
        release();
        count = n;
        src_ip   = (uint32_t*)aligned_alloc(64, ((n * 4 + 63) & ~(size_t)63));
        dst_ip   = (uint32_t*)aligned_alloc(64, ((n * 4 + 63) & ~(size_t)63));
        src_port = (uint16_t*)aligned_alloc(64, ((n * 2 + 63) & ~(size_t)63));
        dst_port = (uint16_t*)aligned_alloc(64, ((n * 2 + 63) & ~(size_t)63));
        protocol = (uint8_t*) aligned_alloc(64, ((n + 63) & ~(size_t)63));
        length   = (uint16_t*)aligned_alloc(64, ((n * 2 + 63) & ~(size_t)63));
        return src_ip && dst_ip && src_port && dst_port && protocol && length;
    }
};

static inline uint16_t ns_bswap16(uint32_t x) {
    return (uint16_t)((x >> 8) | ((x & 0xFF) << 8));
}

// Scalar fallback: extract one packet's fields via constexpr offsets.
static inline void ns_net_extract_one(const uint8_t* pkt, size_t i,
                                      NSNetFields& out) {
    uint32_t sip, dip, ports, plen;
    memcpy(&sip,   pkt + NSIPv4Schema::SRC_IP,    4);
    memcpy(&dip,   pkt + NSIPv4Schema::DST_IP,    4);
    memcpy(&ports, pkt + NSTCPSchema::PORTS,      4);
    memcpy(&plen,  pkt + NSIPv4Schema::TOTAL_LEN, 4);
    out.src_ip[i]   = __builtin_bswap32(sip);
    out.dst_ip[i]   = __builtin_bswap32(dip);
    out.src_port[i] = ns_bswap16(ports & 0xFFFF);
    out.dst_port[i] = ns_bswap16(ports >> 16);
    out.protocol[i] = pkt[NSIPv4Schema::PROTOCOL];
    out.length[i]   = ns_bswap16(plen & 0xFFFF);
}

// Batch extraction: one aligned 64B load per packet, fields pulled by
// a single permutexvar_epi8 into a 16B record (host order baked into the
// index vector — byte swap is free). 16 records are then transposed to
// SoA with permutex2var. No gathers: pure load + shuffle, ~4 uops/packet.
// Packets must be contiguous with fixed stride (>= 64 for 64B frames).
// Caller must have verified schema with ns_net_detect on each packet
// (or generated them uniformly) — UNKNOWN packets get garbage fields.

// Stage-1 index: dest byte j <- packet byte PK_IDX[j].
// Record layout: [sip(4) dip(4) sport(2) dport(2) len(2) proto(1) pad(1)]
// all host-order (indices already byte-reversed).
static const uint8_t NS_PK_IDX[64] __attribute__((aligned(64))) = {
    29,28,27,26,  33,32,31,30,  35,34, 37,36,  17,16, 23, 0,
    0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0
};

// Stage-2 transpose indices (two-level permutex2var over 4 record regs).
static const uint32_t NS_SIP01[16] __attribute__((aligned(64))) =
    {0,4,8,12, 16,20,24,28, 0,0,0,0,0,0,0,0};
static const uint32_t NS_DIP01[16] __attribute__((aligned(64))) =
    {1,5,9,13, 17,21,25,29, 0,0,0,0,0,0,0,0};
static const uint32_t NS_MRG32[16] __attribute__((aligned(64))) =
    {0,1,2,3,4,5,6,7, 16,17,18,19,20,21,22,23};
static const uint16_t NS_SPT01[32] __attribute__((aligned(64))) =
    {4,12,20,28, 36,44,52,60, 0,0,0,0,0,0,0,0,
     0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0};
static const uint16_t NS_DPT01[32] __attribute__((aligned(64))) =
    {5,13,21,29, 37,45,53,61, 0,0,0,0,0,0,0,0,
     0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0};
static const uint16_t NS_LEN01[32] __attribute__((aligned(64))) =
    {6,14,22,30, 38,46,54,62, 0,0,0,0,0,0,0,0,
     0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0};
static const uint16_t NS_MRG16[32] __attribute__((aligned(64))) =
    {0,1,2,3,4,5,6,7, 32,33,34,35,36,37,38,39,
     0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0};
static const uint8_t NS_PROTO01[64] __attribute__((aligned(64))) = {
    14,30,46,62, 78,94,110,126, 0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0
};
static const uint8_t NS_MRG8[64] __attribute__((aligned(64))) = {
    0,1,2,3,4,5,6,7, 64,65,66,67,68,69,70,71,
    0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0
};

static HWY_ATTR void ns_net_extract_batch(const uint8_t* pkts, size_t stride,
                                 size_t count, NSNetFields& out) {
    size_t i = 0;
#if defined(__AVX512F__) && defined(__AVX512VBMI__)
    const __m512i pk_idx  = _mm512_load_si512(NS_PK_IDX);
    const __m512i sip01   = _mm512_load_si512(NS_SIP01);
    const __m512i dip01   = _mm512_load_si512(NS_DIP01);
    const __m512i mrg32   = _mm512_load_si512(NS_MRG32);
    const __m512i spt01   = _mm512_load_si512(NS_SPT01);
    const __m512i dpt01   = _mm512_load_si512(NS_DPT01);
    const __m512i len01   = _mm512_load_si512(NS_LEN01);
    const __m512i mrg16   = _mm512_load_si512(NS_MRG16);
    const __m512i proto01 = _mm512_load_si512(NS_PROTO01);
    const __m512i mrg8    = _mm512_load_si512(NS_MRG8);

    alignas(64) uint8_t staging[256];  // 16 x 16B records

    for (; i + 16 <= count; i += 16) {
        // Stage 1: 16 loads + 16 permutes -> 16 compact records.
        for (int j = 0; j < 16; ++j) {
            __m512i pkt = _mm512_loadu_si512(pkts + (i + j) * stride);
            __m512i rec = _mm512_permutexvar_epi8(pk_idx, pkt);
            _mm_store_si128((__m128i*)(staging + j * 16),
                            _mm512_castsi512_si128(rec));
        }

        // Stage 2: transpose 16 records -> SoA via two-level permutex2var.
        __m512i r0 = _mm512_load_si512(staging);
        __m512i r1 = _mm512_load_si512(staging + 64);
        __m512i r2 = _mm512_load_si512(staging + 128);
        __m512i r3 = _mm512_load_si512(staging + 192);

        __m512i s01 = _mm512_permutex2var_epi32(r0, sip01, r1);
        __m512i s23 = _mm512_permutex2var_epi32(r2, sip01, r3);
        _mm512_storeu_si512(out.src_ip + i,
                            _mm512_permutex2var_epi32(s01, mrg32, s23));

        __m512i d01 = _mm512_permutex2var_epi32(r0, dip01, r1);
        __m512i d23 = _mm512_permutex2var_epi32(r2, dip01, r3);
        _mm512_storeu_si512(out.dst_ip + i,
                            _mm512_permutex2var_epi32(d01, mrg32, d23));

        __m512i p01 = _mm512_permutex2var_epi16(r0, spt01, r1);
        __m512i p23 = _mm512_permutex2var_epi16(r2, spt01, r3);
        _mm256_storeu_si256((__m256i*)(out.src_port + i),
            _mm512_castsi512_si256(
                _mm512_permutex2var_epi16(p01, mrg16, p23)));

        __m512i q01 = _mm512_permutex2var_epi16(r0, dpt01, r1);
        __m512i q23 = _mm512_permutex2var_epi16(r2, dpt01, r3);
        _mm256_storeu_si256((__m256i*)(out.dst_port + i),
            _mm512_castsi512_si256(
                _mm512_permutex2var_epi16(q01, mrg16, q23)));

        __m512i l01 = _mm512_permutex2var_epi16(r0, len01, r1);
        __m512i l23 = _mm512_permutex2var_epi16(r2, len01, r3);
        _mm256_storeu_si256((__m256i*)(out.length + i),
            _mm512_castsi512_si256(
                _mm512_permutex2var_epi16(l01, mrg16, l23)));

        __m512i t01 = _mm512_permutex2var_epi8(r0, proto01, r1);
        __m512i t23 = _mm512_permutex2var_epi8(r2, proto01, r3);
        _mm_storeu_si128((__m128i*)(out.protocol + i),
            _mm512_castsi512_si128(
                _mm512_permutex2var_epi8(t01, mrg8, t23)));
    }
#endif
    for (; i < count; ++i)
        ns_net_extract_one(pkts + i * stride, i, out);
}

#endif // NSNET_HPP

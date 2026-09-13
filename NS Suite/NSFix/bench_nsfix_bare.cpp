// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

// Bare-bones NSFix throughput benchmark.
// Strips away overhead layers to measure how close we can get to the
// memory bandwidth (Shannon) limit. Does NOT replace existing benchmarks.

#include "nsfix.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <immintrin.h>

using namespace nsfix;

static const int NUM_MESSAGES = 100000;
static const char SOH = '\x01';

// ---------------------------------------------------------------------------
// Message generation (same as bench_nsfix.cpp)
// ---------------------------------------------------------------------------
static uint32_t calc_checksum(const std::string& msg) {
    uint32_t sum = 0;
    for (char c : msg) sum += static_cast<uint8_t>(c);
    return sum % 256;
}

static std::string gen_new_order_single(int seq) {
    std::ostringstream body;
    body << "35=D" << SOH
         << "49=CLIENT" << SOH
         << "56=BROKER" << SOH
         << "34=" << seq << SOH
         << "52=20240101-12:00:00" << SOH
         << "11=ORD" << seq << SOH
         << "55=AAPL" << SOH
         << "54=1" << SOH
         << "38=100" << SOH
         << "44=150.50" << SOH
         << "40=2" << SOH;
    std::string body_str = body.str();

    std::ostringstream oss;
    oss << "8=FIX.4.2" << SOH
        << "9=" << body_str.length() << SOH
        << body_str;
    std::string msg = oss.str();

    uint32_t ck = calc_checksum(msg);
    oss.str("");
    oss << "10=" << std::setw(3) << std::setfill('0') << ck << SOH;
    msg += oss.str();
    return msg;
}

// ---------------------------------------------------------------------------
// Contiguous buffer: all messages concatenated (no pointer chasing)
// ---------------------------------------------------------------------------
struct MsgBuf {
    std::vector<char> data;
    std::vector<uint32_t> offsets;
    std::vector<uint32_t> lengths;
};

static MsgBuf build_contiguous(int count) {
    MsgBuf mb;
    mb.offsets.reserve(count);
    mb.lengths.reserve(count);
    for (int i = 2; i < count + 2; i++) {
        std::string msg = gen_new_order_single(i);
        uint32_t off = mb.data.size();
        mb.data.insert(mb.data.end(), msg.begin(), msg.end());
        mb.offsets.push_back(off);
        mb.lengths.push_back(static_cast<uint32_t>(msg.length()));
    }
    return mb;
}

static volatile int g_sink = 0;

// ---------------------------------------------------------------------------
// Layer 0: Raw memory scan: AVX2 SOH count.
//          Shannon limit: pure memory bandwidth, zero parsing.
// ---------------------------------------------------------------------------
static int bench_raw_scan(const MsgBuf& mb, int iters) {
    const char* buf = mb.data.data();
    size_t total_len = mb.data.size();
    int count = 0;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iters; iter++) {
        const __m256i soh_vec = _mm256_set1_epi8(SOH);
        size_t pos = 0;
        int local = 0;
        while (pos + 32 <= total_len) {
            __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(buf + pos));
            int mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, soh_vec));
            local += __builtin_popcount(mask);
            pos += 32;
        }
        for (; pos < total_len; pos++)
            if (buf[pos] == SOH) local++;
        g_sink = local;
        count = local;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    double mps = (double)count * iters / us * 1e6;
    double gbps = (double)total_len * iters / us * 1e6 / 1e9;
    std::cout << "Layer 0 (raw AVX2 SOH scan):  " << std::fixed << std::setprecision(0)
              << mps << " SOH/sec  (" << gbps << " GB/sec)" << std::endl;
    return count;
}
// ---------------------------------------------------------------------------
// Layer 0a: AVX-512 raw memory scan: 64 bytes/cycle.
//           Same as Layer 0 but with AVX-512 for 2x wider SIMD.
// ---------------------------------------------------------------------------
static int bench_raw_scan_avx512(const MsgBuf& mb, int iters) {
    const char* buf = mb.data.data();
    size_t total_len = mb.data.size();
    int count = 0;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iters; iter++) {
        const __m512i soh_vec = _mm512_set1_epi8(SOH);
        size_t pos = 0;
        int local = 0;
        while (pos + 64 <= total_len) {
            __m512i chunk = _mm512_loadu_si512(reinterpret_cast<const void*>(buf + pos));
            __mmask64 mask = _mm512_cmpeq_epi8_mask(chunk, soh_vec);
            local += __builtin_popcountll(mask);
            pos += 64;
        }
        for (; pos < total_len; pos++)
            if (buf[pos] == SOH) local++;
        g_sink = local;
        count = local;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    double mps = (double)count * iters / us * 1e6;
    double gbps = (double)total_len * iters / us * 1e6 / 1e9;
    std::cout << "Layer 0a (AVX-512 raw scan):   " << std::fixed << std::setprecision(0)
              << mps << " SOH/sec  (" << gbps << " GB/sec)" << std::endl;
    return count;
}


// ---------------------------------------------------------------------------
// Layer 1: AVX2 message boundary detection: scan for SOH+"10=" to count
//          complete messages. No field extraction.
// ---------------------------------------------------------------------------
static int bench_msg_count_avx(const MsgBuf& mb, int iters) {
    const char* buf = mb.data.data();
    size_t total_len = mb.data.size();
    int count = 0;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iters; iter++) {
        const __m256i soh_vec = _mm256_set1_epi8(SOH);
        size_t pos = 0;
        int local = 0;
        while (pos + 32 <= total_len) {
            __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(buf + pos));
            int mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, soh_vec));
            while (mask) {
                int bit = __builtin_ctz(mask);
                size_t soh_pos = pos + bit;
                if (soh_pos + 3 < total_len &&
                    buf[soh_pos + 1] == '1' && buf[soh_pos + 2] == '0' &&
                    buf[soh_pos + 3] == '=')
                    local++;
                mask &= mask - 1;
            }
            pos += 32;
        }
        for (; pos < total_len; pos++) {
            if (buf[pos] == SOH && pos + 3 < total_len &&
                buf[pos + 1] == '1' && buf[pos + 2] == '0' && buf[pos + 3] == '=')
                local++;
        }
        g_sink = local;
        count = local;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    double mps = (double)count * iters / us * 1e6;
    double gbps = (double)total_len * iters / us * 1e6 / 1e9;
    std::cout << "Layer 1 (AVX2 msg boundary):  " << std::fixed << std::setprecision(0)
              << mps << " msg/sec  (" << gbps << " GB/sec)" << std::endl;
    return count;
}

// ---------------------------------------------------------------------------
// Layer 2: Bare-bones parser: AVX2 scan + minimal field extraction.
//          No Message struct (520 bytes), no Session, no handle_message,
//          no dispatch array. Compact 56-byte output struct (1 cache line).
//          Extracts only: MsgType, Symbol, Side, OrderQty, Price, SeqNum.
//          This is what a maximally optimized FIX parser hot path looks like.
// ---------------------------------------------------------------------------

// Compact parsed message: 7 string_views = 56 bytes, fits in 1 cache line
struct BareMsg {
    std::string_view msg_type;
    std::string_view symbol;
    std::string_view side;
    std::string_view order_qty;
    std::string_view price;
    std::string_view seq_num;
    std::string_view sender;
};

static int g_bare_count = 0;
static int g_bare_sink = 0;

static void bare_callback(const BareMsg& m) {
    if (m.msg_type.empty()) return;
    g_bare_sink += (int)m.msg_type[0] + m.symbol.length() + m.side.length() +
                   m.order_qty.length() + m.price.length() + m.seq_num.length();
    g_bare_count++;
}

static int bench_bare_parse(const MsgBuf& mb, int iters) {
    const char* buf = mb.data.data();
    size_t total_len = mb.data.size();
    int count = 0;

    // Warmup
    for (int w = 0; w < 3; w++) {
        int local = 0;
        size_t field_start = 0;
        BareMsg msg{};
        for (size_t i = 0; i < total_len; i++) {
            if (buf[i] == SOH) {
                size_t eq = field_start;
                while (eq < i && buf[eq] != '=') eq++;
                if (eq < i) {
                    const char* tp = buf + field_start;
                    uint8_t tag_len = eq - field_start;
                    uint16_t tag_key = (tag_len >= 2) ?
                        (uint16_t)(tp[0] | (tp[1] << 8)) : (uint16_t)tp[0];
                    if (tag_key == ('1'|('0'<<8))) { local++; msg = {}; }
                }
                field_start = i + 1;
            }
        }
        g_sink = local;
    }

    g_bare_count = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iters; iter++) {
        // AVX2 scan for SOH and '=' positions, process fields incrementally
        const __m256i soh_vec = _mm256_set1_epi8(SOH);
        const __m256i eq_vec  = _mm256_set1_epi8('=');

        size_t pos = 0;
        int local = 0;
        size_t field_start = 0;
        BareMsg msg{};

        // Track last '=' position found (for current field)
        size_t last_eq = (size_t)-1;

        while (pos + 32 <= total_len) {
            __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(buf + pos));

            int soh_mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, soh_vec));
            int eq_mask  = _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, eq_vec));

            // Process SOH and EQ events in position order
            while (soh_mask || eq_mask) {
                int soh_bit = soh_mask ? __builtin_ctz(soh_mask) : 32;
                int eq_bit  = eq_mask  ? __builtin_ctz(eq_mask)  : 32;

                if (eq_bit < soh_bit) {
                    // EQ event: remember position
                    last_eq = pos + eq_bit;
                    eq_mask &= eq_mask - 1;
                } else {
                    // SOH event: field complete
                    size_t soh_pos = pos + soh_bit;

                    if (last_eq != (size_t)-1 && last_eq >= field_start && last_eq < soh_pos) {
                        const char* tp = buf + field_start;
                        uint8_t tag_len = last_eq - field_start;
                        uint16_t tag_key = (tag_len >= 2) ?
                            (uint16_t)(tp[0] | (tp[1] << 8)) : (uint16_t)tp[0];

                        const char* val = buf + last_eq + 1;
                        uint16_t val_len = soh_pos - last_eq - 1;

                        switch (tag_key) {
                            case '3'|('5'<<8): msg.msg_type  = {val, val_len}; break; // 35
                            case '5'|('5'<<8): msg.symbol    = {val, val_len}; break; // 55
                            case '5'|('4'<<8): msg.side      = {val, val_len}; break; // 54
                            case '3'|('8'<<8): msg.order_qty = {val, val_len}; break; // 38
                            case '4'|('4'<<8): msg.price     = {val, val_len}; break; // 44
                            case '3'|('4'<<8): msg.seq_num   = {val, val_len}; break; // 34
                            case '4'|('9'<<8): msg.sender    = {val, val_len}; break; // 49
                            case '1'|('0'<<8): // tag 10 = end of message
                                bare_callback(msg);
                                local++;
                                msg = {};
                                break;
                        }
                    }
                    field_start = soh_pos + 1;
                    soh_mask &= soh_mask - 1;
                }
            }

            pos += 32;
        }

        // Scalar tail
        for (; pos < total_len; pos++) {
            if (buf[pos] == '=' && last_eq < field_start) {
                last_eq = pos;
            } else if (buf[pos] == SOH) {
                if (last_eq != (size_t)-1 && last_eq >= field_start && last_eq < pos) {
                    const char* tp = buf + field_start;
                    uint8_t tag_len = last_eq - field_start;
                    uint16_t tag_key = (tag_len >= 2) ?
                        (uint16_t)(tp[0] | (tp[1] << 8)) : (uint16_t)tp[0];
                    const char* val = buf + last_eq + 1;
                    uint16_t val_len = pos - last_eq - 1;

                    switch (tag_key) {
                        case '3'|('5'<<8): msg.msg_type  = {val, val_len}; break;
                        case '5'|('5'<<8): msg.symbol    = {val, val_len}; break;
                        case '5'|('4'<<8): msg.side      = {val, val_len}; break;
                        case '3'|('8'<<8): msg.order_qty = {val, val_len}; break;
                        case '4'|('4'<<8): msg.price     = {val, val_len}; break;
                        case '3'|('4'<<8): msg.seq_num   = {val, val_len}; break;
                        case '4'|('9'<<8): msg.sender    = {val, val_len}; break;
                        case '1'|('0'<<8):
                            bare_callback(msg);
                            local++;
                            msg = {};
                            break;
                    }
                }
                field_start = pos + 1;
                last_eq = (size_t)-1;
            }
        }

        g_sink = local;
        count = local;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    double mps = (double)count * iters / us * 1e6;
    double gbps = (double)total_len * iters / us * 1e6 / 1e9;
    std::cout << "Layer 2 (bare-bones parse):  " << std::fixed << std::setprecision(0)
              << mps << " msg/sec  (" << gbps << " GB/sec)" << std::endl;
    return count;
}

// ---------------------------------------------------------------------------
// Layer 2a: AVX-512 two-phase parser.
//          Phase 1: AVX-512 scan 64 bytes/cycle, collect SOH+EQ positions
//                   into L1-resident arrays (8K entries = 16KB). No branching.
//          Phase 2: Scalar field extraction from position arrays. No SIMD
//                   interleaving: pure scalar at full throughput.
//          Processing in 64KB chunks keeps position arrays in L1.
// ---------------------------------------------------------------------------
static int g_avx512_count = 0;
static int g_avx512_sink = 0;

static int bench_avx512_twophase(const MsgBuf& mb, int iters) {
    const char* buf = mb.data.data();
    size_t total_len = mb.data.size();
    int count = 0;

    static constexpr size_t CHUNK = 65536;
    static constexpr int MAX_POS = 8192;
    uint16_t soh_pos[MAX_POS];
    uint16_t eq_pos[MAX_POS];

    // Warmup
    for (int w = 0; w < 3; w++) {
        int local = 0;
        size_t base = 0;
        while (base < total_len) {
            size_t wend = std::min(base + CHUNK, total_len);
            const __m512i soh_v = _mm512_set1_epi8(SOH);
            const __m512i eq_v  = _mm512_set1_epi8('=');
            int sc = 0, ec = 0;
            size_t pos = base;
            while (pos + 64 <= wend) {
                __m512i ch = _mm512_loadu_si512(reinterpret_cast<const void*>(buf + pos));
                __mmask64 sm = _mm512_cmpeq_epi8_mask(ch, soh_v);
                __mmask64 em = _mm512_cmpeq_epi8_mask(ch, eq_v);
                while (sm && sc < MAX_POS) { soh_pos[sc++] = (uint16_t)(pos + __builtin_ctzll(sm) - base); sm &= sm-1; }
                while (em && ec < MAX_POS) { eq_pos[ec++]  = (uint16_t)(pos + __builtin_ctzll(em) - base);  em &= em-1; }
                pos += 64;
            }
            for (; pos < wend; pos++) {
                if (buf[pos] == SOH && sc < MAX_POS) soh_pos[sc++] = (uint16_t)(pos - base);
                if (buf[pos] == '='  && ec < MAX_POS) eq_pos[ec++]  = (uint16_t)(pos - base);
            }
            size_t fs = base; int ei = 0; BareMsg msg{};
            for (int fi = 0; fi < sc; fi++) {
                size_t soh_abs = base + soh_pos[fi];
                while (ei < ec && (base + eq_pos[ei]) < fs) ei++;
                if (ei >= ec) { fs = soh_abs + 1; continue; }
                size_t eq_abs = base + eq_pos[ei];
                if (eq_abs >= soh_abs) { fs = soh_abs + 1; continue; }
                ei++;
                const char* tp = buf + fs;
                uint8_t tl = eq_abs - fs;
                uint16_t tk = (tl >= 2) ? (uint16_t)(tp[0] | (tp[1] << 8)) : (uint16_t)tp[0];
                if (tk == ('1'|('0'<<8))) { local++; msg = {}; }
                fs = soh_abs + 1;
            }
            base = wend;
        }
        g_sink = local;
    }

    g_bare_count = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iters; iter++) {
        int local = 0;
        size_t base = 0;
        size_t fs = 0;       // field_start carried across chunks
        BareMsg msg{};       // current message carried across chunks
        while (base < total_len) {
            size_t wend = std::min(base + CHUNK, total_len);

            // Phase 1: AVX-512 scan: 64 bytes/cycle, zero branching
            const __m512i soh_v = _mm512_set1_epi8(SOH);
            const __m512i eq_v  = _mm512_set1_epi8('=');
            int sc = 0, ec = 0;
            size_t pos = base;
            while (pos + 64 <= wend) {
                __m512i ch = _mm512_loadu_si512(reinterpret_cast<const void*>(buf + pos));
                __mmask64 sm = _mm512_cmpeq_epi8_mask(ch, soh_v);
                __mmask64 em = _mm512_cmpeq_epi8_mask(ch, eq_v);
                while (sm && sc < MAX_POS) {
                    soh_pos[sc++] = (uint16_t)(pos + __builtin_ctzll(sm) - base);
                    sm &= sm - 1;
                }
                while (em && ec < MAX_POS) {
                    eq_pos[ec++] = (uint16_t)(pos + __builtin_ctzll(em) - base);
                    em &= em - 1;
                }
                pos += 64;
            }
            for (; pos < wend; pos++) {
                if (buf[pos] == SOH && sc < MAX_POS) soh_pos[sc++] = (uint16_t)(pos - base);
                if (buf[pos] == '='  && ec < MAX_POS) eq_pos[ec++]  = (uint16_t)(pos - base);
            }

            // Phase 2: scalar field extraction from L1-resident arrays
            int ei = 0;
            for (int fi = 0; fi < sc; fi++) {
                size_t soh_abs = base + soh_pos[fi];

                // Find EQ for this field
                size_t eq_abs;
                if (fs < base) {
                    // Field started in previous chunk: scan for EQ
                    eq_abs = fs;
                    while (eq_abs < soh_abs && buf[eq_abs] != '=') eq_abs++;
                    if (eq_abs >= soh_abs) { fs = soh_abs + 1; continue; }
                } else {
                    // Field starts in this chunk: use eq_pos array
                    while (ei < ec && (base + eq_pos[ei]) < fs) ei++;
                    if (ei >= ec) { fs = soh_abs + 1; continue; }
                    eq_abs = base + eq_pos[ei];
                    if (eq_abs >= soh_abs) { fs = soh_abs + 1; continue; }
                    ei++;
                }

                const char* tp = buf + fs;
                uint8_t tl = eq_abs - fs;
                uint16_t tk = (tl >= 2) ? (uint16_t)(tp[0] | (tp[1] << 8)) : (uint16_t)tp[0];
                const char* val = buf + eq_abs + 1;
                uint16_t vl = soh_abs - eq_abs - 1;

                switch (tk) {
                    case '3'|('5'<<8): msg.msg_type  = {val, vl}; break;
                    case '5'|('5'<<8): msg.symbol    = {val, vl}; break;
                    case '5'|('4'<<8): msg.side      = {val, vl}; break;
                    case '3'|('8'<<8): msg.order_qty = {val, vl}; break;
                    case '4'|('4'<<8): msg.price     = {val, vl}; break;
                    case '3'|('4'<<8): msg.seq_num   = {val, vl}; break;
                    case '4'|('9'<<8): msg.sender    = {val, vl}; break;
                    case '1'|('0'<<8):
                        bare_callback(msg);
                        local++;
                        msg = {};
                        break;
                }
                fs = soh_abs + 1;
            }
            base = wend;
        }
        g_sink = local;
        count = local;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    double mps = (double)count * iters / us * 1e6;
    double gbps = (double)total_len * iters / us * 1e6 / 1e9;
    std::cout << "Layer 2a (AVX-512 two-phase): " << std::fixed << std::setprecision(0)
              << mps << " msg/sec  (" << gbps << " GB/sec)" << std::endl;
    return count;
}

// ---------------------------------------------------------------------------
// Layer 2b: AVX-512 two-phase, schema-aware (no tag decode).
//          Since all messages are NewOrderSingle with fixed field order:
//            8=, 9=, 35=, 49=, 56=, 34=, 52=, 11=, 55=, 54=, 38=, 44=, 40=, 10=
//          We skip tag parsing entirely and extract values by field INDEX.
//          Field 3 (0-indexed) = MsgType, field 9 = Symbol, etc.
//          This eliminates the tag decode + switch: just direct extraction.
// ---------------------------------------------------------------------------
static int g_schema_count = 0;
static int g_schema_sink = 0;

// Field indices in NewOrderSingle (0-indexed, from SOH count)
// 0:8=  1:9=  2:35=  3:49=  4:56=  5:34=  6:52=  7:11=  8:55=  9:54=  10:38=  11:44=  12:40=  13:10=
static constexpr int IDX_MSGTYPE  = 2;
static constexpr int IDX_SENDER    = 3;
static constexpr int IDX_SEQ       = 5;
static constexpr int IDX_SYMBOL    = 8;
static constexpr int IDX_SIDE      = 9;
static constexpr int IDX_ORDERQTY = 10;
static constexpr int IDX_PRICE     = 11;
static constexpr int IDX_CHECKSUM  = 13;  // last field = end of message

static int bench_avx512_schema(const MsgBuf& mb, int iters) {
    const char* buf = mb.data.data();
    size_t total_len = mb.data.size();
    int count = 0;

    static constexpr size_t CHUNK = 65536;
    static constexpr int MAX_POS = 8192;
    uint16_t soh_pos[MAX_POS];

    // Warmup
    for (int w = 0; w < 3; w++) {
        int local = 0;
        size_t base = 0;
        while (base < total_len) {
            size_t wend = std::min(base + CHUNK, total_len);
            const __m512i soh_v = _mm512_set1_epi8(SOH);
            int sc = 0;
            size_t pos = base;
            while (pos + 64 <= wend) {
                __m512i ch = _mm512_loadu_si512(reinterpret_cast<const void*>(buf + pos));
                __mmask64 sm = _mm512_cmpeq_epi8_mask(ch, soh_v);
                while (sm && sc < MAX_POS) { soh_pos[sc++] = (uint16_t)(pos + __builtin_ctzll(sm) - base); sm &= sm-1; }
                pos += 64;
            }
            for (; pos < wend; pos++)
                if (buf[pos] == SOH && sc < MAX_POS) soh_pos[sc++] = (uint16_t)(pos - base);
            // Count messages: every 14th SOH is the checksum field
            local += sc / 14;
            base = wend;
        }
        g_sink = local;
    }

    g_schema_count = 0;
    g_schema_sink = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iters; iter++) {
        int local = 0;
        size_t base = 0;
        size_t fs = 0;       // field_start carried across chunks
        int fi = 0;          // field index carried across chunks
        BareMsg msg{};       // current message carried across chunks
        while (base < total_len) {
            size_t wend = std::min(base + CHUNK, total_len);

            // Phase 1: AVX-512 scan for SOH only
            const __m512i soh_v = _mm512_set1_epi8(SOH);
            int sc = 0;
            size_t pos = base;
            while (pos + 64 <= wend) {
                __m512i ch = _mm512_loadu_si512(reinterpret_cast<const void*>(buf + pos));
                __mmask64 sm = _mm512_cmpeq_epi8_mask(ch, soh_v);
                while (sm && sc < MAX_POS) {
                    soh_pos[sc++] = (uint16_t)(pos + __builtin_ctzll(sm) - base);
                    sm &= sm - 1;
                }
                pos += 64;
            }
            for (; pos < wend; pos++)
                if (buf[pos] == SOH && sc < MAX_POS) soh_pos[sc++] = (uint16_t)(pos - base);

            // Phase 2: schema-aware extraction by field index
            for (int si = 0; si < sc; si++) {
                size_t soh_abs = base + soh_pos[si];

                // Find '=' by scanning from fs (tag is short: 1-3 chars)
                size_t eq_abs = fs;
                while (eq_abs < soh_abs && buf[eq_abs] != '=') eq_abs++;

                if (eq_abs < soh_abs) {
                    const char* val = buf + eq_abs + 1;
                    uint16_t vl = soh_abs - eq_abs - 1;

                    switch (fi) {
                        case IDX_MSGTYPE:  msg.msg_type  = {val, vl}; break;
                        case IDX_SENDER:   msg.sender    = {val, vl}; break;
                        case IDX_SEQ:      msg.seq_num   = {val, vl}; break;
                        case IDX_SYMBOL:   msg.symbol    = {val, vl}; break;
                        case IDX_SIDE:     msg.side      = {val, vl}; break;
                        case IDX_ORDERQTY: msg.order_qty = {val, vl}; break;
                        case IDX_PRICE:    msg.price     = {val, vl}; break;
                        case IDX_CHECKSUM:
                            bare_callback(msg);
                            local++;
                            msg = {};
                            break;
                    }
                }

                fs = soh_abs + 1;
                fi++;
                if (fi >= 14) fi = 0;
            }
            base = wend;
        }
        g_schema_sink = local;
        count = local;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    double mps = (double)count * iters / us * 1e6;
    double gbps = (double)total_len * iters / us * 1e6 / 1e9;
    std::cout << "Layer 2b (AVX-512 schema):    " << std::fixed << std::setprecision(0)
              << mps << " msg/sec  (" << gbps << " GB/sec)" << std::endl;
    return count;
}

// ---------------------------------------------------------------------------
// Layer 2c: AVX-512 schema-aware + hardcoded tag lengths.
//          Same as 2b but eliminates the '=' scan entirely.
//          For NewOrderSingle, tag lengths are: 1,1,2,2,2,2,2,2,2,2,2,2,2,2
//          So '=' is at field_start + tag_len[fi]. No loop, no branch.
//          This is the absolute minimum work: AVX-512 SOH scan + direct offset.
// ---------------------------------------------------------------------------
static constexpr uint8_t tag_lens[14] = {1,1,2,2,2,2,2,2,2,2,2,2,2,2};

static int bench_avx512_hardcoded(const MsgBuf& mb, int iters) {
    const char* buf = mb.data.data();
    size_t total_len = mb.data.size();
    int count = 0;

    static constexpr size_t CHUNK = 65536;
    static constexpr int MAX_POS = 8192;
    uint16_t soh_pos[MAX_POS];

    // Warmup
    for (int w = 0; w < 3; w++) {
        int local = 0;
        size_t base = 0;
        while (base < total_len) {
            size_t wend = std::min(base + CHUNK, total_len);
            const __m512i soh_v = _mm512_set1_epi8(SOH);
            int sc = 0;
            size_t pos = base;
            while (pos + 64 <= wend) {
                __m512i ch = _mm512_loadu_si512(reinterpret_cast<const void*>(buf + pos));
                __mmask64 sm = _mm512_cmpeq_epi8_mask(ch, soh_v);
                while (sm && sc < MAX_POS) { soh_pos[sc++] = (uint16_t)(pos + __builtin_ctzll(sm) - base); sm &= sm-1; }
                pos += 64;
            }
            for (; pos < wend; pos++)
                if (buf[pos] == SOH && sc < MAX_POS) soh_pos[sc++] = (uint16_t)(pos - base);
            local += sc / 14;
            base = wend;
        }
        g_sink = local;
    }

    g_bare_count = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iters; iter++) {
        int local = 0;
        size_t base = 0;
        size_t fs = 0;
        int fi = 0;
        BareMsg msg{};

        while (base < total_len) {
            size_t wend = std::min(base + CHUNK, total_len);

            // Phase 1: AVX-512 SOH scan
            const __m512i soh_v = _mm512_set1_epi8(SOH);
            int sc = 0;
            size_t pos = base;
            while (pos + 64 <= wend) {
                __m512i ch = _mm512_loadu_si512(reinterpret_cast<const void*>(buf + pos));
                __mmask64 sm = _mm512_cmpeq_epi8_mask(ch, soh_v);
                while (sm && sc < MAX_POS) {
                    soh_pos[sc++] = (uint16_t)(pos + __builtin_ctzll(sm) - base);
                    sm &= sm - 1;
                }
                pos += 64;
            }
            for (; pos < wend; pos++)
                if (buf[pos] == SOH && sc < MAX_POS) soh_pos[sc++] = (uint16_t)(pos - base);

            // Phase 2: direct offset extraction (no EQ scan)
            for (int si = 0; si < sc; si++) {
                size_t soh_abs = base + soh_pos[si];

                // Direct: '=' is at fs + tag_lens[fi]
                size_t eq_abs = fs + tag_lens[fi];
                if (eq_abs < soh_abs && buf[eq_abs] == '=') {
                    const char* val = buf + eq_abs + 1;
                    uint16_t vl = soh_abs - eq_abs - 1;

                    switch (fi) {
                        case IDX_MSGTYPE:  msg.msg_type  = {val, vl}; break;
                        case IDX_SENDER:   msg.sender    = {val, vl}; break;
                        case IDX_SEQ:      msg.seq_num   = {val, vl}; break;
                        case IDX_SYMBOL:   msg.symbol    = {val, vl}; break;
                        case IDX_SIDE:     msg.side      = {val, vl}; break;
                        case IDX_ORDERQTY: msg.order_qty = {val, vl}; break;
                        case IDX_PRICE:    msg.price     = {val, vl}; break;
                        case IDX_CHECKSUM:
                            bare_callback(msg);
                            local++;
                            msg = {};
                            break;
                    }
                }

                fs = soh_abs + 1;
                fi++;
                if (fi >= 14) fi = 0;
            }
            base = wend;
        }
        g_sink = local;
        count = local;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    double mps = (double)count * iters / us * 1e6;
    double gbps = (double)total_len * iters / us * 1e6 / 1e9;
    std::cout << "Layer 2c (AVX-512 hardcoded): " << std::fixed << std::setprecision(0)
              << mps << " msg/sec  (" << gbps << " GB/sec)" << std::endl;
    return count;
}

// ---------------------------------------------------------------------------
// Layer 3: feed_batch() on contiguous buffer: real AVX2 parsing via
//          Session::feed_batch(). Minimal callback (just count).
// ---------------------------------------------------------------------------
static int g_batch_count = 0;
static void batch_callback(Message& msg) {
    g_batch_count++;
}

static int bench_feed_batch(const MsgBuf& mb, int iters) {
    Session session("CLIENT", "BROKER", batch_callback);

    const char* logon = "8=FIX.4.2\x01" "9=70\x01" "35=A\x01" "49=BROKER\x01"
                        "56=CLIENT\x01" "34=1\x01" "52=20240101-12:00:00\x01"
                        "98=0\x01" "108=30\x01" "10=123\x01";
    session.feed(logon, strlen(logon));

    session.feed_batch(mb.data.data(), mb.data.size());

    g_batch_count = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iters; iter++) {
        session.feed_batch(mb.data.data(), mb.data.size());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    double mps = (double)NUM_MESSAGES * iters / us * 1e6;
    double gbps3 = (double)mb.data.size() * iters / us * 1e6 / 1e9;
    std::cout << "Layer 3 (feed_batch, no cb): " << std::fixed << std::setprecision(0)
              << mps << " msg/sec  (" << gbps3 << " GB/sec)" << std::endl;
    return g_batch_count;
}

// ---------------------------------------------------------------------------
// Layer 4: feed_batch() with full callback (field extraction like bench_nsfix)
// ---------------------------------------------------------------------------
static int g_full_count = 0;
static void full_callback(Message& msg) {
    msg.get(Tag::MsgType);
    msg.get(Tag::SenderCompID);
    msg.get(Tag::TargetCompID);
    msg.get(Tag::Symbol);
    msg.get(Tag::Side);
    msg.get_int(Tag::OrderQty);
    msg.get_double(Tag::Price);
    g_full_count++;
}

static int bench_feed_batch_full(const MsgBuf& mb, int iters) {
    Session session("CLIENT", "BROKER", full_callback);

    const char* logon = "8=FIX.4.2\x01" "9=70\x01" "35=A\x01" "49=BROKER\x01"
                        "56=CLIENT\x01" "34=1\x01" "52=20240101-12:00:00\x01"
                        "98=0\x01" "108=30\x01" "10=123\x01";
    session.feed(logon, strlen(logon));

    session.feed_batch(mb.data.data(), mb.data.size());

    g_full_count = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iters; iter++) {
        session.feed_batch(mb.data.data(), mb.data.size());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    double mps = (double)NUM_MESSAGES * iters / us * 1e6;
    double gbps4 = (double)mb.data.size() * iters / us * 1e6 / 1e9;
    std::cout << "Layer 4 (feed_batch + full): " << std::fixed << std::setprecision(0)
              << mps << " msg/sec  (" << gbps4 << " GB/sec)" << std::endl;
    return g_full_count;
}

// ---------------------------------------------------------------------------
// Layer 5: feed() per-message on contiguous buffer (no vector<string>)
// ---------------------------------------------------------------------------
static int g_permsg_count = 0;
static void permsg_callback(Message& msg) {
    msg.get(Tag::MsgType);
    msg.get(Tag::SenderCompID);
    msg.get(Tag::TargetCompID);
    msg.get(Tag::Symbol);
    msg.get(Tag::Side);
    msg.get_int(Tag::OrderQty);
    msg.get_double(Tag::Price);
    g_permsg_count++;
}

static int bench_feed_permsg(const MsgBuf& mb, int iters) {
    Session session("CLIENT", "BROKER", permsg_callback);

    const char* logon = "8=FIX.4.2\x01" "9=70\x01" "35=A\x01" "49=BROKER\x01"
                        "56=CLIENT\x01" "34=1\x01" "52=20240101-12:00:00\x01"
                        "98=0\x01" "108=30\x01" "10=123\x01";
    session.feed(logon, strlen(logon));

    for (int i = 0; i < NUM_MESSAGES; i++)
        session.feed(mb.data.data() + mb.offsets[i], mb.lengths[i]);

    g_permsg_count = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iters; iter++) {
        for (int i = 0; i < NUM_MESSAGES; i++)
            session.feed(mb.data.data() + mb.offsets[i], mb.lengths[i]);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    double mps = (double)NUM_MESSAGES * iters / us * 1e6;
    double gbps5 = (double)mb.data.size() * iters / us * 1e6 / 1e9;
    std::cout << "Layer 5 (feed per-msg, contig):" << std::fixed << std::setprecision(0)
              << mps << " msg/sec  (" << gbps5 << " GB/sec)" << std::endl;
    return g_permsg_count;
}

// ---------------------------------------------------------------------------
// Layer 6: feed() per-message on vector<string> (the existing bench pattern)
// ---------------------------------------------------------------------------
static int g_vec_count = 0;
static void vec_callback(Message& msg) {
    msg.get(Tag::MsgType);
    msg.get(Tag::SenderCompID);
    msg.get(Tag::TargetCompID);
    msg.get(Tag::Symbol);
    msg.get(Tag::Side);
    msg.get_int(Tag::OrderQty);
    msg.get_double(Tag::Price);
    g_vec_count++;
}

static int bench_feed_vector(const std::vector<std::string>& msgs, int iters) {
    Session session("CLIENT", "BROKER", vec_callback);

    const char* logon = "8=FIX.4.2\x01" "9=70\x01" "35=A\x01" "49=BROKER\x01"
                        "56=CLIENT\x01" "34=1\x01" "52=20240101-12:00:00\x01"
                        "98=0\x01" "108=30\x01" "10=123\x01";
    session.feed(logon, strlen(logon));

    for (const auto& m : msgs)
        session.feed(m.data(), m.length());

    g_vec_count = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iters; iter++) {
        for (const auto& m : msgs)
            session.feed(m.data(), m.length());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    double mps = (double)NUM_MESSAGES * iters / us * 1e6;
    double total_bytes6 = 0;
    for (const auto& m : msgs) total_bytes6 += m.length();
    double gbps6 = total_bytes6 * iters / us * 1e6 / 1e9;
    std::cout << "Layer 6 (feed per-msg, vec):  " << std::fixed << std::setprecision(0)
              << mps << " msg/sec  (" << gbps6 << " GB/sec)" << std::endl;
    return g_vec_count;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    std::cout << "=== NSFix Bare-Bones Throughput Ladder ===" << std::endl;
    std::cout << "Each layer adds overhead. Layer 0 = Shannon limit." << std::endl;
    std::cout << std::endl;

    std::cout << "Building contiguous buffer..." << std::endl;
    MsgBuf mb = build_contiguous(NUM_MESSAGES);
    std::cout << "Buffer: " << mb.data.size() << " bytes, " << NUM_MESSAGES << " messages" << std::endl;
    std::cout << "Avg msg size: " << mb.data.size() / NUM_MESSAGES << " bytes" << std::endl;
    std::cout << std::endl;

    std::vector<std::string> msgs;
    msgs.reserve(NUM_MESSAGES);
    for (int i = 0; i < NUM_MESSAGES; i++)
        msgs.emplace_back(mb.data.data() + mb.offsets[i], mb.lengths[i]);

    const int timed = 10;
    std::cout << "Timed: " << timed << " iterations" << std::endl;
    std::cout << std::endl;

    std::cout << "--- Throughput Ladder ---" << std::endl;
    bench_raw_scan(mb, timed);
    bench_raw_scan_avx512(mb, timed);
    bench_msg_count_avx(mb, timed);
    bench_bare_parse(mb, timed);
    bench_avx512_twophase(mb, timed);
    bench_avx512_schema(mb, timed);
    bench_avx512_hardcoded(mb, timed);
    bench_feed_batch(mb, timed);
    bench_feed_batch_full(mb, timed);
    bench_feed_permsg(mb, timed);
    bench_feed_vector(msgs, timed);

    // Memory bandwidth reference
    std::cout << std::endl;
    double total_bytes = (double)mb.data.size() * timed;
    {
        std::vector<char> dst(mb.data.size());
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < timed; i++)
            memcpy(dst.data(), mb.data.data(), mb.data.size());
        auto t1 = std::chrono::high_resolution_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        double gbps = total_bytes / us * 1e6 / 1e9;
        double max_mps = gbps * 1e9 / (mb.data.size() / NUM_MESSAGES);
        std::cout << "--- Memory Bandwidth Reference ---" << std::endl;
        std::cout << "memcpy bandwidth: " << std::fixed << std::setprecision(1)
                  << gbps << " GB/sec" << std::endl;
        std::cout << "Theoretical max:  " << std::fixed << std::setprecision(0)
                  << max_mps << " msg/sec" << std::endl;
    }

    return 0;
}

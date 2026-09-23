// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

// NSNet benchmark: constexpr-schema batch extraction vs naive
// struct-cast + ntohl/ntohs per-packet parsing.
//
// Synthetic 64-byte Ethernet/IPv4/TCP frames, fixed stride.
// King: naive per-packet parse (no DPDK dependency).
// Metric: million packets/sec at batch sizes 64 / 256 / 1024 / 4096.

#include "nsnet.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <random>

// Fill one frame: Ethernet + IPv4(IHL=5) + TCP + payload pad to pkt_size.
static void gen_packet(uint8_t* p, size_t pkt_size,
                       uint32_t src_ip, uint32_t dst_ip,
                       uint16_t sport, uint16_t dport, uint16_t len) {
    memset(p, 0, pkt_size);
    // Ethernet
    for (int k = 0; k < 6; ++k) { p[k] = 0xAA; p[6 + k] = 0xBB; }
    p[NSEthernetSchema::ETHERTYPE]     = 0x08;
    p[NSEthernetSchema::ETHERTYPE + 1] = 0x00;
    // IPv4
    p[NSIPv4Schema::VER_IHL] = 0x45;
    p[NSIPv4Schema::TOTAL_LEN]     = (uint8_t)(len >> 8);
    p[NSIPv4Schema::TOTAL_LEN + 1] = (uint8_t)(len & 0xFF);
    p[NSIPv4Schema::TTL]      = 64;
    p[NSIPv4Schema::PROTOCOL] = 6;  // TCP
    uint32_t s = __builtin_bswap32(src_ip), d = __builtin_bswap32(dst_ip);
    memcpy(p + NSIPv4Schema::SRC_IP, &s, 4);
    memcpy(p + NSIPv4Schema::DST_IP, &d, 4);
    // TCP
    uint16_t sp = __builtin_bswap16(sport), dp = __builtin_bswap16(dport);
    memcpy(p + NSTCPSchema::SRC_PORT, &sp, 2);
    memcpy(p + NSTCPSchema::DST_PORT, &dp, 2);
    p[NSTCPSchema::FLAGS] = 0x50;  // data offset 5
}

// Naive per-packet parser: struct cast + ntoh*, the runtime-parse king.
struct NaiveHdr {
    uint32_t src_ip, dst_ip;
    uint16_t src_port, dst_port;
    uint8_t  protocol;
    uint16_t length;
};

static void naive_parse(const uint8_t* pkts, size_t stride, size_t count,
                        NaiveHdr* out) {
    for (size_t i = 0; i < count; ++i) {
        const uint8_t* p = pkts + i * stride;
        uint32_t s, d, pr; uint16_t l;
        memcpy(&s,  p + NSIPv4Schema::SRC_IP,    4);
        memcpy(&d,  p + NSIPv4Schema::DST_IP,    4);
        memcpy(&pr, p + NSTCPSchema::PORTS,      4);
        memcpy(&l,  p + NSIPv4Schema::TOTAL_LEN, 2);
        out[i].src_ip   = __builtin_bswap32(s);
        out[i].dst_ip   = __builtin_bswap32(d);
        out[i].src_port = __builtin_bswap16((uint16_t)(pr & 0xFFFF));
        out[i].dst_port = __builtin_bswap16((uint16_t)(pr >> 16));
        out[i].protocol = p[NSIPv4Schema::PROTOCOL];
        out[i].length   = __builtin_bswap16(l);
    }
}

int main() {
    printf("NSNet benchmark — constexpr schema + AVX-512 batch extraction\n\n");

    const size_t pkt_sizes[] = {64, 1500};
    const size_t sizes[] = {64, 256, 1024, 4096};
    std::mt19937 rng(42);

    for (size_t pi = 0; pi < 2; ++pi) {
        size_t pkt_size = pkt_sizes[pi];
        size_t stride = (pkt_size + 63) & ~(size_t)63;  // 64B-aligned stride
        printf("--- packet size %zuB (stride %zu) ---\n", pkt_size, stride);

        for (size_t si = 0; si < 4; ++si) {
        size_t count = sizes[si];
        uint8_t* pkts = (uint8_t*)aligned_alloc(64, count * stride);
        for (size_t i = 0; i < count; ++i)
            gen_packet(pkts + i * stride, pkt_size,
                       0x0A000000 | (uint32_t)(rng() & 0xFFFFFF),
                       0xC0A80000 | (uint32_t)(rng() & 0xFFFF),
                       (uint16_t)(1024 + (rng() % 60000)),
                       (uint16_t)(1 + (rng() % 65535)),
                       (uint16_t)(40 + (rng() % 1460)));

        // Schema gate
        NSNetSchemaType st = ns_net_detect(pkts);
        const char* stname = st == NSNET_ETH_IP_TCP ? "ETH_IP_TCP" : "UNKNOWN";

        NSNetFields fields;
        fields.allocate(count);
        NaiveHdr* ref = (NaiveHdr*)malloc(count * sizeof(NaiveHdr));

        // Correctness: NSNet vs naive, exact match on every field.
        naive_parse(pkts, stride, count, ref);
        ns_net_extract_batch(pkts, stride, count, fields);
        size_t bad = 0;
        for (size_t i = 0; i < count; ++i) {
            if (fields.src_ip[i]   != ref[i].src_ip   ||
                fields.dst_ip[i]   != ref[i].dst_ip   ||
                fields.src_port[i] != ref[i].src_port ||
                fields.dst_port[i] != ref[i].dst_port ||
                fields.protocol[i] != ref[i].protocol ||
                fields.length[i]   != ref[i].length)
                ++bad;
        }

        // Timing: enough iterations for >= ~50M packets per measurement.
        size_t target = 50000000;
        size_t iters  = target / count;
        if (iters < 1) iters = 1;

        auto t0 = std::chrono::high_resolution_clock::now();
        for (size_t it = 0; it < iters; ++it)
            naive_parse(pkts, stride, count, ref);
        auto t1 = std::chrono::high_resolution_clock::now();
        double naive_s = std::chrono::duration<double>(t1 - t0).count();

        t0 = std::chrono::high_resolution_clock::now();
        for (size_t it = 0; it < iters; ++it)
            ns_net_extract_batch(pkts, stride, count, fields);
        t1 = std::chrono::high_resolution_clock::now();
        double ns_s = std::chrono::duration<double>(t1 - t0).count();

        double naive_mpps = (double)count * iters / naive_s / 1e6;
        double ns_mpps    = (double)count * iters / ns_s / 1e6;

        printf("batch=%5zu  schema=%s  naive=%8.1f Mpps  nsnet=%8.1f Mpps"
               "  speedup=%6.2fx  mismatches=%zu\n",
               count, stname, naive_mpps, ns_mpps,
               naive_mpps / ns_mpps > 0 ? ns_mpps / naive_mpps : 0.0, bad);

        free(pkts);
        free(ref);
        }
        printf("\n");
    }

    printf("UNKNOWN path (IPv6 / ext headers / TCP options): no fast path,\n");
    printf("falls back to per-packet scalar — documented loss.\n");
    return 0;
}

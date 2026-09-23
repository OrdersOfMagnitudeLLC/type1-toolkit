# NSNet

Compile-time packet schema extraction for Ethernet/IPv4/TCP/UDP.

## Idea

Packet headers are fixed-format schemas: every field sits at a known byte
offset. DPDK and friends treat packets as opaque bytes and parse at runtime.
NSNet encodes the schema as `constexpr` offsets and extracts fields from
16 packets at once with AVX-512 byte permutes — zero runtime field lookup.

## Architecture

- `NSEthernetSchema`, `NSIPv4Schema`, `NSTCPSchema`, `NSUDPSchema` —
  every field a `constexpr size_t` offset.
- `ns_net_detect()` — schema gate from fixed offsets (ethertype 0x0800,
  IHL=5, protocol 6/17). Returns `ETH_IP_TCP`, `ETH_IP_UDP`, or `UNKNOWN`.
- `ns_net_extract_batch()` — one 64B load per packet, then a single
  `permutexvar_epi8` (AVX512-VBMI) compacts all six fields into a 16B
  record with host-order byte swap baked into the index vector. A
  two-level `permutex2var` transpose writes 16 packets to SoA per
  iteration. No gathers — pure load + shuffle. Scalar tail/fallback.

## Build & Run

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --target nsnet -j$(nproc)
./build/nsnet
```

Or build manually with `g++`:

```bash
g++ -O3 -march=native -o bench_nsnet bench_nsnet.cpp
./bench_nsnet
```

Requires AVX-512 F + VBMI (`__AVX512VBMI__`); falls back to scalar
extraction without it.

## Benchmark

Synthetic Ethernet/IPv4/TCP frames at 64B and 1500B (MTU), fixed
64B-aligned stride. King: naive struct-cast + `ntohl`/`ntohs` per packet.
Metric: million packets/sec at batch sizes 64 / 256 / 1024 / 4096.
Correctness: extracted fields verified bit-exact against the naive parser.

Results on Spectre (i7-1165G7, AVX-512 F+VBMI):

64B packets (stride 64):

| Batch | Naive (Mpps) | NSNet (Mpps) | Speedup | Mismatches |
|------:|-------------:|-------------:|--------:|-----------:|
|    64 |        791.7 |        883.9 |   1.12x |          0 |
|   256 |        767.3 |        850.2 |   1.11x |          0 |
|  1024 |        443.2 |        818.7 |   1.85x |          0 |
|  4096 |        638.4 |        652.6 |   1.02x |          0 |

1500B packets (stride 1536):

| Batch | Naive (Mpps) | NSNet (Mpps) | Speedup | Mismatches |
|------:|-------------:|-------------:|--------:|-----------:|
|    64 |        777.1 |        876.8 |   1.13x |          0 |
|   256 |        534.0 |        800.3 |   1.50x |          0 |
|  1024 |        530.3 |        795.3 |   1.50x |          0 |
|  4096 |        306.3 |        410.1 |   1.34x |          0 |

The win is bounded by memory, not compute: extraction touches only the
first 64B of each packet, so at 1500B stride both sides are cache-miss
dominated and NSNet's advantage (~1.4-1.5x) comes from doing one aligned
load + one byte permute per packet instead of five scalar loads + bswaps.
At 64B the working set fits L2 and the naive parser is already cheap, so
the margin narrows.

## Win / Loss

- **Win:** known protocol batch — Ethernet + IPv4 (no options) + TCP/UDP
  at fixed stride. All offsets compile-time, extraction fully vectorized.
- **Loss (documented):** `UNKNOWN` schema — IPv6, IPv6 extension headers,
  IPv4 options (IHL>5), TCP options shifting payload offsets, variable
  stride. No fast path; per-packet scalar fallback only.

Same pattern as the rest of the suite: declare the structure, unlock the win.

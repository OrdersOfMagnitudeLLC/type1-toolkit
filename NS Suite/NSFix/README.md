# NSFix

**Full production FIX engine: faster than bare codecs.**

FIX protocol parser with SIMD-accelerated tag extraction, callback-based message dispatch,
memory-mapped persistence, NSHash multi-session management, and session recovery/replay.

## Build

```bash
cmake -B build && cmake --build build           # core parser + bench
cmake -B build_full && cmake --build build_full  # full feature bench
```

## Run

```bash
./build/nsfix              # core parse benchmark
./build_full/nsfix_full    # full feature benchmark (persistence + sessions + recovery)
```

## Parameters

| Flag | Description |
|------|-------------|
| `NS_FIX_ALL` | CMake option to build all bench variants (batch, diagnostic, parse_only, quickfix, scalar, server, test) |

## Feature Comparison

| Parser | Persistence | Multi-session | Recovery |
|--------|:-----------:|:-------------:|:--------:|
| NSFix | ✅ | ✅ | ✅ |
| hffix | ❌ | ❌ | ❌ |
| QuickFIX | ✅ | ✅ | ✅ |

## Benchmark Results

| Parser | Throughput (msg/sec) |
|--------|---------------------|
| NSFix (full: mmap log + multi-session + recovery) | ~8.5M |
| hffix (equivalent workload, fair) | 8.1M |
| cpp_fix_codec (parse) | 3.1M |
| QuickFIX 1.15.1 (full engine) | 452K |

*Benchmarked on isolated runs: system load affects all FIX parser throughputs.*

## Bare Metal Benchmark

NSFix's production parser (full session: mmap log, multi-session,
recovery) runs at ~9.5M msg/sec: matching or beating bare codecs
that do none of this.

The bare scanner (AVX-512, no field extraction) hits 1.6B SOH/sec
 14 GB/sec, 73% of this machine's 19.1 GB/sec memory bandwidth
ceiling. Theoretical maximum: 154M msg/sec.

The production gap to theoretical is the irreducible cost of
semantic extraction. Every field parsed is a byte touched.

Run: `./build/nsfix_bare`

## Architecture

- **Persistence** (`nsfix_log.hpp`): 256MB file-backed circular buffer with 2MB L2-resident
  in-memory ring buffer for zero-syscall hot-path logging. 32-message pointer-based staging
  buffer batches writes to minimize cache traffic. `pwrite()` + `fdatasync()` on clean shutdown.
- **Multi-session** (`nsfix_session_manager.hpp`): NSHash-based O(1) session lookup mapping
  64-bit composite session keys to `FIXSessionState` (seq_in, seq_out, connection state,
  last heartbeat, Session pointer).
- **Recovery** (`recover_session()` in `nsfix_log.hpp`): Scans ring buffer log for messages
  matching SenderCompID/TargetCompID with sequence numbers > gap_start, returns messages
  for replay.

## Third-party

- **QuickFIX**: BSD License (tested against stable 1.15.1; latest HEAD has memory safety issues under native flags)
- **nlohmann/json**: MIT License (vendored header, used by server only)
- **hffix**: BSL-1.0 License (benchmark comparison only)
- **NSHash**: proprietary (used for multi-session management)

# NSLock

Custom read-write lock with reader-biased design for high-concurrency read-heavy workloads.

## Build

```bash
cmake -B build && cmake --build build
```

## Run

```bash
./build/nslock
```

## Parameters

Optional: `./build/nslock <threads> [reader_ratio]`. Default runs both scenarios below, then the stress test.

## Results

Baseline: std::shared_mutex. 100M ops, pinned to cores 0-7.

**Pure read: 256 threads, 100% readers:**

| Lock | Throughput (ops/sec) | Speedup |
|------|---------------------|---------|
| std::shared_mutex | 11,120,000 | 1.0x |
| NSRWLock | 25,420,000 | **2.29x** |

**Mixed: 8 threads, 95% read / 5% write:**

| Lock | Throughput (ops/sec) | Speedup |
|------|---------------------|---------|
| std::shared_mutex | 5,000,000 | 1.0x |
| NSRWLock | 7,170,000 | **1.43x** |

Headline: **2.29x vs std::shared_mutex** under pure-read load - per-thread reader slots mean zero reader-reader cache contention, while shared_mutex pays a shared atomic update on every acquire/release.

## Third-party

None.

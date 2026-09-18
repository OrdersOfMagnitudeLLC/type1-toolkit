# NSHash

Open-addressing hash map with linear probing, optimized for sequential and bounded-key access patterns.

## Build

```bash
cmake -B build && cmake --build build
```

## Run

```bash
./build/nshash --benchmark_min_time=1x
```

## Parameters

None.

## Results

Competitors (Abseil, ankerl::unordered_dense, robin_hood, fph::unordered_map, boost::unordered_flat_map, std::unordered_map) all built with `-O3 -march=native`. Cold-run medians, 3 runs, 1M keys.

Lookup (ns/op):

| Workload | NSHash | absl | ankerl | robinhood | std | fph | boost | vs absl |
|----------|--------|------|--------|-----------|-----|-----|-------|---------|
| Sequential | 8.52 | 19.38 | 24.09 | 21.88 | 27.44 | 14.23 | 20.52 | 2.28x |
| Random | 12.96 | 18.75 | 17.29 | 22.08 | 33.28 | 14.36 | 19.78 | 1.45x |
| Bounded | 1.19 | 10.14 | 11.01 | 8.29 | 2.59 | 2.96 | 14.39 | 8.53x |
| Timestamp | 1.62 | 19.09 | 14.41 | 22.20 | 3.53 | 15.19 | 21.74 | 11.76x |

Sequential insert (1M keys):

| Map | Time (ms) | ops/sec | vs NSHash |
|-----|-----------|---------|-----------|
| NSHash | 7.04 | 142.0M | - |
| boost | 16.1 | 62.2M | 2.28x |
| ankerl | 21.1 | 47.3M | 3.00x |
| std | 22.1 | 45.3M | 3.1x |
| robin_hood | 24.9 | 40.2M | 3.54x |
| absl | 30.0 | 33.3M | 4.26x |
| fph | 712.9 | 1.4M | 101x |

Headline: **11.76x vs absl::flat_hash_map** on timestamp lookup. Sequential insert: **4.26x absl / 101x fph / 3.1x std**.

## Third-party

- **Abseil**: Apache 2.0 License (built from source with `-O3 -march=native`)
- **ankerl::unordered_dense**: MIT License (built from source)
- **robin_hood::unordered_map**: MIT License (built from source)
- **fph::unordered_map**: MIT License (built from source)
- **Boost unordered_flat_map**: Boost Software License 1.0 (system library, v1.83.0)
- **Google Benchmark**: Apache 2.0 License (built from source)
- **nlohmann/json**: MIT License (vendored header, used by server only)

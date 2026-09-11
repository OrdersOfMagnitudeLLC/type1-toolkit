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

Competitors (Abseil, ankerl::unordered_dense, robin_hood, fph::unordered_map, boost::unordered_flat_map, std::unordered_map) all built with `-O3 -march=native`. Ratios measured under concurrent system load — bounded/timestamp wins are stable, sequential/random compress under load.

| Workload | NSHash (ns) | absl (ns) | ankerl (ns) | robinhood (ns) | std (ns) | fph (ns) | boost (ns) | vs absl |
|----------|------------|-----------|-------------|----------------|----------|----------|-----------|---------|
| Sequential | 15,314 | 49,449 | 40,589 | 26,074 | 25,764 | 20,684 | 26,205 | 3.2x |
| Random | 19,086 | 46,413 | 28,381 | 25,662 | 53,623 | 24,577 | 23,858 | 2.4x |
| Bounded | 2,602 | 47,403 | 18,237 | 13,661 | 3,456 | 7,110 | 18,650 | 18.2x |
| Timestamp | 2,737 | 42,611 | 22,450 | 23,285 | 4,180 | 18,321 | 28,456 | 15.6x |

## Third-party

- **Abseil** — Apache 2.0 License (built from source with `-O3 -march=native`)
- **ankerl::unordered_dense** — MIT License (built from source)
- **robin_hood::unordered_map** — MIT License (built from source)
- **fph::unordered_map** — MIT License (built from source)
- **Boost unordered_flat_map** — Boost Software License 1.0 (system library, v1.83.0)
- **Google Benchmark** — Apache 2.0 License (built from source)
- **nlohmann/json** — MIT License (vendored header, used by server only)

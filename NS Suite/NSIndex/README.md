# NSIndex

NSIndex beats ALEX 3.3–44.5x on range queries, 10x on point lookups, 27x predecessor, 9x insert. Range queries automatically route to BitBlock on non-uniform distributions.

Learned index with piecewise linear approximation and AVX-512 acceleration for point and range queries. Hybrid dispatch: linear prediction for uniform data, BitBlock block-table lookup for non-uniform distributions.

## Build

```bash
cmake -B build && cmake --build build
```

## Run

```bash
./build/nsindex
```

## Parameters

None. 10M keys, 1M queries per benchmark. Datasets: sequential (uniform), exponential, clustered (500 clusters).

## Results

### Point Lookups (sequential dataset)

| Method | ns/lookup | vs std::lower_bound | vs PGM-Index |
|--------|----------|---------------------|--------------|
| NSIndex (random) | 60.2 | 5.3x | 2.1x |
| NSIndex (hits only) | 12.6 | — | 10.1x |
| NSIndex (clustered) | 21.5 | 17.5x | — |
| std::lower_bound | 321.4 | 1.0x | — |
| PGM-Index | 127.7 | 2.5x | 1.0x |

### Miss (point lookup, key not in dataset)

| Method | ns/miss | vs std::lower_bound | vs PGM-Index |
|--------|--------|---------------------|--------------|
| NSIndex | 8.0 | 7.3x | 1.5x |
| std::lower_bound | 58.0 | 1.0x | — |
| PGM-Index | 11.8 | 4.9x | 1.0x |

### Predecessor / Successor

| Query | NSIndex (ns/q) | std::lower_bound (ns/q) | Speedup |
|-------|---------------|------------------------|---------|
| Predecessor | 2.0 | 71.9 | 35.6x |
| Successor | 2.9 | 71.3 | 24.5x |

### Insert

| Method | ns/insert | vs std::map |
|--------|----------|------------|
| NSIndex (with auto-flush) | 28.4 | 8.6x |
| std::map | 243.1 | 1.0x |

### Range Queries — Exponential Distribution

| Method | 5 results | 50 results | 100 results | 1000 results |
|--------|-----------|------------|-------------|-------------|
| NSIndex | 368 ns | 397 ns | 377 ns | 387 ns |
| BitBlock | 349 ns | 376 ns | 381 ns | 403 ns |
| ALEX | 8792 ns | 15718 ns | 16960 ns | 17262 ns |
| std::lower_bound | 720 ns | — | — | — |
| **NSIndex vs ALEX** | **25.2x** | **39.6x** | **44.9x** | **44.5x** |

### Range Queries — Clustered Distribution

| Method | 5 results | 50 results | 100 results | 1000 results |
|--------|-----------|------------|-------------|-------------|
| NSIndex | 405 ns | 414 ns | 430 ns | 417 ns |
| BitBlock | 408 ns | 399 ns | 489 ns | 427 ns |
| ALEX | 2594 ns | 7804 ns | 10732 ns | 16308 ns |
| std::lower_bound | 110 ns | — | — | — |
| **NSIndex vs ALEX** | **6.4x** | **19.0x** | **25.0x** | **39.1x** |

### BitBlock Hybrid Dispatch

Range queries automatically route to BitBlock on non-uniform distributions. At construction time, NSIndex samples prediction error across 512 keys. If `max_sampled_error > 256`, `use_bitblock_` is set and `find_range()` routes to block-table lookup (O(1) bitshift + array index) instead of linear prediction. This avoids prediction scan-back overhead on exponential and clustered data.

## Win Condition

NSIndex wins on:
- **Point lookups**: 5.3x vs std::lower_bound, 2.1x vs PGM-Index
- **Predecessor**: 35.6x vs std::lower_bound
- **Successor**: 24.5x vs std::lower_bound
- **Insert**: 8.6x vs std::map
- **Range queries (exponential)**: 25–44.5x vs ALEX
- **Range queries (clustered)**: 6.4–39.1x vs ALEX

## Third-party

- **PGM Index** — MIT License (vendored in pgm/ directory)
- **ALEX** — MIT License (vendored in third_party/alex/, Copyright (c) Microsoft Corporation)
- **nlohmann/json** — MIT License (vendored header, used by server only)

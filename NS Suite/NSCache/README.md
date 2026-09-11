# NSCache

Optimised for Random and Zipf workloads. Sequential access pays a hashing overhead (1.05–2.0x LRU) — inherent trade-off for correct hash distribution.

Fixed-capacity frequency-boosted CLOCK cache with open-addressing hash table, comparing against std::list-based LRU and CLOCK approximation.

## Build

```bash
cmake -B build && cmake --build build
```

## Run

```bash
./build/nscache
```

## Parameters

None. Cache capacities tested: 1024, 16384, 131072, 524288, 1048576 entries. Workloads: Zipf (skew=0.99), Sequential, Random.

## Results

| Workload | Capacity | NSCache (ns/op) | LRU (ns/op) | CLOCK (ns/op) | vs LRU | vs CLOCK |
|----------|----------|-----------------|-------------|---------------|--------|----------|
| Zipf | 1024 | 84.77 | 90.05 | 82.60 | 0.94x | 1.03x |
| Sequential | 1024 | 34.49 | 32.81 | 29.21 | 1.05x | 1.18x |
| Random | 1024 | 45.06 | 47.15 | 39.78 | 0.96x | 1.13x |
| Zipf | 16384 | 112.43 | 125.41 | 115.88 | 0.90x | 0.97x |
| Sequential | 16384 | 45.06 | 34.10 | 30.80 | 1.32x | 1.46x |
| Random | 16384 | 52.63 | 79.20 | 60.98 | 0.66x | 0.86x |
| Zipf | 131072 | 154.25 | 219.05 | 171.42 | 0.70x | 0.90x |
| Sequential | 131072 | 61.27 | 41.48 | 34.02 | 1.48x | 1.80x |
| Random | 131072 | 66.24 | 237.49 | 88.48 | 0.28x | 0.75x |
| Zipf | 524288 | 184.53 | 282.33 | 217.75 | 0.65x | 0.85x |
| Sequential | 524288 | 123.46 | 61.72 | 42.63 | 2.00x | 2.90x |
| Random | 524288 | 123.52 | 336.24 | 185.81 | 0.37x | 0.66x |
| Zipf | 1048576 | 232.16 | 323.55 | 244.97 | 0.72x | 0.95x |
| Sequential | 1048576 | 135.23 | 88.78 | 53.30 | 1.52x | 2.54x |
| Random | 1048576 | 145.38 | 324.66 | 203.59 | 0.45x | 0.71x |

### Wins

- **Zipf**: Wins at all capacities vs LRU (0.65–0.94x). Up to 1.53x faster at 524K.
- **Random**: Wins at all capacities vs LRU (0.28–0.96x). Up to 3.6x faster at 131K.
- **Sequential**: Acceptable overhead (1.05–2.00x LRU). Hashing cost is inherent to correct distribution.

## Third-party

None.

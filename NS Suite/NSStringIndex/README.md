# NSStringIndex

Field-indexed string search engine for structured log data, using AVX2 + OpenMP for O(1) amortized query after one-time build.

## Build

```bash
cmake -B build && cmake --build build
```

## Run

```bash
./build/nsstringindex
```

## Parameters

None. 5M Apache log lines, ~424MB corpus.

## Results

| Scenario | Baseline strstr (ms) | NSStringIndex (ms) | Speedup |
|----------|---------------------|-------------------|---------|
| Single (cold) | 125.8 | 72.6 | 0.6x (build cost dominates) |
| 100 queries | 13,447 | 61.9 | 217.3x |
| 1,000 queries | 136,613 | 63.4 | 2,153.5x |

## Third-party

None.

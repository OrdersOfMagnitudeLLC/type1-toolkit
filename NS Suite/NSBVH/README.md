# NSBVH

1.37x–1.44x Embree 4.4.1 on clustered/mixed scenes. Dense uniform is a documented loss (1.59x slower).

Bounding Volume Hierarchy (BVH) ray tracer with 8-wide SIMD traversal and spatial grid partitioning.

## Build

```bash
cmake -B build && cmake --build build
```

## Run

```bash
./build/nsbvh
```

## Parameters

None.

## Results

Baseline: Intel Embree 4.4.1, built from source with `-O3 -march=native`. Confirmed 2026-09-13.

| Scene | Ray Type | NS rays/sec | Embree rays/sec | NS (ms) | Embree (ms) | Ratio |
|-------|----------|-------------|-----------------|---------|-------------|-------|
| Dense Uniform (10K random, 500/500 cells) | random | 14.4M | 22.8M | 6.96 | 4.38 | 1.59x slower |
| Sparse Clustered (20 clusters, 20/500 cells) | 70% targeted + 30% random | 14.0M | 10.2M | 7.16 | 9.82 | **1.37x faster** |
| Mixed (10 clusters + 5K random, 500/500 cells) | 70% targeted + 30% random | 11.0M | 7.6M | 9.12 | 13.11 | **1.44x faster** |

**Win condition:** Clustered geometry: sparse clustered and mixed scenes with targeted rays. NS-BVH's spatial grid skips empty cells (96% of cells in the sparse scene), delivering 1.37–1.44x over Embree 4.4.1. No win on structureless random distribution: dense uniform is a documented loss (1.59x slower) - Embree's SAH-optimized BVH2 wins on build quality.

## Secondary Ray Benchmark (10M rays, dense scene)

Secondary rays with origins at object centers, random directions. Sorted by joint origin+direction Morton key (21-bit: 9-bit spatial cell + 12-bit direction). Sort uses Highway VQSort on packed (key, index) uint64 pairs.

| Metric | Unsorted | Sorted |
|--------|----------|--------|
| Traversal (ms) | 2237 | 1283 |
| Throughput (Mrays/s) | 4.47 | 7.79 |
| Total pipeline (ms) | 2237 | 1954 |

- **Sort time:** 671 ms (key computation 137 ms + conversion 92 ms + VQSort 379 ms + back-conversion 63 ms)
- **Traversal speedup:** 1.74x
- **Total pipeline speedup:** 1.14x
- **Hit count match:** YES (10,000,000 / 10,000,000)

## Third-party

- **Embree 4**: Apache 2.0 License (built from source v4.4.1 with `-O3 -march=native`)
- **Highway VQSort**: Apache 2.0 License (v1.2.0, fetched via CMake FetchContent)

# NSBVH

Wins on sparse and mixed scenes. Dense uniform within 1.18x of Embree 4.4.1.

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

Embree 4.4.1 built from source with `-O3 -march=native`.

| Scene | Ray Type | NS rays/sec | Embree rays/sec | NS (ms) | Embree (ms) | Ratio |
|-------|----------|-------------|-----------------|---------|-------------|-------|
| Dense Uniform (10K random, 80/80 cells) | random | 13.6M | 16.0M | 7.37 | 6.26 | 1.18x slower |
| Sparse Clustered (20 clusters, 20/80 cells) | 70% targeted + 30% random | 9.4M | 6.0M | 10.69 | 16.53 | **1.55x faster** |
| Mixed (10 clusters + 5K random, 80/80 cells) | 70% targeted + 30% random | 7.5M | 4.6M | 13.33 | 21.62 | **1.62x faster** |

**Win condition:** Sparse clustered and mixed scenes with targeted rays. NS-BVH's spatial grid skips empty cells (96% of cells in the sparse scene), delivering 1.55–1.62x speedup over Embree 4.4.1. Dense uniform narrowed to 1.18x — Embree's SAH-optimized BVH2 still wins on build quality for uniform distributions.

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

- **Embree 4** — Apache 2.0 License (built from source v4.4.1 with `-O3 -march=native`)
- **Highway VQSort** — Apache 2.0 License (v1.2.0, fetched via CMake FetchContent)

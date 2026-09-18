# NSAlloc

Custom memory allocator with multi-pool and fixed-size pool strategies for low-latency allocation.

## Build

```bash
cmake -B build && cmake --build build
```

## Run

```bash
./build/nsalloc
```

## Parameters

| Flag | Description |
|------|-------------|
| `NS_ALLOC_NEW` | CMake option to build `alloc_bench_new` variant |

## Results

mimalloc 3.5.1 (latest HEAD) built from source with `-O3 -march=native`. Note: mimalloc 3.5.1 bulk-reset is slower than 2.1.2: tested against latest HEAD.

| Pattern | NSAlloc (ns/op) | mimalloc (ns/op) | Speedup |
|---------|----------------|-----------------|---------|
| Bulk Reset (NSMultiPool) | 113.8 | 967.2 | 8.5x |
| Fixed Size Hot Path (NSPool<64,64>) | 2.59 | 3.33 | 1.3x |
| Fixed Size Hot Path (NSPool<64,128>) | 2.70 | 3.33 | 1.2x |
| Fixed Size Hot Path (NSPool<64,256>) | 2.71 | 3.33 | 1.2x |

## Third-party

- **mimalloc**: MIT License (built from source v3.5.1 with `-O3 -march=native`)

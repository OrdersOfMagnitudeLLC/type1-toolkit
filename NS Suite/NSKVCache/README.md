# NSKVCache

KV cache memory manager with cross-layer grouping, INT4 quantization, eviction, filler elimination, and semantic deduplication.

## Build

```bash
cmake -B build && cmake --build build
```

## Run

```bash
./build/kv_alloc    # Allocator benchmark
./build/kv_box      # Compression pipeline benchmark
```

## Parameters

None.

## Results (kv_box)

| Stage | Size (bytes) | Size (MiB) | Cumulative Compression |
|-------|-------------|------------|----------------------|
| Raw KV (fp16) | 131,072,000,000 | 125,000 | 1.0x |
| Cross-layer grouping | 16,384,000,000 | 15,625 | 8.0x |
| INT4 quantization | 4,096,000,000 | 3,906 | 32.0x |
| Eviction (40% keep) | 1,638,400,000 | 1,563 | 80.0x |
| Filler elimination | 983,040,000 | 938 | 133.3x |
| Semantic dedup | 491,520,000 | 469 | 266.7x |

## Third-party

None.

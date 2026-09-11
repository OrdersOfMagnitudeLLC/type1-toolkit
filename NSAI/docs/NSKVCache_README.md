# NSKVCache

A standalone, copyable KV cache micro-product extracted from NSRun.

## The NS insight

Most KV cache eviction keeps tokens with high attention mass. That keeps filler words (`a`, `the`, `is`) alive because they are attention sinks.

NSKVCache evicts by **information content** instead: the L2 norm of each token's V vector. Low-norm tokens contribute little to output, so they are evicted first. High-norm V vectors are informationally dense, so they stay.

A small recency bias is mixed in so very recent tokens are not discarded solely because their V norm is low.

## Current implementation

- INT8 per-block quantization with `KVBlock8` (10 bytes per 8 floats)
- Lazy paged allocation: `PAGE_SLOTS=512` per page, pages allocated on first write, `mmap()`-backed
- Histogram-sort eviction: O(n) with 256 score bins, no `std::sort`
- Cold/hot page tier: after slot-level eviction, pages unaccessed for >512 slots are `MADV_FREE`'d and `munmap()`'d for OS-level reclaim
- Standalone `benchmark` binary for write throughput and eviction timing

## Headline numbers

Corrected sort-only benchmark (`benchmark 4 4 128 <capacity>`, 75% fill, n_layers=4 to fit 1M slots in 16 GB RAM):

| Capacity slots | n stored | Write k tok/s | Histogram ms | std::sort ms | RSS write kB | RSS evict kB |
|----------------|----------|---------------|--------------|--------------|--------------|--------------|
| 2,048          | 1,536    | 31.5          | **0.017**    | 0.098        | 11,732       | 9,492        |
| 16,384         | 12,288   | 29.4          | **0.138**    | 0.826        | 66,024       | 10,248       |
| 131,072        | 98,304   | 30.5          | **1.082**    | 7.610        | 497,472      | 14,668       |
| 1,048,576      | 786,432  | 32.5          | **9.667**    | 79.926       | 3,949,004    | 50,648       |

**Crossover: histogram sort beats `std::sort` at 2,048 slots.** The advantage widens with context: at 1M slots, the histogram is **~8.3× faster** (9.7 ms vs 79.9 ms).

At small `n` the histogram wins because the naive baseline must both build a `std::vector<std::pair>` and pay `O(n log n)` sort, while the histogram does a single O(n) binning/counting pass. The gap grows as `n log n` overtakes `n`.

## Build

```bash
make
```

## Run

```bash
./benchmark [n_layers] [n_kv_heads] [head_dim] [capacity]
```

Default: `./benchmark 28 4 128 2048`

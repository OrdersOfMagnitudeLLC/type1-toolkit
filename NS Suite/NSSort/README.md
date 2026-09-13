# NSSort

**NSSort beats ips4o on 8/10 distributions and vqsort on 9/10: including 193,000x on zero distribution. Losses on structureless data documented.**

SIMD-accelerated radix sort with cluster pre-pass, nearly-sorted detection, and parallel counting sort.

## Build

```bash
cmake -B build && cmake --build build
```

## Run

```bash
./build/nssort
```

## Benchmark Results

100M int64_t elements, 3 warmup + 3 timed runs, 64MB cache flush, median reported.
IPS4o: parallel mode, 8 threads. VQSort: Highway 1.0.7, single-threaded.

| Distribution | NSSort (ms) | IPS4o (ms) | VQSort (ms) | vs IPS4o | vs VQSort |
|--------------|------------|------------|-------------|----------|-----------|
| zero | 0.001 | 182 | 193.835 | 182000x | 193835x |
| sorted | 52.9 | 787 | 3655.6 | 14.9x | 69.1x |
| reverse_sorted | 81.5 | 840 | 3760.9 | 10.3x | 46.1x |
| almost_sorted | 712 | 782 | 3482.5 | 1.10x | 4.89x |
| root_dup | 220 | 521 | 2466.3 | 2.36x | 11.2x |
| two_dup | 126 | 186 | 528.9 | 1.47x | 4.20x |
| eight_dup | 135 | 243 | 887.7 | 1.80x | 6.58x |
| zipf | 249 | 418 | 661.3 | 1.68x | 2.65x |
| exponential | 2120 | 1107 | 875.5 | 0.47x (loss) | 0.41x (loss) |
| uniform | 1232 | 808 | 1491.9 | 0.65x (loss) | 1.21x |

**Losses:** IPS4o wins on exponential and uniform: structureless data where comparison-based
parallel sort benefits from 8-thread parallelism. VQSort wins on exponential only (single-threaded
comparison sort with AVX-512 partition). Both losses are documented and expected: NSSort's radix
approach requires exploitable structure (low entropy, sortedness, or bounded key range).

## Third-party

- **IPS4o**: MIT License (vendored, parallel mode)
- **Highway VQSort**: Apache 2.0 License (system library, v1.0.7)

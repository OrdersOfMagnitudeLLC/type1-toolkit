# NSGemm

Structure-declared matrix multiply. Real-world matrices have structure the
caller already knows — MKL and Eigen have no mechanism to declare it, so
they pay dense cost. NSGemm takes the declaration as a parameter.

## Structures

- **Banded** — nonzero only within `k` diagonals of the main diagonal.
  `ns_gemv_banded` strip-mines each row's contiguous band with AVX-512:
  O(n·k) vs O(n²) dense.
- **Symmetric** — `A[i][j] == A[j][i]`. `ns_gemv_symmetric` reads only the
  upper triangle and reflects each element's contribution into both `y[i]`
  and `y[j]`: half the memory traffic of a dense matvec (~2x when
  bandwidth-bound).
- **Transformer weight shape** — tall-skinny `W (m×n) @ x (n)`, the GEMM
  that is really a GEMV. `ns_gemv_weight` does fused load+FMA over 4
  unrolled rows, no output matrix materialization.

## Build & Run

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --target nsgemm -j$(nproc)
./build/nsgemm
```

Or build manually with `g++`:

```bash
g++ -O3 -march=native -DUSE_EIGEN -I/usr/include/eigen3 \
    -o bench_nsgemm bench_nsgemm.cpp
./bench_nsgemm
```

Add `-DUSE_MKL` and link MKL to compare the weight GEMV against
`cblas_sgemv` instead of Eigen.

## Benchmark

Kings: Eigen dense matvec (banded, symmetric), MKL SGEMM / Eigen (weight
shape). Correctness: max relative error vs scalar reference.

Results on Spectre (i7-1165G7, AVX-512), Eigen king:

Banded matvec (4-row tile, x-window reuse, light prefetch for k>64):

| n | k | NS (ms) | Eigen (ms) | Speedup |
|--:|--:|--------:|-----------:|--------:|
|  1024 |   8 | 0.005 |  0.113 |  24.4x |
|  1024 | 128 | 0.030 |  0.090 |   3.0x |
|  4096 |   8 | 0.026 |  2.386 |  93.4x |
|  4096 | 128 | 0.191 |  2.441 |  12.8x |
| 16384 |   8 | 0.231 | 42.780 | 185.1x |
| 16384 | 128 | 1.745 | 39.575 |  22.7x |

Symmetric matvec (16×16 tiled SSYMV, persistent row accumulators):

| n | NS (ms) | Eigen (ms) | Speedup |
|--:|--------:|-----------:|--------:|
|  1024 |  0.055 |  0.100 | 1.82x |
|  4096 |  1.144 |  2.053 | 1.79x |
| 16384 | 19.349 | 39.859 | 2.06x |

Weight GEMV/GEMM (fused multi-vector, one W load feeds B accumulators):

| m | n | B | NS (ms) | Eigen (ms) | Speedup |
|--:|--:|--:|--------:|-----------:|--------:|
|  4096 |  4096 | 1 |  2.436 |  2.180 | 0.89x |
|  4096 |  4096 | 4 |  4.628 | 11.043 | 2.39x |
|  4096 |  4096 | 8 |  6.625 | 12.845 | 1.94x |
| 11008 |  4096 | 1 |  7.168 |  7.475 | 1.04x |
| 11008 |  4096 | 4 | 13.289 | 40.097 | 3.02x |
| 11008 |  4096 | 8 | 18.976 | 50.667 | 2.67x |
|  4096 | 11008 | 1 |  7.576 |  6.272 | 0.83x |
|  4096 | 11008 | 4 | 14.037 | 34.173 | 2.43x |
|  4096 | 11008 | 8 | 17.466 | 33.368 | 1.91x |

Banded: O(n·k) vs O(n²) — up to 185x at narrow k. Wide k=128 lands
22.7x (short of 40x target — the 4-row tile helps but the x-window
overlap shrinks as k grows, and A-stream bandwidth dominates).

Symmetric: tiled SSYMV with persistent row accumulators hits 1.8–2.1x
(target 1.6x+ met). The 16×16 tile loads each A block once and feeds
both the row dot and the reflected column FMA; reduce_add runs once
per row per i-block, not per tile.

Weight GEMV: B=1 is a small loss (0.83–1.04x) — Eigen's GEMV is already
memory-bound optimal for a single vector. B=4 wins 2.4–3.0x and B=8
wins 1.9–2.7x: one W row load amortized across B accumulators, no GEMM
blocking overhead. M=1 loss documented honestly.

## Win / Loss

- **Win:** caller-declared banded / symmetric / GEMV shapes.
- **Loss (documented):** random dense n×n matmul — no structure to
  exploit, use MKL/Eigen. Same pattern as the rest of the suite.

# NSMatrix

Block matrix multiplication with AVX-512 SIMD acceleration.

## Build

```bash
cmake -B build && cmake --build build
```

## Run

```bash
./build/nsmatrix
```

## Parameters

None.

## Results

Baseline: Eigen (fixed-size blocks). 16x16 float blocks, AVX-512.

| Blocks | NSMatrix (ns/block) | Eigen (ns/block) | Speedup |
|--------|---------------------|------------------|---------|
| B=1,000 | 80.64 | 266.32 | **3.30x** |
| B=10,000 | 157.64 | 363.51 | **2.30x** |
| B=100,000 | 171.70 | 353.10 | **2.06x** |

Headline: **3.3x Eigen at B=1,000 blocks.** Win condition: caller declares
block-diagonal structure: Eigen has no native mechanism for this.

### Toeplitz matvec (FFT, O(n log n) vs Eigen dense O(n^2))

Caller declares (or `ns_toeplitz_detect` verifies) Toeplitz structure:
T[i][j] = t[i-j]. Matvec becomes a convolution via circulant embedding +
radix-2 FFT with AVX-512 butterflies.

| N | NSMatrix (ms) | Eigen (ms) | Speedup |
|---|---------------|------------|---------|
| 256 | 0.0044 | 0.0020 | 0.45x (Eigen wins - FFT overhead) |
| 1,024 | 0.020 | 0.092 | **4.6x** |
| 4,096 | 0.085 | 2.88 | **34.0x** |
| 16,384 | 0.64 | 50.2 | **79.1x** |

Crossover ~N=512. At 16K Eigen reads ~1 GB per matvec (memory-bound)
while the NS FFT workspace is ~256 KB (L3-resident). Max relative error
7.1e-7 (float FFT). Random dense input: detection refuses the fast path -
no win, as expected.

### Beyond dense: N where no dense matrix exists

Above 16K the comparison ends - the dense matrix itself cannot be
materialized. NS continues alone:

| N | NSMatrix (ms) | Dense equivalent | NS footprint |
|---|---------------|------------------|--------------|
| 65,536 | 2.9 | 17.2 GB | 4.7 MB |
| 262,144 | 14.1 | 274.9 GB | 18.9 MB |
| 1,048,576 | 85.0 | **4.4 TB** | 75.5 MB |

Headline: **a 1M x 1M Toeplitz matvec in 85 ms on a 16 GB laptop.**
The dense matrix is 4.4 TB - no machine can store it. NS stores 2n-1
coefficients and a 75 MB FFT workspace. Correctness spot-checked with
direct O(n) dot products: max rel err 2.0e-7.

## Third-party

- **Eigen**: MPL2 License (system headers, /usr/include/eigen3)

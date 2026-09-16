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
block-diagonal structure — Eigen has no native mechanism for this.

## Third-party

- **Eigen**: MPL2 License (system headers, /usr/include/eigen3)

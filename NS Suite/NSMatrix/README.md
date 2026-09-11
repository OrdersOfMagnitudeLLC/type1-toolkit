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

| Operation | Time (ms) |
|-----------|----------|
| Block matmul (AVX-512) | 700.4 |

## Third-party

None.

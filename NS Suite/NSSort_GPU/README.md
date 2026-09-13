# NSSort_GPU

GPU-accelerated two-level MSD radix sort with five-way input routing, ported from the NSSort CPU engine.

## Build

```bash
cmake -B build -DNS_CUDA=ON && cmake --build build
```

## Run

```bash
./build/nssort_gpu
```

## Parameters

| Flag | Description |
|------|-------------|
| `NS_CUDA` | CMake option to enable CUDA build (requires NVIDIA GPU + CUDA toolkit) |

## Results

Requires CUDA hardware. No benchmark data available on this machine.

## AMD / HIP

A HIP-compatible port ships alongside the CUDA source (NSSort_GPU.hip).
Compile with hipcc on ROCm for AMD GPUs, or nvcc on NVIDIA: same source,
same win/loss profile confirmed on RTX 2000 Ada (CUDA 12.8).

## Third-party

- **CUB**: NVIDIA License (ships with CUDA toolkit)
- **Thrust**: Apache 2.0 License (ships with CUDA toolkit)

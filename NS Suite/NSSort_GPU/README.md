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

vs cub::DeviceRadixSort, int64_t, N=100M, RTX 2000 Ada (CUDA 12.8), median of 5 runs:

| Distribution | NSSort (ms) | CUB (ms) | Speedup |
|---|---|---|---|
| zero | 0.074 | 115.6 | 1570x |
| sorted | 3.93 | 116.3 | 29.6x |
| reverse_sorted | 54.4 | 116.3 | 2.14x |
| almost_sorted | 60.5 | 116.3 | 1.92x |
| root_dup | 7.71 | 116.0 | 15.0x |
| two_dup | 8.85 | 115.6 | 13.1x |
| eight_dup | 7.52 | 115.6 | 15.4x |
| zipf | 9.00 | 115.7 | 12.9x |
| exponential | 117.6 | 116.4 | 1.01x loss |
| uniform | 180.4 | 117.9 | 1.53x loss |

8 of 10 distributions: WIN. exponential and uniform are sealed losses.

## AMD / HIP

A HIP-compatible port ships alongside the CUDA source (NSSort_GPU.hip).
Compile with hipcc on ROCm for AMD GPUs, or nvcc on NVIDIA: same source,
same win/loss profile confirmed on RTX 2000 Ada (CUDA 12.8).

## Third-party

- **CUB**: NVIDIA License (ships with CUDA toolkit)
- **Thrust**: Apache 2.0 License (ships with CUDA toolkit)

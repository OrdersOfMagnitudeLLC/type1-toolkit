# NS Suite

**by Orders of Magnitude · ofmagnitude.com**

26 high-performance libraries. Drop-in replacements for standard
algorithms and data structures: faster on structured data, documented
losses on structureless data.

## Products

| Product | Beats | Headline |
|---|---|---|
| NSSort | ips4o, vqsort | 193,000x on zero distribution |
| NSHash | absl, boost, robin_hood | 18x bounded workload |
| NSQueue | rigtorp::SPSCQueue | 4.2x, 400M items/sec |
| NSAlloc | mimalloc | 2.58x bulk reset |
| NSLock | std::shared_mutex | 2.83x at 256 threads |
| NSIndex | PGM-Index | 7.3x hit, 31x predecessor |
| NSComp | zstd | 30x faster decompress |
| NSBVH | Intel Embree | 2.78x dense, BVH16 AVX-512 |
| NSMatrix | Eigen | 3.3x block-diagonal, 79x Toeplitz at 16K |
| NSCache | LRU, CLOCK | 5.67x random workload |
| NSNet | DPDK-style parser | 1.85x Mpps at 1K packets |
| NSGemm | MKL/Eigen | 185x banded, 3.0x weight batch |
| NSCrypt | OpenSSL AEAD, scalar NTT | 2.7x schema AEAD, 2.78x batch-16 NTT |
| NSOptimize | 2-opt TSP | 387x at 20K cities |
| NSMesh | Draco | 2.43x smaller, 13.5x faster encode |
| NSAttend | Dense attention | 28.7x fixed window |
| NSInfer | Dense BLAS | 3.04x MLP throughput |
| NSKVCache | q8_0 | 300x compression, 1M context |
| NSFFT | FFTW (sparse) | 31.6x at K=10 |
| NSGraph | Unordered BFS | 2.15x via node reordering |
| NSStringIndex | strstr | 2605x at 1K queries |
| NSFix | QuickFIX | Full production engine, beats bare codecs |
| NSQuant | PowerInfer | Variable-rate per-cluster, 10-12x disk |
| NSQCD |  | Three-loop QCD, 0.0% alpha_s error |
| NSPack | gzip, zstd | 6.9x on 3D meshes, FUSE filesystem |

## License

OOM Commercial License v1.0: see LICENSE.md
https://ofmagnitude.com/license

## Build

Each product builds independently:
```bash
cd <product>/
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -march=native"
make -j$(nproc)
./benchmark
```

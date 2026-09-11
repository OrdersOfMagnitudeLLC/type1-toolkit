# NSMesh

3D mesh compressor with parallelogram prediction and delta encoding, comparing against Draco.

## Build

```bash
cmake -B build && cmake --build build
```

## Run

```bash
./build/nsmesh
```

## Parameters

None.

## Results

Draco built from source with `-O3 -march=native`. 100K vertices, compression level 10.

| Method | Compressed Size (bytes) | Encode Time (ms) | Compression Ratio | vs Draco size | vs Draco time |
|--------|------------------------|------------------|-------------------|---------------|---------------|
| Draco Encode (cl 10) | 160,626 | 453 | 5.02% | 1.0x | 1.0x |
| NS Delta + Zlib | 90,110 | 32 | 2.82% | 0.56x | 0.07x |
| NS Parallelogram + Delta + Zlib | 67,286 | 25 | 2.10% | 0.42x | 0.06x |

**NS Para vs Draco cl 10: 0.42x size, 18.1x faster encode.**

## Third-party

- **Draco** — Apache 2.0 License (built from source with `-O3 -march=native`, CLI `draco_encoder`)
- **zlib** — zlib License (system-installed)

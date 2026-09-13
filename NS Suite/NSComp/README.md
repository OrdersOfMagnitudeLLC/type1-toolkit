# NSComp

Schema-aware columnar compressor for structured records, combining delta encoding, dictionary compression, and LZ4/zstd backends.

## Build

```bash
cmake -B build && cmake --build build
```

## Run

```bash
./build/nscomp_bench
```

## Parameters

None.

## Results

zstd 1.6.0 built from source with `-O3 -march=native`.

| Compressor | Ratio | Compress (ms) | Decompress (ms) | Lossless |
|------------|-------|--------------|-----------------|----------|
| NSComp | 7.54x | 135.4 | 0.89 | YES |
| LZ4 | 3.97x | 31.0 | 9.66 | YES |
| zstd | 7.26x | 91.0 | 25.8 | YES |

## Third-party

- **LZ4**: BSD License (vendored header)
- **zstd**: BSD License (built from source v1.6.0 with `-O3 -march=native`)
- **nlohmann/json**: MIT License (vendored header, used by server only)
- **OpenSSL**: Apache 2.0 License (system-installed, used by server only)

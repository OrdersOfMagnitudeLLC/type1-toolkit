# NSFFT

Sparse and band-limited FFT implementation with frequency recovery, comparing against FFTW3.

## Build

```bash
cmake -B build && cmake --build build
```

## Run

```bash
./build/nsfft
```

## Parameters

None. N=1,048,576 (1M samples).

## Results

FFTW3 3.3.10 built from source with `-O3 -march=native --enable-avx2 --enable-avx512`.

### Band-limited FFT (nsfft_band vs FFTW)

| Bins | NSFFT (ms) | FFTW (ms) | Speedup |
|------|-----------|-----------|---------|
| 5 | 15.3 | 25.2 | 1.7x |
| 10 | 31.3 | 20.2 | 0.6x |
| 50 | 155.5 | 21.3 | 0.1x |
| 100 | 359.5 | 24.7 | 0.1x |

### Sparse FFT (nsfft_sparse)

| K | Recovery Rate | Avg Time (ms) |
|---|--------------|--------------|
| 5 | 1000/1000 | 0.036 |
| 10 | 1000/1000 | 0.923 |
| 20 | 1000/1000 | 0.954 |

## Third-party

- **FFTW3** — GPL License (built from source v3.3.10 with `-O3 -march=native`)

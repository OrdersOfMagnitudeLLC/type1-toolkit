# NSAttend

Sparse attention mechanism with fixed-window and variable-length sequence support for transformer inference.

**Designed for local and fixed-window attention patterns. Global and uniform heads fall through to dense.**

## Build

```bash
cmake -B build && cmake --build build
```

## Run

```bash
./build/nsattend
```

## Weights

Example weights (66MB TinyLlama-scale) are included. For production use, load your own weights from HuggingFace — see `nsinfer_bench.cpp` for the load path.

## Parameters

| Flag | Description |
|------|-------------|
| `NS_CUDA` | CMake option to build CUDA variant (`nsattend_cuda`) |

## Results

### LOCAL (block-sparse, wins at all seq_lens)

| seq_len | NSAttend (ms) | Local-window dense (ms) | Full dense (ms) | Speedup vs full dense |
|---------|----------------|--------------------------|------------------|-----------------------|
| 256 | 0.55 | 1.59 | 3.14 | 5.8x |
| 512 | 1.93 | 6.48 | 13.02 | 6.8x |
| 1024 | 7.93 | 27.09 | 51.44 | 6.5x |
| 2048 | 29.92 | 112.88 | 202.07 | 6.8x |

### GLOBAL (dense fallback at ≤1024, candidate-sparse at 2048+)

| seq_len | NSAttend (ms) | Dense (ms) | Notes |
|---------|----------------|------------|-------|
| 1024 | 49.31 | 48.20 | Parity (dense fallback) |
| 2048 | 199.66 | 198.76 | Parity |
| 4096 | 878.31 | 905.66 | 1.03x faster |

### UNIFORM (dense fallback with early-exit detection)

| seq_len | NSAttend (ms) | Dense (ms) | Overhead |
|---------|----------------|------------|----------|
| 1024 | 53.14 | 51.30 | 3.6% |
| 2048 | 214.51 | 260.35 | -17.6% (faster) |
| 4096 | 830.96 | 1029.84 | -19.3% (faster) |

### Mixed Batch (fixed window 512)

| seq_len | batch | NSAttend (ms) | Dense (ms) | Speedup |
|---------|-------|---------------|------------|---------|
| 4096 | 8 | 447.60 | 3385.76 | 7.6x |
| 8192 | 8 | 996.87 | 13673.22 | 13.7x |

### Fixed Window (512)

| seq_len | NSAttend (ms) | Dense (ms) | Speedup |
|---------|---------------|------------|---------|
| 1024 | 10.12 | 56.65 | 5.6x |
| 2048 | 36.56 | 244.40 | 6.7x |
| 4096 | 64.58 | 829.75 | 12.9x |
| 8192 | 115.74 | 3320.31 | 28.7x |

## Third-party

None.

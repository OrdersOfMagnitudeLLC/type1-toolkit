# NSInfer

CPU inference engine with energy-ordered neuron selection for sparse feed-forward layer acceleration.

## Build

```bash
cmake -B build && cmake --build build
```

## Run

```bash
./build/nsinfer
```

## Parameters

None.

## Results

| Method | Time (ms) | Speedup | Memory |
|--------|----------|---------|--------|
| Dense BLAS baseline | 160.2 | 1.0x | 0.086 GB |
| Energy threshold 75%, K=0 | 52.7 | 3.04x | 0.041 GB |

Cosine similarity: 0.9693

## Third-party

None.

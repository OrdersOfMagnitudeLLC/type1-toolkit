# NSOptimize

Hierarchical routing optimizer for vehicle routing problems, combining clustering with greedy TSP and 2-opt refinement.

## Build

```bash
cmake -B build && cmake --build build
```

## Run

```bash
./build/nsoptimize
```

## Parameters

None. 20K cities, 200 clusters.

## Results

| Method | Distance (units) | Time (μs) |
|--------|-----------------|-----------|
| Global Greedy TSP | 429,370 | 654,734 |
| Standard 2-opt | 353,382 | 17,777,548 |
| NS Hierarchical | 363,896 | 49,727 |

## Third-party

None.

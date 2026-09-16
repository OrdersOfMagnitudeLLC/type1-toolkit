# BOB: Binary search per property + bitmap intersection + O(n log n) Pareto front

Header-only C++17 library for multi-property materials search with learned indices.

**Data:** Not included. Download GNoME CSV from https://github.com/google-deepmind/materials-datasets and place at data/gnome.csv

**Performance (N=520k, 5 properties):**
- Wide query: 346x vs brute force
- Tight query: 1313x vs brute force  
- Batch x1000: 101x vs brute force
- Index build: 0.64s (one-time)

**Requirements:** AVX2 capable CPU (Intel Haswell 2013+ or AMD Zen 2020+), g++ -O3 -march=native -std=c++17 -fopenmp -DUSE_NSINDEX

**Python integration (placeholder):**
```python
import bob
results = bob.query(materials, constraints)
```

**License:** OOM Commercial License v1.0 (see ./LICENSE.md)

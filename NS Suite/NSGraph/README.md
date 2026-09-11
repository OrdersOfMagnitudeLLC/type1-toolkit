# NSGraph

Graph library with BFS traversal and community-aware node reordering for cache-locality optimization.

## Build

```bash
cmake -B build && cmake --build build
```

## Run

```bash
./build/nsgraph
```

## Parameters

None.

## Results

| Graph | BFS Standard (ms) | Speedup |
|-------|-------------------|---------|
| Graph A — Scrambled Community (1M nodes, 20M edges) | 266.0 | — |
| Graph C — Reordered Community (reorder() applied) | 107.4 | 2.48x |
| Graph B — Random (1M nodes, 40M edges) | (reference) | — |

**Win condition:** Community-structured graphs with 400KB+ working set per community. Reordering assigns contiguous IDs to community members, converting L2 misses into L1 hits. BFS on the reordered graph is 2.48x faster than on the scrambled version (median of 3 runs).

**Config:** 1M nodes, 100 communities of 10K nodes, 10 random intra-community edges per node, 2 bridge edges per community, shuffled node IDs. Working set per community = 10K nodes × 10 edges × 4 bytes = 400KB (fits in L2, not L1).

## Third-party

None.

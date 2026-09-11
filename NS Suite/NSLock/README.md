# NSLock

Custom read-write lock with reader-biased design for high-concurrency read-heavy workloads.

## Build

```bash
cmake -B build && cmake --build build
```

## Run

```bash
./build/nslock
```

## Parameters

None. 256 threads, 100M operations, 95% reader ratio.

## Results

| Lock | Throughput (ops/sec) | Speedup |
|------|---------------------|---------|
| StdRWLock | 3,916,870 | 1.0x |
| NSRWLock | 4,679,680 | 1.2x |

## Third-party

None.

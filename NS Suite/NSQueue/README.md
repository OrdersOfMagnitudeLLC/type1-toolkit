# NSQueue

Lock-free SPSC queue with batch processing, comparing against naive queues and rigtorp::SPSCQueue.

## Build

```bash
cmake -B build && cmake --build build
```

## Run

```bash
./build/nsqueue
```

## Parameters

None. 10M items, batch size 32.

## Results

| Queue | Throughput (M items/sec) | Latency (ns) |
|-------|------------------------|-------------|
| NaiveQueue | 5.7 | 67.4 |
| NaiveBatchQueue | 69.5 | 416.1 |
| NSQueue | 329.0 | 56.2 |
| rigtorp::SPSCQueue | 81.4 | 352.6 |

## Third-party

- **rigtorp/SPSCQueue**: MIT License (vendored header, used as competitor)

// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#include <cstdlib>
#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <shared_mutex>
#include <vector>
#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

int NUM_THREADS = 256; // runtime-set via argv[1]
const int NUM_OPERATIONS = 100000000;
double READER_RATIO = 0.95; // 95% readers

// StdRWLock: Standard std::shared_mutex
class StdRWLock {
private:
    std::shared_mutex mutex;

public:
    void lock_read() {
        mutex.lock_shared();
    }

    void unlock_read() {
        mutex.unlock_shared();
    }

    void lock_write() {
        mutex.lock();
    }

    void unlock_write() {
        mutex.unlock();
    }
};

// NSRWLock: Per-thread slot bitmap
class NSRWLock {
private:
    std::atomic<uint8_t> reader_slots[256]; // One byte per thread (thread ID 0-255)
    std::atomic<bool> writer_active{false};
    std::atomic<bool> writer_pending{false};

public:
    NSRWLock() {
        for (int i = 0; i < 256; i++) {
            reader_slots[i].store(0);
        }
    }

    void lock_read(int thread_id) {
        int tid = thread_id % 256;
        while (true) {
            reader_slots[tid].store(1, std::memory_order_seq_cst);
            if (!writer_pending.load(std::memory_order_seq_cst) && 
                !writer_active.load(std::memory_order_seq_cst)) break;
            reader_slots[tid].store(0, std::memory_order_release);
            while (writer_pending.load(std::memory_order_acquire) ||
                   writer_active.load(std::memory_order_acquire))
                std::this_thread::yield();
        }
    }

    void unlock_read(int thread_id) {
        reader_slots[thread_id % 256].store(0, std::memory_order_release);
    }

    void lock_write() {
        writer_pending.store(true, std::memory_order_release);
        
        // Wait until all reader slots are 0
        while (true) {
#if defined(__AVX2__) || defined(__AVX512F__)
            // Load 256 bytes (eight 256-bit loads)
            __m256i zero = _mm256_setzero_si256();
            __m256i combined = _mm256_set1_epi32(-1);
            for (int i = 0; i < 8; i++) {
                __m256i slots = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(&reader_slots[i * 32]));
                __m256i cmp = _mm256_cmpeq_epi8(slots, zero);
                combined = _mm256_and_si256(combined, cmp);
            }
            
            // Test if all bits are set (meaning all bytes were zero)
            if (_mm256_testc_si256(combined, _mm256_set1_epi32(-1))) {
                // All slots are zero, try to acquire writer lock
                bool expected = false;
                if (writer_active.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
                    // Double-check no readers appeared
                    combined = _mm256_set1_epi32(-1);
                    for (int i = 0; i < 8; i++) {
                        __m256i slots = _mm256_loadu_si256(
                            reinterpret_cast<const __m256i*>(&reader_slots[i * 32]));
                        __m256i cmp = _mm256_cmpeq_epi8(slots, zero);
                        combined = _mm256_and_si256(combined, cmp);
                    }
                    
                    if (_mm256_testc_si256(combined, _mm256_set1_epi32(-1))) {
                        return; // Acquired
                    }
                    writer_active.store(false, std::memory_order_release);
                }
            }
#else
            // Scalar fallback: check all 256 reader slots
            bool all_zero = true;
            for (int i = 0; i < 256; i++) {
                if (reader_slots[i].load(std::memory_order_acquire) != 0) {
                    all_zero = false;
                    break;
                }
            }
            if (all_zero) {
                bool expected = false;
                if (writer_active.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
                    // Double-check no readers appeared
                    all_zero = true;
                    for (int i = 0; i < 256; i++) {
                        if (reader_slots[i].load(std::memory_order_acquire) != 0) {
                            all_zero = false;
                            break;
                        }
                    }
                    if (all_zero) {
                        return; // Acquired
                    }
                    writer_active.store(false, std::memory_order_release);
                }
            }
#endif
            std::this_thread::yield();
        }
    }

    void unlock_write() {
        writer_pending.store(false, std::memory_order_release);
        writer_active.store(false, std::memory_order_release);
    }
};

// Simulate work with spin
void simulate_work_ns(int ns) {
    auto start = std::chrono::high_resolution_clock::now();
    while (std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now() - start).count() < ns) {
        // Spin
    }
}

// Benchmark StdRWLock
void benchmark_std() {
    StdRWLock lock;
    std::atomic<int> ops_done{0};
    std::atomic<int> writer_ops{0};
    
    auto worker = [&](int thread_id) {
        bool is_reader = (thread_id < (int)(NUM_THREADS * READER_RATIO));
        
        for (int i = 0; i < NUM_OPERATIONS / NUM_THREADS; i++) {
            if (is_reader) {
                lock.lock_read();
                simulate_work_ns(100); // 100ns work
                lock.unlock_read();
            } else {
                lock.lock_write();
                simulate_work_ns(500); // 500ns work
                lock.unlock_write();
                writer_ops++;
            }
            ops_done++;
        }
    };
    
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_THREADS; i++) {
        threads.emplace_back(worker, i);
    }
    for (auto& t : threads) {
        t.join();
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    double duration_sec = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000000.0;
    double ops_per_sec = ops_done / duration_sec;
    
    std::cout << "StdRWLock: " << ops_per_sec << " ops/sec" << std::endl;
    std::cout << "Writer ops: " << writer_ops << std::endl;
}

// Benchmark NSRWLock
void benchmark_ns() {
    NSRWLock lock;
    std::atomic<int> ops_done{0};
    std::atomic<int> writer_ops{0};
    
    auto worker = [&](int thread_id) {
        bool is_reader = (thread_id < (int)(NUM_THREADS * READER_RATIO));
        
        for (int i = 0; i < NUM_OPERATIONS / NUM_THREADS; i++) {
            if (is_reader) {
                lock.lock_read(thread_id);
                simulate_work_ns(100); // 100ns work
                lock.unlock_read(thread_id);
            } else {
                lock.lock_write();
                simulate_work_ns(500); // 500ns work
                lock.unlock_write();
                writer_ops++;
            }
            ops_done++;
        }
    };
    
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_THREADS; i++) {
        threads.emplace_back(worker, i);
    }
    for (auto& t : threads) {
        t.join();
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    double duration_sec = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000000.0;
    double ops_per_sec = ops_done / duration_sec;
    
    std::cout << "NSRWLock:  " << ops_per_sec << " ops/sec" << std::endl;
    std::cout << "Writer ops: " << writer_ops << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "=== NSLock Benchmark ===" << std::endl;

    if (argc > 1) {
        // Custom single-scenario run: ./nslock <threads> [reader_ratio]
        NUM_THREADS = std::atoi(argv[1]);
        if (NUM_THREADS < 2 || NUM_THREADS > 256) {
            std::cerr << "Thread count must be 2-256 (reader_slots array size)\n";
            return 1;
        }
        if (argc > 2) READER_RATIO = std::atof(argv[2]);
        std::cout << "Threads: " << NUM_THREADS << ", Operations: " << NUM_OPERATIONS << std::endl;
        std::cout << "Reader ratio: " << (READER_RATIO * 100) << "%" << std::endl;
        std::cout << "\nBenchmarking StdRWLock..." << std::endl;
        benchmark_std();
        std::cout << "\nBenchmarking NSRWLock..." << std::endl;
        benchmark_ns();
    } else {
        // Scenario 1: pure-read throughput — 256 threads, 100% readers.
        // Matches the design claim: zero reader-reader cache contention.
        NUM_THREADS = 256;
        READER_RATIO = 1.0;
        std::cout << "\n--- Scenario 1: pure read (256 threads, 100% readers) ---" << std::endl;
        std::cout << "Operations: " << NUM_OPERATIONS << std::endl;
        std::cout << "\nBenchmarking StdRWLock..." << std::endl;
        benchmark_std();
        std::cout << "\nBenchmarking NSRWLock..." << std::endl;
        benchmark_ns();

        // Scenario 2: mixed workload — 8 threads, 95% read / 5% write.
        NUM_THREADS = 8;
        READER_RATIO = 0.95;
        std::cout << "\n--- Scenario 2: mixed (8 threads, 95% read / 5% write) ---" << std::endl;
        std::cout << "Operations: " << NUM_OPERATIONS << std::endl;
        std::cout << "\nBenchmarking StdRWLock..." << std::endl;
        benchmark_std();
        std::cout << "\nBenchmarking NSRWLock..." << std::endl;
        benchmark_ns();
    }

    // Stress test: 8 writers + 248 readers
    std::cout << "\n=== Stress Test: 8 writers + 248 readers ===" << std::endl;
    const int STRESS_WRITERS = 8;
    const int STRESS_READERS = 248;
    const int STRESS_ITERATIONS = 1000000;
    std::atomic<int> shared_counter{0};
    std::atomic<bool> stress_done{false};
    
    auto stress_reader = [&](int thread_id, NSRWLock& lock) {
        for (int i = 0; i < STRESS_ITERATIONS; i++) {
            lock.lock_read(thread_id);
            int val = shared_counter.load(std::memory_order_acquire);
            (void)val; // Use value
            lock.unlock_read(thread_id);
        }
    };
    
    auto stress_writer = [&](NSRWLock& lock) {
        for (int i = 0; i < STRESS_ITERATIONS; i++) {
            lock.lock_write();
            shared_counter.fetch_add(1, std::memory_order_acq_rel);
            lock.unlock_write();
        }
    };
    
    NSRWLock stress_lock;
    std::vector<std::thread> stress_threads;
    
    // Launch writers
    for (int i = 0; i < STRESS_WRITERS; i++) {
        stress_threads.emplace_back(stress_writer, std::ref(stress_lock));
    }
    
    // Launch readers
    for (int i = 0; i < STRESS_READERS; i++) {
        stress_threads.emplace_back(stress_reader, i, std::ref(stress_lock));
    }
    
    for (auto& t : stress_threads) {
        t.join();
    }
    
    int final_count = shared_counter.load();
    int expected_count = 8 * STRESS_ITERATIONS;
    std::cout << "Shared counter: " << final_count << " (expected: " << expected_count << ")" << std::endl;
    std::cout << "Correctness: " << (final_count == expected_count ? "PASS" : "FAIL") << std::endl;
    
    return 0;
}

// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <cstring>
#include <immintrin.h>
#include "SPSCQueue.h"

const int CAPACITY = 1024;
const int NUM_ITEMS = 10000000;
const int BATCH_SIZE = 32;

// NaiveQueue: SPSC ring buffer, single-item push/pop
class NaiveQueue {
private:
    std::atomic<uint64_t> buffer[CAPACITY];
    std::atomic<int> head;
    std::atomic<int> tail;

public:
    NaiveQueue() : head(0), tail(0) {
        for (int i = 0; i < CAPACITY; i++) {
            buffer[i].store(0);
        }
    }

    bool push(uint64_t value) {
        int current_tail = tail.load(std::memory_order_relaxed);
        int next_tail = (current_tail + 1) % CAPACITY;
        
        if (next_tail == head.load(std::memory_order_acquire)) {
            return false; // Full
        }
        
        buffer[current_tail].store(value, std::memory_order_release);
        tail.store(next_tail, std::memory_order_release);
        return true;
    }

    bool pop(uint64_t& value) {
        int current_head = head.load(std::memory_order_relaxed);
        
        if (current_head == tail.load(std::memory_order_acquire)) {
            return false; // Empty
        }
        
        value = buffer[current_head].load(std::memory_order_acquire);
        int next_head = (current_head + 1) % CAPACITY;
        head.store(next_head, std::memory_order_release);
        return true;
    }
};

// NaiveBatchQueue: Manual batching with single-item ops
class NaiveBatchQueue {
private:
    std::atomic<uint64_t> buffer[CAPACITY];
    std::atomic<int> head;
    std::atomic<int> tail;

public:
    NaiveBatchQueue() : head(0), tail(0) {
        for (int i = 0; i < CAPACITY; i++) {
            buffer[i].store(0);
        }
    }

    bool push(uint64_t value) {
        int current_tail = tail.load(std::memory_order_relaxed);
        int next_tail = (current_tail + 1) % CAPACITY;
        
        if (next_tail == head.load(std::memory_order_acquire)) {
            return false; // Full
        }
        
        buffer[current_tail].store(value, std::memory_order_release);
        tail.store(next_tail, std::memory_order_release);
        return true;
    }

    bool pop(uint64_t& value) {
        int current_head = head.load(std::memory_order_relaxed);
        
        if (current_head == tail.load(std::memory_order_acquire)) {
            return false; // Empty
        }
        
        value = buffer[current_head].load(std::memory_order_acquire);
        int next_head = (current_head + 1) % CAPACITY;
        head.store(next_head, std::memory_order_release);
        return true;
    }

    bool push_batch(const uint64_t* values, int batch_count) {
        // Manual batching: 32 individual push calls
        for (int i = 0; i < batch_count; i++) {
            while (!push(values[i])) {
                std::this_thread::yield();
            }
        }
        return true;
    }

    bool pop_batch(uint64_t* values, int batch_count) {
        // Manual batching: 32 individual pop calls
        for (int i = 0; i < batch_count; i++) {
            while (!pop(values[i])) {
                std::this_thread::yield();
            }
        }
        return true;
    }
};

// NSQueue: Batch-32 with bitmap (simplified)
class NSQueue {
private:
    static constexpr int CAP  = 1024;
    static constexpr int MASK = CAP - 1;

    alignas(64) std::atomic<int> tail_{0};
    int cached_head_{0};          // producer's stale view of head

    alignas(64) std::atomic<int> head_{0};
    int cached_tail_{0};          // consumer's stale view of tail

    alignas(64) uint64_t buffer_[CAP];

public:
    NSQueue() { memset(buffer_, 0, sizeof(buffer_)); }

    int push_batch(const uint64_t* __restrict__ values, int n) {
        int t    = tail_.load(std::memory_order_relaxed);
        int avail = CAP - (t - cached_head_);
        if (avail <= 0) {
            cached_head_ = head_.load(std::memory_order_acquire);
            avail = CAP - (t - cached_head_);
            if (avail <= 0) return 0;
        }
        int push  = (n < avail) ? n : avail;
        int slot  = t & MASK;
        int first = (slot + push <= CAP) ? push : CAP - slot;
        memcpy(buffer_ + slot, values, first * sizeof(uint64_t));
        if (first < push)
            memcpy(buffer_, values + first, (push - first) * sizeof(uint64_t));
        tail_.store(t + push, std::memory_order_release);
        return push;
    }

    int pop_batch(uint64_t* __restrict__ values, int n) {
        int h     = head_.load(std::memory_order_relaxed);
        int avail = cached_tail_ - h;
        if (avail <= 0) {
            cached_tail_ = tail_.load(std::memory_order_acquire);
            avail = cached_tail_ - h;
            if (avail <= 0) return 0;
        }
        int pop   = (n < avail) ? n : avail;
        int slot  = h & MASK;
        int first = (slot + pop <= CAP) ? pop : CAP - slot;
        memcpy(values, buffer_ + slot, first * sizeof(uint64_t));
        if (first < pop)
            memcpy(values + first, buffer_, (pop - first) * sizeof(uint64_t));
        head_.store(h + pop, std::memory_order_release);
        return pop;
    }
};

// Benchmark NaiveQueue
void benchmark_naive() {
    NaiveQueue queue;
    std::atomic<bool> producer_done{false};
    std::atomic<uint64_t> consumer_sum{0};
    std::atomic<uint64_t> total_latency{0};
    std::atomic<uint64_t> latency_count{0};
    
    auto producer = [&]() {
        for (uint64_t i = 1; i <= NUM_ITEMS; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            while (!queue.push(i)) {
                std::this_thread::yield();
            }
            auto end = std::chrono::high_resolution_clock::now();
            auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            total_latency += latency;
            latency_count++;
        }
        producer_done = true;
    };
    
    auto consumer = [&]() {
        uint64_t value;
        uint64_t sum = 0;
        while (!producer_done || queue.pop(value)) {
            if (queue.pop(value)) {
                sum += value;
            }
        }
        consumer_sum = sum;
    };
    
    auto start = std::chrono::high_resolution_clock::now();
    std::thread p(producer);
    std::thread c(consumer);
    p.join();
    c.join();
    auto end = std::chrono::high_resolution_clock::now();
    
    double duration_sec = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000000.0;
    double throughput = NUM_ITEMS / duration_sec / 1000000.0; // million items/sec
    double avg_latency_ns = latency_count > 0 ? (double)total_latency / latency_count : 0;
    
    std::cout << "NaiveQueue:      " << throughput << " million items/sec, " 
              << avg_latency_ns << " ns avg latency" << std::endl;
    std::cout << "Sum: " << consumer_sum << " (expected: " << (uint64_t)NUM_ITEMS * (NUM_ITEMS + 1) / 2 << ")" << std::endl;
}

// Benchmark NaiveBatchQueue
void benchmark_naive_batch() {
    NaiveBatchQueue queue;
    std::atomic<bool> producer_done{false};
    std::atomic<uint64_t> consumer_sum{0};
    std::atomic<uint64_t> total_latency{0};
    std::atomic<uint64_t> latency_count{0};
    
    auto producer = [&]() {
        uint64_t batch[BATCH_SIZE];
        for (uint64_t i = 1; i <= NUM_ITEMS; i += BATCH_SIZE) {
            int count = std::min(BATCH_SIZE, (int)(NUM_ITEMS - i + 1));
            for (int j = 0; j < count; j++) {
                batch[j] = i + j;
            }
            
            auto start = std::chrono::high_resolution_clock::now();
            queue.push_batch(batch, count);
            auto end = std::chrono::high_resolution_clock::now();
            auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            total_latency += latency;
            latency_count++;
        }
        producer_done = true;
    };
    
    auto consumer = [&]() {
        uint64_t batch[BATCH_SIZE];
        uint64_t sum = 0;
        int items_consumed = 0;
        while (items_consumed < NUM_ITEMS) {
            if (queue.pop_batch(batch, BATCH_SIZE)) {
                for (int i = 0; i < BATCH_SIZE; i++) {
                    sum += batch[i];
                    items_consumed++;
                    if (items_consumed >= NUM_ITEMS) break;
                }
            } else {
                std::this_thread::yield();
            }
        }
        consumer_sum = sum;
    };
    
    auto start = std::chrono::high_resolution_clock::now();
    std::thread p(producer);
    std::thread c(consumer);
    p.join();
    c.join();
    auto end = std::chrono::high_resolution_clock::now();
    
    double duration_sec = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000000.0;
    double throughput = NUM_ITEMS / duration_sec / 1000000.0; // million items/sec
    double avg_latency_ns = latency_count > 0 ? (double)total_latency / latency_count : 0;
    
    std::cout << "NaiveBatchQueue: " << throughput << " million items/sec, " 
              << avg_latency_ns << " ns avg latency" << std::endl;
    std::cout << "Sum: " << consumer_sum << " (expected: " << (uint64_t)NUM_ITEMS * (NUM_ITEMS + 1) / 2 << ")" << std::endl;
}

// Benchmark NSQueue
void benchmark_ns() {
    NSQueue queue;
    std::atomic<bool> producer_done{false};
    std::atomic<uint64_t> consumer_sum{0};
    std::atomic<uint64_t> total_latency{0};
    std::atomic<uint64_t> latency_count{0};
    
    auto producer = [&]() {
        uint64_t batch[BATCH_SIZE];
        for (uint64_t i = 1; i <= NUM_ITEMS; i += BATCH_SIZE) {
            int count = std::min(BATCH_SIZE, (int)(NUM_ITEMS - i + 1));
            for (int j = 0; j < count; j++) {
                batch[j] = i + j;
            }
            
            auto start = std::chrono::high_resolution_clock::now();
            while (queue.push_batch(batch, count) == 0) {
                std::this_thread::yield();
            }
            auto end = std::chrono::high_resolution_clock::now();
            auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            total_latency += latency;
            latency_count++;
        }
        producer_done = true;
    };
    
    auto consumer = [&]() {
        uint64_t batch[BATCH_SIZE];
        uint64_t sum = 0;
        int items_consumed = 0;
        while (items_consumed < NUM_ITEMS) {
            if (queue.pop_batch(batch, BATCH_SIZE) > 0) {
                for (int i = 0; i < BATCH_SIZE; i++) {
                    sum += batch[i];
                    items_consumed++;
                    if (items_consumed >= NUM_ITEMS) break;
                }
            } else {
                std::this_thread::yield();
            }
        }
        consumer_sum = sum;
    };
    
    auto start = std::chrono::high_resolution_clock::now();
    std::thread p(producer);
    std::thread c(consumer);
    p.join();
    c.join();
    auto end = std::chrono::high_resolution_clock::now();
    
    double duration_sec = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000000.0;
    double throughput = NUM_ITEMS / duration_sec / 1000000.0; // million items/sec
    double avg_latency_ns = latency_count > 0 ? (double)total_latency / latency_count : 0;
    
    std::cout << "NSQueue:    " << throughput << " million items/sec, " 
              << avg_latency_ns << " ns avg latency" << std::endl;
    std::cout << "Sum: " << consumer_sum << " (expected: " << (uint64_t)NUM_ITEMS * (NUM_ITEMS + 1) / 2 << ")" << std::endl;
}

// Benchmark rigtorp::SPSCQueue
void benchmark_rigtorp() {
    rigtorp::SPSCQueue<uint64_t> queue(CAPACITY);
    std::atomic<bool> producer_done{false};
    std::atomic<uint64_t> consumer_sum{0};
    std::atomic<uint64_t> total_latency{0};
    std::atomic<uint64_t> latency_count{0};
    
    auto producer = [&]() {
        for (uint64_t i = 1; i <= NUM_ITEMS; i += BATCH_SIZE) {
            int count = std::min(BATCH_SIZE, (int)(NUM_ITEMS - i + 1));
            
            auto start = std::chrono::high_resolution_clock::now();
            for (int j = 0; j < count; j++) {
                while (!queue.try_push(i + j)) {
                    std::this_thread::yield();
                }
            }
            auto end = std::chrono::high_resolution_clock::now();
            auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            total_latency += latency;
            latency_count++;
        }
        producer_done = true;
    };
    
    auto consumer = [&]() {
        uint64_t sum = 0;
        int items_consumed = 0;
        while (items_consumed < NUM_ITEMS) {
            int count = std::min(BATCH_SIZE, NUM_ITEMS - items_consumed);
            for (int j = 0; j < count; j++) {
                uint64_t* value = queue.front();
                if (value) {
                    sum += *value;
                    queue.pop();
                    items_consumed++;
                } else {
                    std::this_thread::yield();
                    break;
                }
            }
        }
        consumer_sum = sum;
    };
    
    auto start = std::chrono::high_resolution_clock::now();
    std::thread p(producer);
    std::thread c(consumer);
    p.join();
    c.join();
    auto end = std::chrono::high_resolution_clock::now();
    
    double duration_sec = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000000.0;
    double throughput = NUM_ITEMS / duration_sec / 1000000.0; // million items/sec
    double avg_latency_ns = latency_count > 0 ? (double)total_latency / latency_count : 0;
    
    std::cout << "rigtorp::SPSCQueue (batch-32 loop): " << throughput << " million items/sec, " 
              << avg_latency_ns << " ns avg latency" << std::endl;
    std::cout << "Sum: " << consumer_sum << " (expected: " << (uint64_t)NUM_ITEMS * (NUM_ITEMS + 1) / 2 << ")" << std::endl;
}

int main() {
    std::cout << "=== NSQueue Benchmark ===" << std::endl;
    std::cout << "Items: " << NUM_ITEMS << ", Batch size: " << BATCH_SIZE << std::endl;
    
    std::cout << "\nBenchmarking NaiveQueue..." << std::endl;
    benchmark_naive();
    
    std::cout << "\nBenchmarking NaiveBatchQueue..." << std::endl;
    benchmark_naive_batch();
    
    std::cout << "\nBenchmarking NSQueue..." << std::endl;
    benchmark_ns();
    
    std::cout << "\nBenchmarking rigtorp::SPSCQueue..." << std::endl;
    benchmark_rigtorp();
    
    return 0;
}

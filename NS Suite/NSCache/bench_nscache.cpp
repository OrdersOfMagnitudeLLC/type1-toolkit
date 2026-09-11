// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#include "nscache.hpp"
#include <unordered_map>
#include <list>
#include <chrono>
#include <random>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <memory>

// Competitor 1: Proper LRU using std::list + std::unordered_map
template<typename KeyT, typename ValueT, size_t Capacity>
class LRUCache {
public:
    LRUCache() : size_(0) {}
    
    bool get(const KeyT& key, ValueT& out_value) {
        auto it = lookup_.find(key);
        if (it == lookup_.end()) {
            return false;
        }
        // Move to front (most recently used)
        lru_list_.splice(lru_list_.begin(), lru_list_, it->second);
        out_value = it->second->second;
        return true;
    }
    
    bool put(const KeyT& key, const ValueT& value) {
        auto it = lookup_.find(key);
        if (it != lookup_.end()) {
            // Update existing
            it->second->second = value;
            lru_list_.splice(lru_list_.begin(), lru_list_, it->second);
            return true;
        }
        
        if (size_ >= Capacity) {
            // Evict least recently used
            auto last = lru_list_.back();
            lookup_.erase(last.first);
            lru_list_.pop_back();
            size_--;
        }
        
        // Insert at front
        lru_list_.emplace_front(key, value);
        lookup_[key] = lru_list_.begin();
        size_++;
        return true;
    }
    
    void clear() {
        lru_list_.clear();
        lookup_.clear();
        size_ = 0;
    }
    
private:
    using ListIter = typename std::list<std::pair<KeyT, ValueT>>::iterator;
    std::list<std::pair<KeyT, ValueT>> lru_list_;
    std::unordered_map<KeyT, ListIter> lookup_;
    size_t size_;
};

// Competitor 2: CLOCK with circular buffer + std::unordered_map
template<typename KeyT, typename ValueT, size_t Capacity>
class CLOCKCache {
public:
    CLOCKCache() : hand_(0), size_(0) {
        buffer_ = std::make_unique<ClockSlot[]>(Capacity);
        for (size_t i = 0; i < Capacity; ++i) {
            buffer_[i].occupied = false;
            buffer_[i].reference_bit = false;
        }
    }
    
    bool get(const KeyT& key, ValueT& out_value) {
        auto it = lookup_.find(key);
        if (it == lookup_.end()) {
            return false;
        }
        size_t idx = it->second;
        buffer_[idx].reference_bit = true;
        out_value = buffer_[idx].value;
        return true;
    }
    
    bool put(const KeyT& key, const ValueT& value) {
        auto it = lookup_.find(key);
        if (it != lookup_.end()) {
            // Update existing
            size_t idx = it->second;
            buffer_[idx].value = value;
            buffer_[idx].reference_bit = true;
            return true;
        }
        
        if (size_ >= Capacity) {
            // Evict using CLOCK
            while (true) {
                if (!buffer_[hand_].occupied) {
                    hand_ = (hand_ + 1) % Capacity;
                    continue;
                }
                if (buffer_[hand_].reference_bit) {
                    buffer_[hand_].reference_bit = false;
                    hand_ = (hand_ + 1) % Capacity;
                } else {
                    lookup_.erase(buffer_[hand_].key);
                    buffer_[hand_].occupied = false;
                    size_--;
                    break;
                }
            }
        }
        
        // Insert at current hand position
        buffer_[hand_].key = key;
        buffer_[hand_].value = value;
        buffer_[hand_].reference_bit = true;
        buffer_[hand_].occupied = true;
        lookup_[key] = hand_;
        hand_ = (hand_ + 1) % Capacity;
        size_++;
        return true;
    }
    
    void clear() {
        for (size_t i = 0; i < Capacity; ++i) {
            buffer_[i].occupied = false;
            buffer_[i].reference_bit = false;
        }
        lookup_.clear();
        hand_ = 0;
        size_ = 0;
    }
    
private:
    struct ClockSlot {
        KeyT key;
        ValueT value;
        bool reference_bit;
        bool occupied;
    };
    
    std::unique_ptr<ClockSlot[]> buffer_;
    std::unordered_map<KeyT, size_t> lookup_;
    size_t hand_;
    size_t size_;
};

// Benchmark result structure
struct BenchmarkResult {
    const char* competitor_name;
    const char* workload_name;
    size_t capacity;
    double ns_per_op;
    double ratio_vs_lru;
    double ratio_vs_clock;
};

// Test configurations
constexpr size_t SIZE_1K = 1024;
constexpr size_t SIZE_16K = 16384;  // Changed from 10K to power of 2
constexpr size_t SIZE_128K = 131072;
constexpr size_t SIZE_512K = 524288;
constexpr size_t SIZE_1M = 1048576;

// Workload parameters
constexpr double ZIPF_SKEW = 0.99;  // 80/20 distribution

// Forward declarations
void print_results(const std::vector<BenchmarkResult>& results);
void print_header();
void print_row(const BenchmarkResult& result);

// Zipf sampler with precomputed CDF
class ZipfSampler {
public:
    ZipfSampler(uint64_t capacity, double skew)
        : capacity_(capacity), cdf_(capacity) {
        double sum = 0.0;
        for (uint64_t k = 1; k <= capacity; ++k) {
            sum += 1.0 / std::pow(static_cast<double>(k), skew);
        }
        double c = 0.0;
        for (uint64_t k = 0; k < capacity; ++k) {
            c += 1.0 / std::pow(static_cast<double>(k + 1), skew);
            cdf_[k] = c / sum;
        }
    }

    uint64_t sample(std::mt19937_64& rng) const {
        double u = std::uniform_real_distribution<double>(0, 1)(rng);
        // Binary search in CDF
        uint64_t lo = 0, hi = capacity_ - 1;
        while (lo < hi) {
            uint64_t mid = lo + (hi - lo) / 2;
            if (cdf_[mid] < u)
                lo = mid + 1;
            else
                hi = mid;
        }
        return lo;
    }

private:
    uint64_t capacity_;
    std::vector<double> cdf_;
};

// Workload generators
class WorkloadGenerator {
public:
    // Zipf distribution using precomputed CDF
    static uint64_t zipf_next(const ZipfSampler& sampler, std::mt19937_64& rng) {
        return sampler.sample(rng);
    }
    
    // Sequential access pattern
    static uint64_t sequential_next(uint64_t capacity, uint64_t& counter) {
        uint64_t result = counter % capacity;
        counter++;
        return result;
    }
    
    // Random uniform
    static uint64_t random_next(uint64_t capacity, std::mt19937_64& rng) {
        return std::uniform_int_distribution<uint64_t>(0, capacity - 1)(rng);
    }
};

// Benchmark runner
template<typename KeyT, typename ValueT, size_t Capacity>
class BenchmarkRunner {
public:
    static constexpr size_t NUM_OPERATIONS = 1'000'000; // 1M ops (reduced for speed)
    static constexpr double TIMEOUT_SECONDS = 5.0; // 5 second timeout per benchmark
    
    // Run all three workloads for a given competitor
    static BenchmarkResult run_lru(const char* workload_name) {
        LRUCache<KeyT, ValueT, Capacity> cache;
        double ns_per_op = 0.0;
        
        if (std::strcmp(workload_name, "Zipf") == 0) {
            ns_per_op = run_zipf(cache);
        } else if (std::strcmp(workload_name, "Sequential") == 0) {
            ns_per_op = run_sequential(cache);
        } else if (std::strcmp(workload_name, "Random") == 0) {
            ns_per_op = run_random(cache);
        }
        
        return {"LRU", workload_name, Capacity, ns_per_op, 1.0, 0.0};
    }
    
    static BenchmarkResult run_clock(const char* workload_name) {
        CLOCKCache<KeyT, ValueT, Capacity> cache;
        double ns_per_op = 0.0;
        
        if (std::strcmp(workload_name, "Zipf") == 0) {
            ns_per_op = run_zipf(cache);
        } else if (std::strcmp(workload_name, "Sequential") == 0) {
            ns_per_op = run_sequential(cache);
        } else if (std::strcmp(workload_name, "Random") == 0) {
            ns_per_op = run_random(cache);
        }
        
        return {"CLOCK", workload_name, Capacity, ns_per_op, 0.0, 1.0};
    }
    
    static BenchmarkResult run_nscache(const char* workload_name) {
        NSCache<KeyT, ValueT, Capacity> cache;
        double ns_per_op = 0.0;
        
        if (std::strcmp(workload_name, "Zipf") == 0) {
            ns_per_op = run_zipf(cache);
        } else if (std::strcmp(workload_name, "Sequential") == 0) {
            ns_per_op = run_sequential(cache);
        } else if (std::strcmp(workload_name, "Random") == 0) {
            ns_per_op = run_random(cache);
        }
        
        return {"NSCache", workload_name, Capacity, ns_per_op, 0.0, 0.0};
    }

    static BenchmarkResult run_nscache_noboost(const char* workload_name) {
        std::cout << "run_nscache_noboost: not implemented" << std::endl;
        return {"NSCache-NoBoost", workload_name, Capacity, 0.0, 0.0, 0.0};
    }
    
private:
    // Individual workload runners
    static double run_zipf(auto& cache) {
        ZipfSampler sampler(Capacity, ZIPF_SKEW);
        std::mt19937_64 rng(42);
        auto start = std::chrono::high_resolution_clock::now();
        
        for (size_t i = 0; i < NUM_OPERATIONS; ++i) {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed_sec = std::chrono::duration<double>(now - start).count();
            if (elapsed_sec > TIMEOUT_SECONDS) {
                // Return early with partial results
                double duration_ns = std::chrono::duration<double, std::nano>(now - start).count();
                return duration_ns / (i + 1);
            }
            
            uint64_t key = WorkloadGenerator::zipf_next(sampler, rng);
            ValueT value = static_cast<ValueT>(key);
            ValueT out;
            if (!cache.get(key, out)) {
                cache.put(key, value);
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double duration_ns = std::chrono::duration<double, std::nano>(end - start).count();
        return duration_ns / NUM_OPERATIONS;
    }
    
    static double run_sequential(auto& cache) {
        uint64_t counter = 0;
        auto start = std::chrono::high_resolution_clock::now();
        
        for (size_t i = 0; i < NUM_OPERATIONS; ++i) {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed_sec = std::chrono::duration<double>(now - start).count();
            if (elapsed_sec > TIMEOUT_SECONDS) {
                double duration_ns = std::chrono::duration<double, std::nano>(now - start).count();
                return duration_ns / (i + 1);
            }
            
            uint64_t key = WorkloadGenerator::sequential_next(Capacity, counter);
            ValueT value = static_cast<ValueT>(key);
            ValueT out;
            if (!cache.get(key, out)) {
                cache.put(key, value);
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double duration_ns = std::chrono::duration<double, std::nano>(end - start).count();
        return duration_ns / NUM_OPERATIONS;
    }
    
    static double run_random(auto& cache) {
        std::mt19937_64 rng(42);
        auto start = std::chrono::high_resolution_clock::now();
        
        for (size_t i = 0; i < NUM_OPERATIONS; ++i) {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed_sec = std::chrono::duration<double>(now - start).count();
            if (elapsed_sec > TIMEOUT_SECONDS) {
                double duration_ns = std::chrono::duration<double, std::nano>(now - start).count();
                return duration_ns / (i + 1);
            }
            
            uint64_t key = WorkloadGenerator::random_next(Capacity, rng);
            ValueT value = static_cast<ValueT>(key);
            ValueT out;
            if (!cache.get(key, out)) {
                cache.put(key, value);
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double duration_ns = std::chrono::duration<double, std::nano>(end - start).count();
        return duration_ns / NUM_OPERATIONS;
    }
};

// Main benchmark function
void run_all_benchmarks() {
    std::vector<BenchmarkResult> results;
    
    const char* workloads[] = {"Zipf", "Sequential", "Random"};
    
    // 128K only
    for (const char* workload : workloads) {
        auto lru = BenchmarkRunner<uint64_t, uint64_t, SIZE_128K>::run_lru(workload);
        auto clock = BenchmarkRunner<uint64_t, uint64_t, SIZE_128K>::run_clock(workload);
        auto nscache = BenchmarkRunner<uint64_t, uint64_t, SIZE_128K>::run_nscache(workload);
        auto nscache_noboost = BenchmarkRunner<uint64_t, uint64_t, SIZE_128K>::run_nscache_noboost(workload);
        
        nscache.ratio_vs_lru = nscache.ns_per_op / lru.ns_per_op;
        nscache.ratio_vs_clock = nscache.ns_per_op / clock.ns_per_op;
        nscache_noboost.ratio_vs_lru = nscache_noboost.ns_per_op / lru.ns_per_op;
        nscache_noboost.ratio_vs_clock = nscache_noboost.ns_per_op / clock.ns_per_op;
        clock.ratio_vs_lru = clock.ns_per_op / lru.ns_per_op;
        
        results.push_back(lru);
        results.push_back(clock);
        results.push_back(nscache);
        results.push_back(nscache_noboost);
    }
    
    print_results(results);
}

// Output formatting
void print_results(const std::vector<BenchmarkResult>& results) {
    print_header();
    for (const auto& result : results) {
        print_row(result);
    }
}

void print_header() {
    std::cout << std::left << std::setw(16) << "Competitor"
              << std::setw(12) << "Workload"
              << std::setw(10) << "Capacity"
              << std::setw(15) << "ns/op"
              << std::setw(15) << "vs LRU"
              << std::setw(15) << "vs CLOCK"
              << std::endl;
    std::cout << std::string(83, '-') << std::endl;
}

void print_row(const BenchmarkResult& result) {
    std::cout << std::left << std::setw(16) << result.competitor_name
              << std::setw(12) << result.workload_name
              << std::setw(10) << result.capacity
              << std::setw(15) << std::fixed << std::setprecision(2) << result.ns_per_op
              << std::setw(15) << std::fixed << std::setprecision(3) << result.ratio_vs_lru
              << std::setw(15) << std::fixed << std::setprecision(3) << result.ratio_vs_clock
              << std::endl;
}

int main() {
    // Full benchmark: all sizes with all workloads
    std::vector<BenchmarkResult> results;
    
    const char* workloads[] = {"Zipf", "Sequential", "Random"};
    
    // 1K
    for (const char* workload : workloads) {
        auto lru = BenchmarkRunner<uint64_t, uint64_t, SIZE_1K>::run_lru(workload);
        auto clock = BenchmarkRunner<uint64_t, uint64_t, SIZE_1K>::run_clock(workload);
        auto nscache = BenchmarkRunner<uint64_t, uint64_t, SIZE_1K>::run_nscache(workload);
        
        nscache.ratio_vs_lru = nscache.ns_per_op / lru.ns_per_op;
        nscache.ratio_vs_clock = nscache.ns_per_op / clock.ns_per_op;
        clock.ratio_vs_lru = clock.ns_per_op / lru.ns_per_op;
        
        results.push_back(lru);
        results.push_back(clock);
        results.push_back(nscache);
    }
    
    // 16K
    for (const char* workload : workloads) {
        auto lru = BenchmarkRunner<uint64_t, uint64_t, SIZE_16K>::run_lru(workload);
        auto clock = BenchmarkRunner<uint64_t, uint64_t, SIZE_16K>::run_clock(workload);
        auto nscache = BenchmarkRunner<uint64_t, uint64_t, SIZE_16K>::run_nscache(workload);
        
        nscache.ratio_vs_lru = nscache.ns_per_op / lru.ns_per_op;
        nscache.ratio_vs_clock = nscache.ns_per_op / clock.ns_per_op;
        clock.ratio_vs_lru = clock.ns_per_op / lru.ns_per_op;
        
        results.push_back(lru);
        results.push_back(clock);
        results.push_back(nscache);
    }
    
    // 128K
    for (const char* workload : workloads) {
        auto lru = BenchmarkRunner<uint64_t, uint64_t, SIZE_128K>::run_lru(workload);
        auto clock = BenchmarkRunner<uint64_t, uint64_t, SIZE_128K>::run_clock(workload);
        auto nscache = BenchmarkRunner<uint64_t, uint64_t, SIZE_128K>::run_nscache(workload);
        
        nscache.ratio_vs_lru = nscache.ns_per_op / lru.ns_per_op;
        nscache.ratio_vs_clock = nscache.ns_per_op / clock.ns_per_op;
        clock.ratio_vs_lru = clock.ns_per_op / lru.ns_per_op;
        
        results.push_back(lru);
        results.push_back(clock);
        results.push_back(nscache);
    }
    
    // 512K
    for (const char* workload : workloads) {
        auto lru = BenchmarkRunner<uint64_t, uint64_t, SIZE_512K>::run_lru(workload);
        auto clock = BenchmarkRunner<uint64_t, uint64_t, SIZE_512K>::run_clock(workload);
        auto nscache = BenchmarkRunner<uint64_t, uint64_t, SIZE_512K>::run_nscache(workload);
        
        nscache.ratio_vs_lru = nscache.ns_per_op / lru.ns_per_op;
        nscache.ratio_vs_clock = nscache.ns_per_op / clock.ns_per_op;
        clock.ratio_vs_lru = clock.ns_per_op / lru.ns_per_op;
        
        results.push_back(lru);
        results.push_back(clock);
        results.push_back(nscache);
    }
    
    // 1M
    for (const char* workload : workloads) {
        auto lru = BenchmarkRunner<uint64_t, uint64_t, SIZE_1M>::run_lru(workload);
        auto clock = BenchmarkRunner<uint64_t, uint64_t, SIZE_1M>::run_clock(workload);
        auto nscache = BenchmarkRunner<uint64_t, uint64_t, SIZE_1M>::run_nscache(workload);
        
        nscache.ratio_vs_lru = nscache.ns_per_op / lru.ns_per_op;
        nscache.ratio_vs_clock = nscache.ns_per_op / clock.ns_per_op;
        clock.ratio_vs_lru = clock.ns_per_op / lru.ns_per_op;
        
        results.push_back(lru);
        results.push_back(clock);
        results.push_back(nscache);
    }
    
    print_results(results);
    
    return 0;
}

// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
#include <chrono>
#include <random>
#include "nsindex.hpp"
#include "pgm/pgm_index.hpp"
#include "alex.h"
#include <cstring>

// Cold cache flush - clear CPU caches
void flush_cache() {
    const size_t cache_size = 32 * 1024 * 1024; // 32MB
    std::vector<char> flusher(cache_size);
    std::fill(flusher.begin(), flusher.end(), 0);
    // Prevent optimization
    volatile char sum = 0;
    for (char c : flusher) sum += c;
    (void)sum;
}

const int NUM_KEYS = 10000000;
const int NUM_LOOKUPS = 1000000;
const int NUM_INSERTS = 1000000;
const int NUM_RANGE_QUERIES = 1000000;

// Generate exponential distribution dataset (many small, few large)
std::vector<int64_t> generate_exponential_dataset(int n) {
    std::vector<int64_t> data;
    data.reserve(n);
    std::mt19937_64 rng(42);
    std::exponential_distribution<double> exp_dist(1.0);
    for (int i = 0; i < n; i++) {
        double val = exp_dist(rng);
        // Scale to reasonable range and ensure uniqueness
        int64_t key = static_cast<int64_t>(val * 1000000) + i;
        data.push_back(key);
    }
    std::sort(data.begin(), data.end());
    return data;
}

// Generate clustered distribution dataset (500 clusters of 20K consecutive keys)
std::vector<int64_t> generate_clustered_dataset(int n) {
    std::vector<int64_t> data;
    data.reserve(n);
    std::mt19937_64 rng(42);
    const int num_clusters = 500;
    const int cluster_size = n / num_clusters;
    
    for (int c = 0; c < num_clusters; c++) {
        // Random gap between clusters
        int64_t gap = rng() % 10000;
        int64_t cluster_start = c * (cluster_size + gap);
        for (int i = 0; i < cluster_size; i++) {
            data.push_back(cluster_start + i);
        }
    }
    return data;
}

// Helper function to run range benchmark for a given dataset
void run_range_benchmark(const std::vector<int64_t>& dataset, const std::string& dataset_name) {
    std::cout << "\n=== RANGE BENCHMARK: " << dataset_name << " ===" << std::endl;
    std::cout << "ALEX baseline: ~30ns/query for 100+ results (learned index)" << std::endl;
    
    NSIndex<int64_t> idx_range(dataset.data(), dataset.size());
    static int64_t range_out[10000];
    
    // Generate range query keys
    std::vector<int64_t> range_lo_keys;
    range_lo_keys.reserve(NUM_RANGE_QUERIES);
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<size_t> dist(0, dataset.size() - 1);
    for (int i = 0; i < NUM_RANGE_QUERIES; i++) {
        range_lo_keys.push_back(dataset[dist(rng)]);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    
    // 5 results - NSIndex
    {
        std::vector<int64_t> range_hi_keys;
        range_hi_keys.reserve(NUM_RANGE_QUERIES);
        for (int i = 0; i < NUM_RANGE_QUERIES; i++) {
            range_hi_keys.push_back(range_lo_keys[i] + 5000);
        }
        flush_cache();
        volatile size_t total_found = 0;
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_RANGE_QUERIES; i++) {
            int count = idx_range.range(range_lo_keys[i], range_hi_keys[i], range_out, 10000);
            total_found += count;
        }
        end = std::chrono::high_resolution_clock::now();
        double range_ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>
            (end - start).count() / NUM_RANGE_QUERIES;
        std::cout << "NSIndex range (5 results): " << range_ns 
                  << " ns/query (avg " << (total_found/NUM_RANGE_QUERIES) << " results)" << std::endl;
    }
    
    // 5 results - BitBlock
    {
        std::vector<int64_t> range_hi_keys;
        range_hi_keys.reserve(NUM_RANGE_QUERIES);
        for (int i = 0; i < NUM_RANGE_QUERIES; i++) {
            range_hi_keys.push_back(range_lo_keys[i] + 5000);
        }
        flush_cache();
        volatile size_t total_found = 0;
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_RANGE_QUERIES; i++) {
            int count = idx_range.find_range_bitblock(range_lo_keys[i], range_hi_keys[i], range_out, 10000);
            total_found += count;
        }
        end = std::chrono::high_resolution_clock::now();
        double range_ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>
            (end - start).count() / NUM_RANGE_QUERIES;
        std::cout << "BitBlock range (5 results): " << range_ns 
                  << " ns/query (avg " << (total_found/NUM_RANGE_QUERIES) << " results)" << std::endl;
    }
    
    // 5 results - std::lower_bound
    {
        std::vector<int64_t> range_hi_keys;
        range_hi_keys.reserve(NUM_RANGE_QUERIES);
        for (int i = 0; i < NUM_RANGE_QUERIES; i++) {
            range_hi_keys.push_back(range_lo_keys[i] + 5000);
        }
        flush_cache();
        volatile size_t total_found = 0;
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_RANGE_QUERIES; i++) {
            auto lo_it = std::lower_bound(dataset.begin(), dataset.end(), range_lo_keys[i]);
            auto hi_it = std::upper_bound(dataset.begin(), dataset.end(), range_hi_keys[i]);
            total_found += (hi_it - lo_it);
        }
        end = std::chrono::high_resolution_clock::now();
        double range_ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>
            (end - start).count() / NUM_RANGE_QUERIES;
        std::cout << "std::lower_bound range (5 results): " << range_ns 
                  << " ns/query (avg " << (total_found/NUM_RANGE_QUERIES) << " results)" << std::endl;
    }
    
    // 50 results - NSIndex
    {
        std::vector<int64_t> range_hi_keys;
        range_hi_keys.reserve(NUM_RANGE_QUERIES);
        for (int i = 0; i < NUM_RANGE_QUERIES; i++) {
            range_hi_keys.push_back(range_lo_keys[i] + 50000);
        }
        flush_cache();
        volatile size_t total_found = 0;
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_RANGE_QUERIES; i++) {
            int count = idx_range.range(range_lo_keys[i], range_hi_keys[i], range_out, 10000);
            total_found += count;
        }
        end = std::chrono::high_resolution_clock::now();
        double range_ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>
            (end - start).count() / NUM_RANGE_QUERIES;
        std::cout << "NSIndex range (50 results): " << range_ns 
                  << " ns/query (avg " << (total_found/NUM_RANGE_QUERIES) << " results)" << std::endl;
    }
    
    // 50 results - BitBlock
    {
        std::vector<int64_t> range_hi_keys;
        range_hi_keys.reserve(NUM_RANGE_QUERIES);
        for (int i = 0; i < NUM_RANGE_QUERIES; i++) {
            range_hi_keys.push_back(range_lo_keys[i] + 50000);
        }
        flush_cache();
        volatile size_t total_found = 0;
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_RANGE_QUERIES; i++) {
            int count = idx_range.find_range_bitblock(range_lo_keys[i], range_hi_keys[i], range_out, 10000);
            total_found += count;
        }
        end = std::chrono::high_resolution_clock::now();
        double range_ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>
            (end - start).count() / NUM_RANGE_QUERIES;
        std::cout << "BitBlock range (50 results): " << range_ns 
                  << " ns/query (avg " << (total_found/NUM_RANGE_QUERIES) << " results)" << std::endl;
    }
    
    // 100 results - NSIndex
    {
        std::vector<int64_t> range_hi_keys;
        range_hi_keys.reserve(NUM_RANGE_QUERIES);
        for (int i = 0; i < NUM_RANGE_QUERIES; i++) {
            range_hi_keys.push_back(range_lo_keys[i] + 100000);
        }
        flush_cache();
        volatile size_t total_found = 0;
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_RANGE_QUERIES; i++) {
            int count = idx_range.range(range_lo_keys[i], range_hi_keys[i], range_out, 10000);
            total_found += count;
        }
        end = std::chrono::high_resolution_clock::now();
        double range_ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>
            (end - start).count() / NUM_RANGE_QUERIES;
        std::cout << "NSIndex range (100 results): " << range_ns 
                  << " ns/query (avg " << (total_found/NUM_RANGE_QUERIES) << " results)" << std::endl;
    }
    
    // 100 results - BitBlock
    {
        std::vector<int64_t> range_hi_keys;
        range_hi_keys.reserve(NUM_RANGE_QUERIES);
        for (int i = 0; i < NUM_RANGE_QUERIES; i++) {
            range_hi_keys.push_back(range_lo_keys[i] + 100000);
        }
        flush_cache();
        volatile size_t total_found = 0;
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_RANGE_QUERIES; i++) {
            int count = idx_range.find_range_bitblock(range_lo_keys[i], range_hi_keys[i], range_out, 10000);
            total_found += count;
        }
        end = std::chrono::high_resolution_clock::now();
        double range_ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>
            (end - start).count() / NUM_RANGE_QUERIES;
        std::cout << "BitBlock range (100 results): " << range_ns 
                  << " ns/query (avg " << (total_found/NUM_RANGE_QUERIES) << " results)" << std::endl;
    }
    
    // 1000 results - NSIndex
    {
        std::vector<int64_t> range_hi_keys;
        range_hi_keys.reserve(NUM_RANGE_QUERIES);
        for (int i = 0; i < NUM_RANGE_QUERIES; i++) {
            range_hi_keys.push_back(range_lo_keys[i] + 1000000);
        }
        flush_cache();
        volatile size_t total_found = 0;
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_RANGE_QUERIES; i++) {
            int count = idx_range.range(range_lo_keys[i], range_hi_keys[i], range_out, 10000);
            total_found += count;
        }
        end = std::chrono::high_resolution_clock::now();
        double range_ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>
            (end - start).count() / NUM_RANGE_QUERIES;
        std::cout << "NSIndex range (1000 results): " << range_ns 
                  << " ns/query (avg " << (total_found/NUM_RANGE_QUERIES) << " results)" << std::endl;
    }
    
    // 1000 results - BitBlock
    {
        std::vector<int64_t> range_hi_keys;
        range_hi_keys.reserve(NUM_RANGE_QUERIES);
        for (int i = 0; i < NUM_RANGE_QUERIES; i++) {
            range_hi_keys.push_back(range_lo_keys[i] + 1000000);
        }
        flush_cache();
        volatile size_t total_found = 0;
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_RANGE_QUERIES; i++) {
            int count = idx_range.find_range_bitblock(range_lo_keys[i], range_hi_keys[i], range_out, 10000);
            total_found += count;
        }
        end = std::chrono::high_resolution_clock::now();
        double range_ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>
            (end - start).count() / NUM_RANGE_QUERIES;
        std::cout << "BitBlock range (1000 results): " << range_ns 
                  << " ns/query (avg " << (total_found/NUM_RANGE_QUERIES) << " results)" << std::endl;
    }
    
    // ALEX range benchmark
    std::cout << "\n--- ALEX Range Benchmark ---" << std::endl;
    {
        std::vector<std::pair<int64_t, int64_t>> alex_values;
        alex_values.reserve(dataset.size());
        for (size_t i = 0; i < dataset.size(); i++) {
            alex_values.push_back({dataset[i], (int64_t)i});
        }
        alex::Alex<int64_t, int64_t> alex_index;
        alex_index.bulk_load(alex_values.data(), (int)dataset.size());
        
        struct RangeTest { const char* name; int64_t span; };
        RangeTest tests[] = {
            {"5 results", 5000},
            {"50 results", 50000},
            {"100 results", 100000},
            {"1000 results", 1000000}
        };
        
        int64_t max_key = dataset.back();
        for (const auto& test : tests) {
            std::vector<int64_t> range_hi_keys;
            range_hi_keys.reserve(NUM_RANGE_QUERIES);
            for (int i = 0; i < NUM_RANGE_QUERIES; i++) {
                int64_t hi = range_lo_keys[i] + test.span;
                if (hi > max_key) hi = max_key;
                if (hi < range_lo_keys[i]) hi = range_lo_keys[i];
                range_hi_keys.push_back(hi);
            }
            flush_cache();
            volatile size_t total_found = 0;
            start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < NUM_RANGE_QUERIES; i++) {
                auto it = alex_index.lower_bound(range_lo_keys[i]);
                auto end_it = alex_index.upper_bound(range_hi_keys[i]);
                int cnt = 0;
                while (it != end_it && cnt < 10000) {
                    cnt++;
                    ++it;
                }
                total_found += cnt;
            }
            end = std::chrono::high_resolution_clock::now();
            double alex_ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>
                (end - start).count() / NUM_RANGE_QUERIES;
            std::cout << "ALEX range (" << test.name << "): " << alex_ns 
                      << " ns/query (avg " << (total_found/NUM_RANGE_QUERIES) << " results)" << std::endl;
        }
    }
}

int main() {
#ifdef __AVX2__
    std::cout << "AVX2: YES" << std::endl;
#else
    std::cout << "AVX2: NO (scalar)" << std::endl;
#endif
#ifdef __AVX512F__
    std::cout << "AVX-512: YES" << std::endl;
#else
    std::cout << "AVX-512: NO (scalar)" << std::endl;
#endif
    std::cout << "=== NSIndex Benchmark ===" << std::endl;
    std::cout << "Keys: " << NUM_KEYS << ", Lookups: " << NUM_LOOKUPS << std::endl;
    
    // Build flat sorted array of int64 timestamp keys
    std::vector<int64_t> flat_array;
    flat_array.reserve(NUM_KEYS);
    for (int64_t i = 0; i < NUM_KEYS; i++) {
        flat_array.push_back(i * 1000); // Sequential timestamps
    }
    
    // Build std::map with same keys
    std::map<int64_t, int64_t> std_map;
    for (int64_t i = 0; i < NUM_KEYS; i++) {
        std_map[i * 1000] = i;
    }
    
    // Generate random lookup keys
    std::vector<int64_t> lookup_keys;
    lookup_keys.reserve(NUM_LOOKUPS);
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<int64_t> dist(0, (int64_t)(NUM_KEYS - 1) * 1000);
    for (int i = 0; i < NUM_LOOKUPS; i++) {
        lookup_keys.push_back(dist(rng));
    }
    
    // Benchmark std::lower_bound on flat array
    flush_cache();
    std::cout << "\nBenchmarking std::lower_bound..." << std::endl;
    volatile int64_t sum = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_LOOKUPS; i++) {
        auto it = std::lower_bound(flat_array.begin(), flat_array.end(), lookup_keys[i]);
        if (it != flat_array.end() && *it == lookup_keys[i]) {
            sum += *it;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto lower_bound_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double lower_bound_ns_per_lookup = (double)lower_bound_duration / NUM_LOOKUPS;
    
    // Benchmark std::map lookup
    flush_cache();
    std::cout << "Benchmarking std::map..." << std::endl;
    sum = 0;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_LOOKUPS; i++) {
        auto it = std_map.find(lookup_keys[i]);
        if (it != std_map.end()) {
            sum += it->second;
        }
    }
    end = std::chrono::high_resolution_clock::now();
    auto map_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double map_ns_per_lookup = (double)map_duration / NUM_LOOKUPS;
    
    // Benchmark NSIndex lookup
    flush_cache();
    std::cout << "Benchmarking NSIndex..." << std::endl;
    NSIndex<int64_t> idx(flat_array.data(), flat_array.size());
    // No flush needed - delta_buffer_ is already empty on fresh index
    sum = 0;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_LOOKUPS; i++) {
        auto it = idx.find(flat_array.data(), flat_array.size(), lookup_keys[i]);
        if (it != flat_array.data() + flat_array.size() && *it == lookup_keys[i]) {
            sum += *it;
        }
    }
    end = std::chrono::high_resolution_clock::now();
    auto nsindex_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double nsindex_ns = (double)nsindex_duration / NUM_LOOKUPS;
    
    // Generate guaranteed hit lookup keys
    std::vector<int64_t> hit_lookup_keys;
    hit_lookup_keys.reserve(NUM_LOOKUPS);
    std::uniform_int_distribution<int> index_dist(0, NUM_KEYS - 1);
    for (int i = 0; i < NUM_LOOKUPS; i++) {
        hit_lookup_keys.push_back(flat_array[index_dist(rng)]);
    }
    
    // Benchmark NSIndex (hits only)
    flush_cache();
    std::cout << "Benchmarking NSIndex (hits only)..." << std::endl;
    // No flush needed - delta_buffer_ is already empty
    sum = 0;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_LOOKUPS; i++) {
        auto it = idx.find(flat_array.data(), flat_array.size(), hit_lookup_keys[i]);
        if (it != flat_array.data() + flat_array.size() && *it == hit_lookup_keys[i]) {
            sum += *it;
        }
    }
    end = std::chrono::high_resolution_clock::now();
    auto nsindex_hits_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double nsindex_hits_ns = (double)nsindex_hits_duration / NUM_LOOKUPS;
    
    // Build clustered array with random jitter
    std::vector<int64_t> flat_array_clustered;
    flat_array_clustered.reserve(NUM_KEYS);
    std::uniform_int_distribution<int> jitter_dist(-50, 49);
    for (int64_t i = 0; i < NUM_KEYS; i++) {
        flat_array_clustered.push_back(i * 1000 + jitter_dist(rng));
    }
    std::sort(flat_array_clustered.begin(), flat_array_clustered.end());
    
    // Generate guaranteed hit lookup keys for clustered array
    std::vector<int64_t> clustered_hit_keys;
    clustered_hit_keys.reserve(NUM_LOOKUPS);
    for (int i = 0; i < NUM_LOOKUPS; i++) {
        clustered_hit_keys.push_back(flat_array_clustered[index_dist(rng)]);
    }
    
    // Benchmark NSIndex on clustered keys
    flush_cache();
    std::cout << "Benchmarking NSIndex (clustered keys)..." << std::endl;
    NSIndex<int64_t> idx_clustered(flat_array_clustered.data(), flat_array_clustered.size());
    // No flush needed - delta_buffer_ is already empty
    sum = 0;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_LOOKUPS; i++) {
        auto it = idx_clustered.find(flat_array_clustered.data(), flat_array_clustered.size(), clustered_hit_keys[i]);
        if (it != flat_array_clustered.data() + flat_array_clustered.size() && *it == clustered_hit_keys[i]) {
            sum += *it;
        }
    }
    end = std::chrono::high_resolution_clock::now();
    auto nsindex_clustered_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double nsindex_clustered_ns = (double)nsindex_clustered_duration / NUM_LOOKUPS;
    
    // Benchmark std::lower_bound on clustered keys for comparison
    flush_cache();
    std::cout << "Benchmarking std::lower_bound (clustered keys)..." << std::endl;
    sum = 0;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_LOOKUPS; i++) {
        auto it = std::lower_bound(flat_array_clustered.begin(), flat_array_clustered.end(), clustered_hit_keys[i]);
        if (it != flat_array_clustered.end() && *it == clustered_hit_keys[i]) {
            sum += *it;
        }
    }
    end = std::chrono::high_resolution_clock::now();
    auto lower_bound_clustered_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double lower_bound_clustered_ns = (double)lower_bound_clustered_duration / NUM_LOOKUPS;
    
    // Benchmark PGM-Index
    flush_cache();
    std::cout << "Benchmarking PGM-Index..." << std::endl;
    pgm::PGMIndex<int64_t, 64> pgm_index(flat_array.begin(), flat_array.end());
    sum = 0;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_LOOKUPS; i++) {
        auto range = pgm_index.search(lookup_keys[i]);
        auto it = std::lower_bound(
            flat_array.begin() + range.lo,
            flat_array.begin() + range.hi,
            lookup_keys[i]);
        if (it != flat_array.end() && *it == lookup_keys[i]) {
            sum += *it;
        }
    }
    end = std::chrono::high_resolution_clock::now();
    auto pgm_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double pgm_ns = (double)pgm_duration / NUM_LOOKUPS;
    
    // Range query benchmark (1M queries on 10M keys)
    // ALEX baseline: ~30ns at 100+ results (from BLI 2025 paper)
    NSIndex<int64_t> idx_range(flat_array.data(), flat_array.size());
    static int64_t range_out[10000]; // Pre-allocated output array
    
    // Test different result sizes
    std::vector<int64_t> range_lo_keys;
    range_lo_keys.reserve(NUM_RANGE_QUERIES);
    for (int i = 0; i < NUM_RANGE_QUERIES; i++) {
        range_lo_keys.push_back(hit_lookup_keys[i % NUM_LOOKUPS]);
    }
    
    // 5 results
    {
        std::vector<int64_t> range_hi_keys;
        range_hi_keys.reserve(NUM_RANGE_QUERIES);
        for (int i = 0; i < NUM_RANGE_QUERIES; i++) {
            range_hi_keys.push_back(range_lo_keys[i] + 5000);
        }
        flush_cache();
        volatile size_t total_found = 0;
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_RANGE_QUERIES; i++) {
            int count = idx_range.range(range_lo_keys[i], range_hi_keys[i], range_out, 10000);
            total_found += count;
        }
        end = std::chrono::high_resolution_clock::now();
        double range_ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>
            (end - start).count() / NUM_RANGE_QUERIES;
        std::cout << "NSIndex range (5 results): " << range_ns 
                  << " ns/query (avg " << (total_found/NUM_RANGE_QUERIES) << " results)" << std::endl;
        
        // Timing breakdown removed from production code
        // uint64_t pred_cycles, setup_cycles, avx_cycles, delta_cycles, call_count;
        // idx_range.get_timing_breakdown(pred_cycles, setup_cycles, avx_cycles, delta_cycles, call_count);
        // double ghz = 3.5;
        // printf("Timing breakdown (avg per query, %.1f GHz):\n", ghz);
        // printf("  Prediction:    %.2f cycles (%.2f ns)\n", (double)pred_cycles/call_count, (double)pred_cycles/call_count/ghz);
        // printf("  Setup:         %.2f cycles (%.2f ns)\n", (double)setup_cycles/call_count, (double)setup_cycles/call_count/ghz);
        // printf("  AVX scan:      %.2f cycles (%.2f ns)\n", (double)avx_cycles/call_count, (double)avx_cycles/call_count/ghz);
        // printf("  Delta scan:    %.2f cycles (%.2f ns)\n", (double)delta_cycles/call_count, (double)delta_cycles/call_count/ghz);
        // printf("  Total:         %.2f cycles (%.2f ns)\n", (double)(pred_cycles+setup_cycles+avx_cycles+delta_cycles)/call_count, (double)(pred_cycles+setup_cycles+avx_cycles+delta_cycles)/call_count/ghz);
    }
    
    // 50 results
    {
        std::vector<int64_t> range_hi_keys;
        range_hi_keys.reserve(NUM_RANGE_QUERIES);
        for (int i = 0; i < NUM_RANGE_QUERIES; i++) {
            range_hi_keys.push_back(range_lo_keys[i] + 50000);
        }
        flush_cache();
        volatile size_t total_found = 0;
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_RANGE_QUERIES; i++) {
            int count = idx_range.range(range_lo_keys[i], range_hi_keys[i], range_out, 10000);
            total_found += count;
        }
        end = std::chrono::high_resolution_clock::now();
        double range_ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>
            (end - start).count() / NUM_RANGE_QUERIES;
        std::cout << "NSIndex range (50 results): " << range_ns 
                  << " ns/query (avg " << (total_found/NUM_RANGE_QUERIES) << " results)" << std::endl;
    }
    
    // 100 results
    {
        std::vector<int64_t> range_hi_keys;
        range_hi_keys.reserve(NUM_RANGE_QUERIES);
        for (int i = 0; i < NUM_RANGE_QUERIES; i++) {
            range_hi_keys.push_back(range_lo_keys[i] + 100000);
        }
        flush_cache();
        volatile size_t total_found = 0;
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_RANGE_QUERIES; i++) {
            int count = idx_range.range(range_lo_keys[i], range_hi_keys[i], range_out, 10000);
            total_found += count;
        }
        end = std::chrono::high_resolution_clock::now();
        double range_ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>
            (end - start).count() / NUM_RANGE_QUERIES;
        std::cout << "NSIndex range (100 results): " << range_ns 
                  << " ns/query (avg " << (total_found/NUM_RANGE_QUERIES) << " results)" << std::endl;
    }
    
    // 1000 results
    {
        std::vector<int64_t> range_hi_keys;
        range_hi_keys.reserve(NUM_RANGE_QUERIES);
        for (int i = 0; i < NUM_RANGE_QUERIES; i++) {
            range_hi_keys.push_back(range_lo_keys[i] + 1000000);
        }
        flush_cache();
        volatile size_t total_found = 0;
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_RANGE_QUERIES; i++) {
            int count = idx_range.range(range_lo_keys[i], range_hi_keys[i], range_out, 10000);
            total_found += count;
        }
        end = std::chrono::high_resolution_clock::now();
        double range_ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>
            (end - start).count() / NUM_RANGE_QUERIES;
        std::cout << "NSIndex range (1000 results): " << range_ns 
                  << " ns/query (avg " << (total_found/NUM_RANGE_QUERIES) << " results)" << std::endl;
        
        // AVX stats removed from production code
        // uint64_t avx_calls, scalar_calls;
        // idx_range.get_avx_stats(avx_calls, scalar_calls);
        // printf("AVX calls: %lu, Scalar calls: %lu\n", avx_calls, scalar_calls);
        // printf("Key type: int64 (sizeof=%zu bytes)\n", sizeof(int64_t));
    }
    
    // Fast range benchmark (pure scan cost, no prediction overhead)
    std::cout << "\n=== FAST RANGE BENCHMARK (pure scan) ===" << std::endl;
    
    // 5 results
    {
        std::vector<int64_t> range_hi_keys;
        range_hi_keys.reserve(NUM_RANGE_QUERIES);
        for (int i = 0; i < NUM_RANGE_QUERIES; i++) {
            range_hi_keys.push_back(range_lo_keys[i] + 5000);
        }
        flush_cache();
        volatile size_t total_found = 0;
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_RANGE_QUERIES; i++) {
            // Find start position first (not timed)
            auto start_pos = idx_range.predict(range_lo_keys[i]);
            // Time only the fast_range scan
            int count = idx_range.fast_range(start_pos, range_hi_keys[i], range_out, 10000);
            total_found += count;
        }
        end = std::chrono::high_resolution_clock::now();
        double fast_range_ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>
            (end - start).count() / NUM_RANGE_QUERIES;
        std::cout << "fast_range (5 results): " << fast_range_ns 
                  << " ns/query (avg " << (total_found/NUM_RANGE_QUERIES) << " results)" << std::endl;
    }
    
    // 50 results
    {
        std::vector<int64_t> range_hi_keys;
        range_hi_keys.reserve(NUM_RANGE_QUERIES);
        for (int i = 0; i < NUM_RANGE_QUERIES; i++) {
            range_hi_keys.push_back(range_lo_keys[i] + 50000);
        }
        flush_cache();
        volatile size_t total_found = 0;
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_RANGE_QUERIES; i++) {
            auto start_pos = idx_range.predict(range_lo_keys[i]);
            int count = idx_range.fast_range(start_pos, range_hi_keys[i], range_out, 10000);
            total_found += count;
        }
        end = std::chrono::high_resolution_clock::now();
        double fast_range_ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>
            (end - start).count() / NUM_RANGE_QUERIES;
        std::cout << "fast_range (50 results): " << fast_range_ns 
                  << " ns/query (avg " << (total_found/NUM_RANGE_QUERIES) << " results)" << std::endl;
    }
    
    // 100 results
    {
        std::vector<int64_t> range_hi_keys;
        range_hi_keys.reserve(NUM_RANGE_QUERIES);
        for (int i = 0; i < NUM_RANGE_QUERIES; i++) {
            range_hi_keys.push_back(range_lo_keys[i] + 100000);
        }
        flush_cache();
        volatile size_t total_found = 0;
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_RANGE_QUERIES; i++) {
            auto start_pos = idx_range.predict(range_lo_keys[i]);
            int count = idx_range.fast_range(start_pos, range_hi_keys[i], range_out, 10000);
            total_found += count;
        }
        end = std::chrono::high_resolution_clock::now();
        double fast_range_ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>
            (end - start).count() / NUM_RANGE_QUERIES;
        std::cout << "fast_range (100 results): " << fast_range_ns 
                  << " ns/query (avg " << (total_found/NUM_RANGE_QUERIES) << " results)" << std::endl;
    }
    
    // 1000 results
    {
        std::vector<int64_t> range_hi_keys;
        range_hi_keys.reserve(NUM_RANGE_QUERIES);
        for (int i = 0; i < NUM_RANGE_QUERIES; i++) {
            range_hi_keys.push_back(range_lo_keys[i] + 1000000);
        }
        flush_cache();
        volatile size_t total_found = 0;
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_RANGE_QUERIES; i++) {
            auto start_pos = idx_range.predict(range_lo_keys[i]);
            int count = idx_range.fast_range(start_pos, range_hi_keys[i], range_out, 10000);
            total_found += count;
        }
        end = std::chrono::high_resolution_clock::now();
        double fast_range_ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>
            (end - start).count() / NUM_RANGE_QUERIES;
        std::cout << "fast_range (1000 results): " << fast_range_ns 
                  << " ns/query (avg " << (total_found/NUM_RANGE_QUERIES) << " results)" << std::endl;
    }
    
    // BitBlock range benchmark
    std::cout << "\n=== BITBLOCK RANGE BENCHMARK ===" << std::endl;
    
    // 5 results
    {
        std::vector<int64_t> range_hi_keys;
        range_hi_keys.reserve(NUM_RANGE_QUERIES);
        for (int i = 0; i < NUM_RANGE_QUERIES; i++) {
            range_hi_keys.push_back(range_lo_keys[i] + 5000);
        }
        flush_cache();
        volatile size_t total_found = 0;
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_RANGE_QUERIES; i++) {
            int count = idx_range.find_range_bitblock(range_lo_keys[i], range_hi_keys[i], range_out, 10000);
            total_found += count;
        }
        end = std::chrono::high_resolution_clock::now();
        double bitblock_ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>
            (end - start).count() / NUM_RANGE_QUERIES;
        std::cout << "BitBlock range (5 results): " << bitblock_ns 
                  << " ns/query (avg " << (total_found/NUM_RANGE_QUERIES) << " results)" << std::endl;
    }
    
    // 50 results
    {
        std::vector<int64_t> range_hi_keys;
        range_hi_keys.reserve(NUM_RANGE_QUERIES);
        for (int i = 0; i < NUM_RANGE_QUERIES; i++) {
            range_hi_keys.push_back(range_lo_keys[i] + 50000);
        }
        flush_cache();
        volatile size_t total_found = 0;
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_RANGE_QUERIES; i++) {
            int count = idx_range.find_range_bitblock(range_lo_keys[i], range_hi_keys[i], range_out, 10000);
            total_found += count;
        }
        end = std::chrono::high_resolution_clock::now();
        double bitblock_ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>
            (end - start).count() / NUM_RANGE_QUERIES;
        std::cout << "BitBlock range (50 results): " << bitblock_ns 
                  << " ns/query (avg " << (total_found/NUM_RANGE_QUERIES) << " results)" << std::endl;
    }
    
    // 100 results
    {
        std::vector<int64_t> range_hi_keys;
        range_hi_keys.reserve(NUM_RANGE_QUERIES);
        for (int i = 0; i < NUM_RANGE_QUERIES; i++) {
            range_hi_keys.push_back(range_lo_keys[i] + 100000);
        }
        flush_cache();
        volatile size_t total_found = 0;
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_RANGE_QUERIES; i++) {
            int count = idx_range.find_range_bitblock(range_lo_keys[i], range_hi_keys[i], range_out, 10000);
            total_found += count;
        }
        end = std::chrono::high_resolution_clock::now();
        double bitblock_ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>
            (end - start).count() / NUM_RANGE_QUERIES;
        std::cout << "BitBlock range (100 results): " << bitblock_ns 
                  << " ns/query (avg " << (total_found/NUM_RANGE_QUERIES) << " results)" << std::endl;
    }
    
    // 1000 results
    {
        std::vector<int64_t> range_hi_keys;
        range_hi_keys.reserve(NUM_RANGE_QUERIES);
        for (int i = 0; i < NUM_RANGE_QUERIES; i++) {
            range_hi_keys.push_back(range_lo_keys[i] + 1000000);
        }
        flush_cache();
        volatile size_t total_found = 0;
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_RANGE_QUERIES; i++) {
            int count = idx_range.find_range_bitblock(range_lo_keys[i], range_hi_keys[i], range_out, 10000);
            total_found += count;
        }
        end = std::chrono::high_resolution_clock::now();
        double bitblock_ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>
            (end - start).count() / NUM_RANGE_QUERIES;
        std::cout << "BitBlock range (1000 results): " << bitblock_ns 
                  << " ns/query (avg " << (total_found/NUM_RANGE_QUERIES) << " results)" << std::endl;
    }
    
    // Benchmark std::lower_bound range scan for comparison (5 results)
    {
        std::vector<int64_t> range_hi_keys;
        range_hi_keys.reserve(NUM_RANGE_QUERIES);
        for (int i = 0; i < NUM_RANGE_QUERIES; i++) {
            range_hi_keys.push_back(range_lo_keys[i] + 5000);
        }
        flush_cache();
        volatile size_t total_found = 0;
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_RANGE_QUERIES; i++) {
            auto lo_it = std::lower_bound(flat_array.begin(), flat_array.end(), range_lo_keys[i]);
            auto hi_it = std::upper_bound(flat_array.begin(), flat_array.end(), range_hi_keys[i]);
            total_found += (hi_it - lo_it);
        }
        end = std::chrono::high_resolution_clock::now();
        double lower_bound_range_ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>
            (end - start).count() / NUM_RANGE_QUERIES;
        std::cout << "std::lower_bound range (5 results): " << lower_bound_range_ns
                  << " ns/query (avg " << (total_found/NUM_RANGE_QUERIES) << " results)" << std::endl;
    }

    // Miss benchmark (1M lookups on keys NOT in dataset)
    std::cout << "\n=== MISS BENCHMARK ===" << std::endl;
    std::vector<int64_t> miss_keys;
    miss_keys.reserve(NUM_LOOKUPS);
    for (int i = 0; i < NUM_LOOKUPS; i++) {
        // Generate keys that are NOT in the dataset (odd numbers)
        miss_keys.push_back(i * 2000 + 1);
    }
    
    flush_cache();
    sum = 0;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_LOOKUPS; i++) {
        auto it = std::lower_bound(flat_array.begin(), flat_array.end(), miss_keys[i]);
        if (it != flat_array.end()) {
            sum += *it;
        }
    }
    end = std::chrono::high_resolution_clock::now();
    auto lower_bound_miss_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double lower_bound_miss_ns = (double)lower_bound_miss_duration / NUM_LOOKUPS;
    std::cout << "std::lower_bound miss: " << lower_bound_miss_ns << " ns/miss" << std::endl;
    
    flush_cache();
    NSIndex<int64_t> idx_miss(flat_array.data(), flat_array.size());
    // No flush needed - delta_buffer_ is already empty
    sum = 0;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_LOOKUPS; i++) {
        auto it = idx_miss.find(flat_array.data(), flat_array.size(), miss_keys[i]);
        if (it != flat_array.data() + flat_array.size()) {
            sum += *it;
        }
    }
    end = std::chrono::high_resolution_clock::now();
    auto nsindex_miss_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double nsindex_miss_ns = (double)nsindex_miss_duration / NUM_LOOKUPS;
    std::cout << "NSIndex miss: " << nsindex_miss_ns << " ns/miss" << std::endl;
    std::cout << "Speedup vs lower_bound miss: " << (lower_bound_miss_ns / nsindex_miss_ns) << "x" << std::endl;
    
    // PGM miss benchmark
    flush_cache();
    sum = 0;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_LOOKUPS; i++) {
        auto range = pgm_index.search(miss_keys[i]);
        auto it = std::lower_bound(
            flat_array.begin() + range.lo,
            flat_array.begin() + range.hi,
            miss_keys[i]);
        if (it != flat_array.end()) {
            sum += *it;
        }
    }
    end = std::chrono::high_resolution_clock::now();
    auto pgm_miss_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double pgm_miss_ns = (double)pgm_miss_duration / NUM_LOOKUPS;
    std::cout << "PGM-Index miss: " << pgm_miss_ns << " ns/miss" << std::endl;
    std::cout << "NSIndex vs PGM miss: " << (pgm_miss_ns / nsindex_miss_ns) << "x faster" << std::endl;

    // Predecessor benchmark
    std::cout << "\n=== PREDECESSOR BENCHMARK ===" << std::endl;
    flush_cache();
    NSIndex<int64_t> idx_pred(flat_array.data(), flat_array.size());
    // No flush needed - delta_buffer_ is already empty
    sum = 0;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_LOOKUPS; i++) {
        auto it = idx_pred.predecessor(flat_array.data(), flat_array.size(), miss_keys[i]);
        if (it != nullptr) {
            sum += *it;
        }
    }
    end = std::chrono::high_resolution_clock::now();
    auto nsindex_pred_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double nsindex_pred_ns = (double)nsindex_pred_duration / NUM_LOOKUPS;
    std::cout << "NSIndex predecessor: " << nsindex_pred_ns << " ns/query" << std::endl;
    
    flush_cache();
    sum = 0;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_LOOKUPS; i++) {
        auto it = std::lower_bound(flat_array.begin(), flat_array.end(), miss_keys[i]);
        if (it != flat_array.begin()) {
            --it;
            sum += *it;
        }
    }
    end = std::chrono::high_resolution_clock::now();
    auto lower_bound_pred_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double lower_bound_pred_ns = (double)lower_bound_pred_duration / NUM_LOOKUPS;
    std::cout << "std::lower_bound predecessor: " << lower_bound_pred_ns << " ns/query" << std::endl;
    std::cout << "Speedup vs lower_bound: " << (lower_bound_pred_ns / nsindex_pred_ns) << "x" << std::endl;

    // Successor benchmark
    std::cout << "\n=== SUCCESSOR BENCHMARK ===" << std::endl;
    flush_cache();
    NSIndex<int64_t> idx_succ(flat_array.data(), flat_array.size());
    // No flush needed - delta_buffer_ is already empty
    sum = 0;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_LOOKUPS; i++) {
        auto it = idx_succ.successor(flat_array.data(), flat_array.size(), miss_keys[i]);
        if (it != nullptr) {
            sum += *it;
        }
    }
    end = std::chrono::high_resolution_clock::now();
    auto nsindex_succ_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double nsindex_succ_ns = (double)nsindex_succ_duration / NUM_LOOKUPS;
    std::cout << "NSIndex successor: " << nsindex_succ_ns << " ns/query" << std::endl;
    
    flush_cache();
    sum = 0;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_LOOKUPS; i++) {
        auto it = std::lower_bound(flat_array.begin(), flat_array.end(), miss_keys[i]);
        if (it != flat_array.end()) {
            sum += *it;
        }
    }
    end = std::chrono::high_resolution_clock::now();
    auto lower_bound_succ_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double lower_bound_succ_ns = (double)lower_bound_succ_duration / NUM_LOOKUPS;
    std::cout << "std::lower_bound successor: " << lower_bound_succ_ns << " ns/query" << std::endl;
    std::cout << "Speedup vs lower_bound: " << (lower_bound_succ_ns / nsindex_succ_ns) << "x" << std::endl;

    // Insert benchmark (1M sequential inserts vs std::map)
    std::cout << "\n=== INSERT BENCHMARK ===" << std::endl;
    
    // Benchmark std::map insert
    std::map<int64_t, int64_t> insert_map;
    flush_cache();
    start = std::chrono::high_resolution_clock::now();
    for (int64_t i = 0; i < NUM_INSERTS; i++) {
        insert_map[i * 1000] = i;
    }
    end = std::chrono::high_resolution_clock::now();
    auto map_insert_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double map_insert_ns = (double)map_insert_duration / NUM_INSERTS;
    std::cout << "std::map insert: " << map_insert_ns << " ns/insert" << std::endl;
    
    // Benchmark NSIndex insert (with auto-flush)
    std::vector<int64_t> insert_array;
    insert_array.reserve(NUM_KEYS);
    for (int64_t i = 0; i < NUM_KEYS; i++) {
        insert_array.push_back(i * 1000);
    }
    NSIndex<int64_t> idx_insert(insert_array.data(), insert_array.size());
    flush_cache();
    start = std::chrono::high_resolution_clock::now();
    for (int64_t i = NUM_KEYS; i < NUM_KEYS + NUM_INSERTS; i++) {
        idx_insert.insert(i * 1000);
        if (idx_insert.needs_flush()) {
            idx_insert.flush(insert_array);
        }
    }
    end = std::chrono::high_resolution_clock::now();
    auto nsindex_insert_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double nsindex_insert_ns = (double)nsindex_insert_duration / NUM_INSERTS;
    std::cout << "NSIndex insert (with auto-flush): " << nsindex_insert_ns << " ns/insert" << std::endl;
    std::cout << "Speedup vs std::map: " << (map_insert_ns / nsindex_insert_ns) << "x" << std::endl;
    
    // Insert + flush correctness
    std::vector<int64_t> dynamic_array(flat_array.begin(), flat_array.end());
    NSIndex<int64_t> idx_dyn(dynamic_array.data(), dynamic_array.size());
    idx_dyn.insert(9999999500LL);
    idx_dyn.insert(9999999750LL);
    idx_dyn.flush(dynamic_array);
    NSIndex<int64_t> idx_after(dynamic_array.data(), dynamic_array.size());
    auto c1 = idx_after.find(dynamic_array.data(), dynamic_array.size(), 9999999500LL);
    auto c2 = idx_after.find(dynamic_array.data(), dynamic_array.size(), 9999999750LL);
    std::cout << "Insert+flush correctness: "
              << (c1 != dynamic_array.data()+dynamic_array.size() &&
                  c2 != dynamic_array.data()+dynamic_array.size() ? "PASS" : "FAIL")
              << std::endl;
    
    // Results
    std::cout << "\n=== RESULTS ===" << std::endl;
    std::cout << "std::lower_bound: " << lower_bound_ns_per_lookup << " ns/lookup" << std::endl;
    std::cout << "std::map:        " << map_ns_per_lookup << " ns/lookup" << std::endl;
    std::cout << "NSIndex:          " << nsindex_ns << " ns/lookup" << std::endl;
    std::cout << "NSIndex (hits only): " << nsindex_hits_ns << " ns/lookup" << std::endl;
    std::cout << "NSIndex (clustered): " << nsindex_clustered_ns << " ns/lookup" << std::endl;
    std::cout << "std::lower_bound (clustered): " << lower_bound_clustered_ns << " ns/lookup" << std::endl;
    std::cout << "PGM-Index:        " << pgm_ns << " ns/lookup" << std::endl;
    std::cout << "Speedup vs lower_bound: " << (lower_bound_ns_per_lookup / nsindex_ns) << "x" << std::endl;
    std::cout << "Speedup vs lower_bound (clustered): " << (lower_bound_clustered_ns / nsindex_clustered_ns) << "x" << std::endl;
    std::cout << "NSIndex vs PGM:   " << (pgm_ns / nsindex_ns) << "x faster" << std::endl;
    std::cout << "NSIndex hits-only vs PGM: " << (pgm_ns / nsindex_hits_ns) << "x faster" << std::endl;
    std::cout << "\nMiss results:" << std::endl;
    std::cout << "NSIndex miss: " << nsindex_miss_ns << " ns/miss" << std::endl;
    std::cout << "std::lower_bound miss: " << lower_bound_miss_ns << " ns/miss" << std::endl;
    std::cout << "PGM-Index miss: " << pgm_miss_ns << " ns/miss" << std::endl;
    std::cout << "NSIndex vs PGM miss: " << (pgm_miss_ns / nsindex_miss_ns) << "x faster" << std::endl;
    std::cout << "\nPredecessor results:" << std::endl;
    std::cout << "NSIndex predecessor: " << nsindex_pred_ns << " ns/query" << std::endl;
    std::cout << "std::lower_bound predecessor: " << lower_bound_pred_ns << " ns/query" << std::endl;
    std::cout << "Speedup vs lower_bound: " << (lower_bound_pred_ns / nsindex_pred_ns) << "x" << std::endl;
    std::cout << "\nSuccessor results:" << std::endl;
    std::cout << "NSIndex successor: " << nsindex_succ_ns << " ns/query" << std::endl;
    std::cout << "std::lower_bound successor: " << lower_bound_succ_ns << " ns/query" << std::endl;
    std::cout << "Speedup vs lower_bound: " << (lower_bound_succ_ns / nsindex_succ_ns) << "x" << std::endl;
    
    std::cout << "AVX2 path: " << (__builtin_cpu_supports("avx2") ? "YES" : "NO") << std::endl;
    std::cout << "AVX-512 path: " << (__builtin_cpu_supports("avx512f") ? "YES" : "NO") << std::endl;
    
    // Run range benchmarks on skewed datasets
    auto exponential_data = generate_exponential_dataset(NUM_KEYS);
    run_range_benchmark(exponential_data, "Exponential Distribution");
    
    auto clustered_data = generate_clustered_dataset(NUM_KEYS);
    run_range_benchmark(clustered_data, "Clustered Distribution");
    
    return 0;
}

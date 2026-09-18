// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#include <benchmark/benchmark.h>
#include <absl/container/flat_hash_map.h>
#include "third_party/ankerl/unordered_dense.h"
#include "third_party/robin_hood.h"
#include "third_party/fph-table/include/fph/dynamic_fph_table.h"
#include <boost/unordered/unordered_flat_map.hpp>
#include "nshash.hpp"
#include <random>
#include <vector>
#include <cstdint>
#include <unordered_map>
#include <string>
#include <cstdio>

constexpr size_t kNumKeys = 1000000;

std::vector<uint64_t> GenerateSequentialKeys() {
    std::vector<uint64_t> keys;
    keys.reserve(kNumKeys);
    for (size_t i = 1; i <= kNumKeys; ++i) {
        keys.push_back(i);
    }
    return keys;
}

std::vector<uint64_t> GenerateRandomKeys() {
    std::vector<uint64_t> keys;
    keys.reserve(kNumKeys);
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint64_t> dist;
    for (size_t i = 0; i < kNumKeys; ++i) {
        keys.push_back(dist(rng));
    }
    return keys;
}

std::vector<uint64_t> GenerateBoundedKeys() {
    std::vector<uint64_t> keys;
    keys.reserve(kNumKeys);
    for (size_t i = 0; i < kNumKeys; ++i) {
        keys.push_back(i % 65536);
    }
    return keys;
}

std::vector<uint64_t> GenerateTimestampKeys() {
    std::vector<uint64_t> keys;
    keys.reserve(kNumKeys);
    uint64_t base = 1700000000ULL;
    for (size_t i = 0; i < kNumKeys; ++i) {
        keys.push_back(base + i);
    }
    return keys;
}

// NSHash benchmarks
static void BM_NSHash_Sequential(benchmark::State& state) {
    auto keys = GenerateSequentialKeys();
    NSHash<uint64_t, uint64_t> map;
    map.reserve(2097152);
    for (auto key : keys) {
        map.insert(key, key);
    }
    
    std::mt19937_64 rng(42);
    std::shuffle(keys.begin(), keys.end(), rng);
    
    for (auto _ : state) {
        for (auto key : keys) {
            benchmark::DoNotOptimize(map.lookup(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_NSHash_Random(benchmark::State& state) {
    auto keys = GenerateRandomKeys();
    NSHash<uint64_t, uint64_t> map;
    map.reserve(2097152);
    for (auto key : keys) {
        map.insert(key, key);
    }
    
    for (auto _ : state) {
        for (auto key : keys) {
            benchmark::DoNotOptimize(map.lookup(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_NSHash_Bounded(benchmark::State& state) {
    auto keys = GenerateBoundedKeys();
    NSHash<uint64_t, uint64_t> map;
    map.reserve(2097152);
    for (auto key : keys) {
        map.insert(key, key);
    }
    
    for (auto _ : state) {
        for (auto key : keys) {
            benchmark::DoNotOptimize(map.lookup(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_NSHash_Timestamp(benchmark::State& state) {
    auto keys = GenerateTimestampKeys();
    NSHash<uint64_t, uint64_t> map;
    map.reserve(2097152);
    for (auto key : keys) {
        map.insert(key, key);
    }
    
    for (auto _ : state) {
        for (auto key : keys) {
            benchmark::DoNotOptimize(map.lookup(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

// absl::flat_hash_map benchmarks
static void BM_Absl_Sequential(benchmark::State& state) {
    auto keys = GenerateSequentialKeys();
    absl::flat_hash_map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    for (auto key : keys) {
        map[key] = key;
    }
    
    std::mt19937_64 rng(42);
    std::shuffle(keys.begin(), keys.end(), rng);
    
    for (auto _ : state) {
        for (auto key : keys) {
            benchmark::DoNotOptimize(map.find(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_Absl_Random(benchmark::State& state) {
    auto keys = GenerateRandomKeys();
    absl::flat_hash_map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    for (auto key : keys) {
        map[key] = key;
    }
    
    for (auto _ : state) {
        for (auto key : keys) {
            benchmark::DoNotOptimize(map.find(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_Absl_Bounded(benchmark::State& state) {
    auto keys = GenerateBoundedKeys();
    absl::flat_hash_map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    for (auto key : keys) {
        map[key] = key;
    }
    
    for (auto _ : state) {
        for (auto key : keys) {
            benchmark::DoNotOptimize(map.find(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_Absl_Timestamp(benchmark::State& state) {
    auto keys = GenerateTimestampKeys();
    absl::flat_hash_map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    for (auto key : keys) {
        map[key] = key;
    }
    
    for (auto _ : state) {
        for (auto key : keys) {
            benchmark::DoNotOptimize(map.find(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

// ankerl::unordered_dense benchmarks
static void BM_Ankerl_Sequential(benchmark::State& state) {
    auto keys = GenerateSequentialKeys();
    ankerl::unordered_dense::map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    for (auto key : keys) {
        map[key] = key;
    }
    
    std::mt19937_64 rng(42);
    std::shuffle(keys.begin(), keys.end(), rng);
    
    for (auto _ : state) {
        for (auto key : keys) {
            benchmark::DoNotOptimize(map.find(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_Ankerl_Random(benchmark::State& state) {
    auto keys = GenerateRandomKeys();
    ankerl::unordered_dense::map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    for (auto key : keys) {
        map[key] = key;
    }
    
    for (auto _ : state) {
        for (auto key : keys) {
            benchmark::DoNotOptimize(map.find(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_Ankerl_Bounded(benchmark::State& state) {
    auto keys = GenerateBoundedKeys();
    ankerl::unordered_dense::map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    for (auto key : keys) {
        map[key] = key;
    }
    
    for (auto _ : state) {
        for (auto key : keys) {
            benchmark::DoNotOptimize(map.find(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_Ankerl_Timestamp(benchmark::State& state) {
    auto keys = GenerateTimestampKeys();
    ankerl::unordered_dense::map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    for (auto key : keys) {
        map[key] = key;
    }
    
    for (auto _ : state) {
        for (auto key : keys) {
            benchmark::DoNotOptimize(map.find(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

// robin_hood::unordered_flat_map benchmarks
static void BM_RobinHood_Sequential(benchmark::State& state) {
    auto keys = GenerateSequentialKeys();
    robin_hood::unordered_flat_map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    for (auto key : keys) {
        map[key] = key;
    }
    
    std::mt19937_64 rng(42);
    std::shuffle(keys.begin(), keys.end(), rng);
    
    for (auto _ : state) {
        for (auto key : keys) {
            benchmark::DoNotOptimize(map.find(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_RobinHood_Random(benchmark::State& state) {
    auto keys = GenerateRandomKeys();
    robin_hood::unordered_flat_map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    for (auto key : keys) {
        map[key] = key;
    }
    
    for (auto _ : state) {
        for (auto key : keys) {
            benchmark::DoNotOptimize(map.find(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_RobinHood_Bounded(benchmark::State& state) {
    auto keys = GenerateBoundedKeys();
    robin_hood::unordered_flat_map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    for (auto key : keys) {
        map[key] = key;
    }
    
    for (auto _ : state) {
        for (auto key : keys) {
            benchmark::DoNotOptimize(map.find(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_RobinHood_Timestamp(benchmark::State& state) {
    auto keys = GenerateTimestampKeys();
    robin_hood::unordered_flat_map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    for (auto key : keys) {
        map[key] = key;
    }
    
    for (auto _ : state) {
        for (auto key : keys) {
            benchmark::DoNotOptimize(map.find(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

// std::unordered_map benchmarks
static void BM_Std_Sequential(benchmark::State& state) {
    auto keys = GenerateSequentialKeys();
    std::unordered_map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    for (auto key : keys) {
        map[key] = key;
    }
    
    std::mt19937_64 rng(42);
    std::shuffle(keys.begin(), keys.end(), rng);
    
    for (auto _ : state) {
        for (auto key : keys) {
            benchmark::DoNotOptimize(map.find(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_Std_Random(benchmark::State& state) {
    auto keys = GenerateRandomKeys();
    std::unordered_map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    for (auto key : keys) {
        map[key] = key;
    }
    
    for (auto _ : state) {
        for (auto key : keys) {
            benchmark::DoNotOptimize(map.find(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_Std_Bounded(benchmark::State& state) {
    auto keys = GenerateBoundedKeys();
    std::unordered_map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    for (auto key : keys) {
        map[key] = key;
    }
    
    for (auto _ : state) {
        for (auto key : keys) {
            benchmark::DoNotOptimize(map.find(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_Std_Timestamp(benchmark::State& state) {
    auto keys = GenerateTimestampKeys();
    std::unordered_map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    for (auto key : keys) {
        map[key] = key;
    }
    
    for (auto _ : state) {
        for (auto key : keys) {
            benchmark::DoNotOptimize(map.find(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

// fph::DynamicFphMap benchmarks
static void BM_Fph_Sequential(benchmark::State& state) {
    auto keys = GenerateSequentialKeys();
    fph::DynamicFphMap<uint64_t, uint64_t> map;
    for (auto key : keys) {
        map[key] = key;
    }
    
    std::mt19937_64 rng(42);
    std::shuffle(keys.begin(), keys.end(), rng);
    
    for (auto _ : state) {
        for (auto key : keys) {
            benchmark::DoNotOptimize(map.find(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_Fph_Random(benchmark::State& state) {
    auto keys = GenerateRandomKeys();
    fph::DynamicFphMap<uint64_t, uint64_t> map;
    for (auto key : keys) {
        map[key] = key;
    }
    
    for (auto _ : state) {
        for (auto key : keys) {
            benchmark::DoNotOptimize(map.find(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_Fph_Bounded(benchmark::State& state) {
    auto keys = GenerateBoundedKeys();
    fph::DynamicFphMap<uint64_t, uint64_t> map;
    for (auto key : keys) {
        map[key] = key;
    }
    
    for (auto _ : state) {
        for (auto key : keys) {
            benchmark::DoNotOptimize(map.find(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_Fph_Timestamp(benchmark::State& state) {
    auto keys = GenerateTimestampKeys();
    fph::DynamicFphMap<uint64_t, uint64_t> map;
    for (auto key : keys) {
        map[key] = key;
    }
    
    for (auto _ : state) {
        for (auto key : keys) {
            benchmark::DoNotOptimize(map.find(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}


// boost::unordered_flat_map benchmarks
static void BM_Boost_Sequential(benchmark::State& state) {
    auto keys = GenerateSequentialKeys();
    boost::unordered_flat_map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    for (auto key : keys) {
        map[key] = key;
    }
    
    std::mt19937_64 rng(42);
    std::shuffle(keys.begin(), keys.end(), rng);
    
    for (auto _ : state) {
        for (auto key : keys) {
            benchmark::DoNotOptimize(map.find(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_Boost_Random(benchmark::State& state) {
    auto keys = GenerateRandomKeys();
    boost::unordered_flat_map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    for (auto key : keys) {
        map[key] = key;
    }
    
    for (auto _ : state) {
        for (auto key : keys) {
            benchmark::DoNotOptimize(map.find(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_Boost_Bounded(benchmark::State& state) {
    auto keys = GenerateBoundedKeys();
    boost::unordered_flat_map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    for (auto key : keys) {
        map[key] = key;
    }
    
    for (auto _ : state) {
        for (auto key : keys) {
            benchmark::DoNotOptimize(map.find(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_Boost_Timestamp(benchmark::State& state) {
    auto keys = GenerateTimestampKeys();
    boost::unordered_flat_map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    for (auto key : keys) {
        map[key] = key;
    }
    
    for (auto _ : state) {
        for (auto key : keys) {
            benchmark::DoNotOptimize(map.find(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

// Miss benchmarks
static void BM_NSHash_Miss_Sequential(benchmark::State& state) {
    auto keys = GenerateSequentialKeys();
    NSHash<uint64_t, uint64_t> map;
    map.reserve(2097152);
    for (auto key : keys) {
        map.insert(key, key);
    }
    
    std::vector<uint64_t> miss_keys;
    miss_keys.reserve(kNumKeys);
    for (auto key : keys) {
        miss_keys.push_back(key + 1000000ULL);
    }
    
    for (auto _ : state) {
        for (auto key : miss_keys) {
            benchmark::DoNotOptimize(map.lookup(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_NSHash_Miss_Random(benchmark::State& state) {
    auto keys = GenerateRandomKeys();
    NSHash<uint64_t, uint64_t> map;
    map.reserve(2097152);
    for (auto key : keys) {
        map.insert(key, key);
    }
    
    std::vector<uint64_t> miss_keys;
    miss_keys.reserve(kNumKeys);
    std::mt19937_64 rng(999);
    std::uniform_int_distribution<uint64_t> dist;
    for (size_t i = 0; i < kNumKeys; ++i) {
        miss_keys.push_back(dist(rng));
    }
    
    for (auto _ : state) {
        for (auto key : miss_keys) {
            benchmark::DoNotOptimize(map.lookup(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_NSHash_Miss_Timestamp(benchmark::State& state) {
    auto keys = GenerateTimestampKeys();
    NSHash<uint64_t, uint64_t> map;
    map.reserve(2097152);
    for (auto key : keys) {
        map.insert(key, key);
    }
    
    std::vector<uint64_t> miss_keys;
    miss_keys.reserve(kNumKeys);
    uint64_t base = 1800000000ULL;
    for (size_t i = 0; i < kNumKeys; ++i) {
        miss_keys.push_back(base + i);
    }
    
    for (auto _ : state) {
        for (auto key : miss_keys) {
            benchmark::DoNotOptimize(map.lookup(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_Absl_Miss_Sequential(benchmark::State& state) {
    auto keys = GenerateSequentialKeys();
    absl::flat_hash_map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    for (auto key : keys) {
        map[key] = key;
    }
    
    std::vector<uint64_t> miss_keys;
    miss_keys.reserve(kNumKeys);
    for (auto key : keys) {
        miss_keys.push_back(key + 1000000ULL);
    }
    
    for (auto _ : state) {
        for (auto key : miss_keys) {
            benchmark::DoNotOptimize(map.find(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_Absl_Miss_Random(benchmark::State& state) {
    auto keys = GenerateRandomKeys();
    absl::flat_hash_map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    for (auto key : keys) {
        map[key] = key;
    }
    
    std::vector<uint64_t> miss_keys;
    miss_keys.reserve(kNumKeys);
    std::mt19937_64 rng(999);
    std::uniform_int_distribution<uint64_t> dist;
    for (size_t i = 0; i < kNumKeys; ++i) {
        miss_keys.push_back(dist(rng));
    }
    
    for (auto _ : state) {
        for (auto key : miss_keys) {
            benchmark::DoNotOptimize(map.find(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_Absl_Miss_Timestamp(benchmark::State& state) {
    auto keys = GenerateTimestampKeys();
    absl::flat_hash_map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    for (auto key : keys) {
        map[key] = key;
    }
    
    std::vector<uint64_t> miss_keys;
    miss_keys.reserve(kNumKeys);
    uint64_t base = 1800000000ULL;
    for (size_t i = 0; i < kNumKeys; ++i) {
        miss_keys.push_back(base + i);
    }
    
    for (auto _ : state) {
        for (auto key : miss_keys) {
            benchmark::DoNotOptimize(map.find(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_Ankerl_Miss_Sequential(benchmark::State& state) {
    auto keys = GenerateSequentialKeys();
    ankerl::unordered_dense::map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    for (auto key : keys) {
        map[key] = key;
    }
    
    std::vector<uint64_t> miss_keys;
    miss_keys.reserve(kNumKeys);
    for (auto key : keys) {
        miss_keys.push_back(key + 1000000ULL);
    }
    
    for (auto _ : state) {
        for (auto key : miss_keys) {
            benchmark::DoNotOptimize(map.find(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_Ankerl_Miss_Random(benchmark::State& state) {
    auto keys = GenerateRandomKeys();
    ankerl::unordered_dense::map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    for (auto key : keys) {
        map[key] = key;
    }
    
    std::vector<uint64_t> miss_keys;
    miss_keys.reserve(kNumKeys);
    std::mt19937_64 rng(999);
    std::uniform_int_distribution<uint64_t> dist;
    for (size_t i = 0; i < kNumKeys; ++i) {
        miss_keys.push_back(dist(rng));
    }
    
    for (auto _ : state) {
        for (auto key : miss_keys) {
            benchmark::DoNotOptimize(map.find(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_Ankerl_Miss_Timestamp(benchmark::State& state) {
    auto keys = GenerateTimestampKeys();
    ankerl::unordered_dense::map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    for (auto key : keys) {
        map[key] = key;
    }
    
    std::vector<uint64_t> miss_keys;
    miss_keys.reserve(kNumKeys);
    uint64_t base = 1800000000ULL;
    for (size_t i = 0; i < kNumKeys; ++i) {
        miss_keys.push_back(base + i);
    }
    
    for (auto _ : state) {
        for (auto key : miss_keys) {
            benchmark::DoNotOptimize(map.find(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_RobinHood_Miss_Sequential(benchmark::State& state) {
    auto keys = GenerateSequentialKeys();
    robin_hood::unordered_flat_map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    for (auto key : keys) {
        map[key] = key;
    }
    
    std::vector<uint64_t> miss_keys;
    miss_keys.reserve(kNumKeys);
    for (auto key : keys) {
        miss_keys.push_back(key + 1000000ULL);
    }
    
    for (auto _ : state) {
        for (auto key : miss_keys) {
            benchmark::DoNotOptimize(map.find(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_RobinHood_Miss_Random(benchmark::State& state) {
    auto keys = GenerateRandomKeys();
    robin_hood::unordered_flat_map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    for (auto key : keys) {
        map[key] = key;
    }
    
    std::vector<uint64_t> miss_keys;
    miss_keys.reserve(kNumKeys);
    std::mt19937_64 rng(999);
    std::uniform_int_distribution<uint64_t> dist;
    for (size_t i = 0; i < kNumKeys; ++i) {
        miss_keys.push_back(dist(rng));
    }
    
    for (auto _ : state) {
        for (auto key : miss_keys) {
            benchmark::DoNotOptimize(map.find(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_RobinHood_Miss_Timestamp(benchmark::State& state) {
    auto keys = GenerateTimestampKeys();
    robin_hood::unordered_flat_map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    for (auto key : keys) {
        map[key] = key;
    }
    
    std::vector<uint64_t> miss_keys;
    miss_keys.reserve(kNumKeys);
    uint64_t base = 1800000000ULL;
    for (size_t i = 0; i < kNumKeys; ++i) {
        miss_keys.push_back(base + i);
    }
    
    for (auto _ : state) {
        for (auto key : miss_keys) {
            benchmark::DoNotOptimize(map.find(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_Std_Miss_Sequential(benchmark::State& state) {
    auto keys = GenerateSequentialKeys();
    std::unordered_map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    for (auto key : keys) {
        map[key] = key;
    }
    
    std::vector<uint64_t> miss_keys;
    miss_keys.reserve(kNumKeys);
    for (auto key : keys) {
        miss_keys.push_back(key + 1000000ULL);
    }
    
    for (auto _ : state) {
        for (auto key : miss_keys) {
            benchmark::DoNotOptimize(map.find(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_Std_Miss_Random(benchmark::State& state) {
    auto keys = GenerateRandomKeys();
    std::unordered_map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    for (auto key : keys) {
        map[key] = key;
    }
    
    std::vector<uint64_t> miss_keys;
    miss_keys.reserve(kNumKeys);
    std::mt19937_64 rng(999);
    std::uniform_int_distribution<uint64_t> dist;
    for (size_t i = 0; i < kNumKeys; ++i) {
        miss_keys.push_back(dist(rng));
    }
    
    for (auto _ : state) {
        for (auto key : miss_keys) {
            benchmark::DoNotOptimize(map.find(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_Std_Miss_Timestamp(benchmark::State& state) {
    auto keys = GenerateTimestampKeys();
    std::unordered_map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    for (auto key : keys) {
        map[key] = key;
    }
    
    std::vector<uint64_t> miss_keys;
    miss_keys.reserve(kNumKeys);
    uint64_t base = 1800000000ULL;
    for (size_t i = 0; i < kNumKeys; ++i) {
        miss_keys.push_back(base + i);
    }
    
    for (auto _ : state) {
        for (auto key : miss_keys) {
            benchmark::DoNotOptimize(map.find(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_Fph_Miss_Sequential(benchmark::State& state) {
    auto keys = GenerateSequentialKeys();
    fph::DynamicFphMap<uint64_t, uint64_t> map;
    for (auto key : keys) {
        map[key] = key;
    }
    
    std::vector<uint64_t> miss_keys;
    miss_keys.reserve(kNumKeys);
    for (auto key : keys) {
        miss_keys.push_back(key + 1000000ULL);
    }
    
    for (auto _ : state) {
        for (auto key : miss_keys) {
            benchmark::DoNotOptimize(map.find(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_Fph_Miss_Random(benchmark::State& state) {
    auto keys = GenerateRandomKeys();
    fph::DynamicFphMap<uint64_t, uint64_t> map;
    for (auto key : keys) {
        map[key] = key;
    }
    
    std::vector<uint64_t> miss_keys;
    miss_keys.reserve(kNumKeys);
    std::mt19937_64 rng(999);
    std::uniform_int_distribution<uint64_t> dist;
    for (size_t i = 0; i < kNumKeys; ++i) {
        miss_keys.push_back(dist(rng));
    }
    
    for (auto _ : state) {
        for (auto key : miss_keys) {
            benchmark::DoNotOptimize(map.find(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_Fph_Miss_Timestamp(benchmark::State& state) {
    auto keys = GenerateTimestampKeys();
    fph::DynamicFphMap<uint64_t, uint64_t> map;
    for (auto key : keys) {
        map[key] = key;
    }
    
    std::vector<uint64_t> miss_keys;
    miss_keys.reserve(kNumKeys);
    uint64_t base = 1800000000ULL;
    for (size_t i = 0; i < kNumKeys; ++i) {
        miss_keys.push_back(base + i);
    }
    
    for (auto _ : state) {
        for (auto key : miss_keys) {
            benchmark::DoNotOptimize(map.find(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}


// boost miss benchmarks
static void BM_Boost_Miss_Sequential(benchmark::State& state) {
    auto keys = GenerateSequentialKeys();
    boost::unordered_flat_map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    for (auto key : keys) {
        map[key] = key;
    }
    
    std::vector<uint64_t> miss_keys;
    miss_keys.reserve(kNumKeys);
    for (auto key : keys) {
        miss_keys.push_back(key + 1000000ULL);
    }
    
    for (auto _ : state) {
        for (auto key : miss_keys) {
            benchmark::DoNotOptimize(map.find(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_Boost_Miss_Random(benchmark::State& state) {
    auto keys = GenerateRandomKeys();
    boost::unordered_flat_map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    for (auto key : keys) {
        map[key] = key;
    }
    
    std::vector<uint64_t> miss_keys;
    miss_keys.reserve(kNumKeys);
    std::mt19937_64 rng(999);
    std::uniform_int_distribution<uint64_t> dist;
    for (size_t i = 0; i < kNumKeys; ++i) {
        miss_keys.push_back(dist(rng));
    }
    
    for (auto _ : state) {
        for (auto key : miss_keys) {
            benchmark::DoNotOptimize(map.find(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_Boost_Miss_Timestamp(benchmark::State& state) {
    auto keys = GenerateTimestampKeys();
    boost::unordered_flat_map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    for (auto key : keys) {
        map[key] = key;
    }
    
    std::vector<uint64_t> miss_keys;
    miss_keys.reserve(kNumKeys);
    uint64_t base = 1800000000ULL;
    for (size_t i = 0; i < kNumKeys; ++i) {
        miss_keys.push_back(base + i);
    }
    
    for (auto _ : state) {
        for (auto key : miss_keys) {
            benchmark::DoNotOptimize(map.find(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

// Insert benchmarks (std::unordered_map)
static void BM_Std_Insert_Sequential(benchmark::State& state) {
    auto keys = GenerateSequentialKeys();
    std::unordered_map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    
    for (auto _ : state) {
        map.clear();
        map.reserve(1000000);
        for (auto key : keys) {
            map[key] = key;
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_Std_Insert_Random(benchmark::State& state) {
    auto keys = GenerateRandomKeys();
    std::unordered_map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    
    for (auto _ : state) {
        map.clear();
        map.reserve(1000000);
        for (auto key : keys) {
            map[key] = key;
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_Std_Insert_Bounded(benchmark::State& state) {
    auto keys = GenerateBoundedKeys();
    std::unordered_map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    
    for (auto _ : state) {
        map.clear();
        map.reserve(1000000);
        for (auto key : keys) {
            map[key] = key;
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_Std_Insert_Timestamp(benchmark::State& state) {
    auto keys = GenerateTimestampKeys();
    std::unordered_map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    
    for (auto _ : state) {
        map.clear();
        map.reserve(1000000);
        for (auto key : keys) {
            map[key] = key;
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

// Insert benchmarks (fph)
static void BM_Fph_Insert_Sequential(benchmark::State& state) {
    auto keys = GenerateSequentialKeys();
    
    for (auto _ : state) {
        fph::DynamicFphMap<uint64_t, uint64_t> map;
        for (auto key : keys) {
            map[key] = key;
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_Fph_Insert_Random(benchmark::State& state) {
    auto keys = GenerateRandomKeys();
    
    for (auto _ : state) {
        fph::DynamicFphMap<uint64_t, uint64_t> map;
        for (auto key : keys) {
            map[key] = key;
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_Fph_Insert_Bounded(benchmark::State& state) {
    auto keys = GenerateBoundedKeys();
    
    for (auto _ : state) {
        fph::DynamicFphMap<uint64_t, uint64_t> map;
        for (auto key : keys) {
            map[key] = key;
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_Fph_Insert_Timestamp(benchmark::State& state) {
    auto keys = GenerateTimestampKeys();
    
    for (auto _ : state) {
        fph::DynamicFphMap<uint64_t, uint64_t> map;
        for (auto key : keys) {
            map[key] = key;
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

// Insert benchmarks
static void BM_NSHash_Insert_Sequential(benchmark::State& state) {
    auto keys = GenerateSequentialKeys();
    NSHash<uint64_t, uint64_t> map;
    map.reserve(2097152);
    
    for (auto _ : state) {
        map.clear();
        map.reserve(2097152);
        for (auto key : keys) {
            map.insert(key, key);
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_NSHash_Insert_Random(benchmark::State& state) {
    auto keys = GenerateRandomKeys();
    NSHash<uint64_t, uint64_t> map;
    map.reserve(2097152);
    
    for (auto _ : state) {
        map.clear();
        map.reserve(2097152);
        for (auto key : keys) {
            map.insert(key, key);
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_NSHash_Insert_Bounded(benchmark::State& state) {
    auto keys = GenerateBoundedKeys();
    NSHash<uint64_t, uint64_t> map;
    map.reserve(2097152);
    
    for (auto _ : state) {
        map.clear();
        map.reserve(2097152);
        for (auto key : keys) {
            map.insert(key, key);
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_NSHash_Insert_Timestamp(benchmark::State& state) {
    auto keys = GenerateTimestampKeys();
    NSHash<uint64_t, uint64_t> map;
    map.reserve(2097152);
    
    for (auto _ : state) {
        map.clear();
        map.reserve(2097152);
        for (auto key : keys) {
            map.insert(key, key);
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_Absl_Insert_Sequential(benchmark::State& state) {
    auto keys = GenerateSequentialKeys();
    absl::flat_hash_map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    
    for (auto _ : state) {
        map.clear();
        map.reserve(1000000);
        for (auto key : keys) {
            map[key] = key;
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_Absl_Insert_Random(benchmark::State& state) {
    auto keys = GenerateRandomKeys();
    absl::flat_hash_map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    
    for (auto _ : state) {
        map.clear();
        map.reserve(1000000);
        for (auto key : keys) {
            map[key] = key;
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_Absl_Insert_Bounded(benchmark::State& state) {
    auto keys = GenerateBoundedKeys();
    absl::flat_hash_map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    
    for (auto _ : state) {
        map.clear();
        map.reserve(1000000);
        for (auto key : keys) {
            map[key] = key;
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_Absl_Insert_Timestamp(benchmark::State& state) {
    auto keys = GenerateTimestampKeys();
    absl::flat_hash_map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    
    for (auto _ : state) {
        map.clear();
        map.reserve(1000000);
        for (auto key : keys) {
            map[key] = key;
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_Ankerl_Insert_Sequential(benchmark::State& state) {
    auto keys = GenerateSequentialKeys();
    ankerl::unordered_dense::map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    
    for (auto _ : state) {
        map.clear();
        map.reserve(1000000);
        for (auto key : keys) {
            map[key] = key;
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_Ankerl_Insert_Random(benchmark::State& state) {
    auto keys = GenerateRandomKeys();
    ankerl::unordered_dense::map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    
    for (auto _ : state) {
        map.clear();
        map.reserve(1000000);
        for (auto key : keys) {
            map[key] = key;
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_Ankerl_Insert_Bounded(benchmark::State& state) {
    auto keys = GenerateBoundedKeys();
    ankerl::unordered_dense::map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    
    for (auto _ : state) {
        map.clear();
        map.reserve(1000000);
        for (auto key : keys) {
            map[key] = key;
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_Ankerl_Insert_Timestamp(benchmark::State& state) {
    auto keys = GenerateTimestampKeys();
    ankerl::unordered_dense::map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    
    for (auto _ : state) {
        map.clear();
        map.reserve(1000000);
        for (auto key : keys) {
            map[key] = key;
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_RobinHood_Insert_Sequential(benchmark::State& state) {
    auto keys = GenerateSequentialKeys();
    robin_hood::unordered_flat_map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    
    for (auto _ : state) {
        map.clear();
        map.reserve(1000000);
        for (auto key : keys) {
            map[key] = key;
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_RobinHood_Insert_Random(benchmark::State& state) {
    auto keys = GenerateRandomKeys();
    robin_hood::unordered_flat_map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    
    for (auto _ : state) {
        map.clear();
        map.reserve(1000000);
        for (auto key : keys) {
            map[key] = key;
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_RobinHood_Insert_Bounded(benchmark::State& state) {
    auto keys = GenerateBoundedKeys();
    robin_hood::unordered_flat_map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    
    for (auto _ : state) {
        map.clear();
        map.reserve(1000000);
        for (auto key : keys) {
            map[key] = key;
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_RobinHood_Insert_Timestamp(benchmark::State& state) {
    auto keys = GenerateTimestampKeys();
    robin_hood::unordered_flat_map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    
    for (auto _ : state) {
        map.clear();
        map.reserve(1000000);
        for (auto key : keys) {
            map[key] = key;
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}


// Insert benchmarks (boost)
static void BM_Boost_Insert_Sequential(benchmark::State& state) {
    auto keys = GenerateSequentialKeys();
    boost::unordered_flat_map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    
    for (auto _ : state) {
        map.clear();
        map.reserve(1000000);
        for (auto key : keys) {
            map[key] = key;
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_Boost_Insert_Random(benchmark::State& state) {
    auto keys = GenerateRandomKeys();
    boost::unordered_flat_map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    
    for (auto _ : state) {
        map.clear();
        map.reserve(1000000);
        for (auto key : keys) {
            map[key] = key;
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_Boost_Insert_Bounded(benchmark::State& state) {
    auto keys = GenerateBoundedKeys();
    boost::unordered_flat_map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    
    for (auto _ : state) {
        map.clear();
        map.reserve(1000000);
        for (auto key : keys) {
            map[key] = key;
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_Boost_Insert_Timestamp(benchmark::State& state) {
    auto keys = GenerateTimestampKeys();
    boost::unordered_flat_map<uint64_t, uint64_t> map;
    map.reserve(1000000);
    
    for (auto _ : state) {
        map.clear();
        map.reserve(1000000);
        for (auto key : keys) {
            map[key] = key;
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

// String key benchmarks
constexpr size_t kNumStringKeys = 1000000;

std::vector<std::string> GenerateKeyStrings() {
    std::vector<std::string> strings;
    strings.reserve(kNumStringKeys);
    for (size_t i = 0; i < kNumStringKeys; ++i) {
        char buf[16];
        snprintf(buf, sizeof(buf), "key_%09zu", i);
        strings.push_back(std::string(buf));
    }
    return strings;
}

std::vector<std::string> GenerateMissStrings() {
    std::vector<std::string> strings;
    strings.reserve(kNumStringKeys);
    for (size_t i = 0; i < kNumStringKeys; ++i) {
        char buf[16];
        snprintf(buf, sizeof(buf), "mss_%09zu", i);
        strings.push_back(std::string(buf));
    }
    return strings;
}

static void BM_NSHash_String_Hit(benchmark::State& state) {
    auto strings = GenerateKeyStrings();
    std::vector<uint64_t> hashes;
    hashes.reserve(kNumStringKeys);
    for (const auto& s : strings) {
        hashes.push_back(nsh_hash_str(s.c_str(), s.length()));
    }
    
    NSHash<uint64_t, uint64_t> map;
    map.reserve(2097152);
    for (auto h : hashes) {
        map.insert(h, h);
    }
    
    for (auto _ : state) {
        for (auto h : hashes) {
            benchmark::DoNotOptimize(map.lookup(h));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumStringKeys);
}

static void BM_NSHash_String_Miss(benchmark::State& state) {
    auto strings = GenerateKeyStrings();
    std::vector<uint64_t> hashes;
    hashes.reserve(kNumStringKeys);
    for (const auto& s : strings) {
        hashes.push_back(nsh_hash_str(s.c_str(), s.length()));
    }
    
    NSHash<uint64_t, uint64_t> map;
    map.reserve(2097152);
    for (auto h : hashes) {
        map.insert(h, h);
    }
    
    auto miss_strings = GenerateMissStrings();
    std::vector<uint64_t> miss_hashes;
    miss_hashes.reserve(kNumStringKeys);
    for (const auto& s : miss_strings) {
        miss_hashes.push_back(nsh_hash_str(s.c_str(), s.length()));
    }
    
    for (auto _ : state) {
        for (auto h : miss_hashes) {
            benchmark::DoNotOptimize(map.lookup(h));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumStringKeys);
}

static void BM_Absl_String_Hit(benchmark::State& state) {
    auto strings = GenerateKeyStrings();
    absl::flat_hash_map<std::string, uint64_t> map;
    map.reserve(kNumStringKeys);
    for (const auto& s : strings) {
        map[s] = static_cast<uint64_t>(map.size());
    }
    
    for (auto _ : state) {
        for (const auto& s : strings) {
            benchmark::DoNotOptimize(map.find(s));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumStringKeys);
}

static void BM_Absl_String_Miss(benchmark::State& state) {
    auto strings = GenerateKeyStrings();
    absl::flat_hash_map<std::string, uint64_t> map;
    map.reserve(kNumStringKeys);
    for (const auto& s : strings) {
        map[s] = static_cast<uint64_t>(map.size());
    }
    
    auto miss_strings = GenerateMissStrings();
    for (auto _ : state) {
        for (const auto& s : miss_strings) {
            benchmark::DoNotOptimize(map.find(s));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumStringKeys);
}

static void BM_Ankerl_String_Hit(benchmark::State& state) {
    auto strings = GenerateKeyStrings();
    ankerl::unordered_dense::map<std::string, uint64_t> map;
    map.reserve(kNumStringKeys);
    for (const auto& s : strings) {
        map[s] = static_cast<uint64_t>(map.size());
    }
    
    for (auto _ : state) {
        for (const auto& s : strings) {
            benchmark::DoNotOptimize(map.find(s));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumStringKeys);
}

static void BM_Ankerl_String_Miss(benchmark::State& state) {
    auto strings = GenerateKeyStrings();
    ankerl::unordered_dense::map<std::string, uint64_t> map;
    map.reserve(kNumStringKeys);
    for (const auto& s : strings) {
        map[s] = static_cast<uint64_t>(map.size());
    }
    
    auto miss_strings = GenerateMissStrings();
    for (auto _ : state) {
        for (const auto& s : miss_strings) {
            benchmark::DoNotOptimize(map.find(s));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumStringKeys);
}

static void BM_RobinHood_String_Hit(benchmark::State& state) {
    auto strings = GenerateKeyStrings();
    robin_hood::unordered_flat_map<std::string, uint64_t> map;
    map.reserve(kNumStringKeys);
    for (const auto& s : strings) {
        map[s] = static_cast<uint64_t>(map.size());
    }
    
    for (auto _ : state) {
        for (const auto& s : strings) {
            benchmark::DoNotOptimize(map.find(s));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumStringKeys);
}

static void BM_RobinHood_String_Miss(benchmark::State& state) {
    auto strings = GenerateKeyStrings();
    robin_hood::unordered_flat_map<std::string, uint64_t> map;
    map.reserve(kNumStringKeys);
    for (const auto& s : strings) {
        map[s] = static_cast<uint64_t>(map.size());
    }
    
    auto miss_strings = GenerateMissStrings();
    for (auto _ : state) {
        for (const auto& s : miss_strings) {
            benchmark::DoNotOptimize(map.find(s));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumStringKeys);
}

static void BM_Std_String_Hit(benchmark::State& state) {
    auto strings = GenerateKeyStrings();
    std::unordered_map<std::string, uint64_t> map;
    map.reserve(kNumStringKeys);
    for (const auto& s : strings) {
        map[s] = static_cast<uint64_t>(map.size());
    }
    
    for (auto _ : state) {
        for (const auto& s : strings) {
            benchmark::DoNotOptimize(map.find(s));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumStringKeys);
}

static void BM_Std_String_Miss(benchmark::State& state) {
    auto strings = GenerateKeyStrings();
    std::unordered_map<std::string, uint64_t> map;
    map.reserve(kNumStringKeys);
    for (const auto& s : strings) {
        map[s] = static_cast<uint64_t>(map.size());
    }
    
    auto miss_strings = GenerateMissStrings();
    for (auto _ : state) {
        for (const auto& s : miss_strings) {
            benchmark::DoNotOptimize(map.find(s));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumStringKeys);
}

static void BM_Fph_String_Hit(benchmark::State& state) {
    auto strings = GenerateKeyStrings();
    fph::DynamicFphMap<std::string, uint64_t> map;
    for (const auto& s : strings) {
        map[s] = static_cast<uint64_t>(map.size());
    }
    
    for (auto _ : state) {
        for (const auto& s : strings) {
            benchmark::DoNotOptimize(map.find(s));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumStringKeys);
}

static void BM_Fph_String_Miss(benchmark::State& state) {
    auto strings = GenerateKeyStrings();
    fph::DynamicFphMap<std::string, uint64_t> map;
    for (const auto& s : strings) {
        map[s] = static_cast<uint64_t>(map.size());
    }
    
    auto miss_strings = GenerateMissStrings();
    for (auto _ : state) {
        for (const auto& s : miss_strings) {
            benchmark::DoNotOptimize(map.find(s));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumStringKeys);
}


static void BM_Boost_String_Hit(benchmark::State& state) {
    auto strings = GenerateKeyStrings();
    boost::unordered_flat_map<std::string, uint64_t> map;
    map.reserve(kNumStringKeys);
    for (const auto& s : strings) {
        map[s] = static_cast<uint64_t>(map.size());
    }
    
    for (auto _ : state) {
        for (const auto& s : strings) {
            benchmark::DoNotOptimize(map.find(s));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumStringKeys);
}

static void BM_Boost_String_Miss(benchmark::State& state) {
    auto strings = GenerateKeyStrings();
    boost::unordered_flat_map<std::string, uint64_t> map;
    map.reserve(kNumStringKeys);
    for (const auto& s : strings) {
        map[s] = static_cast<uint64_t>(map.size());
    }
    
    auto miss_strings = GenerateMissStrings();
    for (auto _ : state) {
        for (const auto& s : miss_strings) {
            benchmark::DoNotOptimize(map.find(s));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumStringKeys);
}

// Mixed workload benchmark (70% insert, 30% lookup)
constexpr size_t kMixedOps = 1000000;

static void BM_NSHash_Mixed(benchmark::State& state) {
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint64_t> key_dist(1, 10000000);
    std::uniform_real_distribution<double> op_dist(0.0, 1.0);
    
    for (auto _ : state) {
        NSHash<uint64_t, uint64_t> map;
        map.reserve(2097152);
        for (size_t i = 0; i < kMixedOps; ++i) {
            uint64_t key = key_dist(rng);
            if (op_dist(rng) < 0.7) {
                map.insert(key, key);
            } else {
                benchmark::DoNotOptimize(map.lookup(key));
            }
        }
    }
    state.SetItemsProcessed(state.iterations() * kMixedOps);
}

// NSHash with WyHash benchmarks
static void BM_NSHashWy_Random(benchmark::State& state) {
    auto keys = GenerateRandomKeys();
    NSHash<uint64_t, uint64_t, WyHash> map;
    map.reserve(2097152);
    for (auto key : keys) {
        map.insert(key, key);
    }
    
    for (auto _ : state) {
        for (auto key : keys) {
            benchmark::DoNotOptimize(map.lookup(key));
        }
    }
    state.SetItemsProcessed(state.iterations() * kNumKeys);
}

static void BM_NSHashWy_Mixed(benchmark::State& state) {
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint64_t> key_dist(1, 10000000);
    std::uniform_real_distribution<double> op_dist(0.0, 1.0);
    
    for (auto _ : state) {
        NSHash<uint64_t, uint64_t, WyHash> map;
        map.reserve(2097152);
        for (size_t i = 0; i < kMixedOps; ++i) {
            uint64_t key = key_dist(rng);
            if (op_dist(rng) < 0.7) {
                map.insert(key, key);
            } else {
                benchmark::DoNotOptimize(map.lookup(key));
            }
        }
    }
    state.SetItemsProcessed(state.iterations() * kMixedOps);
}

static void BM_Absl_Mixed(benchmark::State& state) {
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint64_t> key_dist(1, 10000000);
    std::uniform_real_distribution<double> op_dist(0.0, 1.0);
    
    for (auto _ : state) {
        absl::flat_hash_map<uint64_t, uint64_t> map;
        map.reserve(1000000);
        for (size_t i = 0; i < kMixedOps; ++i) {
            uint64_t key = key_dist(rng);
            if (op_dist(rng) < 0.7) {
                map[key] = key;
            } else {
                benchmark::DoNotOptimize(map.find(key));
            }
        }
    }
    state.SetItemsProcessed(state.iterations() * kMixedOps);
}

static void BM_Ankerl_Mixed(benchmark::State& state) {
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint64_t> key_dist(1, 10000000);
    std::uniform_real_distribution<double> op_dist(0.0, 1.0);
    
    for (auto _ : state) {
        ankerl::unordered_dense::map<uint64_t, uint64_t> map;
        map.reserve(1000000);
        for (size_t i = 0; i < kMixedOps; ++i) {
            uint64_t key = key_dist(rng);
            if (op_dist(rng) < 0.7) {
                map[key] = key;
            } else {
                benchmark::DoNotOptimize(map.find(key));
            }
        }
    }
    state.SetItemsProcessed(state.iterations() * kMixedOps);
}

static void BM_RobinHood_Mixed(benchmark::State& state) {
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint64_t> key_dist(1, 10000000);
    std::uniform_real_distribution<double> op_dist(0.0, 1.0);
    
    for (auto _ : state) {
        robin_hood::unordered_flat_map<uint64_t, uint64_t> map;
        map.reserve(1000000);
        for (size_t i = 0; i < kMixedOps; ++i) {
            uint64_t key = key_dist(rng);
            if (op_dist(rng) < 0.7) {
                map[key] = key;
            } else {
                benchmark::DoNotOptimize(map.find(key));
            }
        }
    }
    state.SetItemsProcessed(state.iterations() * kMixedOps);
}

static void BM_Std_Mixed(benchmark::State& state) {
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint64_t> key_dist(1, 10000000);
    std::uniform_real_distribution<double> op_dist(0.0, 1.0);
    
    for (auto _ : state) {
        std::unordered_map<uint64_t, uint64_t> map;
        map.reserve(1000000);
        for (size_t i = 0; i < kMixedOps; ++i) {
            uint64_t key = key_dist(rng);
            if (op_dist(rng) < 0.7) {
                map[key] = key;
            } else {
                benchmark::DoNotOptimize(map.find(key));
            }
        }
    }
    state.SetItemsProcessed(state.iterations() * kMixedOps);
}

static void BM_Fph_Mixed(benchmark::State& state) {
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint64_t> key_dist(1, 10000000);
    std::uniform_real_distribution<double> op_dist(0.0, 1.0);
    
    for (auto _ : state) {
        fph::DynamicFphMap<uint64_t, uint64_t> map;
        for (size_t i = 0; i < kMixedOps; ++i) {
            uint64_t key = key_dist(rng);
            if (op_dist(rng) < 0.7) {
                map[key] = key;
            } else {
                benchmark::DoNotOptimize(map.find(key));
            }
        }
    }
    state.SetItemsProcessed(state.iterations() * kMixedOps);
}

static void BM_Boost_Mixed(benchmark::State& state) {
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint64_t> key_dist(1, 10000000);
    std::uniform_real_distribution<double> op_dist(0.0, 1.0);
    
    for (auto _ : state) {
        boost::unordered_flat_map<uint64_t, uint64_t> map;
        map.reserve(1000000);
        for (size_t i = 0; i < kMixedOps; ++i) {
            uint64_t key = key_dist(rng);
            if (op_dist(rng) < 0.7) {
                map[key] = key;
            } else {
                benchmark::DoNotOptimize(map.find(key));
            }
        }
    }
    state.SetItemsProcessed(state.iterations() * kMixedOps);
}

BENCHMARK(BM_NSHash_Sequential)->Threads(1);
BENCHMARK(BM_NSHash_Random)->Threads(1);
BENCHMARK(BM_NSHash_Bounded)->Threads(1);
BENCHMARK(BM_NSHash_Timestamp)->Threads(1);

BENCHMARK(BM_Absl_Sequential)->Threads(1);
BENCHMARK(BM_Absl_Random)->Threads(1);
BENCHMARK(BM_Absl_Bounded)->Threads(1);
BENCHMARK(BM_Absl_Timestamp)->Threads(1);

BENCHMARK(BM_Ankerl_Sequential)->Threads(1);
BENCHMARK(BM_Ankerl_Random)->Threads(1);
BENCHMARK(BM_Ankerl_Bounded)->Threads(1);
BENCHMARK(BM_Ankerl_Timestamp)->Threads(1);

BENCHMARK(BM_RobinHood_Sequential)->Threads(1);
BENCHMARK(BM_RobinHood_Random)->Threads(1);
BENCHMARK(BM_RobinHood_Bounded)->Threads(1);
BENCHMARK(BM_RobinHood_Timestamp)->Threads(1);

BENCHMARK(BM_Std_Sequential)->Threads(1);
BENCHMARK(BM_Std_Random)->Threads(1);
BENCHMARK(BM_Std_Bounded)->Threads(1);
BENCHMARK(BM_Std_Timestamp)->Threads(1);

BENCHMARK(BM_Fph_Sequential)->Threads(1);
BENCHMARK(BM_Fph_Random)->Threads(1);
BENCHMARK(BM_Fph_Bounded)->Threads(1);
BENCHMARK(BM_Fph_Timestamp)->Threads(1);

BENCHMARK(BM_Boost_Sequential)->Threads(1);
BENCHMARK(BM_Boost_Random)->Threads(1);
BENCHMARK(BM_Boost_Bounded)->Threads(1);
BENCHMARK(BM_Boost_Timestamp)->Threads(1);

BENCHMARK(BM_NSHash_Miss_Sequential)->Threads(1);
BENCHMARK(BM_NSHash_Miss_Random)->Threads(1);
BENCHMARK(BM_NSHash_Miss_Timestamp)->Threads(1);

BENCHMARK(BM_Absl_Miss_Sequential)->Threads(1);
BENCHMARK(BM_Absl_Miss_Random)->Threads(1);
BENCHMARK(BM_Absl_Miss_Timestamp)->Threads(1);

BENCHMARK(BM_Ankerl_Miss_Sequential)->Threads(1);
BENCHMARK(BM_Ankerl_Miss_Random)->Threads(1);
BENCHMARK(BM_Ankerl_Miss_Timestamp)->Threads(1);

BENCHMARK(BM_RobinHood_Miss_Sequential)->Threads(1);
BENCHMARK(BM_RobinHood_Miss_Random)->Threads(1);
BENCHMARK(BM_RobinHood_Miss_Timestamp)->Threads(1);

BENCHMARK(BM_Std_Miss_Sequential)->Threads(1);
BENCHMARK(BM_Std_Miss_Random)->Threads(1);
BENCHMARK(BM_Std_Miss_Timestamp)->Threads(1);

BENCHMARK(BM_Fph_Miss_Sequential)->Threads(1);
BENCHMARK(BM_Fph_Miss_Random)->Threads(1);
BENCHMARK(BM_Fph_Miss_Timestamp)->Threads(1);

BENCHMARK(BM_Boost_Miss_Sequential)->Threads(1);
BENCHMARK(BM_Boost_Miss_Random)->Threads(1);
BENCHMARK(BM_Boost_Miss_Timestamp)->Threads(1);

BENCHMARK(BM_NSHash_String_Hit)->Threads(1);
BENCHMARK(BM_NSHash_String_Miss)->Threads(1);

BENCHMARK(BM_Absl_String_Hit)->Threads(1);
BENCHMARK(BM_Absl_String_Miss)->Threads(1);

BENCHMARK(BM_Ankerl_String_Hit)->Threads(1);
BENCHMARK(BM_Ankerl_String_Miss)->Threads(1);

BENCHMARK(BM_RobinHood_String_Hit)->Threads(1);
BENCHMARK(BM_RobinHood_String_Miss)->Threads(1);

BENCHMARK(BM_Std_String_Hit)->Threads(1);
BENCHMARK(BM_Std_String_Miss)->Threads(1);

BENCHMARK(BM_Fph_String_Hit)->Threads(1);
BENCHMARK(BM_Fph_String_Miss)->Threads(1);

BENCHMARK(BM_Boost_String_Hit)->Threads(1);
BENCHMARK(BM_Boost_String_Miss)->Threads(1);

BENCHMARK(BM_NSHash_Mixed)->Threads(1);
BENCHMARK(BM_NSHashWy_Random)->Threads(1);
BENCHMARK(BM_NSHashWy_Mixed)->Threads(1);
BENCHMARK(BM_Absl_Mixed)->Threads(1);
BENCHMARK(BM_Ankerl_Mixed)->Threads(1);
BENCHMARK(BM_RobinHood_Mixed)->Threads(1);
BENCHMARK(BM_Std_Mixed)->Threads(1);
BENCHMARK(BM_Fph_Mixed)->Threads(1);
BENCHMARK(BM_Boost_Mixed)->Threads(1);

BENCHMARK(BM_NSHash_Insert_Sequential)->Threads(1);
BENCHMARK(BM_NSHash_Insert_Random)->Threads(1);
BENCHMARK(BM_NSHash_Insert_Bounded)->Threads(1);
BENCHMARK(BM_NSHash_Insert_Timestamp)->Threads(1);

BENCHMARK(BM_Absl_Insert_Sequential)->Threads(1);
BENCHMARK(BM_Absl_Insert_Random)->Threads(1);
BENCHMARK(BM_Absl_Insert_Bounded)->Threads(1);
BENCHMARK(BM_Absl_Insert_Timestamp)->Threads(1);

BENCHMARK(BM_Ankerl_Insert_Sequential)->Threads(1);
BENCHMARK(BM_Ankerl_Insert_Random)->Threads(1);
BENCHMARK(BM_Ankerl_Insert_Bounded)->Threads(1);
BENCHMARK(BM_Ankerl_Insert_Timestamp)->Threads(1);

BENCHMARK(BM_RobinHood_Insert_Sequential)->Threads(1);
BENCHMARK(BM_RobinHood_Insert_Random)->Threads(1);
BENCHMARK(BM_RobinHood_Insert_Bounded)->Threads(1);
BENCHMARK(BM_RobinHood_Insert_Timestamp)->Threads(1);

BENCHMARK(BM_Std_Insert_Sequential)->Threads(1);
BENCHMARK(BM_Std_Insert_Random)->Threads(1);
BENCHMARK(BM_Std_Insert_Bounded)->Threads(1);
BENCHMARK(BM_Std_Insert_Timestamp)->Threads(1);

BENCHMARK(BM_Fph_Insert_Sequential)->Threads(1);
BENCHMARK(BM_Fph_Insert_Random)->Threads(1);
BENCHMARK(BM_Fph_Insert_Bounded)->Threads(1);
BENCHMARK(BM_Fph_Insert_Timestamp)->Threads(1);

BENCHMARK(BM_Boost_Insert_Sequential)->Threads(1);
BENCHMARK(BM_Boost_Insert_Random)->Threads(1);
BENCHMARK(BM_Boost_Insert_Bounded)->Threads(1);
BENCHMARK(BM_Boost_Insert_Timestamp)->Threads(1);

BENCHMARK_MAIN();

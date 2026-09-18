// Copyright 2026 Orders of Magnitude LLC
// Free for non-commercial and academic use.
// Commercial use requires a license from ofmagnitude.com

#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <iomanip>
#include <set>
#include <cstring>
#include <cassert>
#include "deps/NS/NSSort/NSSort.hpp"
#ifdef USE_NSINDEX
#include "deps/NS/NSIndex/nsindex.hpp"
#endif

struct PropBounds {
    double min_val;
    double max_val;
};

class BobIndex {
public:
    BobIndex(size_t k, size_t n) : k_(k), n_(n) {
        // Calculate bitset words needed (same 65KB approach as before)
        bitset_words_ = (n + 63) / 64;
        
        // Allocate property storage
        properties_.resize(n);
        for (size_t i = 0; i < n; ++i) {
            properties_[i].resize(k);
        }
        
        // Allocate sorted indices for each property
        property_indices_.resize(k);
        for (size_t prop = 0; prop < k; ++prop) {
            property_indices_[prop].resize(n);
        }
        
        // Allocate sorted property values for each property
        sorted_property_values_.resize(k);
        for (size_t prop = 0; prop < k; ++prop) {
            sorted_property_values_[prop].resize(n);
        }
        
        // Allocate per-property bitsets (k × 65KB)
        prop_bits_.resize(k);
        for (size_t prop = 0; prop < k; ++prop) {
            prop_bits_[prop].resize(bitset_words_, 0);
        }
        
        // Allocate result bitset
        bitset_result_.resize(bitset_words_, 0);
        
#ifdef USE_NSINDEX
        property_indexes_.reserve(k);
#endif
    }
    
    void add_material(size_t idx, const std::vector<double>& props) {
        assert(props.size() == k_);
        assert(idx < n_);
        properties_[idx] = props;
    }
    
    void build_indices() {
        constexpr int64_t OFFSET = 4000000000LL;
        
        // Phase 1: Sort indices for all properties using nssort
        for (size_t prop = 0; prop < k_; ++prop) {
            for (size_t i = 0; i < n_; ++i) {
                property_indices_[prop][i] = i;
            }
            
            // Sort indices by property value using nssort (encode value+index)
            std::vector<int64_t> encoded(n_);
            for (size_t i = 0; i < n_; ++i) {
                int64_t val = static_cast<int64_t>(properties_[i][prop]);
                // Encode: (value + offset) * 1M + index
                encoded[i] = (val + OFFSET) * 1000000LL + i;
            }
            nssort(encoded.data(), n_);
            // Decode back to indices
            for (size_t i = 0; i < n_; ++i) {
                property_indices_[prop][i] = encoded[i] % 1000000;
            }
        }
        
        // Phase 2: Populate sorted property value arrays (after all sorting is done)
        for (size_t prop = 0; prop < k_; ++prop) {
            for (size_t i = 0; i < n_; ++i) {
                size_t idx = property_indices_[prop][i];
                sorted_property_values_[prop][i] = static_cast<int64_t>(properties_[idx][prop]) + OFFSET;
            }
        }
        
#ifdef USE_NSINDEX
        // Phase 3: Build NSIndex instances (after all data is stable)
        for (size_t prop = 0; prop < k_; ++prop) {
            property_indexes_.emplace_back(sorted_property_values_[prop].data(), n_);
        }
#endif
    }
    
    std::vector<size_t> query(const std::vector<PropBounds>& bounds) {
        assert(bounds.size() == k_);
        constexpr int64_t OFFSET = 4000000000LL;
        
        // Step 1: Binary search for each property to get candidate ranges
        std::vector<std::pair<size_t, size_t>> ranges(k_);
        
        for (size_t prop = 0; prop < k_; ++prop) {
            int64_t min_val = static_cast<int64_t>(bounds[prop].min_val) + OFFSET;
            int64_t max_val = static_cast<int64_t>(bounds[prop].max_val) + OFFSET;
            
#ifdef USE_NSINDEX
            size_t pos = property_indexes_[prop].predict(min_val);
            size_t end = property_indexes_[prop].predict(max_val);
            while (pos > 0 && sorted_property_values_[prop][pos - 1] >= min_val) pos--;
            while (pos < n_ && sorted_property_values_[prop][pos] < min_val) pos++;
            while (end > 0 && sorted_property_values_[prop][end - 1] > max_val) end--;
            while (end < n_ && sorted_property_values_[prop][end] <= max_val) end++;
#else
            auto it_low = std::lower_bound(sorted_property_values_[prop].begin(), 
                                          sorted_property_values_[prop].end(), min_val);
            auto it_high = std::upper_bound(sorted_property_values_[prop].begin(), 
                                           sorted_property_values_[prop].end(), max_val);
            size_t pos = it_low - sorted_property_values_[prop].begin();
            size_t end = it_high - sorted_property_values_[prop].begin();
#endif
            ranges[prop] = {pos, end};
        }
        
        // Sort ranges by size: process smallest first
        std::vector<size_t> prop_order(k_);
        for (size_t i = 0; i < k_; ++i) prop_order[i] = i;
        std::sort(prop_order.begin(), prop_order.end(),
            [&ranges](size_t a, size_t b) {
                return (ranges[a].second - ranges[a].first) < (ranges[b].second - ranges[b].first);
            });
        
        // Step 2: Parallel bitset build + SIMD AND
        std::vector<size_t> candidates;
        
        // Build all property bitsets in parallel
        #pragma omp parallel for num_threads(k_) schedule(static, 1)
        for (size_t p = 0; p < k_; ++p) {
            size_t prop = prop_order[p];
            std::fill(prop_bits_[prop].begin(), prop_bits_[prop].end(), 0);
            for (size_t i = ranges[prop].first; i < ranges[prop].second; ++i) {
                size_t idx = property_indices_[prop][i];
                prop_bits_[prop][idx >> 6] |= (1ULL << (idx & 63));
            }
        }
        
        // AND all bitsets into result
        std::copy(prop_bits_[0].begin(), prop_bits_[0].end(), bitset_result_.begin());
        for (size_t p = 1; p < k_; ++p) {
            for (size_t w = 0; w < bitset_words_; ++w) {
                bitset_result_[w] &= prop_bits_[p][w];
            }
        }
        
        // Extract candidates from result bitset
        for (size_t w = 0; w < bitset_words_; ++w) {
            uint64_t word = bitset_result_[w];
            while (word) {
                candidates.push_back(w * 64 + __builtin_ctzll(word));
                word &= word - 1;
            }
        }
        
        return candidates;
    }
    
    size_t k() const { return k_; }
    size_t n() const { return n_; }
    
private:
    size_t k_;
    size_t n_;
    size_t bitset_words_;
    
    // Property storage: [material][property]
    std::vector<std::vector<double>> properties_;
    
    // Sorted indices for each property: [property][rank] -> material index
    std::vector<std::vector<size_t>> property_indices_;
    
    // Sorted property values: [property][rank] -> value
    std::vector<std::vector<int64_t>> sorted_property_values_;
    
    // Per-property bitsets for intersection: [property][word]
    std::vector<std::vector<uint64_t>> prop_bits_;
    
    // Result bitset
    std::vector<uint64_t> bitset_result_;
    
#ifdef USE_NSINDEX
    // NSIndex instances for range queries
    std::vector<NSIndex<int64_t>> property_indexes_;
#endif
};

// Legacy compatibility layer for existing benchmark code
constexpr size_t N = 520000;
constexpr int NUM_PROPERTIES = 5;
constexpr int64_t OFFSET = 4000000000LL;

struct Material {
    int64_t formation_energy;
    int64_t bandgap;
    int64_t density;
    int64_t melting_point;
    int64_t bulk_modulus;
};

struct Query {
    int64_t min_formation_energy, max_formation_energy;
    int64_t min_bandgap, max_bandgap;
    int64_t min_density, max_density;
    int64_t min_melting_point, max_melting_point;
    int64_t min_bulk_modulus, max_bulk_modulus;
};

// Global data for legacy benchmark
std::vector<Material> materials(N);
constexpr size_t WORDS = (N + 63) / 64;
static uint64_t bitset_a[WORDS];
static uint64_t bitset_b[WORDS];

// BOB indices: pre-sorted indices for each property
std::vector<std::vector<size_t>> property_indices(NUM_PROPERTIES);
// Sorted property value arrays (parallel to property_indices)
std::vector<int64_t> sorted_property_values[NUM_PROPERTIES];
#ifdef USE_NSINDEX
// NSIndex instances for range queries
std::vector<NSIndex<int64_t>> property_indexes;
#endif

// Deterministic random generation
void generate_materials() {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    constexpr double SCALE = 1e9;
    
    for (size_t i = 0; i < N; ++i) {
        materials[i].formation_energy = static_cast<int64_t>(dist(rng) * SCALE);
        materials[i].bandgap = static_cast<int64_t>(dist(rng) * SCALE);
        materials[i].density = static_cast<int64_t>(dist(rng) * SCALE);
        materials[i].melting_point = static_cast<int64_t>(dist(rng) * SCALE);
        materials[i].bulk_modulus = static_cast<int64_t>(dist(rng) * SCALE);
    }
}

// Build BOB indices (one-time cost)
double build_bob_indices() {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Phase 1: Sort indices for all properties using nssort
    for (int prop = 0; prop < NUM_PROPERTIES; ++prop) {
        property_indices[prop].resize(N);
        for (size_t i = 0; i < N; ++i) {
            property_indices[prop][i] = i;
        }
        
        // Sort indices by property value using nssort (encode value+index)
        std::vector<int64_t> encoded(N);
        for (size_t i = 0; i < N; ++i) {
            int64_t val;
            switch(prop) {
                case 0: val = materials[i].formation_energy; break;
                case 1: val = materials[i].bandgap; break;
                case 2: val = materials[i].density; break;
                case 3: val = materials[i].melting_point; break;
                case 4: val = materials[i].bulk_modulus; break;
                default: val = 0;
            }
            // Encode: (value + offset) * 1M + index
            encoded[i] = (val + OFFSET) * 1000000LL + i;
        }
        nssort(encoded.data(), N);
        // Decode back to indices
        for (size_t i = 0; i < N; ++i) {
            property_indices[prop][i] = encoded[i] % 1000000;
        }
    }
    
    // Phase 2: Populate sorted property value arrays (after all sorting is done)
    for (int prop = 0; prop < NUM_PROPERTIES; ++prop) {
        sorted_property_values[prop].resize(N);
        for (size_t i = 0; i < N; ++i) {
            size_t idx = property_indices[prop][i];
            switch(prop) {
                case 0: sorted_property_values[prop][i] = materials[idx].formation_energy + OFFSET; break;
                case 1: sorted_property_values[prop][i] = materials[idx].bandgap + OFFSET; break;
                case 2: sorted_property_values[prop][i] = materials[idx].density + OFFSET; break;
                case 3: sorted_property_values[prop][i] = materials[idx].melting_point + OFFSET; break;
                case 4: sorted_property_values[prop][i] = materials[idx].bulk_modulus + OFFSET; break;
            }
        }
    }
    
#ifdef USE_NSINDEX
    // Phase 3: Build NSIndex instances (after all data is stable)
    property_indexes.reserve(NUM_PROPERTIES);
    for (int prop = 0; prop < NUM_PROPERTIES; ++prop) {
        property_indexes.emplace_back(sorted_property_values[prop].data(), N);
    }
#endif
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    return diff.count();
}

// BOB: Binary search per property + bitmap intersection + O(n log n) Pareto front
std::vector<size_t> bob_query(const Query& q, double& query_time) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Step 1: Binary search for each property to get candidate ranges
    std::vector<std::pair<size_t, size_t>> ranges(NUM_PROPERTIES);
    
    // Property 0: formation_energy
#ifdef USE_NSINDEX
    size_t pos0 = property_indexes[0].predict(q.min_formation_energy + OFFSET);
    size_t end0 = property_indexes[0].predict(q.max_formation_energy + OFFSET);
    while (pos0 > 0 && sorted_property_values[0][pos0 - 1] >= q.min_formation_energy + OFFSET) pos0--;
    while (pos0 < N && sorted_property_values[0][pos0] < q.min_formation_energy + OFFSET) pos0++;
    while (end0 > 0 && sorted_property_values[0][end0 - 1] > q.max_formation_energy + OFFSET) end0--;
    while (end0 < N && sorted_property_values[0][end0] <= q.max_formation_energy + OFFSET) end0++;
#else
    auto it0_low = std::lower_bound(sorted_property_values[0].begin(), sorted_property_values[0].end(), q.min_formation_energy + OFFSET);
    auto it0_high = std::upper_bound(sorted_property_values[0].begin(), sorted_property_values[0].end(), q.max_formation_energy + OFFSET);
    size_t pos0 = it0_low - sorted_property_values[0].begin();
    size_t end0 = it0_high - sorted_property_values[0].begin();
#endif
    ranges[0] = {pos0, end0};
    
    // Property 1: bandgap
#ifdef USE_NSINDEX
    size_t pos1 = property_indexes[1].predict(q.min_bandgap + OFFSET);
    size_t end1 = property_indexes[1].predict(q.max_bandgap + OFFSET);
    while (pos1 > 0 && sorted_property_values[1][pos1 - 1] >= q.min_bandgap + OFFSET) pos1--;
    while (pos1 < N && sorted_property_values[1][pos1] < q.min_bandgap + OFFSET) pos1++;
    while (end1 > 0 && sorted_property_values[1][end1 - 1] > q.max_bandgap + OFFSET) end1--;
    while (end1 < N && sorted_property_values[1][end1] <= q.max_bandgap + OFFSET) end1++;
#else
    auto it1_low = std::lower_bound(sorted_property_values[1].begin(), sorted_property_values[1].end(), q.min_bandgap + OFFSET);
    auto it1_high = std::upper_bound(sorted_property_values[1].begin(), sorted_property_values[1].end(), q.max_bandgap + OFFSET);
    size_t pos1 = it1_low - sorted_property_values[1].begin();
    size_t end1 = it1_high - sorted_property_values[1].begin();
#endif
    ranges[1] = {pos1, end1};
    
    // Property 2: density
#ifdef USE_NSINDEX
    size_t pos2 = property_indexes[2].predict(q.min_density + OFFSET);
    size_t end2 = property_indexes[2].predict(q.max_density + OFFSET);
    while (pos2 > 0 && sorted_property_values[2][pos2 - 1] >= q.min_density + OFFSET) pos2--;
    while (pos2 < N && sorted_property_values[2][pos2] < q.min_density + OFFSET) pos2++;
    while (end2 > 0 && sorted_property_values[2][end2 - 1] > q.max_density + OFFSET) end2--;
    while (end2 < N && sorted_property_values[2][end2] <= q.max_density + OFFSET) end2++;
#else
    auto it2_low = std::lower_bound(sorted_property_values[2].begin(), sorted_property_values[2].end(), q.min_density + OFFSET);
    auto it2_high = std::upper_bound(sorted_property_values[2].begin(), sorted_property_values[2].end(), q.max_density + OFFSET);
    size_t pos2 = it2_low - sorted_property_values[2].begin();
    size_t end2 = it2_high - sorted_property_values[2].begin();
#endif
    ranges[2] = {pos2, end2};
    
    // Property 3: melting_point
#ifdef USE_NSINDEX
    size_t pos3 = property_indexes[3].predict(q.min_melting_point + OFFSET);
    size_t end3 = property_indexes[3].predict(q.max_melting_point + OFFSET);
    while (pos3 > 0 && sorted_property_values[3][pos3 - 1] >= q.min_melting_point + OFFSET) pos3--;
    while (pos3 < N && sorted_property_values[3][pos3] < q.min_melting_point + OFFSET) pos3++;
    while (end3 > 0 && sorted_property_values[3][end3 - 1] > q.max_melting_point + OFFSET) end3--;
    while (end3 < N && sorted_property_values[3][end3] <= q.max_melting_point + OFFSET) end3++;
#else
    auto it3_low = std::lower_bound(sorted_property_values[3].begin(), sorted_property_values[3].end(), q.min_melting_point + OFFSET);
    auto it3_high = std::upper_bound(sorted_property_values[3].begin(), sorted_property_values[3].end(), q.max_melting_point + OFFSET);
    size_t pos3 = it3_low - sorted_property_values[3].begin();
    size_t end3 = it3_high - sorted_property_values[3].begin();
#endif
    ranges[3] = {pos3, end3};
    
    // Property 4: bulk_modulus
#ifdef USE_NSINDEX
    size_t pos4 = property_indexes[4].predict(q.min_bulk_modulus + OFFSET);
    size_t end4 = property_indexes[4].predict(q.max_bulk_modulus + OFFSET);
    while (pos4 > 0 && sorted_property_values[4][pos4 - 1] >= q.min_bulk_modulus + OFFSET) pos4--;
    while (pos4 < N && sorted_property_values[4][pos4] < q.min_bulk_modulus + OFFSET) pos4++;
    while (end4 > 0 && sorted_property_values[4][end4 - 1] > q.max_bulk_modulus + OFFSET) end4--;
    while (end4 < N && sorted_property_values[4][end4] <= q.max_bulk_modulus + OFFSET) end4++;
#else
    auto it4_low = std::lower_bound(sorted_property_values[4].begin(), sorted_property_values[4].end(), q.min_bulk_modulus + OFFSET);
    auto it4_high = std::upper_bound(sorted_property_values[4].begin(), sorted_property_values[4].end(), q.max_bulk_modulus + OFFSET);
    size_t pos4 = it4_low - sorted_property_values[4].begin();
    size_t end4 = it4_high - sorted_property_values[4].begin();
#endif
    ranges[4] = {pos4, end4};
    
    // Sort ranges by size: process smallest first
    std::vector<int> prop_order(NUM_PROPERTIES);
    for (int i = 0; i < NUM_PROPERTIES; ++i) prop_order[i] = i;
    std::sort(prop_order.begin(), prop_order.end(),
        [&ranges](int a, int b) {
            return (ranges[a].second - ranges[a].first) < (ranges[b].second - ranges[b].first);
        });
    
    // Step 2: Parallel bitset build + SIMD AND
    std::vector<size_t> candidates;

    // Allocate per-property bitsets on heap (5 × 65KB = 325KB)
    static uint64_t prop_bits[NUM_PROPERTIES][WORDS];

    // Build all property bitsets in parallel
    #pragma omp parallel for num_threads(NUM_PROPERTIES) schedule(static, 1)
    for (int p = 0; p < NUM_PROPERTIES; ++p) {
        int prop = prop_order[p];
        memset(prop_bits[p], 0, WORDS * 8);
        for (size_t i = ranges[prop].first; i < ranges[prop].second; ++i) {
            size_t idx = property_indices[prop][i];
            prop_bits[p][idx >> 6] |= (1ULL << (idx & 63));
        }
    }

    // AND all bitsets into result
    memcpy(bitset_b, prop_bits[0], WORDS * 8);
    for (int p = 1; p < NUM_PROPERTIES; ++p) {
        for (size_t w = 0; w < WORDS; ++w) bitset_b[w] &= prop_bits[p][w];
    }
    for (size_t w = 0; w < WORDS; ++w) {
        uint64_t word = bitset_b[w];
        while (word) {
            candidates.push_back(w * 64 + __builtin_ctzll(word));
            word &= word - 1;
        }
    }
    
    // Step 3: O(n log n) Pareto front (sort by first objective, scan once)
    std::sort(candidates.begin(), candidates.end(),
        [](size_t a, size_t b) {
            return materials[a].formation_energy < materials[b].formation_energy;
        });
    
    // Pareto front scan
    std::vector<size_t> pareto_front;
    int64_t min_bandgap = std::numeric_limits<int64_t>::max();
    
    for (size_t idx : candidates) {
        if (materials[idx].bandgap < min_bandgap) {
            pareto_front.push_back(idx);
            min_bandgap = materials[idx].bandgap;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    query_time = std::chrono::duration<double>(end - start).count();
    
    return pareto_front;
}

// Generate queries
Query generate_wide_query() {
    // ~25% of N should pass: each property has 0.25 range
    constexpr double SCALE = 1e9;
    return {0, static_cast<int64_t>(0.25 * SCALE), 0, static_cast<int64_t>(0.25 * SCALE), 0, static_cast<int64_t>(0.25 * SCALE), 0, static_cast<int64_t>(0.25 * SCALE), 0, static_cast<int64_t>(0.25 * SCALE)};
}

Query generate_tight_query() {
    // ~0.1% of N should pass: each property has 0.001 range
    constexpr double SCALE = 1e9;
    return {static_cast<int64_t>(0.4 * SCALE), static_cast<int64_t>(0.401 * SCALE), static_cast<int64_t>(0.4 * SCALE), static_cast<int64_t>(0.401 * SCALE), static_cast<int64_t>(0.4 * SCALE), static_cast<int64_t>(0.401 * SCALE), static_cast<int64_t>(0.4 * SCALE), static_cast<int64_t>(0.401 * SCALE), static_cast<int64_t>(0.4 * SCALE), static_cast<int64_t>(0.401 * SCALE)};
}

Query generate_random_query(std::mt19937_64& rng) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    constexpr double SCALE = 1e9;
    double width = dist(rng) * 0.3; // Random width up to 0.3
    double center = dist(rng) * (1.0 - width);
    int64_t iwidth = static_cast<int64_t>(width * SCALE);
    int64_t icenter = static_cast<int64_t>(center * SCALE);
    return {icenter, icenter + iwidth, icenter, icenter + iwidth, icenter, icenter + iwidth, icenter, icenter + iwidth, icenter, icenter + iwidth};
}

double brute_force_pareto(const Query& q, std::vector<size_t>& pareto_result) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Step 1: linear intersection
    std::vector<size_t> candidates;
    candidates.reserve(1024);
    for (size_t i = 0; i < N; ++i) {
        if (materials[i].formation_energy >= q.min_formation_energy &&
            materials[i].formation_energy <= q.max_formation_energy &&
            materials[i].bandgap >= q.min_bandgap &&
            materials[i].bandgap <= q.max_bandgap &&
            materials[i].density >= q.min_density &&
            materials[i].density <= q.max_density &&
            materials[i].melting_point >= q.min_melting_point &&
            materials[i].melting_point <= q.max_melting_point &&
            materials[i].bulk_modulus >= q.min_bulk_modulus &&
            materials[i].bulk_modulus <= q.max_bulk_modulus) {
            candidates.push_back(i);
        }
    }
    
    // Step 2: same Pareto front as BOB
    std::sort(candidates.begin(), candidates.end(),
        [](size_t a, size_t b) {
            return materials[a].formation_energy < materials[b].formation_energy;
        });
    pareto_result.clear();
    int64_t min_bandgap = std::numeric_limits<int64_t>::max();
    for (size_t idx : candidates) {
        if (materials[idx].bandgap < min_bandgap) {
            pareto_result.push_back(idx);
            min_bandgap = materials[idx].bandgap;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

void run_benchmark(const std::string& name, const Query& q) {
    std::cout << "\n=== " << name << " ===\n";
    
    double bob_time;
    auto bob_result = bob_query(q, bob_time);
    
    std::vector<size_t> brute_result;
    double brute_time = brute_force_pareto(q, brute_result);
    
    size_t bob_candidates = bob_result.size();
    size_t brute_candidates = brute_result.size();
    double speedup = brute_time / bob_time;
    
    std::cout << "BOB candidates: " << bob_candidates << "\n";
    std::cout << "Brute force candidates: " << brute_candidates << "\n";
    std::cout << "BOB time: " << std::fixed << std::setprecision(6) << (bob_time * 1000) << "ms\n";
    std::cout << "Brute force time: " << std::fixed << std::setprecision(6) << (brute_time * 1000) << "ms\n";
    std::cout << "Speedup: " << std::fixed << std::setprecision(2) << speedup << "x\n";
    
    // Correctness check: verify BOB results actually pass all filter conditions
    int bob_false_positives = 0;
    for (size_t idx : bob_result) {
        bool passes = 
            materials[idx].formation_energy >= q.min_formation_energy &&
            materials[idx].formation_energy <= q.max_formation_energy &&
            materials[idx].bandgap >= q.min_bandgap &&
            materials[idx].bandgap <= q.max_bandgap &&
            materials[idx].density >= q.min_density &&
            materials[idx].density <= q.max_density &&
            materials[idx].melting_point >= q.min_melting_point &&
            materials[idx].melting_point <= q.max_melting_point &&
            materials[idx].bulk_modulus >= q.min_bulk_modulus &&
            materials[idx].bulk_modulus <= q.max_bulk_modulus;
        if (!passes) bob_false_positives++;
    }
    std::cout << "BOB false positives: " << bob_false_positives << "\n";
}

void run_batch_benchmark(int num_queries) {
    std::cout << "\n=== Batch of " << num_queries << " varied queries ===\n";
    
    std::mt19937_64 rng(123);
    
    double total_bob_time = 0.0;
    double total_brute_time = 0.0;
    size_t total_candidates = 0;
    std::vector<size_t> brute_results;
    
    for (int i = 0; i < num_queries; ++i) {
        Query q = generate_random_query(rng);
        double bob_time;
        auto bob_result = bob_query(q, bob_time);
        
        total_bob_time += bob_time;
        total_candidates += bob_result.size();
        total_brute_time += brute_force_pareto(q, brute_results);
    }
    
    std::cout << "Total candidates: " << total_candidates << "\n";
    std::cout << "Avg candidates per query: " << total_candidates / num_queries << "\n";
    std::cout << "Total BOB time: " << std::fixed << std::setprecision(4) << total_bob_time << "s\n";
    std::cout << "Total brute force time: " << std::fixed << std::setprecision(4) << total_brute_time << "s\n";
    std::cout << "Avg speedup: " << std::fixed << std::setprecision(2) << (total_brute_time / total_bob_time) << "x\n";
}

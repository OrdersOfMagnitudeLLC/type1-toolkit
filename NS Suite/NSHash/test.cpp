// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#include "nshash.hpp"
#include <iostream>
#include <random>
#include <cstdint>

bool test1_sequential_1M() {
    NSHash<uint64_t, uint64_t> map;
    
    // Insert keys 1..1,000,000 with value = key * 2
    for (uint64_t i = 1; i <= 1000000; ++i) {
        map.insert(i, i * 2);
    }
    
    // Verify lookup(k) == k*2 for all k
    for (uint64_t i = 1; i <= 1000000; ++i) {
        uint64_t* val = map.lookup(i);
        if (val == nullptr || *val != i * 2) {
            std::cout << "FAIL: Test 1 - lookup(" << i << ") returned wrong value" << std::endl;
            return false;
        }
    }
    
    // Verify lookup(0) returns nullptr (not found)
    if (map.lookup(0) != nullptr) {
        std::cout << "FAIL: Test 1 - lookup(0) should return nullptr" << std::endl;
        return false;
    }
    
    std::cout << "PASS: Test 1 - Sequential 1M insert + lookup" << std::endl;
    return true;
}

bool test2_erase_correctness() {
    NSHash<uint64_t, uint64_t> map;
    
    // Insert keys 1..1000
    for (uint64_t i = 1; i <= 1000; ++i) {
        map.insert(i, i);
    }
    
    // Erase all even keys
    for (uint64_t i = 2; i <= 1000; i += 2) {
        map.erase(i);
    }
    
    // Verify even keys not found
    for (uint64_t i = 2; i <= 1000; i += 2) {
        if (map.lookup(i) != nullptr) {
            std::cout << "FAIL: Test 2 - even key " << i << " should not be found after erase" << std::endl;
            return false;
        }
    }
    
    // Verify odd keys still found
    for (uint64_t i = 1; i <= 1000; i += 2) {
        uint64_t* val = map.lookup(i);
        if (val == nullptr || *val != i) {
            std::cout << "FAIL: Test 2 - odd key " << i << " should still be found" << std::endl;
            return false;
        }
    }
    
    std::cout << "PASS: Test 2 - Erase correctness" << std::endl;
    return true;
}

bool test3_resize_trigger() {
    NSHash<uint64_t, uint64_t> map;
    
    // Insert 100,000 keys
    for (uint64_t i = 1; i <= 100000; ++i) {
        map.insert(i, i * 3);
    }
    
    // Check all 100,000 keys still found after inserts complete
    for (uint64_t i = 1; i <= 100000; ++i) {
        uint64_t* val = map.lookup(i);
        if (val == nullptr || *val != i * 3) {
            std::cout << "FAIL: Test 3 - key " << i << " not found or wrong value after resize" << std::endl;
            return false;
        }
    }
    
    std::cout << "PASS: Test 3 - Resize trigger" << std::endl;
    return true;
}

bool test4_large_n() {
    NSHash<uint64_t, uint64_t> map;
    
    // Insert 5,000,000 sequential keys
    for (uint64_t i = 1; i <= 5000000; ++i) {
        map.insert(i, i * 5);
    }
    
    // Verify 1000 random lookups correct
    std::mt19937_64 rng(123);
    std::uniform_int_distribution<uint64_t> dist(1, 5000000);
    
    for (int i = 0; i < 1000; ++i) {
        uint64_t key = dist(rng);
        uint64_t* val = map.lookup(key);
        if (val == nullptr || *val != key * 5) {
            std::cout << "FAIL: Test 4 - random lookup for key " << key << " failed" << std::endl;
            return false;
        }
    }
    
    std::cout << "PASS: Test 4 - Large n" << std::endl;
    return true;
}

bool test5_benchmark_erase() {
    NSHash<uint64_t, uint64_t> map;
    
    // Insert 100,000 keys
    for (uint64_t i = 1; i <= 100000; ++i) {
        map.insert(i, i * 2);
    }
    
    // Erase all even keys
    for (uint64_t i = 2; i <= 100000; i += 2) {
        map.erase(i);
    }
    
    // Verify even keys not found
    for (uint64_t i = 2; i <= 100000; i += 2) {
        if (map.lookup(i) != nullptr) {
            std::cout << "FAIL: Test 5 - even key " << i << " should not be found after erase" << std::endl;
            return false;
        }
    }
    
    // Verify odd keys still found
    for (uint64_t i = 1; i <= 100000; i += 2) {
        uint64_t* val = map.lookup(i);
        if (val == nullptr || *val != i * 2) {
            std::cout << "FAIL: Test 5 - odd key " << i << " should still be found" << std::endl;
            return false;
        }
    }
    
    std::cout << "PASS: Test 5 - Benchmark erase (100K insert, erase even, verify)" << std::endl;
    return true;
}

int main() {
    bool all_passed = true;
    
    all_passed &= test1_sequential_1M();
    all_passed &= test2_erase_correctness();
    all_passed &= test3_resize_trigger();
    all_passed &= test4_large_n();
    all_passed &= test5_benchmark_erase();
    
    if (all_passed) {
        std::cout << "\nAll tests PASSED" << std::endl;
        return 0;
    } else {
        std::cout << "\nSome tests FAILED" << std::endl;
        return 1;
    }
}

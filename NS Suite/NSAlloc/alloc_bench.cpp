// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#include <cstdint>
#include <cstring>
#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <mimalloc.h>
#include "nsalloc.hpp"

// Benchmark patterns
constexpr size_t NUM_OPS = 100000;
size_t sizes[] = {8, 16, 32, 64, 128};

void pattern_d_bulk_reset() {
    std::cout << "Pattern D - Bulk Reset:" << std::endl;
    
    // NSMultiPool
    {
        NSMultiPool alloc;
        std::vector<void*> ptrs;
        std::vector<size_t> alloc_sizes;
        ptrs.reserve(NUM_OPS);
        alloc_sizes.reserve(NUM_OPS);
        
        // Allocate 100K blocks
        for (size_t i = 0; i < NUM_OPS; ++i) {
            size_t size = sizes[i % 5];
            void* p = alloc.alloc(size);
            if (!p) {
                std::cout << "OOM at iteration " << i << std::endl;
                break;
            }
            ptrs.push_back(p);
            alloc_sizes.push_back(size);
        }
        
        // Reset 1000x
        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < 1000; ++iter) {
            alloc.reset();
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        double ns_per_op = std::chrono::duration<double, std::nano>(end - start).count() / 1000;
        std::cout << "  NSMultiPool: " << ns_per_op << " ns/op" << std::endl;
    }
    
    // mi_heap (mimalloc arena)
    {
        std::vector<void*> ptrs;
        std::vector<size_t> alloc_sizes;
        ptrs.reserve(NUM_OPS);
        alloc_sizes.reserve(NUM_OPS);
        
        // Allocate 100K blocks
        mi_heap_t* heap = mi_heap_new();
        for (size_t i = 0; i < NUM_OPS; ++i) {
            size_t size = sizes[i % 5];
            void* p = mi_heap_malloc(heap, size);
            if (!p) {
                std::cout << "OOM at iteration " << i << std::endl;
                break;
            }
            ptrs.push_back(p);
            alloc_sizes.push_back(size);
        }
        
        // Reset 1000x (destroy + new)
        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < 1000; ++iter) {
            mi_heap_destroy(heap);
            heap = mi_heap_new();
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        double ns_per_op = std::chrono::duration<double, std::nano>(end - start).count() / 1000;
        std::cout << "  mi_heap (mimalloc arena): " << ns_per_op << " ns/op" << std::endl;
        mi_heap_destroy(heap);
    }
}

void pattern_e_fixed_size_hot_path() {
    std::cout << "Pattern E - Fixed Size Hot Path:" << std::endl;
    
    // NSPool<64, 64>
    {
        NSPool<64, 64> pool;
        std::vector<void*> ptrs;
        ptrs.reserve(64);
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < 100000; ++iter) {
            // Alloc 64 objects
            for (size_t i = 0; i < 64; ++i) {
                void* p = pool.alloc();
                if (!p) {
                    std::cout << "OOM at iteration " << iter << std::endl;
                    break;
                }
                ptrs.push_back(p);
            }
            // Free all
            for (size_t i = 0; i < 64; ++i) {
                pool.free(ptrs[i]);
            }
            ptrs.clear();
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        double ns_per_op = std::chrono::duration<double, std::nano>(end - start).count() / (100000 * 64 * 2);
        std::cout << "  NSPool<64, 64>: " << ns_per_op << " ns/op" << std::endl;
    }
    
    // NSPool<64, 128>
    {
        NSPool<64, 128> pool;
        std::vector<void*> ptrs;
        ptrs.reserve(128);
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < 100000; ++iter) {
            // Alloc 128 objects
            for (size_t i = 0; i < 128; ++i) {
                void* p = pool.alloc();
                if (!p) {
                    std::cout << "OOM at iteration " << iter << std::endl;
                    break;
                }
                ptrs.push_back(p);
            }
            // Free all
            for (size_t i = 0; i < 128; ++i) {
                pool.free(ptrs[i]);
            }
            ptrs.clear();
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        double ns_per_op = std::chrono::duration<double, std::nano>(end - start).count() / (100000 * 128 * 2);
        std::cout << "  NSPool<64, 128>: " << ns_per_op << " ns/op" << std::endl;
    }
    
    // NSPool<64, 256>
    {
        NSPool<64, 256> pool;
        std::vector<void*> ptrs;
        ptrs.reserve(256);
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < 100000; ++iter) {
            // Alloc 256 objects
            for (size_t i = 0; i < 256; ++i) {
                void* p = pool.alloc();
                if (!p) {
                    std::cout << "OOM at iteration " << iter << std::endl;
                    break;
                }
                ptrs.push_back(p);
            }
            // Free all
            for (size_t i = 0; i < 256; ++i) {
                pool.free(ptrs[i]);
            }
            ptrs.clear();
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        double ns_per_op = std::chrono::duration<double, std::nano>(end - start).count() / (100000 * 256 * 2);
        std::cout << "  NSPool<64, 256>: " << ns_per_op << " ns/op" << std::endl;
    }
    
    // mi_heap (mimalloc arena)
    {
        mi_heap_t* heap = mi_heap_new();
        std::vector<void*> ptrs;
        ptrs.reserve(64);
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < 100000; ++iter) {
            // Alloc 64 objects
            for (size_t i = 0; i < 64; ++i) {
                void* p = mi_heap_malloc(heap, 64);
                if (!p) {
                    std::cout << "OOM at iteration " << iter << std::endl;
                    break;
                }
                ptrs.push_back(p);
            }
            // Free all
            for (size_t i = 0; i < 64; ++i) {
                mi_free(ptrs[i]);
            }
            ptrs.clear();
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        double ns_per_op = std::chrono::duration<double, std::nano>(end - start).count() / (100000 * 64 * 2);
        std::cout << "  mi_heap (mimalloc arena): " << ns_per_op << " ns/op" << std::endl;
        mi_heap_destroy(heap);
    }
}

int main() {
    pattern_d_bulk_reset();
    pattern_e_fixed_size_hot_path();
    return 0;
}

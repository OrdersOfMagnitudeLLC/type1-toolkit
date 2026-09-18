// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#pragma once

#include <atomic>
#include <thread>
#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

class NSRWLock {
private:
    static constexpr int NUM_SLOTS = 1024; // One byte per thread (thread ID 0-1023)
    std::atomic<uint8_t> reader_slots[NUM_SLOTS];
    std::atomic<bool> writer_active{false};
    std::atomic<bool> writer_pending{false};

public:
    NSRWLock() {
        for (int i = 0; i < NUM_SLOTS; i++) {
            reader_slots[i].store(0);
        }
    }

    void lock_read(int thread_id) {
        int tid = thread_id % NUM_SLOTS;
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
        reader_slots[thread_id % NUM_SLOTS].store(0, std::memory_order_release);
    }

    void lock_write() {
        writer_pending.store(true, std::memory_order_release);
        
        // Wait until all reader slots are 0
        while (true) {
#if defined(__AVX2__) || defined(__AVX512F__)
            // Load NUM_SLOTS bytes (NUM_SLOTS/32 256-bit loads)
            __m256i zero = _mm256_setzero_si256();
            __m256i combined = _mm256_set1_epi32(-1);
            for (int i = 0; i < NUM_SLOTS / 32; i++) {
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
                    for (int i = 0; i < NUM_SLOTS / 32; i++) {
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
            // Scalar fallback: check all NUM_SLOTS reader slots
            bool all_zero = true;
            for (int i = 0; i < NUM_SLOTS; i++) {
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
                    for (int i = 0; i < NUM_SLOTS; i++) {
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

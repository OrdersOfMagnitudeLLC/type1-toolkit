// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#pragma once

#include <cstdint>
#include <cstring>

template <size_t ObjSize, size_t Capacity>
class NSPool {
    static_assert(Capacity % 64 == 0, "Capacity must be a multiple of 64");

private:
    alignas(64) char arena_[ObjSize * Capacity];
    alignas(64) uint64_t bitmap_[Capacity / 64];  // 1 = free, 0 = allocated

public:
    NSPool() {
        memset(bitmap_, 0xFF, sizeof(bitmap_));
    }

    void* alloc() {
        for (size_t word_idx = 0; word_idx < Capacity / 64; ++word_idx) {
            if (bitmap_[word_idx] == 0) continue;
            
            int bit = __builtin_ctzll(bitmap_[word_idx]);
            bitmap_[word_idx] &= ~(1ULL << bit);
            
            return arena_ + (word_idx * 64 + bit) * ObjSize;
        }
        return nullptr;
    }

    void free(void* ptr) {
        size_t idx = (static_cast<char*>(ptr) - arena_) / ObjSize;
        bitmap_[idx / 64] |= (1ULL << (idx % 64));
    }

    void reset() {
        memset(bitmap_, 0xFF, sizeof(bitmap_));
    }
};

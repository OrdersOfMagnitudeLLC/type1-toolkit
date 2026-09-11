// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#pragma once

#include <atomic>
#include <cstring>
#include <cstdint>

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

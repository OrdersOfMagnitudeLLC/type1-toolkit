// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#pragma once
#include <cstdint>
#include <cstring>
#include <new>
#include <cstdlib>
#include <utility>
#include <type_traits>

static inline uint64_t nsh_mix(uint64_t v) {
    v ^= v >> 30; v *= 0xbf58476d1ce4e5b9ULL;
    v ^= v >> 27; v *= 0x94d049bb133111ebULL;
    return v ^ (v >> 31);
}

static inline uint64_t nsh_hash_str(const char* p, size_t len) {
    uint64_t h = 0x9e3779b97f4a7c15ULL ^ (uint64_t)len;
    const char* end = p + (len & ~7ULL);
    for (; p < end; p += 8) {
        uint64_t w = 0; std::memcpy(&w, p, 8);
        h ^= nsh_mix(w); h = nsh_mix(h);
    }
    if (len & 7) {
        uint64_t w = 0; std::memcpy(&w, p, len & 7);
        h ^= nsh_mix(w); h = nsh_mix(h);
    }
    return h;
}

// Identity hash (default): optimal for sequential/timestamp/structured keys
struct IdentityHash {
    size_t operator()(uint64_t k) const { return static_cast<size_t>(k); }
};

// General hash: use when key distribution is unknown or adversarial
struct WyHash {
    size_t operator()(uint64_t k) const { return static_cast<size_t>(nsh_mix(k)); }
};

template<typename Key = uint64_t, typename Value = uint64_t,
         typename Hasher = void>
class NSHash {
public:
    explicit NSHash(double max_load_factor = 0.5)
        : slots_(nullptr), values_(nullptr),
          capacity_(0), mask_(0), size_(0), grow_at_(0), max_load_factor_(max_load_factor) {}
    
    ~NSHash() {
        std::free(slots_);
        std::free(values_);
    }
    
    void insert(Key key, Value value) {
        if (size_ >= grow_at_) [[unlikely]] grow();
        size_t slot = slot_for(key);
        uint8_t dist = 0;
        for (;;) {
            if (slots_[slot].psl == 0xFF) [[likely]] {
                slots_[slot].key = key; values_[slot] = value; slots_[slot].psl = dist;
                size_++; return;
            }
            if (slots_[slot].key == key) { values_[slot] = value; return; }
            if (slots_[slot].psl < dist) {
                std::swap(key, slots_[slot].key);
                std::swap(value, values_[slot]);
                std::swap(dist, slots_[slot].psl);
            }
            slot = (slot + 1) & mask_;
            dist++;
        }
    }
    
    __attribute__((always_inline)) Value* lookup(Key key) {
        if (!capacity_) return nullptr;
        size_t slot = slot_for(key);
        uint8_t dist = 0;
        while (slots_[slot].psl != 0xFF && slots_[slot].psl >= dist) {
            if (slots_[slot].key == key) return &values_[slot];
            slot = (slot + 1) & mask_; dist++;
        }
        return nullptr;
    }
    
    bool erase(Key key) {
        if (!capacity_) return false;
        size_t slot = slot_for(key);
        uint8_t dist = 0;
        while (slots_[slot].psl != 0xFF && slots_[slot].psl >= dist) {
            if (slots_[slot].key == key) {
                for (;;) {
                    size_t next = (slot + 1) & mask_;
                    if (slots_[next].psl == 0xFF || slots_[next].psl == 0) {
                        slots_[slot].psl = 0xFF; size_--; return true;
                    }
                    slots_[slot].key   = slots_[next].key;
                    values_[slot]      = values_[next];
                    slots_[slot].psl   = slots_[next].psl - 1;
                    slot = next;
                }
            }
            slot = (slot + 1) & mask_; dist++;
        }
        return false;
    }
    
    size_t size() const { return size_; }
    size_t capacity() const { return capacity_; }
    double load_factor() const { return capacity_ ? (double)size_ / capacity_ : 0.0; }
    
    void reserve(size_t min_elements) {
        size_t needed = static_cast<size_t>(min_elements / max_load_factor_) + 1;
        size_t cap = capacity_ == 0 ? 16 : capacity_;
        while (cap < needed) cap *= 2;
        if (cap > capacity_) {
            Slot* os = slots_; Value* ov = values_;
            size_t oc = capacity_;
            size_ = 0; slots_ = nullptr; values_ = nullptr;
            alloc_tables(cap);
            if (os) {
                for (size_t i = 0; i < oc; i++)
                    if (os[i].psl != 0xFF) insert(os[i].key, ov[i]);
                std::free(os); std::free(ov);
            }
        }
    }
    
    void clear() {
        if (slots_) std::memset(slots_, 0xFF, capacity_ * sizeof(Slot));
        size_ = 0;
    }
    
private:
    struct alignas(8) Slot { uint8_t psl; Key key; };
    
    __attribute__((always_inline)) size_t slot_for(Key key) const {
        if constexpr (std::is_same_v<Hasher, void> || std::is_same_v<Hasher, IdentityHash>) {
            return static_cast<size_t>(key) & mask_;
        } else {
            Hasher h;
            return static_cast<size_t>(h(key)) & mask_;
        }
    }
    
    void alloc_tables(size_t cap) {
        slots_  = static_cast<Slot*>(std::aligned_alloc(64, cap * sizeof(Slot)));
        values_ = static_cast<Value*>(std::aligned_alloc(64, cap * sizeof(Value)));
        if (!slots_ || !values_) throw std::bad_alloc();
        std::memset(slots_, 0xFF, cap * sizeof(Slot));
        capacity_ = cap; mask_ = cap - 1;
        grow_at_ = static_cast<size_t>(cap * max_load_factor_);
    }
    
    void grow() {
        size_t new_cap = capacity_ == 0 ? 16 : capacity_ * 2;
        Slot* os = slots_; Value* ov = values_;
        size_t oc = capacity_;
        size_ = 0; slots_ = nullptr; values_ = nullptr;
        alloc_tables(new_cap);
        if (os) {
            for (size_t i = 0; i < oc; i++)
                if (os[i].psl != 0xFF) insert(os[i].key, ov[i]);
            std::free(os); std::free(ov);
        }
    }
    
    Slot*    slots_;
    Value*  values_;
    size_t  capacity_;
    size_t  mask_;
    size_t  size_;
    size_t  grow_at_;
    double  max_load_factor_;
};

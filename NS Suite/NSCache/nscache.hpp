// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <array>
#include <algorithm>
#include <memory>
#include <climits>

// wyhash implementation
namespace wyhash {
    constexpr uint64_t _wyp0 = 0xa0761d6478bd642f;
    constexpr uint64_t _wyp1 = 0xe7037ed1a0b428db;
    constexpr uint64_t _wyp2 = 0x8ebc6af09c88c6e3;
    constexpr uint64_t _wyp3 = 0x589965cc75374cc3;
    
    inline uint64_t wyhash64(const void* key, uint64_t len, uint64_t seed = 0) {
        const uint8_t* p = static_cast<const uint8_t*>(key);
        uint64_t a, b;
        if (len <= 16) {
            if (len >= 4) {
                a = (p[len - 4] << 24) | (p[len - 3] << 16) | (p[len - 2] << 8) | p[len - 1];
                b = (p[0] << 24) | (p[1] << 16) | (p[2] << 8) | p[3];
            } else if (len > 0) {
                a = p[len - 1];
                b = p[0];
            } else {
                a = b = 0;
            }
        } else {
            uint64_t i = len;
            a = _wyp0 + _wyp1 * seed;
            b = _wyp2 + _wyp3 * seed;
            for (; i > 16; i -= 16) {
                uint64_t v0 = *reinterpret_cast<const uint64_t*>(p);
                uint64_t v1 = *reinterpret_cast<const uint64_t*>(p + 8);
                a ^= v0; b ^= v1;
                a = (a << 32) | (a >> 32);
                b = (b << 32) | (b >> 32);
                a *= _wyp0; b *= _wyp2;
                p += 16;
            }
            if (i > 0) {
                uint64_t v0 = 0, v1 = 0;
                if (i >= 8) {
                    v0 = *reinterpret_cast<const uint64_t*>(p);
                    v1 = *reinterpret_cast<const uint64_t*>(p + i - 8);
                } else {
                    for (uint64_t j = 0; j < i; j++) {
                        v0 = (v0 << 8) | p[j];
                    }
                }
                a ^= v0; b ^= v1;
            }
        }
        a ^= _wyp0; b ^= _wyp1;
        a = (a << 32) | (a >> 32);
        b = (b << 32) | (b >> 32);
        a *= _wyp2; b *= _wyp3;
        a ^= _wyp0; b ^= _wyp1;
        a = (a << 32) | (a >> 32);
        b = (b << 32) | (b >> 32);
        uint64_t h = a ^ b;
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33;
        h *= 0xc4ceb9fe1a85ec53ULL;
        h ^= h >> 33;
        return h;
    }
}

// Cache slot entry for frequency-boosted CLOCK
template<typename KeyT, typename ValueT>
struct CacheSlot {
    KeyT key;
    ValueT value;
    uint32_t hit_count;
    uint8_t chances;  // Number of CLOCK passes before eviction
    bool occupied;
    bool tombstone;   // Marker for deleted slots in open addressing
    bool is_hot;      // Hot tier flag
    
    CacheSlot() : hit_count(0), chances(0), occupied(false), tombstone(false), is_hot(false) {}
};

// Frequency-boosted CLOCK cache
template<typename KeyT, typename ValueT, size_t Capacity>
class NSCache {
public:
    using key_type = KeyT;
    using value_type = ValueT;
    using size_type = size_t;
    
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");
    static constexpr uint32_t kHotThreshold = 4;
    static constexpr size_type SlotCount = Capacity * 2;  // 50% load factor
    
    NSCache();
    
    bool get(const KeyT& key, ValueT& out_value);
    bool put(const KeyT& key, const ValueT& value);
    bool contains(const KeyT& key) const;
    void clear();
    
    size_type size() const;
    double hit_rate() const;

private:
    std::unique_ptr<CacheSlot<KeyT, ValueT>[]> slots_;
    size_type clock_hand_;
    size_type size_;
    size_type hot_count_;
    
    uint64_t total_lookups_;
    uint64_t total_hits_;
    
    size_type find_empty_slot();
    size_type evict();
    void advance_clock_hand();
    size_type hash_key(const KeyT& key) const;
    void demote_hot();
};

template<typename KeyT, typename ValueT, size_t Capacity>
NSCache<KeyT, ValueT, Capacity>::NSCache()
    : clock_hand_(0)
    , size_(0)
    , hot_count_(0)
    , total_lookups_(0)
    , total_hits_(0)
{
    slots_ = std::make_unique<CacheSlot<KeyT, ValueT>[]>(SlotCount);
    for (size_t i = 0; i < SlotCount; ++i) {
        slots_[i] = CacheSlot<KeyT, ValueT>{};
    }
}

template<typename KeyT, typename ValueT, size_t Capacity>
typename NSCache<KeyT, ValueT, Capacity>::size_type
NSCache<KeyT, ValueT, Capacity>::hash_key(const KeyT& key) const {
    return static_cast<size_type>(
        wyhash::wyhash64(&key, sizeof(KeyT), 0) & (SlotCount - 1)
    );
}

template<typename KeyT, typename ValueT, size_t Capacity>
bool NSCache<KeyT, ValueT, Capacity>::get(const KeyT& key, ValueT& out_value) {
    total_lookups_++;
    
    size_type start_idx = hash_key(key);
    size_type idx = start_idx;
    
    while (true) {
        if (!slots_[idx].occupied && !slots_[idx].tombstone) {
            // Empty slot - key not found
            return false;
        }
        
        if (slots_[idx].occupied && slots_[idx].key == key) {
            // Found the key
            out_value = slots_[idx].value;
            
            // Update hit count and chances based on frequency
            slots_[idx].hit_count++;
            if (slots_[idx].hit_count >= kHotThreshold && !slots_[idx].is_hot) {
                slots_[idx].is_hot = true;
                hot_count_++;
            }
            slots_[idx].chances = 1;
            
            total_hits_++;
            return true;
        }
        
        // Linear probe
        idx = (idx + 1) & (SlotCount - 1);
        if (idx == start_idx) {
            // Wrapped around - table full, key not found
            return false;
        }
    }
}

template<typename KeyT, typename ValueT, size_t Capacity>
bool NSCache<KeyT, ValueT, Capacity>::put(const KeyT& key, const ValueT& value) {
    size_type start_idx = hash_key(key);
    size_type idx = start_idx;
    size_type first_empty = SlotCount;
    
    while (true) {
        if (!slots_[idx].occupied && !slots_[idx].tombstone) {
            // Empty slot
            if (first_empty == SlotCount) {
                first_empty = idx;
            }
            break;
        }
        
        if (slots_[idx].occupied && slots_[idx].key == key) {
            // Update existing
            slots_[idx].value = value;
            slots_[idx].hit_count++;
            if (slots_[idx].hit_count >= kHotThreshold && !slots_[idx].is_hot) {
                slots_[idx].is_hot = true;
                hot_count_++;
            }
            slots_[idx].chances = 1;
            return true;
        }
        
        if (slots_[idx].tombstone && first_empty == SlotCount) {
            first_empty = idx;
        }
        
        // Linear probe
        idx = (idx + 1) & (SlotCount - 1);
        if (idx == start_idx) {
            // Wrapped around - table full
            break;
        }
    }
    
    // New key
    if (size_ >= Capacity) {
        evict();
        first_empty = find_empty_slot();
    }
    
    if (first_empty == SlotCount) {
        first_empty = find_empty_slot();
    }
    if (first_empty == SlotCount) {
        return false;
    }
    
    slots_[first_empty].key = key;
    slots_[first_empty].value = value;
    slots_[first_empty].hit_count = 0;
    slots_[first_empty].chances = 1;
    slots_[first_empty].occupied = true;
    slots_[first_empty].tombstone = false;
    slots_[first_empty].is_hot = false;
    size_++;
    
    return true;
}

template<typename KeyT, typename ValueT, size_t Capacity>
bool NSCache<KeyT, ValueT, Capacity>::contains(const KeyT& key) const {
    size_type start_idx = hash_key(key);
    size_type idx = start_idx;
    
    while (true) {
        if (!slots_[idx].occupied && !slots_[idx].tombstone) {
            return false;
        }
        
        if (slots_[idx].occupied && slots_[idx].key == key) {
            return true;
        }
        
        idx = (idx + 1) & (SlotCount - 1);
        if (idx == start_idx) {
            return false;
        }
    }
}

template<typename KeyT, typename ValueT, size_t Capacity>
void NSCache<KeyT, ValueT, Capacity>::clear() {
    for (size_t i = 0; i < SlotCount; ++i) {
        slots_[i].occupied = false;
        slots_[i].hit_count = 0;
        slots_[i].chances = 0;
        slots_[i].tombstone = false;
        slots_[i].is_hot = false;
    }
    clock_hand_ = 0;
    size_ = 0;
    total_lookups_ = 0;
    total_hits_ = 0;
}

template<typename KeyT, typename ValueT, size_t Capacity>
typename NSCache<KeyT, ValueT, Capacity>::size_type
NSCache<KeyT, ValueT, Capacity>::size() const {
    return size_;
}

template<typename KeyT, typename ValueT, size_t Capacity>
double NSCache<KeyT, ValueT, Capacity>::hit_rate() const {
    return total_lookups_ > 0 ? static_cast<double>(total_hits_) / total_lookups_ : 0.0;
}

template<typename KeyT, typename ValueT, size_t Capacity>
typename NSCache<KeyT, ValueT, Capacity>::size_type
NSCache<KeyT, ValueT, Capacity>::find_empty_slot() {
    for (size_type i = 0; i < SlotCount; ++i) {
        if (!slots_[i].occupied && !slots_[i].tombstone) {
            return i;
        }
    }
    // If no truly empty slot, look for tombstone
    for (size_type i = 0; i < SlotCount; ++i) {
        if (!slots_[i].occupied && slots_[i].tombstone) {
            return i;
        }
    }
    return SlotCount;
}

template<typename KeyT, typename ValueT, size_t Capacity>
typename NSCache<KeyT, ValueT, Capacity>::size_type
NSCache<KeyT, ValueT, Capacity>::evict() {
    // Check if hot tier exceeds 10% - demote if needed
    size_type hot_limit = Capacity / 10;
    if (hot_count_ > hot_limit) {
        demote_hot();
    }

    size_type max_scan = std::min(Capacity / 4, static_cast<size_type>(256));
    size_type scan_count = 0;
    
    while (scan_count < max_scan) {
        size_type idx = clock_hand_;
        advance_clock_hand();
        scan_count++;
        
        if (!slots_[idx].occupied) {
            // Recycle tombstone during scan
            if (slots_[idx].tombstone) {
                slots_[idx].tombstone = false;
            }
            continue;
        }
        
        // Skip hot items entirely
        if (slots_[idx].is_hot) {
            continue;
        }
        
        if (slots_[idx].chances > 0) {
            slots_[idx].chances--;
        } else {
            // Evict this slot - set tombstone
            slots_[idx].occupied = false;
            slots_[idx].tombstone = true;
            slots_[idx].hit_count = 0;
            slots_[idx].chances = 0;
            slots_[idx].is_hot = false;
            size_--;
            return idx;
        }
    }
    
    // No victim found within window, forcibly evict current hand position
    size_type idx = clock_hand_;
    advance_clock_hand();
    if (slots_[idx].occupied) {
        if (slots_[idx].is_hot) {
            slots_[idx].is_hot = false;
            hot_count_--;
        }
        slots_[idx].occupied = false;
        slots_[idx].tombstone = true;
        slots_[idx].hit_count = 0;
        slots_[idx].chances = 0;
        slots_[idx].is_hot = false;
        size_--;
    }
    return idx;
}

template<typename KeyT, typename ValueT, size_t Capacity>
void NSCache<KeyT, ValueT, Capacity>::advance_clock_hand() {
    clock_hand_ = (clock_hand_ + 1) % SlotCount;
}

template<typename KeyT, typename ValueT, size_t Capacity>
void NSCache<KeyT, ValueT, Capacity>::demote_hot() {
    // Scan up to max(64, Capacity/32) slots from clock_hand_ to find oldest hot item
    size_type scan_limit = std::max(static_cast<size_type>(64), Capacity / 32);
    size_type candidate_idx = SlotCount;
    uint32_t min_hit_count = UINT32_MAX;
    
    size_type scan_idx = clock_hand_;
    for (size_type i = 0; i < scan_limit; i++) {
        if (slots_[scan_idx].occupied && slots_[scan_idx].is_hot) {
            if (slots_[scan_idx].hit_count < min_hit_count) {
                min_hit_count = slots_[scan_idx].hit_count;
                candidate_idx = scan_idx;
            }
        }
        scan_idx = (scan_idx + 1) % SlotCount;
    }
    
    // Demote the candidate
    if (candidate_idx != SlotCount) {
        slots_[candidate_idx].is_hot = false;
        hot_count_--;
    }
}

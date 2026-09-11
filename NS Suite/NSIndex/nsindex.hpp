// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#ifndef NSINDEX_HPP
#define NSINDEX_HPP

// NSIndex — Learned Index with AVX-512 Range Query Acceleration
// License: AGPL-3.0
// Description: Cache-optimized learned index for sorted key arrays with hybrid prediction/BitBlock dispatch

#include <algorithm>
#include <cstddef>
#include <vector>
#include <immintrin.h>
#include <sys/types.h>
#include <cstdio>

template<typename Key>
class NSIndex {
public:
    static constexpr int K_BITS = 20;
    
    struct alignas(64) HotData {
        Key min_key;
        Key max_key;
        double stride_estimate;
        size_t n;
        size_t error_bound;
    } hot_;  // fits in one cache line

    NSIndex(const Key* keys, size_t n)
        : keys_(keys), needs_flush_(false) {
        hot_.n = n;
        hot_.error_bound = 64;
        delta_buffer_.reserve(std::max(hot_.n / 10, size_t(16)));
        if (n > 0) {
            hot_.min_key = keys[0];
            hot_.max_key = keys[n - 1];
            if (n > 1) {
                hot_.stride_estimate = static_cast<double>(hot_.max_key - hot_.min_key) / (hot_.n - 1);
                if (hot_.stride_estimate == 0) {
                    hot_.stride_estimate = 1.0; // Avoid division by zero
                }
            } else {
                hot_.stride_estimate = 1.0;
            }
        } else {
            hot_.min_key = Key{};
            hot_.max_key = Key{};
            hot_.stride_estimate = 1.0;
        }
        
        // Compute adaptive bit selection for block table
        Key key_range = hot_.max_key - hot_.min_key;
        int top_bit = 63 - __builtin_clzll(key_range | 1);  // highest set bit
        shift_ = std::max(0, top_bit - (K_BITS - 1));
        
        // Build block table for range queries
        build_block_table();
        
        // Build-time distribution detection: sample keys to measure prediction error
        size_t sample_size = std::min(size_t(512), hot_.n);
        size_t step = (hot_.n > 1) ? (hot_.n - 1) / (sample_size - 1) : 1;
        max_sampled_error_ = 0;
        
        for (size_t i = 0; i < sample_size; i++) {
            size_t actual_pos = i * step;
            if (actual_pos >= hot_.n) actual_pos = hot_.n - 1;
            
            Key sampled_key = keys_[actual_pos];
            // Same linear prediction as predict()
            double offset = static_cast<double>(sampled_key - hot_.min_key);
            if (offset < 0.0) offset = 0.0;
            size_t predicted_pos = static_cast<size_t>(offset / hot_.stride_estimate);
            if (predicted_pos >= hot_.n) predicted_pos = hot_.n - 1;
            
            size_t error = (predicted_pos > actual_pos) ? (predicted_pos - actual_pos) : (actual_pos - predicted_pos);
            if (error > max_sampled_error_) max_sampled_error_ = error;
        }
        
        use_bitblock_ = (max_sampled_error_ > 256);
    }

    size_t predict(Key k) const {
        if (hot_.n == 0) return 0;
        double offset = static_cast<double>(k - hot_.min_key);
        if (offset < 0.0) return 0;
        size_t pos = static_cast<size_t>(offset / hot_.stride_estimate);
        if (pos >= hot_.n) pos = hot_.n - 1;
        return pos;
    }

    const Key* find(const Key* keys, size_t n, Key k) const {
        if (n == 0) return nullptr;
        
        size_t pos = predict(k);
        __builtin_prefetch(&keys[pos], 0, 1);
        
        // Check predicted position
        if (keys[pos] == k) {
            return &keys[pos];
        }
        
        // Linear scan outward ±8 positions
        const int scan_range = 8;
#ifdef __AVX2__
        // AVX2 SIMD scan for int64_t keys
        if constexpr (sizeof(Key) == 8) {
            __m256i key_vec = _mm256_set1_epi64x(k);
            // Scan forward
            for (int i = 0; i < scan_range && pos + i + 4 <= n; i += 4) {
                __m256i data = _mm256_loadu_si256((__m256i*)&keys[pos + i]);
                __m256i cmp = _mm256_cmpeq_epi64(data, key_vec);
                int mask = _mm256_movemask_epi8(cmp);
                if (mask != 0) {
                    // Find first match
                    for (int j = 0; j < 4; j++) {
                        if (keys[pos + i + j] == k) {
                            return &keys[pos + i + j];
                        }
                    }
                }
                if (keys[pos + i] > k) break;
            }
            // Scan backward
            for (int i = 1; i <= scan_range && pos >= static_cast<size_t>(i); i++) {
                if (keys[pos - i] == k) {
                    return &keys[pos - i];
                }
            }
        } else {
#endif
            // Scalar fallback
            for (int i = 1; i <= scan_range; i++) {
                // Check forward
                if (pos + static_cast<size_t>(i) < n) {
                    if (keys[pos + i] == k) {
                        return &keys[pos + i];
                    }
                    if (keys[pos + i] > k) {
                        break; // Gone past the key
                    }
                }
                // Check backward
                if (pos >= static_cast<size_t>(i)) {
                    if (keys[pos - i] == k) {
                        return &keys[pos - i];
                    }
                }
            }
#ifdef __AVX2__
        }
#endif
        
        // Binary search sorted delta buffer (only if non-empty)
        if (!delta_buffer_.empty()) {
            auto it = std::lower_bound(delta_buffer_.begin(), delta_buffer_.end(), k);
            if (it != delta_buffer_.end() && *it == k) {
                return &(*it);
            }
        }
        
        return keys + n; // Return end pointer on miss
    }

    bool in_delta(Key k) const {
        // Binary search — delta_buffer_ is kept sorted
        return std::binary_search(delta_buffer_.begin(), delta_buffer_.end(), k);
    }

    const Key* predecessor(const Key* keys, size_t n, Key k) const {
        if (n == 0) return nullptr;
        
        size_t pos = predict(k);
        
        // Scan left from predicted position
        for (ssize_t i = static_cast<ssize_t>(pos); i >= 0; i--) {
            if (keys[i] <= k) {
                return &keys[i];
            }
        }
        
        // Check delta buffer
        if (!delta_buffer_.empty()) {
            auto it = std::upper_bound(delta_buffer_.begin(), delta_buffer_.end(), k);
            if (it != delta_buffer_.begin()) {
                --it;
                return &(*it);
            }
        }
        
        return nullptr;
    }

    const Key* successor(const Key* keys, size_t n, Key k) const {
        if (n == 0) return nullptr;
        
        size_t pos = predict(k);
        
        // Scan right from predicted position
        for (size_t i = pos; i < n; i++) {
            if (keys[i] >= k) {
                return &keys[i];
            }
        }
        
        // Check delta buffer
        if (!delta_buffer_.empty()) {
            auto it = std::lower_bound(delta_buffer_.begin(), delta_buffer_.end(), k);
            if (it != delta_buffer_.end()) {
                return &(*it);
            }
        }
        
        return nullptr;
    }

    int find_range(const Key* keys, size_t n, Key lo, Key hi, Key* out, int max_out) const {
        if (hot_.n == 0 || max_out == 0) return 0;
        size_t pos;
        if (use_bitblock_) {
            uint32_t block = (uint32_t)((lo - hot_.min_key) >> shift_);
            pos = (block > 0) ? block_table_[block - 1] : 0;
        } else {
            pos = predict(lo);
            while (pos > 0 && keys[pos] > lo) pos--;
        }
        return _avx_scan(pos, lo, hi, out, max_out);
    }
    
    int fast_range(size_t start_pos, Key hi, Key* out, int max_out) const {
        int count = 0;
        size_t i = start_pos;
#ifdef __AVX512F__
        __m512i hi_vec = _mm512_set1_epi64(hi);
        while (i + 8 <= hot_.n && count < max_out) {
            __m512i data = _mm512_loadu_si512((__m512i*)&keys_[i]);
            __mmask8 gt = _mm512_cmpgt_epi64_mask(data, hi_vec);
            if (gt == 0xFF) break;
            __mmask8 le = _mm512_cmple_epi64_mask(data, hi_vec);
            int matched = _mm_popcnt_u32(le);
            if (count + matched > max_out) {
                matched = max_out - count;
                __mmask8 trunc_mask = le;
                for (int j = 0; j < matched; j++) {
                    if (trunc_mask & (1 << j)) {
                        out[count++] = keys_[i + j];
                    }
                }
                break;
            }
            _mm512_mask_compressstoreu_epi64(out + count, le, data);
            count += matched;
            i += 8;
        }
#else
        // Scalar fallback
        while (i < hot_.n && count < max_out) {
            if (keys_[i] <= hi) {
                out[count++] = keys_[i];
            } else {
                break;
            }
            i++;
        }
#endif
        return count;
    }
    
    int range(Key lo, Key hi, Key* out, int max_out) const {
        return find_range(keys_, hot_.n, lo, hi, out, max_out);
    }
    
    int find_range_bitblock(Key lo, Key hi, Key* out, int max_out) const {
        if (hot_.n == 0 || max_out == 0) return 0;
        
        // one bitshift = start position, no prediction math
        uint32_t block = (uint32_t)((lo - hot_.min_key) >> shift_);
        size_t pos = (block > 0) ? block_table_[block - 1] : 0;
        
        return _avx_scan(pos, lo, hi, out, max_out);
    }

    void insert(Key k) {
        // Insert into delta_buffer_ maintaining sorted order
        auto it = std::lower_bound(delta_buffer_.begin(), delta_buffer_.end(), k);
        delta_buffer_.insert(it, k);
        if (delta_buffer_.size() > hot_.n / 10 && hot_.n > 0) {
            needs_flush_ = true;
        }
    }

    bool needs_flush() const { return needs_flush_; }

    void flush(std::vector<Key>& main_array) {
        if (delta_buffer_.empty()) return;
        
        // O(n) merge: both arrays are already sorted
        std::vector<Key> merged;
        merged.reserve(main_array.size() + delta_buffer_.size());
        std::merge(main_array.begin(), main_array.end(), 
                   delta_buffer_.begin(), delta_buffer_.end(),
                   std::back_inserter(merged));
        main_array = std::move(merged);
        
        // Update pointers and metadata
        keys_ = main_array.data();
        hot_.n = main_array.size();
        
        if (hot_.n > 0) {
            hot_.min_key = main_array[0];
            hot_.max_key = main_array[hot_.n - 1];
            if (hot_.n > 1) {
                hot_.stride_estimate = static_cast<double>(hot_.max_key - hot_.min_key) / (hot_.n - 1);
                if (hot_.stride_estimate == 0) {
                    hot_.stride_estimate = 1.0;
                }
            } else {
                hot_.stride_estimate = 1.0;
            }
        }
        
        // Clear delta buffer and reset flag
        delta_buffer_.clear();
        needs_flush_ = false;
        
        // Recompute adaptive bit selection for block table
        Key key_range = hot_.max_key - hot_.min_key;
        int top_bit = 63 - __builtin_clzll(key_range | 1);
        shift_ = std::max(0, top_bit - (K_BITS - 1));
        
        // Rebuild block table after flush
        build_block_table();
        
        // Re-run distribution detection after flush
        size_t sample_size = std::min(size_t(512), hot_.n);
        size_t step = (hot_.n > 1) ? (hot_.n - 1) / (sample_size - 1) : 1;
        max_sampled_error_ = 0;
        
        for (size_t i = 0; i < sample_size; i++) {
            size_t actual_pos = i * step;
            if (actual_pos >= hot_.n) actual_pos = hot_.n - 1;
            
            Key sampled_key = keys_[actual_pos];
            double offset = static_cast<double>(sampled_key - hot_.min_key);
            if (offset < 0.0) offset = 0.0;
            size_t predicted_pos = static_cast<size_t>(offset / hot_.stride_estimate);
            if (predicted_pos >= hot_.n) predicted_pos = hot_.n - 1;
            
            size_t error = (predicted_pos > actual_pos) ? (predicted_pos - actual_pos) : (actual_pos - predicted_pos);
            if (error > max_sampled_error_) max_sampled_error_ = error;
        }
        
        use_bitblock_ = (max_sampled_error_ > 256);
    }
    
    void build_block_table() {
        block_table_.assign(1 << K_BITS, hot_.n);  // default = end
        // walk keys_ backwards, fill block_table_
        for (int64_t i = (int64_t)hot_.n - 1; i >= 0; i--) {
            uint32_t block = (uint32_t)((keys_[i] - hot_.min_key) >> shift_);
            block = std::min(block, static_cast<uint32_t>(block_table_.size() - 1));
            block_table_[block] = i;
        }
        // forward fill gaps: if block_table_[i] == n_, use block_table_[i-1]
        for (size_t i = 1; i < block_table_.size(); i++)
            if (block_table_[i] == hot_.n) block_table_[i] = block_table_[i-1];
    }

private:
    __attribute__((always_inline)) int _avx_scan(size_t pos, Key lo, Key hi, Key* out, int max_out) const {
        int out_count = 0;
        if (hot_.n == 0 || max_out == 0) return 0;
        
        __m512i lo_vec = _mm512_set1_epi64(lo);
        __m512i hi_vec = _mm512_set1_epi64(hi);
        
        // Estimate how many array positions span (hi - lo) based on key density
        size_t key_range = (size_t)(hi - lo);
        size_t density = hot_.n / (hot_.max_key - hot_.min_key + 1);
        size_t estimated_span = key_range * density + hot_.error_bound * 2 + 16;
        size_t pos_hi = std::min(pos + estimated_span, hot_.n);
        
        // Prefetch all cache lines in scan range before entering AVX loop
        const char* base = (const char*)&keys_[pos];
        const char* end  = (const char*)&keys_[pos_hi];
        for (const char* p = base; p < end; p += 64)
            _mm_prefetch(p, _MM_HINT_T0);
        
#ifdef __AVX512F__
        // AVX-512 vectorized scan for int64 keys
        if constexpr (sizeof(Key) == 8) {
            // Process 8 keys at a time (512-bit = 8 x int64)
            size_t i = pos;
            while (i + 8 <= pos_hi && out_count < max_out) {
                __m512i data = _mm512_loadu_si512((__m512i*)&keys_[i]);
                
                // Check if all keys > hi (stop condition)
                __mmask8 gt_hi_mask = _mm512_cmpgt_epi64_mask(data, hi_vec);
                if (gt_hi_mask == 0xFF) {
                    break;
                }
                
                // Check if keys >= lo and <= hi
                __mmask8 ge_lo_mask = _mm512_cmpge_epi64_mask(data, lo_vec);
                __mmask8 le_hi_mask = _mm512_cmple_epi64_mask(data, hi_vec);
                __mmask8 match_mask = ge_lo_mask & le_hi_mask;
                
                // Compress and store matching keys directly to out[]
                if (match_mask != 0) {
                    int matched = _mm_popcnt_u32(match_mask);
                    if (out_count + matched > max_out) {
                        matched = max_out - out_count;
                        __mmask8 trunc_mask = match_mask;
                        for (int j = 0; j < matched; j++) {
                            if (trunc_mask & (1 << j)) {
                                out[out_count++] = keys_[i + j];
                            }
                        }
                        break;
                    }
                    _mm512_mask_compressstoreu_epi64(out + out_count, match_mask, data);
                    out_count += matched;
                }
                
                i += 8;
            }
            
            // Scalar tail from pos_hi to pos_hi+error_bound_+1 to catch clipped keys
            size_t tail_end = std::min(pos_hi + hot_.error_bound + 1, hot_.n);
            while (i < tail_end && keys_[i] <= hi && out_count < max_out) {
                if (keys_[i] >= lo) {
                    out[out_count++] = keys_[i];
                }
                i++;
            }
        } else {
#endif
            // Scalar fallback for non-int64 or no AVX-512
            while (pos < hot_.n && keys_[pos] <= hi && out_count < max_out) {
                if (keys_[pos] >= lo) {
                    out[out_count++] = keys_[pos];
                }
                pos++;
            }
#ifdef __AVX512F__
        }
#endif
        
        // Also check sorted delta buffer via lower_bound (scalar) - skip if empty
        if (!delta_buffer_.empty() && out_count < max_out) {
            auto lo_it = std::lower_bound(delta_buffer_.begin(), delta_buffer_.end(), lo);
            auto hi_it = std::upper_bound(delta_buffer_.begin(), delta_buffer_.end(), hi);
            for (auto it = lo_it; it != hi_it && out_count < max_out; ++it) {
                out[out_count++] = *it;
            }
        }
        
        return out_count;
    }

private:
    const Key* keys_;
    std::vector<Key> delta_buffer_;
    std::vector<size_t> block_table_;
    int shift_ = 0;
    bool use_bitblock_ = false;
    size_t max_sampled_error_ = 0;
    bool needs_flush_;
};

#include <shared_mutex>
#include <mutex>
template<typename Key>
class NSIndexMT {
public:
    NSIndexMT(const Key* keys, size_t n) : index_(keys, n) {}

    const Key* find(const Key* keys, size_t n, Key k) const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return index_.find(keys, n, k);
    }
    int find_range(const Key* keys, size_t n, Key lo, Key hi, Key* out, int max_out) const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return index_.find_range(keys, n, lo, hi, out, max_out);
    }
    void insert(Key k) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        index_.insert(k);
    }
    void flush(std::vector<Key>& arr) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        index_.flush(arr);
    }
    bool needs_flush() const { return index_.needs_flush(); }
    size_t predict(Key k) const { return index_.predict(k); }

private:
    NSIndex<Key> index_;
    mutable std::shared_mutex mutex_;
};

#endif // NSINDEX_HPP

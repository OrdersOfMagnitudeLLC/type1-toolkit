#pragma once

// KVBox — compressed KV cache mirror using dynamic-range INT8 quantization
// + position-indexed hash map. No cross-layer grouping: each layer stored
// independently. Per-head dynamic scale preserves full value range.
// Collision-free: every position gets its own slot via std::unordered_map.

#include "ggml.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cmath>
#include <cfloat>
#include <vector>
#include <algorithm>
#include <unordered_map>

struct KVBox {
    // Full per-layer K+V storage. Cross-layer K averaging for scoring only.
    uint32_t n_layers;
    uint32_t group_size;  // K averaging factor for scoring (4)
    uint32_t n_kv_heads;
    uint32_t head_dim;

    uint64_t n_tokens;    // logical capacity (matches ik_llama kv_size)
    uint64_t n_slots;     // max physical slots = n_tokens (one per position)
    size_t   head_bytes;  // per head: 2 * head_dim * sizeof(fp16) (K + V)
    size_t   slot_bytes;  // per slot: n_layers * n_kv_heads * head_bytes
    size_t   total_bytes; // total allocated
    uint8_t* buf;

    uint64_t writes;      // number of write_slot calls
    uint64_t retrievals;  // number of read_slot cache misses served
    uint64_t prefill_retrievals;  // number of read_slot cache misses served
    uint64_t abs_pos;     // absolute token position (for small working cache mode)

    // Stored Q from last token for decode-time retrieval scoring
    std::vector<float> retrieval_q;  // full per-head Q [n_heads * head_dim]
    int64_t retrieval_q_pos;         // position of the stored Q token
    bool decode_injected_once;       // true after the one-shot decode retrieval/injection has run
    uint32_t retrieval_layer;        // which layer's post-RoPE Q/K to use for retrieval scoring

    // Position-indexed hash map: absolute position → slot index
    std::unordered_map<int64_t, uint32_t> pos_to_slot;
    // Free list of available slot indices (stack)
    std::vector<uint32_t> free_slots;
    // Which slots are currently in use
    std::vector<bool> slot_used;

    KVBox()
        : n_layers(0), group_size(4), n_kv_heads(0), head_dim(0)
        , n_tokens(0), n_slots(0), head_bytes(0), slot_bytes(0)
        , total_bytes(0), buf(nullptr)
        , writes(0), retrievals(0), prefill_retrievals(0), abs_pos(0)
        , retrieval_q_pos(0), decode_injected_once(false), retrieval_layer(0)
    {}

    ~KVBox() {
        if (buf) free(buf);
    }

    void init(uint32_t layers, uint32_t heads, uint32_t dim, uint64_t tokens) {
        if (buf) { free(buf); buf = nullptr; }
        n_layers = layers;
        group_size = 4;
        n_kv_heads = heads;
        head_dim = dim;
        n_tokens = tokens;
        n_slots = tokens;  // one slot per possible position (collision-free)
        // Per head: raw FP16 K[head_dim] + FP16 V[head_dim] (no quantization)
        head_bytes = (size_t)head_dim * 2 * sizeof(ggml_fp16_t);
        slot_bytes = (size_t)n_layers * n_kv_heads * head_bytes;
        total_bytes = (size_t)n_slots * slot_bytes;
        buf = (uint8_t*)malloc(total_bytes);
        assert(buf && "KVBox malloc failed");
        memset(buf, 0, total_bytes);
        writes = 0;
        retrievals = 0;
        prefill_retrievals = 0;
        abs_pos = 0;
        decode_injected_once = false;
        // Layer 0 empirically gives the best score discrimination.
        retrieval_layer = 0;
        pos_to_slot.clear();
        slot_used.assign(n_slots, false);
        free_slots.clear();
        free_slots.reserve(n_slots);
        for (uint32_t i = 0; i < n_slots; i++) {
            free_slots.push_back((uint32_t)(n_slots - 1 - i));
        }
    }

    bool initialized() const { return buf != nullptr; }

    // Check if a position has been written to KVBox
    bool has_position(int64_t pos) const {
        return pos_to_slot.find(pos) != pos_to_slot.end();
    }

    // Get all stored positions (for retrieval candidate selection)
    std::vector<int64_t> get_stored_positions() const {
        std::vector<int64_t> positions;
        positions.reserve(pos_to_slot.size());
        for (const auto & kv : pos_to_slot) {
            positions.push_back(kv.first);
        }
        return positions;
    }

    // Write K and V for (token_idx, layer, head).
    // k_f16 / v_f16: each head_dim ggml_fp16_t values.
    // Raw FP16 storage — no quantization.
    void write_slot(uint64_t token_idx, uint32_t layer,
                    uint32_t head, const ggml_fp16_t* k_f16,
                    const ggml_fp16_t* v_f16) {
        assert(buf && "KVBox not allocated");
        assert(layer < n_layers);
        assert(head < n_kv_heads);

        int64_t pos = (int64_t)token_idx;
        uint32_t slot;
        auto it = pos_to_slot.find(pos);
        if (it != pos_to_slot.end()) {
            slot = it->second;
        } else {
            if (free_slots.empty()) return;
            slot = free_slots.back();
            free_slots.pop_back();
            pos_to_slot[pos] = slot;
            slot_used[slot] = true;
        }

        size_t offset = (size_t)slot * slot_bytes
                      + (size_t)(layer * n_kv_heads + head) * head_bytes;

        ggml_fp16_t* dst_k = (ggml_fp16_t*)(buf + offset);
        ggml_fp16_t* dst_v = dst_k + head_dim;
        memcpy(dst_k, k_f16, head_dim * sizeof(ggml_fp16_t));
        memcpy(dst_v, v_f16, head_dim * sizeof(ggml_fp16_t));
        writes++;
    }

    // Read K and V for (token_idx, layer, head) from storage.
    // Returns true if the position was previously written (cache hit).
    // Raw FP16 — no dequantization needed.
    bool read_slot(uint64_t token_idx, uint32_t layer,
                   uint32_t head, ggml_fp16_t* k_f16,
                   ggml_fp16_t* v_f16) const {
        assert(buf && "KVBox not allocated");
        assert(layer < n_layers);
        assert(head < n_kv_heads);

        int64_t pos = (int64_t)token_idx;
        auto it = pos_to_slot.find(pos);
        if (it == pos_to_slot.end()) return false;
        uint32_t slot = it->second;

        size_t offset = (size_t)slot * slot_bytes
                      + (size_t)(layer * n_kv_heads + head) * head_bytes;

        const ggml_fp16_t* src_k = (const ggml_fp16_t*)(buf + offset);
        const ggml_fp16_t* src_v = src_k + head_dim;
        memcpy(k_f16, src_k, head_dim * sizeof(ggml_fp16_t));
        memcpy(v_f16, src_v, head_dim * sizeof(ggml_fp16_t));
        return true;
    }

    // Read averaged K for scoring: average K across group_size layers (0..group_size-1)
    // for the given head. V is not needed for scoring.
    // Returns true if the position was previously written.
    bool read_score_k(uint64_t token_idx, uint32_t head, ggml_fp16_t* k_f16) const {
        assert(buf && "KVBox not allocated");
        assert(head < n_kv_heads);

        int64_t pos = (int64_t)token_idx;
        auto it = pos_to_slot.find(pos);
        if (it == pos_to_slot.end()) return false;
        uint32_t slot = it->second;

        uint32_t n_avg = std::min(group_size, n_layers);
        std::vector<float> k_sum(head_dim, 0.0f);

        for (uint32_t il = 0; il < n_avg; ++il) {
            size_t offset = (size_t)slot * slot_bytes
                          + (size_t)(il * n_kv_heads + head) * head_bytes;
            const ggml_fp16_t* src_k = (const ggml_fp16_t*)(buf + offset);
            for (uint32_t d = 0; d < head_dim; ++d) {
                k_sum[d] += ggml_fp16_to_fp32(src_k[d]);
            }
        }

        for (uint32_t d = 0; d < head_dim; ++d) {
            k_f16[d] = ggml_fp32_to_fp16(k_sum[d] / (float)n_avg);
        }
        return true;
    }

    // Evict a position from KVBox, freeing its slot
    void evict_position(int64_t pos) {
        auto it = pos_to_slot.find(pos);
        if (it == pos_to_slot.end()) return;
        uint32_t slot = it->second;
        pos_to_slot.erase(it);
        slot_used[slot] = false;
        free_slots.push_back(slot);
    }

    uint64_t slots_used_count() const {
        return (uint64_t)pos_to_slot.size();
    }

    // Compression ratio: ik_llama allocated bytes / KVBox allocated bytes
    double compression_ratio(size_t ik_llama_kv_bytes) const {
        if (total_bytes == 0) return 0.0;
        return (double)ik_llama_kv_bytes / (double)total_bytes;
    }
};

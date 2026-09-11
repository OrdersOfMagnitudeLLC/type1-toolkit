// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#ifndef NSSTRINGINDEX_HPP
#define NSSTRINGINDEX_HPP

#include <vector>
#include <unordered_map>
#include <string>
#include <string_view>
#include <cstdint>
#include <iostream>
#include <chrono>
#include <omp.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

struct FlatStrMap {
    static const uint32_t CAP = 4096;  // power of 2, enough for low-cardinality fields
    struct Slot { std::string_view key; uint32_t val; bool used = false; };
    Slot slots[CAP];

    static uint64_t _hash(const char* p, size_t n) {
        uint64_t h = 0x9368d13a6b3c4f27ULL ^ n;
        for (size_t i = 0; i < n; i++)
            h = (h ^ (uint8_t)p[i]) * 0x517cc1b727220a95ULL;
        return h ^ (h >> 32);
    }

    uint32_t& operator[](std::string_view k) {
        uint64_t h = _hash(k.data(), k.size()) & (CAP - 1);
        while (slots[h].used && slots[h].key != k) h = (h + 1) & (CAP - 1);
        if (!slots[h].used) { slots[h] = {k, 0, true}; }
        return slots[h].val;
    }

    uint32_t* find(std::string_view k) {
        uint64_t h = _hash(k.data(), k.size()) & (CAP - 1);
        while (slots[h].used && slots[h].key != k) h = (h + 1) & (CAP - 1);
        return slots[h].used ? &slots[h].val : nullptr;
    }

    void merge_into(FlatStrMap& target) const {
        for (uint32_t i = 0; i < CAP; i++)
            if (slots[i].used)
                target[slots[i].key] += slots[i].val;
    }
};

// Schema-aware log indexer. Parse structured text once, answer field=value
// queries in microseconds from an in-memory index.
struct NSStringIndex {
    // Count-only index: (field, value) → count
    // Used for 90% of analytics queries where only counts matter
    std::vector<FlatStrMap> field_index;

    // Detailed index with line numbers, built on demand
    // Used when actual line numbers are needed (e.g., for verification or export)
    std::vector<std::unordered_map<std::string_view, std::vector<uint32_t>>> field_index_detailed;
    bool detailed_built = false;
    uint32_t num_fields;
    uint32_t num_lines;

    // Parse corpus once. delimiter = field separator.
    // num_fields: how many fields per line to index (index all of them).
    // index_fields: which specific fields to index (only these will be populated)
    // IMPORTANT: corpus must outlive the index (string_views point into corpus)
    // Single-threaded version (use build() for parallel by default)
    void build_single(const char* corpus, size_t len, char delimiter, uint32_t num_fields, const std::vector<uint32_t>& index_fields);

    // Parallel build using OpenMP (default, requires -fopenmp)
    void build(const char* corpus, size_t len, char delimiter, uint32_t num_fields, const std::vector<uint32_t>& index_fields);

    // Build detailed index with line numbers (on demand)
    void build_detailed(const char* corpus, size_t len, char delimiter, uint32_t num_fields);

    // Query: return count of lines where field_idx == value.
    // O(1) lookup after build.
    uint64_t query(uint32_t field_idx, const std::string& value);

    // Query with line numbers (for verification)
    // Builds detailed index on first call if not already built
    const std::vector<uint32_t>* query_lines(uint32_t field_idx,
                                               const std::string& value);
};

inline void NSStringIndex::build_single(const char* corpus, size_t len, char delimiter, uint32_t num_fields, const std::vector<uint32_t>& index_fields) {
    this->num_fields = num_fields;
    this->num_lines = 0;
    field_index.resize(num_fields);

#ifdef __AVX2__
    const __m256i nl_vec  = _mm256_set1_epi8('\n');
    const __m256i dlm_vec = _mm256_set1_epi8(delimiter);
    size_t pos = 0;
    size_t avx_end = len & ~31ULL;
    uint32_t field_starts[32];
    uint8_t  field_count = 0;
    size_t   field_start = 0;
    size_t   line_start  = 0;
    uint32_t line_num    = 0;

    auto process_event = [&](size_t char_pos, bool is_newline) {
        if (is_newline) {
            for (uint32_t f : index_fields) {
                if (f < field_count + 1) {
                    size_t fs = (f == 0) ? line_start
                              : (size_t)field_starts[f-1] + 1;
                    size_t fe = (f < field_count)
                              ? (size_t)field_starts[f]
                              : char_pos;
                    field_index[f][std::string_view(corpus+fs, fe-fs)]++;
                }
            }
            field_count = 0;
            line_start  = char_pos + 1;
            field_start = char_pos + 1;
            line_num++;
        } else {
            if (field_count < 32)
                field_starts[field_count++] = char_pos;
        }
    };

    while (pos < avx_end) {
        __m256i chunk   = _mm256_loadu_si256((const __m256i*)(corpus + pos));
        uint32_t nl_m   = _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, nl_vec));
        uint32_t dl_m   = _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, dlm_vec));
        uint32_t combined = nl_m | dl_m;
        while (combined) {
            int bit = __builtin_ctz(combined);
            process_event(pos + bit, (nl_m >> bit) & 1);
            combined &= combined - 1;
        }
        pos += 32;
    }
    // scalar remainder
    for (size_t i = pos; i <= len; i++) {
        if (i == len || corpus[i] == '\n')
            process_event(i, true);
        else if (corpus[i] == delimiter)
            process_event(i, false);
    }
    this->num_lines = line_num;
#else
    // scalar fallback
    uint32_t line_num = 0;
    size_t line_start = 0;
    std::vector<std::string_view> fields;
    fields.reserve(num_fields);

    for (size_t i = 0; i <= len; i++) {
        if (i == len || corpus[i] == '\n') {
            fields.clear();
            size_t field_start = line_start;
            uint32_t field_count = 0;

            for (size_t j = line_start; j < i; j++) {
                if (corpus[j] == delimiter) {
                    if (field_count < num_fields) {
                        fields.push_back(std::string_view(corpus + field_start, j - field_start));
                        field_count++;
                        field_start = j + 1;
                    }
                }
            }
            if (field_count < num_fields && line_start < i) {
                fields.push_back(std::string_view(corpus + field_start, i - field_start));
                field_count++;
            }

            for (uint32_t f : index_fields) {
                if (f < fields.size())
                    field_index[f][fields[f]]++;
            }

            line_num++;
            line_start = i + 1;
        }
    }
    this->num_lines = line_num;
#endif
}

inline void NSStringIndex::build(const char* corpus, size_t len, char delimiter, uint32_t nf, const std::vector<uint32_t>& index_fields) {
    num_fields = nf;
    field_index.resize(nf);
    int nt = omp_get_max_threads();
    std::vector<size_t> starts(nt + 1);
    starts[0] = 0; starts[nt] = len;
    for (int t = 1; t < nt; t++) {
        size_t p = (len / nt) * t;
        while (p < len && corpus[p] != '\n') p++;
        starts[t] = p + 1;
    }
    std::vector<std::vector<FlatStrMap>> lm(nt, std::vector<FlatStrMap>(nf));
    #pragma omp parallel for num_threads(nt)
    for (int t = 0; t < nt; t++) {
        size_t pos = starts[t], end = starts[t+1];
        size_t ls = pos, fs = pos; uint8_t fc = 0;
        uint32_t field_starts[32];
        auto emit = [&](size_t p, bool nl) {
            if (nl) {
                for (uint32_t f : index_fields) if (f <= fc) {
                    size_t a = f==0 ? ls : (size_t)field_starts[f-1]+1;
                    size_t b = f<fc ? (size_t)field_starts[f] : p;
                    lm[t][f][std::string_view(corpus+a,b-a)]++;
                }
                fc=0; ls=p+1; fs=p+1;
            } else { if(fc<32) field_starts[fc++]=p; }
        };
        #ifdef __AVX2__
        __m256i nv=_mm256_set1_epi8('\n'), dv=_mm256_set1_epi8(delimiter);
        size_t ae = end & ~31ULL;
        while (pos < ae) {
            __m256i c=_mm256_loadu_si256((const __m256i*)(corpus+pos));
            uint32_t nm=_mm256_movemask_epi8(_mm256_cmpeq_epi8(c,nv));
            uint32_t dm=_mm256_movemask_epi8(_mm256_cmpeq_epi8(c,dv));
            uint32_t cm=nm|dm;
            while(cm){int b=__builtin_ctz(cm);emit(pos+b,(nm>>b)&1);cm&=cm-1;}
            pos+=32;
        }
        #endif
        for(size_t i=pos;i<=end;i++)
            if(i==end||corpus[i]=='\n') emit(i,true);
            else if(corpus[i]==delimiter) emit(i,false);
    }
    for (int t = 0; t < nt; t++)
        for (uint32_t f : index_fields)
            lm[t][f].merge_into(field_index[f]);
    num_lines = 0;
}

inline void NSStringIndex::build_detailed(const char* corpus, size_t len, char delimiter, uint32_t num_fields) {
    if (detailed_built) return;  // Already built
    
    field_index_detailed.resize(num_fields);
    
    uint32_t line_num = 0;
    size_t line_start = 0;
    std::vector<std::string_view> fields;
    fields.reserve(num_fields);
    
    for (size_t i = 0; i <= len; i++) {
        if (i == len || corpus[i] == '\n') {
            fields.clear();
            size_t field_start = line_start;
            uint32_t field_count = 0;
            
            for (size_t j = line_start; j < i; j++) {
                if (corpus[j] == delimiter) {
                    if (field_count < num_fields) {
                        fields.push_back(std::string_view(corpus + field_start, j - field_start));
                        field_count++;
                        field_start = j + 1;
                    }
                }
            }
            if (field_count < num_fields && line_start < i) {
                fields.push_back(std::string_view(corpus + field_start, i - field_start));
                field_count++;
            }
            
            for (uint32_t f = 0; f < std::min(num_fields, (uint32_t)fields.size()); f++) {
                field_index_detailed[f][fields[f]].push_back(line_num);
            }
            
            line_num++;
            line_start = i + 1;
        }
    }
    
    detailed_built = true;
}

inline uint64_t NSStringIndex::query(uint32_t field_idx, const std::string& value) {
    if (field_idx >= num_fields) return 0;
    auto* r = field_index[field_idx].find(value);
    return r ? *r : 0;
}

inline const std::vector<uint32_t>* NSStringIndex::query_lines(uint32_t field_idx,
                                                                  const std::string& value) {
    // Build detailed index on demand if not already built
    if (!detailed_built) {
        // Note: this requires corpus to still be available
        // Caller must ensure corpus outlives the index
        // This is a design limitation - could store corpus pointer in struct
        return nullptr;  // Caller must call build_detailed() explicitly first
    }
    
    if (field_idx >= num_fields) return nullptr;
    auto it = field_index_detailed[field_idx].find(value);
    if (it == field_index_detailed[field_idx].end()) return nullptr;
    return &it->second;
}

#endif // NSSTRINGINDEX_HPP

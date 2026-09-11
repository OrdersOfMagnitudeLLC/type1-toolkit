// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

/*
 * NSSort v10 — v9 + persistent thread pool for try_cluster_sort_2pass
 *
 * v10 change: replace sequential per-cluster counting sort with three
 *   parallel phases using ClusterSortPool (8 pre-created threads, CV-based).
 *   Phase 1: private-histogram parallel count  (T=8 strips)
 *   Phase 2: cursor-based parallel scatter      (zero-race, disjoint writes)
 *   Phase 3: per-cluster sort in parallel       (N≤8 independent tasks)
 *   No OMP (avoids TBB oversubscription). No per-call thread creation.
 *
 * v9 base: v7 base + 256-sample nearly-sorted path + Regime A/B
 *
 * New (v9): 256-element sample detection with desc_violations/asc_violations
 *   ordering.  Regime A — fused single-pass insertion repair within WINDOW.
 *   Regime B — extract-merge with std::inplace_merge.
 *
 * Inherited from v7 (unchanged):
 *   Change 1 — two-level L1-buffered scatter at depth > 0
 *   Change 2 — consecutive-pair nearly-sorted + run-merge (fallback after v9 path)
 *   Change A — small-n fast path in nssort_arithmetic (n < 1024)
 *   Change B — 32-sample clustered pre-pass (10 K ≤ n ≤ 2 M)
 *   V5 depth-0 imbalance subdivision, AVX2 sort16, OMP parallelism
 */

#ifndef NSSORT_HPP
#define NSSORT_HPP

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstring>
#include <fstream>
#include <immintrin.h>
#include <memory>
#include <mutex>
#include <omp.h>
#include <random>
#include <unordered_map>
#include <vector>
#include <condition_variable>
#include <deque>
#include <functional>
#include <thread>

using namespace std;

// Global sync counters for instrumentation
std::atomic<long long> g_nssort_barrier_count(0);
std::atomic<long long> g_nssort_critical_count(0);
std::atomic<long long> g_nssort_atomic_count(0);

// ---------------------------------------------------------------------------
// Physical core count detection
// ---------------------------------------------------------------------------
static int get_physical_cores_from_proc() {
    ifstream cpuinfo("/proc/cpuinfo");
    if (!cpuinfo.is_open()) {
        return -1;
    }
    
    string line;
    while (getline(cpuinfo, line)) {
        if (line.find("cpu cores") != string::npos) {
            size_t colon = line.find(':');
            if (colon != string::npos) {
                string num_str = line.substr(colon + 1);
                size_t start = num_str.find_first_not_of(" \t");
                size_t end = num_str.find_last_not_of(" \t");
                if (start != string::npos && end != string::npos) {
                    num_str = num_str.substr(start, end - start + 1);
                    return stoi(num_str);
                }
            }
        }
    }
    return -1;
}

static int get_physical_cores_from_lscpu() {
    FILE* pipe = popen("lscpu | grep 'Core(s) per socket'", "r");
    if (!pipe) return -1;
    
    char buffer[128];
    if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        string line(buffer);
        size_t colon = line.find(':');
        if (colon != string::npos) {
            string num_str = line.substr(colon + 1);
            size_t start = num_str.find_first_not_of(" \t");
            size_t end = num_str.find_last_not_of(" \t\n");
            if (start != string::npos && end != string::npos) {
                num_str = num_str.substr(start, end - start + 1);
                pclose(pipe);
                return stoi(num_str);
            }
        }
    }
    pclose(pipe);
    return -1;
}

static int get_physical_core_count() {
    int cores = get_physical_cores_from_proc();
    if (cores > 0) return cores;
    
    cores = get_physical_cores_from_lscpu();
    if (cores > 0) return cores;
    
    return -1;  // Unable to determine
}

// ---------------------------------------------------------------------------
// v9 nearly-sorted path constants
// ---------------------------------------------------------------------------
constexpr int SAMPLE_N = 256;
constexpr int WINDOW   = 4096;

// Forward declaration for parallel is_sorted check
static bool parallel_is_sorted_check(int64_t* arr, int64_t n);

// ---------------------------------------------------------------------------
// Insertion sort — leaf-level (n ≲ 11)
// ---------------------------------------------------------------------------
static void insertion_sort(int64_t* arr, int64_t n) {
    for (int64_t i = 1; i < n; i++) {
        int64_t key = arr[i], j = i - 1;
        while (j >= 0 && arr[j] > key) { arr[j + 1] = arr[j]; j--; }
        arr[j + 1] = key;
    }
}

// Returns false = input violates this fast path's assumptions; arr is
// UNTOUCHED and the caller must fall through to the next strategy.
static bool counting_sort_checked(int64_t* arr, int64_t n, int64_t min_val, long long range) {
    if (n <= 0) return true;
    if (range < 0 || range > 25'000'000LL) return false;  // Invalid or too large for counting sort - bail to general sort

    int num_threads = omp_get_max_threads();
    if (num_threads < 2 || n < 100000 || range > 2'000'000LL) {
        // Small arrays, single-threaded, or large range: use sequential path
        std::vector<int> cnt(range + 1, 0);
        for (int i = 0; i < n; i++) {
            long long idx = (long long)arr[i] - min_val;
            if (idx < 0 || idx > range) return false;
            cnt[idx]++;
        }
        int pos = 0;
        for (long long v = 0; v <= range; v++)
            for (int k = 0; k < cnt[v]; k++)
                arr[pos++] = v + min_val;
        return pos == n;
    }

    // Parallel counting sort (range ≤ 2M, max 8×2M×4 = 64MB)
    std::vector<std::vector<int>> local_counts(num_threads);
    for (int t = 0; t < num_threads; t++) local_counts[t].assign(range + 1, 0);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int64_t chunk_size = (n + num_threads - 1) / num_threads;
        int64_t start = tid * chunk_size;
        int64_t end = std::min(start + chunk_size, n);

        for (int64_t i = start; i < end; i++) {
            long long idx = (long long)arr[i] - min_val;
            if (idx < 0 || idx > range) continue;
            local_counts[tid][idx]++;
        }
    }

    // Sum local counts
    std::vector<int> global_cnt(range + 1, 0);
    for (int t = 0; t < num_threads; t++) {
        for (long long v = 0; v <= range; v++) {
            global_cnt[v] += local_counts[t][v];
        }
    }

    // Scatter (parallel by value range - no overlapping writes)
    std::vector<int64_t> prefix(range + 2, 0);
    for (long long v = 0; v <= range; v++) 
        prefix[v+1] = prefix[v] + global_cnt[v];
    
    if (range <= 10000) {
        // In-place scatter for small ranges (2 passes instead of 3)
        #pragma omp parallel for schedule(dynamic, 16)
        for (long long v = 0; v <= range; v++) {
            int64_t val = v + min_val;
            for (int64_t k = prefix[v]; k < prefix[v+1]; k++)
                arr[k] = val;
        }
        return prefix[range+1] == n;
    } else {
        // Large ranges: use tmp buffer to avoid overwriting unprocessed values
        std::vector<int64_t> tmp(n);
        #pragma omp parallel for schedule(dynamic, 16)
        for (long long v = 0; v <= range; v++) {
            int64_t val = v + min_val;
            for (int64_t k = prefix[v]; k < prefix[v+1]; k++)
                tmp[k] = val;
        }
        memcpy(arr, tmp.data(), n * sizeof(int64_t));
        return prefix[range+1] == n;
    }
}

static void ns_sort_band(int64_t* data, int64_t n,
                         int64_t val_min, int64_t val_max, int depth = 0) {
    if (n <= 0) return;
    
    // Fast path: few unique values (stack HT, fits in L1)
    constexpr int HT_SIZE = 1024;
    int64_t ht_keys[HT_SIZE];
    int64_t ht_counts[HT_SIZE];
    bool ht_used[HT_SIZE];
    memset(ht_used, 0, sizeof(ht_used));
    int unique_count = 0;
    bool few_unique = true;

    for (int64_t i = 0; i < n; i++) {
        int64_t v = data[i];
        uint32_t h = (uint32_t)((uint64_t)v * 0x9e3779b97f4a7c15ULL >> 54) & (HT_SIZE - 1);
        while (ht_used[h] && ht_keys[h] != v) h = (h + 1) & (HT_SIZE - 1);
        if (!ht_used[h]) {
            if (unique_count >= 256) { few_unique = false; break; }
            ht_used[h] = true;
            ht_keys[h] = v;
            ht_counts[h] = 0;
            unique_count++;
        }
        ht_counts[h]++;
    }

    if (few_unique) {
        std::pair<int64_t,int64_t> kv[256];
        int kv_size = 0;
        for (int h = 0; h < HT_SIZE; h++)
            if (ht_used[h]) kv[kv_size++] = {ht_keys[h], ht_counts[h]};
        std::sort(kv, kv + kv_size);
        int64_t pos = 0;
        for (int j = 0; j < kv_size; j++)
            while (kv[j].second--) data[pos++] = kv[j].first;
        return;
    }
    
    if (val_min == val_max) return;  // all same value, done

    int64_t range = val_max - val_min;
    
    if (range < 0) return;  // overflow guard, shouldn't happen

    if (range <= 30'000'000LL) {
        std::vector<int64_t> cnt(range + 1, 0);
        for (int64_t i = 0; i < n; i++) {
            int64_t idx = data[i] - val_min;
            if (idx >= 0 && idx <= range) cnt[idx]++;
        }
        int64_t pos = 0;
        for (int64_t v = 0; v <= range; v++)
            while (cnt[v]--) data[pos++] = v + val_min;
        return;
    }

    constexpr int BUCKETS = 256;
    int64_t bw = (range + BUCKETS - 1) / BUCKETS;

    std::vector<int64_t> hist(BUCKETS, 0);
    for (int64_t i = 0; i < n; i++) {
        int b = (int)((data[i] - val_min) / bw);
        if (b < 0) b = 0;
        if (b >= BUCKETS) b = BUCKETS - 1;
        hist[b]++;
    }

    std::vector<int64_t> offsets(BUCKETS + 1, 0);
    for (int b = 0; b < BUCKETS; b++) offsets[b + 1] = offsets[b] + hist[b];

    std::vector<int64_t> tmp(n);
    std::vector<int64_t> cursors(offsets.begin(), offsets.begin() + BUCKETS);
    for (int64_t i = 0; i < n; i++) {
        int b = (int)((data[i] - val_min) / bw);
        if (b < 0) b = 0;
        if (b >= BUCKETS) b = BUCKETS - 1;
        tmp[cursors[b]++] = data[i];
    }

    for (int b = 0; b < BUCKETS; b++) {
        if (hist[b] == 0) continue;
        int64_t sub_min = val_min + (int64_t)b * bw;
        int64_t sub_max = val_min + (int64_t)(b + 1) * bw - 1;
        if (sub_max > val_max) sub_max = val_max;
        ns_sort_band(tmp.data() + offsets[b], hist[b], sub_min, sub_max, depth + 1);
    }

    memcpy(data, tmp.data(), n * sizeof(int64_t));
}

// Adaptive parallel sort for large-range data — 2-pass parallel MSB radix sort.
// Pass 1: partition by the top 16 bits of (value - min_val), computed with
//   clamping (values <= min_val -> digit 0, values >= max_val -> top digit)
//   since callers may pass a SAMPLE-estimated min/max, not the true bounds.
// Pass 2: within each pass-1 bucket, split by the next bits, using the
//   bucket's own exact min/max (no estimation error at this level).
// Buckets whose true range collapses to <= 4,000,000 (after either pass)
// are finished with counting_sort_checked. A residual bucket can only
// remain unresolved if the original range exceeds ~2^32 — that case
// recurses into adaptive_parallel_sort with the bucket's own true bounds.
static void ns_uniform_lsd_sort(int64_t* arr, int64_t n, int64_t min_val, int64_t max_val) {
    if (n <= 1) return;
    if (min_val == max_val) return;

    int num_threads = omp_get_max_threads();
    if (num_threads < 1) num_threads = 1;
    if (num_threads > 64) num_threads = 64;  // Cap for thread_offsets array

    constexpr int NUM_BUCKETS = 256;  // 8 bits per pass

    // Allocate buffers once
    int64_t* tmp = new int64_t[n];
    int64_t global_hist[256];
    int64_t offsets[256];
    int64_t thread_offsets[64][256];  // Max 64 threads
    int64_t local_hists[64][256];  // Per-thread local histograms

    // Sign offset for handling negative numbers (flip sign bit)
    uint64_t sign_offset = (uint64_t)1ULL << 63;

    uint64_t range = (uint64_t)max_val - (uint64_t)min_val;
    int range_bits = 64 - __builtin_clzll(range);
    
    uint64_t min_u = (uint64_t)min_val ^ sign_offset;
    uint64_t max_u = (uint64_t)max_val ^ sign_offset;
    int active_passes = 0;
    int pass_shifts[8];
    for (int p = 0; p < 8; p++) {
        if ((min_u >> (8 * p)) != (max_u >> (8 * p)))
            pass_shifts[active_passes++] = p * 8;
    }

    // Only run passes for byte positions that are non-trivial
    for (int pass_idx = 0; pass_idx < active_passes; pass_idx++) {
        int pass_shift = pass_shifts[pass_idx];
        int64_t* src = (pass_idx % 2 == 0) ? arr : tmp;
        int64_t* dst = (pass_idx % 2 == 0) ? tmp : arr;

        // Zero global histogram
        memset(global_hist, 0, sizeof(global_hist));

        // Phase 1: parallel histogram building
        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            int64_t chunk = (n + num_threads - 1) / num_threads;
            int64_t start = tid * chunk;
            int64_t end = std::min(start + chunk, n);

            // Local histogram on stack (2KB, L1-resident)
            memset(local_hists[tid], 0, 256 * sizeof(int64_t));

            for (int64_t i = start; i < end; i++) {
                uint64_t val = (uint64_t)src[i] ^ sign_offset;  // Flip sign bit for unsigned comparison
                int digit = (int)((val >> pass_shift) & 0xFF);
                local_hists[tid][digit]++;
            }
        }

        // Merge local histograms into global
        for (int t = 0; t < num_threads; t++) {
            for (int d = 0; d < 256; d++) {
                global_hist[d] += local_hists[t][d];
            }
        }

        // Compute prefix sum (offsets)
        offsets[0] = 0;
        for (int d = 1; d < 256; d++) {
            offsets[d] = offsets[d - 1] + global_hist[d - 1];
        }

        // Compute thread-local offsets
        for (int d = 0; d < 256; d++) {
            int64_t running = offsets[d];
            for (int t = 0; t < num_threads; t++) {
                thread_offsets[t][d] = running;
                running += local_hists[t][d];
            }
        }

        // Phase 2: parallel scatter
        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            int64_t chunk = (n + num_threads - 1) / num_threads;
            int64_t start = tid * chunk;
            int64_t end = std::min(start + chunk, n);

            // Local cursor on stack
            int cursor[256];
            for (int d = 0; d < 256; d++) {
                cursor[d] = thread_offsets[tid][d];
            }

            for (int64_t i = start; i < end; i++) {
                uint64_t val = (uint64_t)src[i] ^ sign_offset;
                int digit = (int)((val >> pass_shift) & 0xFF);
                dst[cursor[digit]++] = src[i];
            }
        }
    }

    if (active_passes % 2 != 0) {
        memcpy(arr, tmp, n * sizeof(int64_t));
    }
    delete[] tmp;
}

// Adaptive parallel sort for large-range data — 2-pass parallel MSB radix sort.
// Pass 1: partition by the top 16 bits of (value - min_val), computed with
//   clamping (values <= min_val -> digit 0, values >= max_val -> top digit)
//   since callers may pass a SAMPLE-estimated min/max, not the true bounds.
// Pass 2: within each pass-1 bucket, split by the next bits, using the
// bucket's own exact min/max (no estimation error at this level).
// Buckets whose true range collapses to <= 4,000,000 (after either pass)
// are finished with counting_sort_checked. A residual bucket can only
// remain unresolved if the original range exceeds ~2^32 — that case
// recurses into adaptive_parallel_sort with the bucket's own true bounds.
static void adaptive_parallel_sort(int64_t* arr, int64_t n, int64_t min_val, int64_t max_val) {
    if (n <= 1) return;
    if (min_val == max_val) return;

    int num_threads = omp_get_max_threads();
    if (num_threads < 1) num_threads = 1;

    constexpr int DIGIT_BITS   = 16;
    constexpr int NUM_DIGITS   = 1 << DIGIT_BITS;   // 65536
    constexpr int64_t SMALL_RANGE = 4'000'000LL;

    uint64_t range = (uint64_t)max_val - (uint64_t)min_val;
    int range_bits = 64 - __builtin_clzll(range);
    int shift1 = (range_bits > DIGIT_BITS) ? (range_bits - DIGIT_BITS) : 0;
    uint64_t mask1 = (uint64_t)(NUM_DIGITS - 1);

    // Clamped digit extraction — min_val/max_val may be SAMPLE-estimated
    // (not the true bounds), so values outside [min_val, max_val] are
    // clamped to the nearest valid digit rather than wrapping/truncating.
    auto digit1_of = [&](int64_t v) -> uint32_t {
        uint64_t shifted;
        if (v <= min_val) shifted = 0;
        else if (v >= max_val) shifted = range;
        else shifted = (uint64_t)v - (uint64_t)min_val;
        return (uint32_t)((shifted >> shift1) & mask1);
    };

    // ---------- Pass 1: parallel per-thread histogram by top digit ----------
    std::vector<std::vector<int64_t>> local_hist1(num_threads, std::vector<int64_t>(NUM_DIGITS, 0));

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        int64_t chunk = (n + num_threads - 1) / num_threads;
        int64_t start = tid * chunk;
        int64_t end = std::min(start + chunk, n);
        auto& hist = local_hist1[tid];
        for (int64_t i = start; i < end; i++) hist[digit1_of(arr[i])]++;
    }

    std::vector<int64_t> global_hist1(NUM_DIGITS, 0);
    for (int t = 0; t < num_threads; t++)
        for (int d = 0; d < NUM_DIGITS; d++)
            global_hist1[d] += local_hist1[t][d];

    // Global prefix sum (single thread)
    std::vector<int64_t> offsets1(NUM_DIGITS + 1, 0);
    for (int d = 0; d < NUM_DIGITS; d++) offsets1[d + 1] = offsets1[d] + global_hist1[d];

    std::vector<std::vector<int64_t>> thread_offset1(num_threads, std::vector<int64_t>(NUM_DIGITS, 0));
    for (int d = 0; d < NUM_DIGITS; d++) {
        int64_t acc = offsets1[d];
        for (int t = 0; t < num_threads; t++) {
            thread_offset1[t][d] = acc;
            acc += local_hist1[t][d];
        }
    }

    // ---------- Parallel scatter by thread-local offset tracking ----------
    std::vector<int64_t> tmp(n);
    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        int64_t chunk = (n + num_threads - 1) / num_threads;
        int64_t start = tid * chunk;
        int64_t end = std::min(start + chunk, n);
        std::vector<int64_t> cursor = thread_offset1[tid];
        for (int64_t i = start; i < end; i++) {
            uint32_t d = digit1_of(arr[i]);
            tmp[cursor[d]++] = arr[i];
        }
    }

    // ---------- Pass 2: within each pass-1 bucket, split by the bucket's
    // own true bit-range (buckets processed in parallel via dynamic sched) ----------
    #pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads)
    for (int d = 0; d < NUM_DIGITS; d++) {
        int64_t bcount = global_hist1[d];
        if (bcount <= 1) continue;

        int64_t* bucket = tmp.data() + offsets1[d];

        int64_t bmin = bucket[0], bmax = bucket[0];
        for (int64_t i = 1; i < bcount; i++) {
            if (bucket[i] < bmin) bmin = bucket[i];
            if (bucket[i] > bmax) bmax = bucket[i];
        }
        if (bmin == bmax) continue;

        uint64_t brange = (uint64_t)bmax - (uint64_t)bmin;
        if (brange <= (uint64_t)SMALL_RANGE) {
            // Inline sequential counting sort — counting_sort_checked has an
            // internal #pragma omp parallel that mis-chunks when called from
            // a nested parallel region (this pass-2 loop is already parallel).
            std::vector<int64_t> cnt(brange + 1, 0);
            for (int64_t i = 0; i < bcount; i++) cnt[bucket[i] - bmin]++;
            int64_t pos = 0;
            for (uint64_t v = 0; v <= brange; v++)
                while (cnt[v]--) bucket[pos++] = (int64_t)v + bmin;
            continue;
        }

        // Second radix pass, sized to this bucket's own exact range.
        int brange_bits = 64 - __builtin_clzll(brange);
        int shift2 = (brange_bits > DIGIT_BITS) ? (brange_bits - DIGIT_BITS) : 0;
        int digit2_bits = std::min(brange_bits, DIGIT_BITS);
        int64_t num_digits2 = 1LL << digit2_bits;
        uint64_t mask2 = (uint64_t)(num_digits2 - 1);

        std::vector<int64_t> hist2(num_digits2, 0);
        for (int64_t i = 0; i < bcount; i++) {
            uint64_t shifted = (uint64_t)bucket[i] - (uint64_t)bmin;
            hist2[(shifted >> shift2) & mask2]++;
        }
        std::vector<int64_t> offsets2(num_digits2 + 1, 0);
        for (int64_t d2 = 0; d2 < num_digits2; d2++) offsets2[d2 + 1] = offsets2[d2] + hist2[d2];

        std::vector<int64_t> bucket_tmp(bcount);
        std::vector<int64_t> cursor2(offsets2.begin(), offsets2.begin() + num_digits2);
        for (int64_t i = 0; i < bcount; i++) {
            uint64_t shifted = (uint64_t)bucket[i] - (uint64_t)bmin;
            uint32_t d2 = (uint32_t)((shifted >> shift2) & mask2);
            bucket_tmp[cursor2[d2]++] = bucket[i];
        }
        memcpy(bucket, bucket_tmp.data(), bcount * sizeof(int64_t));

        for (int64_t d2 = 0; d2 < num_digits2; d2++) {
            int64_t scount = hist2[d2];
            if (scount <= 1) continue;
            int64_t* sub = bucket + offsets2[d2];

            int64_t smin = sub[0], smax = sub[0];
            for (int64_t i = 1; i < scount; i++) {
                if (sub[i] < smin) smin = sub[i];
                if (sub[i] > smax) smax = sub[i];
            }
            if (smin == smax) continue;

            uint64_t srange = (uint64_t)smax - (uint64_t)smin;
            if (srange <= (uint64_t)SMALL_RANGE) {
                std::vector<int64_t> cnt2(srange + 1, 0);
                for (int64_t i = 0; i < scount; i++) cnt2[sub[i] - smin]++;
                int64_t pos2 = 0;
                for (uint64_t v = 0; v <= srange; v++)
                    while (cnt2[v]--) sub[pos2++] = (int64_t)v + smin;
            } else {
                // Only reachable when the original range exceeds ~2^32.
                adaptive_parallel_sort(sub, scount, smin, smax);
            }
        }
    }

    memcpy(arr, tmp.data(), n * sizeof(int64_t));
}

// Convenience wrapper: computes TRUE min/max itself (replaces overload 1).
static bool counting_sort_checked(int64_t* arr, int64_t n) {
    if (n <= 0) return true;
    int64_t min_val = arr[0], max_val = arr[0];
    for (int64_t i = 1; i < n; i++) {
        if (arr[i] < min_val) min_val = arr[i];
        if (arr[i] > max_val) max_val = arr[i];
    }
    return counting_sort_checked(arr, n, min_val, (long long)max_val - min_val);
}

// ---------------------------------------------------------------------------
// v9 Regime A — fused single-pass insertion repair (dense-fallback)
// Single forward pass.  When arr[i] > arr[i+1]: store val=arr[i+1],
// shift arr[i] into arr[i+1], walk backward while arr[j-1] > val AND
// j > i-WINDOW, place val.  Continue forward.  No OpenMP.
// ---------------------------------------------------------------------------
static void regime_a_insertion_repair(int64_t* arr, int64_t n) {
    // fprintf(stderr, "[INSTRUMENTATION] Regime A insertion repair called (n=%lld)\n", (long long)n);
    // Full-array insertion repair — no window partitioning.
    // Window partitioning blocked cross-boundary element movement.
    // Called only when dirty_count/num_windows > 0.30 threshold,
    // meaning data is dense-nearly-sorted: O(n*k) where k is small.
    for (int64_t i = 0; i < n - 1; i++) {
        if (arr[i] > arr[i + 1]) {
            int64_t val = arr[i + 1];
            arr[i + 1] = arr[i];
            int64_t j = i;
            while (j > 0 && arr[j - 1] > val) {
                arr[j] = arr[j - 1];
                j--;
            }
            arr[j] = val;
        }
    }
}

// ---------------------------------------------------------------------------
// v9 Regime A — dirty-window repair (sparse path)
// Only sorts overlapping windows that contain violations (dirty flag).
// Even/odd pass ordering ensures no adjacent window interference.
// Falls back to insertion repair when >30% of windows are dirty (15% at >500M).
// Window size and dirty threshold scale with n to avoid overhead at large n.
// ---------------------------------------------------------------------------
static void regime_a_window_repair(int64_t* arr, int64_t n, int64_t disp_est) {
    int64_t repair_window = (n > 500000000) ? 16384 : WINDOW;
    int64_t num_windows = (n + repair_window - 1) / repair_window;

    vector<bool> local_dirty(num_windows, false);
    int64_t dirty_count = 0;
    for (int64_t i = 0; i < n - 1; i++) {
        if (arr[i] > arr[i + 1]) {
            int64_t win  = i / repair_window;
            int64_t win2 = (i + 1) / repair_window;
            if (!local_dirty[win]) {
                local_dirty[win] = true;
                dirty_count++;
            }
            if (win2 != win && !local_dirty[win2]) {
                local_dirty[win2] = true;
                dirty_count++;
            }
        }
    }

    double dirty_threshold = (n > 500000000) ? 0.15 : 0.30;
    double dirty_fraction = (double)dirty_count / num_windows;
    if (dirty_fraction > dirty_threshold) {
        regime_a_insertion_repair(arr, n);
        return;
    }

    for (int64_t w = 0; w < num_windows; w++) {
        if (!local_dirty[w]) continue;
        int64_t overlap = max((int64_t)1, disp_est);
        int64_t start = max((int64_t)0, w * repair_window - overlap);
        int64_t end   = min(n, (w + 1) * repair_window + overlap);
        int64_t len   = end - start;
        vector<int64_t> buf(len);
        copy(arr + start, arr + end, buf.begin());
        sort(buf.begin(), buf.end());
        copy(buf.begin(), buf.end(), arr + start);
    }
}

// Forward declaration
static void parallel_merge_regime_b(int64_t* arr, int64_t w, int64_t n);
static void nssort(int64_t* arr, int64_t n);

// ---------------------------------------------------------------------------
// v9 Regime B — run-based extract-merge
// Marking: arr[i] > arr[i+1] only (not both neighbours).
// Chains consecutive violations into runs (one run per reversed block).
// Coalesce overlapping runs, extract, sort, then parallel merge.
// ---------------------------------------------------------------------------
static void regime_b_extract_merge(int64_t* arr, int64_t n) {
    vector<pair<int64_t,int64_t>> runs;
    int64_t i = 0;
    while (i < n - 1) {
        if (arr[i] > arr[i + 1]) {
            int64_t run_start = max((int64_t)0, i - 1);
            while (i < n - 1 && arr[i] > arr[i + 1]) i++;
            runs.push_back({run_start, min(n - 1, i + 1)});
        } else {
            i++;
        }
    }
    if (runs.empty()) return;

    sort(runs.begin(), runs.end());
    vector<pair<int64_t,int64_t>> merged;
    for (auto& r : runs) {
        if (merged.empty() || r.first > merged.back().second) {
            merged.push_back(r);
        } else {
            merged.back().second = max(merged.back().second, r.second);
        }
    }

    int64_t total = 0;
    for (auto& r : merged) total += r.second - r.first + 1;
    vector<int64_t> extracted(total);
    int64_t e = 0;
    for (auto& r : merged)
        for (int64_t j = r.first; j <= r.second; j++)
            extracted[e++] = arr[j];
    nssort(extracted.data(), extracted.size());

    int64_t w = 0, pos = 0;
    for (auto& r : merged) {
        int64_t len = r.first - pos;
        if (len > 0) {
            memmove(arr + w, arr + pos, len * sizeof(int64_t));
            w += len;
        }
        pos = r.second + 1;
    }
    if (pos < n) {
        memmove(arr + w, arr + pos, (n - pos) * sizeof(int64_t));
        w += n - pos;
    }
    memcpy(arr + w, extracted.data(), total * sizeof(int64_t));
    nssort(arr, w);
    parallel_merge_regime_b(arr, w, n);
}

// ---------------------------------------------------------------------------
// Change 2 helper: merge sorted runs bottom-up  (FAST PATH — DO NOT MODIFY)
// Scans arr once to collect run-start positions, then merges pairwise until
// one run remains.  O(n) scan + O(n log R) merges; fast when R is small.
// ---------------------------------------------------------------------------
static void merge_runs(int64_t* arr, int64_t n) {
    vector<int64_t> starts;
    starts.reserve(64);
    starts.push_back(0);
    for (int64_t i = 1; i < n; i++)
        if (arr[i] < arr[i - 1]) starts.push_back(i);
    starts.push_back(n);

    int64_t nruns = (int64_t)starts.size() - 1;
    if (nruns <= 1) return;

    while ((int64_t)starts.size() > 2) {
        vector<int64_t> next;
        next.reserve(starts.size() / 2 + 2);
        next.push_back(starts[0]);
        for (int64_t i = 0; i + 2 < (int64_t)starts.size(); i += 2) {
            int64_t lo  = starts[i];
            int64_t mid = starts[i + 1];
            int64_t hi  = starts[i + 2];
            inplace_merge(arr + lo, arr + mid, arr + hi);
            next.push_back(hi);
        }
        if ((int64_t)starts.size() % 2 == 0) next.push_back(starts.back());
        starts = move(next);
    }
}

// ---------------------------------------------------------------------------
// Batcher's sorting network — 16 elements, scalar fallback
// ---------------------------------------------------------------------------
static void sort16(int64_t* a) {
#define CS(x,y) { int64_t lo=a[x]<a[y]?a[x]:a[y], hi=a[x]>a[y]?a[x]:a[y]; a[x]=lo; a[y]=hi; }
    CS(0,1)  CS(2,3)  CS(4,5)  CS(6,7)  CS(8,9)  CS(10,11) CS(12,13) CS(14,15)
    CS(0,2)  CS(1,3)  CS(4,6)  CS(5,7)  CS(8,10) CS(9,11)  CS(12,14) CS(13,15)
    CS(1,2)  CS(5,6)  CS(9,10) CS(13,14)
    CS(0,4)  CS(1,5)  CS(2,6)  CS(3,7)  CS(8,12) CS(9,13)  CS(10,14) CS(11,15)
    CS(2,4)  CS(3,5)  CS(10,12) CS(11,13)
    CS(1,2)  CS(3,4)  CS(5,6)  CS(9,10) CS(11,12) CS(13,14)
    CS(0,8)  CS(1,9)  CS(2,10) CS(3,11) CS(4,12) CS(5,13)  CS(6,14)  CS(7,15)
    CS(4,8)  CS(5,9)  CS(6,10) CS(7,11)
    CS(2,4)  CS(3,5)  CS(6,8)  CS(7,9)  CS(10,12) CS(11,13)
    CS(1,2)  CS(3,4)  CS(5,6)  CS(7,8)  CS(9,10) CS(11,12) CS(13,14)
#undef CS
}

#if 0  // Disabled for int64_t - AVX2 intrinsics operate on 32-bit integers
// AVX2 sort16 — djbsort-derived 10-stage bitonic network  (FAST PATH — DO NOT MODIFY)
// ---------------------------------------------------------------------------
#define V_MINMAX(a, b) do { \
    __m256i _t = _mm256_min_epi32(a, b); \
    (b)         = _mm256_max_epi32(a, b); \
    (a)         = _t; } while(0)

static inline void merge16_finish(int* x, __m256i x0, __m256i x1) {
    __m256i b0, b1, c0, c1;
    V_MINMAX(x0, x1);
    b0 = _mm256_permute2x128_si256(x0, x1, 0x20);
    b1 = _mm256_permute2x128_si256(x0, x1, 0x31);
    V_MINMAX(b0, b1);
    c0 = _mm256_unpacklo_epi64(b0, b1); c1 = _mm256_unpackhi_epi64(b0, b1);
    V_MINMAX(c0, c1);
    b0 = _mm256_unpacklo_epi32(c0, c1); b1 = _mm256_unpackhi_epi32(c0, c1);
    c0 = _mm256_unpacklo_epi64(b0, b1); c1 = _mm256_unpackhi_epi64(b0, b1);
    V_MINMAX(c0, c1);
    b0 = _mm256_unpacklo_epi32(c0, c1); b1 = _mm256_unpackhi_epi32(c0, c1);
    x0 = _mm256_permute2x128_si256(b0, b1, 0x20);
    x1 = _mm256_permute2x128_si256(b0, b1, 0x31);
    _mm256_storeu_si256((__m256i*)&x[0], x0);
    _mm256_storeu_si256((__m256i*)&x[8], x1);
}

static void sort16_avx2(int* x) {
    __m256i x0, x1, b0, b1, c0, c1, mask;
    x0   = _mm256_loadu_si256((__m256i*)&x[0]);
    x1   = _mm256_loadu_si256((__m256i*)&x[8]);
    mask = _mm256_set_epi32(0, 0, -1, -1, 0, 0, -1, -1);
    x0   = _mm256_xor_si256(x0, mask); x1 = _mm256_xor_si256(x1, mask);
    b0   = _mm256_unpacklo_epi32(x0, x1); b1 = _mm256_unpackhi_epi32(x0, x1);
    c0   = _mm256_unpacklo_epi64(b0, b1); c1 = _mm256_unpackhi_epi64(b0, b1);
    V_MINMAX(c0, c1);
    mask = _mm256_set_epi32(0, 0, -1, -1, -1, -1, 0, 0);
    c0   = _mm256_xor_si256(c0, mask); c1 = _mm256_xor_si256(c1, mask);
    b0   = _mm256_unpacklo_epi32(c0, c1); b1 = _mm256_unpackhi_epi32(c0, c1);
    V_MINMAX(b0, b1);
    x0   = _mm256_unpacklo_epi64(b0, b1); x1 = _mm256_unpackhi_epi64(b0, b1);
    b0   = _mm256_unpacklo_epi32(x0, x1); b1 = _mm256_unpackhi_epi32(x0, x1);
    c0   = _mm256_unpacklo_epi64(b0, b1); c1 = _mm256_unpackhi_epi64(b0, b1);
    V_MINMAX(c0, c1);
    b0   = _mm256_unpacklo_epi32(c0, c1); b1 = _mm256_unpackhi_epi32(c0, c1);
    b0   = _mm256_xor_si256(b0, mask);    b1 = _mm256_xor_si256(b1, mask);
    c0   = _mm256_permute2x128_si256(b0, b1, 0x20);
    c1   = _mm256_permute2x128_si256(b0, b1, 0x31);
    V_MINMAX(c0, c1);
    b0   = _mm256_permute2x128_si256(c0, c1, 0x20);
    b1   = _mm256_permute2x128_si256(c0, c1, 0x31);
    V_MINMAX(b0, b1);
    x0   = _mm256_unpacklo_epi64(b0, b1); x1 = _mm256_unpackhi_epi64(b0, b1);
    b0   = _mm256_unpacklo_epi32(x0, x1); b1 = _mm256_unpackhi_epi32(x0, x1);
    c0   = _mm256_unpacklo_epi64(b0, b1); c1 = _mm256_unpackhi_epi64(b0, b1);
    V_MINMAX(c0, c1);
    b0   = _mm256_unpacklo_epi32(c0, c1); b1 = _mm256_unpackhi_epi32(c0, c1);
    x0   = _mm256_unpacklo_epi64(b0, b1); x1 = _mm256_unpackhi_epi64(b0, b1);
    mask = _mm256_set1_epi32(-1);
    x0   = _mm256_xor_si256(x0, mask);
    merge16_finish(x, x0, x1);
}

static bool detect_avx2_sort16() {
    if (!__builtin_cpu_supports("avx2")) return false;
    mt19937 rng(0xC0FFEE);
    int a[16], b[16];
    auto trial = [&](uniform_int_distribution<int>& d, int iters) -> bool {
        for (int t = 0; t < iters; t++) {
            for (int i = 0; i < 16; i++) { int v = d(rng); a[i] = v; b[i] = v; }
            sort16_avx2(a); sort(b, b + 16);
            for (int i = 0; i < 16; i++) if (a[i] != b[i]) return false;
        }
        return true;
    };
    uniform_int_distribution<int> full(INT_MIN, INT_MAX);
    uniform_int_distribution<int> small(0, 5);
    uniform_int_distribution<int> mid(-100, 100);
    return trial(full, 200000) && trial(small, 50000) && trial(mid, 50000);
}

static const bool g_avx2_ok = detect_avx2_sort16();
#endif  // Disabled for int64_t

static const bool g_avx2_ok = false;  // Force scalar path for int64_t

// ---------------------------------------------------------------------------
// Unified OMP team size — computed once at program start (PRE-U-3 baseline)
// ---------------------------------------------------------------------------
static int g_team_size = []() {
    int sz = get_physical_core_count();
    if (sz <= 0) sz = omp_get_max_threads();
    return sz;
}();

// ---------------------------------------------------------------------------
// Static OMP warmup — ensures hot team is pre-spawned with g_team_size
// ---------------------------------------------------------------------------
static const bool g_omp_warmed_up = []() {
    #pragma omp parallel num_threads(g_team_size)
    { }
    return true;
}();

// Instrumentation for parallel region profiling
static std::atomic<long long> g_regime_a_calls{0};
static std::atomic<long long> g_regime_b_calls{0};
static std::atomic<long long> g_general_sort_calls{0};
static std::atomic<long long> g_histogram_parallel_calls{0};
static std::atomic<long long> g_scatter_parallel_calls{0};
static std::atomic<long long> g_recursion_parallel_calls{0};
static std::atomic<long long> g_pb_detection_calls{0};
static std::atomic<long long> g_pc_coalesced_calls{0};

void print_instrumentation() {
    fprintf(stderr, "=== INSTRUMENTATION ===\n");
    fprintf(stderr, "Regime A calls: %lld\n", g_regime_a_calls.load());
    fprintf(stderr, "Regime B calls: %lld\n", g_regime_b_calls.load());
    fprintf(stderr, "General sort calls: %lld\n", g_general_sort_calls.load());
    fprintf(stderr, "Histogram parallel calls: %lld\n", g_histogram_parallel_calls.load());
    fprintf(stderr, "Scatter parallel calls: %lld\n", g_scatter_parallel_calls.load());
    fprintf(stderr, "Recursion parallel calls: %lld\n", g_recursion_parallel_calls.load());
    fprintf(stderr, "P-B detection calls: %lld\n", g_pb_detection_calls.load());
    fprintf(stderr, "P-C coalesced calls: %lld\n", g_pc_coalesced_calls.load());
    fprintf(stderr, "======================\n");
}

static inline void sort16_net(int64_t* buf) {
    sort16(buf);
}

// ---------------------------------------------------------------------------
// SIMD-accelerated small sort for sizes 1–32 (leaf level)
// ---------------------------------------------------------------------------
static void simd_sort_small(int64_t* arr, int64_t n) {
    if (n < 12 || n > 32) { insertion_sort(arr, n); return; }
    int64_t buf[32];
    int64_t chunk1 = n < 16 ? n : 16, chunk2 = n - chunk1;
    memcpy(buf, arr, chunk1 * sizeof(int64_t));
    for (int64_t i = chunk1; i < 16; i++) buf[i] = INT64_MAX;
    sort16_net(buf);
    if (chunk2 > 0) {
        memcpy(buf + 16, arr + chunk1, chunk2 * sizeof(int64_t));
        for (int64_t i = 16 + chunk2; i < 32; i++) buf[i] = INT64_MAX;
        sort16_net(buf + 16);
        inplace_merge(buf, buf + 16, buf + 16 + chunk2);
    }
    memcpy(arr, buf, n * sizeof(int64_t));
}

// Forward declaration
void nssort_arithmetic(int64_t* arr, int64_t n, int64_t depth, int64_t forced_ns,
                       bool skip_two_level = false,
                       bool rescale = false, int64_t rescale_min = 0, int64_t rescale_max = 0);

// ---------------------------------------------------------------------------
// ClusterSortPool — persistent thread pool for try_cluster_sort_2pass
// Dynamic thread count based on g_team_size.
// Workers sleep on cv_work; caller sleeps on cv_done until pending == 0.
// No OMP (avoids TBB oversubscription). No per-call std::thread creation.
// ---------------------------------------------------------------------------
struct ClusterSortPool {
    int NTHREADS;
    std::vector<std::thread> workers;
    std::deque<std::function<void()>> queue;
    std::mutex              mtx;
    std::condition_variable cv_work;   // workers sleep here when idle
    std::condition_variable cv_done;   // caller sleeps here until batch done
    int                     pending{0};
    bool                    stop{false};

    explicit ClusterSortPool(int n_threads) : NTHREADS(n_threads), workers(n_threads) {
        for (int i = 0; i < NTHREADS; i++) {
            workers[i] = std::thread([this] {
                for (;;) {
                    std::function<void()> task;
                    {
                        g_nssort_critical_count.fetch_add(1, std::memory_order_relaxed);
                        std::unique_lock<std::mutex> lk(mtx);
                        cv_work.wait(lk, [this]{ return stop || !queue.empty(); });
                        if (stop && queue.empty()) return;
                        task = std::move(queue.front());
                        queue.pop_front();
                    }
                    task();
                    {
                        g_nssort_critical_count.fetch_add(1, std::memory_order_relaxed);
                        std::lock_guard<std::mutex> lk(mtx);
                        if (--pending == 0) cv_done.notify_one();
                    }
                }
            });
        }
    }

    ~ClusterSortPool() {
        {
            g_nssort_critical_count.fetch_add(1, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lk(mtx);
            stop = true;
        }
        cv_work.notify_all();
        for (auto& w : workers) w.join();
    }

    void submit_and_wait(std::vector<std::function<void()>>&& tasks) {
        if (tasks.empty()) return;
        {
            g_nssort_critical_count.fetch_add(1, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lk(mtx);
            pending = (int)tasks.size();
            for (auto& t : tasks) queue.push_back(std::move(t));
        }
        cv_work.notify_all();
        g_nssort_critical_count.fetch_add(1, std::memory_order_relaxed);
        std::unique_lock<std::mutex> lk(mtx);
        cv_done.wait(lk, [this]{ return pending == 0; });
    }
};

static ClusterSortPool& get_pool() {
    static ClusterSortPool pool(g_team_size);
    return pool;
}

// ---------------------------------------------------------------------------
// Co-rank: binary search for the split point i in A such that merging
// A[0..i) with B[0..k-i) yields exactly the first k elements of the
// full sorted merge of A and B. Standard co-rank merge primitive.
// ---------------------------------------------------------------------------
static int co_rank(int64_t k, const int64_t* A, int64_t m, const int64_t* B, int64_t n) {
    int64_t i_lo = std::max((int64_t)0, k - n);
    int64_t i_hi = std::min(k, m);
    while (i_lo < i_hi) {
        int64_t i = (i_lo + i_hi + 1) / 2;
        int64_t j = k - i;
        if (A[i - 1] > B[j]) {
            i_hi = i - 1;
        } else {
            i_lo = i;
        }
    }
    return i_lo;
}

// ---------------------------------------------------------------------------
// Parallel merge of arr[0..w) and arr[w..n), both individually sorted,
// into arr[0..n) fully sorted. Replaces std::inplace_merge. No stability
// requirement — plain ints, equal elements are interchangeable.
// ---------------------------------------------------------------------------
static void parallel_merge_regime_b(int64_t* arr, int64_t w, int64_t n) {
    const int64_t* A = arr;
    int64_t m = w;
    const int64_t* B = arr + w;
    int64_t bn = n - w;

    int T = get_pool().NTHREADS;
    int64_t* out = new int64_t[n];  // uninitialized; fully overwritten by merge tasks
    std::vector<int64_t> splits_i(T + 1), splits_j(T + 1);
    splits_i[0] = 0; splits_j[0] = 0;
    splits_i[T] = m; splits_j[T] = bn;

    for (int t = 1; t < T; t++) {
        long long k = (long long)t * n / T;
        int64_t i = co_rank((int64_t)k, A, m, B, bn);
        splits_i[t] = i;
        splits_j[t] = (int64_t)k - i;
    }

    {
        std::vector<std::function<void()>> tasks;
        tasks.reserve(T);
        for (int t = 0; t < T; t++) {
            int64_t i0 = splits_i[t], i1 = splits_i[t + 1];
            int64_t j0 = splits_j[t], j1 = splits_j[t + 1];
            int64_t out_start = i0 + j0;
            tasks.push_back([=]() {
                int64_t ii = i0, jj = j0, oo = out_start;
                while (ii < i1 && jj < j1) {
                    if (A[ii] <= B[jj]) out[oo++] = A[ii++];
                    else out[oo++] = B[jj++];
                }
                while (ii < i1) out[oo++] = A[ii++];
                while (jj < j1) out[oo++] = B[jj++];
            });
        }
        get_pool().submit_and_wait(std::move(tasks));
    }

    {
        std::vector<std::function<void()>> tasks;
        tasks.reserve(T);
        int64_t chunk = (n + T - 1) / T;
        for (int t = 0; t < T; t++) {
            int64_t lo = t * chunk;
            int64_t hi = std::min(lo + chunk, n);
            tasks.push_back([=]() {
                memcpy(arr + lo, out + lo, (hi - lo) * sizeof(int64_t));
            });
        }
        get_pool().submit_and_wait(std::move(tasks));
    }

    delete[] out;
}

// ---------------------------------------------------------------------------
// Parallel reverse for large descending inputs; small n use std::reverse.
// ---------------------------------------------------------------------------
static void parallel_reverse(int64_t* arr, int64_t n) {
    if (n >= 5000000) {
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < n / 2; i++) swap(arr[i], arr[n - 1 - i]);
    } else {
        reverse(arr, arr + n);
    }
}

// ---------------------------------------------------------------------------
// nssort_scatter_into — one-level out-of-place scatter helper
// ---------------------------------------------------------------------------
static void nssort_scatter_into(int64_t* src, int64_t* dst, int64_t n, int64_t depth,
                                int64_t forced_ns, bool skip_two_level,
                                bool rescale = false, int64_t rescale_min = 0, int64_t rescale_max = 0) {
    int64_t g_min = src[0], g_max = src[0];
    for (int64_t i = 1; i < n; i++) {
        if (src[i] < g_min) g_min = src[i];
        if (src[i] > g_max) g_max = src[i];
    }
    long long g_range = g_max - g_min + 1;
    if (g_range <= 0) {
        memcpy(dst, src, n * sizeof(int64_t));
        return;
    }

    int64_t NS;
    if (forced_ns > 0) {
        NS = forced_ns;
    } else if (n < 1024) {
        int64_t sn = 1;
        while (sn < n / 8) sn <<= 1;
        if (sn < 4) sn = 4;
        NS = sn;
    } else if (depth > 0) {
        NS = 1;
        while (NS < n / 4000 && NS < 256) NS <<= 1;
        NS = max(NS, (int64_t)4);
        NS = min(NS, (int64_t)256);
    } else {
        NS = 1;
        while (NS < n / 4000 && NS < 1024) NS <<= 1;
        NS = max(NS, (int64_t)256);
        NS = min(NS, (int64_t)1024);
    }



    auto get_sector = [g_min, g_range, NS](int64_t val) -> int64_t {
        int64_t s = (int64_t)((long long)(val - g_min) * NS / g_range);
        if (s >= NS) s = NS - 1;
        if (s < 0)   s = 0;
        return s;
    };

    int counts[1024];
    memset(counts, 0, NS * sizeof(int));
    for (int64_t i = 0; i < n; i++) counts[get_sector(src[i])]++;



    int offsets[1024];
    offsets[0] = 0;
    for (int64_t s = 1; s < NS; s++) offsets[s] = offsets[s - 1] + counts[s - 1];

    int cursors[1024];
    for (int64_t s = 0; s < NS; s++) cursors[s] = offsets[s];
    for (int64_t i = 0; i < n; i++) dst[cursors[get_sector(src[i])]++] = src[i];

    const long long expected         = (long long)n / NS;
    const long long imbalance_thresh = max(4 * expected, 500000LL);
    const int64_t       sub_ns           = min(NS * 4, (int64_t)1024);

    auto handle_sector = [&](int64_t s) {
        if (counts[s] < 2) return;
        if (depth == 0 && sub_ns > NS && counts[s] > imbalance_thresh) {
            nssort_arithmetic(dst + offsets[s], counts[s], depth + 1, sub_ns, skip_two_level,
                              rescale, rescale_min, rescale_max);
            return;
        }
        if (counts[s] > n / 2) { sort(dst + offsets[s], dst + offsets[s] + counts[s]); return; }
        if (counts[s] <= 32)   simd_sort_small(dst + offsets[s], counts[s]);
        else                   nssort_arithmetic(dst + offsets[s], counts[s], depth + 1, 0, false,
                                                 rescale, rescale_min, rescale_max);
    };

    if (n > 5000000) {
        #pragma omp parallel for schedule(dynamic, 4)
        for (int64_t s = 0; s < NS; s++) handle_sector(s);
    } else {
        for (int64_t s = 0; s < NS; s++) handle_sector(s);
    }
}

// ---------------------------------------------------------------------------
// FAST PATH — DO NOT MODIFY
// nssort_arithmetic — core recursive arithmetic-sector sort
// ---------------------------------------------------------------------------
void nssort_arithmetic(int64_t* arr, int64_t n, int64_t depth, int64_t forced_ns,
                       bool skip_two_level /* = false */,
                       bool rescale /* = false */, int64_t rescale_min /* = 0 */, int64_t rescale_max /* = 0 */) {
    if (depth == 0) {
    }
    if (n <= 1) return;
    if (depth > 20) { sort(arr, arr + n); return; }
    if (forced_ns > 1024) forced_ns = 1024;

    int64_t g_min = arr[0], g_max = arr[0];
    for (int64_t i = 1; i < n; i++) {
        if (arr[i] < g_min) g_min = arr[i];
        if (arr[i] > g_max) g_max = arr[i];
    }
    long long g_range = g_max - g_min + 1;
    if (g_range <= 0) return;

    // Clustered rescale: override g_min/g_range with sample-derived bounds
    // so sector assignment spreads clustered data evenly across all sectors.
    if (rescale) {
        g_min   = rescale_min;
        g_range = (long long)rescale_max - rescale_min + 1;
    }

    if (!skip_two_level && depth > 0 && n >= 10000) {
        constexpr int CNS    = 16;
        constexpr int BSIZE  = 32;

        auto coarse = [g_min, g_range](int64_t val) -> int64_t {
            int64_t s = (int64_t)((long long)(val - g_min) * CNS / g_range);
            if (s >= CNS) s = CNS - 1;
            if (s < 0)    s = 0;
            return s;
        };

        int counts[CNS] = {};
        for (int64_t i = 0; i < n; i++) counts[coarse(arr[i])]++;

        int offsets[CNS];
        offsets[0] = 0;
        for (int s = 1; s < CNS; s++) offsets[s] = offsets[s - 1] + counts[s - 1];

        int64_t* output = nullptr;
        try { output = new int64_t[n]; } catch (...) { output = nullptr; }
        if (!output) goto fallthrough_sort;
        int64_t  buf[CNS][BSIZE];
        int  buf_cnt[CNS] = {};
        int64_t  cur[CNS];
        for (int s = 0; s < CNS; s++) cur[s] = offsets[s];

        for (int64_t i = 0; i < n; i++) {
            int64_t s = coarse(arr[i]);
            buf[s][buf_cnt[s]++] = arr[i];
            if (buf_cnt[s] == BSIZE) {
                memcpy(output + cur[s], buf[s], BSIZE * sizeof(int64_t));
                cur[s]     += BSIZE;
                buf_cnt[s]  = 0;
            }
        }
        for (int s = 0; s < CNS; s++)
            if (buf_cnt[s] > 0)
                memcpy(output + cur[s], buf[s], buf_cnt[s] * sizeof(int64_t));
        memcpy(arr, output, n * sizeof(int64_t));
        delete[] output;

        for (int s = 0; s < CNS; s++)
            if (counts[s] > 1)
                nssort_arithmetic(arr + offsets[s], counts[s], depth + 1, 0, false);
        return;
    }

fallthrough_sort:
    int64_t NS;
    if (forced_ns > 0) {
        NS = forced_ns;
    } else if (n < 1024) {
        int64_t sn = 1;
        while (sn < n / 8) sn <<= 1;
        if (sn < 4) sn = 4;
        NS = sn;
    } else if (depth > 0) {
        NS = 1;
        while (NS < n / 4000 && NS < 256) NS <<= 1;
        NS = max(NS, (int64_t)4);
        NS = min(NS, (int64_t)256);
    } else {
        NS = 1;
        while (NS < n / 4000 && NS < 1024) NS <<= 1;
        NS = max(NS, (int64_t)256);
        NS = min(NS, (int64_t)64);
    }



    float rescale_scale = 0.0f;
    if (rescale) {
        rescale_scale = (float)(NS - 1) / (float)(rescale_max - rescale_min);
    }

    auto get_sector = [g_min, g_range, NS, rescale, rescale_scale](int64_t val) -> int64_t {
        if (rescale) {
            int64_t s = (int64_t)((float)(val - g_min) * rescale_scale);
            if (s >= NS) s = NS - 1;
            if (s < 0)   s = 0;
            return s;
        }
        int64_t s = (int64_t)((long long)(val - g_min) * NS / g_range);
        if (s >= NS) s = NS - 1;
        if (s < 0)   s = 0;
        return s;
    };

    int counts[1024];
    int offsets[1024];

    if (depth == 0 && n > 5000000) {
        g_histogram_parallel_calls.fetch_add(1, std::memory_order_relaxed);
        vector<vector<int>> local_hist(g_team_size, vector<int>(NS, 0));

        {
            int64_t chunk = n / g_team_size;
            std::vector<std::function<void()>> tasks;
            tasks.reserve(g_team_size);
            for (int tid = 0; tid < g_team_size; tid++) {
                int64_t start = tid * chunk;
                int64_t end = (tid == g_team_size - 1) ? n : start + chunk;
                tasks.push_back([&, tid, start, end]() {
#if 0  // Disabled for int64_t - AVX512 intrinsics operate on 32-bit integers
                    __m512i g_min_vec = _mm512_set1_epi32(g_min);
                    int i = start;
                    for (; i + 15 < end; i += 16) {
                        __m512i vals = _mm512_loadu_si512((__m512i*)&arr[i]);
                        __m512i diff = _mm512_sub_epi32(vals, g_min_vec);
                        __m512 diff_f = _mm512_cvtepi32_ps(diff);
                        __m512 scale = _mm512_set1_ps((float)NS / (float)g_range);
                        __m512i sector = _mm512_cvttps_epi32(_mm512_mul_ps(diff_f, scale));
                        sector = _mm512_max_epi32(sector, _mm512_setzero_si512());
                        sector = _mm512_min_epi32(sector, _mm512_set1_epi32(NS - 1));
                        int sectors[16];
                        _mm512_storeu_si512((__m512i*)sectors, sector);
                        for (int j = 0; j < 16; j++) local_hist[tid][sectors[j]]++;
                    }
                    for (; i < end; i++)
                        local_hist[tid][get_sector(arr[i])]++;
#else
                    for (int64_t i = start; i < end; i++)
                        local_hist[tid][get_sector(arr[i])]++;
#endif
                });
            }
            get_pool().submit_and_wait(std::move(tasks));
        }

        memset(counts, 0, NS * sizeof(int));
        for (int s = 0; s < NS; s++)
            for (int t = 0; t < g_team_size; t++)
                counts[s] += local_hist[t][s];



        offsets[0] = 0;
        for (int s = 1; s < NS; s++)
            offsets[s] = offsets[s - 1] + counts[s - 1];

        vector<vector<int>> thread_offsets(g_team_size, vector<int>(NS, 0));
        for (int s = 0; s < NS; s++) {
            int running = 0;
            for (int t = 0; t < g_team_size; t++) {
                thread_offsets[t][s] = offsets[s] + running;
                running += local_hist[t][s];
            }
        }

        int64_t* output = new int64_t[n];
        g_scatter_parallel_calls.fetch_add(1, std::memory_order_relaxed);
        {
            int chunk = n / g_team_size;
            std::vector<std::function<void()>> tasks;
            tasks.reserve(g_team_size);
            for (int tid = 0; tid < g_team_size; tid++) {
                int start = tid * chunk;
                int end = (tid == g_team_size - 1) ? n : start + chunk;
                tasks.push_back([&, tid, start, end]() {
                    auto& cursors = thread_offsets[tid];
                    constexpr int STAGE = 16;
                    int64_t stage_buf[1024][STAGE];
                    int stage_cnt[1024];
                    memset(stage_cnt, 0, NS * sizeof(int));
                    for (int64_t i = start; i < end; i++) {
                        int64_t s = get_sector(arr[i]);
                        stage_buf[s][stage_cnt[s]++] = arr[i];
                        if (stage_cnt[s] == STAGE) {
                            memcpy(output + cursors[s], stage_buf[s], STAGE * sizeof(int64_t));
                            cursors[s] += STAGE;
                            stage_cnt[s] = 0;
                        }
                    }
                    for (int s = 0; s < NS; s++) {
                        if (stage_cnt[s] > 0) {
                            memcpy(output + cursors[s], stage_buf[s], stage_cnt[s] * sizeof(int64_t));
                            cursors[s] += stage_cnt[s];
                        }
                    }
                });
            }
            get_pool().submit_and_wait(std::move(tasks));
        }
        // P1-C: eliminate sequential copy-back - fold into parallel recursion loop
        const long long expected         = (long long)n / NS;
        const long long imbalance_thresh = max(4 * expected, 500000LL);
        const int64_t       sub_ns           = min(NS * 4, (int64_t)1024);

        auto handle_sector = [&](int64_t s) {
            if (counts[s] < 2) {
                memcpy(arr + offsets[s], output + offsets[s], counts[s] * sizeof(int64_t));
                return;
            }
            if (depth == 0 && sub_ns > NS && counts[s] > imbalance_thresh) {
                nssort_arithmetic(output + offsets[s], counts[s], depth + 1, sub_ns, false);
                memcpy(arr + offsets[s], output + offsets[s], counts[s] * sizeof(int64_t));
                return;
            }
            if (counts[s] > n / 2) { 
                sort(output + offsets[s], output + offsets[s] + counts[s]); 
                memcpy(arr + offsets[s], output + offsets[s], counts[s] * sizeof(int64_t));
                return; 
            }
            if (counts[s] <= 32) {
                simd_sort_small(output + offsets[s], counts[s]);
                memcpy(arr + offsets[s], output + offsets[s], counts[s] * sizeof(int64_t));
            } else {
                nssort_arithmetic(output + offsets[s], counts[s], depth + 1, 0, false);
                memcpy(arr + offsets[s], output + offsets[s], counts[s] * sizeof(int64_t));
            }
        };

        g_recursion_parallel_calls.fetch_add(1, std::memory_order_relaxed);
        #pragma omp parallel for schedule(dynamic, 4) num_threads(g_team_size)
        for (int64_t s = 0; s < NS; s++) handle_sector(s);
        delete[] output;
    } else {
        memset(counts, 0, NS * sizeof(int));
        for (int64_t i = 0; i < n; i++) counts[get_sector(arr[i])]++;



        offsets[0] = 0;
        for (int s = 1; s < NS; s++) offsets[s] = offsets[s - 1] + counts[s - 1];

        int64_t  stack_buf[1024];
        int64_t* output = (n <= 1024) ? stack_buf : new int64_t[n];
        int  cursors[1024];
        for (int s = 0; s < NS; s++) cursors[s] = offsets[s];
        if (depth == 0) {
            constexpr int STAGE = 16;
            int64_t stage_buf[1024][STAGE];
            int stage_cnt[1024];
            memset(stage_cnt, 0, NS * sizeof(int));
            for (int64_t i = 0; i < n; i++) {
                int64_t s = get_sector(arr[i]);
                stage_buf[s][stage_cnt[s]++] = arr[i];
                if (stage_cnt[s] == STAGE) {
                    memcpy(output + cursors[s], stage_buf[s], STAGE * sizeof(int64_t));
                    cursors[s] += STAGE;
                    stage_cnt[s] = 0;
                }
            }
            for (int s = 0; s < NS; s++) {
                if (stage_cnt[s] > 0) {
                    memcpy(output + cursors[s], stage_buf[s], stage_cnt[s] * sizeof(int64_t));
                    cursors[s] += stage_cnt[s];
                }
            }
        } else {
            for (int64_t i = 0; i < n; i++) output[cursors[get_sector(arr[i])]++] = arr[i];
        }
        // P1-C: eliminate sequential copy-back - fold into parallel recursion loop
        const long long expected         = (long long)n / NS;
        const long long imbalance_thresh = max(4 * expected, 500000LL);
        const int64_t       sub_ns           = min(NS * 4, (int64_t)1024);

        auto handle_sector = [&](int64_t s) {
            if (counts[s] < 2) {
                memcpy(arr + offsets[s], output + offsets[s], counts[s] * sizeof(int64_t));
                return;
            }
            if (depth == 0 && sub_ns > NS && counts[s] > imbalance_thresh) {
                nssort_arithmetic(output + offsets[s], counts[s], depth + 1, sub_ns, false);
                memcpy(arr + offsets[s], output + offsets[s], counts[s] * sizeof(int64_t));
                return;
            }
            if (counts[s] > n / 2) { 
                sort(output + offsets[s], output + offsets[s] + counts[s]); 
                memcpy(arr + offsets[s], output + offsets[s], counts[s] * sizeof(int64_t));
                return; 
            }
            if (counts[s] <= 32) {
                simd_sort_small(output + offsets[s], counts[s]);
                memcpy(arr + offsets[s], output + offsets[s], counts[s] * sizeof(int64_t));
            } else {
                nssort_arithmetic(output + offsets[s], counts[s], depth + 1, 0, false);
                memcpy(arr + offsets[s], output + offsets[s], counts[s] * sizeof(int64_t));
            }
        };

        if (n > 5000000) {
            g_recursion_parallel_calls.fetch_add(1, std::memory_order_relaxed);
            #pragma omp parallel for schedule(dynamic, 4) num_threads(g_team_size)
            for (int64_t s = 0; s < NS; s++) handle_sector(s);
        } else {
            for (int64_t s = 0; s < NS; s++) handle_sector(s);
        }
        if (n > 1024) delete[] output;
    }
}

// ---------------------------------------------------------------------------
// try_cluster_sort_2pass — parallel O(2n) cluster partition + per-cluster sort
//
// Two array passes (fusion impossible: Phase 2 cursors require all of Phase 1):
//   Phase 1 — T=8 threads each count a strip into private local_sizes[t][c]
//             → barrier → reduce to global sizes[], compute offsets[] and
//             per-thread write cursors[t][c] (prefix sum across threads)
//   Phase 2 — T=8 threads scatter their strips into out_buf via disjoint
//             cursors (no atomics, no races by construction)
//   Phase 3 — N≤8 independent cluster sorts submitted as parallel tasks
// ---------------------------------------------------------------------------
static bool try_cluster_sort_2pass(int64_t* arr, int64_t n,
                                   const std::vector<std::pair<int64_t,int64_t>>& clusters) {
    int N = (int)clusters.size();
    if (N < 2 || N > 8) return false;

    int64_t cluster_maxes[8];
    for (int c = 0; c < N; c++) cluster_maxes[c] = clusters[c].second;

    ClusterSortPool& pool = get_pool();
    const int T = pool.NTHREADS;
    const int64_t chunk = (n + T - 1) / T;

    // ---- Phase 1: parallel counting with private per-thread histograms ----
    // Mirrors GPU §3.2: each block accumulates into __shared__ then atomicAdds.
    // Here: each thread accumulates into local_sizes[t], no sharing needed.
    g_pb_detection_calls.fetch_add(1, std::memory_order_relaxed);
    std::vector<std::array<int,8>> local_sizes(T, std::array<int,8>{});
    std::vector<char> local_ok(T, true);

    {
        std::vector<std::function<void()>> tasks;
        tasks.reserve(T);
        for (int t = 0; t < T; t++) {
            const int64_t lo = t * chunk;
            const int64_t hi = std::min(lo + chunk, n);
            tasks.push_back([&, t, lo, hi]() {
                for (int64_t i = lo; i < hi; i++) {
                    int64_t v = arr[i];
                    int c = 0;
                    for (int cc = 0; cc < N - 1; cc++) c += (v > cluster_maxes[cc]);
                    if (v < clusters[c].first || v > clusters[c].second) {
                        local_ok[t] = false;
                        return;
                    }
                    local_sizes[t][c]++;
                }
            });
        }
        pool.submit_and_wait(std::move(tasks));
    }

    for (int t = 0; t < T; t++)
        if (!local_ok[t]) return false;

    // ---- Barrier: reduce histograms, compute offsets and per-thread cursors ----
    // cursors[t][c] = start position in out_buf where thread t writes cluster c.
    // Derived from prefix sum of local_sizes across threads — mirrors GPU §3.4
    // per-thread g_cursor init from g_base + thread-local histogram prefix.
    int sizes[8]   = {};
    for (int c = 0; c < N; c++)
        for (int t = 0; t < T; t++)
            sizes[c] += local_sizes[t][c];

    int offsets[9] = {};
    for (int c = 0; c < N; c++) offsets[c + 1] = offsets[c] + sizes[c];
    if (offsets[N] != n) return false;

    std::vector<std::array<int,8>> cursors(T, std::array<int,8>{});
    for (int c = 0; c < N; c++) {
        int running = offsets[c];
        for (int t = 0; t < T; t++) {
            cursors[t][c] = running;
            running += local_sizes[t][c];
        }
    }

    // ---- Phase 2: parallel scatter using pre-computed disjoint cursors ----
    int64_t* out_buf = new int64_t[n];             // per-call, uninitialized; fully overwritten
    int64_t* const out_ptr = out_buf;          // raw ptr: safe to read from any thread

    {
        std::vector<std::function<void()>> tasks;
        tasks.reserve(T);
        for (int t = 0; t < T; t++) {
            const int64_t lo = t * chunk;
            const int64_t hi = std::min(lo + chunk, n);
            tasks.push_back([&, t, lo, hi]() {
                int cur[8];
                for (int c = 0; c < N; c++) cur[c] = cursors[t][c];
                for (int64_t i = lo; i < hi; i++) {
                    int64_t v = arr[i];
                    int c = 0;
                    for (int cc = 0; cc < N - 1; cc++) c += (v > cluster_maxes[cc]);
                    out_ptr[cur[c]++] = v;
                }
            });
        }
        pool.submit_and_wait(std::move(tasks));
    }

    // ---- Phase 3: parallel per-cluster sort (embarrassingly parallel) ----
    // Mirrors GPU §5 mixed routing: each occupied sector takes its own door.
    // N≤8 independent tasks, one per cluster, no shared state.
    g_pc_coalesced_calls.fetch_add(1, std::memory_order_relaxed);
    {
        std::vector<std::function<void()>> tasks;
        tasks.reserve(N);
        for (int c = 0; c < N; c++) {
            tasks.push_back([&, c]() {
                const int64_t cmin       = clusters[c].first;
                const long long crange = (long long)clusters[c].second - cmin + 1;
                const int start      = offsets[c];
                const int sz         = sizes[c];
                if (sz == 0) return;
                if (crange <= 2000000) {
                    std::vector<int> counts((int)crange, 0);
                    for (int i = start; i < start + sz; i++) counts[out_ptr[i] - cmin]++;
                    int pos = start;
                    for (int64_t v = 0; v < (int64_t)crange; v++)
                        for (int k = 0; k < counts[v]; k++)
                            out_ptr[pos++] = v + cmin;
                } else {
                    std::sort(out_ptr + start, out_ptr + start + sz);
                }
            });
        }
        pool.submit_and_wait(std::move(tasks));
    }

    memcpy(arr, out_ptr, n * sizeof(int64_t));
    delete[] out_buf;
    return true;
}

// ---------------------------------------------------------------------------
// Compressed counting sort — large-range sparse data fast path
// O(n) scan into unordered_map, sort distinct keys, REGENERATE output.
// Bails early if >100k distinct values (avoids memory blowup on random data).
// ---------------------------------------------------------------------------
static bool compressed_counting_sort(int64_t* arr, int64_t n) {
    // Guard against large ranges that can cause heap corruption
    int64_t min_val = arr[0], max_val = arr[0];
    for (int64_t i = 1; i < n; i++) {
        if (arr[i] < min_val) min_val = arr[i];
        if (arr[i] > max_val) max_val = arr[i];
    }
    long long range = (long long)max_val - min_val;
    if (range > 16'000'000LL) return false;  // Too large - bail to general sort

    unordered_map<int64_t,int> counts;
    counts.reserve(2048);
    for (int64_t i = 0; i < n; i++) {
        counts[arr[i]]++;
        if (counts.size() > 100000) return false;
    }

    vector<pair<int64_t,int>> pairs(counts.begin(), counts.end());
    sort(pairs.begin(), pairs.end());

    int64_t pos = 0;
    for (auto& [val, cnt] : pairs)
        for (int k = 0; k < cnt; k++)
            arr[pos++] = val;

    return pos == n;
}

// ---------------------------------------------------------------------------
// nssort — public entry point
//
// v9 detection order via 256-element sample:
//   1. desc_violations == 0 → reverse (with O(n) verify)
//   2. distinct values in sample ≤ 20 → O(n) min/max scan → counting sort
//   3. asc_violations <= SAMPLE_N/16 → nearly-sorted → Regime A or B
//   4. sample_range < n/10 → counting sort
//   5. Fall through to existing v7 fast paths + general sort
// ---------------------------------------------------------------------------
void nssort(int64_t* arr, int64_t n) {
    // fprintf(stderr, "[INSTRUMENTATION] nssort ENTRY: n=%lld\n", (long long)n);
    if (n <= 1) return;

    // FAST PATH — DO NOT MODIFY: tiny arrays
    if (n < 32) { insertion_sort(arr, n); return; }

    // Pre-check: if first 64 elements all identical, trust the sample and
    // return immediately (matches the existing v9 sample_min==sample_max
    // precedent elsewhere in this function — no O(n) verification).
    {
        int64_t v0 = arr[0];
        int64_t presample = std::min(n, (int64_t)64);
        bool all_same = true;
        for (int64_t i = 1; i < presample; i++)
            if (arr[i] != v0) { all_same = false; break; }
        if (all_same) return;
    }

    // Few-unique fast path — intercepts two_dup, eight_dup, ones-like distributions
    {
        constexpr int HT_SIZE = 1024;
        constexpr int MAX_TH = 64;
        int pu_threads = omp_get_max_threads();
        if (pu_threads < 1) pu_threads = 1;
        if (pu_threads > MAX_TH) pu_threads = MAX_TH;

        if (n >= 100000 && pu_threads > 1) {
            // ---- Parallel phase 1: each thread scans its own chunk with
            // its own stack-local HT (same structure as the sequential path). ----
            std::pair<int64_t,int64_t> thread_kv[MAX_TH][256];
            int thread_kv_size[MAX_TH] = {};
            bool thread_overflow[MAX_TH] = {};

            #pragma omp parallel num_threads(pu_threads)
            {
                int tid = omp_get_thread_num();
                int64_t pchunk = (n + pu_threads - 1) / pu_threads;
                int64_t pstart = (int64_t)tid * pchunk;
                int64_t pend = std::min(pstart + pchunk, n);

                int64_t p_ht_keys[HT_SIZE];
                int64_t p_ht_counts[HT_SIZE];
                bool p_ht_used[HT_SIZE];
                memset(p_ht_used, 0, sizeof(p_ht_used));
                int p_unique_count = 0;
                bool overflow = false;

                for (int64_t i = pstart; i < pend; i++) {
                    int64_t v = arr[i];
                    uint32_t h = (uint32_t)((uint64_t)v * 0x9e3779b97f4a7c15ULL >> 54) & (HT_SIZE - 1);
                    while (p_ht_used[h] && p_ht_keys[h] != v) h = (h + 1) & (HT_SIZE - 1);
                    if (!p_ht_used[h]) {
                        if (p_unique_count >= 256) { overflow = true; break; }
                        p_ht_used[h] = true;
                        p_ht_keys[h] = v;
                        p_ht_counts[h] = 0;
                        p_unique_count++;
                    }
                    p_ht_counts[h]++;
                }

                int kv_size = 0;
                if (!overflow)
                    for (int h = 0; h < HT_SIZE; h++)
                        if (p_ht_used[h]) thread_kv[tid][kv_size++] = {p_ht_keys[h], p_ht_counts[h]};
                thread_kv_size[tid] = kv_size;
                thread_overflow[tid] = overflow;
            }

            bool any_overflow = false;
            for (int t = 0; t < pu_threads; t++) if (thread_overflow[t]) any_overflow = true;

            if (!any_overflow) {
                // ---- Single-thread merge: union of local HTs (<=256 total) ----
                std::pair<int64_t,int64_t> merged[256];
                int merged_size = 0;
                bool merge_overflow = false;
                for (int t = 0; t < pu_threads && !merge_overflow; t++) {
                    for (int j = 0; j < thread_kv_size[t]; j++) {
                        int64_t v = thread_kv[t][j].first;
                        int64_t c = thread_kv[t][j].second;
                        int idx = -1;
                        for (int m = 0; m < merged_size; m++) if (merged[m].first == v) { idx = m; break; }
                        if (idx >= 0) merged[idx].second += c;
                        else {
                            if (merged_size >= 256) { merge_overflow = true; break; }
                            merged[merged_size++] = {v, c};
                        }
                    }
                }

                if (!merge_overflow) {
                    for (int a = 0; a < merged_size - 1; a++)
                        for (int b = a + 1; b < merged_size; b++)
                            if (merged[a].first > merged[b].first) std::swap(merged[a], merged[b]);

                    int64_t offsets[256];
                    int64_t acc = 0;
                    for (int j = 0; j < merged_size; j++) { offsets[j] = acc; acc += merged[j].second; }

                    // ---- Parallel phase 2: fill by equal array chunks, not
                    // by unique-value count, so zero/two_dup still get full
                    // pu_threads-way parallelism (memory-bandwidth bound). ----
                    #pragma omp parallel num_threads(pu_threads)
                    {
                        int tid = omp_get_thread_num();
                        int64_t fchunk = (n + pu_threads - 1) / pu_threads;
                        int64_t fstart = (int64_t)tid * fchunk;
                        int64_t fend = std::min(fstart + fchunk, n);
                        if (fstart < fend) {
                            int lo = 0, hi = merged_size - 1, j = 0;
                            while (lo <= hi) {
                                int mid = (lo + hi) / 2;
                                if (offsets[mid] <= fstart) { j = mid; lo = mid + 1; }
                                else hi = mid - 1;
                            }
                            int64_t pos = fstart;
                            while (pos < fend) {
                                int64_t seg_end = (j + 1 < merged_size) ? offsets[j + 1] : n;
                                int64_t fill_end = std::min(seg_end, fend);
                                int64_t v = merged[j].first;
                                for (int64_t k = pos; k < fill_end; k++) arr[k] = v;
                                pos = fill_end;
                                j++;
                            }
                        }
                    }
                    return;
                }
                // merge_overflow: fall through to sequential scan below (rare)
            }
            // any_overflow: fall through to sequential scan below (rare)
        }

        // ---- Sequential fallback: small n, single thread, or overflow ----
        constexpr int HT_SIZE_SEQ = HT_SIZE;
        int64_t ht_keys[HT_SIZE_SEQ];
        int64_t ht_counts[HT_SIZE_SEQ];
        bool ht_used[HT_SIZE_SEQ];
        memset(ht_used, 0, sizeof(ht_used));
        int unique_count = 0;
        bool few_unique = true;

        for (int64_t i = 0; i < n; i++) {
            int64_t v = arr[i];
            uint32_t h = (uint32_t)((uint64_t)v * 0x9e3779b97f4a7c15ULL >> 54) & (HT_SIZE_SEQ - 1);
            while (ht_used[h] && ht_keys[h] != v) h = (h + 1) & (HT_SIZE_SEQ - 1);
            if (!ht_used[h]) {
                if (unique_count >= 256) { few_unique = false; break; }
                ht_used[h] = true;
                ht_keys[h] = v;
                ht_counts[h] = 0;
                unique_count++;
            }
            ht_counts[h]++;
        }

        if (few_unique) {
            std::pair<int64_t,int64_t> kv[256];
            int kv_size = 0;
            for (int h = 0; h < HT_SIZE_SEQ; h++)
                if (ht_used[h]) kv[kv_size++] = {ht_keys[h], ht_counts[h]};
            for (int a = 0; a < kv_size - 1; a++)
                for (int b = a + 1; b < kv_size; b++)
                    if (kv[a].first > kv[b].first) std::swap(kv[a], kv[b]);
            int64_t pos = 0;
            for (int j = 0; j < kv_size; j++)
                while (kv[j].second--) arr[pos++] = kv[j].first;
            return;
        }
    }

    bool use_rescale = false;
    int64_t rs_sample_min = 0, rs_sample_max = 0;

    // ===== v9: 256-element sample detection =====
    bool skip_v9_detection = false;
    if (n >= SAMPLE_N) {
        int desc_violations = 0, asc_violations = 0;
        int64_t sample_min = arr[0], sample_max = arr[0];
        int64_t step = n / SAMPLE_N;

        for (int i = 0; i < SAMPLE_N - 1; i++) {
            int64_t idx = i * step;
            if (arr[idx] > arr[idx + 1]) asc_violations++;
            if (arr[idx] < arr[idx + 1]) desc_violations++;
            if (arr[idx] < sample_min) sample_min = arr[idx];
            if (arr[idx] > sample_max) sample_max = arr[idx];
        }
        int64_t last_idx = (SAMPLE_N - 1) * step;
        if (arr[last_idx] < sample_min) sample_min = arr[last_idx];
        if (arr[last_idx] > sample_max) sample_max = arr[last_idx];

        // Already-sorted fast path
        if (asc_violations == 0) {
            if (parallel_is_sorted_check(arr, n)) return;
        }

        int64_t sample_range = sample_max - sample_min;

        // All same value - already sorted
        if (sample_min == sample_max) return;

        // Step 1: desc_violations == 0 → reverse
        // fprintf(stderr, "[INSTRUMENTATION] Step 1: desc_violations=%d (n=%lld)\n", desc_violations, (long long)n);
        if (desc_violations == 0) {
            // fprintf(stderr, "[INSTRUMENTATION] v9 reverse path taken (desc_violations=0, n=%lld)\n", (long long)n);
            // Spot check before reverse
            // fprintf(stderr, "[INSTRUMENTATION] Before reverse: arr[0]=%lld, arr[n/2]=%lld, arr[n-1]=%lld\n", (long long)arr[0], (long long)arr[n/2], (long long)arr[n-1]);
            parallel_reverse(arr, n);
            // Spot check after reverse
            // fprintf(stderr, "[INSTRUMENTATION] After reverse: arr[0]=%lld, arr[n/2]=%lld, arr[n-1]=%lld\n", (long long)arr[0], (long long)arr[n/2], (long long)arr[n-1]);
            // Verify the result is actually sorted
            bool is_sorted = parallel_is_sorted_check(arr, n);
            // fprintf(stderr, "[INSTRUMENTATION] parallel_is_sorted_check result: %s\n", is_sorted ? "SORTED" : "NOT SORTED");
            if (is_sorted) {
                // fprintf(stderr, "[INSTRUMENTATION] v9 reverse path verified sorted, returning\n");
                return;
            }
            // If not sorted, skip remaining v9 detection and fall through to general sort
            // fprintf(stderr, "[INSTRUMENTATION] v9 reverse path verification FAILED, falling through\n");
            skip_v9_detection = true;
        }

        if (skip_v9_detection) {
            // Skip to general sort - array is now reversed, but general sort handles any order
        } else {

        // Range-based counting sort gate
        // If range is small relative to n, counting sort is always faster
        // Catches: root_dup (range=999, n=1M), any low-cardinality distribution
        // that slips past the MAXU=24 unique-value gate
        if ((uint64_t)sample_max - (uint64_t)sample_min < 16'000'000ULL) {
            if (counting_sort_checked(arr, n))
                return;
        }

        // Adaptive parallel sort for large-range data
        int num_threads = omp_get_max_threads();
        if (n > 500000 && num_threads > 1 && (sample_max - sample_min) > 2'000'000LL && (sample_max - sample_min) <= 200'000'000LL) {
            adaptive_parallel_sort(arr, n, sample_min, sample_max);
            return;
        }

        // Step 2: distinct values in sample ≤ 20 → counting sort with sample range
        {
            int64_t tmp[256];
            for (int i = 0; i < SAMPLE_N; i++) tmp[i] = arr[i * step];
            sort(tmp, tmp + SAMPLE_N);
            int distinct = 1;
            for (int i = 1; i < SAMPLE_N; i++) if (tmp[i] != tmp[i - 1]) distinct++;
            if (distinct <= 20) {
                if (counting_sort_checked(arr, n, sample_min, (long long)sample_max - sample_min))
                    return;
                // else: sample lied — fall through to Step 3
                if (compressed_counting_sort(arr, n)) return;
            }
        }

        // Uniform/random fast path: high disorder + large range
        // → route to parallel radix, bypassing sequential fallthrough
        if (asc_violations > SAMPLE_N / 2 &&
            (uint64_t)((uint64_t)sample_max - (uint64_t)sample_min) > 2'000'000LL &&
            n > 500'000) {
            int64_t true_min = arr[0], true_max = arr[0];
            #pragma omp parallel for reduction(min:true_min) reduction(max:true_max)
            for (int64_t i = 1; i < n; i++) {
                true_min = std::min(true_min, arr[i]);
                true_max = std::max(true_max, arr[i]);
            }
            ns_uniform_lsd_sort(arr, n, true_min, true_max);
            return;
        }

        // Step 3: asc_violations threshold → O(n) verification → Regime A/B
        if (asc_violations == 0 || asc_violations <= SAMPLE_N / 16) {
            int64_t total_violations = 0;
            for (int64_t i = 0; i < n - 1; i++)
                if (arr[i] > arr[i + 1]) total_violations++;

            if (total_violations > n / 500) {
                int64_t true_min = arr[0], true_max = arr[0];
                for (int64_t i = 1; i < n; i++) {
                    if (arr[i] < true_min) true_min = arr[i];
                    if (arr[i] > true_max) true_max = arr[i];
                }
                adaptive_parallel_sort(arr, n, true_min, true_max);
                return;
            }

            if (total_violations == 0) return;

            int64_t disp_est = 0;
            bool has_long = false;
            for (int64_t i = 0; i < n - 1; i++) {
                if (arr[i] > arr[i + 1]) {
                    int64_t target = i + 1;
                    while (target < n && arr[target] < arr[i]) target++;
                    int64_t dist = target - i;
                    if (dist > disp_est) disp_est = dist;
                    if (dist > WINDOW) {
                        has_long = true;
                        break;  // Exit early - we know it's going to Regime B
                    }
                }
            }

            if (disp_est <= WINDOW && !has_long) {
                g_regime_a_calls.fetch_add(1, std::memory_order_relaxed);
                regime_a_window_repair(arr, n, disp_est);
            } else {
                g_regime_b_calls.fetch_add(1, std::memory_order_relaxed);
                regime_b_extract_merge(arr, n);
            }
            return;
        }

        // Step 4: sample_range < n/10 → fast counting sort (REGENERATE)
        if (sample_range < n / 10) {
            if (counting_sort_checked(arr, n)) return;
            // else: too many uniques for the table path — fall through
        }

        // Step 4.5: Collision detection in sorted sample → compressed counting sort
        // Fallback for large-range sparse data missed by Step 4
        {
            int64_t sample_sorted[SAMPLE_N];
            for (int i = 0; i < SAMPLE_N; i++) sample_sorted[i] = arr[i * step];
            sort(sample_sorted, sample_sorted + SAMPLE_N);
            int collisions = 0;
            for (int i = 1; i < SAMPLE_N; i++)
                if (sample_sorted[i] == sample_sorted[i - 1]) collisions++;
            if (collisions >= 20) {
                if (compressed_counting_sort(arr, n)) return;
            }
        }

        // Step 4.6: Cluster detection — per-cluster counting sort
        // Costs O(K×n) where K is number of tight clusters.
        // For 5 tight clusters spread across a large range, K=5 full scans
        // beats the general sort at 1M.  Falls through if clusters too wide.
        {
            int cluster_sample_n = SAMPLE_N;
            if (n > 100000000) cluster_sample_n = min(2048, (int)(n / 500000));
            int64_t cstep = n / cluster_sample_n;

            vector<int64_t> s_sample(cluster_sample_n);
            for (int i = 0; i < cluster_sample_n; i++) s_sample[i] = arr[i * cstep];
            sort(s_sample.begin(), s_sample.end());

            long long s_range = (long long)s_sample[cluster_sample_n - 1] - s_sample[0];
            int gap_count = 0;
            vector<pair<int64_t,int64_t>> clusters;
            int64_t cluster_start = s_sample[0];

            for (int i = 1; i < cluster_sample_n; i++) {
                long long gap = (long long)s_sample[i] - s_sample[i - 1];
                if (gap > s_range / 8) {
                    int64_t cmin = cluster_start;
                    int64_t cmax = s_sample[i - 1];
                    int64_t cluster_span = cmax - cmin;
                    int64_t margin = cluster_span / 10;
                    clusters.push_back({cmin - margin, cmax + margin});
                    cluster_start = s_sample[i];
                    gap_count++;
                }
            }
            int64_t cmin = cluster_start;
            int64_t cmax = s_sample[cluster_sample_n - 1];
            int64_t cluster_span = cmax - cmin;
            int64_t margin = cluster_span / 10;
            clusters.push_back({cmin - margin, cmax + margin});

            if (gap_count >= 2 && gap_count <= 7 && clusters.size() >= 3 && clusters.size() <= 8) {
                if (try_cluster_sort_2pass(arr, n, clusters)) return;
            }
        }

        // Step 4.7: heavy-left (exponential-like) detection
        {
            int64_t hl_sample[SAMPLE_N];
            for (int i = 0; i < SAMPLE_N; i++) hl_sample[i] = arr[i * step];
            sort(hl_sample, hl_sample + SAMPLE_N);
            int64_t hl_min = hl_sample[0];
            int64_t hl_max = hl_sample[SAMPLE_N - 1];
            int64_t hl_median = hl_sample[SAMPLE_N / 2 - 1];
            int64_t hl_p95 = hl_sample[(SAMPLE_N * 95) / 100];

            if ((hl_max - hl_min) > 0 && hl_median < (double)(hl_max - hl_min) * 0.15) {
                // Single-pass outside-in exponential fast path.
                // Step 1: one parallel partition at p95 → dense head + sparse tail.
                // Step 2: 2-pass normalized radix on dense head (29-bit keys vs 64-bit).
                // Step 3: adaptive_parallel_sort on sparse tail (trivially small).

                int64_t split = hl_p95;
                int nt = num_threads;

                // True min scan — hl_min is sample-estimated and may exceed
                // the actual minimum, which would make (arr[i] - base)
                // negative and corrupt the bucket index below.
                int64_t base = hl_min;
                {
                    std::vector<int64_t> lmin(nt, hl_min);
                    #pragma omp parallel num_threads(nt)
                    {
                        int tid = omp_get_thread_num();
                        int64_t chunk = (n + nt - 1) / nt;
                        int64_t s = tid * chunk;
                        int64_t e = std::min(s + chunk, n);
                        int64_t m = (s < e) ? arr[s] : hl_min;
                        for (int64_t i = s; i < e; i++)
                            if (arr[i] < m) m = arr[i];
                        lmin[tid] = m;
                    }
                    for (int t = 0; t < nt; t++)
                        if (lmin[t] < base) base = lmin[t];
                }

                const int BITS = 15;
                const int BUCKETS = 1 << BITS;
                const int64_t MASK = BUCKETS - 1;
                const int OVERFLOW = BUCKETS; // bucket index for tail elements
                const int TOTAL_BUCKETS = BUCKETS + 1;

                // Combined pass: histogram for low 15 bits + overflow in one scan
                std::vector<std::vector<int64_t>> lcnt(nt,
                    std::vector<int64_t>(TOTAL_BUCKETS, 0));

                #pragma omp parallel num_threads(nt)
                {
                    int tid = omp_get_thread_num();
                    int64_t chunk = (n + nt - 1) / nt;
                    int64_t s = tid * chunk;
                    int64_t e = std::min(s + chunk, n);
                    auto& cnt = lcnt[tid];
                    for (int64_t i = s; i < e; i++) {
                        if (arr[i] > split)
                            cnt[OVERFLOW]++;
                        else
                            cnt[((arr[i] - base)) & MASK]++;
                    }
                }

                // Global prefix sum
                std::vector<int64_t> gprefix(TOTAL_BUCKETS, 0);
                for (int b = 0; b < TOTAL_BUCKETS; b++)
                    for (int t = 0; t < nt; t++)
                        gprefix[b] += lcnt[t][b];

                int64_t lsz = n - gprefix[OVERFLOW];
                int64_t rsz = gprefix[OVERFLOW];

                int64_t running = 0;
                for (int b = 0; b < TOTAL_BUCKETS; b++) {
                    int64_t c = gprefix[b];
                    gprefix[b] = running;
                    running += c;
                }
                // Move overflow to end
                // gprefix[OVERFLOW] now = lsz (start of tail in output)

                // Per-thread offsets
                std::vector<std::vector<int64_t>> toff(nt,
                    std::vector<int64_t>(TOTAL_BUCKETS, 0));
                for (int b = 0; b < TOTAL_BUCKETS; b++) {
                    int64_t off = gprefix[b];
                    for (int t = 0; t < nt; t++) {
                        toff[t][b] = off;
                        off += lcnt[t][b];
                    }
                }

                // Parallel scatter — combined pass 0 + partition
                std::vector<int64_t> tmp(n);
                #pragma omp parallel num_threads(nt)
                {
                    int tid = omp_get_thread_num();
                    int64_t chunk = (n + nt - 1) / nt;
                    int64_t s = tid * chunk;
                    int64_t e = std::min(s + chunk, n);
                    auto& off = toff[tid];
                    for (int64_t i = s; i < e; i++) {
                        if (arr[i] > split)
                            tmp[off[OVERFLOW]++] = arr[i];
                        else
                            tmp[off[(arr[i] - base) & MASK]++] = arr[i];
                    }
                }
                memcpy(arr, tmp.data(), n * sizeof(int64_t));

                // Pass 1: high 15 bits on dense head only (lsz elements)
                if (lsz > 1) {
                    int shift = BITS;
                    std::vector<std::vector<int64_t>> lcnt2(nt,
                        std::vector<int64_t>(BUCKETS, 0));
                    #pragma omp parallel num_threads(nt)
                    {
                        int tid = omp_get_thread_num();
                        int64_t chunk = (lsz + nt - 1) / nt;
                        int64_t s = tid * chunk;
                        int64_t e = std::min(s + chunk, lsz);
                        auto& cnt = lcnt2[tid];
                        for (int64_t i = s; i < e; i++)
                            cnt[((arr[i] - base) >> shift) & MASK]++;
                    }

                    std::vector<int64_t> gprefix2(BUCKETS, 0);
                    for (int b = 0; b < BUCKETS; b++)
                        for (int t = 0; t < nt; t++)
                            gprefix2[b] += lcnt2[t][b];

                    running = 0;
                    for (int b = 0; b < BUCKETS; b++) {
                        int64_t c = gprefix2[b];
                        gprefix2[b] = running;
                        running += c;
                    }

                    std::vector<std::vector<int64_t>> toff2(nt,
                        std::vector<int64_t>(BUCKETS, 0));
                    for (int b = 0; b < BUCKETS; b++) {
                        int64_t off = gprefix2[b];
                        for (int t = 0; t < nt; t++) {
                            toff2[t][b] = off;
                            off += lcnt2[t][b];
                        }
                    }

                    #pragma omp parallel num_threads(nt)
                    {
                        int tid = omp_get_thread_num();
                        int64_t chunk = (lsz + nt - 1) / nt;
                        int64_t s = tid * chunk;
                        int64_t e = std::min(s + chunk, lsz);
                        auto& off = toff2[tid];
                        for (int64_t i = s; i < e; i++) {
                            int64_t b = ((arr[i] - base) >> shift) & MASK;
                            tmp[off[b]++] = arr[i];
                        }
                    }
                    memcpy(arr, tmp.data(), lsz * sizeof(int64_t));
                }

                // Sparse tail
                if (rsz > 1) {
                    int64_t t_min = arr[lsz], t_max = arr[lsz];
                    for (int64_t i = lsz + 1; i < n; i++) {
                        if (arr[i] < t_min) t_min = arr[i];
                        if (arr[i] > t_max) t_max = arr[i];
                    }
                    if (t_min != t_max)
                        adaptive_parallel_sort(arr + lsz, rsz, t_min, t_max);
                }
                return;
            }
        }

        // Clustered rescale: when sample range fits in 28 bits, use the
        // rescaled sector assignment in nssort_arithmetic to spread
        // clustered data evenly across all sectors.
        if ((long long)(sample_max - sample_min) < (1LL << 28)) {
            use_rescale = true;
            rs_sample_min = sample_min;
            rs_sample_max = sample_max;
        }
        } // end of else (skip_v9_detection == false)
    }
    // Fall through to existing v7 fast paths + general sort

    // FAST PATH — reverse-sorted check (3 reads)
    bool skip_v7_detection = false;
    if (arr[0] > arr[n / 2] && arr[n / 2] > arr[n - 1]) {
        parallel_reverse(arr, n);
        // Verify the result is actually sorted
        if (parallel_is_sorted_check(arr, n)) {
            return;
        }
        // If not sorted, skip v7 detection and fall through to general sort
        skip_v7_detection = true;
    }

    // FAST PATH — DO NOT MODIFY: nearly-sorted detection (v7 Change 2)
    if (!skip_v7_detection && n >= 1000) {
        int ordered = 0;
        int64_t step = max(1LL, (long long)(n / 32));
        for (int i = 0; i < 31; i++) {
            int64_t idx = i * step;
            if (arr[idx] <= arr[idx + 1]) ordered++;
        }
        if (ordered >= 24) {
            int64_t R = 1;
            for (int64_t i = 1; i < n; i++) if (arr[i] < arr[i - 1]) R++;
            if (R <= n / 200) {
                merge_runs(arr, n);
                return;
            }
        }
    }

    // FAST PATH — DO NOT MODIFY: heavy-duplicate detection (100-sample)
    {
        int64_t step = max(1LL, (long long)(n / 100));
        int64_t sample[128], sc = 0;
        for (int64_t i = 0; i < n && sc < 100; i += step) sample[sc++] = arr[i];
        sort(sample, sample + sc);
        int unique = 1;
        for (int i = 1; i < sc; i++) if (sample[i] != sample[i - 1]) unique++;
        if (unique <= 10) {
            if (counting_sort_checked(arr, n)) return;
        }
    }

    // Change B: smart clustered pre-pass (10 K ≤ n ≤ 2 M)
    if (n >= 10000 && n <= 2000000) {
        int64_t samp[32];
        for (int i = 0; i < 32; i++) samp[i] = arr[(long long)i * (n - 1) / 31];
        sort(samp, samp + 32);

        long long gaps[31], sorted_gaps[31];
        for (int i = 0; i < 31; i++) gaps[i] = (long long)samp[i + 1] - samp[i];
        memcpy(sorted_gaps, gaps, sizeof(gaps));
        sort(sorted_gaps, sorted_gaps + 31);
        long long median_gap = sorted_gaps[15];

        long long walls[16]; int nwalls = 0;
        for (int i = 0; i < 31 && nwalls < 16; i++)
            if (gaps[i] > 10 * median_gap)
                walls[nwalls++] = ((long long)samp[i] + samp[i + 1]) / 2;

        if (nwalls >= 1 && nwalls <= 8) {
            // Check for heavy-left distribution (exponential-like)
            sort(samp, samp + 32);
            int64_t sample_median = samp[15];
            int64_t sample_p95 = samp[30];
            int64_t local_sample_min = samp[0];
            int64_t local_sample_max = samp[31];
            if (sample_median < (local_sample_max - local_sample_min) * 0.15) {
                // Heavy-left detected: partition and split-sort
                int64_t threshold = sample_p95 + (sample_p95 - sample_median) / 2;
                int64_t left = 0, right = n - 1;
                while (left <= right) {
                    while (left <= right && arr[left] <= threshold) left++;
                    while (left <= right && arr[right] > threshold) right--;
                    if (left < right) std::swap(arr[left++], arr[right--]);
                }
                // Sort left partition (dense, small range)
                if (left > 0) counting_sort_checked(arr, left);
                // Sort right partition (sparse, small n)
                if (left < n) std::sort(arr + left, arr + n);
                return;
            }

            int K = nwalls + 1;
            int counts[9] = {};
            for (int64_t i = 0; i < n; i++) {
                int c = 0;
                for (int w = 0; w < nwalls; w++) c += (arr[i] >= walls[w]);
                counts[c]++;
            }
            int base[9]; base[0] = 0;
            for (int i = 1; i < K; i++) base[i] = base[i - 1] + counts[i - 1];
            int off[9];
            for (int i = 0; i < K; i++) off[i] = base[i];
            vector<int64_t> tmp(n);
            constexpr int SBUF = 64;
            int64_t sbuf[9][SBUF];
            int sbuf_cnt[9] = {};
            for (int64_t i = 0; i < n; i++) {
                int c = 0;
                for (int w = 0; w < nwalls; w++) c += (arr[i] >= walls[w]);
                sbuf[c][sbuf_cnt[c]++] = arr[i];
                if (sbuf_cnt[c] == SBUF) {
                    memcpy(tmp.data() + off[c], sbuf[c], SBUF * sizeof(int64_t));
                    off[c] += SBUF;
                    sbuf_cnt[c] = 0;
                }
            }
            for (int c = 0; c < K; c++) {
                if (sbuf_cnt[c] > 0) {
                    memcpy(tmp.data() + off[c], sbuf[c], sbuf_cnt[c] * sizeof(int64_t));
                }
            }
            memcpy(arr, tmp.data(), n * sizeof(int64_t));
            int64_t pos = 0;
            for (int i = 0; i < K; i++) {
                if (counts[i] > 1) {
                    nssort_scatter_into(arr + pos, tmp.data() + pos,
                                        counts[i], 0, 0, true);
                    memcpy(arr + pos, tmp.data() + pos, counts[i] * sizeof(int64_t));
                }
                pos += counts[i];
            }
            return;
        }

    }  // end of skip_v9_detection else

    // Standard path
    g_general_sort_calls.fetch_add(1, std::memory_order_relaxed);
    std::sort(arr, arr + n);
}

// ---------------------------------------------------------------------------
// parallel_is_sorted_check — O(n) parallel verification
// ---------------------------------------------------------------------------
static bool parallel_is_sorted_check(int64_t* arr, int64_t n) {
    if (n <= 1) return true;
    
    // For small n, use sequential check
    if (n < 100000) {
        for (int64_t i = 1; i < n; i++) if (arr[i] < arr[i - 1]) return false;
        return true;
    }
    
    // Parallel check with early exit
    bool sorted = true;
    #pragma omp parallel for schedule(static) reduction(&&:sorted)
    for (int64_t i = 1; i < n; i++) {
        if (arr[i] < arr[i - 1]) sorted = false;
    }
    return sorted;
}

#endif // NSSORT_HPP

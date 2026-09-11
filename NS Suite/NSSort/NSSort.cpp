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

#include <algorithm>
#include <cstdlib>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <iostream>
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
#include <fstream>
#include <sstream>
#include <ctime>

// Global sync counters for instrumentation
std::atomic<long long> g_nssort_barrier_count(0);
std::atomic<long long> g_nssort_critical_count(0);
std::atomic<long long> g_nssort_atomic_count(0);

// IPS4o headers are only included in bench_v10.cpp to avoid linker conflicts
// when NSSort_20260704_active.cpp is linked with benchmark code

using namespace std;
using namespace std::chrono;

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
static bool parallel_is_sorted_check(int* arr, int n);

// ---------------------------------------------------------------------------
// Insertion sort — leaf-level (n ≲ 11)
// ---------------------------------------------------------------------------
static void insertion_sort(int* arr, int n) {
    for (int i = 1; i < n; i++) {
        int key = arr[i], j = i - 1;
        while (j >= 0 && arr[j] > key) { arr[j + 1] = arr[j]; j--; }
        arr[j + 1] = key;
    }
}

// Returns false = input violates this fast path's assumptions; arr is
// UNTOUCHED and the caller must fall through to the next strategy.
static bool counting_sort_checked(int* arr, int n, int min_val, long long range) {
    fprintf(stderr, "[INSTRUMENTATION] counting_sort_checked called: n=%d, min=%d, range=%lld\n", n, min_val, range);
    if (n <= 0) return true;

    if (range <= 4096) {
        int cnt[4097] = {};
        std::atomic<bool> ok{true};
        #pragma omp parallel
        {
            int local_cnt[4097] = {};
            #pragma omp for schedule(static)
            for (int i = 0; i < n; i++) {
                long long idx = (long long)arr[i] - min_val;
                if (idx < 0 || idx > range) {
                    ok.store(false, std::memory_order_relaxed);
                    idx = 0;              // keep loop structure valid; result discarded
                }
                local_cnt[idx]++;
            }
            if (ok.load(std::memory_order_relaxed)) {
                #pragma omp critical
                for (long long v = 0; v <= range; v++) cnt[v] += local_cnt[v];
            }
        }
        if (!ok.load()) return false;     // arr untouched — safe bail
        int pos = 0;
        for (long long v = 0; v <= range; v++)
            for (int k = 0; k < cnt[v]; k++)
                arr[pos++] = (int)(v + min_val);
        return pos == n;                  // belt-and-suspenders
    }

    // Large-range path: bounded unique-value table with bail-out.
    constexpr int MAXU = 24;
    int uvals[MAXU], nu = 0;
    for (int i = 0; i < n; i++) {
        int v = arr[i], j = 0;
        while (j < nu && uvals[j] != v) j++;
        if (j == nu) {
            if (nu == MAXU) return false;         // too many uniques — bail
            uvals[nu++] = v;
        }
    }
    for (int i = 1; i < nu; i++) {
        int kv = uvals[i], j = i - 1;
        while (j >= 0 && uvals[j] > kv) { uvals[j+1] = uvals[j]; j--; }
        uvals[j+1] = kv;
    }

    long long ucnts[MAXU] = {};
    std::atomic<bool> ok{true};
    #pragma omp parallel
    {
        long long local_cnt[MAXU] = {};
        #pragma omp for schedule(static)
        for (int i = 0; i < n; i++) {
            int v = arr[i], j = 0;
            while (j < nu && uvals[j] != v) j++;
            if (j == nu) { ok.store(false, std::memory_order_relaxed); j = 0; }
            local_cnt[j]++;
        }
        if (ok.load(std::memory_order_relaxed)) {
            #pragma omp critical
            for (int j = 0; j < nu; j++) ucnts[j] += local_cnt[j];
        }
    }
    if (!ok.load()) return false;

    long long pos = 0;
    for (int i = 0; i < nu; i++)
        for (long long k = 0; k < ucnts[i]; k++)
            arr[pos++] = uvals[i];
    return pos == n;
}

// Convenience wrapper: computes TRUE min/max itself (replaces overload 1).
static bool counting_sort_checked(int* arr, int n) {
    if (n <= 0) return true;
    int min_val = arr[0], max_val = arr[0];
    #pragma omp parallel for schedule(static) reduction(min:min_val) reduction(max:max_val)
    for (int i = 1; i < n; i++) {
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
static void regime_a_insertion_repair(int* arr, int n) {
    fprintf(stderr, "[INSTRUMENTATION] Regime A insertion repair called (n=%d)\n", n);
    // Full-array insertion repair — no window partitioning.
    // Window partitioning blocked cross-boundary element movement.
    // Called only when dirty_count/num_windows > 0.30 threshold,
    // meaning data is dense-nearly-sorted: O(n*k) where k is small.
    for (int i = 0; i < n - 1; i++) {
        if (arr[i] > arr[i + 1]) {
            int val = arr[i + 1];
            arr[i + 1] = arr[i];
            int j = i;
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
static void regime_a_window_repair(int* arr, int n, int disp_est) {
    int repair_window = (n > 500000000) ? 16384 : WINDOW;
    int num_windows = (n + repair_window - 1) / repair_window;

    vector<bool> local_dirty(num_windows, false);
    int dirty_count = 0;
    for (int i = 0; i < n - 1; i++) {
        if (arr[i] > arr[i + 1]) {
            int win  = i / repair_window;
            int win2 = (i + 1) / repair_window;
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

    for (int w = 0; w < num_windows; w++) {
        if (!local_dirty[w]) continue;
        int overlap = max(1, disp_est);
        int start = max(0, w * repair_window - overlap);
        int end   = min(n, (w + 1) * repair_window + overlap);
        int len   = end - start;
        vector<int> buf(len);
        copy(arr + start, arr + end, buf.begin());
        sort(buf.begin(), buf.end());
        copy(buf.begin(), buf.end(), arr + start);
    }
}

// Forward declaration
static void parallel_merge_regime_b(int* arr, int w, int n);

// ---------------------------------------------------------------------------
// v9 Regime B — run-based extract-merge
// Marking: arr[i] > arr[i+1] only (not both neighbours).
// Chains consecutive violations into runs (one run per reversed block).
// Coalesce overlapping runs, extract, sort, then parallel merge.
// ---------------------------------------------------------------------------
static void regime_b_extract_merge(int* arr, int n) {
    vector<pair<int,int>> runs;
    int i = 0;
    while (i < n - 1) {
        if (arr[i] > arr[i + 1]) {
            int run_start = max(0, i - 1);
            while (i < n - 1 && arr[i] > arr[i + 1]) i++;
            runs.push_back({run_start, min(n - 1, i + 1)});
        } else {
            i++;
        }
    }
    if (runs.empty()) return;

    sort(runs.begin(), runs.end());
    vector<pair<int,int>> merged;
    for (auto& r : runs) {
        if (merged.empty() || r.first > merged.back().second) {
            merged.push_back(r);
        } else {
            merged.back().second = max(merged.back().second, r.second);
        }
    }

    int total = 0;
    for (auto& r : merged) total += r.second - r.first + 1;
    vector<int> extracted(total);
    int e = 0;
    for (auto& r : merged)
        for (int j = r.first; j <= r.second; j++)
            extracted[e++] = arr[j];
    sort(extracted.begin(), extracted.end());

    int w = 0, pos = 0;
    for (auto& r : merged) {
        int len = r.first - pos;
        if (len > 0) {
            memmove(arr + w, arr + pos, len * sizeof(int));
            w += len;
        }
        pos = r.second + 1;
    }
    if (pos < n) {
        memmove(arr + w, arr + pos, (n - pos) * sizeof(int));
        w += n - pos;
    }
    memcpy(arr + w, extracted.data(), total * sizeof(int));
    sort(arr, arr + w);
    parallel_merge_regime_b(arr, w, n);
}

// ---------------------------------------------------------------------------
// Change 2 helper: merge sorted runs bottom-up  (FAST PATH — DO NOT MODIFY)
// Scans arr once to collect run-start positions, then merges pairwise until
// one run remains.  O(n) scan + O(n log R) merges; fast when R is small.
// ---------------------------------------------------------------------------
static void merge_runs(int* arr, int n) {
    vector<int> starts;
    starts.reserve(64);
    starts.push_back(0);
    for (int i = 1; i < n; i++)
        if (arr[i] < arr[i - 1]) starts.push_back(i);
    starts.push_back(n);

    int nruns = (int)starts.size() - 1;
    if (nruns <= 1) return;

    while ((int)starts.size() > 2) {
        vector<int> next;
        next.reserve(starts.size() / 2 + 2);
        next.push_back(starts[0]);
        for (int i = 0; i + 2 < (int)starts.size(); i += 2) {
            int lo  = starts[i];
            int mid = starts[i + 1];
            int hi  = starts[i + 2];
            inplace_merge(arr + lo, arr + mid, arr + hi);
            next.push_back(hi);
        }
        if ((int)starts.size() % 2 == 0) next.push_back(starts.back());
        starts = move(next);
    }
}

// ---------------------------------------------------------------------------
// Batcher's sorting network — 16 elements, scalar fallback
// ---------------------------------------------------------------------------
static void sort16(int* a) {
#define CS(x,y) { int lo=a[x]<a[y]?a[x]:a[y], hi=a[x]>a[y]?a[x]:a[y]; a[x]=lo; a[y]=hi; }
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

#ifdef __AVX2__
// ---------------------------------------------------------------------------
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
#else
static const bool g_avx2_ok = false;
#endif

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

static inline void sort16_net(int* buf) {
#ifdef __AVX2__
    if (g_avx2_ok) sort16_avx2(buf); else sort16(buf);
#else
    sort16(buf);
#endif
}

// ---------------------------------------------------------------------------
// SIMD-accelerated small sort for sizes 1–32 (leaf level)
// ---------------------------------------------------------------------------
static void simd_sort_small(int* arr, int n) {
    if (n < 12 || n > 32) { insertion_sort(arr, n); return; }
    int buf[32];
    int chunk1 = n < 16 ? n : 16, chunk2 = n - chunk1;
    memcpy(buf, arr, chunk1 * sizeof(int));
    for (int i = chunk1; i < 16; i++) buf[i] = INT_MAX;
    sort16_net(buf);
    if (chunk2 > 0) {
        memcpy(buf + 16, arr + chunk1, chunk2 * sizeof(int));
        for (int i = 16 + chunk2; i < 32; i++) buf[i] = INT_MAX;
        sort16_net(buf + 16);
        inplace_merge(buf, buf + 16, buf + 16 + chunk2);
    }
    memcpy(arr, buf, n * sizeof(int));
}

// Forward declaration
void nssort_arithmetic(int* arr, int n, int depth, int forced_ns,
                       bool skip_two_level = false,
                       bool rescale = false, int rescale_min = 0, int rescale_max = 0);

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
static int co_rank(int k, const int* A, int m, const int* B, int n) {
    int i_lo = std::max(0, k - n);
    int i_hi = std::min(k, m);
    while (i_lo < i_hi) {
        int i = (i_lo + i_hi + 1) / 2;
        int j = k - i;
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
static void parallel_merge_regime_b(int* arr, int w, int n) {
    const int* A = arr;
    int m = w;
    const int* B = arr + w;
    int bn = n - w;

    int T = get_pool().NTHREADS;
    int* out = new int[n];  // uninitialized; fully overwritten by merge tasks
    std::vector<int> splits_i(T + 1), splits_j(T + 1);
    splits_i[0] = 0; splits_j[0] = 0;
    splits_i[T] = m; splits_j[T] = bn;

    for (int t = 1; t < T; t++) {
        long long k = (long long)t * n / T;
        int i = co_rank((int)k, A, m, B, bn);
        splits_i[t] = i;
        splits_j[t] = (int)k - i;
    }

    {
        std::vector<std::function<void()>> tasks;
        tasks.reserve(T);
        for (int t = 0; t < T; t++) {
            int i0 = splits_i[t], i1 = splits_i[t + 1];
            int j0 = splits_j[t], j1 = splits_j[t + 1];
            int out_start = i0 + j0;
            tasks.push_back([=]() {
                int ii = i0, jj = j0, oo = out_start;
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
        int chunk = (n + T - 1) / T;
        for (int t = 0; t < T; t++) {
            int lo = t * chunk;
            int hi = std::min(lo + chunk, n);
            tasks.push_back([=]() {
                memcpy(arr + lo, out + lo, (hi - lo) * sizeof(int));
            });
        }
        get_pool().submit_and_wait(std::move(tasks));
    }

    delete[] out;
}

// ---------------------------------------------------------------------------
// Parallel reverse for large descending inputs; small n use std::reverse.
// ---------------------------------------------------------------------------
static void parallel_reverse(int* arr, int n) {
    if (n >= 5000000) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n / 2; i++) swap(arr[i], arr[n - 1 - i]);
    } else {
        reverse(arr, arr + n);
    }
}

// ---------------------------------------------------------------------------
// nssort_scatter_into — one-level out-of-place scatter helper
// ---------------------------------------------------------------------------
static void nssort_scatter_into(int* src, int* dst, int n, int depth,
                                int forced_ns, bool skip_two_level,
                                bool rescale = false, int rescale_min = 0, int rescale_max = 0) {
    long long g_min = src[0], g_max = src[0];
    for (int i = 1; i < n; i++) {
        if (src[i] < g_min) g_min = src[i];
        if (src[i] > g_max) g_max = src[i];
    }
    long long g_range = g_max - g_min + 1;
    if (g_range <= 0) {
        memcpy(dst, src, n * sizeof(int));
        return;
    }

    int NS;
    if (forced_ns > 0) {
        NS = forced_ns;
    } else if (n < 1024) {
        int sn = 1;
        while (sn < n / 8) sn <<= 1;
        if (sn < 4) sn = 4;
        NS = sn;
    } else if (depth > 0) {
        NS = 1;
        while (NS < n / 4000 && NS < 256) NS <<= 1;
        NS = max(NS, 4);
        NS = min(NS, 256);
    } else {
        NS = 1;
        while (NS < n / 4000 && NS < 1024) NS <<= 1;
        NS = max(NS, 256);
        NS = min(NS, 1024);
    }



    auto get_sector = [g_min, g_range, NS](int val) -> int {
        int s = (int)((long long)(val - g_min) * NS / g_range);
        if (s >= NS) s = NS - 1;
        if (s < 0)   s = 0;
        return s;
    };

    int counts[1024];
    memset(counts, 0, NS * sizeof(int));
    for (int i = 0; i < n; i++) counts[get_sector(src[i])]++;



    int offsets[1024];
    offsets[0] = 0;
    for (int s = 1; s < NS; s++) offsets[s] = offsets[s - 1] + counts[s - 1];

    int cursors[1024];
    for (int s = 0; s < NS; s++) cursors[s] = offsets[s];
    for (int i = 0; i < n; i++) dst[cursors[get_sector(src[i])]++] = src[i];

    const long long expected         = (long long)n / NS;
    const long long imbalance_thresh = max(4 * expected, 500000LL);
    const int       sub_ns           = min(NS * 4, 1024);

    auto handle_sector = [&](int s) {
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
        for (int s = 0; s < NS; s++) handle_sector(s);
    } else {
        for (int s = 0; s < NS; s++) handle_sector(s);
    }
}

// ---------------------------------------------------------------------------
// FAST PATH — DO NOT MODIFY
// nssort_arithmetic — core recursive arithmetic-sector sort
// ---------------------------------------------------------------------------
void nssort_arithmetic(int* arr, int n, int depth, int forced_ns,
                       bool skip_two_level /* = false */,
                       bool rescale /* = false */, int rescale_min /* = 0 */, int rescale_max /* = 0 */) {
    if (n <= 1) return;
    if (depth > 20) { sort(arr, arr + n); return; }
    if (forced_ns > 1024) forced_ns = 1024;

    long long g_min = arr[0], g_max = arr[0];
    for (int i = 1; i < n; i++) {
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

    if (!skip_two_level && depth > 0 && n >= 10000 && n <= 4000000) {
        constexpr int CNS    = 16;
        constexpr int BSIZE  = 32;

        auto coarse = [g_min, g_range](int val) -> int {
            int s = (int)((long long)(val - g_min) * CNS / g_range);
            if (s >= CNS) s = CNS - 1;
            if (s < 0)    s = 0;
            return s;
        };

        int counts[CNS] = {};
        for (int i = 0; i < n; i++) counts[coarse(arr[i])]++;

        int offsets[CNS];
        offsets[0] = 0;
        for (int s = 1; s < CNS; s++) offsets[s] = offsets[s - 1] + counts[s - 1];

        int* output = new int[n];
        int  buf[CNS][BSIZE];
        int  buf_cnt[CNS] = {};
        int  cur[CNS];
        for (int s = 0; s < CNS; s++) cur[s] = offsets[s];

        for (int i = 0; i < n; i++) {
            int s = coarse(arr[i]);
            buf[s][buf_cnt[s]++] = arr[i];
            if (buf_cnt[s] == BSIZE) {
                memcpy(output + cur[s], buf[s], BSIZE * sizeof(int));
                cur[s]     += BSIZE;
                buf_cnt[s]  = 0;
            }
        }
        for (int s = 0; s < CNS; s++)
            if (buf_cnt[s] > 0)
                memcpy(output + cur[s], buf[s], buf_cnt[s] * sizeof(int));
        memcpy(arr, output, n * sizeof(int));
        delete[] output;

        for (int s = 0; s < CNS; s++)
            if (counts[s] > 1)
                nssort_arithmetic(arr + offsets[s], counts[s], depth + 1, 0, false);
        return;
    }

    int NS;
    if (forced_ns > 0) {
        NS = forced_ns;
    } else if (n < 1024) {
        int sn = 1;
        while (sn < n / 8) sn <<= 1;
        if (sn < 4) sn = 4;
        NS = sn;
    } else if (depth > 0) {
        NS = 1;
        while (NS < n / 4000 && NS < 256) NS <<= 1;
        NS = max(NS, 4);
        NS = min(NS, 256);
    } else {
        NS = 1;
        while (NS < n / 4000 && NS < 1024) NS <<= 1;
        NS = max(NS, 256);
        NS = min(NS, 1024);
    }



    float rescale_scale = 0.0f;
    if (rescale) {
        rescale_scale = (float)(NS - 1) / (float)(rescale_max - rescale_min);
    }

    auto get_sector = [g_min, g_range, NS, rescale, rescale_scale](int val) -> int {
        if (rescale) {
            int s = (int)((float)(val - g_min) * rescale_scale);
            if (s >= NS) s = NS - 1;
            if (s < 0)   s = 0;
            return s;
        }
        int s = (int)((long long)(val - g_min) * NS / g_range);
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
            int chunk = n / g_team_size;
            std::vector<std::function<void()>> tasks;
            tasks.reserve(g_team_size);
            for (int tid = 0; tid < g_team_size; tid++) {
                int start = tid * chunk;
                int end = (tid == g_team_size - 1) ? n : start + chunk;
                tasks.push_back([&, tid, start, end]() {
                    for (int i = start; i < end; i++)
                        local_hist[tid][get_sector(arr[i])]++;
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

        int* output = new int[n];
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
                    int stage_buf[1024][STAGE];
                    int stage_cnt[1024];
                    memset(stage_cnt, 0, NS * sizeof(int));
                    for (int i = start; i < end; i++) {
                        int s = get_sector(arr[i]);
                        stage_buf[s][stage_cnt[s]++] = arr[i];
                        if (stage_cnt[s] == STAGE) {
                            memcpy(output + cursors[s], stage_buf[s], STAGE * sizeof(int));
                            cursors[s] += STAGE;
                            stage_cnt[s] = 0;
                        }
                    }
                    for (int s = 0; s < NS; s++) {
                        if (stage_cnt[s] > 0) {
                            memcpy(output + cursors[s], stage_buf[s], stage_cnt[s] * sizeof(int));
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
        const int       sub_ns           = min(NS * 4, 1024);

        auto handle_sector = [&](int s) {
            if (counts[s] < 2) {
                memcpy(arr + offsets[s], output + offsets[s], counts[s] * sizeof(int));
                return;
            }
            if (depth == 0 && sub_ns > NS && counts[s] > imbalance_thresh) {
                nssort_arithmetic(output + offsets[s], counts[s], depth + 1, sub_ns, false);
                memcpy(arr + offsets[s], output + offsets[s], counts[s] * sizeof(int));
                return;
            }
            if (counts[s] > n / 2) { 
                sort(output + offsets[s], output + offsets[s] + counts[s]); 
                memcpy(arr + offsets[s], output + offsets[s], counts[s] * sizeof(int));
                return; 
            }
            if (counts[s] <= 32) {
                simd_sort_small(output + offsets[s], counts[s]);
                memcpy(arr + offsets[s], output + offsets[s], counts[s] * sizeof(int));
            } else {
                nssort_arithmetic(output + offsets[s], counts[s], depth + 1, 0, false);
                memcpy(arr + offsets[s], output + offsets[s], counts[s] * sizeof(int));
            }
        };

        g_recursion_parallel_calls.fetch_add(1, std::memory_order_relaxed);
        #pragma omp parallel for schedule(dynamic, 4) num_threads(g_team_size)
        for (int s = 0; s < NS; s++) handle_sector(s);
        delete[] output;
    } else {
        memset(counts, 0, NS * sizeof(int));
        for (int i = 0; i < n; i++) counts[get_sector(arr[i])]++;



        offsets[0] = 0;
        for (int s = 1; s < NS; s++) offsets[s] = offsets[s - 1] + counts[s - 1];

        int  stack_buf[1024];
        int* output = (n <= 1024) ? stack_buf : new int[n];
        int  cursors[1024];
        for (int s = 0; s < NS; s++) cursors[s] = offsets[s];
        if (depth == 0) {
            constexpr int STAGE = 16;
            int stage_buf[1024][STAGE];
            int stage_cnt[1024];
            memset(stage_cnt, 0, NS * sizeof(int));
            for (int i = 0; i < n; i++) {
                int s = get_sector(arr[i]);
                stage_buf[s][stage_cnt[s]++] = arr[i];
                if (stage_cnt[s] == STAGE) {
                    memcpy(output + cursors[s], stage_buf[s], STAGE * sizeof(int));
                    cursors[s] += STAGE;
                    stage_cnt[s] = 0;
                }
            }
            for (int s = 0; s < NS; s++) {
                if (stage_cnt[s] > 0) {
                    memcpy(output + cursors[s], stage_buf[s], stage_cnt[s] * sizeof(int));
                    cursors[s] += stage_cnt[s];
                }
            }
        } else {
            for (int i = 0; i < n; i++) output[cursors[get_sector(arr[i])]++] = arr[i];
        }
        // P1-C: eliminate sequential copy-back - fold into parallel recursion loop
        const long long expected         = (long long)n / NS;
        const long long imbalance_thresh = max(4 * expected, 500000LL);
        const int       sub_ns           = min(NS * 4, 1024);

        auto handle_sector = [&](int s) {
            if (counts[s] < 2) {
                memcpy(arr + offsets[s], output + offsets[s], counts[s] * sizeof(int));
                return;
            }
            if (depth == 0 && sub_ns > NS && counts[s] > imbalance_thresh) {
                nssort_arithmetic(output + offsets[s], counts[s], depth + 1, sub_ns, false);
                memcpy(arr + offsets[s], output + offsets[s], counts[s] * sizeof(int));
                return;
            }
            if (counts[s] > n / 2) { 
                sort(output + offsets[s], output + offsets[s] + counts[s]); 
                memcpy(arr + offsets[s], output + offsets[s], counts[s] * sizeof(int));
                return; 
            }
            if (counts[s] <= 32) {
                simd_sort_small(output + offsets[s], counts[s]);
                memcpy(arr + offsets[s], output + offsets[s], counts[s] * sizeof(int));
            } else {
                nssort_arithmetic(output + offsets[s], counts[s], depth + 1, 0, false);
                memcpy(arr + offsets[s], output + offsets[s], counts[s] * sizeof(int));
            }
        };

        if (n > 5000000) {
            g_recursion_parallel_calls.fetch_add(1, std::memory_order_relaxed);
            #pragma omp parallel for schedule(dynamic, 4) num_threads(g_team_size)
            for (int s = 0; s < NS; s++) handle_sector(s);
        } else {
            for (int s = 0; s < NS; s++) handle_sector(s);
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
static bool try_cluster_sort_2pass(int* arr, int n,
                                   const std::vector<std::pair<int,int>>& clusters) {
    int N = (int)clusters.size();
    if (N < 2 || N > 8) return false;

    int cluster_maxes[8];
    for (int c = 0; c < N; c++) cluster_maxes[c] = clusters[c].second;

    ClusterSortPool& pool = get_pool();
    const int T = pool.NTHREADS;
    const int chunk = (n + T - 1) / T;

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
            const int lo = t * chunk;
            const int hi = std::min(lo + chunk, n);
            tasks.push_back([&, t, lo, hi]() {
                for (int i = lo; i < hi; i++) {
                    int v = arr[i];
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
    int* out_buf = new int[n];             // per-call, uninitialized; fully overwritten
    int* const out_ptr = out_buf;          // raw ptr: safe to read from any thread

    {
        std::vector<std::function<void()>> tasks;
        tasks.reserve(T);
        for (int t = 0; t < T; t++) {
            const int lo = t * chunk;
            const int hi = std::min(lo + chunk, n);
            tasks.push_back([&, t, lo, hi]() {
                int cur[8];
                for (int c = 0; c < N; c++) cur[c] = cursors[t][c];
                for (int i = lo; i < hi; i++) {
                    int v = arr[i];
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
                const int cmin       = clusters[c].first;
                const long long crange = (long long)clusters[c].second - cmin + 1;
                const int start      = offsets[c];
                const int sz         = sizes[c];
                if (sz == 0) return;
                if (crange <= 2000000) {
                    std::vector<int> counts((int)crange, 0);
                    for (int i = start; i < start + sz; i++) counts[out_ptr[i] - cmin]++;
                    int pos = start;
                    for (int v = 0; v < (int)crange; v++)
                        for (int k = 0; k < counts[v]; k++)
                            out_ptr[pos++] = v + cmin;
                } else {
                    std::sort(out_ptr + start, out_ptr + start + sz);
                }
            });
        }
        pool.submit_and_wait(std::move(tasks));
    }

    memcpy(arr, out_ptr, n * sizeof(int));
    delete[] out_buf;
    return true;
}

// ---------------------------------------------------------------------------
// Compressed counting sort — large-range sparse data fast path
// O(n) scan into unordered_map, sort distinct keys, REGENERATE output.
// Bails early if >100k distinct values (avoids memory blowup on random data).
// ---------------------------------------------------------------------------
static bool compressed_counting_sort(int* arr, int n) {
    unordered_map<int,int> counts;
    counts.reserve(2048);
    for (int i = 0; i < n; i++) {
        counts[arr[i]]++;
        if (counts.size() > 100000) return false;
    }

    vector<pair<int,int>> pairs(counts.begin(), counts.end());
    sort(pairs.begin(), pairs.end());

    int pos = 0;
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
void nssort(int* arr, int n) {
    fprintf(stderr, "[INSTRUMENTATION] nssort ENTRY: n=%d\n", n);
    if (n <= 1) return;

    // FAST PATH — DO NOT MODIFY: tiny arrays
    if (n < 32) { insertion_sort(arr, n); return; }
    
    bool use_rescale = false;
    int rs_sample_min = 0, rs_sample_max = 0;

    // ===== v9: 256-element sample detection =====
    bool skip_v9_detection = false;
    if (n >= SAMPLE_N) {
        int desc_violations = 0, asc_violations = 0;
        int sample_min = arr[0], sample_max = arr[0];
        int step = n / SAMPLE_N;

        for (int i = 0; i < SAMPLE_N - 1; i++) {
            int idx = i * step;
            if (arr[idx] > arr[idx + 1]) asc_violations++;
            if (arr[idx] < arr[idx + 1]) desc_violations++;
            if (arr[idx] < sample_min) sample_min = arr[idx];
            if (arr[idx] > sample_max) sample_max = arr[idx];
        }
        int last_idx = (SAMPLE_N - 1) * step;
        if (arr[last_idx] < sample_min) sample_min = arr[last_idx];
        if (arr[last_idx] > sample_max) sample_max = arr[last_idx];

        int sample_range = sample_max - sample_min;

        // Step 1: desc_violations == 0 → reverse
        fprintf(stderr, "[INSTRUMENTATION] Step 1: desc_violations=%d (n=%d)\n", desc_violations, n);
        if (desc_violations == 0) {
            fprintf(stderr, "[INSTRUMENTATION] v9 reverse path taken (desc_violations=0, n=%d)\n", n);
            // Spot check before reverse
            fprintf(stderr, "[INSTRUMENTATION] Before reverse: arr[0]=%d, arr[n/2]=%d, arr[n-1]=%d\n", arr[0], arr[n/2], arr[n-1]);
            parallel_reverse(arr, n);
            // Spot check after reverse
            fprintf(stderr, "[INSTRUMENTATION] After reverse: arr[0]=%d, arr[n/2]=%d, arr[n-1]=%d\n", arr[0], arr[n/2], arr[n-1]);
            // Verify the result is actually sorted
            bool is_sorted = parallel_is_sorted_check(arr, n);
            fprintf(stderr, "[INSTRUMENTATION] parallel_is_sorted_check result: %s\n", is_sorted ? "SORTED" : "NOT SORTED");
            if (is_sorted) {
                fprintf(stderr, "[INSTRUMENTATION] v9 reverse path verified sorted, returning\n");
                return;
            }
            // If not sorted, skip remaining v9 detection and fall through to general sort
            fprintf(stderr, "[INSTRUMENTATION] v9 reverse path verification FAILED, falling through\n");
            skip_v9_detection = true;
        }

        if (skip_v9_detection) {
            // Skip to general sort - array is now reversed, but general sort handles any order
        } else {

        // Step 2: distinct values in sample ≤ 20 → counting sort with sample range
        {
            int tmp[256];
            for (int i = 0; i < SAMPLE_N; i++) tmp[i] = arr[i * step];
            sort(tmp, tmp + SAMPLE_N);
            int distinct = 1;
            for (int i = 1; i < SAMPLE_N; i++) if (tmp[i] != tmp[i - 1]) distinct++;
            if (distinct <= 20) {
                if (counting_sort_checked(arr, n, sample_min, (long long)sample_max - sample_min))
                    return;
                // else: sample lied — fall through to Step 3
            }
        }

        // Step 3: asc_violations threshold → O(n) verification → Regime A/B
        fprintf(stderr, "[NS-GATE-ACTUAL] n=%d sample_asc_violations=%d threshold=%d\n", n, asc_violations, SAMPLE_N / 16);
        if (asc_violations == 0 || asc_violations <= SAMPLE_N / 16) {
            int total_violations = 0;
            for (int i = 0; i < n - 1; i++)
                if (arr[i] > arr[i + 1]) total_violations++;


            if (total_violations == 0) return;

            int disp_est = 0;
            bool has_long = false;
            for (int i = 0; i < n - 1; i++) {
                if (arr[i] > arr[i + 1]) {
                    int target = i + 1;
                    while (target < n && arr[target] < arr[i]) target++;
                    int dist = target - i;
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
            int sample_sorted[SAMPLE_N];
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
            if (n > 100000000) cluster_sample_n = min(2048, n / 500000);
            int cstep = n / cluster_sample_n;

            vector<int> s_sample(cluster_sample_n);
            for (int i = 0; i < cluster_sample_n; i++) s_sample[i] = arr[i * cstep];
            sort(s_sample.begin(), s_sample.end());

            long long s_range = (long long)s_sample[cluster_sample_n - 1] - s_sample[0];
            int gap_count = 0;
            vector<pair<int,int>> clusters;
            int cluster_start = s_sample[0];

            for (int i = 1; i < cluster_sample_n; i++) {
                long long gap = (long long)s_sample[i] - s_sample[i - 1];
                if (gap > s_range / 8) {
                    int cmin = cluster_start;
                    int cmax = s_sample[i - 1];
                    int cluster_span = cmax - cmin;
                    int margin = cluster_span / 10;
                    clusters.push_back({cmin - margin, cmax + margin});
                    cluster_start = s_sample[i];
                    gap_count++;
                }
            }
            int cmin = cluster_start;
            int cmax = s_sample[cluster_sample_n - 1];
            int cluster_span = cmax - cmin;
            int margin = cluster_span / 10;
            clusters.push_back({cmin - margin, cmax + margin});

            if (gap_count >= 2 && gap_count <= 7 && clusters.size() >= 3 && clusters.size() <= 8) {
                if (try_cluster_sort_2pass(arr, n, clusters)) return;
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
        int step = max(1, n / 32);
        for (int i = 0; i < 31; i++) {
            int idx = i * step;
            if (arr[idx] <= arr[idx + 1]) ordered++;
        }
        if (ordered >= 24) {
            int R = 1;
            for (int i = 1; i < n; i++) if (arr[i] < arr[i - 1]) R++;
            if (R <= n / 200) {
                merge_runs(arr, n);
                return;
            }
        }
    }

    // FAST PATH — DO NOT MODIFY: heavy-duplicate detection (100-sample)
    {
        int step = max(1, n / 100);
        int sample[128], sc = 0;
        for (int i = 0; i < n && sc < 100; i += step) sample[sc++] = arr[i];
        sort(sample, sample + sc);
        int unique = 1;
        for (int i = 1; i < sc; i++) if (sample[i] != sample[i - 1]) unique++;
        if (unique <= 10) {
            if (counting_sort_checked(arr, n)) return;
        }
    }

    // Change B: smart clustered pre-pass (10 K ≤ n ≤ 2 M)
    if (n >= 10000 && n <= 2000000) {
        int samp[32];
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
            int K = nwalls + 1;
            int counts[9] = {};
            for (int i = 0; i < n; i++) {
                int c = 0;
                for (int w = 0; w < nwalls; w++) c += (arr[i] >= walls[w]);
                counts[c]++;
            }
            int base[9]; base[0] = 0;
            for (int i = 1; i < K; i++) base[i] = base[i - 1] + counts[i - 1];
            int off[9];
            for (int i = 0; i < K; i++) off[i] = base[i];
            vector<int> tmp(n);
            constexpr int SBUF = 64;
            int sbuf[9][SBUF];
            int sbuf_cnt[9] = {};
            for (int i = 0; i < n; i++) {
                int c = 0;
                for (int w = 0; w < nwalls; w++) c += (arr[i] >= walls[w]);
                sbuf[c][sbuf_cnt[c]++] = arr[i];
                if (sbuf_cnt[c] == SBUF) {
                    memcpy(tmp.data() + off[c], sbuf[c], SBUF * sizeof(int));
                    off[c] += SBUF;
                    sbuf_cnt[c] = 0;
                }
            }
            for (int c = 0; c < K; c++) {
                if (sbuf_cnt[c] > 0) {
                    memcpy(tmp.data() + off[c], sbuf[c], sbuf_cnt[c] * sizeof(int));
                }
            }
            memcpy(arr, tmp.data(), n * sizeof(int));
            int pos = 0;
            for (int i = 0; i < K; i++) {
                if (counts[i] > 1) {
                    nssort_scatter_into(arr + pos, tmp.data() + pos,
                                        counts[i], 0, 0, true);
                    memcpy(arr + pos, tmp.data() + pos, counts[i] * sizeof(int));
                }
                pos += counts[i];
            }
            return;
        }
    }

    // Standard path
    g_general_sort_calls.fetch_add(1, std::memory_order_relaxed);
    nssort_arithmetic(arr, n, 0, 0, false, use_rescale, rs_sample_min, rs_sample_max);
}

// ---------------------------------------------------------------------------
// IPS4o reference wrapper (only used when not linked with bench_v10.cpp)
// ---------------------------------------------------------------------------
void ips4o_sort(int* arr, int n) {
    sort(arr, arr + n);
}

// ===========================================================================
// Data generators
// ===========================================================================

void generate_random_uniform(int* arr, int n, mt19937& rng) {
    uniform_int_distribution<int> d(0, 1000000000);
    for (int i = 0; i < n; i++) arr[i] = d(rng);
}

void generate_random_clustered(int* arr, int n, mt19937& rng) {
    int cs = n / 5;
    for (int i = 0; i < 5; i++) {
        int lo = i * 200000000, hi = lo + 100000;
        uniform_int_distribution<int> d(lo, hi);
        for (int j = 0; j < cs; j++) arr[i * cs + j] = d(rng);
    }
    uniform_int_distribution<int> fill(0, 1000000000);
    for (int i = 5 * cs; i < n; i++) arr[i] = fill(rng);
}

void generate_sparse(int* arr, int n, mt19937& rng) {
    for (int i = 0; i < n; i++) arr[i] = rng() % 1001;
}

void generate_nearly_sorted(int* arr, int n, mt19937& rng) {
    for (int i = 0; i < n; i++) arr[i] = i;
    int range_max = n - 1;
    if (range_max < 0) range_max = 0;
    uniform_int_distribution<int> sd(0, range_max);
    for (int i = 0; i < n / 10; i++) swap(arr[sd(rng)], arr[sd(rng)]);
}

void generate_reverse_sorted(int* arr, int n) {
    for (int i = 0; i < n; i++) arr[i] = n - i;
}

void generate_many_duplicates(int* arr, int n, mt19937& rng) {
    uniform_int_distribution<int> d(0, 9);
    for (int i = 0; i < n; i++) arr[i] = d(rng);
}

void generate_already_sorted(int* arr, int n) {
    for (int i = 0; i < n; i++) arr[i] = i;
}

void generate_ns_a(int* arr, int n, mt19937& rng) {
    for (int i = 0; i < n; i++) arr[i] = i;
    int pos_max = n - 1000;
    if (pos_max < 0) pos_max = 0;
    uniform_int_distribution<int> pos_dist(0, pos_max);
    uniform_int_distribution<int> offset_dist(0, 999);
    for (int i = 0; i < 1000; i++) {
        int p = pos_dist(rng);
        int q = p + offset_dist(rng);
        if (q >= n) q = n - 1;
        swap(arr[p], arr[q]);
    }
}

void generate_ns_b(int* arr, int n, mt19937& rng) {
    for (int i = 0; i < n; i++) arr[i] = i;
    int range_max = n - 1;
    if (range_max < 0) range_max = 0;
    uniform_int_distribution<int> dist(0, range_max);
    for (int i = 0; i < 1000; i++) {
        int p = dist(rng);
        int q = dist(rng);
        swap(arr[p], arr[q]);
    }
}

void generate_ns_c(int* arr, int n, mt19937& rng) {
    (void)rng;
    for (int i = 0; i < n; i++) arr[i] = i;
    int block_size = min(100000, n / 4);
    int mid = n / 2;
    int start = max(0, mid - block_size / 2);
    int end = min(n, start + block_size);
    reverse(arr + start, arr + end);
}

// T0-a: SentinelDuplicates - mostly one duplicate-heavy value (99.99% one value)
// with rare out-of-sample outliers including negative and INT_MAX-scale values.
// Purpose: reproduce CRIT-1 Path A (silent data loss from sample-derived min/max).
void generate_sentinel_duplicates(int* arr, int n, mt19937& rng) {
    uniform_int_distribution<int> d(0, 9);
    for (int i = 0; i < n; i++) arr[i] = d(rng);
    // Add rare outliers at random positions
    uniform_int_distribution<int> pos_dist(0, n - 1);
    arr[pos_dist(rng)] = -1000000000;
    arr[pos_dist(rng)] = 2000000000;
    arr[pos_dist(rng)] = 2100000000;
}

// T0-b: ElevenToTwentyDistinct - exactly 11 to 20 distinct values (vary across instances),
// spread over a range > 4096. Purpose: reproduce CRIT-1 Path B (uvals[10] overflow).
void generate_eleven_to_twenty_distinct(int* arr, int n, mt19937& rng) {
    uniform_int_distribution<int> count_dist(11, 20);
    int num_distinct = count_dist(rng);
    uniform_int_distribution<int> val_dist(-1000000, 1000000);
    vector<int> distinct_vals;
    for (int i = 0; i < num_distinct; i++) {
        distinct_vals.push_back(val_dist(rng));
    }
    uniform_int_distribution<int> idx_dist(0, num_distinct - 1);
    for (int i = 0; i < n; i++) arr[i] = distinct_vals[idx_dist(rng)];
}

// T0-c: MidRangeDistinct - large n (10M+), values uniform over a range where
// sample_range < n/10 but true distinct count is in the hundreds of thousands.
// Purpose: reproduce CRIT-2 (Step 4 gate admits ranges that overflow uvals[10]).
void generate_midrange_distinct(int* arr, int n, mt19937& rng) {
    int hi = max(500000, n / 20);
    uniform_int_distribution<int> d(0, hi);
    for (int i = 0; i < n; i++) arr[i] = d(rng);
}

// T0-d: EightWallClustered - clustered data engineered to produce exactly 8 walls
// in the sampling step (nwalls == 8). Purpose: reproduce CRIT-3 (sbuf[8]/sbuf_cnt[8]
// off-by-one, cluster index 0..8). Note: nwalls = clusters - 1, so need 9 clusters.
void generate_eight_wall_clustered(int* arr, int n, mt19937& rng) {
    int per = n / 9;
    for (int c = 0; c < 9; c++) {
        int lo = c * 200000000;
        uniform_int_distribution<int> d(lo, lo + 1000);
        int start = c * per;
        int end = (c == 8) ? n : start + per;
        for (int i = start; i < end; i++) arr[i] = d(rng);
    }
}

// ===========================================================================
// Correctness tests
// ===========================================================================

static bool is_sorted_check(int* arr, int n) {
    for (int i = 1; i < n; i++) if (arr[i] < arr[i - 1]) return false;
    return true;
}

// Fast parallel is_sorted check (reduction scans all elements, no early exit)
static bool parallel_is_sorted_check(int* arr, int n) {
    if (n <= 1) return true;
    
    // For small n, use sequential check
    if (n < 100000) {
        for (int i = 1; i < n; i++) if (arr[i] < arr[i - 1]) return false;
        return true;
    }
    
    // Parallel check with early exit
    bool sorted = true;
    #pragma omp parallel for schedule(static) reduction(&&:sorted)
    for (int i = 1; i < n; i++) {
        if (arr[i] < arr[i - 1]) sorted = false;
    }
    return sorted;
}

static void test_one(int n, const char* name, void (*gen)(int*, int, mt19937&)) {
    vector<int> arr(n), ref(n);
    mt19937 rng(42);
    gen(arr.data(), n, rng); ref = arr;
    nssort(arr.data(), n); sort(ref.begin(), ref.end());
    bool ok = is_sorted_check(arr.data(), n) && (arr == ref);
    printf("  %-25s n=%-8d  %s\n", name, n, ok ? "PASS" : "FAIL");
}

static void test_rev(int n) {
    vector<int> arr(n), ref(n);
    generate_reverse_sorted(arr.data(), n); ref = arr;
    nssort(arr.data(), n); sort(ref.begin(), ref.end());
    bool ok = is_sorted_check(arr.data(), n) && (arr == ref);
    printf("  %-25s n=%-8d  %s\n", "Reverse Sorted", n, ok ? "PASS" : "FAIL");
}

static void test_dup(int n) {
    vector<int> arr(n), ref(n);
    mt19937 rng(42);
    generate_many_duplicates(arr.data(), n, rng); ref = arr;
    nssort(arr.data(), n); sort(ref.begin(), ref.end());
    bool ok = is_sorted_check(arr.data(), n) && (arr == ref);
    printf("  %-25s n=%-8d  %s\n", "Many Duplicates", n, ok ? "PASS" : "FAIL");
}

static void test_ns(const char* name, void (*gen)(int*, int, mt19937&), int n) {
    vector<int> arr(n), ref(n);
    mt19937 rng(42);
    gen(arr.data(), n, rng); ref = arr;
    nssort(arr.data(), n); sort(ref.begin(), ref.end());
    bool ok = is_sorted_check(arr.data(), n) && (arr == ref);
    printf("  %-25s n=%-8d  %s\n", name, n, ok ? "PASS" : "FAIL");
}

void run_correctness_tests() {
    puts("=== CORRECTNESS TESTS ===");

    {
        int t[] = {950, 50, 970, 30, 980, 10, 960, 70, 940, 90};
        nssort(t, 10);
        bool ok = true;
        for (int i = 1; i < 10; i++) if (t[i] < t[i - 1]) ok = false;
        printf("  %-25s n=%-8d  %s\n", "Micro (<1024)", 10, ok ? "PASS" : "FAIL");
    }
    {
        int t[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        nssort(t, 10);
        bool ok = true;
        for (int i = 1; i < 10; i++) if (t[i] < t[i - 1]) ok = false;
        printf("  %-25s n=%-8d  %s\n", "Already sorted", 10, ok ? "PASS" : "FAIL");
    }

    int sizes[] = {100, 1000, 10000, 100000, 1000000};
    for (int n : sizes) {
        printf("--- n = %d ---\n", n);
        test_one(n, "Random Uniform",    generate_random_uniform);
        test_one(n, "Random Clustered",  generate_random_clustered);
        test_one(n, "Sparse",            generate_sparse);
        test_one(n, "Nearly Sorted",     generate_nearly_sorted);
        test_rev(n);
        test_dup(n);
        test_ns("NS-A", generate_ns_a, n);
        test_ns("NS-B", generate_ns_b, n);
        if (n >= 200000) test_ns("NS-C", generate_ns_c, n);
    }
    
    // T0-c: MidRangeDistinct - needs large n (10M+) to trigger sample_range < n/10 gate
    printf("--- n = 10000000 ---\n");
    test_one(10000000, "T0-c MidRange Distinct", generate_midrange_distinct);
    
    puts("");
}

// ===========================================================================
// Wall-clock benchmark (median of 20 runs, µs)
// ===========================================================================

static double bench_ns(int* base, int n, int iters) {
    vector<long long> t; t.reserve(iters);
    for (int it = 0; it < iters; it++) {
        vector<int> tmp(base, base + n);
        auto s = high_resolution_clock::now();
        nssort(tmp.data(), n);
        auto e = high_resolution_clock::now();
        t.push_back(duration_cast<microseconds>(e - s).count());
    }
    sort(t.begin(), t.end());
    return (double)t[iters / 2];
}

static double bench_ip(int* base, int n, int iters) {
    vector<long long> t; t.reserve(iters);
    for (int it = 0; it < iters; it++) {
        vector<int> tmp(base, base + n);
        auto s = high_resolution_clock::now();
        ips4o_sort(tmp.data(), n);
        auto e = high_resolution_clock::now();
        t.push_back(duration_cast<microseconds>(e - s).count());
    }
    sort(t.begin(), t.end());
    return (double)t[iters / 2];
}

static void bench_case(int n, const char* name, void (*gen)(int*, int, mt19937&)) {
    vector<int> arr(n); mt19937 rng(42); gen(arr.data(), n, rng);
    double ns_us = bench_ns(arr.data(), n, 20);
    double ip_us = bench_ip(arr.data(), n, 20);
    printf("  %-25s n=%-9d  NSSort=%7.0f µs  IPS4o=%7.0f µs  ratio=%.3f\n",
           name, n, ns_us, ip_us, ip_us / ns_us);
}

static void bench_rev(int n) {
    vector<int> arr(n); generate_reverse_sorted(arr.data(), n);
    double ns_us = bench_ns(arr.data(), n, 20);
    double ip_us = bench_ip(arr.data(), n, 20);
    printf("  %-25s n=%-9d  NSSort=%7.0f µs  IPS4o=%7.0f µs  ratio=%.3f\n",
           "Reverse Sorted", n, ns_us, ip_us, ip_us / ns_us);
}

static void bench_dup(int n) {
    vector<int> arr(n); mt19937 rng(42); generate_many_duplicates(arr.data(), n, rng);
    double ns_us = bench_ns(arr.data(), n, 20);
    double ip_us = bench_ip(arr.data(), n, 20);
    printf("  %-25s n=%-9d  NSSort=%7.0f µs  IPS4o=%7.0f µs  ratio=%.3f\n",
           "Many Duplicates", n, ns_us, ip_us, ip_us / ns_us);
}

static void bench_sorted(int n) {
    vector<int> arr(n); generate_already_sorted(arr.data(), n);
    double ns_us = bench_ns(arr.data(), n, 20);
    double ip_us = bench_ip(arr.data(), n, 20);
    printf("  %-25s n=%-9d  NSSort=%7.0f µs  IPS4o=%7.0f µs  ratio=%.3f\n",
           "Already Sorted", n, ns_us, ip_us, ip_us / ns_us);
}

static void bench_ns_abc(const char* name, void (*gen)(int*, int, mt19937&), int n) {
    vector<int> arr(n); mt19937 rng(42); gen(arr.data(), n, rng);
    double ns_us = bench_ns(arr.data(), n, 20);
    double ip_us = bench_ip(arr.data(), n, 20);
    printf("  %-25s n=%-9d  NSSort=%7.0f µs  IPS4o=%7.0f µs  ratio=%.3f\n",
           name, n, ns_us, ip_us, ip_us / ns_us);
}

void run_perf_tests() {
    puts("=== PERFORMANCE (median 20 runs, µs) ===");
    for (int n : {1000000, 10000000, 100000000}) {
        printf("--- n = %d ---\n", n);
        bench_case(n, "Random Uniform",    generate_random_uniform);
        bench_case(n, "Random Clustered",  generate_random_clustered);
        bench_case(n, "Sparse",            generate_sparse);
        bench_case(n, "Nearly Sorted",     generate_nearly_sorted);
        bench_rev(n);
        bench_dup(n);
        bench_sorted(n);
        bench_ns_abc("NS-A", generate_ns_a, n);
        bench_ns_abc("NS-B", generate_ns_b, n);
        if (n >= 200000) bench_ns_abc("NS-C", generate_ns_c, n);
    }
}

#ifndef BENCH_MODE
int main() {

    puts("NSSort v9 — v7 + 256-sample nearly-sorted path + Regime A/B");
    puts("===========================================================\n");
    run_correctness_tests();
    // run_perf_tests();
    
    // double sectors_per_call = g_depth0_calls > 0 ? (double)g_total_sectors / g_depth0_calls : 0.0;
    // printf("NS scaling instrumentation: depth>0 calls=%lld, total sectors=%lld, sectors/call=%.2f\n",
    //        g_depth0_calls, g_total_sectors, sectors_per_call);
    
    // printf("NS histogram (NS value -> count):\n");
    // for (int ns = 4; ns <= 256; ns *= 2) {
    //     if (g_ns_histogram[ns] > 0) {
    //         printf("  NS=%3d: %lld calls\n", ns, g_ns_histogram[ns]);
    //     }
    // }
    
    return 0;
}
#endif

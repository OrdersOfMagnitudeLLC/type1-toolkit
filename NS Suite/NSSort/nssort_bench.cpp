// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

/*
 * nssort_bench.cpp — Manual chrono benchmark for NSSort v10 vs IPS4o
 *
 * Methodology matches the original sealed benchmark:
 *   - Element type: int64_t for both NSSort and IPS4o
 *   - IPS4o mode: ips4o::parallel::sort (multi-threaded)
 *   - Timing: manual std::chrono, 3 warmup + 3 timed runs
 *   - Cache flush: 64MB buffer filled before each timed run
 *   - Report: 3 raw times + median
 *
 * 10 distributions at n=100M.
 */

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <algorithm>
#include <array>
#include <cmath>

#define _REENTRANT
#include "ips4o/ips4o.hpp"
#include "NSSort.hpp"

using namespace std;
using namespace std::chrono;

constexpr size_t N = 100'000'000;  // 100M

// ---------------------------------------------------------------------------
// Distribution generators — all produce int64_t directly
// ---------------------------------------------------------------------------

void gen_zero(int64_t* data, size_t n) {
    memset(data, 0, n * sizeof(int64_t));
}

void gen_sorted(int64_t* data, size_t n) {
    for (size_t i = 0; i < n; i++)
        data[i] = static_cast<int64_t>(i);
}

void gen_reverse(int64_t* data, size_t n) {
    for (size_t i = 0; i < n; i++)
        data[i] = static_cast<int64_t>(n - i);
}

void gen_almost_sorted(int64_t* data, size_t n) {
    for (size_t i = 0; i < n; i++)
        data[i] = static_cast<int64_t>(i);
    mt19937 rng(42);
    size_t swaps = n / 100;  // n/100, not n/10
    uniform_int_distribution<size_t> dist(0, n - 1);
    for (size_t i = 0; i < swaps; i++) {
        size_t a = dist(rng), b = dist(rng);
        std::swap(data[a], data[b]);
    }
}

void gen_root_dup(int64_t* data, size_t n) {
    for (size_t i = 0; i < n; i++)
        data[i] = static_cast<int64_t>(sqrt((double)i));
}

void gen_two_dup(int64_t* data, size_t n) {
    mt19937 rng(42);
    uniform_int_distribution<int64_t> dist(0, 1);
    for (size_t i = 0; i < n; i++)
        data[i] = dist(rng);
}

void gen_eight_dup(int64_t* data, size_t n) {
    mt19937 rng(42);
    uniform_int_distribution<int64_t> dist(0, 7);
    for (size_t i = 0; i < n; i++)
        data[i] = dist(rng);
}

void gen_zipf(int64_t* data, size_t n) {
    mt19937 rng(42);
    const int max_unique = 1000;
    const double s = 1.0;

    vector<double> probs(max_unique);
    double sum = 0.0;
    for (int i = 1; i <= max_unique; i++) {
        probs[i - 1] = 1.0 / pow((double)i, s);
        sum += probs[i - 1];
    }

    vector<double> cdf(max_unique);
    cdf[0] = probs[0] / sum;
    for (int i = 1; i < max_unique; i++)
        cdf[i] = cdf[i - 1] + probs[i] / sum;

    uniform_real_distribution<double> udist(0.0, 1.0);
    for (size_t i = 0; i < n; i++) {
        double r = udist(rng);
        int64_t val = 0;
        for (int j = 0; j < max_unique; j++) {
            if (r <= cdf[j]) {
                val = j;
                break;
            }
        }
        data[i] = val;
    }
}

void gen_exponential(int64_t* data, size_t n) {
    mt19937 rng(42);
    exponential_distribution<double> exp_dist(0.0001);
    for (size_t i = 0; i < n; i++)
        data[i] = static_cast<int64_t>(exp_dist(rng));
}

void gen_uniform(int64_t* data, size_t n) {
    mt19937_64 rng(42);
    for (size_t i = 0; i < n; i++)
        data[i] = static_cast<int64_t>(rng());
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

bool is_sorted_check(int64_t* data, size_t n) {
    for (size_t i = 1; i < n; i++)
        if (data[i] < data[i - 1]) return false;
    return true;
}

// ---------------------------------------------------------------------------
// Benchmark harness — matches original methodology
// ---------------------------------------------------------------------------

array<double, 3> benchmark_ns_sort(int64_t* data, size_t n, int warm_runs) {
    vector<int64_t> copy(n);
    static vector<char> flush_buf(64 * 1024 * 1024);

    // Warmup
    for (int i = 0; i < warm_runs; i++) {
        std::copy(data, data + n, copy.data());
        nssort(copy.data(), n);
    }

    // 3 timed runs
    array<double, 3> times;
    for (int run = 0; run < 3; run++) {
        // Flush CPU cache
        std::fill(flush_buf.begin(), flush_buf.end(), 1);
        (void)flush_buf[0];

        // Fresh copy
        std::copy(data, data + n, copy.data());

        auto start = high_resolution_clock::now();
        nssort(copy.data(), n);
        auto end = high_resolution_clock::now();

        if (!is_sorted_check(copy.data(), n)) {
            cerr << "ERROR: nssort produced unsorted output!" << endl;
        }

        times[run] = duration_cast<microseconds>(end - start).count() / 1000.0;
    }

    return times;
}

array<double, 3> benchmark_ips4o_sort(int64_t* data, size_t n, int warm_runs) {
    vector<int64_t> copy(n);
    static vector<char> flush_buf(64 * 1024 * 1024);

    // Warmup
    for (int i = 0; i < warm_runs; i++) {
        std::copy(data, data + n, copy.data());
        ips4o::parallel::sort(copy.data(), copy.data() + n);
    }

    // 3 timed runs
    array<double, 3> times;
    for (int run = 0; run < 3; run++) {
        // Flush CPU cache
        std::fill(flush_buf.begin(), flush_buf.end(), 1);
        (void)flush_buf[0];

        // Fresh copy
        std::copy(data, data + n, copy.data());

        auto start = high_resolution_clock::now();
        ips4o::parallel::sort(copy.data(), copy.data() + n);
        auto end = high_resolution_clock::now();

        if (!is_sorted_check(copy.data(), n)) {
            cerr << "ERROR: ips4o produced unsorted output!" << endl;
        }

        times[run] = duration_cast<microseconds>(end - start).count() / 1000.0;
    }

    return times;
}

double median_of_three(array<double, 3> times) {
    array<double, 3> sorted = times;
    std::sort(sorted.begin(), sorted.end());
    return sorted[1];
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    cout << "=== NSSort vs IPS4o (parallel) — 100M int64_t ===" << endl;
    cout << "Methodology: 3 warmup + 3 timed runs, 64MB cache flush, median reported" << endl;
    cout << endl;

    vector<int64_t> data(N);

    struct Dist {
        const char* name;
        void (*gen)(int64_t*, size_t);
    };

    Dist dists[] = {
        {"zero",          gen_zero},
        {"sorted",        gen_sorted},
        {"reverse",       gen_reverse},
        {"almost_sorted", gen_almost_sorted},
        {"root_dup",      gen_root_dup},
        {"two_dup",       gen_two_dup},
        {"eight_dup",     gen_eight_dup},
        {"zipf",          gen_zipf},
        {"exponential",   gen_exponential},
        {"uniform",       gen_uniform},
    };

    cout << left;
    cout << setw(16) << "Distribution"
         << setw(20) << "NSSort (ms)"
         << setw(20) << "IPS4o (ms)"
         << setw(12) << "Speedup"
         << endl;
    cout << string(68, '-') << endl;

    for (auto& d : dists) {
        d.gen(data.data(), N);

        auto ns_times = benchmark_ns_sort(data.data(), N, 3);
        auto ips4o_times = benchmark_ips4o_sort(data.data(), N, 3);

        double ns_median = median_of_three(ns_times);
        double ips4o_median = median_of_three(ips4o_times);
        double speedup = ips4o_median / ns_median;

        cout << setw(16) << d.name
             << setw(20) << fixed << setprecision(3) << ns_median
             << setw(20) << fixed << setprecision(3) << ips4o_median
             << setw(12) << fixed << setprecision(2) << speedup << "x"
             << endl;

        // Raw times to stderr for logging
        cerr << "[" << d.name << "]"
             << " ns: " << ns_times[0] << " " << ns_times[1] << " " << ns_times[2]
             << " | ips4o: " << ips4o_times[0] << " " << ips4o_times[1] << " " << ips4o_times[2]
             << endl;
    }

    return 0;
}

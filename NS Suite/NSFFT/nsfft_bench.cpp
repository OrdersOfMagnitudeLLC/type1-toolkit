// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#include "NSFFT.hpp"
#include <fftw3.h>
#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <algorithm>
#include <chrono>
#include <fstream>

using namespace nsfft;

// Generate sparse signal with K non-zero frequencies
std::vector<double> generate_sparse_signal(int N, int K) {
    std::vector<double> signal(N, 0.0);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> freq_dist(0, N/2);
    std::uniform_real_distribution<double> phase_dist(0.0, 2.0 * M_PI);
    std::uniform_real_distribution<double> amp_dist(0.5, 2.0);
    
    std::vector<int> used_freqs;
    for (int k = 0; k < K; ) {
        int freq = freq_dist(gen);
        if (std::find(used_freqs.begin(), used_freqs.end(), freq) == used_freqs.end()) {
            used_freqs.push_back(freq);
            double phase = phase_dist(gen);
            double amplitude = amp_dist(gen);
            
            for (int n = 0; n < N; ++n) {
                signal[n] += amplitude * std::cos(2.0 * M_PI * freq * n / N + phase);
            }
            k++;
        }
    }
    
    return signal;
}

std::vector<double> generate_band_limited_signal(int N, int f_low, int f_high) {
    std::vector<double> signal(N, 0.0);
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> phase_dist(0.0, 2.0 * M_PI);
    std::uniform_real_distribution<double> amp_dist(0.5, 2.0);
    
    for (int freq = f_low; freq <= f_high; ++freq) {
        double phase = phase_dist(gen);
        double amplitude = amp_dist(gen);
        
        for (int n = 0; n < N; ++n) {
            signal[n] += amplitude * std::cos(2.0 * M_PI * freq * n / N + phase);
        }
    }
    
    return signal;
}

double run_fftw_band(const double* signal, int N, int f_low, int f_high) {
    fftw_complex* in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
    
    for (int n = 0; n < N; ++n) {
        in[n][0] = signal[n];
        in[n][1] = 0.0;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    fftw_plan plan = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    auto end = std::chrono::high_resolution_clock::now();
    
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);
    
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double run_fftw_sparse(const double* signal, int N, int K) {
    fftw_complex* in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
    
    for (int n = 0; n < N; ++n) {
        in[n][0] = signal[n];
        in[n][1] = 0.0;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    fftw_plan plan = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    auto end = std::chrono::high_resolution_clock::now();
    
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);
    
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main() {
    const int N = 1048576;
    const int K = 10;
    
    std::ofstream out("benchmark.txt");
    
    // nsfft_band vs FFTW benchmarks
    std::cout << "nsfft_band vs FFTW" << std::endl;
    std::cout << "==================" << std::endl;
    out << "nsfft_band vs FFTW" << std::endl;
    out << "==================" << std::endl;
    
    // Test 1: 5 bins
    {
        auto signal = generate_band_limited_signal(N, 100, 104);
        
        double nsfft_min = 1e9;
        for (int i = 0; i < 3; ++i) {
            auto result = nsfft_band(signal.data(), N, 100, 104);
            nsfft_min = std::min(nsfft_min, result.elapsed_ms);
        }
        
        double fftw_min = 1e9;
        for (int i = 0; i < 3; ++i) {
            fftw_min = std::min(fftw_min, run_fftw_band(signal.data(), N, 100, 104));
        }
        
        std::cout << "N=" << N << " f_low=100 f_high=104 (5 bins)" << std::endl;
        std::cout << "nsfft_band: " << nsfft_min << " ms" << std::endl;
        std::cout << "fftw: " << fftw_min << " ms" << std::endl;
        std::cout << std::endl;
        
        out << "N=" << N << " f_low=100 f_high=104 (5 bins)" << std::endl;
        out << "nsfft_band: " << nsfft_min << " ms" << std::endl;
        out << "fftw: " << fftw_min << " ms" << std::endl;
        out << std::endl;
    }
    
    // Test 2: 10 bins
    {
        auto signal = generate_band_limited_signal(N, 100, 109);
        
        double nsfft_min = 1e9;
        for (int i = 0; i < 3; ++i) {
            auto result = nsfft_band(signal.data(), N, 100, 109);
            nsfft_min = std::min(nsfft_min, result.elapsed_ms);
        }
        
        double fftw_min = 1e9;
        for (int i = 0; i < 3; ++i) {
            fftw_min = std::min(fftw_min, run_fftw_band(signal.data(), N, 100, 109));
        }
        
        std::cout << "N=" << N << " f_low=100 f_high=109 (10 bins)" << std::endl;
        std::cout << "nsfft_band: " << nsfft_min << " ms" << std::endl;
        std::cout << "fftw: " << fftw_min << " ms" << std::endl;
        std::cout << std::endl;
        
        out << "N=" << N << " f_low=100 f_high=109 (10 bins)" << std::endl;
        out << "nsfft_band: " << nsfft_min << " ms" << std::endl;
        out << "fftw: " << fftw_min << " ms" << std::endl;
        out << std::endl;
    }
    
    // Test 3: 50 bins
    {
        auto signal = generate_band_limited_signal(N, 100, 149);
        
        double nsfft_min = 1e9;
        for (int i = 0; i < 3; ++i) {
            auto result = nsfft_band(signal.data(), N, 100, 149);
            nsfft_min = std::min(nsfft_min, result.elapsed_ms);
        }
        
        double fftw_min = 1e9;
        for (int i = 0; i < 3; ++i) {
            fftw_min = std::min(fftw_min, run_fftw_band(signal.data(), N, 100, 149));
        }
        
        std::cout << "N=" << N << " f_low=100 f_high=149 (50 bins)" << std::endl;
        std::cout << "nsfft_band: " << nsfft_min << " ms" << std::endl;
        std::cout << "fftw: " << fftw_min << " ms" << std::endl;
        std::cout << std::endl;
        
        out << "N=" << N << " f_low=100 f_high=149 (50 bins)" << std::endl;
        out << "nsfft_band: " << nsfft_min << " ms" << std::endl;
        out << "fftw: " << fftw_min << " ms" << std::endl;
        out << std::endl;
    }
    
    // Test 4: 100 bins
    {
        auto signal = generate_band_limited_signal(N, 100, 199);
        
        double nsfft_min = 1e9;
        for (int i = 0; i < 3; ++i) {
            auto result = nsfft_band(signal.data(), N, 100, 199);
            nsfft_min = std::min(nsfft_min, result.elapsed_ms);
        }
        
        double fftw_min = 1e9;
        for (int i = 0; i < 3; ++i) {
            fftw_min = std::min(fftw_min, run_fftw_band(signal.data(), N, 100, 199));
        }
        
        std::cout << "N=" << N << " f_low=100 f_high=199 (100 bins)" << std::endl;
        std::cout << "nsfft_band: " << nsfft_min << " ms" << std::endl;
        std::cout << "fftw: " << fftw_min << " ms" << std::endl;
        std::cout << std::endl;
        
        out << "N=" << N << " f_low=100 f_high=199 (100 bins)" << std::endl;
        out << "nsfft_band: " << nsfft_min << " ms" << std::endl;
        out << "fftw: " << fftw_min << " ms" << std::endl;
        out << std::endl;
    }
    
    // nsfft_sparse vs FFTW benchmarks
    std::cout << "nsfft_sparse vs FFTW" << std::endl;
    std::cout << "====================" << std::endl;
    out << "nsfft_sparse vs FFTW" << std::endl;
    out << "====================" << std::endl;
    
    // Test different K values
    std::vector<int> K_values = {5, 10, 20, 50};
    for (int test_K : K_values) {
        int exact_matches = 0;
        double total_time = 0.0;
        
        for (int trial = 0; trial < 1000; ++trial) {
            auto signal = generate_sparse_signal(N, test_K);
            auto result = nsfft_sparse(signal.data(), N, test_K);
            total_time += result.elapsed_ms;
            
            // Simple correctness check: non-zero magnitudes
            int non_zero = 0;
            for (double mag : result.magnitudes) {
                if (mag > 0.01) non_zero++;
            }
            if (non_zero >= test_K) exact_matches++;
        }
        
        std::cout << "N=" << N << " K=" << test_K << " 1000 trials" << std::endl;
        std::cout << "recovery_rate: " << exact_matches << "/1000" << std::endl;
        std::cout << "avg_time: " << (total_time / 1000.0) << " ms" << std::endl;
        std::cout << std::endl;
        
        out << "N=" << N << " K=" << test_K << " 1000 trials" << std::endl;
        out << "recovery_rate: " << exact_matches << "/1000" << std::endl;
        out << "avg_time: " << (total_time / 1000.0) << " ms" << std::endl;
        out << std::endl;
    }
    
    // Single benchmark run for K=10 vs FFTW
    {
        auto signal = generate_sparse_signal(N, K);
        
        double nsfft_min = 1e9;
        for (int i = 0; i < 3; ++i) {
            auto result = nsfft_sparse(signal.data(), N, K);
            nsfft_min = std::min(nsfft_min, result.elapsed_ms);
        }
        
        double fftw_min = 1e9;
        for (int i = 0; i < 3; ++i) {
            fftw_min = std::min(fftw_min, run_fftw_sparse(signal.data(), N, K));
        }
        
        std::cout << "N=" << N << " K=" << K << " single benchmark" << std::endl;
        std::cout << "nsfft_sparse: " << nsfft_min << " ms" << std::endl;
        std::cout << "fftw: " << fftw_min << " ms" << std::endl;
        std::cout << std::endl;
        
        out << "N=" << N << " K=" << K << " single benchmark" << std::endl;
        out << "nsfft_sparse: " << nsfft_min << " ms" << std::endl;
        out << "fftw: " << fftw_min << " ms" << std::endl;
        out << std::endl;
    }
    
    out.close();
    
    return 0;
}

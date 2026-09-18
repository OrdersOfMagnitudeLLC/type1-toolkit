// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#ifndef NSFFT_HPP
#define NSFFT_HPP

#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>

namespace nsfft {

struct NSFFTResult {
    std::vector<int>    freqs;
    std::vector<double> magnitudes;
    double              elapsed_ms;
};

namespace internal {

// Small FFT for bucket processing (Cooley-Tukey, power of 2 only)
inline void small_fft(std::complex<double>* data, int n) {
    if (n <= 1) return;
    
    // Bit-reversal permutation
    int j = 0;
    for (int i = 1; i < n; ++i) {
        int m = n >> 1;
        while (j >= m) {
            j -= m;
            m >>= 1;
        }
        j += m;
        if (i < j) {
            std::swap(data[i], data[j]);
        }
    }
    
    // Cooley-Tukey FFT
    for (int len = 2; len <= n; len <<= 1) {
        double angle = -2.0 * M_PI / len;
        std::complex<double> wlen(std::cos(angle), std::sin(angle));
        
        for (int i = 0; i < n; i += len) {
            std::complex<double> w(1.0, 0.0);
            for (int j = 0; j < len / 2; ++j) {
                std::complex<double> u = data[i + j];
                std::complex<double> v = data[i + j + len / 2] * w;
                data[i + j] = u + v;
                data[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

// Compute DFT at single frequency using Goertzel algorithm
inline std::complex<double> goertzel(const double* signal, int N, int k) {
    const double omega = 2.0 * M_PI * k / N;
    const double cos_omega = std::cos(omega);
    const double sin_omega = std::sin(omega);
    const double cos_2omega = 2.0 * cos_omega;
    
    double s_prev = 0.0;
    double s_prev2 = 0.0;
    
    for (int n = 0; n < N; ++n) {
        double s = signal[n] + cos_2omega * s_prev - s_prev2;
        s_prev2 = s_prev;
        s_prev = s;
    }
    
    double re = s_prev - cos_omega * s_prev2;
    double im = sin_omega * s_prev2;
    
    return std::complex<double>(re, im);
}

// Compute DFT in frequency range [f_low, f_high] using Goertzel
inline std::vector<std::complex<double>> goertzel_range(const double* signal, int N, 
                                                         int f_low, int f_high) {
    int num_freqs = f_high - f_low + 1;
    std::vector<std::complex<double>> result(num_freqs);
    
    for (int k = f_low; k <= f_high; ++k) {
        result[k - f_low] = goertzel(signal, N, k);
    }
    
    return result;
}

} // namespace internal

// FFAST-based sparse FFT using subsampling + peeling decoder with adaptive B retry
inline NSFFTResult nsfft_sparse_crt(const double* signal, int N, int K) {
    NSFFTResult result;
    result.freqs.resize(K);
    result.magnitudes.resize(K);
    auto start = std::chrono::high_resolution_clock::now();
    
    if (K <= 0 || N <= 0) {
        result.elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - start).count();
        return result;
    }
    
    // Helper function to run FFAST with specific B and timeout
    auto run_ffast_with_B = [&](int B_val, double timeout_ms) {
        std::vector<int> found_freqs;
        std::vector<double> found_mags;
        auto pass_start = std::chrono::high_resolution_clock::now();
        int stride_val = N / B_val;
        
        // Random delay offset to break structural collisions
        srand((unsigned)time(nullptr));
        int delay_offset = rand() % stride_val;
        
        // Subsample with 3 delays and compute FFTs
        std::vector<std::vector<std::complex<double>>> Y(3);
        for (int d = 0; d < 3; ++d) {
            Y[d].resize(B_val, std::complex<double>(0.0, 0.0));
            for (int b = 0; b < B_val; ++b) {
                int idx = (b * stride_val + d + delay_offset) % N;
                Y[d][b] = signal[idx];
            }
            internal::small_fft(Y[d].data(), B_val);
        }
        
        // Peeling decoder with timeout check
        std::vector<std::vector<std::complex<double>>> Y_copy = Y;
        
        while (found_freqs.size() < K) {
            // Check timeout
            double elapsed = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - pass_start).count();
            if (elapsed > timeout_ms) {
                return std::make_pair(found_freqs, found_mags);
            }
            
            size_t prev_found = found_freqs.size();
            
            for (int i = 0; i < B_val; ++i) {
                if (found_freqs.size() >= K) break;
                if (std::abs(Y_copy[0][i]) < 1e-6) continue;
                
                std::complex<double> ratio1 = Y_copy[1][i] / Y_copy[0][i];
                std::complex<double> ratio2 = Y_copy[2][i] / Y_copy[0][i];
                
                double phase1 = std::arg(ratio1);
                double phase2 = std::arg(ratio2);
                
                double expected_phase2 = 2.0 * phase1;
                while (expected_phase2 > M_PI) expected_phase2 -= 2.0 * M_PI;
                while (expected_phase2 < -M_PI) expected_phase2 += 2.0 * M_PI;
                
                double phase_diff = std::abs(phase2 - expected_phase2);
                while (phase_diff > M_PI) phase_diff -= 2.0 * M_PI;
                phase_diff = std::abs(phase_diff);
                
                if (phase_diff < 0.5) {
                    int f = (int)std::round(N * phase1 / (2.0 * M_PI));
                    if (f < 0) f += N;
                    if (f >= N/2) f = N - f;
                    
                    bool already_found = false;
                    for (int existing : found_freqs) {
                        if (existing == f) { already_found = true; break; }
                    }
                    if (already_found) continue;
                    
                    double mag = (f == 0) ? std::abs(Y_copy[0][i]) / B_val : std::abs(Y_copy[0][i]) * 2.0 / B_val;
                    
                    found_freqs.push_back(f);
                    found_mags.push_back(mag);
                    
                    int k_pos = ((f % B_val) + B_val) % B_val;
                    int k_neg = (B_val - k_pos) % B_val;
                    for (int d = 0; d < 3; ++d) {
                        Y_copy[d][k_pos] -= (mag * B_val / 2.0) * std::exp(std::complex<double>(0, 2.0*M_PI*f*d/N));
                        if (k_neg != k_pos)
                            Y_copy[d][k_neg] -= (mag * B_val / 2.0) * std::exp(std::complex<double>(0, -2.0*M_PI*f*d/N));
                    }
                }
            }
            
            if (found_freqs.size() == prev_found) {
                break;
            }
        }
        
        return std::make_pair(found_freqs, found_mags);
    };
    
    // Adaptive B retry: start at B=256, double if K not found or timeout
    int B = 256;
    const double TIMEOUT_MS = 5.0;
    const int MAX_RETRIES = 5;
    std::vector<int> found_freqs;
    std::vector<double> found_mags;
    
    for (int retry = 0; retry <= MAX_RETRIES; ++retry) {
        auto [freqs, mags] = run_ffast_with_B(B, TIMEOUT_MS);
        found_freqs = freqs;
        found_mags = mags;
        
        if (found_freqs.size() >= K) {
            // Fill results
            for (size_t i = 0; i < K; ++i) {
                if (i < found_freqs.size()) {
                    result.freqs[i] = found_freqs[i];
                    result.magnitudes[i] = found_mags[i];
                } else {
                    result.freqs[i] = 0;
                    result.magnitudes[i] = 0.0;
                }
            }
            
            result.elapsed_ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - start).count();
            
            return result;
        }
        
        B *= 2;
    }
    
    // Fallback: use whatever we found from last attempt
    for (size_t i = 0; i < K; ++i) {
        if (i < found_freqs.size()) {
            result.freqs[i] = found_freqs[i];
            result.magnitudes[i] = found_mags[i];
        } else {
            result.freqs[i] = 0;
            result.magnitudes[i] = 0.0;
        }
    }
    
    result.elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - start).count();
    
    return result;
}

// Public sparse FFT wrapper
inline NSFFTResult nsfft_sparse(const double* signal, int N, int K) {
    return nsfft_sparse_crt(signal, N, K);
}

// Band-limited mode: compute DFT only in [f_low, f_high]
inline NSFFTResult nsfft_band(const double* signal, int N, int f_low, int f_high) {
    auto start = std::chrono::high_resolution_clock::now();
    
    NSFFTResult result;
    
    if (N <= 0 || f_low < 0 || f_high >= N || f_low > f_high) {
        result.elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - start).count();
        return result;
    }
    
    // Use Goertzel algorithm for the specified frequency range
    auto coeffs = internal::goertzel_range(signal, N, f_low, f_high);
    
    int num_freqs = f_high - f_low + 1;
    result.freqs.resize(num_freqs);
    result.magnitudes.resize(num_freqs);
    
    for (int i = 0; i < num_freqs; ++i) {
        result.freqs[i] = f_low + i;
        result.magnitudes[i] = std::abs(coeffs[i]);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    return result;
}

// Pruned FFT: compute only butterflies needed for target output bins
} // namespace nsfft

#endif // NSFFT_HPP

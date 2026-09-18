// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cstring>
#include <immintrin.h>

const char SOH = '\x01';
const int NUM_MESSAGES = 1000000;

std::vector<std::string> generate_fix_messages(int count) {
    std::vector<std::string> messages;
    messages.reserve(count);
    
    std::string template_msg = "8=FIX.4.4";
    template_msg += SOH;
    template_msg += "9=150";
    template_msg += SOH;
    template_msg += "35=D";
    template_msg += SOH;
    template_msg += "49=CLIENT";
    template_msg += SOH;
    template_msg += "56=BROKER";
    template_msg += SOH;
    template_msg += "34=1";
    template_msg += SOH;
    template_msg += "52=20240101-12:00:00";
    template_msg += SOH;
    template_msg += "55=AAPL";
    template_msg += SOH;
    template_msg += "54=1";
    template_msg += SOH;
    template_msg += "38=100";
    template_msg += SOH;
    template_msg += "40=2";
    template_msg += SOH;
    template_msg += "44=150.00";
    template_msg += SOH;
    template_msg += "10=099";
    template_msg += SOH;
    
    for (int i = 0; i < count; i++) {
        messages.push_back(template_msg);
    }
    
    return messages;
}

int naive_parse(const std::vector<std::string>& messages) {
    int total_fields = 0;
    int msgtype_count = 0;
    int symbol_count = 0;
    
    for (const auto& msg : messages) {
        const char* data = msg.data();
        size_t len = msg.length();
        
        size_t field_start = 0;
        for (size_t i = 0; i < len; i++) {
            if (data[i] == SOH) {
                // Parse field
                const char* field = data + field_start;
                size_t field_len = i - field_start;
                
                // Find '=' to split tag from value
                for (size_t j = 0; j < field_len; j++) {
                    if (field[j] == '=') {
                        // Extract tag
                        if (field_len >= 3 && field[0] == '3' && field[1] == '5') {
                            msgtype_count++;
                        }
                        if (field_len >= 3 && field[0] == '5' && field[1] == '5') {
                            symbol_count++;
                        }
                        break;
                    }
                }
                
                total_fields++;
                field_start = i + 1;
            }
        }
    }
    
    return total_fields;
}

int simd_parse(const std::vector<std::string>& messages) {
    int total_fields = 0;
    int msgtype_count = 0;
    int symbol_count = 0;
    
    __m256i soh_vec = _mm256_set1_epi8(SOH);
    
    for (const auto& msg : messages) {
        const char* data = msg.data();
        size_t len = msg.length();
        
        size_t pos = 0;
        size_t field_start = 0;
        
        while (pos + 32 <= len) {
            __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + pos));
            __m256i cmp = _mm256_cmpeq_epi8(chunk, soh_vec);
            int mask = _mm256_movemask_epi8(cmp);
            
            while (mask != 0) {
                int bit_pos = __builtin_ctz(mask);
                size_t delimiter_pos = pos + bit_pos;
                
                // Parse field
                const char* field = data + field_start;
                size_t field_len = delimiter_pos - field_start;
                
                // Find '=' to split tag from value
                for (size_t j = 0; j < field_len; j++) {
                    if (field[j] == '=') {
                        // Extract tag
                        if (field_len >= 3 && field[0] == '3' && field[1] == '5') {
                            msgtype_count++;
                        }
                        if (field_len >= 3 && field[0] == '5' && field[1] == '5') {
                            symbol_count++;
                        }
                        break;
                    }
                }
                
                total_fields++;
                field_start = delimiter_pos + 1;
                
                mask &= mask - 1;
            }
            
            pos += 32;
        }
        
        // Handle remaining bytes
        for (size_t i = pos; i < len; i++) {
            if (data[i] == SOH) {
                const char* field = data + field_start;
                size_t field_len = i - field_start;
                
                for (size_t j = 0; j < field_len; j++) {
                    if (field[j] == '=') {
                        if (field_len >= 3 && field[0] == '3' && field[1] == '5') {
                            msgtype_count++;
                        }
                        if (field_len >= 3 && field[0] == '5' && field[1] == '5') {
                            symbol_count++;
                        }
                        break;
                    }
                }
                
                total_fields++;
                field_start = i + 1;
            }
        }
    }
    
    return total_fields;
}

int main() {
    std::cout << "Generating " << NUM_MESSAGES << " FIX messages..." << std::endl;
    auto messages = generate_fix_messages(NUM_MESSAGES);
    std::cout << "Generated " << messages.size() << " messages, total size: " 
              << (messages.size() * messages[0].length() / 1024 / 1024) << " MB" << std::endl;
    
    // Warmup
    naive_parse(messages);
    simd_parse(messages);
    
    // Benchmark NaiveParser (multiple iterations for accuracy)
    std::cout << "\nBenchmarking NaiveParser..." << std::endl;
    const int iterations = 100;
    int naive_total = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        naive_total += naive_parse(messages);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto naive_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    double naive_msg_per_sec = (double)NUM_MESSAGES * iterations / naive_duration * 1000000.0;
    std::cout << "Naive total fields: " << naive_total << std::endl;
    
    // Benchmark SIMDParser (multiple iterations for accuracy)
    std::cout << "Benchmarking SIMDParser..." << std::endl;
    int simd_total = 0;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        simd_total += simd_parse(messages);
    }
    end = std::chrono::high_resolution_clock::now();
    auto simd_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    double simd_msg_per_sec = (double)NUM_MESSAGES * iterations / simd_duration * 1000000.0;
    std::cout << "SIMD total fields: " << simd_total << std::endl;
    
    // Single run for correctness check
    int naive_fields = naive_parse(messages);
    int simd_fields = simd_parse(messages);
    
    // Results
    std::cout << "\n=== RESULTS ===" << std::endl;
    std::cout << "NaiveParser: " << naive_msg_per_sec << " messages/sec (" << naive_duration / 1000.0 << " ms)" << std::endl;
    std::cout << "SIMDParser:  " << simd_msg_per_sec << " messages/sec (" << simd_duration / 1000.0 << " ms)" << std::endl;
    std::cout << "Speedup:     " << (simd_msg_per_sec / naive_msg_per_sec) << "x" << std::endl;
    std::cout << "Naive fields: " << naive_fields << std::endl;
    std::cout << "SIMD fields:  " << simd_fields << std::endl;
    std::cout << "Correctness:  " << (naive_fields == simd_fields ? "PASS" : "FAIL") << std::endl;
    
    return 0;
}

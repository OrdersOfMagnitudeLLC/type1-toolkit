// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#include "nsfix.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <immintrin.h>

using namespace nsfix;

const int NUM_MESSAGES = 10000000;  // 10M messages for pure parse measurement
const char SOH = '\x01';

uint32_t calculate_checksum(const std::string& msg) {
    uint32_t checksum = 0;
    for (char c : msg) {
        checksum += static_cast<uint8_t>(c);
    }
    return checksum % 256;
}

std::string generate_new_order_single(int seq_num) {
    std::ostringstream body_oss;
    body_oss << "35=D" << SOH;
    body_oss << "49=CLIENT" << SOH;
    body_oss << "56=BROKER" << SOH;
    body_oss << "34=" << seq_num << SOH;
    body_oss << "52=20240101-12:00:00" << SOH;
    body_oss << "11=ORD" << seq_num << SOH;
    body_oss << "55=AAPL" << SOH;
    body_oss << "54=1" << SOH;
    body_oss << "38=100" << SOH;
    body_oss << "44=150.50" << SOH;
    body_oss << "40=2" << SOH;
    
    std::string body = body_oss.str();
    
    std::ostringstream oss;
    oss << "8=FIX.4.2" << SOH;
    oss << "9=" << body.length() << SOH;
    oss << body;
    
    std::string msg = oss.str();
    
    uint32_t checksum = calculate_checksum(msg);
    
    oss.str("");
    oss << "10=" << std::setw(3) << std::setfill('0') << checksum << SOH;
    msg += oss.str();
    
    return msg;
}

std::vector<std::string> generate_messages(int count) {
    std::vector<std::string> messages;
    messages.reserve(count);
    
    for (int i = 2; i < count + 2; i++) {
        messages.push_back(generate_new_order_single(i));
    }
    
    return messages;
}

std::string concatenate_messages(const std::vector<std::string>& messages) {
    std::string result;
    size_t total_size = 0;
    for (const auto& msg : messages) {
        total_size += msg.length();
    }
    result.reserve(total_size);
    for (const auto& msg : messages) {
        result += msg;
    }
    return result;
}

// Pure parse benchmark - no callbacks, no session, just parsing
size_t total_fields_written = 0;

double bench_parse_only(const std::string& buffer, int iterations) {
    total_fields_written = 0;
    
    // Use same dispatch array as Session
    uint8_t dispatch_array[10001] = {0};
    dispatch_array[35] = 1;  // MsgType
    dispatch_array[49] = 2;  // SenderCompID
    dispatch_array[56] = 3;  // TargetCompID
    dispatch_array[34] = 4;  // MsgSeqNum
    dispatch_array[52] = 5;  // SendingTime
    dispatch_array[11] = 6;  // ClOrdID
    dispatch_array[55] = 7;  // Symbol
    dispatch_array[54] = 8;  // Side
    dispatch_array[38] = 9;  // OrderQty
    dispatch_array[44] = 10; // Price
    dispatch_array[40] = 11; // OrdType
    dispatch_array[8]  = 12; // BeginString
    dispatch_array[9]  = 13; // BodyLength
    dispatch_array[10] = 14; // CheckSum

    static constexpr int MAX_TOTAL = 4096;
    uint16_t soh_pos[MAX_TOTAL];
    uint16_t eq_pos[MAX_TOTAL];
    
    // Warmup
    for (int iter = 0; iter < 1; iter++) {
        int soh_count = 0, eq_count = 0;
        const __m256i soh_vec = _mm256_set1_epi8('\x01');
        const __m256i eq_vec  = _mm256_set1_epi8('=');
        size_t pos = 0;
        while (pos + 32 <= buffer.length() && soh_count < MAX_TOTAL - 8) {
            __m256i chunk = _mm256_loadu_si256(
                              reinterpret_cast<const __m256i*>(buffer.data() + pos));
            int sm = _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, soh_vec));
            int em = _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, eq_vec));
            while (sm) {
                soh_pos[soh_count++] = static_cast<uint16_t>(pos + __builtin_ctz(sm));
                sm &= sm - 1;
            }
            while (em) {
                eq_pos[eq_count++]  = static_cast<uint16_t>(pos + __builtin_ctz(em));
                em &= em - 1;
            }
            pos += 32;
        }
        for (size_t i = pos; i < buffer.length(); i++) {
            if (buffer[i] == '\x01' && soh_count < MAX_TOTAL) soh_pos[soh_count++] = i;
            if (buffer[i] == '='    && eq_count  < MAX_TOTAL) eq_pos[eq_count++]   = i;
        }

        // Parse fields
        uint16_t field_start = 0;
        int ei = 0;
        for (int fi = 0; fi < soh_count; fi++) {
            uint16_t soh = soh_pos[fi];
            while (ei < eq_count && eq_pos[ei] < field_start) ei++;
            if (ei >= eq_count || eq_pos[ei] >= soh) {
                field_start = soh + 1;
                continue;
            }
            uint16_t eq      = eq_pos[ei++];
            uint8_t  tag_len = static_cast<uint8_t>(eq - field_start);
            const char* tp   = buffer.data() + field_start;

            uint32_t tag = 0;
            switch (tag_len) {
                case 1: tag = tp[0]-'0'; break;
                case 2: tag = (tp[0]-'0')*10  + (tp[1]-'0'); break;
                case 3: tag = (tp[0]-'0')*100 + (tp[1]-'0')*10  + (tp[2]-'0'); break;
                case 4: tag = (tp[0]-'0')*1000+ (tp[1]-'0')*100
                            + (tp[2]-'0')*10  + (tp[3]-'0'); break;
                default:
                    for (uint8_t k = 0; k < tag_len; k++)
                        tag = tag*10 + (tp[k]-'0');
            }

            uint8_t slot = (tag < 10001) ? dispatch_array[tag] : 0;
            if (slot) {
                total_fields_written++;
            }
            field_start = soh + 1;
        }
    }

    // Timed runs
    total_fields_written = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; iter++) {
        int soh_count = 0, eq_count = 0;
        const __m256i soh_vec = _mm256_set1_epi8('\x01');
        const __m256i eq_vec  = _mm256_set1_epi8('=');
        size_t pos = 0;
        while (pos + 32 <= buffer.length() && soh_count < MAX_TOTAL - 8) {
            __m256i chunk = _mm256_loadu_si256(
                              reinterpret_cast<const __m256i*>(buffer.data() + pos));
            int sm = _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, soh_vec));
            int em = _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, eq_vec));
            while (sm) {
                soh_pos[soh_count++] = static_cast<uint16_t>(pos + __builtin_ctz(sm));
                sm &= sm - 1;
            }
            while (em) {
                eq_pos[eq_count++]  = static_cast<uint16_t>(pos + __builtin_ctz(em));
                em &= em - 1;
            }
            pos += 32;
        }
        for (size_t i = pos; i < buffer.length(); i++) {
            if (buffer[i] == '\x01' && soh_count < MAX_TOTAL) soh_pos[soh_count++] = i;
            if (buffer[i] == '='    && eq_count  < MAX_TOTAL) eq_pos[eq_count++]   = i;
        }

        // Parse fields
        uint16_t field_start = 0;
        int ei = 0;
        for (int fi = 0; fi < soh_count; fi++) {
            uint16_t soh = soh_pos[fi];
            while (ei < eq_count && eq_pos[ei] < field_start) ei++;
            if (ei >= eq_count || eq_pos[ei] >= soh) {
                field_start = soh + 1;
                continue;
            }
            uint16_t eq      = eq_pos[ei++];
            uint8_t  tag_len = static_cast<uint8_t>(eq - field_start);
            const char* tp   = buffer.data() + field_start;

            uint32_t tag = 0;
            switch (tag_len) {
                case 1: tag = tp[0]-'0'; break;
                case 2: tag = (tp[0]-'0')*10  + (tp[1]-'0'); break;
                case 3: tag = (tp[0]-'0')*100 + (tp[1]-'0')*10  + (tp[2]-'0'); break;
                case 4: tag = (tp[0]-'0')*1000+ (tp[1]-'0')*100
                            + (tp[2]-'0')*10  + (tp[3]-'0'); break;
                default:
                    for (uint8_t k = 0; k < tag_len; k++)
                        tag = tag*10 + (tp[k]-'0');
            }

            uint8_t slot = (tag < 10001) ? dispatch_array[tag] : 0;
            if (slot) {
                total_fields_written++;
            }
            field_start = soh + 1;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    double msg_per_sec = (double)NUM_MESSAGES * iterations / duration * 1000000.0;
    
    return msg_per_sec;
}

int main() {
    std::cout << "=== Pure Parse Benchmark ===" << std::endl;
    std::cout << "Generating " << NUM_MESSAGES << " NewOrderSingle messages..." << std::endl;
    
    auto messages = generate_messages(NUM_MESSAGES);
    std::cout << "Generated " << messages.size() << " messages" << std::endl;
    
    std::string buffer = concatenate_messages(messages);
    std::cout << "Concatenated buffer size: " << buffer.length() << " bytes" << std::endl;
    
    const int warmup_runs = 2;
    const int timed_runs = 5;
    
    std::cout << "\n=== Pure Parse (no callbacks, no session) ===" << std::endl;
    std::cout << "Warmup runs: " << warmup_runs << std::endl;
    std::cout << "Timed runs: " << timed_runs << std::endl;
    
    double parse_speed = bench_parse_only(buffer, timed_runs);
    std::cout << "Pure parse: " << parse_speed << " messages/sec" << std::endl;
    std::cout << "Total fields written: " << total_fields_written << std::endl;
    
    return 0;
}

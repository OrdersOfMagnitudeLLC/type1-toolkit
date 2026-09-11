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
#include <cstring>

using namespace nsfix;

const int NUM_MESSAGES = 100000;
const char SOH = '\x01';

// Calculate FIX checksum
uint32_t calculate_checksum(const std::string& msg) {
    uint32_t checksum = 0;
    for (char c : msg) {
        checksum += static_cast<uint8_t>(c);
    }
    return checksum % 256;
}

// Generate a single NewOrderSingle message with valid checksum
std::string generate_new_order_single(int seq_num) {
    // Build body first (everything after BeginString and BodyLength)
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
    
    // Build full message
    std::ostringstream oss;
    oss << "8=FIX.4.2" << SOH;
    oss << "9=" << body.length() << SOH;
    oss << body;
    
    std::string msg = oss.str();
    
    // Calculate checksum
    uint32_t checksum = calculate_checksum(msg);
    
    // Append checksum
    oss.str("");
    oss << "10=" << std::setw(3) << std::setfill('0') << checksum << SOH;
    msg += oss.str();
    
    return msg;
}

// Generate all messages
std::vector<std::string> generate_messages(int count) {
    std::vector<std::string> messages;
    messages.reserve(count);
    
    // Start from seq=2 since seq=1 was used for logon
    for (int i = 2; i < count + 2; i++) {
        messages.push_back(generate_new_order_single(i));
    }
    
    return messages;
}

// === DIAGNOSTIC 1: Raw parse benchmark (no Session, no callback) ===
int raw_parse_field_count = 0;

double benchmark_raw_parse(const std::vector<std::string>& messages, int iterations) {
    raw_parse_field_count = 0;
    
    const char SOH = '\x01';
    
    // Warmup
    for (const auto& msg : messages) {
        const char* buf = msg.data();
        size_t len = msg.length();
        
        size_t field_start = 0;
        
        Message msg_obj;
        msg_obj.reset();
        
        // Scalar parse for diagnostic
        for (size_t i = 0; i < len; i++) {
            if (buf[i] == SOH) {
                const char* eq = static_cast<const char*>(memchr(buf + field_start, '=', i - field_start));
                if (eq) {
                    size_t tag_len = eq - (buf + field_start);
                    uint16_t tag = 0;
                    for (size_t j = 0; j < tag_len; j++) {
                        if (buf[field_start + j] >= '0' && buf[field_start + j] <= '9') {
                            tag = tag * 10 + (buf[field_start + j] - '0');
                        }
                    }
                    
                    uint8_t slot = dispatch_array[tag];
                    if (slot != 0) {
                        const char* value = eq + 1;
                        size_t value_len = i - field_start - tag_len - 1;
                        msg_obj.slots[slot] = std::string_view(value, value_len);
                        msg_obj.dirty_[slot] = 1;
                        raw_parse_field_count++;
                    }
                }
                field_start = i + 1;
            }
        }
    }
    
    // Reset for timed run
    raw_parse_field_count = 0;
    
    // Timed runs
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; iter++) {
        for (const auto& msg : messages) {
            const char* buf = msg.data();
            size_t len = msg.length();
            
            size_t field_start = 0;
            
            Message msg_obj;
            msg_obj.reset();
            
            // Scalar parse for diagnostic
            for (size_t i = 0; i < len; i++) {
                if (buf[i] == SOH) {
                    const char* eq = static_cast<const char*>(memchr(buf + field_start, '=', i - field_start));
                    if (eq) {
                        size_t tag_len = eq - (buf + field_start);
                        uint16_t tag = 0;
                        for (size_t j = 0; j < tag_len; j++) {
                            if (buf[field_start + j] >= '0' && buf[field_start + j] <= '9') {
                                tag = tag * 10 + (buf[field_start + j] - '0');
                            }
                        }
                        
                        uint8_t slot = dispatch_array[tag];
                        if (slot != 0) {
                            const char* value = eq + 1;
                            size_t value_len = i - field_start - tag_len - 1;
                            msg_obj.slots[slot] = std::string_view(value, value_len);
                            msg_obj.dirty_[slot] = 1;
                            raw_parse_field_count++;
                        }
                    }
                    field_start = i + 1;
                }
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    double msg_per_sec = (double)NUM_MESSAGES * iterations / duration * 1000000.0;
    
    return msg_per_sec;
}

// === DIAGNOSTIC 2: Contiguous buffer benchmark ===
int contiguous_msg_count = 0;
void contiguous_callback(Message& msg) {
    contiguous_msg_count++;
}

double benchmark_contiguous_buffer(const std::vector<std::string>& messages, int iterations) {
    contiguous_msg_count = 0;
    
    // Concatenate all messages into one flat buffer
    size_t total_size = 0;
    for (const auto& msg : messages) {
        total_size += msg.length();
    }
    
    char* buffer = new char[total_size];
    size_t offset = 0;
    for (const auto& msg : messages) {
        memcpy(buffer + offset, msg.data(), msg.length());
        offset += msg.length();
    }
    
    Session session("CLIENT", "BROKER", contiguous_callback);
    
    // Simulate logon to get session into ACTIVE state
    const char* logon_ack = "8=FIX.4.2\x01"
                           "9=70\x01"
                           "35=A\x01"
                           "49=BROKER\x01"
                           "56=CLIENT\x01"
                           "34=1\x01"
                           "52=20240101-12:00:00\x01"
                           "98=0\x01"
                           "108=30\x01"
                           "10=123\x01";
    session.feed(logon_ack, strlen(logon_ack));
    
    const size_t chunk_size = 4096;
    
    // Warmup - feed each message individually (simulating realistic chunking)
    for (const auto& msg : messages) {
        size_t msg_len = msg.length();
        size_t pos = 0;
        while (pos < msg_len) {
            size_t chunk_len = std::min(chunk_size, msg_len - pos);
            session.feed(msg.data() + pos, chunk_len);
            pos += chunk_len;
        }
    }
    
    // Reset for timed run
    contiguous_msg_count = 0;
    
    // Timed runs
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; iter++) {
        for (const auto& msg : messages) {
            size_t msg_len = msg.length();
            size_t pos = 0;
            while (pos < msg_len) {
                size_t chunk_len = std::min(chunk_size, msg_len - pos);
                session.feed(msg.data() + pos, chunk_len);
                pos += chunk_len;
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    double msg_per_sec = (double)NUM_MESSAGES * iterations / duration * 1000000.0;
    
    delete[] buffer;
    return msg_per_sec;
}

int main() {
    std::cout << "=== Diagnostic Benchmarks ===" << std::endl;
    std::cout << "Generating " << NUM_MESSAGES << " NewOrderSingle messages..." << std::endl;
    
    auto messages = generate_messages(NUM_MESSAGES);
    std::cout << "Generated " << messages.size() << " messages" << std::endl;
    std::cout << "Sample message: " << messages[0] << std::endl;
    
    const int warmup_runs = 5;
    const int timed_runs = 10;
    
    // Diagnostic 1: Raw parse benchmark
    std::cout << "\n=== Diagnostic 1: Raw Parse (No Session, No Callback) ===" << std::endl;
    std::cout << "Warmup runs: " << warmup_runs << std::endl;
    std::cout << "Timed runs: " << timed_runs << std::endl;
    
    double raw_parse_speed = benchmark_raw_parse(messages, timed_runs);
    std::cout << "Raw parse: " << raw_parse_speed << " messages/sec" << std::endl;
    std::cout << "Fields touched: " << raw_parse_field_count << std::endl;
    
    // Diagnostic 2: Contiguous buffer benchmark
    std::cout << "\n=== Diagnostic 2: Contiguous Buffer (4096-byte chunks) ===" << std::endl;
    std::cout << "Warmup runs: " << warmup_runs << std::endl;
    std::cout << "Timed runs: " << timed_runs << std::endl;
    
    double contiguous_speed = benchmark_contiguous_buffer(messages, timed_runs);
    std::cout << "Contiguous buffer: " << contiguous_speed << " messages/sec" << std::endl;
    std::cout << "Messages processed: " << contiguous_msg_count << std::endl;
    
    return 0;
}

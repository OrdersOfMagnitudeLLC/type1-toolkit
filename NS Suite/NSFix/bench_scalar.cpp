// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <sstream>

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

// Generate all messages
std::vector<std::string> generate_messages(int count) {
    std::vector<std::string> messages;
    messages.reserve(count);
    
    for (int i = 2; i < count + 2; i++) {
        messages.push_back(generate_new_order_single(i));
    }
    
    return messages;
}

// Clean scalar FIX parser - no SIMD, no hash dispatch
struct ScalarParser {
    const char* msg_type = nullptr;
    size_t msg_type_len = 0;
    const char* sender = nullptr;
    size_t sender_len = 0;
    const char* target = nullptr;
    size_t target_len = 0;
    const char* symbol = nullptr;
    size_t symbol_len = 0;
    const char* side = nullptr;
    size_t side_len = 0;
    const char* order_qty = nullptr;
    size_t order_qty_len = 0;
    const char* price = nullptr;
    size_t price_len = 0;
    
    void reset() {
        msg_type = nullptr;
        msg_type_len = 0;
        sender = nullptr;
        sender_len = 0;
        target = nullptr;
        target_len = 0;
        symbol = nullptr;
        symbol_len = 0;
        side = nullptr;
        side_len = 0;
        order_qty = nullptr;
        order_qty_len = 0;
        price = nullptr;
        price_len = 0;
    }
    
    void parse(const char* data, size_t len) {
        size_t field_start = 0;
        
        for (size_t i = 0; i < len; i++) {
            if (data[i] == SOH) {
                // Find '=' in this field
                size_t eq_pos = field_start;
                while (eq_pos < i && data[eq_pos] != '=') {
                    eq_pos++;
                }
                
                if (eq_pos < i) {
                    // Parse tag number
                    uint16_t tag = 0;
                    for (size_t j = field_start; j < eq_pos; j++) {
                        if (data[j] >= '0' && data[j] <= '9') {
                            tag = tag * 10 + (data[j] - '0');
                        }
                    }
                    
                    // Store value pointer based on tag
                    const char* value = data + eq_pos + 1;
                    size_t value_len = i - eq_pos - 1;
                    
                    switch (tag) {
                        case 35: // MsgType
                            msg_type = value;
                            msg_type_len = value_len;
                            break;
                        case 49: // SenderCompID
                            sender = value;
                            sender_len = value_len;
                            break;
                        case 56: // TargetCompID
                            target = value;
                            target_len = value_len;
                            break;
                        case 55: // Symbol
                            symbol = value;
                            symbol_len = value_len;
                            break;
                        case 54: // Side
                            side = value;
                            side_len = value_len;
                            break;
                        case 38: // OrderQty
                            order_qty = value;
                            order_qty_len = value_len;
                            break;
                        case 44: // Price
                            price = value;
                            price_len = value_len;
                            break;
                    }
                }
                
                field_start = i + 1;
            }
        }
    }
};

// Scalar parser benchmark
int scalar_field_count = 0;
double benchmark_scalar(const std::vector<std::string>& messages, int iterations) {
    scalar_field_count = 0;
    
    // Warmup
    for (const auto& msg : messages) {
        ScalarParser parser;
        parser.parse(msg.data(), msg.length());
        if (parser.msg_type) scalar_field_count++;
        if (parser.sender) scalar_field_count++;
        if (parser.target) scalar_field_count++;
        if (parser.symbol) scalar_field_count++;
        if (parser.side) scalar_field_count++;
        if (parser.order_qty) scalar_field_count++;
        if (parser.price) scalar_field_count++;
    }
    
    // Reset for timed run
    scalar_field_count = 0;
    
    // Timed runs
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; iter++) {
        for (const auto& msg : messages) {
            ScalarParser parser;
            parser.parse(msg.data(), msg.length());
            if (parser.msg_type) scalar_field_count++;
            if (parser.sender) scalar_field_count++;
            if (parser.target) scalar_field_count++;
            if (parser.symbol) scalar_field_count++;
            if (parser.side) scalar_field_count++;
            if (parser.order_qty) scalar_field_count++;
            if (parser.price) scalar_field_count++;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    double msg_per_sec = (double)NUM_MESSAGES * iterations / duration * 1000000.0;
    
    return msg_per_sec;
}

int main() {
    std::cout << "=== Scalar Parser Benchmark ===" << std::endl;
    std::cout << "Generating " << NUM_MESSAGES << " NewOrderSingle messages..." << std::endl;
    
    auto messages = generate_messages(NUM_MESSAGES);
    std::cout << "Generated " << messages.size() << " messages" << std::endl;
    std::cout << "Sample message: " << messages[0] << std::endl;
    
    const int warmup_runs = 5;
    const int timed_runs = 10;
    
    std::cout << "\n=== Scalar Parser Benchmark ===" << std::endl;
    std::cout << "Warmup runs: " << warmup_runs << std::endl;
    std::cout << "Timed runs: " << timed_runs << std::endl;
    
    double scalar_speed = benchmark_scalar(messages, timed_runs);
    std::cout << "Scalar parser: " << scalar_speed << " messages/sec" << std::endl;
    std::cout << "Scalar field count: " << scalar_field_count << std::endl;
    
    return 0;
}

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

using namespace nsfix;

const int NUM_MESSAGES = 100000;
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

int callback_count = 0;
void callback(Message& msg) {
    callback_count++;
}

double benchmark_feed(const std::string& buffer, int iterations) {
    callback_count = 0;
    
    Session session("CLIENT", "BROKER", callback);
    
    // Simulate logon
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
    
    // Warmup - feed message by message
    for (int i = 0; i < 100; i++) {
        session.feed(buffer.data() + i * 100, 100);
    }
    
    // Reset for timed run
    callback_count = 0;
    
    // Timed runs - feed message by message
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; iter++) {
        size_t pos = 0;
        for (int i = 0; i < NUM_MESSAGES; i++) {
            size_t msg_len = buffer.find(SOH, pos) - pos + 1;
            // Find actual message length by finding next "8=FIX.4.2" or end
            size_t next_start = buffer.find("8=FIX.4.2", pos + 1);
            if (next_start == std::string::npos) {
                msg_len = buffer.length() - pos;
            } else {
                msg_len = next_start - pos;
            }
            session.feed(buffer.data() + pos, msg_len);
            pos += msg_len;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    double msg_per_sec = (double)NUM_MESSAGES * iterations / duration * 1000000.0;
    
    return msg_per_sec;
}

double benchmark_feed_batch(const std::string& buffer, int iterations) {
    callback_count = 0;
    
    Session session("CLIENT", "BROKER", callback);
    
    // Simulate logon
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
    
    // Warmup
    session.feed_batch(buffer.data(), buffer.length());
    
    // Reset for timed run
    callback_count = 0;
    
    // Timed runs
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; iter++) {
        session.feed_batch(buffer.data(), buffer.length());
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    double msg_per_sec = (double)NUM_MESSAGES * iterations / duration * 1000000.0;
    
    return msg_per_sec;
}

int main() {
#ifdef __AVX2__
    std::cout << "AVX2: YES" << std::endl;
#else
    std::cout << "AVX2: NO (scalar)" << std::endl;
#endif
    std::cout << "=== Batch Feed Benchmark ===" << std::endl;
    std::cout << "Generating " << NUM_MESSAGES << " NewOrderSingle messages..." << std::endl;
    
    auto messages = generate_messages(NUM_MESSAGES);
    std::cout << "Generated " << messages.size() << " messages" << std::endl;
    
    std::string buffer = concatenate_messages(messages);
    std::cout << "Concatenated buffer size: " << buffer.length() << " bytes" << std::endl;
    
    const int warmup_runs = 5;
    const int timed_runs = 10;
    
    std::cout << "\n=== Feed (message-by-message) ===" << std::endl;
    std::cout << "Warmup runs: " << warmup_runs << std::endl;
    std::cout << "Timed runs: " << timed_runs << std::endl;
    
    double feed_speed = benchmark_feed(buffer, timed_runs);
    std::cout << "Feed: " << feed_speed << " messages/sec" << std::endl;
    std::cout << "Feed callback count: " << callback_count << std::endl;
    
    std::cout << "\n=== Feed Batch (single sweep) ===" << std::endl;
    std::cout << "Warmup runs: " << warmup_runs << std::endl;
    std::cout << "Timed runs: " << timed_runs << std::endl;
    
    Session::reset_cap_hits();
    double batch_speed = benchmark_feed_batch(buffer, timed_runs);
    std::cout << "Feed Batch: " << batch_speed << " messages/sec" << std::endl;
    std::cout << "Feed Batch callback count: " << callback_count << std::endl;
    std::cout << "DIAGNOSTIC: MAX_TOTAL cap_hits: " << Session::get_cap_hits() << std::endl;
    std::cout << "DIAGNOSTIC: Messages actually processed: " << callback_count << std::endl;
    
    return 0;
}

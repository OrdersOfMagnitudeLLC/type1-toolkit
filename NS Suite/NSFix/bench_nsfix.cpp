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
#include <fstream>

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

// NSFix benchmark
int nsfix_callback_count = 0;
void nsfix_callback(Message& msg) {
    // Access all required fields
    msg.get(Tag::MsgType);
    msg.get(Tag::SenderCompID);
    msg.get(Tag::TargetCompID);
    msg.get(Tag::Symbol);
    msg.get(Tag::Side);
    msg.get_int(Tag::OrderQty);
    msg.get_double(Tag::Price);
    nsfix_callback_count++;
}

double benchmark_nsfix(const std::vector<std::string>& messages, int iterations) {
    nsfix_callback_count = 0;
    Session session("CLIENT", "BROKER", nsfix_callback);
    
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
    
    // Warmup
    for (const auto& msg : messages) {
        session.feed(msg.data(), msg.length());
    }
    
    // Reset callback count for timed run
    nsfix_callback_count = 0;
    
    // Timed runs
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; iter++) {
        for (const auto& msg : messages) {
            session.feed(msg.data(), msg.length());
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    double msg_per_sec = (double)NUM_MESSAGES * iterations / duration * 1000000.0;
    
    return msg_per_sec;
}

int main() {
    std::cout << "=== NSFix Benchmark ===" << std::endl;
    std::cout << "Generating " << NUM_MESSAGES << " NewOrderSingle messages..." << std::endl;
    
    auto messages = generate_messages(NUM_MESSAGES);
    std::cout << "Generated " << messages.size() << " messages" << std::endl;
    std::cout << "Sample message: " << messages[0] << std::endl;
    
    // Save messages to file for QuickFIX benchmark
    std::ofstream msg_file("messages.dat", std::ios::binary);
    for (const auto& msg : messages) {
        uint32_t len = msg.length();
        msg_file.write(reinterpret_cast<const char*>(&len), sizeof(len));
        msg_file.write(msg.data(), len);
    }
    msg_file.close();
    std::cout << "Messages saved to messages.dat" << std::endl;
    
    const int warmup_runs = 5;
    const int timed_runs = 10;
    
    std::cout << "\n=== NSFix Benchmark ===" << std::endl;
    std::cout << "Warmup runs: " << warmup_runs << std::endl;
    std::cout << "Timed runs: " << timed_runs << std::endl;
    
    double nsfix_speed = benchmark_nsfix(messages, timed_runs);
    std::cout << "NSFix: " << nsfix_speed << " messages/sec" << std::endl;
    std::cout << "NSFix callback count: " << nsfix_callback_count << std::endl;
    
    return 0;
}

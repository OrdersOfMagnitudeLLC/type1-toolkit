// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#include <quickfix/Message.h>
#include <quickfix/Field.h>
#include <quickfix/fix42/NewOrderSingle.h>
#include <quickfix/SessionID.h>
#include <quickfix/DataDictionary.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <fstream>

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
    // Use the same format as NSFix benchmark (which works for NSFix)
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

// QuickFIX benchmark
int quickfix_parse_count = 0;
int quickfix_field_count = 0;
double benchmark_quickfix(const std::vector<std::string>& messages, int iterations) {
    quickfix_parse_count = 0;
    quickfix_field_count = 0;
    
    // Load DataDictionary
    FIX::DataDictionary dataDictionary("/tmp/FIX42.xml");
    
    // Warmup - just test first message with minimal parsing
    try {
        FIX::Message msg;
        msg.setString(messages[0], false, &dataDictionary);  // validate=false, with DataDictionary
        // Call getField() for at least 5 tags per message
        msg.getHeader().getField(FIX::FIELD::MsgType);
        msg.getHeader().getField(FIX::FIELD::SenderCompID);
        msg.getHeader().getField(FIX::FIELD::TargetCompID);
        msg.getField(FIX::FIELD::Symbol);
        msg.getField(FIX::FIELD::Side);
        msg.getField(FIX::FIELD::OrderQty);
        msg.getField(FIX::FIELD::Price);
        quickfix_parse_count++;
        quickfix_field_count += 7;
    } catch (std::exception& e) {
        std::cout << "Parse error on first message: " << e.what() << std::endl;
        std::cout << "Message: " << messages[0] << std::endl;
        return 0.0;
    }
    
    // Warmup - rest of messages
    for (size_t i = 1; i < messages.size(); i++) {
        try {
            FIX::Message msg;
            msg.setString(messages[i], false, &dataDictionary);
            // Call getField() for at least 5 tags per message
            msg.getHeader().getField(FIX::FIELD::MsgType);
            msg.getHeader().getField(FIX::FIELD::SenderCompID);
            msg.getHeader().getField(FIX::FIELD::TargetCompID);
            msg.getField(FIX::FIELD::Symbol);
            msg.getField(FIX::FIELD::Side);
            msg.getField(FIX::FIELD::OrderQty);
            msg.getField(FIX::FIELD::Price);
            quickfix_parse_count++;
            quickfix_field_count += 7;
        } catch (...) {
            // Skip messages that fail to parse
        }
    }
    
    // Reset for timed run
    quickfix_parse_count = 0;
    quickfix_field_count = 0;
    
    // Timed runs
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; iter++) {
        for (const auto& msg_str : messages) {
            try {
                FIX::Message msg;
                msg.setString(msg_str, false, &dataDictionary);
                // Call getField() for at least 5 tags per message
                msg.getHeader().getField(FIX::FIELD::MsgType);
                msg.getHeader().getField(FIX::FIELD::SenderCompID);
                msg.getHeader().getField(FIX::FIELD::TargetCompID);
                msg.getField(FIX::FIELD::Symbol);
                msg.getField(FIX::FIELD::Side);
                msg.getField(FIX::FIELD::OrderQty);
                msg.getField(FIX::FIELD::Price);
                quickfix_parse_count++;
                quickfix_field_count += 7;
            } catch (...) {
                // Skip messages that fail to parse
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    double msg_per_sec = (double)NUM_MESSAGES * iterations / duration * 1000000.0;
    
    return msg_per_sec;
}

int main() {
    std::cout << "=== QuickFIX Benchmark ===" << std::endl;
    std::cout << "Generating " << NUM_MESSAGES << " NewOrderSingle messages..." << std::endl;
    
    auto messages = generate_messages(NUM_MESSAGES);
    std::cout << "Generated " << messages.size() << " messages" << std::endl;
    std::cout << "Sample message: " << messages[0] << std::endl;
    
    const int warmup_runs = 5;
    const int timed_runs = 10;
    
    std::cout << "\n=== QuickFIX Benchmark ===" << std::endl;
    std::cout << "Warmup runs: " << warmup_runs << std::endl;
    std::cout << "Timed runs: " << timed_runs << std::endl;
    
    double quickfix_speed = benchmark_quickfix(messages, timed_runs);
    std::cout << "QuickFIX: " << quickfix_speed << " messages/sec" << std::endl;
    std::cout << "QuickFIX parse count: " << quickfix_parse_count << std::endl;
    std::cout << "QuickFIX field count: " << quickfix_field_count << std::endl;
    
    return 0;
}

// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#include "nsfix.hpp"
#include "nsfix_log.hpp"
#include "nsfix_session_manager.hpp"
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

int nsfix_callback_count = 0;
void nsfix_callback(Message& msg) {
    msg.get(Tag::MsgType);
    msg.get(Tag::SenderCompID);
    msg.get(Tag::TargetCompID);
    msg.get(Tag::Symbol);
    msg.get(Tag::Side);
    msg.get_int(Tag::OrderQty);
    msg.get_double(Tag::Price);
    nsfix_callback_count++;
}

int main() {
    std::cout << "=== NSFix Full Feature Benchmark ===" << std::endl;
    std::cout << "Features: mmap persistence + NSHash session manager + recovery" << std::endl;
    std::cout << std::endl;

    // Generate test messages
    std::cout << "Generating " << NUM_MESSAGES << " NewOrderSingle messages..." << std::endl;
    auto messages = generate_messages(NUM_MESSAGES);
    std::cout << "Generated " << messages.size() << " messages" << std::endl;
    std::cout << "Sample message: " << messages[0] << std::endl;
    std::cout << std::endl;

    // === Feature 1: Memory-mapped persistence ===
    std::cout << "--- Feature 1: Memory-mapped persistence ---" << std::endl;
    FixLog log("nsfix_session.log");
    std::cout << "Log open: " << (log.is_open() ? "YES" : "NO") << std::endl;
    std::cout << "Log file: nsfix_session.log (256MB circular buffer)" << std::endl;

    // === Feature 2: Multi-session via NSHash ===
    std::cout << std::endl << "--- Feature 2: Multi-session via NSHash ---" << std::endl;
    SessionManager manager(64);
    FIXSessionState* state = manager.get_or_create_session("CLIENT", "BROKER", nsfix_callback);
    std::cout << "Session count: " << manager.session_count() << std::endl;
    std::cout << "Session state: seq_in=" << state->seq_in << " seq_out=" << state->seq_out << std::endl;

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
    state->session->feed(logon_ack, strlen(logon_ack));
    std::cout << "After logon: state=" << static_cast<int>(state->session->state()) << std::endl;

    // Create a second session to test multi-session
    FIXSessionState* state2 = manager.get_or_create_session("CLIENT2", "BROKER2", nsfix_callback);
    std::cout << "After 2nd session: session count: " << manager.session_count() << std::endl;
    std::cout << std::endl;

    // === Benchmark: parse + log (hot path) ===
    std::cout << "--- Benchmark: parse + mmap log (hot path) ---" << std::endl;
    const int warmup_runs = 5;
    const int timed_runs = 10;

    // Warmup
    for (int w = 0; w < warmup_runs; w++) {
        for (const auto& msg : messages) {
            state->session->feed(msg.data(), msg.length());
            log.log_message(msg.data(), msg.length());
        }
    }

    nsfix_callback_count = 0;
    log.reset(); // Clear log to avoid dirty page writeback contention
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < timed_runs; iter++) {
        for (const auto& msg : messages) {
            state->session->feed(msg.data(), msg.length());
            log.log_message(msg.data(), msg.length());
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    double msg_per_sec = (double)NUM_MESSAGES * timed_runs / duration_us * 1000000.0;

    std::cout << "Warmup runs: " << warmup_runs << std::endl;
    std::cout << "Timed runs: " << timed_runs << std::endl;
    std::cout << "NSFix (parse + mmap log): " << msg_per_sec << " messages/sec" << std::endl;
    std::cout << "Callback count: " << nsfix_callback_count << std::endl;
    std::cout << "Log write position: " << log.write_position() << " bytes" << std::endl;
    std::cout << std::endl;

    // === Feature 3: Recovery/replay ===
    std::cout << "--- Feature 3: Recovery/replay ---" << std::endl;
    // Simulate reconnect: peer reports gap at seq 99990, need to replay messages with seq > 99990
    uint32_t gap_start = NUM_MESSAGES - 10;
    std::cout << "Recovering messages with seq > " << gap_start << " for CLIENT/BROKER..." << std::endl;
    auto replay = log.recover_session("CLIENT", "BROKER", gap_start);
    std::cout << "Recovered " << replay.size() << " messages for replay" << std::endl;
    if (!replay.empty()) {
        std::cout << "First recovered message: " << replay[0].substr(0, 60) << "..." << std::endl;
    }

    // Test recovery for non-existent session
    auto replay2 = log.recover_session("NOBODY", "NOWHERE", 0);
    std::cout << "Recovery for non-existent session: " << replay2.size() << " messages" << std::endl;
    std::cout << std::endl;

    // === Summary ===
    std::cout << "=== Summary ===" << std::endl;
    std::cout << "NSFix (parse + mmap log): " << (int)msg_per_sec << " msg/sec" << std::endl;
    std::cout << "hffix equivalent workload: 8,137,954 msg/sec (reference)" << std::endl;
    std::cout << "Threshold: 8,000,000 msg/sec" << std::endl;
    if (msg_per_sec > 8000000.0) {
        std::cout << "PASS: Throughput above 8M msg/sec threshold" << std::endl;
    } else {
        std::cout << "FAIL: Throughput below 8M msg/sec threshold" << std::endl;
    }

    return 0;
}

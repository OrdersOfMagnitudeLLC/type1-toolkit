// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cstring>
#include <unordered_map>

const char SOH = '\x01';
const int NUM_MESSAGES = 1000000;

// Most common FIX tags (10-15 out of 500)
const char* COMMON_TAGS[] = {
    "8",   // BeginString
    "9",   // BodyLength
    "35",  // MsgType
    "49",  // SenderCompID
    "56",  // TargetCompID
    "34",  // MsgSeqNum
    "52",  // SendingTime
    "55",  // Symbol
    "54",  // Side
    "38",  // OrderQty
    "40",  // OrdType
    "44",  // Price
    "10"   // CheckSum
};
const int NUM_COMMON_TAGS = 13;

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

// Perfect hash using simple integer hash for common tags
int perfect_hash(const char* tag, int len) {
    // Simple but effective hash for our known tags
    if (len == 1) {
        return tag[0] - '0';  // Single digit tags: 8, 9
    } else if (len == 2) {
        return 10 + (tag[0] - '0') * 10 + (tag[1] - '0');  // Two digit tags: 10-99
    }
    return -1;  // Not a common tag
}

bool is_common_tag(int hash) {
    // Check if hash corresponds to one of our common tags
    // Common tags: 8, 9, 10, 34, 35, 38, 40, 44, 49, 52, 54, 55, 56
    switch (hash) {
        case 8: case 9: case 10: case 34: case 35: case 38:
        case 40: case 44: case 49: case 52: case 54: case 55: case 56:
            return true;
        default:
            return false;
    }
}

int naive_parse(const std::vector<std::string>& messages) {
    int total_fields = 0;
    
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
                        total_fields++;
                        break;
                    }
                }
                
                field_start = i + 1;
            }
        }
    }
    
    return total_fields;
}

int hash_dispatch_parse(const std::vector<std::string>& messages) {
    int total_fields = 0;
    int common_tag_hits = 0;
    
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
                        // Extract tag using perfect hash
                        int tag_hash = perfect_hash(field, j);
                        
                        if (is_common_tag(tag_hash)) {
                            common_tag_hits++;
                            // Fast path for common tags - skip value parsing
                            total_fields++;
                        } else {
                            // Slow path for uncommon tags - still count but skip
                            total_fields++;
                        }
                        break;
                    }
                }
                
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
    hash_dispatch_parse(messages);
    
    // Benchmark NaiveParser
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
    
    // Benchmark HashDispatchParser
    std::cout << "Benchmarking HashDispatchParser..." << std::endl;
    int hash_total = 0;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        hash_total += hash_dispatch_parse(messages);
    }
    end = std::chrono::high_resolution_clock::now();
    auto hash_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    double hash_msg_per_sec = (double)NUM_MESSAGES * iterations / hash_duration * 1000000.0;
    std::cout << "Hash total fields: " << hash_total << std::endl;
    
    // Single run for correctness check
    int naive_fields = naive_parse(messages);
    int hash_fields = hash_dispatch_parse(messages);
    
    // Results
    std::cout << "\n=== RESULTS ===" << std::endl;
    std::cout << "NaiveParser:        " << naive_msg_per_sec << " messages/sec (" << naive_duration / 1000.0 << " ms)" << std::endl;
    std::cout << "HashDispatchParser: " << hash_msg_per_sec << " messages/sec (" << hash_duration / 1000.0 << " ms)" << std::endl;
    std::cout << "Speedup:             " << (hash_msg_per_sec / naive_msg_per_sec) << "x" << std::endl;
    std::cout << "Naive fields:        " << naive_fields << std::endl;
    std::cout << "Hash fields:         " << hash_fields << std::endl;
    std::cout << "Correctness:         " << (naive_fields == hash_fields ? "PASS" : "FAIL") << std::endl;
    
    return 0;
}

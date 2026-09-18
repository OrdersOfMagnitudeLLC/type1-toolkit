// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <chrono>
#include <cstring>
#include "nlohmann_json.hpp"
#include "nsfix.hpp"

using json = nlohmann::json;

// Global storage for parsed messages
std::vector<json> g_parsed_messages;

// Callback function to capture parsed messages
void message_callback(nsfix::Message& msg) {
    json msg_json;
    
    // Extract common fields
    auto get_field = [&](uint16_t tag) -> std::string {
        std::string_view sv = msg.get(tag);
        return std::string(sv.data(), sv.length());
    };
    
    msg_json["BeginString"] = get_field(nsfix::Tag::BeginString);
    msg_json["MsgType"] = get_field(nsfix::Tag::MsgType);
    msg_json["SenderCompID"] = get_field(nsfix::Tag::SenderCompID);
    msg_json["TargetCompID"] = get_field(nsfix::Tag::TargetCompID);
    msg_json["MsgSeqNum"] = msg.get_int(nsfix::Tag::MsgSeqNum);
    msg_json["SendingTime"] = get_field(nsfix::Tag::SendingTime);
    msg_json["Symbol"] = get_field(nsfix::Tag::Symbol);
    msg_json["Side"] = get_field(nsfix::Tag::Side);
    msg_json["OrderQty"] = msg.get_int(nsfix::Tag::OrderQty);
    msg_json["Price"] = msg.get_double(nsfix::Tag::Price);
    msg_json["ClOrdID"] = get_field(nsfix::Tag::ClOrdID);
    
    g_parsed_messages.push_back(msg_json);
}

int main() {
    try {
        // Read entire stdin
        std::string input((std::istreambuf_iterator<char>(std::cin)),
                          std::istreambuf_iterator<char>());

        json request = json::parse(input);
        std::string action = request["action"];
        
        auto start = std::chrono::high_resolution_clock::now();

        json response;

        if (action == "parse" || action == "batch") {
            g_parsed_messages.clear();
            
            std::string sender = request.value("sender", "SENDER");
            std::string target = request.value("target", "TARGET");
            
            nsfix::Session session(sender, target, message_callback);
            
            if (action == "parse" && request.contains("message")) {
                // Single message parse
                std::string message = request["message"];
                session.feed(message.data(), message.length());
            } else if (action == "batch" && request.contains("messages")) {
                // Batch parse
                std::vector<std::string> messages = request["messages"];
                std::string combined;
                for (const auto& msg : messages) {
                    combined += msg;
                }
                size_t count = session.feed_batch(combined.data(), combined.length());
                response["processed_count"] = count;
            }
            
            response["parsed"] = g_parsed_messages;
            response["valid"] = true;
            
        } else if (action == "validate") {
            // Validation - basic structure check
            if (request.contains("message")) {
                std::string message = request["message"];
                bool has_checksum = message.find("10=") != std::string::npos;
                bool has_body_length = message.find("9=") != std::string::npos;
                bool has_msg_type = message.find("35=") != std::string::npos;
                
                response["valid"] = has_checksum && has_body_length && has_msg_type;
                response["errors"] = json::array();
                if (!has_checksum) response["errors"].push_back("Missing checksum");
                if (!has_body_length) response["errors"].push_back("Missing body length");
                if (!has_msg_type) response["errors"].push_back("Missing message type");
            } else {
                response["valid"] = false;
                response["errors"] = {"No message provided"};
            }
            
        } else {
            json error = {{"error", "Unknown action"}};
            std::cout << error.dump() << std::endl;
            return 1;
        }

        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        response["time_ms"] = time_ms;

        std::cout << response.dump() << std::endl;

    } catch (const std::exception& e) {
        json error = {{"error", e.what()}};
        std::cout << error.dump() << std::endl;
        return 1;
    }

    return 0;
}

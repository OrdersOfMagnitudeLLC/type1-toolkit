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
#include "nsindex.hpp"

using json = nlohmann::json;

// Global index storage
static std::vector<uint64_t> g_index_data;
static NSIndex<uint64_t>* g_index = nullptr;

int main() {
    try {
        // Read entire stdin
        std::string input((std::istreambuf_iterator<char>(std::cin)),
                          std::istreambuf_iterator<char>());

        json request = json::parse(input);
        std::string action = request["action"];
        
        auto start = std::chrono::high_resolution_clock::now();

        json response;

        if (action == "build") {
            if (!request.contains("data")) {
                json error = {{"error", "data array is required"}};
                std::cout << error.dump() << std::endl;
                return 1;
            }
            
            std::vector<uint64_t> data = request["data"];
            
            // Sort data (NSIndex requires sorted keys)
            std::sort(data.begin(), data.end());
            
            // Store globally for subsequent queries
            g_index_data = data;
            
            // Clean up existing index
            delete g_index;
            g_index = new NSIndex<uint64_t>(g_index_data.data(), g_index_data.size());
            
            response["index_id"] = "default";
            response["indexed_count"] = g_index_data.size();
            response["build_time_ms"] = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - start).count();
            
        } else if (action == "query") {
            if (!g_index) {
                json error = {{"error", "Index not built. Call build first."}};
                std::cout << error.dump() << std::endl;
                return 1;
            }
            
            if (!request.contains("key")) {
                json error = {{"error", "key is required for query"}};
                std::cout << error.dump() << std::endl;
                return 1;
            }
            
            uint64_t key = request["key"];
            
            const uint64_t* result = g_index->find(g_index_data.data(), g_index_data.size(), key);
            
            if (result != nullptr && result != g_index_data.data() + g_index_data.size()) {
                response["found"] = true;
                response["result"] = *result;
                response["position"] = result - g_index_data.data();
            } else {
                response["found"] = false;
                response["result"] = nullptr;
            }
            
            response["query_time_ms"] = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - start).count();
            
        } else if (action == "range") {
            if (!g_index) {
                json error = {{"error", "Index not built. Call build first."}};
                std::cout << error.dump() << std::endl;
                return 1;
            }
            
            if (!request.contains("lo") || !request.contains("hi")) {
                json error = {{"error", "lo and hi are required for range query"}};
                std::cout << error.dump() << std::endl;
                return 1;
            }
            
            uint64_t lo = request["lo"];
            uint64_t hi = request["hi"];
            int max_out = request.value("max_out", 1000);
            
            std::vector<uint64_t> results(max_out);
            int count = g_index->find_range(g_index_data.data(), g_index_data.size(), lo, hi, results.data(), max_out);
            
            results.resize(count);
            response["results"] = results;
            response["match_count"] = count;
            response["query_time_ms"] = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - start).count();
            
        } else {
            json error = {{"error", "Unknown action"}};
            std::cout << error.dump() << std::endl;
            return 1;
        }

        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        if (!response.contains("build_time_ms") && !response.contains("query_time_ms")) {
            response["time_ms"] = time_ms;
        }

        std::cout << response.dump() << std::endl;

    } catch (const std::exception& e) {
        json error = {{"error", e.what()}};
        std::cout << error.dump() << std::endl;
        return 1;
    }

    return 0;
}

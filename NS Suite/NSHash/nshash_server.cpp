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
#include "nshash.hpp"

using json = nlohmann::json;

int main() {
    try {
        // Read entire stdin
        std::string input((std::istreambuf_iterator<char>(std::cin)),
                          std::istreambuf_iterator<char>());

        json request = json::parse(input);
        std::string action = request["action"];
        
        auto start = std::chrono::high_resolution_clock::now();

        json response;

        if (action == "dedup") {
            if (!request.contains("items")) {
                json error = {{"error", "items array is required"}};
                std::cout << error.dump() << std::endl;
                return 1;
            }
            
            std::vector<std::string> items = request["items"];
            size_t original_count = items.size();
            
            // Use NSHash for deduplication with string hashing
            NSHash<uint64_t, uint64_t, WyHash> hash_table;
            hash_table.reserve(original_count);
            
            std::vector<std::string> unique_items;
            size_t duplicates_removed = 0;
            
            for (const auto& item : items) {
                uint64_t hash = nsh_hash_str(item.data(), item.length());
                
                if (hash_table.lookup(hash) == nullptr) {
                    hash_table.insert(hash, 1);
                    unique_items.push_back(item);
                } else {
                    duplicates_removed++;
                }
            }
            
            response["unique"] = unique_items;
            response["original_count"] = original_count;
            response["unique_count"] = unique_items.size();
            response["duplicates_removed"] = duplicates_removed;
            
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

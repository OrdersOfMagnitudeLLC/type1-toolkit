// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#include "nscomp.hpp"
#include <iostream>
#include <random>
#include <chrono>
#include <cstring>
#include "lz4.h"
#include <zstd.h>

using namespace nscomp;
using namespace std::chrono;

// Entity struct (36 bytes with padding)
struct Entity {
    float x, y, z;        // position: 12 bytes
    float vx, vy, vz;     // velocity: 12 bytes
    int32_t hp;           // health: 4 bytes
    int32_t id;           // sequential IDs: 4 bytes
    uint8_t state;        // state flags: 1 byte
    uint8_t team;         // 0 or 1: 1 byte
    uint16_t flags;       // bitmask: 2 bytes
};  // 36 bytes total (with padding)

// Schema definition for Entity
Schema entity_schema = {
    36, // struct_size (actual size with padding)
    {
        {"position", FieldType::FLOAT32, 0, 3},
        {"velocity", FieldType::FLOAT32, 12, 3},
        {"hp",       FieldType::INT32,   24, 1},
        {"id",       FieldType::INT32,   28, 1},
        {"state",    FieldType::UINT8,   32, 1},
        {"team",     FieldType::UINT8,   33, 1},
        {"flags",    FieldType::INT16,   34, 1},
    }
};

// Generate synthetic game data
std::vector<Entity> generate_entities(size_t n_records) {
    std::vector<Entity> entities(n_records);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> pos_small(0.0f, 0.1f);
    std::uniform_real_distribution<float> pos_medium(0.1f, 1.0f);
    std::uniform_real_distribution<float> vel_small(-0.5f, 0.5f);
    std::uniform_int_distribution<int32_t> hp_change(-5, 0);
    std::uniform_int_distribution<uint8_t> state_dist(0, 3);
    std::uniform_int_distribution<uint16_t> flags_dist(0, 255);
    std::uniform_int_distribution<uint32_t> rare_check(0, 999);

    float x = 0.0f, y = 0.0f, z = 0.0f;
    int32_t hp = 100;
    uint8_t state = 0;
    uint8_t team = 0;
    uint16_t flags = 0;

    for (size_t i = 0; i < n_records; i++) {
        // Position: 80% standing still, 15% small delta, 5% medium delta
        uint32_t pos_roll = rare_check(rng);
        if (pos_roll < 800) {
            // 80%: no movement
        } else if (pos_roll < 950) {
            // 15%: small delta
            x += pos_small(rng);
            y += pos_small(rng);
            z += pos_small(rng);
        } else {
            // 5%: medium delta
            x += pos_medium(rng);
            y += pos_medium(rng);
            z += pos_medium(rng);
        }
        entities[i].x = x;
        entities[i].y = y;
        entities[i].z = z;

        // Velocity: 90% exactly 0.0f, 10% small random
        if (rare_check(rng) < 900) {
            entities[i].vx = 0.0f;
            entities[i].vy = 0.0f;
            entities[i].vz = 0.0f;
        } else {
            entities[i].vx = vel_small(rng);
            entities[i].vy = vel_small(rng);
            entities[i].vz = vel_small(rng);
        }

        // HP: changes only 1% of records
        if (rare_check(rng) < 10) {
            hp += hp_change(rng);
            if (hp < 0) hp = 0;
        }
        entities[i].hp = hp;

        // ID: strictly sequential
        entities[i].id = static_cast<int32_t>(i);

        // State: changes only 0.1% of records
        if (rare_check(rng) < 1) {
            state = state_dist(rng);
        }
        entities[i].state = state;

        // Team: changes only 0.1% of records
        if (rare_check(rng) < 1) {
            team = (team == 0) ? 1 : 0;
        }
        entities[i].team = team;

        // Flags: changes only 0.5% of records
        if (rare_check(rng) < 5) {
            flags = flags_dist(rng);
        }
        entities[i].flags = flags;
    }

    return entities;
}

int main() {
    const size_t n_records = 1000000;
    std::cout << "=== NSComp Schema-Aware Benchmark ===" << std::endl;
    std::cout << "Records: " << n_records << std::endl;
    std::cout << "Struct size: " << sizeof(Entity) << " bytes" << std::endl;
    std::cout << std::endl;

    // Generate synthetic data
    auto entities = generate_entities(n_records);
    size_t original_size = n_records * sizeof(Entity);
    std::cout << "Original size: " << original_size << " bytes (" << original_size / 1024.0 / 1024.0 << " MiB)" << std::endl;
    std::cout << std::endl;

    // NSComp benchmark
    {
        NSComp compressor;
        const uint8_t* data = reinterpret_cast<const uint8_t*>(entities.data());

        auto start = high_resolution_clock::now();
        auto compressed = compressor.compress_records(data, n_records, entity_schema);
        auto end = high_resolution_clock::now();
        auto compress_time = duration_cast<microseconds>(end - start).count() / 1000.0;

        std::vector<uint8_t> decompressed(n_records * sizeof(Entity));
        std::memcpy(decompressed.data(), data, decompressed.size());  // Preserve padding
        start = high_resolution_clock::now();
        bool success = compressor.decompress_records(compressed.data(), decompressed.data(), n_records, entity_schema);
        end = high_resolution_clock::now();
        auto decompress_time = duration_cast<microseconds>(end - start).count() / 1000.0;

        double ratio = static_cast<double>(original_size) / compressed.size();
        bool lossless = (std::memcmp(data, decompressed.data(), original_size) == 0);

        std::cout << "NSComp:  " << ratio << "x ratio | compress " << compress_time << "ms | decompress " << decompress_time << "ms | lossless " << (lossless ? "YES" : "NO") << std::endl;
    }

    // LZ4 raw benchmark
    {
        const uint8_t* data = reinterpret_cast<const uint8_t*>(entities.data());
        int max_compressed_size = LZ4_compressBound(original_size);
        std::vector<uint8_t> compressed(max_compressed_size);

        auto start = high_resolution_clock::now();
        int compressed_size = LZ4_compress_default(
            reinterpret_cast<const char*>(data),
            reinterpret_cast<char*>(compressed.data()),
            original_size,
            max_compressed_size
        );
        auto end = high_resolution_clock::now();
        auto compress_time = duration_cast<microseconds>(end - start).count() / 1000.0;

        compressed.resize(compressed_size);

        std::vector<uint8_t> decompressed(original_size);
        start = high_resolution_clock::now();
        int result = LZ4_decompress_safe(
            reinterpret_cast<const char*>(compressed.data()),
            reinterpret_cast<char*>(decompressed.data()),
            compressed_size,
            original_size
        );
        auto end2 = high_resolution_clock::now();
        auto decompress_time = duration_cast<microseconds>(end2 - start).count() / 1000.0;

        double ratio = static_cast<double>(original_size) / compressed_size;

        std::cout << "LZ4 raw: " << ratio << "x ratio | compress " << compress_time << "ms | decompress " << decompress_time << "ms" << std::endl;
    }

    // zstd raw benchmark (level 3)
    {
        const uint8_t* data = reinterpret_cast<const uint8_t*>(entities.data());
        size_t max_compressed_size = ZSTD_compressBound(original_size);
        std::vector<uint8_t> compressed(max_compressed_size);

        auto start = high_resolution_clock::now();
        size_t compressed_size = ZSTD_compress(
            compressed.data(),
            max_compressed_size,
            data,
            original_size,
            3  // compression level
        );
        auto end = high_resolution_clock::now();
        auto compress_time = duration_cast<microseconds>(end - start).count() / 1000.0;

        compressed.resize(compressed_size);

        std::vector<uint8_t> decompressed(original_size);
        start = high_resolution_clock::now();
        size_t result = ZSTD_decompress(
            decompressed.data(),
            original_size,
            compressed.data(),
            compressed_size
        );
        auto end2 = high_resolution_clock::now();
        auto decompress_time = duration_cast<microseconds>(end2 - start).count() / 1000.0;

        double ratio = static_cast<double>(original_size) / compressed_size;

        std::cout << "zstd raw: " << ratio << "x ratio | compress " << compress_time << "ms | decompress " << decompress_time << "ms" << std::endl;
    }

    return 0;
}

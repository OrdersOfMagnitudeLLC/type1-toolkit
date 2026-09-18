// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstring>
#include "nscomp.hpp"
#include <zstd.h>

int main() {
    const char* csv_path = "data/gnome.csv";
    
    // Load raw CSV bytes into buffer
    std::ifstream file(csv_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open: " << csv_path << std::endl;
        return 1;
    }
    
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<uint8_t> buffer(size);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        std::cerr << "Failed to read file" << std::endl;
        return 1;
    }
    
    std::cout << "Original size: " << size << " bytes (" << size / 1024.0 / 1024.0 << " MB)" << std::endl;
    
    // Test NSComp compression
    // Create a simple schema treating data as UINT8 array
    nscomp::Schema schema;
    schema.struct_size = 1;  // 1 byte per "record"
    schema.fields.push_back({"data", nscomp::FieldType::UINT8, 0, 1});
    
    nscomp::NSComp nscomp;
    
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<uint8_t> nscomp_compressed = nscomp.compress_records(buffer.data(), size, schema);
    auto end = std::chrono::high_resolution_clock::now();
    double nscomp_compress_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    // NSComp decompress
    std::vector<uint8_t> nscomp_decompressed(size);
    start = std::chrono::high_resolution_clock::now();
    bool nscomp_ok = nscomp.decompress_records(nscomp_compressed.data(), nscomp_decompressed.data(), size, schema);
    end = std::chrono::high_resolution_clock::now();
    double nscomp_decompress_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    if (!nscomp_ok) {
        std::cerr << "NSComp decompression failed" << std::endl;
        return 1;
    }
    
    // Verify NSComp roundtrip
    if (nscomp_decompressed != buffer) {
        std::cerr << "NSComp roundtrip verification failed" << std::endl;
        return 1;
    }
    
    // Test zstd compression (default level)
    size_t zstd_max_size = ZSTD_compressBound(size);
    std::vector<uint8_t> zstd_compressed(zstd_max_size);
    
    start = std::chrono::high_resolution_clock::now();
    size_t zstd_compressed_size = ZSTD_compress(zstd_compressed.data(), zstd_max_size, 
                                                buffer.data(), size, 3);  // default level
    end = std::chrono::high_resolution_clock::now();
    double zstd_compress_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    zstd_compressed.resize(zstd_compressed_size);
    
    // zstd decompress
    std::vector<uint8_t> zstd_decompressed(size);
    start = std::chrono::high_resolution_clock::now();
    size_t zstd_decompressed_size = ZSTD_decompress(zstd_decompressed.data(), size,
                                                     zstd_compressed.data(), zstd_compressed_size);
    end = std::chrono::high_resolution_clock::now();
    double zstd_decompress_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    if (zstd_decompressed_size != size) {
        std::cerr << "zstd decompression size mismatch" << std::endl;
        return 1;
    }
    
    // Verify zstd roundtrip
    if (zstd_decompressed != buffer) {
        std::cerr << "zstd roundtrip verification failed" << std::endl;
        return 1;
    }
    
    // Print results table
    std::cout << "\n| Method       | Size        | Ratio  | Compress ms | Decompress ms |\n";
    std::cout << "|--------------|-------------|--------|--------------|---------------|\n";
    
    double nscomp_ratio = static_cast<double>(nscomp_compressed.size()) / size;
    std::cout << "| NSComp       | " << std::setw(11) << nscomp_compressed.size() 
              << " | " << std::fixed << std::setprecision(3) << nscomp_ratio
              << " | " << std::setw(12) << nscomp_compress_ms
              << " | " << std::setw(13) << nscomp_decompress_ms << " |\n";
    
    double zstd_ratio = static_cast<double>(zstd_compressed.size()) / size;
    std::cout << "| zstd (level 3)| " << std::setw(11) << zstd_compressed.size() 
              << " | " << std::fixed << std::setprecision(3) << zstd_ratio
              << " | " << std::setw(12) << zstd_compress_ms
              << " | " << std::setw(13) << zstd_decompress_ms << " |\n";
    
    std::cout << std::defaultfloat;
    
    return 0;
}

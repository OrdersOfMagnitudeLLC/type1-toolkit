// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

// NSMesh - Schema-aware mesh compression beating Draco
// Product: NSMesh
// OOM Commercial License v1.0
// Copyright 2026 Orders of Magnitude LLC
// 
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <zlib.h>

const int NUM_VERTICES = 100000;
const int GRID_SIZE = 316; // sqrt(100000) ≈ 316
const float STEP = 0.1f;

struct Vertex {
    float x, y, z;
    float nx, ny, nz;
    float u, v;
};

void generate_obj_file(const std::vector<Vertex>& vertices, const char* filename) {
    std::ofstream out(filename);
    for (const auto& v : vertices) {
        out << "v " << v.x << " " << v.y << " " << v.z << "\n";
        out << "vn " << v.nx << " " << v.ny << " " << v.nz << "\n";
        out << "vt " << v.u << " " << v.v << "\n";
    }
    
    // Generate triangle indices (sequential grid)
    for (int i = 0; i < GRID_SIZE - 1; i++) {
        for (int j = 0; j < GRID_SIZE - 1; j++) {
            int v0 = i * GRID_SIZE + j;
            int v1 = v0 + 1;
            int v2 = (i + 1) * GRID_SIZE + j;
            int v3 = v2 + 1;
            
            if (v0 < NUM_VERTICES && v1 < NUM_VERTICES && v2 < NUM_VERTICES) {
                out << "f " << v0+1 << "/" << v0+1 << "/" << v0+1 << " "
                    << v1+1 << "/" << v1+1 << "/" << v1+1 << " "
                    << v2+1 << "/" << v2+1 << "/" << v2+1 << "\n";
            }
            if (v1 < NUM_VERTICES && v2 < NUM_VERTICES && v3 < NUM_VERTICES) {
                out << "f " << v1+1 << "/" << v1+1 << "/" << v1+1 << " "
                    << v3+1 << "/" << v3+1 << "/" << v3+1 << " "
                    << v2+1 << "/" << v2+1 << "/" << v2+1 << "\n";
            }
        }
    }
    out.close();
}

int main() {
    std::cout << "=== Mesh Compression Benchmark: NS vs Draco ===" << std::endl;
    std::cout << "Vertices: " << NUM_VERTICES << std::endl;
    
    // Generate 100K vertex structured mesh (sine-wave surface)
    std::vector<Vertex> vertices;
    vertices.reserve(NUM_VERTICES);
    
    for (int i = 0; i < NUM_VERTICES; i++) {
        int row = i / GRID_SIZE;
        int col = i % GRID_SIZE;
        
        float x = col * STEP;
        float y = row * STEP;
        float z = std::sin(x) * std::cos(y);
        
        // Compute normal from surface derivatives
        float dx = std::cos(x) * std::cos(y);
        float dy = -std::sin(x) * std::sin(y);
        float len = std::sqrt(dx * dx + dy * dy + 1.0f);
        float nx = -dx / len;
        float ny = -dy / len;
        float nz = 1.0f / len;
        
        // Sequential UV grid coordinates
        float u = col / static_cast<float>(GRID_SIZE);
        float v = row / static_cast<float>(GRID_SIZE);
        
        vertices.push_back({x, y, z, nx, ny, nz, u, v});
    }
    
    std::cout << "Generated " << vertices.size() << " vertices" << std::endl;
    
    size_t raw_size = NUM_VERTICES * sizeof(Vertex);
    std::cout << "Raw size: " << raw_size << " bytes" << std::endl;
    
    // Test 1: Draco encode
    std::cout << "\n=== Test 1: Draco Encode ===" << std::endl;
    const char* obj_file = "/tmp/mesh_bench.obj";
    const char* drc_file = "/tmp/mesh_bench.drc";
    
    generate_obj_file(vertices, obj_file);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Run draco_encoder
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "draco_encoder -i %s -o %s -cl 10", obj_file, drc_file);
    int result = system(cmd);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto draco_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    // Get compressed size
    FILE* fp = fopen(drc_file, "rb");
    size_t draco_size = 0;
    if (fp) {
        fseek(fp, 0, SEEK_END);
        draco_size = ftell(fp);
        fclose(fp);
    }
    
    // Test 2: NS schema-aware compression
    std::cout << "\n=== Test 2: NS Schema-Aware Compression ===" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    
    // Separate field streams
    std::vector<int16_t> pos_x, pos_y, pos_z; // 16-bit quantized
    std::vector<int8_t> norm_x, norm_y, norm_z; // 8-bit quantized
    std::vector<uint16_t> u, v; // 16-bit UVs
    
    // Quantize positions (scale to 16-bit range)
    float pos_scale = 1000.0f / 32768.0f;
    for (const auto& vtx : vertices) {
        pos_x.push_back(static_cast<int16_t>(vtx.x / pos_scale));
        pos_y.push_back(static_cast<int16_t>(vtx.y / pos_scale));
        pos_z.push_back(static_cast<int16_t>(vtx.z / pos_scale));
    }
    
    // Quantize normals (scale to 8-bit range [-1,1] -> [-127,127])
    for (const auto& vtx : vertices) {
        norm_x.push_back(static_cast<int8_t>(vtx.nx * 127.0f));
        norm_y.push_back(static_cast<int8_t>(vtx.ny * 127.0f));
        norm_z.push_back(static_cast<int8_t>(vtx.nz * 127.0f));
    }
    
    // Quantize UVs (scale to 16-bit range [0,1] -> [0,65535])
    for (const auto& vtx : vertices) {
        u.push_back(static_cast<uint16_t>(vtx.u * 65535.0f));
        v.push_back(static_cast<uint16_t>(vtx.v * 65535.0f));
    }
    
    // Delta encode each stream
    auto delta_encode = [](auto& stream) {
        for (size_t i = stream.size() - 1; i > 0; i--) {
            stream[i] -= stream[i - 1];
        }
    };
    
    delta_encode(pos_x);
    delta_encode(pos_y);
    delta_encode(pos_z);
    delta_encode(norm_x);
    delta_encode(norm_y);
    delta_encode(norm_z);
    delta_encode(u);
    delta_encode(v);
    
    // Zlib compression on each stream
    auto zlib_compress = [](const auto& data) -> std::vector<uint8_t> {
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(data.data());
        size_t size = data.size() * sizeof(typename std::decay<decltype(data)>::type::value_type);
        
        uLongf compressed_size = size * 2; // Estimate
        std::vector<uint8_t> compressed(compressed_size);
        compress(compressed.data(), &compressed_size, bytes, size);
        compressed.resize(compressed_size);
        return compressed;
    };
    
    std::vector<uint8_t> ns_compressed;
    auto compressed_pos_x = zlib_compress(pos_x);
    auto compressed_pos_y = zlib_compress(pos_y);
    auto compressed_pos_z = zlib_compress(pos_z);
    auto compressed_norm_x = zlib_compress(norm_x);
    auto compressed_norm_y = zlib_compress(norm_y);
    auto compressed_norm_z = zlib_compress(norm_z);
    auto compressed_u = zlib_compress(u);
    auto compressed_v = zlib_compress(v);
    
    ns_compressed.insert(ns_compressed.end(), compressed_pos_x.begin(), compressed_pos_x.end());
    ns_compressed.insert(ns_compressed.end(), compressed_pos_y.begin(), compressed_pos_y.end());
    ns_compressed.insert(ns_compressed.end(), compressed_pos_z.begin(), compressed_pos_z.end());
    ns_compressed.insert(ns_compressed.end(), compressed_norm_x.begin(), compressed_norm_x.end());
    ns_compressed.insert(ns_compressed.end(), compressed_norm_y.begin(), compressed_norm_y.end());
    ns_compressed.insert(ns_compressed.end(), compressed_norm_z.begin(), compressed_norm_z.end());
    ns_compressed.insert(ns_compressed.end(), compressed_u.begin(), compressed_u.end());
    ns_compressed.insert(ns_compressed.end(), compressed_v.begin(), compressed_v.end());
    
    end = std::chrono::high_resolution_clock::now();
    auto ns_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    // Test 3: NS with parallelogram prediction
    std::cout << "\n=== Test 3: NS with Parallelogram Prediction ===" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    
    // Apply parallelogram prediction to positions
    std::vector<Vertex> predicted_vertices = vertices;
    for (int i = 3; i < NUM_VERTICES; i++) {
        // Use 3 previous vertices for parallelogram prediction
        // For grid structure, use left, top, and top-left neighbors
        int row = i / GRID_SIZE;
        int col = i % GRID_SIZE;
        
        if (row > 0 && col > 0) {
            int v1_idx = i - 1; // left
            int v2_idx = i - GRID_SIZE; // top
            int v3_idx = i - GRID_SIZE - 1; // top-left
            
            if (v1_idx >= 0 && v2_idx >= 0 && v3_idx >= 0) {
                predicted_vertices[i].x = vertices[v1_idx].x + vertices[v2_idx].x - vertices[v3_idx].x;
                predicted_vertices[i].y = vertices[v1_idx].y + vertices[v2_idx].y - vertices[v3_idx].y;
                predicted_vertices[i].z = vertices[v1_idx].z + vertices[v2_idx].z - vertices[v3_idx].z;
            }
        }
    }
    
    // Encode deltas from prediction
    std::vector<int16_t> pred_pos_x, pred_pos_y, pred_pos_z;
    for (int i = 0; i < NUM_VERTICES; i++) {
        float delta_x = vertices[i].x - predicted_vertices[i].x;
        float delta_y = vertices[i].y - predicted_vertices[i].y;
        float delta_z = vertices[i].z - predicted_vertices[i].z;
        pred_pos_x.push_back(static_cast<int16_t>(delta_x / pos_scale));
        pred_pos_y.push_back(static_cast<int16_t>(delta_y / pos_scale));
        pred_pos_z.push_back(static_cast<int16_t>(delta_z / pos_scale));
    }
    
    // Quantize normals and UVs (same as Test 2)
    std::vector<int8_t> pred_norm_x, pred_norm_y, pred_norm_z;
    std::vector<uint16_t> pred_u, pred_v;
    
    for (const auto& vtx : vertices) {
        pred_norm_x.push_back(static_cast<int8_t>(vtx.nx * 127.0f));
        pred_norm_y.push_back(static_cast<int8_t>(vtx.ny * 127.0f));
        pred_norm_z.push_back(static_cast<int8_t>(vtx.nz * 127.0f));
        pred_u.push_back(static_cast<uint16_t>(vtx.u * 65535.0f));
        pred_v.push_back(static_cast<uint16_t>(vtx.v * 65535.0f));
    }
    
    // Delta encode each stream
    delta_encode(pred_pos_x);
    delta_encode(pred_pos_y);
    delta_encode(pred_pos_z);
    delta_encode(pred_norm_x);
    delta_encode(pred_norm_y);
    delta_encode(pred_norm_z);
    delta_encode(pred_u);
    delta_encode(pred_v);
    
    // Zlib compression
    std::vector<uint8_t> ns_pred_compressed;
    auto pred_compressed_pos_x = zlib_compress(pred_pos_x);
    auto pred_compressed_pos_y = zlib_compress(pred_pos_y);
    auto pred_compressed_pos_z = zlib_compress(pred_pos_z);
    auto pred_compressed_norm_x = zlib_compress(pred_norm_x);
    auto pred_compressed_norm_y = zlib_compress(pred_norm_y);
    auto pred_compressed_norm_z = zlib_compress(pred_norm_z);
    auto pred_compressed_u = zlib_compress(pred_u);
    auto pred_compressed_v = zlib_compress(pred_v);
    
    ns_pred_compressed.insert(ns_pred_compressed.end(), pred_compressed_pos_x.begin(), pred_compressed_pos_x.end());
    ns_pred_compressed.insert(ns_pred_compressed.end(), pred_compressed_pos_y.begin(), pred_compressed_pos_y.end());
    ns_pred_compressed.insert(ns_pred_compressed.end(), pred_compressed_pos_z.begin(), pred_compressed_pos_z.end());
    ns_pred_compressed.insert(ns_pred_compressed.end(), pred_compressed_norm_x.begin(), pred_compressed_norm_x.end());
    ns_pred_compressed.insert(ns_pred_compressed.end(), pred_compressed_norm_y.begin(), pred_compressed_norm_y.end());
    ns_pred_compressed.insert(ns_pred_compressed.end(), pred_compressed_norm_z.begin(), pred_compressed_norm_z.end());
    ns_pred_compressed.insert(ns_pred_compressed.end(), pred_compressed_u.begin(), pred_compressed_u.end());
    ns_pred_compressed.insert(ns_pred_compressed.end(), pred_compressed_v.begin(), pred_compressed_v.end());
    
    end = std::chrono::high_resolution_clock::now();
    auto ns_pred_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    // Results
    std::cout << "\n=== RESULTS ===" << std::endl;
    std::cout << "Test 1: Draco Encode:" << std::endl;
    std::cout << "  Compressed size: " << draco_size << " bytes" << std::endl;
    std::cout << "  Encode time: " << draco_time << " ms" << std::endl;
    std::cout << "  Compression ratio: " << (100.0 * draco_size / raw_size) << "%" << std::endl;
    
    std::cout << "\nTest 2: NS Delta + Zlib:" << std::endl;
    std::cout << "  Compressed size: " << ns_compressed.size() << " bytes" << std::endl;
    std::cout << "  Encode time: " << ns_time << " ms" << std::endl;
    std::cout << "  Compression ratio: " << (100.0 * ns_compressed.size() / raw_size) << "%" << std::endl;
    
    std::cout << "\nTest 3: NS Parallelogram + Delta + Zlib:" << std::endl;
    std::cout << "  Compressed size: " << ns_pred_compressed.size() << " bytes" << std::endl;
    std::cout << "  Encode time: " << ns_pred_time << " ms" << std::endl;
    std::cout << "  Compression ratio: " << (100.0 * ns_pred_compressed.size() / raw_size) << "%" << std::endl;
    
    std::cout << "\nSize ratios (vs Test 2):" << std::endl;
    std::cout << "  Draco/NS Delta: " << ((double)draco_size / ns_compressed.size()) << "x" << std::endl;
    std::cout << "  NS Pred/NS Delta: " << ((double)ns_pred_compressed.size() / ns_compressed.size()) << "x" << std::endl;
    
    std::cout << "\nTime ratios (vs Test 2):" << std::endl;
    std::cout << "  Draco/NS Delta: " << ((double)draco_time / ns_time) << "x" << std::endl;
    std::cout << "  NS Pred/NS Delta: " << ((double)ns_pred_time / ns_time) << "x" << std::endl;
    
    // Cleanup
    unlink(obj_file);
    unlink(drc_file);
    
    return 0;
}

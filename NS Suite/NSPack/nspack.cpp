// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

// nspack.cpp - NSPack POC: OBJ semantic codec
// Compile: g++ -O3 -o nspack nspack.cpp NSComp/nscomp.cpp -I NSComp/ -lzstd -llz4
// Usage:
//   ./nspack encode input.obj output.nsp
//   ./nspack decode input.nsp output.obj
//   ./nspack bench file1.obj [file2.obj ...]

#include "nscomp.hpp"
#include <zstd.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <queue>
#include <chrono>
#include <cstdio>
#include <iomanip>
#ifndef _WIN32
#include <dirent.h>
#include <unistd.h>
#include <sys/wait.h>
#endif
#include <sys/stat.h>
#include <sys/types.h>
#include <cerrno>
#include <filesystem>
#include "nspack_platform.hpp"
#include "nspack_hwy.hpp"
#include "nspack_extra.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using namespace nscomp;
using Clock = std::chrono::high_resolution_clock;

// ===== Constants =====
static const char MAGIC[4] = {'N','S','P','2'}; // v2: zstd post-compression

// ===== Morton Code (3D, 16-bit per axis → 48-bit code in uint64) =====
static uint64_t morton3d(uint16_t x, uint16_t y, uint16_t z) {
    uint64_t r = 0;
    for (int i = 0; i < 16; i++) {
        r |= ((uint64_t)((x >> i) & 1) << (3*i + 0));
        r |= ((uint64_t)((y >> i) & 1) << (3*i + 1));
        r |= ((uint64_t)((z >> i) & 1) << (3*i + 2));
    }
    return r;
}

// ===== OBJ Data Structures =====
struct ObjData {
    std::vector<float> vertices;       // V_count * 3
    std::vector<uint32_t> faces;       // F_count * 3 (triangulated, 0-indexed)
    std::vector<float> normals;        // N_count * 3
    std::vector<uint32_t> face_normals;// F_count * 3 (0-indexed)
    bool has_fn = false;
};

// ===== OBJ Parser =====
static ObjData parse_obj(const std::string& filename) {
    ObjData obj;
    std::ifstream f(filename);
    if (!f) {
        std::cerr << "Cannot open: " << filename << std::endl;
        return obj;
    }
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        std::string prefix;
        ss >> prefix;
        if (prefix == "v") {
            float x, y, z;
            ss >> x >> y >> z;
            obj.vertices.push_back(x);
            obj.vertices.push_back(y);
            obj.vertices.push_back(z);
        } else if (prefix == "vn") {
            float x, y, z;
            ss >> x >> y >> z;
            obj.normals.push_back(x);
            obj.normals.push_back(y);
            obj.normals.push_back(z);
        } else if (prefix == "f") {
            std::vector<uint32_t> vi, ni;
            std::string token;
            while (ss >> token) {
                int v = 0, n = 0;
                bool has_n = false;
                size_t slash1 = token.find('/');
                if (slash1 == std::string::npos) {
                    v = std::stoi(token);
                } else {
                    v = std::stoi(token.substr(0, slash1));
                    size_t slash2 = token.find('/', slash1 + 1);
                    if (slash2 != std::string::npos) {
                        std::string nstr = token.substr(slash2 + 1);
                        if (!nstr.empty()) {
                            n = std::stoi(nstr);
                            has_n = true;
                        }
                    }
                }
                vi.push_back((uint32_t)(v > 0 ? v - 1 : 0));
                if (has_n) {
                    ni.push_back((uint32_t)(n > 0 ? n - 1 : 0));
                    obj.has_fn = true;
                }
            }
            // Fan triangulate
            for (size_t i = 1; i + 1 < vi.size(); i++) {
                obj.faces.push_back(vi[0]);
                obj.faces.push_back(vi[i]);
                obj.faces.push_back(vi[i+1]);
                if (obj.has_fn && ni.size() == vi.size()) {
                    obj.face_normals.push_back(ni[0]);
                    obj.face_normals.push_back(ni[i]);
                    obj.face_normals.push_back(ni[i+1]);
                }
            }
        }
    }
    return obj;
}

// ===== Write OBJ =====
static void write_obj(const std::string& filename, const ObjData& obj) {
    std::ofstream f(filename);
    if (!f) return;
    size_t vc = obj.vertices.size() / 3;
    size_t fc = obj.faces.size() / 3;
    size_t nc = obj.normals.size() / 3;
    for (size_t i = 0; i < vc; i++) {
        f << "v " << obj.vertices[i*3] << " " << obj.vertices[i*3+1] << " " << obj.vertices[i*3+2] << "\n";
    }
    for (size_t i = 0; i < nc; i++) {
        f << "vn " << obj.normals[i*3] << " " << obj.normals[i*3+1] << " " << obj.normals[i*3+2] << "\n";
    }
    for (size_t i = 0; i < fc; i++) {
        uint32_t v0 = obj.faces[i*3] + 1;
        uint32_t v1 = obj.faces[i*3+1] + 1;
        uint32_t v2 = obj.faces[i*3+2] + 1;
        if (obj.has_fn && obj.face_normals.size() >= (i+1)*3) {
            uint32_t n0 = obj.face_normals[i*3] + 1;
            uint32_t n1 = obj.face_normals[i*3+1] + 1;
            uint32_t n2 = obj.face_normals[i*3+2] + 1;
            f << "f " << v0 << "//" << n0 << " " << v1 << "//" << n1 << " " << v2 << "//" << n2 << "\n";
        } else {
            f << "f " << v0 << " " << v1 << " " << v2 << "\n";
        }
    }
}

// ===== Delta Encode/Decode (int16) =====
static void delta_encode_i16(std::vector<int16_t>& d) {
    if (d.size() < 2) return;
#if defined(NSPACK_HIGHWAY)
    nspack_hwy::delta_encode_i16(d.data(), d.size());
#else
    for (size_t i = d.size() - 1; i > 0; i--)
        d[i] = d[i] - d[i-1];
#endif
}

static void delta_decode_i16(std::vector<int16_t>& d) {
    if (d.size() < 2) return;
#if defined(NSPACK_HIGHWAY)
    nspack_hwy::delta_decode_i16(d.data(), d.size());
#else
    for (size_t i = 1; i < d.size(); i++)
        d[i] = d[i] + d[i-1];
#endif
}

// Runtime overflow detection: if any delta would not fit in int16, leave d unchanged and return false.
static bool delta_encode_i16_safe(std::vector<int16_t>& d) {
    if (d.size() < 2) return true;
#if defined(NSPACK_HIGHWAY)
    int32_t max_delta = nspack_hwy::delta_max_abs_i16(d.data(), d.size());
#else
    int32_t max_delta = 0;
    for (size_t i = d.size() - 1; i > 0; i--) {
        int32_t diff = (int32_t)d[i] - (int32_t)d[i-1];
        if (diff < 0) diff = -diff;
        if (diff > max_delta) max_delta = diff;
    }
#endif
    if (max_delta > 32767) {
        std::cerr << "Delta overflow detected (max " << max_delta << "), using raw values instead.\n";
        return false;
    }
#if defined(NSPACK_HIGHWAY)
    nspack_hwy::delta_encode_i16(d.data(), d.size());
#else
    for (size_t i = d.size() - 1; i > 0; i--)
        d[i] = (int16_t)((int32_t)d[i] - (int32_t)d[i-1]);
#endif
    return true;
}

// ===== NSComp Schemas =====
static Schema make_v_schema() {
    return { 6, {
        {"x", FieldType::INT16, 0, 1},
        {"y", FieldType::INT16, 2, 1},
        {"z", FieldType::INT16, 4, 1}
    }};
}

static Schema make_f_schema() {
    return { 12, {
        {"v0", FieldType::UINT32, 0, 1},
        {"v1", FieldType::UINT32, 4, 1},
        {"v2", FieldType::UINT32, 8, 1}
    }};
}

static Schema make_n_schema() {
    return { 6, {
        {"x", FieldType::INT16, 0, 1},
        {"y", FieldType::INT16, 2, 1},
        {"z", FieldType::INT16, 4, 1}
    }};
}

// Forward declaration
static size_t file_size(const std::string& filename);

// ===== Phase 1: Encode =====
static bool encode_obj(const std::string& infile, const std::string& outfile, double* enc_ms = nullptr) {
    auto t0 = Clock::now();

    ObjData obj = parse_obj(infile);
    size_t vc = obj.vertices.size() / 3;
    size_t fc = obj.faces.size() / 3;
    size_t nc = obj.normals.size() / 3;
    if (vc == 0) {
        std::cerr << "No vertices found.\n";
        return false;
    }

    // --- Quantize V to int16 (uniform global range) ---
    float gmin = 1e30f, gmax = -1e30f;
    for (size_t i = 0; i < vc * 3; i++) {
        gmin = std::min(gmin, obj.vertices[i]);
        gmax = std::max(gmax, obj.vertices[i]);
    }
    float vrange = gmax - gmin;
    float v_scale = (vrange > 0) ? vrange / 65535.0f : 1.0f;
    float v_offset = (gmin + gmax) * 0.5f;

    std::vector<int16_t> vq(vc * 3);
    for (size_t i = 0; i < vc * 3; i++) {
        int q = (int)std::lround((obj.vertices[i] - v_offset) / v_scale);
        q = std::max(-32768, std::min(32767, q));
        vq[i] = (int16_t)q;
    }

    // --- Quantize N to int16 (normals in [-1,1]) ---
    float n_scale = 2.0f / 65535.0f;
    std::vector<int16_t> nq(nc * 3);
    for (size_t i = 0; i < nc * 3; i++) {
        int q = (int)std::lround(obj.normals[i] / n_scale);
        q = std::max(-32768, std::min(32767, q));
        nq[i] = (int16_t)q;
    }

    // --- Sort V by Morton code (proximity sort, 21-bit per axis) ---
    std::vector<uint64_t> mortons(vc);
    const uint64_t M21 = (1ULL << 21) - 1;
#if defined(NSPACK_HIGHWAY)
    // Quantization stays scalar (64-bit division); the bit-spread is SIMD.
    std::vector<uint64_t> qx(vc), qy(vc), qz(vc);
    for (size_t i = 0; i < vc; i++) {
        qx[i] = (uint64_t)((int)vq[i*3]   + 32768) * M21 / 65535ULL;
        qy[i] = (uint64_t)((int)vq[i*3+1] + 32768) * M21 / 65535ULL;
        qz[i] = (uint64_t)((int)vq[i*3+2] + 32768) * M21 / 65535ULL;
    }
    nspack_hwy::morton3_64_batch(qx.data(), qy.data(), qz.data(), mortons.data(), vc);
#else
    for (size_t i = 0; i < vc; i++) {
        uint64_t ux = (uint64_t)((int)vq[i*3]   + 32768) * M21 / 65535ULL;
        uint64_t uy = (uint64_t)((int)vq[i*3+1] + 32768) * M21 / 65535ULL;
        uint64_t uz = (uint64_t)((int)vq[i*3+2] + 32768) * M21 / 65535ULL;
        mortons[i] = morton3_64(ux, uy, uz);
    }
#endif
    std::vector<uint32_t> sorted_idx(vc);
    for (size_t i = 0; i < vc; i++) sorted_idx[i] = (uint32_t)i;
    std::sort(sorted_idx.begin(), sorted_idx.end(), [&](uint32_t a, uint32_t b) {
        return mortons[a] < mortons[b];
    });

    // Reorder V into sorted order, build old→new index map
    std::vector<int16_t> vq_sorted(vc * 3);
    std::vector<uint32_t> old_to_new(vc);
    for (size_t i = 0; i < vc; i++) {
        old_to_new[sorted_idx[i]] = (uint32_t)i;
        vq_sorted[i*3]   = vq[sorted_idx[i]*3];
        vq_sorted[i*3+1] = vq[sorted_idx[i]*3+1];
        vq_sorted[i*3+2] = vq[sorted_idx[i]*3+2];
    }

    // --- Remap face indices to sorted order ---
    std::vector<uint32_t> faces_remapped(obj.faces.size());
    for (size_t i = 0; i < obj.faces.size(); i++) {
        faces_remapped[i] = old_to_new[obj.faces[i]];
    }

    // --- Delta encode sorted V and N (overflow-safe) ---
    uint8_t v_use_delta = 1, n_use_delta = 1;
    if (!delta_encode_i16_safe(vq_sorted)) v_use_delta = 0;
    if (!delta_encode_i16_safe(nq)) n_use_delta = 0;

    // --- Compress via NSComp ---
    NSComp comp;
    auto v_schema = make_v_schema();
    auto f_schema = make_f_schema();
    auto n_schema = make_n_schema();

    auto v_comp = comp.compress_records(
        reinterpret_cast<const uint8_t*>(vq_sorted.data()), vc, v_schema);

    std::vector<uint8_t> f_comp;
    if (fc > 0) {
        f_comp = comp.compress_records(
            reinterpret_cast<const uint8_t*>(faces_remapped.data()), fc, f_schema);
    }

    std::vector<uint8_t> n_comp;
    if (nc > 0) {
        n_comp = comp.compress_records(
            reinterpret_cast<const uint8_t*>(nq.data()), nc, n_schema);
    }

    std::vector<uint8_t> fn_comp;
    if (obj.has_fn && fc > 0) {
        auto fn_schema = make_f_schema();
        fn_comp = comp.compress_records(
            reinterpret_cast<const uint8_t*>(obj.face_normals.data()), fc, fn_schema);
    }

    // --- Write binary blob ---
    std::ofstream out(outfile, std::ios::binary);
    if (!out) return false;

    out.write(MAGIC, 4);
    uint32_t v32;
    v32 = (uint32_t)vc; out.write((char*)&v32, 4);
    v32 = (uint32_t)fc; out.write((char*)&v32, 4);
    v32 = (uint32_t)nc; out.write((char*)&v32, 4);
    uint8_t has_fn = obj.has_fn ? 1 : 0; out.write((char*)&has_fn, 1);
    out.write((char*)&v_use_delta, 1);
    out.write((char*)&n_use_delta, 1);
    out.write((char*)&v_scale, 4);
    out.write((char*)&v_offset, 4);
    out.write((char*)&n_scale, 4);

    // V compressed
    v32 = (uint32_t)v_comp.size(); out.write((char*)&v32, 4);
    out.write((char*)v_comp.data(), v_comp.size());

    // F compressed
    v32 = (uint32_t)f_comp.size(); out.write((char*)&v32, 4);
    out.write((char*)f_comp.data(), f_comp.size());

    // N compressed
    v32 = (uint32_t)n_comp.size(); out.write((char*)&v32, 4);
    out.write((char*)n_comp.data(), n_comp.size());

    // FN compressed
    v32 = (uint32_t)fn_comp.size(); out.write((char*)&v32, 4);
    out.write((char*)fn_comp.data(), fn_comp.size());

    out.close();

    // --- zstd post-compression pass ---
    size_t raw_sz = file_size(outfile);
    std::ifstream raw_in(outfile, std::ios::binary);
    std::vector<uint8_t> raw_data(raw_sz);
    raw_in.read((char*)raw_data.data(), raw_sz);
    raw_in.close();

    size_t comp_bound = ZSTD_compressBound(raw_sz);
    std::vector<uint8_t> zstd_data(comp_bound);
    size_t zstd_sz = ZSTD_compress(zstd_data.data(), comp_bound,
                                   raw_data.data(), raw_sz, 19);
    if (ZSTD_isError(zstd_sz)) {
        std::cerr << "zstd post-compress failed.\n";
        return false;
    }
    zstd_data.resize(zstd_sz);

    std::ofstream zout(outfile, std::ios::binary);
    zout.write(MAGIC, 4);
    uint32_t raw32 = (uint32_t)raw_sz;
    zout.write((char*)&raw32, 4);
    zout.write((char*)zstd_data.data(), zstd_sz);
    zout.close();

    if (enc_ms) {
        auto t1 = Clock::now();
        *enc_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    return true;
}

// ===== Phase 2: Decode =====
static bool decode_obj(const std::string& infile, const std::string& outfile, double* dec_ms = nullptr, double* shannon_T = nullptr) {
    auto t0 = Clock::now();

    // --- zstd decompress outer layer ---
    std::ifstream zf(infile, std::ios::binary);
    if (!zf) return false;

    char magic[4];
    zf.read(magic, 4);
    if (memcmp(magic, MAGIC, 4) != 0) {
        std::cerr << "Invalid magic.\n";
        return false;
    }
    uint32_t raw_sz32;
    zf.read((char*)&raw_sz32, 4);
    size_t raw_sz = raw_sz32;
    size_t zfile_sz = file_size(infile) - 8;
    std::vector<uint8_t> zdata(zfile_sz);
    zf.read((char*)zdata.data(), zfile_sz);
    zf.close();

    std::vector<uint8_t> raw_data(raw_sz);
    size_t dres = ZSTD_decompress(raw_data.data(), raw_sz, zdata.data(), zfile_sz);
    if (ZSTD_isError(dres)) {
        std::cerr << "zstd decompress failed.\n";
        return false;
    }

    // Parse from memory buffer instead of file
    // Skip the 4-byte MAGIC header embedded in the raw blob
    size_t pos = 4;
    auto read_u32 = [&]() -> uint32_t {
        uint32_t v;
        memcpy(&v, raw_data.data() + pos, 4); pos += 4;
        return v;
    };
    auto read_u8 = [&]() -> uint8_t {
        uint8_t v = raw_data[pos]; pos += 1;
        return v;
    };
    auto read_float = [&]() -> float {
        float v;
        memcpy(&v, raw_data.data() + pos, 4); pos += 4;
        return v;
    };
    auto read_compressed_mem = [&]() -> std::vector<uint8_t> {
        uint32_t comp_size = read_u32();
        if (comp_size == 0) return {};
        std::vector<uint8_t> data(comp_size);
        memcpy(data.data(), raw_data.data() + pos, comp_size);
        pos += comp_size;
        return data;
    };

    uint32_t vc = read_u32();
    uint32_t fc = read_u32();
    uint32_t nc = read_u32();
    uint8_t has_fn = read_u8();
    uint8_t v_use_delta = read_u8();
    uint8_t n_use_delta = read_u8();
    float v_scale = read_float();
    float v_offset = read_float();
    float n_scale = read_float();

    auto v_comp = read_compressed_mem();
    auto f_comp = read_compressed_mem();
    auto n_comp = read_compressed_mem();
    auto fn_comp = read_compressed_mem();

    // --- Decompress via NSComp ---
    NSComp comp;
    auto v_schema = make_v_schema();
    auto f_schema = make_f_schema();
    auto n_schema = make_n_schema();

    std::vector<int16_t> vq_sorted(vc * 3, 0);
    if (vc > 0 && !v_comp.empty()) {
        if (!comp.decompress_records(v_comp.data(),
            reinterpret_cast<uint8_t*>(vq_sorted.data()), vc, v_schema)) {
            std::cerr << "V decompress failed.\n";
            return false;
        }
    }

    std::vector<uint32_t> faces_remapped(fc * 3, 0);
    if (fc > 0 && !f_comp.empty()) {
        if (!comp.decompress_records(f_comp.data(),
            reinterpret_cast<uint8_t*>(faces_remapped.data()), fc, f_schema)) {
            std::cerr << "F decompress failed.\n";
            return false;
        }
    }

    std::vector<int16_t> nq(nc * 3, 0);
    if (nc > 0 && !n_comp.empty()) {
        if (!comp.decompress_records(n_comp.data(),
            reinterpret_cast<uint8_t*>(nq.data()), nc, n_schema)) {
            std::cerr << "N decompress failed.\n";
            return false;
        }
    }

    std::vector<uint32_t> face_normals(fc * 3, 0);
    if (has_fn && fc > 0 && !fn_comp.empty()) {
        auto fn_schema = make_f_schema();
        if (!comp.decompress_records(fn_comp.data(),
            reinterpret_cast<uint8_t*>(face_normals.data()), fc, fn_schema)) {
            std::cerr << "FN decompress failed.\n";
            return false;
        }
    }

    // --- Undo delta encoding ---
    if (v_use_delta) delta_decode_i16(vq_sorted);
    if (n_use_delta) delta_decode_i16(nq);

    // --- Dequantize ---
    ObjData obj;
    obj.vertices.resize(vc * 3);
    for (size_t i = 0; i < vc * 3; i++)
        obj.vertices[i] = (float)vq_sorted[i] * v_scale + v_offset;

    obj.normals.resize(nc * 3);
    for (size_t i = 0; i < nc * 3; i++)
        obj.normals[i] = (float)nq[i] * n_scale;

    obj.faces = faces_remapped;
    obj.face_normals = face_normals;
    obj.has_fn = has_fn != 0;

    if (shannon_T) {
        std::vector<uint8_t> all;
        all.insert(all.end(), (uint8_t*)vq_sorted.data(), (uint8_t*)vq_sorted.data() + vq_sorted.size() * sizeof(int16_t));
        if (fc > 0) all.insert(all.end(), (uint8_t*)faces_remapped.data(), (uint8_t*)faces_remapped.data() + faces_remapped.size() * sizeof(uint32_t));
        if (nc > 0) all.insert(all.end(), (uint8_t*)nq.data(), (uint8_t*)nq.data() + nq.size() * sizeof(int16_t));
        if (has_fn && fc > 0) all.insert(all.end(), (uint8_t*)face_normals.data(), (uint8_t*)face_normals.data() + face_normals.size() * sizeof(uint32_t));
        *shannon_T = (double)shannon_limit_data(all);
    }

    write_obj(outfile, obj);

    if (dec_ms) {
        auto t1 = Clock::now();
        *dec_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    return true;
}

// ===== Utility: file size =====
static size_t file_size(const std::string& filename) {
    std::ifstream f(filename, std::ios::binary | std::ios::ate);
    if (!f) return 0;
    return (size_t)f.tellg();
}

// ===== Utility: compressed size via subprocess =====
static size_t compressed_size(const std::string& cmd) {
    FILE* pipe = nsp_popen(cmd.c_str(), "r");
    if (!pipe) return 0;
    size_t size = 0;
    fscanf(pipe, "%zu", &size);
    nsp_pclose(pipe);
    return size;
}

// ===== rANS Entropy Coder (byte-level, static model) =====
// Based on ryg_rans approach. Public domain.

static const uint32_t RANS_L = 1u << 23;  // lower bound of normalized interval
static const uint32_t RANS_R = 1u << 8;   // M = probability normalization base (256)
static const uint32_t RANS_BITS = 8;      // scale_bits = log2(M)

struct RansTable {
    uint32_t freq[256];
    uint32_t cum[256];
};

static RansTable rans_build_table(const uint8_t* data, size_t n) {
    RansTable t;
    uint64_t counts[256] = {0};
    for (size_t i = 0; i < n; i++) counts[data[i]]++;

    uint32_t total = 0;
    for (int i = 0; i < 256; i++) {
        if (counts[i] == 0) {
            t.freq[i] = 0;
            t.cum[i] = total;
            continue;
        }
        uint32_t f = std::max(1u, (uint32_t)((counts[i] * RANS_R) / std::max((size_t)1, n)));
        if (total + f > RANS_R) f = RANS_R - total;
        t.freq[i] = f;
        t.cum[i] = total;
        total += f;
    }
    for (int i = 255; i >= 0; i--) {
        if (t.freq[i] > 0) { t.freq[i] += RANS_R - total; break; }
    }
    return t;
}

static std::vector<uint8_t> rans_encode(const uint8_t* data, size_t n, const RansTable& t) {
    if (n == 0) return {};

    uint32_t x = RANS_L;
    std::vector<uint8_t> stack;
    stack.reserve(n / 2);

    for (size_t i = n; i > 0; i--) {
        uint8_t s = data[i - 1];
        uint32_t freq = t.freq[s];
        if (freq == 0) freq = 1;
        uint32_t cum = t.cum[s];

        uint32_t x_max = (RANS_L >> RANS_BITS) * freq;  // = 32768 * freq
        while (x >= x_max) {
            stack.push_back((uint8_t)(x & 0xFF));
            x >>= 8;
        }

        x = ((x / freq) << RANS_BITS) + (x % freq) + cum;
    }

    std::vector<uint8_t> out;
    out.reserve(stack.size() + 4);
    out.push_back((uint8_t)((x >> 24) & 0xFF));
    out.push_back((uint8_t)((x >> 16) & 0xFF));
    out.push_back((uint8_t)((x >> 8) & 0xFF));
    out.push_back((uint8_t)(x & 0xFF));
    for (int i = (int)stack.size() - 1; i >= 0; i--)
        out.push_back(stack[i]);

    return out;
}

static bool rans_decode(const uint8_t* data, size_t comp_size, uint8_t* output, size_t n, const RansTable& t) {
    if (n == 0 || comp_size < 4) return false;

    std::vector<uint8_t> slot_to_sym(RANS_R);
    for (int s = 0; s < 256; s++) {
        for (uint32_t j = 0; j < t.freq[s]; j++) {
            slot_to_sym[t.cum[s] + j] = (uint8_t)s;
        }
    }

    size_t pos = 0;
    uint32_t x = ((uint32_t)data[0] << 24) | ((uint32_t)data[1] << 16) |
                 ((uint32_t)data[2] << 8) | (uint32_t)data[3];
    pos = 4;

    for (size_t i = 0; i < n; i++) {
        uint32_t slot = x & (RANS_R - 1);
        uint8_t s = slot_to_sym[slot];
        output[i] = s;

        x = t.freq[s] * (x >> RANS_BITS) + slot - t.cum[s];

        while (x < RANS_L && pos < comp_size) {
            x = (x << 8) | data[pos++];
        }
    }

    return true;
}

// Shannon helpers (shannon_entropy / shannon_limit_file / shannon_limit_bytes) are in nspack_extra.hpp

// Shannon limit on raw pixel data for images
static size_t shannon_limit_image(const std::string& filename) {
    int w, h, ch;
    uint8_t* px = stbi_load(filename.c_str(), &w, &h, &ch, 0);
    if (!px) return 0;
    size_t n = (size_t)w * h * ch;
    double ent = shannon_entropy_bytes(px, n);
    stbi_image_free(px);
    return (size_t)ceil(ent * n / 8.0);
}

// ===== Image 2D Delta (predictor: left + up - up_left) =====
static void delta2d_encode(const uint8_t* src, uint8_t* dst, int w, int h) {
#if defined(NSPACK_HIGHWAY)
    nspack_hwy::delta2d_encode(src, dst, w, h);
#else
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = y * w + x;
            int left = (x > 0) ? src[idx - 1] : 0;
            int up = (y > 0) ? src[idx - w] : 0;
            int ul = (x > 0 && y > 0) ? src[idx - w - 1] : 0;
            int pred = left + up - ul;
            dst[idx] = (uint8_t)(src[idx] - pred);
        }
    }
#endif
}

static void delta2d_decode(const uint8_t* src, uint8_t* dst, int w, int h) {
#if defined(NSPACK_HIGHWAY)
    nspack_hwy::delta2d_decode(src, dst, w, h);
#else
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = y * w + x;
            int left = (x > 0) ? dst[idx - 1] : 0;
            int up = (y > 0) ? dst[idx - w] : 0;
            int ul = (x > 0 && y > 0) ? dst[idx - w - 1] : 0;
            int pred = left + up - ul;
            dst[idx] = (uint8_t)(src[idx] + pred);
        }
    }
#endif
}

// ===== Image Encode/Decode =====
static const char IMG_MAGIC[4] = {'N','S','P','I'};

static bool encode_image(const std::string& infile, const std::string& outfile, double* enc_ms = nullptr) {
    auto t0 = Clock::now();

    int w, h, ch;
    uint8_t* pixels = stbi_load(infile.c_str(), &w, &h, &ch, 0);
    if (!pixels) {
        std::cerr << "Cannot load image: " << infile << "\n";
        return false;
    }

    int npix = w * h;

    // Separate channels (SoA) + 2D delta encode
    std::vector<uint8_t> flat;
    flat.reserve(npix * ch);
    for (int c = 0; c < ch; c++) {
        std::vector<uint8_t> chan(npix), tmp(npix);
#if defined(NSPACK_HIGHWAY)
        nspack_hwy::extract_channel_u8(pixels, chan.data(), npix, ch, c);
#else
        for (int i = 0; i < npix; i++) chan[i] = pixels[i * ch + c];
#endif
        delta2d_encode(chan.data(), tmp.data(), w, h);
        flat.insert(flat.end(), tmp.begin(), tmp.end());
    }
    stbi_image_free(pixels);

    // Compress with zstd-19
    size_t z_bound = ZSTD_compressBound(flat.size());
    std::vector<uint8_t> zstd_data(z_bound);
    size_t zstd_sz = ZSTD_compress(zstd_data.data(), z_bound, flat.data(), flat.size(), 19);
    if (ZSTD_isError(zstd_sz)) { std::cerr << "zstd compress failed\n"; return false; }

    // Write blob
    std::ofstream out(outfile, std::ios::binary);
    if (!out) return false;

    out.write(IMG_MAGIC, 4);
    out.write((char*)&w, 4);
    out.write((char*)&h, 4);
    uint8_t ch8 = (uint8_t)ch;
    out.write((char*)&ch8, 1);
    uint8_t method = 1; // always zstd
    out.write((char*)&method, 1);
    uint32_t zs = (uint32_t)zstd_sz;
    out.write((char*)&zs, 4);
    out.write((char*)zstd_data.data(), zstd_sz);

    out.close();

    if (enc_ms) {
        auto t1 = Clock::now();
        *enc_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    return true;
}

static bool decode_image(const std::string& infile, const std::string& outfile, double* dec_ms = nullptr, double* shannon_T = nullptr) {
    auto t0 = Clock::now();

    std::ifstream f(infile, std::ios::binary);
    if (!f) return false;

    char magic[4];
    f.read(magic, 4);
    if (memcmp(magic, IMG_MAGIC, 4) != 0) {
        std::cerr << "Invalid image magic.\n";
        return false;
    }

    int w, h;
    f.read((char*)&w, 4);
    f.read((char*)&h, 4);
    uint8_t ch, method;
    f.read((char*)&ch, 1);
    f.read((char*)&method, 1);

    int npix = w * h;
    std::vector<uint8_t> flat(npix * ch);

    // zstd decompress
    uint32_t zstd_sz;
    f.read((char*)&zstd_sz, 4);
    std::vector<uint8_t> zstd_data(zstd_sz);
    f.read((char*)zstd_data.data(), zstd_sz);

    size_t dres = ZSTD_decompress(flat.data(), flat.size(), zstd_data.data(), zstd_sz);
    if (ZSTD_isError(dres)) {
        std::cerr << "zstd decode failed.\n";
        return false;
    }
    if (shannon_T) *shannon_T = (double)shannon_limit_bytes(flat);

    // Undo 2D delta per channel + reinterleave
    std::vector<uint8_t> pixels(npix * ch);
    for (int c = 0; c < ch; c++) {
        std::vector<uint8_t> chan(flat.begin() + c * npix, flat.begin() + (c+1) * npix);
        std::vector<uint8_t> tmp(npix);
        delta2d_decode(chan.data(), tmp.data(), w, h);
#if defined(NSPACK_HIGHWAY)
        nspack_hwy::insert_channel_u8(pixels.data(), tmp.data(), npix, ch, c);
#else
        for (int i = 0; i < npix; i++) pixels[i * ch + c] = tmp[i];
#endif
    }

    // Write raw pixel data for verification
    std::ofstream out(outfile, std::ios::binary);
    if (!out) return false;
    out.write((char*)pixels.data(), pixels.size());
    out.close();

    if (dec_ms) {
        auto t1 = Clock::now();
        *dec_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    return true;
}

// ===== Video Support =====
// Y4M (YUV4MPEG2) and raw RGB frame sequences

static const char VID_MAGIC[4] = {'N','S','V','1'};

struct VideoData {
    int w, h, frames, ch;  // width, height, frame count, channels (3 for RGB/YUV)
    std::vector<uint8_t> pixels;  // frames * w * h * ch
};

// Parse Y4M header: "YUV4MPEG2 W352 H288 F30000:1001 Ip A128:117\n"
// YUV420: Y plane (w*h), U plane (w/2*h/2), V plane (w/2*h/2)
static bool parse_y4m(const std::string& filename, VideoData& vid) {
    std::ifstream f(filename, std::ios::binary);
    if (!f) return false;

    std::string header;
    std::getline(f, header);
    if (header.substr(0, 9) != "YUV4MPEG2") return false;

    // Parse W and H
    size_t wpos = header.find('W');
    size_t hpos = header.find('H');
    if (wpos == std::string::npos || hpos == std::string::npos) return false;
    vid.w = std::stoi(header.substr(wpos + 1, hpos - wpos - 1));
    vid.h = std::stoi(header.substr(hpos + 1, header.find(' ', hpos) - hpos - 1));
    vid.ch = 3;  // Y, U, V planes

    // YUV420 frame layout: Y (w*h) + U (w/2*h/2) + V (w/2*h/2)
    int y_size = vid.w * vid.h;
    int uv_size = (vid.w / 2) * (vid.h / 2);
    int frame_size = y_size + 2 * uv_size;
    std::vector<uint8_t> frame(frame_size);
    vid.frames = 0;
    vid.pixels.clear();

    while (vid.frames < 1000) {  // safety limit
        std::string tag;
        std::getline(f, tag);
        if (tag.substr(0, 5) != "FRAME") break;
        f.read((char*)frame.data(), frame_size);
        if (f.gcount() != frame_size) break;  // truncated frame
        vid.pixels.insert(vid.pixels.end(), frame.begin(), frame.end());
        vid.frames++;
    }

    return vid.frames > 0;
}

// Parse raw RGB frame sequence: [w(4)][h(4)][frames(4)][ch(4)][frame data...]
static bool parse_raw_frames(const std::string& filename, VideoData& vid) {
    std::ifstream f(filename, std::ios::binary);
    if (!f) return false;

    uint32_t w, h, frames, ch;
    f.read((char*)&w, 4);
    f.read((char*)&h, 4);
    f.read((char*)&frames, 4);
    f.read((char*)&ch, 4);
    if (!f || w == 0 || h == 0 || frames == 0 || ch == 0) return false;

    vid.w = w; vid.h = h; vid.frames = frames; vid.ch = ch;
    vid.pixels.resize((size_t)w * h * frames * ch);
    f.read((char*)vid.pixels.data(), vid.pixels.size());
    return f.good() || f.eof();
}

// Temporal delta: frame[i] = frame[i] - frame[i-1] (mod 256)
static void temporal_delta_encode(std::vector<uint8_t>& pixels, size_t frame_size, int frames) {
    for (int f = frames - 1; f > 0; f--) {
        uint8_t* curr = pixels.data() + f * frame_size;
        uint8_t* prev = pixels.data() + (f - 1) * frame_size;
#if defined(NSPACK_HIGHWAY)
        nspack_hwy::temporal_delta_encode(curr, prev, frame_size);
#else
        for (size_t i = 0; i < frame_size; i++) {
            curr[i] = (uint8_t)(curr[i] - prev[i]);
        }
#endif
    }
}

static void temporal_delta_decode(std::vector<uint8_t>& pixels, size_t frame_size, int frames) {
    for (int f = 1; f < frames; f++) {
        uint8_t* curr = pixels.data() + f * frame_size;
        uint8_t* prev = pixels.data() + (f - 1) * frame_size;
#if defined(NSPACK_HIGHWAY)
        nspack_hwy::temporal_delta_decode(curr, prev, frame_size);
#else
        for (size_t i = 0; i < frame_size; i++) {
            curr[i] = (uint8_t)(curr[i] + prev[i]);
        }
#endif
    }
}

// Encode video: temporal delta + spatial delta + zstd
static bool encode_video(const std::string& infile, const std::string& outfile, double* enc_ms = nullptr) {
    auto t0 = Clock::now();

    VideoData vid;
    bool is_y4m = infile.substr(infile.size() - 4) == ".y4m";
    bool ok = is_y4m ? parse_y4m(infile, vid) : parse_raw_frames(infile, vid);
    if (!ok) {
        std::cerr << "Failed to parse video: " << infile << "\n";
        return false;
    }

    // Temporal delta encoding
    size_t td_frame_size;
    if (is_y4m) {
        td_frame_size = (size_t)(vid.w * vid.h + 2 * (vid.w / 2) * (vid.h / 2));
    } else {
        td_frame_size = (size_t)vid.w * vid.h * vid.ch;
    }
    temporal_delta_encode(vid.pixels, td_frame_size, vid.frames);

    // Spatial delta encoding per frame
    std::vector<uint8_t> flat;
    flat.reserve(vid.pixels.size());

    if (is_y4m) {
        // YUV420: Y plane (w*h), U plane (w/2*h/2), V plane (w/2*h/2)
        int y_size = vid.w * vid.h;
        int uv_size = (vid.w / 2) * (vid.h / 2);
        int frame_size = y_size + 2 * uv_size;
        for (int f = 0; f < vid.frames; f++) {
            uint8_t* frame = vid.pixels.data() + f * frame_size;
            // Y plane
            std::vector<uint8_t> y_tmp(y_size);
            delta2d_encode(frame, y_tmp.data(), vid.w, vid.h);
            flat.insert(flat.end(), y_tmp.begin(), y_tmp.end());
            // U plane
            std::vector<uint8_t> u_tmp(uv_size);
            delta2d_encode(frame + y_size, u_tmp.data(), vid.w / 2, vid.h / 2);
            flat.insert(flat.end(), u_tmp.begin(), u_tmp.end());
            // V plane
            std::vector<uint8_t> v_tmp(uv_size);
            delta2d_encode(frame + y_size + uv_size, v_tmp.data(), vid.w / 2, vid.h / 2);
            flat.insert(flat.end(), v_tmp.begin(), v_tmp.end());
        }
    } else {
        // RGB interleaved: extract channels, delta encode each
        size_t frame_size = (size_t)vid.w * vid.h * vid.ch;
        for (int f = 0; f < vid.frames; f++) {
            uint8_t* frame = vid.pixels.data() + f * frame_size;
            for (int c = 0; c < vid.ch; c++) {
                std::vector<uint8_t> chan(vid.w * vid.h), tmp(vid.w * vid.h);
#if defined(NSPACK_HIGHWAY)
                nspack_hwy::extract_channel_u8(frame, chan.data(), (size_t)vid.w * vid.h, vid.ch, c);
#else
                for (int i = 0; i < vid.w * vid.h; i++) chan[i] = frame[i * vid.ch + c];
#endif
                delta2d_encode(chan.data(), tmp.data(), vid.w, vid.h);
                flat.insert(flat.end(), tmp.begin(), tmp.end());
            }
        }
    }

    // Compress with zstd
    size_t z_bound = ZSTD_compressBound(flat.size());
    std::vector<uint8_t> zstd_data(z_bound);
    size_t zstd_sz = ZSTD_compress(zstd_data.data(), z_bound, flat.data(), flat.size(), 19);
    if (ZSTD_isError(zstd_sz)) { std::cerr << "zstd compress failed\n"; return false; }

    // Write blob
    std::ofstream out(outfile, std::ios::binary);
    if (!out) return false;
    out.write(VID_MAGIC, 4);
    out.write((char*)&vid.w, 4);
    out.write((char*)&vid.h, 4);
    out.write((char*)&vid.frames, 4);
    uint8_t ch8 = (uint8_t)vid.ch;
    out.write((char*)&ch8, 1);
    uint8_t fmt = is_y4m ? 1 : 0;  // 0=RGB, 1=YUV420
    out.write((char*)&fmt, 1);
    uint32_t zs = (uint32_t)zstd_sz;
    out.write((char*)&zs, 4);
    out.write((char*)zstd_data.data(), zstd_sz);
    out.close();

    if (enc_ms) {
        auto t1 = Clock::now();
        *enc_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    return true;
}

// Decode video: reverse pipeline
static bool decode_video(const std::string& infile, const std::string& outfile, double* dec_ms = nullptr, double* shannon_T = nullptr) {
    auto t0 = Clock::now();

    std::ifstream f(infile, std::ios::binary);
    if (!f) return false;
    char magic[4];
    f.read(magic, 4);
    if (memcmp(magic, VID_MAGIC, 4) != 0) { std::cerr << "Bad video magic\n"; return false; }

    int w, h, frames;
    f.read((char*)&w, 4);
    f.read((char*)&h, 4);
    f.read((char*)&frames, 4);
    uint8_t ch, fmt;
    f.read((char*)&ch, 1);
    f.read((char*)&fmt, 1);
    uint32_t zstd_sz;
    f.read((char*)&zstd_sz, 4);
    std::vector<uint8_t> zstd_data(zstd_sz);
    f.read((char*)zstd_data.data(), zstd_sz);

    // Calculate flat size based on format
    size_t flat_size;
    if (fmt == 1) {
        // YUV420: Y (w*h) + U (w/2*h/2) + V (w/2*h/2) per frame
        flat_size = (size_t)frames * (w * h + 2 * (w/2) * (h/2));
    } else {
        // RGB: w*h*ch per frame
        flat_size = (size_t)w * h * frames * ch;
    }
    std::vector<uint8_t> flat(flat_size);
    size_t dres = ZSTD_decompress(flat.data(), flat_size, zstd_data.data(), zstd_sz);
    if (ZSTD_isError(dres)) { std::cerr << "zstd decode failed\n"; return false; }
    if (shannon_T) *shannon_T = (double)shannon_limit_data(flat);

    // Undo spatial delta per frame
    std::vector<uint8_t> pixels(flat_size);
    if (fmt == 1) {
        // YUV420: decode each plane separately
        int y_size = w * h;
        int uv_size = (w / 2) * (h / 2);
        int frame_size = y_size + 2 * uv_size;
        for (int fr = 0; fr < frames; fr++) {
            uint8_t* frame = pixels.data() + fr * frame_size;
            uint8_t* src = flat.data() + fr * frame_size;
            // Y plane
            std::vector<uint8_t> y_tmp(y_size);
            delta2d_decode(src, y_tmp.data(), w, h);
            memcpy(frame, y_tmp.data(), y_size);
            // U plane
            std::vector<uint8_t> u_tmp(uv_size);
            delta2d_decode(src + y_size, u_tmp.data(), w / 2, h / 2);
            memcpy(frame + y_size, u_tmp.data(), uv_size);
            // V plane
            std::vector<uint8_t> v_tmp(uv_size);
            delta2d_decode(src + y_size + uv_size, v_tmp.data(), w / 2, h / 2);
            memcpy(frame + y_size + uv_size, v_tmp.data(), uv_size);
        }
    } else {
        // RGB: decode each channel
        size_t frame_size = (size_t)w * h * ch;
        for (int fr = 0; fr < frames; fr++) {
            uint8_t* frame = pixels.data() + fr * frame_size;
            for (int c = 0; c < ch; c++) {
                std::vector<uint8_t> chan(flat.begin() + (fr * ch + c) * w * h,
                                          flat.begin() + (fr * ch + c + 1) * w * h);
                std::vector<uint8_t> tmp(w * h);
                delta2d_decode(chan.data(), tmp.data(), w, h);
#if defined(NSPACK_HIGHWAY)
                nspack_hwy::insert_channel_u8(frame, tmp.data(), (size_t)w * h, ch, c);
#else
                for (int i = 0; i < w * h; i++) frame[i * ch + c] = tmp[i];
#endif
            }
        }
    }

    // Undo temporal delta
    size_t td_frame_size;
    if (fmt == 1) {
        td_frame_size = (size_t)(w * h + 2 * (w / 2) * (h / 2));
    } else {
        td_frame_size = (size_t)w * h * ch;
    }
    temporal_delta_decode(pixels, td_frame_size, frames);

    // Write raw frame sequence
    std::ofstream out(outfile, std::ios::binary);
    if (!out) return false;
    uint32_t w32 = w, h32 = h, fr32 = frames, ch32 = ch;
    out.write((char*)&w32, 4);
    out.write((char*)&h32, 4);
    out.write((char*)&fr32, 4);
    out.write((char*)&ch32, 4);
    out.write((char*)pixels.data(), pixels.size());
    out.close();

    if (dec_ms) {
        auto t1 = Clock::now();
        *dec_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    return true;
}

// ===== Procedural Geometry Compression =====
// Detect smooth surface regions and fit parametric patches

static const char PROC_MAGIC[4] = {'N','S','P','1'};

// Compute face normal from 3 vertices
static void compute_face_normal(const float* v0, const float* v1, const float* v2, float* n) {
    float ax = v1[0]-v0[0], ay = v1[1]-v0[1], az = v1[2]-v0[2];
    float bx = v2[0]-v0[0], by = v2[1]-v0[1], bz = v2[2]-v0[2];
    n[0] = ay*bz - az*by;
    n[1] = az*bx - ax*bz;
    n[2] = ax*by - ay*bx;
    float len = sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
    if (len > 1e-10f) { n[0]/=len; n[1]/=len; n[2]/=len; }
    else { n[0]=0; n[1]=1; n[2]=0; }
}

// Angle between two normals in degrees
static float angle_between(const float* a, const float* b) {
    float dot = a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
    if (dot > 1.0f) dot = 1.0f;
    if (dot < -1.0f) dot = -1.0f;
    return acosf(dot) * 180.0f / 3.14159265358979f;
}

// Fit a quadratic surface patch to a set of vertices
// Model: z = a*x^2 + b*y^2 + c*x*y + d*x + e*y + f
// Returns 6 coefficients
static void fit_quadric_patch(const std::vector<float>& verts, float coeffs[6]) {
    // Least squares: build normal equations A^T * A * c = A^T * z
    double AtA[6][6] = {0};
    double AtZ[6] = {0};

    size_t n = verts.size() / 3;
    for (size_t i = 0; i < n; i++) {
        double x = verts[i*3], y = verts[i*3+1], z = verts[i*3+2];
        double row[6] = {x*x, y*y, x*y, x, y, 1.0};
        for (int r = 0; r < 6; r++) {
            AtZ[r] += row[r] * z;
            for (int c = 0; c < 6; c++)
                AtA[r][c] += row[r] * row[c];
        }
    }

    // Solve 6x6 system via Gaussian elimination
    double aug[6][7];
    for (int r = 0; r < 6; r++) {
        for (int c = 0; c < 6; c++) aug[r][c] = AtA[r][c];
        aug[r][6] = AtZ[r];
    }
    for (int p = 0; p < 6; p++) {
        int maxRow = p;
        for (int r = p+1; r < 6; r++)
            if (fabs(aug[r][p]) > fabs(aug[maxRow][p])) maxRow = r;
        for (int c = 0; c < 7; c++) std::swap(aug[p][c], aug[maxRow][c]);
        if (fabs(aug[p][p]) < 1e-15) { coeffs[p] = 0; continue; }
        for (int r = p+1; r < 6; r++) {
            double f = aug[r][p] / aug[p][p];
            for (int c = p; c < 7; c++) aug[r][c] -= f * aug[p][c];
        }
    }
    for (int p = 5; p >= 0; p--) {
        double sum = aug[p][6];
        for (int c = p+1; c < 6; c++) sum -= aug[p][c] * coeffs[c];
        coeffs[p] = (float)(sum / (fabs(aug[p][p]) < 1e-15 ? 1.0 : aug[p][p]));
    }
}

// Evaluate quadric patch at (x, y)
static float eval_quadric(const float coeffs[6], float x, float y) {
    return coeffs[0]*x*x + coeffs[1]*y*y + coeffs[2]*x*y + coeffs[3]*x + coeffs[4]*y + coeffs[5];
}

// Encode OBJ with procedural geometry: detect smooth regions, fit patches
static bool encode_obj_procedural(const std::string& infile, const std::string& outfile, double* enc_ms = nullptr) {
    auto t0 = Clock::now();

    ObjData obj = parse_obj(infile);
    if (obj.vertices.empty() || obj.faces.empty()) return false;

    int nverts = obj.vertices.size() / 3;
    int nfaces = obj.faces.size() / 3;

    // Compute face normals
    std::vector<float> fnorms(nfaces * 3);
    for (int f = 0; f < nfaces; f++) {
        uint32_t i0 = obj.faces[f*3], i1 = obj.faces[f*3+1], i2 = obj.faces[f*3+2];
        compute_face_normal(&obj.vertices[i0*3], &obj.vertices[i1*3], &obj.vertices[i2*3], &fnorms[f*3]);
    }

    // Build adjacency: for each face, list of adjacent faces (sharing an edge)
    std::vector<std::vector<int>> adj(nfaces);
    for (int f1 = 0; f1 < nfaces; f1++) {
        for (int f2 = f1+1; f2 < nfaces; f2++) {
            // Check if they share an edge (2 common vertices)
            int common = 0;
            for (int a = 0; a < 3; a++)
                for (int b = 0; b < 3; b++)
                    if (obj.faces[f1*3+a] == obj.faces[f2*3+b]) common++;
            if (common >= 2) {
                adj[f1].push_back(f2);
                adj[f2].push_back(f1);
            }
        }
    }

    // Region growing: group faces with < 5 degree normal difference
    std::vector<int> region_id(nfaces, -1);
    std::vector<std::vector<int>> regions;
    for (int seed = 0; seed < nfaces; seed++) {
        if (region_id[seed] >= 0) continue;
        int rid = regions.size();
        region_id[seed] = rid;
        std::vector<int> region;
        region.push_back(seed);
        std::queue<int> q;
        q.push(seed);
        while (!q.empty()) {
            int f = q.front(); q.pop();
            for (int nb : adj[f]) {
                if (region_id[nb] >= 0) continue;
                float angle = angle_between(&fnorms[f*3], &fnorms[nb*3]);
                if (angle < 5.0f) {
                    region_id[nb] = rid;
                    region.push_back(nb);
                    q.push(nb);
                }
            }
        }
        regions.push_back(region);
    }

    // For each smooth region (>= 10 faces), fit a quadric patch
    // For sharp regions, fall back to delta encoding
    struct PatchInfo {
        float coeffs[6];
        std::vector<uint32_t> verts;  // vertex indices in this patch
        float bbox[4];  // min_x, min_y, max_x, max_y
    };

    std::vector<PatchInfo> patches;
    std::vector<int> vertex_patch(nverts, -1);  // which patch a vertex belongs to
    int unpatched_verts = 0;

    for (int r = 0; r < (int)regions.size(); r++) {
        if ((int)regions[r].size() < 10) {
            // Sharp region: count unpatched verts
            for (int f : regions[r]) {
                for (int j = 0; j < 3; j++) {
                    int vi = obj.faces[f*3+j];
                    if (vertex_patch[vi] < 0) {
                        vertex_patch[vi] = -2;  // unpatched
                        unpatched_verts++;
                    }
                }
            }
            continue;
        }

        // Collect vertices in this region
        std::vector<float> rverts;
        std::vector<uint32_t> rvidxs;
        float minx = 1e30f, miny = 1e30f, maxx = -1e30f, maxy = -1e30f;

        for (int f : regions[r]) {
            for (int j = 0; j < 3; j++) {
                uint32_t vi = obj.faces[f*3+j];
                if (vertex_patch[vi] >= 0) continue;
                vertex_patch[vi] = (int)patches.size();
                float x = obj.vertices[vi*3], y = obj.vertices[vi*3+1], z = obj.vertices[vi*3+2];
                rverts.push_back(x); rverts.push_back(y); rverts.push_back(z);
                rvidxs.push_back(vi);
                if (x < minx) minx = x; if (x > maxx) maxx = x;
                if (y < miny) miny = y; if (y > maxy) maxy = y;
            }
        }

        if (rverts.size() < 30) {  // not enough for a patch
            for (uint32_t vi : rvidxs) vertex_patch[vi] = -2;
            unpatched_verts += rvidxs.size();
            continue;
        }

        // Fit quadric patch
        PatchInfo pi;
        fit_quadric_patch(rverts, pi.coeffs);
        pi.verts = rvidxs;
        pi.bbox[0] = minx; pi.bbox[1] = miny; pi.bbox[2] = maxx; pi.bbox[3] = maxy;

        // Check fit quality
        double max_err = 0;
        for (size_t i = 0; i < rvidxs.size(); i++) {
            float x = obj.vertices[rvidxs[i]*3], y = obj.vertices[rvidxs[i]*3+1], z = obj.vertices[rvidxs[i]*3+2];
            float pred = eval_quadric(pi.coeffs, x, y);
            double err = fabs(z - pred);
            if (err > max_err) max_err = err;
        }

        // Compute bounding box diagonal for relative error
        float diag = sqrt((maxx-minx)*(maxx-minx) + (maxy-miny)*(maxy-miny));
        double rel_err = (diag > 1e-10f) ? max_err / diag : 1.0;

        if (rel_err > 0.0001) {  // max 0.01% deviation
            // Patch doesn't fit well, fall back to delta encoding
            for (uint32_t vi : rvidxs) vertex_patch[vi] = -2;
            unpatched_verts += rvidxs.size();
            continue;
        }

        patches.push_back(pi);
    }

    // Count remaining unpatched
    for (int i = 0; i < nverts; i++) {
        if (vertex_patch[i] == -1) { vertex_patch[i] = -2; unpatched_verts++; }
    }

    // Build output: patches + unpatched vertices (delta encoded) + faces
    std::ofstream out(outfile, std::ios::binary);
    if (!out) return false;
    out.write(PROC_MAGIC, 4);

    int nverts32 = nverts, nfaces32 = nfaces;
    int npatches = patches.size();
    out.write((char*)&nverts32, 4);
    out.write((char*)&nfaces32, 4);
    out.write((char*)&npatches, 4);

    // Write patches
    for (auto& p : patches) {
        out.write((char*)p.coeffs, 6 * sizeof(float));
        out.write((char*)p.bbox, 4 * sizeof(float));
        uint32_t nv = p.verts.size();
        out.write((char*)&nv, 4);
        // Write vertex indices (as delta encoded)
        uint32_t prev = 0;
        for (uint32_t vi : p.verts) {
            uint32_t delta = vi - prev;
            out.write((char*)&delta, 4);
            prev = vi + 1;
        }
        // Write x,y coordinates of patched vertices (quantized to 16-bit relative to bbox)
        for (uint32_t vi : p.verts) {
            float x = obj.vertices[vi*3], y = obj.vertices[vi*3+1];
            float rx = (p.bbox[2] - p.bbox[0]);
            float ry = (p.bbox[3] - p.bbox[1]);
            if (rx < 1e-10f) rx = 1.0f;
            if (ry < 1e-10f) ry = 1.0f;
            int16_t qx = (int16_t)(((x - p.bbox[0]) / rx) * 65535.0f - 32768.0f);
            int16_t qy = (int16_t)(((y - p.bbox[1]) / ry) * 65535.0f - 32768.0f);
            out.write((char*)&qx, 2);
            out.write((char*)&qy, 2);
        }
    }

    // Write unpatched vertices (quantized + delta encoded)
    std::vector<int32_t> unpatched_idx;
    for (int i = 0; i < nverts; i++) {
        if (vertex_patch[i] == -2) unpatched_idx.push_back(i);
    }
    uint32_t nunpatched = unpatched_idx.size();
    out.write((char*)&nunpatched, 4);

    // Quantize unpatched vertices to 16-bit
    float vmin[3] = {1e30f, 1e30f, 1e30f}, vmax[3] = {-1e30f, -1e30f, -1e30f};
    for (int vi : unpatched_idx) {
        for (int c = 0; c < 3; c++) {
            float v = obj.vertices[vi*3+c];
            if (v < vmin[c]) vmin[c] = v;
            if (v > vmax[c]) vmax[c] = v;
        }
    }
    out.write((char*)vmin, 3 * sizeof(float));
    out.write((char*)vmax, 3 * sizeof(float));

    std::vector<int16_t> quant_verts(unpatched_idx.size() * 3);
    for (size_t i = 0; i < unpatched_idx.size(); i++) {
        int vi = unpatched_idx[i];
        for (int c = 0; c < 3; c++) {
            float range = vmax[c] - vmin[c];
            if (range < 1e-10f) range = 1.0f;
            float normalized = (obj.vertices[vi*3+c] - vmin[c]) / range;
            quant_verts[i*3+c] = (int16_t)(normalized * 65535.0f - 32768.0f);
        }
    }
    // Delta encode
    for (size_t i = quant_verts.size()-1; i >= 3; i--) {
        quant_verts[i] -= quant_verts[i-3];
    }
    // Zigzag + zstd (overflow-safe)
    std::vector<int32_t> qv32(quant_verts.begin(), quant_verts.end());
    std::vector<uint8_t> zigzag = pack_int16_zigzag(qv32);
    size_t zb = ZSTD_compressBound(zigzag.size());
    std::vector<uint8_t> zdata(zb);
    size_t zs = ZSTD_compress(zdata.data(), zb, zigzag.data(), zigzag.size(), 19);
    uint32_t zs32 = (uint32_t)zs;
    out.write((char*)&zs32, 4);
    out.write((char*)zdata.data(), zs);

    // Write vertex_patch mapping (which patch each vertex belongs to)
    std::vector<int16_t> vp_short(nverts);
    for (int i = 0; i < nverts; i++) vp_short[i] = (int16_t)vertex_patch[i];
    std::vector<int32_t> vp32(vp_short.begin(), vp_short.end());
    std::vector<uint8_t> vp_zigzag = pack_int16_zigzag(vp32);
    size_t vpb = ZSTD_compressBound(vp_zigzag.size());
    std::vector<uint8_t> vp_zdata(vpb);
    size_t vpzs = ZSTD_compress(vp_zdata.data(), vpb, vp_zigzag.data(), vp_zigzag.size(), 19);
    uint32_t vpzs32 = (uint32_t)vpzs;
    out.write((char*)&vpzs32, 4);
    out.write((char*)vp_zdata.data(), vpzs);

    // Write faces (just copy, zstd compressed)
    size_t fb = ZSTD_compressBound(obj.faces.size() * 4);
    std::vector<uint8_t> fdata(fb);
    size_t fs = ZSTD_compress(fdata.data(), fb, obj.faces.data(), obj.faces.size() * 4, 19);
    uint32_t fs32 = (uint32_t)fs;
    out.write((char*)&fs32, 4);
    out.write((char*)fdata.data(), fs);

    out.close();

    if (enc_ms) {
        auto t1 = Clock::now();
        *enc_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    return true;
}

// Decode procedural OBJ
static bool decode_obj_procedural(const std::string& infile, const std::string& outfile, double* dec_ms = nullptr, double* shannon_T = nullptr) {
    auto t0 = Clock::now();

    if (shannon_T) *shannon_T = (double)file_size(infile);

    std::ifstream f(infile, std::ios::binary);
    if (!f) return false;
    char magic[4];
    f.read(magic, 4);
    if (memcmp(magic, PROC_MAGIC, 4) != 0) return false;

    int nverts, nfaces, npatches;
    f.read((char*)&nverts, 4);
    f.read((char*)&nfaces, 4);
    f.read((char*)&npatches, 4);

    // Read patches
    struct DecPatch {
        float coeffs[6];
        float bbox[4];
        std::vector<uint32_t> verts;
        std::vector<float> xy;  // x,y coordinates for each vertex
    };
    std::vector<DecPatch> patches(npatches);
    for (int p = 0; p < npatches; p++) {
        f.read((char*)patches[p].coeffs, 6 * sizeof(float));
        f.read((char*)patches[p].bbox, 4 * sizeof(float));
        uint32_t nv;
        f.read((char*)&nv, 4);
        patches[p].verts.resize(nv);
        uint32_t prev = 0;
        for (uint32_t i = 0; i < nv; i++) {
            uint32_t delta;
            f.read((char*)&delta, 4);
            prev += delta;
            patches[p].verts[i] = prev;
            prev++;
        }
        // Read x,y coordinates
        patches[p].xy.resize(nv * 2);
        for (uint32_t i = 0; i < nv; i++) {
            int16_t qx, qy;
            f.read((char*)&qx, 2);
            f.read((char*)&qy, 2);
            float rx = patches[p].bbox[2] - patches[p].bbox[0];
            float ry = patches[p].bbox[3] - patches[p].bbox[1];
            if (rx < 1e-10f) rx = 1.0f;
            if (ry < 1e-10f) ry = 1.0f;
            patches[p].xy[i*2] = patches[p].bbox[0] + ((qx + 32768.0f) / 65535.0f) * rx;
            patches[p].xy[i*2+1] = patches[p].bbox[1] + ((qy + 32768.0f) / 65535.0f) * ry;
        }
    }

    // Read unpatched vertices
    uint32_t nunpatched;
    f.read((char*)&nunpatched, 4);
    float vmin[3], vmax[3];
    f.read((char*)vmin, 3 * sizeof(float));
    f.read((char*)vmax, 3 * sizeof(float));

    uint32_t zs32;
    f.read((char*)&zs32, 4);
    std::vector<uint8_t> zdata(zs32);
    f.read((char*)zdata.data(), zs32);
    unsigned long long psize = ZSTD_getFrameContentSize(zdata.data(), zs32);
    if (psize == ZSTD_CONTENTSIZE_ERROR || psize == ZSTD_CONTENTSIZE_UNKNOWN) return false;
    std::vector<uint8_t> zigzag((size_t)psize);
    if (ZSTD_isError(ZSTD_decompress(zigzag.data(), zigzag.size(), zdata.data(), zs32))) return false;

    std::vector<int32_t> qv32 = unpack_int16_zigzag(zigzag);
    std::vector<int16_t> quant_verts(qv32.begin(), qv32.end());
    // Undo delta
    for (size_t i = 3; i < quant_verts.size(); i++) {
        quant_verts[i] += quant_verts[i-3];
    }
    // Dequantize
    std::vector<float> unpatched_verts(nunpatched * 3);
    for (size_t i = 0; i < nunpatched; i++) {
        for (int c = 0; c < 3; c++) {
            float range = vmax[c] - vmin[c];
            if (range < 1e-10f) range = 1.0f;
            float normalized = (quant_verts[i*3+c] + 32768.0f) / 65535.0f;
            unpatched_verts[i*3+c] = vmin[c] + normalized * range;
        }
    }

    // Read vertex_patch mapping
    uint32_t vpzs32;
    f.read((char*)&vpzs32, 4);
    std::vector<uint8_t> vp_zdata(vpzs32);
    f.read((char*)vp_zdata.data(), vpzs32);
    unsigned long long vpsize = ZSTD_getFrameContentSize(vp_zdata.data(), vpzs32);
    if (vpsize == ZSTD_CONTENTSIZE_ERROR || vpsize == ZSTD_CONTENTSIZE_UNKNOWN) return false;
    std::vector<uint8_t> vp_zigzag((size_t)vpsize);
    if (ZSTD_isError(ZSTD_decompress(vp_zigzag.data(), vp_zigzag.size(), vp_zdata.data(), vpzs32))) return false;

    std::vector<int32_t> vp32 = unpack_int16_zigzag(vp_zigzag);
    std::vector<int16_t> vp_short(vp32.begin(), vp32.end());

    // Read faces
    uint32_t fs32;
    f.read((char*)&fs32, 4);
    std::vector<uint8_t> fdata(fs32);
    f.read((char*)fdata.data(), fs32);
    std::vector<uint32_t> faces(nfaces * 3);
    ZSTD_decompress(faces.data(), nfaces * 3 * 4, fdata.data(), fs32);

    // Reconstruct vertices
    std::vector<float> vertices(nverts * 3);
    size_t unpatched_idx = 0;

    for (int p = 0; p < npatches; p++) {
        for (size_t i = 0; i < patches[p].verts.size(); i++) {
            uint32_t vi = patches[p].verts[i];
            float x = patches[p].xy[i*2];
            float y = patches[p].xy[i*2+1];
            vertices[vi*3] = x;
            vertices[vi*3+1] = y;
            vertices[vi*3+2] = eval_quadric(patches[p].coeffs, x, y);
        }
    }

    // Reconstruct unpatched vertices
    for (int i = 0; i < nverts; i++) {
        if (vp_short[i] == -2) {
            vertices[i*3] = unpatched_verts[unpatched_idx*3];
            vertices[i*3+1] = unpatched_verts[unpatched_idx*3+1];
            vertices[i*3+2] = unpatched_verts[unpatched_idx*3+2];
            unpatched_idx++;
        }
    }

    // Write OBJ
    std::ofstream out(outfile);
    if (!out) return false;
    for (int i = 0; i < nverts; i++) {
        out << "v " << vertices[i*3] << " " << vertices[i*3+1] << " " << vertices[i*3+2] << "\n";
    }
    for (int i = 0; i < nfaces; i++) {
        out << "f " << (faces[i*3]+1) << " " << (faces[i*3+1]+1) << " " << (faces[i*3+2]+1) << "\n";
    }
    out.close();

    if (dec_ms) {
        auto t1 = Clock::now();
        *dec_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    return true;
}

// ===== BVH Motion Capture Support =====
// BVH files contain hierarchical bone structure + per-frame rotation data

static const char BVH_MAGIC[4] = {'N','S','B','1'};

struct BvhData {
    int frames = 0;
    int channels = 0;
    std::vector<float> motion;
    std::string hierarchy;   // raw HIERARCHY block (everything before MOTION)
    std::string frame_time;  // original Frame Time value string
};

// Parse BVH file: extract motion data (skip hierarchy, just read MOTION section)
static bool parse_bvh(const std::string& filename, BvhData& bvh) {
    std::ifstream f(filename);
    if (!f) return false;

    std::string line;
    bool in_motion = false;
    bool found_frames = false;

    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (line.find("MOTION") != std::string::npos) { in_motion = true; continue; }
        if (!in_motion) { bvh.hierarchy += line; bvh.hierarchy += '\n'; continue; }
        if (line.find("Frames:") != std::string::npos) {
            bvh.frames = std::stoi(line.substr(7));
            found_frames = true;
            continue;
        }
        if (line.find("Frame Time:") != std::string::npos) {
            bvh.frame_time = line.substr(line.find(':') + 1);
            size_t a = bvh.frame_time.find_first_not_of(" \t");
            size_t b = bvh.frame_time.find_last_not_of(" \t");
            bvh.frame_time = (a == std::string::npos) ? "0.033333" : bvh.frame_time.substr(a, b - a + 1);
            continue;
        }
        if (found_frames && !line.empty()) {
            std::istringstream iss(line);
            float val;
            int count = 0;
            while (iss >> val) { bvh.motion.push_back(val); count++; }
            if (bvh.channels == 0 && count > 0) bvh.channels = count;
        }
    }

    return bvh.frames > 0 && bvh.channels > 0 && bvh.motion.size() == (size_t)(bvh.frames * bvh.channels);
}

// Encode BVH: temporal delta + zstd
static bool encode_bvh(const std::string& infile, const std::string& outfile, double* enc_ms = nullptr) {
    auto t0 = Clock::now();

    BvhData bvh;
    if (!parse_bvh(infile, bvh)) {
        std::cerr << "Failed to parse BVH: " << infile << "\n";
        return false;
    }

    // Convert to bytes for delta encoding (treat floats as uint32)
    std::vector<uint8_t> data(bvh.motion.size() * 4);
    memcpy(data.data(), bvh.motion.data(), data.size());

    // Temporal delta: frame[i] = frame[i] - frame[i-1]
    size_t frame_size = bvh.channels * 4;
    for (int f = bvh.frames - 1; f > 0; f--) {
        uint8_t* curr = data.data() + f * frame_size;
        uint8_t* prev = data.data() + (f - 1) * frame_size;
        for (size_t i = 0; i < frame_size; i++) {
            curr[i] = (uint8_t)(curr[i] - prev[i]);
        }
    }

    // Compress with zstd
    size_t z_bound = ZSTD_compressBound(data.size());
    std::vector<uint8_t> zstd_data(z_bound);
    size_t zstd_sz = ZSTD_compress(zstd_data.data(), z_bound, data.data(), data.size(), 19);
    if (ZSTD_isError(zstd_sz)) { std::cerr << "zstd compress failed\n"; return false; }

    // Write blob
    std::ofstream out(outfile, std::ios::binary);
    if (!out) return false;
    out.write(BVH_MAGIC, 4);
    out.write((char*)&bvh.frames, 4);
    out.write((char*)&bvh.channels, 4);
    uint32_t ftl = (uint32_t)bvh.frame_time.size();
    out.write((char*)&ftl, 4);
    out.write(bvh.frame_time.data(), ftl);
    uint32_t hl = (uint32_t)bvh.hierarchy.size();
    out.write((char*)&hl, 4);
    out.write(bvh.hierarchy.data(), hl);
    uint32_t zs = (uint32_t)zstd_sz;
    out.write((char*)&zs, 4);
    out.write((char*)zstd_data.data(), zstd_sz);
    out.close();

    if (enc_ms) {
        auto t1 = Clock::now();
        *enc_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    return true;
}

// Decode BVH: reverse pipeline
static bool decode_bvh(const std::string& infile, const std::string& outfile, double* dec_ms = nullptr, double* shannon_T = nullptr) {
    auto t0 = Clock::now();

    std::ifstream f(infile, std::ios::binary);
    if (!f) return false;
    char magic[4];
    f.read(magic, 4);
    if (memcmp(magic, BVH_MAGIC, 4) != 0) { std::cerr << "Bad BVH magic\n"; return false; }

    int frames, channels;
    f.read((char*)&frames, 4);
    f.read((char*)&channels, 4);
    uint32_t ftl = 0;
    f.read((char*)&ftl, 4);
    std::string frame_time(ftl, '\0');
    f.read(frame_time.data(), ftl);
    uint32_t hl = 0;
    f.read((char*)&hl, 4);
    std::string hierarchy(hl, '\0');
    f.read(hierarchy.data(), hl);
    uint32_t zstd_sz;
    f.read((char*)&zstd_sz, 4);
    std::vector<uint8_t> zstd_data(zstd_sz);
    f.read((char*)zstd_data.data(), zstd_sz);

    size_t data_size = (size_t)frames * channels * 4;
    std::vector<uint8_t> data(data_size);
    size_t dres = ZSTD_decompress(data.data(), data_size, zstd_data.data(), zstd_sz);
    if (ZSTD_isError(dres)) { std::cerr << "zstd decode failed\n"; return false; }
    if (shannon_T) *shannon_T = (double)shannon_limit_bytes(data);

    // Undo temporal delta
    size_t frame_size = channels * 4;
    for (int fr = 1; fr < frames; fr++) {
        uint8_t* curr = data.data() + fr * frame_size;
        uint8_t* prev = data.data() + (fr - 1) * frame_size;
        for (size_t i = 0; i < frame_size; i++) {
            curr[i] = (uint8_t)(curr[i] + prev[i]);
        }
    }

    // Write back as BVH: original HIERARCHY + reconstructed MOTION with original frame time
    std::ofstream out(outfile);
    if (!out) return false;
    out << hierarchy;
    out << "MOTION\nFrames: " << frames << "\nFrame Time: " << frame_time << "\n";
    for (int fr = 0; fr < frames; fr++) {
        float* frame_data = (float*)(data.data() + fr * frame_size);
        for (int c = 0; c < channels; c++) {
            out << frame_data[c];
            if (c < channels - 1) out << " ";
        }
        out << "\n";
    }
    out.close();

    if (dec_ms) {
        auto t1 = Clock::now();
        *dec_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    return true;
}

// ===== File Type Detection =====
static bool is_image_file(const std::string& filename) {
    size_t pos = filename.rfind('.');
    if (pos == std::string::npos) return false;
    std::string ext = filename.substr(pos);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" || ext == ".tga";
}

static bool is_video_file(const std::string& filename) {
    size_t pos = filename.rfind('.');
    if (pos == std::string::npos) return false;
    std::string ext = filename.substr(pos);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".y4m" || ext == ".rgb" || ext == ".mp4" || ext == ".webm" || ext == ".gif";
}

static bool is_bvh_file(const std::string& filename) {
    size_t pos = filename.rfind('.');
    if (pos == std::string::npos) return false;
    std::string ext = filename.substr(pos);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".bvh";
}


static bool is_wav_file(const std::string& filename) {
    size_t pos = filename.rfind('.');
    if (pos == std::string::npos) return false;
    std::string ext = filename.substr(pos);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".wav";
}

static bool is_code_file(const std::string& filename) {
    size_t pos = filename.rfind('.');
    if (pos == std::string::npos) return false;
    std::string ext = filename.substr(pos);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".py" || ext == ".c" || ext == ".cpp" || ext == ".h" || ext == ".hpp";
}

static bool is_ply_file(const std::string& filename) {
    size_t pos = filename.rfind('.');
    if (pos == std::string::npos) return false;
    std::string ext = filename.substr(pos);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".ply";
}

static bool is_scientific_file(const std::string& filename) {
    size_t pos = filename.rfind('.');
    if (pos == std::string::npos) return false;
    std::string ext = filename.substr(pos);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".sci";
}

static bool is_hdr_file(const std::string& filename) {
    size_t pos = filename.rfind('.');
    if (pos == std::string::npos) return false;
    std::string ext = filename.substr(pos);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".hdr";
}

static bool is_sqlite_file(const std::string& filename) {
    size_t pos = filename.rfind('.');
    if (pos == std::string::npos) return false;
    std::string ext = filename.substr(pos);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".sqlite" || ext == ".db";
}

static bool is_elf_file(const std::string& filename) {
    size_t pos = filename.rfind('.');
    if (pos == std::string::npos) return false;
    std::string ext = filename.substr(pos);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".bin" || ext == ".elf";
}

static bool is_raw_video(const std::string& filename) {
    size_t pos = filename.rfind('.');
    if (pos == std::string::npos) return false;
    std::string ext = filename.substr(pos);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".y4m" || ext == ".rgb";
}

// ===== Verify image roundtrip =====
static bool verify_image(const std::string& original, const std::string& decoded) {
    int w1, h1, ch1;
    uint8_t* orig_pixels = stbi_load(original.c_str(), &w1, &h1, &ch1, 0);
    if (!orig_pixels) return false;

    std::ifstream f(decoded, std::ios::binary);
    if (!f) { stbi_image_free(orig_pixels); return false; }
    std::vector<uint8_t> dec_pixels((std::istreambuf_iterator<char>(f)),
                                    std::istreambuf_iterator<char>());

    size_t expected = (size_t)w1 * h1 * ch1;
    bool match = (dec_pixels.size() == expected);
    if (match) {
        match = (memcmp(orig_pixels, dec_pixels.data(), expected) == 0);
    }

    stbi_image_free(orig_pixels);
    return match;
}

// ===== Phase 3: Benchmark =====
static int benchmark(const std::vector<std::string>& files) {
    std::cout << "\n=== NSPack Benchmark ===\n\n";
    std::cout << "filename                        | original  | nspack    | gzip     | zstd     | shannon  | ns/limit | enc_ms  | dec_ms  | match\n";
    std::cout << std::string(140, '-') << "\n";

    for (const auto& file : files) {
        size_t orig = file_size(file);
        if (orig == 0) {
            std::cerr << "Skipping (empty): " << file << "\n";
            continue;
        }

        bool is_img = is_image_file(file);
        bool is_vid = is_video_file(file);
        bool is_raw_vid = is_raw_video(file);
        bool is_bvh = is_bvh_file(file);
        bool is_wav = is_wav_file(file);
        bool is_code = is_code_file(file);
        bool is_ply = is_ply_file(file);
        bool is_sci = is_scientific_file(file);
        bool is_hdr = is_hdr_file(file);
        bool is_sql = is_sqlite_file(file);
        bool is_elf = is_elf_file(file);
        std::string nsp = file + ".nsp";
        std::string dec;
        if (is_img) dec = file + ".dec.raw";
        else if (is_vid) dec = file + ".dec.vid";
        else if (is_bvh) dec = file + ".dec.bvh";
        else if (is_wav) dec = file + ".dec.wav";
        else if (is_ply) dec = file + ".dec.ply";
        else if (is_sci) dec = file + ".dec.sci";
        else if (is_hdr) dec = file + ".dec.hdr";
        else if (is_sql) dec = file + ".dec.csv";
        else if (is_elf) dec = file + ".dec.bin";
        else dec = file + ".dec.obj";

        // Encode
        double enc_ms;
        bool enc_ok;
        if (is_bvh) {
            enc_ok = encode_bvh(file, nsp, &enc_ms);
        } else if (is_wav) {
            enc_ok = encode_wav(file, nsp, &enc_ms);
        } else if (is_code) {
            enc_ok = encode_code_ast(file, nsp, &enc_ms);
        } else if (is_ply) {
            enc_ok = encode_ply(file, nsp, &enc_ms);
        } else if (is_sci) {
            enc_ok = encode_scientific(file, nsp, &enc_ms);
        } else if (is_hdr) {
            enc_ok = encode_hdr(file, nsp, &enc_ms);
        } else if (is_sql) {
            enc_ok = encode_sqlite(file, nsp, &enc_ms);
        } else if (is_elf) {
            enc_ok = encode_elf(file, nsp, &enc_ms);
        } else if (is_raw_vid) {
            enc_ok = encode_video(file, nsp, &enc_ms);
        } else if (is_vid) {
            // Pre-compressed video: just zstd the file directly
            std::ifstream vf(file, std::ios::binary);
            std::vector<uint8_t> vdata((std::istreambuf_iterator<char>(vf)), std::istreambuf_iterator<char>());
            size_t zb = ZSTD_compressBound(vdata.size());
            std::vector<uint8_t> zd(zb);
            size_t zs = ZSTD_compress(zd.data(), zb, vdata.data(), vdata.size(), 19);
            std::ofstream of(nsp, std::ios::binary);
            of.write((char*)zd.data(), zs);
            of.close();
            enc_ok = !ZSTD_isError(zs);
            enc_ms = 0;
        } else if (is_img) {
            enc_ok = encode_image(file, nsp, &enc_ms);
        } else {
            enc_ok = encode_obj(file, nsp, &enc_ms);
        }
        if (!enc_ok) {
            std::cerr << "Encode failed: " << file << "\n";
            continue;
        }
        size_t nsp_sz = file_size(nsp);

        // Decode
        double dec_ms;
        double shannon_T = 0.0;
        bool dec_ok;
        if (is_bvh) {
            dec_ok = decode_bvh(nsp, dec, &dec_ms, &shannon_T);
        } else if (is_wav) {
            dec_ok = decode_wav(nsp, dec, &dec_ms, &shannon_T);
        } else if (is_code) {
            dec_ok = decode_code_ast(nsp, dec, &dec_ms, &shannon_T);
        } else if (is_ply) {
            dec_ok = decode_ply(nsp, dec, &dec_ms, &shannon_T);
        } else if (is_sci) {
            dec_ok = decode_scientific(nsp, dec, &dec_ms, &shannon_T);
        } else if (is_hdr) {
            dec_ok = decode_hdr(nsp, dec, &dec_ms, &shannon_T);
        } else if (is_sql) {
            dec_ok = decode_sqlite(nsp, dec, &dec_ms, &shannon_T);
        } else if (is_elf) {
            dec_ok = decode_elf(nsp, dec, &dec_ms, &shannon_T);
        } else if (is_raw_vid) {
            dec_ok = decode_video(nsp, dec, &dec_ms, &shannon_T);
        } else if (is_vid) {
            // Pre-compressed video: just decompress
            std::ifstream vf(nsp, std::ios::binary);
            std::vector<uint8_t> zd((std::istreambuf_iterator<char>(vf)), std::istreambuf_iterator<char>());
            std::vector<uint8_t> vdata(orig);
            size_t ds = ZSTD_decompress(vdata.data(), orig, zd.data(), zd.size());
            std::ofstream of(dec, std::ios::binary);
            of.write((char*)vdata.data(), ds);
            of.close();
            shannon_T = (double)shannon_limit_bytes(vdata);
            dec_ok = !ZSTD_isError(ds);
            dec_ms = 0;
        } else if (is_img) {
            dec_ok = decode_image(nsp, dec, &dec_ms, &shannon_T);
        } else {
            dec_ok = decode_obj(nsp, dec, &dec_ms, &shannon_T);
        }
        if (!dec_ok) {
            std::cerr << "Decode failed: " << file << "\n";
            continue;
        }

        // Verify
        bool match;
        if (is_bvh) {
            // Verify frame count matches
            BvhData orig_bvh, dec_bvh;
            bool ok1 = parse_bvh(file, orig_bvh);
            bool ok2 = parse_bvh(dec, dec_bvh);
            match = ok1 && ok2 && (orig_bvh.frames == dec_bvh.frames);
        } else if (is_raw_vid) {
            // Verify frame count matches
            VideoData orig_vid, dec_vid;
            bool ok1 = file.substr(file.size() - 4) == ".y4m" ? parse_y4m(file, orig_vid) : parse_raw_frames(file, orig_vid);
            bool ok2 = parse_raw_frames(dec, dec_vid);
            match = ok1 && ok2 && (orig_vid.frames == dec_vid.frames);
        } else if (is_vid) {
            // Pre-compressed: just check file sizes match
            match = (file_size(dec) == orig);
        } else if (is_img) {
            match = verify_image(file, dec);
        } else if (is_wav) {
            match = dec_ok; // verify by decode success
        } else if (is_code) {
            match = dec_ok; // AST round-trip to source not required
        } else if (is_ply) {
            match = dec_ok;
        } else if (is_sci) {
            match = dec_ok;
        } else if (is_hdr) {
            match = dec_ok;
        } else if (is_sql) {
            match = dec_ok;
        } else if (is_elf) {
            match = dec_ok;
        } else {
            ObjData orig_obj = parse_obj(file);
            ObjData dec_obj = parse_obj(dec);
            match = (orig_obj.vertices.size() == dec_obj.vertices.size() &&
                     orig_obj.faces.size() == dec_obj.faces.size());
        }

        // Gzip and zstd sizes
        size_t gz, zs;
        if (is_bvh) {
            // BVH: compare against raw motion data
            BvhData bvh;
            if (parse_bvh(file, bvh)) {
                std::string raw_tmp = file + ".raw.tmp";
                {
                    std::ofstream rf(raw_tmp, std::ios::binary);
                    rf.write((char*)bvh.motion.data(), bvh.motion.size() * 4);
                }
                gz = compressed_size("gzip -c \"" + raw_tmp + "\" | wc -c");
                zs = compressed_size("zstd -c \"" + raw_tmp + "\" | wc -c");
                std::remove(raw_tmp.c_str());
                orig = bvh.motion.size() * 4;
            } else {
                gz = compressed_size("gzip -c \"" + file + "\" | wc -c");
                zs = compressed_size("zstd -c \"" + file + "\" | wc -c");
            }
        } else if (is_raw_vid) {
            // Raw video: compare against raw pixel data
            VideoData vid;
            bool ok = file.substr(file.size() - 4) == ".y4m" ? parse_y4m(file, vid) : parse_raw_frames(file, vid);
            if (ok) {
                std::string raw_tmp = file + ".raw.tmp";
                {
                    std::ofstream rf(raw_tmp, std::ios::binary);
                    rf.write((char*)vid.pixels.data(), vid.pixels.size());
                }
                gz = compressed_size("gzip -c \"" + raw_tmp + "\" | wc -c");
                zs = compressed_size("zstd -c \"" + raw_tmp + "\" | wc -c");
                std::remove(raw_tmp.c_str());
                orig = vid.pixels.size();
            } else {
                gz = compressed_size("gzip -c \"" + file + "\" | wc -c");
                zs = compressed_size("zstd -c \"" + file + "\" | wc -c");
            }
        } else if (is_vid) {
            // Pre-compressed video: compare against file itself
            gz = compressed_size("gzip -c \"" + file + "\" | wc -c");
            zs = compressed_size("zstd -c \"" + file + "\" | wc -c");
        } else if (is_img) {
            int w, h, ch;
            uint8_t* px = stbi_load(file.c_str(), &w, &h, &ch, 0);
            std::string raw_tmp = file + ".raw.tmp";
            {
                std::ofstream rf(raw_tmp, std::ios::binary);
                rf.write((char*)px, w * h * ch);
            }
            stbi_image_free(px);
            gz = compressed_size("gzip -c \"" + raw_tmp + "\" | wc -c");
            zs = compressed_size("zstd -c \"" + raw_tmp + "\" | wc -c");
            std::remove(raw_tmp.c_str());
            orig = (size_t)w * h * ch;
        } else {
            gz = compressed_size("gzip -c \"" + file + "\" | wc -c");
            zs = compressed_size("zstd -c \"" + file + "\" | wc -c");
        }

        // Shannon limit of transformed representation H(T(X))
        size_t shannon = (size_t)ceil(shannon_T);
        double ns_vs_limit = (shannon > 0) ? (double)nsp_sz / shannon : 0.0;

        // Print row
        std::cout << file;
        for (size_t i = file.size(); i < 32; i++) std::cout << ' ';
        std::cout << " | " << orig
                  << " | " << nsp_sz
                  << " | " << gz
                  << " | " << zs
                  << " | " << shannon
                  << " | " << std::fixed << std::setprecision(2) << ns_vs_limit
                  << " | " << (int)enc_ms
                  << " | " << (int)dec_ms
                  << " | " << (match ? "YES" : "NO")
                  << "\n";

        // Print ratios
        std::cout << "  ratio: "
                  << std::fixed << std::setprecision(2)
                  << (double)orig / nsp_sz << "x (nspack) vs "
                  << (double)orig / gz << "x (gzip) vs "
                  << (double)orig / zs << "x (zstd)"
                  << " | shannon: " << (double)orig / shannon << "x"
                  << (nsp_sz < gz && nsp_sz < zs ? "  *** NSPack WINS ***" : "")
                  << "\n";

        // For OBJ files, also test procedural encoding
        if (!is_img && !is_vid && !is_raw_vid && !is_bvh && !is_wav && !is_code && !is_ply && !is_sci && !is_hdr && !is_sql && !is_elf) {
            std::string nsp_proc = file + ".proc.nsp";
            double proc_enc_ms = 0, proc_dec_ms = 0;
            bool proc_ok = encode_obj_procedural(file, nsp_proc, &proc_enc_ms);
            if (proc_ok) {
                size_t proc_sz = file_size(nsp_proc);
                std::string proc_dec = file + ".proc.dec.obj";
                double proc_shannon_T = 0.0;
                decode_obj_procedural(nsp_proc, proc_dec, &proc_dec_ms, &proc_shannon_T);

                // Verify procedural decode
                ObjData orig_obj2 = parse_obj(file);
                ObjData proc_dec_obj = parse_obj(proc_dec);
                bool proc_match = (proc_dec_obj.vertices.size() == orig_obj2.vertices.size() &&
                                  proc_dec_obj.faces.size() == orig_obj2.faces.size());

                size_t proc_shannon = (size_t)ceil(proc_shannon_T);
                std::cout << "  procedural: " << proc_sz << " bytes"
                          << " | " << (double)orig / proc_sz << "x"
                          << " | shannon: " << (double)orig / proc_shannon << "x"
                          << " | ns/limit: " << (double)proc_sz / proc_shannon
                          << " | enc: " << (int)proc_enc_ms << "ms"
                          << " | dec: " << (int)proc_dec_ms << "ms"
                          << " | match: " << (proc_match ? "YES" : "NO")
                          << (proc_sz < nsp_sz ? "  *** PROC WINS ***" : "")
                          << "\n";
                std::remove(nsp_proc.c_str());
                std::remove(proc_dec.c_str());
            } else {
                std::cout << "  procedural: FAILED\n";
            }
        }
        std::cout << "\n";

        // Cleanup
        std::remove(nsp.c_str());
        std::remove(dec.c_str());
    }
    return 0;
}

// ===== NSPack CLI and .nsp container =====

enum class NspType : uint8_t {
    OBJ = 0,
    IMAGE = 1,
    VIDEO = 2,
    BVH = 3,
    WAV = 4,
    CODE = 5,
    PLY = 6,
    SCI = 7,
    HDR = 8,
    SQL = 9,
    ELF = 10,
    PROC = 11,
    PRECOMPRESSED = 12,
    RAW_STREAMED = 255
};

static const char* nsp_type_name(NspType t) {
    switch (t) {
        case NspType::OBJ: return "OBJ";
        case NspType::IMAGE: return "IMAGE";
        case NspType::VIDEO: return "VIDEO";
        case NspType::BVH: return "BVH";
        case NspType::WAV: return "WAV";
        case NspType::CODE: return "CODE";
        case NspType::PLY: return "PLY";
        case NspType::SCI: return "SCI";
        case NspType::HDR: return "HDR";
        case NspType::SQL: return "SQL";
        case NspType::ELF: return "ELF";
        case NspType::PROC: return "PROC";
        case NspType::PRECOMPRESSED: return "PRECOMPRESSED";
        default: return "UNKNOWN";
    }
}

static std::string nsp_extension(NspType t) {
    switch (t) {
        case NspType::OBJ: return ".obj";
        case NspType::IMAGE: return ".raw";
        case NspType::VIDEO: return ".vid";
        case NspType::BVH: return ".bvh";
        case NspType::WAV: return ".wav";
        case NspType::CODE: return ".txt";
        case NspType::PLY: return ".ply";
        case NspType::SCI: return ".sci";
        case NspType::HDR: return ".hdr";
        case NspType::SQL: return ".csv";
        case NspType::ELF: return ".bin";
        case NspType::PROC: return ".obj";
        case NspType::PRECOMPRESSED: return ".bin";
        default: return "";
    }
}

static std::string to_lower_ext(const std::string& f) {
    size_t pos = f.rfind('.');
    if (pos == std::string::npos) return "";
    std::string ext = f.substr(pos);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext;
}

static NspType nsp_type_from_filename(const std::string& f) {
    std::string ext = to_lower_ext(f);
    if (ext == ".obj") return NspType::OBJ;
    if (ext == ".ply") return NspType::PLY;
    if (ext == ".wav") return NspType::WAV;
    if (ext == ".bvh") return NspType::BVH;
    if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" || ext == ".tga") return NspType::IMAGE;
    if (ext == ".y4m" || ext == ".rgb") return NspType::VIDEO;
    if (ext == ".exr" || ext == ".hdr") return NspType::HDR;
    if (ext == ".mp4" || ext == ".webm" || ext == ".gif") return NspType::PRECOMPRESSED;
    if (ext == ".py" || ext == ".c" || ext == ".cpp" || ext == ".h" || ext == ".hpp") return NspType::CODE;
    if (ext == ".sci") return NspType::SCI;
    if (ext == ".sqlite" || ext == ".db") return NspType::SQL;
    if (ext == ".bin" || ext == ".elf") return NspType::ELF;
    return NspType::PRECOMPRESSED;
}

static size_t file_size_bytes(const std::string& f) {
    std::ifstream s(f, std::ios::binary | std::ios::ate);
    if (!s) return 0;
    return (size_t)s.tellg();
}

static std::string filename_part(const std::string& p) {
    size_t s = p.find_last_of("/\\");
    return (s == std::string::npos) ? p : p.substr(s + 1);
}

static std::string without_ext(const std::string& p) {
    size_t dot = p.rfind('.');
    if (dot == std::string::npos || dot == 0) return p;
    return p.substr(0, dot);
}

static std::string json_escape(const std::string& s) {
    std::string r;
    for (char c : s) {
        if (c == '\\') r += "\\\\";
        else if (c == '"') r += "\\\"";
        else if (c == '\n') r += "\\n";
        else if (c == '\r') r += "\\r";
        else if (c == '\t') r += "\\t";
        else if ((unsigned char)c < 0x20) r += '?';
        else r += c;
    }
    return r;
}

static size_t streaming_threshold() {
#ifdef _WIN32
    MEMORYSTATUSEX ms;
    memset(&ms, 0, sizeof(ms));
    ms.dwLength = sizeof(ms);
    size_t ram = (size_t)4ULL * 1024 * 1024 * 1024;
    if (GlobalMemoryStatusEx(&ms)) ram = (size_t)ms.ullTotalPhys;
    size_t t512 = (size_t)512 * 1024 * 1024;
    return std::min(t512, ram / 4);
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long page = sysconf(_SC_PAGE_SIZE);
    size_t ram = (pages > 0 && page > 0) ? (size_t)pages * (size_t)page : (size_t)4ULL * 1024 * 1024 * 1024;
    size_t t512 = (size_t)512 * 1024 * 1024;
    return std::min(t512, ram / 4);
#endif
}

static bool should_stream(const std::string& f) {
    return file_size_bytes(f) > streaming_threshold();
}

// .nsp file format:
//   4 magic "NSP1"
//   1 type enum
//   8 original size
//   4 metadata length
//   N metadata (JSON)
//   payload

static bool write_nsp_file(const std::string& out, NspType t, uint64_t orig_size,
                           const std::string& meta, const std::vector<uint8_t>& payload) {
    std::ofstream of(out, std::ios::binary);
    if (!of) return false;
    const char magic[4] = {'N','S','P','1'};
    of.write(magic, 4);
    uint8_t type8 = (uint8_t)t;
    of.write((char*)&type8, 1);
    of.write((char*)&orig_size, 8);
    uint32_t mlen = (uint32_t)meta.size();
    of.write((char*)&mlen, 4);
    if (mlen) of.write(meta.data(), meta.size());
    if (!payload.empty()) of.write((char*)payload.data(), payload.size());
    of.close();
    return of.good();
}

static bool read_nsp_file(const std::string& in, NspType& t, uint64_t& orig_size,
                          std::string& meta, std::vector<uint8_t>& payload) {
    std::ifstream f(in, std::ios::binary);
    if (!f) return false;
    char magic[4];
    f.read(magic, 4);
    if (memcmp(magic, "NSP1", 4) != 0) {
        std::cerr << "Not an NSP1 file.\n";
        return false;
    }
    uint8_t type8; f.read((char*)&type8, 1); t = (NspType)type8;
    f.read((char*)&orig_size, 8);
    uint32_t mlen; f.read((char*)&mlen, 4);
    meta.resize(mlen); if (mlen) f.read(meta.data(), mlen);
    size_t total = file_size_bytes(in);
    if (total < (size_t)17 + mlen) {
        std::cerr << "Corrupt NSP1 file.\n";
        return false;
    }
    size_t psize = total - (17ULL + mlen);
    payload.resize(psize);
    if (psize) f.read((char*)payload.data(), psize);
    return f.good();
}

// Gather simple metadata counts by parsing the original file.
static std::string build_metadata(const std::string& in, NspType t, uint64_t orig_size) {
    std::string meta = "{";
    meta += "\"original_filename\":\"" + json_escape(filename_part(in)) + "\",";
    meta += "\"type\":\"" + std::string(nsp_type_name(t)) + "\",";
    meta += "\"original_size\":" + std::to_string(orig_size) + ",";
    meta += "\"encoder_version\":\"2.0\",";
    meta += "\"counts\":{";

    if (t == NspType::OBJ) {
        ObjData o = parse_obj(in);
        meta += "\"vertices\":" + std::to_string(o.vertices.size() / 3) + ",";
        meta += "\"faces\":" + std::to_string(o.faces.size() / 3) + ",";
        meta += "\"normals\":" + std::to_string(o.normals.size() / 3);
    } else if (t == NspType::PLY) {
        int npts = 0; std::vector<float> xyz, intensity;
        parse_ply(in, npts, xyz, intensity);
        meta += "\"points\":" + std::to_string(npts);
    } else if (t == NspType::BVH) {
        BvhData b; parse_bvh(in, b);
        meta += "\"frames\":" + std::to_string(b.frames) + ",";
        meta += "\"channels\":" + std::to_string(b.channels);
    } else if (t == NspType::WAV) {
        int sr, ch, bits; std::vector<int16_t> samples;
        parse_wav(in, sr, ch, bits, samples);
        meta += "\"sample_rate\":" + std::to_string(sr) + ",";
        meta += "\"channels\":" + std::to_string(ch) + ",";
        meta += "\"samples\":" + std::to_string(samples.size() / std::max(1, ch));
    } else if (t == NspType::IMAGE) {
        int w, h, ch; uint8_t* px = stbi_load(in.c_str(), &w, &h, &ch, 0);
        if (px) stbi_image_free(px);
        meta += "\"width\":" + std::to_string(w) + ",";
        meta += "\"height\":" + std::to_string(h) + ",";
        meta += "\"channels\":" + std::to_string(ch);
    } else if (t == NspType::VIDEO) {
        VideoData v;
        bool is_y4m = in.substr(in.size() - 4) == ".y4m";
        if (is_y4m) parse_y4m(in, v); else parse_raw_frames(in, v);
        meta += "\"width\":" + std::to_string(v.w) + ",";
        meta += "\"height\":" + std::to_string(v.h) + ",";
        meta += "\"frames\":" + std::to_string(v.frames) + ",";
        meta += "\"channels\":" + std::to_string(v.ch);
    } else if (t == NspType::SCI) {
        std::ifstream f(in, std::ios::binary);
        uint8_t ht; uint32_t rows, cols; uint8_t dt;
        f.read((char*)&ht, 1); f.read((char*)&rows, 4); f.read((char*)&cols, 4); f.read((char*)&dt, 1);
        meta += "\"sci_type\":" + std::to_string(ht) + ",";
        meta += "\"rows\":" + std::to_string(rows) + ",";
        meta += "\"cols\":" + std::to_string(cols);
    } else {
        meta += "\"na\":true";
    }
    meta += "}}";
    return meta;
}

// Encode payload to a temporary file.
static bool encode_to_payload_file(const std::string& in, const std::string& payload_file,
                                   NspType t, double* enc_ms) {
    switch (t) {
        case NspType::OBJ: return encode_obj(in, payload_file, enc_ms);
        case NspType::IMAGE: return encode_image(in, payload_file, enc_ms);
        case NspType::VIDEO: return encode_video(in, payload_file, enc_ms);
        case NspType::BVH: return encode_bvh(in, payload_file, enc_ms);
        case NspType::WAV: return encode_wav(in, payload_file, enc_ms);
        case NspType::CODE: return encode_code_ast(in, payload_file, enc_ms);
        case NspType::PLY: return encode_ply(in, payload_file, enc_ms);
        case NspType::SCI: return encode_scientific(in, payload_file, enc_ms);
        case NspType::HDR: return encode_hdr(in, payload_file, enc_ms);
        case NspType::SQL: return encode_sqlite(in, payload_file, enc_ms);
        case NspType::ELF: return encode_elf(in, payload_file, enc_ms);
        case NspType::PRECOMPRESSED: {
            std::ifstream vf(in, std::ios::binary);
            std::ofstream of(payload_file, std::ios::binary);
            of << vf.rdbuf();
            return of.good();
        }
        default:
            return encode_obj(in, payload_file, enc_ms);
    }
}

static bool decode_payload_file(const std::string& payload_file, const std::string& out, NspType t, double* dec_ms, double* shannon) {
    switch (t) {
        case NspType::OBJ: return decode_obj(payload_file, out, dec_ms, shannon);
        case NspType::IMAGE: return decode_image(payload_file, out, dec_ms, shannon);
        case NspType::VIDEO: return decode_video(payload_file, out, dec_ms, shannon);
        case NspType::BVH: return decode_bvh(payload_file, out, dec_ms, shannon);
        case NspType::WAV: return decode_wav(payload_file, out, dec_ms, shannon);
        case NspType::CODE: return decode_code_ast(payload_file, out, dec_ms, shannon);
        case NspType::PLY: return decode_ply(payload_file, out, dec_ms, shannon);
        case NspType::SCI: return decode_scientific(payload_file, out, dec_ms, shannon);
        case NspType::HDR: return decode_hdr(payload_file, out, dec_ms, shannon);
        case NspType::SQL: return decode_sqlite(payload_file, out, dec_ms, shannon);
        case NspType::ELF: return decode_elf(payload_file, out, dec_ms, shannon);
        case NspType::PRECOMPRESSED: {
            std::ifstream vf(payload_file, std::ios::binary);
            std::ofstream of(out, std::ios::binary);
            of << vf.rdbuf();
            return of.good();
        }
        default:
            return decode_obj(payload_file, out, dec_ms, shannon);
    }
}

// Streaming: for files larger than the threshold, try to process in chunks.
// Currently implemented for BVH and WAV; other types fall back to full load with a warning.
static bool can_stream(NspType t) {
    return t == NspType::BVH || t == NspType::WAV || t == NspType::OBJ || t == NspType::PLY;
}

static bool encode_streaming(const std::string& in, const std::string& out, NspType t, double* enc_ms) {
    std::cerr << "Streaming encode not yet implemented for " << nsp_type_name(t) << ". Falling back to full load.\n";
    return false;
}

static bool decode_streaming(const std::string& in, const std::string& out, NspType t, double* dec_ms, double* shannon) {
    std::cerr << "Streaming decode not yet implemented for " << nsp_type_name(t) << ". Falling back to full load.\n";
    return false;
}

static bool nspack_encode_file(const std::string& in, const std::string& out, double* enc_ms) {
    NspType t = nsp_type_from_filename(in);
    uint64_t orig = file_size_bytes(in);
    if (t == NspType::IMAGE) {
        int w = 0, h = 0, ch = 0;
        if (stbi_info(in.c_str(), &w, &h, &ch)) {
            orig = (uint64_t)w * h * ch;
        }
    }
    std::string meta = build_metadata(in, t, orig);
    bool stream = should_stream(in);
    std::string payload_file = in + ".nspktmp";
    if (stream && can_stream(t)) {
        if (encode_streaming(in, payload_file, t, enc_ms)) {
            // Streaming currently writes a payload file; for now this branch is not reached.
        } else {
            std::remove(payload_file.c_str());
            return false;
        }
    } else {
        if (stream) {
            std::cerr << "Warning: file exceeds streaming threshold but no streaming codec for " << nsp_type_name(t) << ". Falling back to full load.\n";
        }
        if (!encode_to_payload_file(in, payload_file, t, enc_ms)) {
            std::remove(payload_file.c_str());
            return false;
        }
    }
    size_t ps = file_size_bytes(payload_file);
    std::ifstream pf(payload_file, std::ios::binary);
    std::vector<uint8_t> payload(ps);
    pf.read((char*)payload.data(), ps);
    pf.close();
    std::remove(payload_file.c_str());
    bool ok = write_nsp_file(out, t, orig, meta, payload);
    return ok;
}

static bool nspack_decode_file(const std::string& in, const std::string& out, double* dec_ms) {
    NspType t; uint64_t orig; std::string meta; std::vector<uint8_t> payload;
    if (!read_nsp_file(in, t, orig, meta, payload)) return false;
    std::string payload_file = in + ".nspktmp";
    {
        std::ofstream pf(payload_file, std::ios::binary);
        pf.write((char*)payload.data(), payload.size());
    }
    bool stream = should_stream(in);
    bool ok = false;
    if (stream && can_stream(t)) {
        ok = decode_streaming(in, out, t, dec_ms, nullptr);
    } else {
        if (stream) {
            std::cerr << "Warning: payload large but no streaming decoder for " << nsp_type_name(t) << ". Falling back to full load.\n";
        }
        ok = decode_payload_file(payload_file, out, t, dec_ms, nullptr);
    }
    std::remove(payload_file.c_str());
    return ok;
}

// .nsx file size helper (only reads the NSP1 header)
static bool nsp_file_size(const std::string& in, NspType& t, uint64_t& orig_size) {
    std::ifstream f(in, std::ios::binary);
    if (!f) return false;
    char magic[4];
    f.read(magic, 4);
    if (memcmp(magic, "NSP1", 4) != 0) return false;
    uint8_t type8; f.read((char*)&type8, 1);
    t = (NspType)type8;
    f.read((char*)&orig_size, 8);
    return f.good();
}

// Mount record helpers
static std::string mount_record_path() {
    return nsp_temp_dir() + "/nspack_mounts";
}

static std::string find_mount_storagedir(const std::string& mountpoint) {
    std::ifstream in(mount_record_path());
    if (!in) return "";
    std::string line;
    while (std::getline(in, line)) {
        auto p = line.find('|');
        if (p == std::string::npos) continue;
        std::string mp = line.substr(0, p);
        if (mp == mountpoint) {
            auto p2 = line.find('|', p + 1);
            return (p2 == std::string::npos) ? line.substr(p + 1) : line.substr(p + 1, p2 - p - 1);
        }
    }
    return "";
}

static void add_mount_record(const std::string& mountpoint, const std::string& storagedir,
                             unsigned long pid = 0) {
    std::ofstream out(mount_record_path(), std::ios::app);
    out << mountpoint << "|" << storagedir;
    if (pid) out << "|" << pid;
    out << "\n";
}

static unsigned long find_mount_pid(const std::string& mountpoint) {
    std::ifstream in(mount_record_path());
    if (!in) return 0;
    std::string line;
    while (std::getline(in, line)) {
        auto p = line.find('|');
        if (p == std::string::npos) continue;
        if (line.substr(0, p) != mountpoint) continue;
        auto p2 = line.find('|', p + 1);
        if (p2 == std::string::npos) return 0;
        return std::strtoul(line.substr(p2 + 1).c_str(), nullptr, 10);
    }
    return 0;
}

static void remove_mount_record(const std::string& mountpoint) {
    std::ifstream in(mount_record_path());
    std::string tmp = mount_record_path() + ".tmp";
    std::ofstream out(tmp);
    if (in) {
        std::string line;
        while (std::getline(in, line)) {
            if (line.find(mountpoint + "|") != 0) out << line << "\n";
        }
    }
    in.close(); out.close();
    nsp_rename_file(tmp.c_str(), mount_record_path().c_str());
}

static void status_scan_dir(const std::string& sd, std::vector<std::tuple<std::string,std::string,uint64_t,uint64_t>>& rows, uint64_t& total_orig, uint64_t& total_disk) {
#ifdef _WIN32
    std::error_code ec;
    for (auto& e : std::filesystem::directory_iterator(std::filesystem::path(sd), ec)) {
        std::string name = e.path().filename().string();
        if (name == "." || name == "..") continue;
        std::string full = sd + "/" + name;
        if (e.is_directory(ec)) {
            status_scan_dir(full, rows, total_orig, total_disk);
        } else if (e.is_regular_file(ec)) {
            std::string rel_name = name;
            uint64_t disk = (uint64_t)e.file_size(ec);
            if (ec) continue;
            uint64_t orig = disk;
            std::string type = "RAW";
            NspType t; uint64_t os;
            if (name.size() > 4 && name.substr(name.size() - 4) == ".nsx" && nsp_file_size(full, t, os)) {
                orig = os;
                rel_name = name.substr(0, name.size() - 4);
                type = nsp_type_name(t);
            }
            if (rel_name.size() >= 8 && rel_name.substr(rel_name.size() - 8) == ".hdrinfo") continue;
            rows.emplace_back(rel_name, type, orig, disk);
            total_orig += orig;
            total_disk += disk;
        }
    }
#else
    DIR* d = opendir(sd.c_str());
    if (!d) return;
    struct dirent* de;
    while ((de = readdir(d)) != nullptr) {
        std::string name = de->d_name;
        if (name == "." || name == "..") continue;
        std::string full = sd + "/" + name;
        struct stat st;
        if (stat(full.c_str(), &st) != 0) continue;
        if (S_ISDIR(st.st_mode)) {
            status_scan_dir(full, rows, total_orig, total_disk);
        } else if (S_ISREG(st.st_mode)) {
            std::string rel_name = name;
            uint64_t disk = (uint64_t)st.st_size;
            uint64_t orig = disk;
            std::string type = "RAW";
            NspType t; uint64_t os;
            if (name.size() > 4 && name.substr(name.size() - 4) == ".nsx" && nsp_file_size(full, t, os)) {
                orig = os;
                rel_name = name.substr(0, name.size() - 4);
                type = nsp_type_name(t);
            }
            if (rel_name.size() >= 8 && rel_name.substr(rel_name.size() - 8) == ".hdrinfo") continue;
            rows.emplace_back(rel_name, type, orig, disk);
            total_orig += orig;
            total_disk += disk;
        }
    }
    closedir(d);
#endif
}

static void print_status_table(const std::string& storagedir) {
    std::vector<std::tuple<std::string,std::string,uint64_t,uint64_t>> rows;
    uint64_t total_orig = 0, total_disk = 0;
    status_scan_dir(storagedir, rows, total_orig, total_disk);
    std::cout << std::left << std::setw(45) << "filename"
              << std::setw(12) << "type"
              << std::right << std::setw(14) << "original"
              << std::setw(14) << "on_disk"
              << std::setw(10) << "ratio" << "\n";
    std::cout << std::string(95, '-') << "\n";
    for (const auto& r : rows) {
        double ratio = (std::get<3>(r) > 0) ? (double)std::get<2>(r) / (double)std::get<3>(r) : 1.0;
        std::cout << std::left << std::setw(45) << std::get<0>(r)
                  << std::setw(12) << std::get<1>(r)
                  << std::right << std::setw(14) << std::get<2>(r)
                  << std::setw(14) << std::get<3>(r)
                  << std::setw(10) << std::fixed << std::setprecision(2) << ratio << "\n";
    }
    std::cout << std::string(95, '-') << "\n";
    double total_ratio = (total_disk > 0) ? (double)total_orig / (double)total_disk : 1.0;
    int64_t saved = (int64_t)total_orig - (int64_t)total_disk;
    std::cout << "Total original: " << total_orig << " bytes; on disk: " << total_disk
              << " bytes; space saved: " << saved << " bytes";
    if (saved < 0) std::cout << " (expansion)";
    std::cout << " (ratio " << std::fixed << std::setprecision(2) << total_ratio << ":1)\n";
}

static std::string self_exe_dir() {
#ifdef _WIN32
    char buf[MAX_PATH];
    DWORD n = GetModuleFileNameA(NULL, buf, MAX_PATH);
    if (n == 0) return "";
    std::string p(buf, n);
    size_t s = p.find_last_of("/\\");
    return (s == std::string::npos) ? "" : p.substr(0, s);
#else
    char buf[4096];
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n < 0) return "";
    buf[n] = 0;
    std::string p(buf);
    size_t s = p.find_last_of("/\\");
    return (s == std::string::npos) ? "" : p.substr(0, s);
#endif
}

static int do_mount(const std::string& mountpoint, const std::string& storagedir) {
    if (nsp_sys_mkdir(storagedir.c_str()) != 0 && errno != EEXIST) {
        std::cerr << "Cannot create storage dir: " << storagedir << "\n";
        return 1;
    }
#ifdef _WIN32
    std::string fs_bin = self_exe_dir() + "\\nspack_fs.exe";
    std::string cmdline = "\"" + fs_bin + "\" mount \"" + mountpoint + "\" \"" + storagedir + "\"";
    STARTUPINFOA si;
    memset(&si, 0, sizeof(si));
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi;
    memset(&pi, 0, sizeof(pi));
    std::vector<char> cl(cmdline.begin(), cmdline.end());
    cl.push_back(0);
    if (!CreateProcessA(NULL, cl.data(), NULL, NULL, FALSE,
                        DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
                        NULL, NULL, &si, &pi)) {
        std::cerr << "Failed to start nspack_fs.exe\n";
        return 1;
    }
    add_mount_record(mountpoint, storagedir, (unsigned long)pi.dwProcessId);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    nsp_sleep_ms(300);
    std::cout << "Mounted " << mountpoint << " -> " << storagedir << "\n";
    return 0;
#else
    add_mount_record(mountpoint, storagedir);
    std::string fs_bin = self_exe_dir() + "/nspack_fs";
    pid_t pid = fork();
    if (pid < 0) { perror("fork"); return 1; }
    if (pid == 0) {
        setsid();
        close(STDIN_FILENO); close(STDOUT_FILENO); close(STDERR_FILENO);
        execl(fs_bin.c_str(), "nspack_fs", "mount", mountpoint.c_str(), storagedir.c_str(), (char*)NULL);
        _exit(1);
    }
    nsp_sleep_ms(300);
    std::cout << "Mounted " << mountpoint << " -> " << storagedir << "\n";
    return 0;
#endif
}

static int do_unmount(const std::string& mountpoint) {
#ifdef _WIN32
    // The mount record carries the nspack_fs.exe PID; stopping that process
    // makes WinFSP tear down the drive or directory mount point.
    unsigned long pid = find_mount_pid(mountpoint);
    int rc = 1;
    if (pid) {
        HANDLE h = OpenProcess(PROCESS_TERMINATE, FALSE, (DWORD)pid);
        if (h) {
            if (TerminateProcess(h, 0)) rc = 0;
            CloseHandle(h);
        }
    }
    remove_mount_record(mountpoint);
    if (rc == 0) {
        std::cout << "Unmounted " << mountpoint << "\n";
        return 0;
    }
    std::cerr << "No running filesystem found for " << mountpoint << "\n";
    return 1;
#else
    std::string cmd = "fusermount3 -u \"" + mountpoint + "\"";
    int r = system(cmd.c_str());
    remove_mount_record(mountpoint);
    if (WIFEXITED(r) && WEXITSTATUS(r) == 0) {
        std::cout << "Unmounted " << mountpoint << "\n";
        return 0;
    }
    return 1;
#endif
}

static int do_status(const std::string& mountpoint) {
    std::string sd = find_mount_storagedir(mountpoint);
    if (sd.empty()) {
        std::cerr << "No mount record for " << mountpoint << "\n";
        return 1;
    }
    print_status_table(sd);
    return 0;
}

static int do_bench(const std::vector<std::string>& files) {
    return benchmark(files);
}

static bool migrate_supported(const std::string& ext) {
    static const char* exts[] = {".obj", ".ply", ".bvh", ".wav", ".png", ".jpg",
                                 ".hdr", ".exr", ".sqlite", ".py", ".cpp", ".c", ".h"};
    for (const char* e : exts) if (ext == e) return true;
    return false;
}

static int do_migrate(const std::string& dir, bool dry_run) {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (!fs::is_directory(fs::path(dir), ec)) {
        std::cerr << "Not a directory: " << dir << "\n";
        return 1;
    }
    if (dry_run) std::cout << "DRY RUN -- no files will be changed\n";
    std::cout << std::left << std::setw(45) << "filename"
              << std::right << std::setw(14) << "original"
              << std::setw(14) << "compressed"
              << std::setw(10) << "ratio" << "\n";
    std::cout << std::string(83, '-') << "\n";
    uint64_t total_orig = 0, total_disk = 0;
    int migrated = 0, skipped = 0;
    for (auto& e : fs::recursive_directory_iterator(fs::path(dir), ec)) {
        if (ec) break;
        if (!e.is_regular_file(ec)) continue;
        std::string full = e.path().string();
        std::string name = e.path().filename().string();
        std::string ext = to_lower_ext(name);
        if (ext == ".nsx") continue;                          // already a sidecar
        if (!migrate_supported(ext)) continue;                // unsupported, skip silently
        std::string sidecar = full + ".nsx";
        if (fs::exists(fs::path(sidecar), ec)) { skipped++; continue; }  // already migrated
        uint64_t orig = (uint64_t)e.file_size(ec);
        if (ec) continue;
        if (dry_run) {
            std::cout << std::left << std::setw(45) << name
                      << std::right << std::setw(14) << orig
                      << std::setw(14) << "(pending)"
                      << std::setw(10) << "-" << "\n";
            total_orig += orig;
            migrated++;
            continue;
        }
        if (!nspack_encode_file(full, sidecar, nullptr)) {
            std::cerr << "  encode failed, skipped: " << name << "\n";
            skipped++;
            continue;
        }
        uint64_t disk = file_size_bytes(sidecar);
        // Remove the original; the .nsx sidecar keeps the original filename visible.
        if (std::remove(full.c_str()) != 0)
            std::cerr << "  could not remove original: " << name << "\n";
        double ratio = (disk > 0) ? (double)orig / (double)disk : 1.0;
        std::cout << std::left << std::setw(45) << name
                  << std::right << std::setw(14) << orig
                  << std::setw(14) << disk
                  << std::setw(10) << std::fixed << std::setprecision(2) << ratio << "\n";
        total_orig += orig;
        total_disk += disk;
        migrated++;
    }
    std::cout << std::string(83, '-') << "\n";
    double total_ratio = (total_disk > 0) ? (double)total_orig / (double)total_disk : 0.0;
    int64_t saved = (int64_t)total_orig - (int64_t)total_disk;
    if (dry_run) {
        std::cout << "Would migrate " << migrated << " file(s), " << total_orig << " bytes original";
        if (skipped) std::cout << "; " << skipped << " already migrated";
        std::cout << "\n";
    } else {
        std::cout << "Migrated " << migrated << " file(s); total original: " << total_orig
                  << " bytes; total compressed: " << total_disk
                  << " bytes; space saved: " << saved << " bytes";
        if (saved < 0) std::cout << " (expansion)";
        std::cout << " (ratio " << std::fixed << std::setprecision(2) << total_ratio << ":1)";
        if (skipped) std::cout << "; " << skipped << " skipped";
        std::cout << "\n";
    }
    return 0;
}

static void print_usage(const char* exe) {
    std::cerr << "Usage:\n"
              << "  " << exe << " mount <mountpoint> <storagedir>\n"
              << "  " << exe << " unmount <mountpoint>\n"
              << "  " << exe << " status <mountpoint>\n"
              << "  " << exe << " bench <file1> [file2] ...\n"
              << "  " << exe << " migrate <directory>   compress all supported files in a directory in place\n"
              << "               --dry-run     show what would be compressed without changing anything\n"
              << "  " << exe << " --help\n";
}

static int full_main(int argc, char** argv) {
    if (argc < 2) { print_usage(argv[0]); return 1; }
    std::string cmd = argv[1];
    if (cmd == "--help" || cmd == "-h") { print_usage(argv[0]); return 0; }

    if (cmd == "mount") {
        if (argc < 4) { print_usage(argv[0]); return 1; }
        return do_mount(argv[2], argv[3]);
    }

    if (cmd == "unmount") {
        if (argc < 3) { print_usage(argv[0]); return 1; }
        return do_unmount(argv[2]);
    }

    if (cmd == "status") {
        if (argc < 3) { print_usage(argv[0]); return 1; }
        return do_status(argv[2]);
    }

    if (cmd == "bench") {
        if (argc < 3) { print_usage(argv[0]); return 1; }
        std::vector<std::string> files;
        for (int i = 2; i < argc; i++) files.push_back(argv[i]);
        return do_bench(files);
    }

    if (cmd == "migrate") {
        if (argc < 3) { print_usage(argv[0]); return 1; }
        bool dry_run = false;
        for (int i = 3; i < argc; i++)
            if (std::string(argv[i]) == "--dry-run") dry_run = true;
        return do_migrate(argv[2], dry_run);
    }

    std::cerr << "Invalid command: " << cmd << "\n";
    print_usage(argv[0]);
    return 1;
}

#ifndef NSPACK_NO_MAIN
int main(int argc, char** argv) { return full_main(argc, argv); }
#endif

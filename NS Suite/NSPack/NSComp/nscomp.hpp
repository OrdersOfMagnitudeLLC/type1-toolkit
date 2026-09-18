// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#pragma once

#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include "lz4.h"
#include <zstd.h>

namespace nscomp {

enum class FieldType { FLOAT32, INT32, INT16, UINT8, UINT32 };

struct FieldDef {
    const char* name;
    FieldType type;
    size_t offset;  // byte offset in struct
    size_t count;   // number of elements (1 for scalar, 3 for vec3)
};

struct Schema {
    size_t struct_size;           // total bytes per record
    std::vector<FieldDef> fields;
};

class NSComp {
public:
    NSComp() = default;

    // Compress records using schema-aware pipeline
    std::vector<uint8_t> compress_records(const uint8_t* data, size_t n_records, const Schema& schema);

    // Decompress records back to original format
    bool decompress_records(const uint8_t* compressed, uint8_t* output, size_t n_records, const Schema& schema);

private:
    // Column extraction: AoS → SoA
    std::vector<std::vector<uint8_t>> extract_columns(const uint8_t* data, size_t n_records, const Schema& schema);

    // Per-type delta encoding
    void delta_encode_uint32(std::vector<uint8_t>& data);
    void delta_encode_uint16(std::vector<uint8_t>& data);
    void delta_encode_uint8(std::vector<uint8_t>& data);

    // Per-type delta decoding
    void delta_decode_uint32(std::vector<uint8_t>& data);
    void delta_decode_uint16(std::vector<uint8_t>& data);
    void delta_decode_uint8(std::vector<uint8_t>& data);

    // Zigzag encoding for signed integers
    void zigzag_encode_int32(std::vector<uint8_t>& data);
    void zigzag_decode_int32(std::vector<uint8_t>& data);
    void zigzag_encode_int16(std::vector<uint8_t>& data);
    void zigzag_decode_int16(std::vector<uint8_t>& data);

    // Byte plane split for float columns
    std::vector<std::vector<uint8_t>> plane_split_float(const std::vector<uint8_t>& data);

    // Byte plane unsplit for float columns
    std::vector<uint8_t> plane_unsplit_float(const std::vector<std::vector<uint8_t>>& planes);

    // SoA → AoS reconstruction
    void reconstruct_records(const std::vector<std::vector<uint8_t>>& columns, uint8_t* output, size_t n_records, const Schema& schema);
};

} // namespace nscomp

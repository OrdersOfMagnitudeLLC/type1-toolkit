// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#include "nscomp.hpp"
#include <stdexcept>

namespace nscomp {

std::vector<uint8_t> NSComp::compress_records(const uint8_t* data, size_t n_records, const Schema& schema) {
    if (n_records == 0 || data == nullptr) return {};

    // Step 1: Column extraction (AoS → SoA)
    auto columns = extract_columns(data, n_records, schema);

    // Step 2 & 3: Per-type delta encoding + plane split for floats
    std::vector<std::vector<uint8_t>> processed_columns;
    for (size_t i = 0; i < columns.size(); i++) {
        const auto& field = schema.fields[i];
        auto& col = columns[i];

        // Delta encode based on type
        switch (field.type) {
            case FieldType::FLOAT32:
            case FieldType::UINT32:
                delta_encode_uint32(col);
                break;
            case FieldType::INT32:
                delta_encode_uint32(col);
                zigzag_encode_int32(col);
                break;
            case FieldType::INT16:
                delta_encode_uint16(col);
                zigzag_encode_int16(col);
                break;
            case FieldType::UINT8:
                delta_encode_uint8(col);
                break;
        }

        // Plane split for float columns
        if (field.type == FieldType::FLOAT32) {
            auto planes = plane_split_float(col);
            processed_columns.insert(processed_columns.end(), planes.begin(), planes.end());
        } else {
            processed_columns.push_back(col);
        }
    }

    // Step 4: LZ4 compress each plane/column independently (parallel)
    std::vector<std::vector<uint8_t>> compressed_columns(processed_columns.size());
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < processed_columns.size(); i++) {
        const auto& col = processed_columns[i];
        int max_compressed_size = LZ4_compressBound(col.size());
        std::vector<uint8_t> compressed(max_compressed_size);
        int compressed_size = LZ4_compress_default(
            reinterpret_cast<const char*>(col.data()),
            reinterpret_cast<char*>(compressed.data()),
            col.size(),
            max_compressed_size
        );
        if (compressed_size <= 0) {
            throw std::runtime_error("LZ4 compression failed");
        }
        compressed.resize(compressed_size);
        compressed_columns[i] = compressed;
    }

    // Step 5: Pack output
    std::vector<uint8_t> output;
    output.reserve(8 + compressed_columns.size() * 8); // rough estimate

    // Header
    uint32_t n_rec = static_cast<uint32_t>(n_records);
    uint32_t n_cols = static_cast<uint32_t>(compressed_columns.size());
    output.insert(output.end(), reinterpret_cast<uint8_t*>(&n_rec), reinterpret_cast<uint8_t*>(&n_rec) + 4);
    output.insert(output.end(), reinterpret_cast<uint8_t*>(&n_cols), reinterpret_cast<uint8_t*>(&n_cols) + 4);

    // Per column data
    size_t comp_idx = 0;
    for (size_t i = 0; i < columns.size(); i++) {
        const auto& field = schema.fields[i];
        
        // For float columns, we have 4 planes
        if (field.type == FieldType::FLOAT32) {
            for (size_t p = 0; p < 4; p++) {
                size_t plane_orig_size = n_records;
                uint32_t orig_size = static_cast<uint32_t>(plane_orig_size);
                uint32_t comp_size = static_cast<uint32_t>(compressed_columns[comp_idx].size());
                
                output.insert(output.end(), reinterpret_cast<uint8_t*>(&orig_size), reinterpret_cast<uint8_t*>(&orig_size) + 4);
                output.insert(output.end(), reinterpret_cast<uint8_t*>(&comp_size), reinterpret_cast<uint8_t*>(&comp_size) + 4);
                output.insert(output.end(), compressed_columns[comp_idx].begin(), compressed_columns[comp_idx].end());
                comp_idx++;
            }
        } else {
            size_t original_size = field.count * n_records * (field.type == FieldType::UINT32 || field.type == FieldType::INT32 ? 4 :
                                                             field.type == FieldType::INT16 ? 2 : 1);
            uint32_t orig_size = static_cast<uint32_t>(original_size);
            uint32_t comp_size = static_cast<uint32_t>(compressed_columns[comp_idx].size());
            
            output.insert(output.end(), reinterpret_cast<uint8_t*>(&orig_size), reinterpret_cast<uint8_t*>(&orig_size) + 4);
            output.insert(output.end(), reinterpret_cast<uint8_t*>(&comp_size), reinterpret_cast<uint8_t*>(&comp_size) + 4);
            output.insert(output.end(), compressed_columns[comp_idx].begin(), compressed_columns[comp_idx].end());
            comp_idx++;
        }
    }

    return output;
}

bool NSComp::decompress_records(const uint8_t* compressed, uint8_t* output, size_t n_records, const Schema& schema) {
    if (n_records == 0 || compressed == nullptr || output == nullptr) return false;

    const uint8_t* ptr = compressed;

    // Read header
    uint32_t n_rec, n_cols;
    std::memcpy(&n_rec, ptr, 4); ptr += 4;
    std::memcpy(&n_cols, ptr, 4); ptr += 4;

    if (n_rec != n_records) return false;

    // Decompress each column
    std::vector<std::vector<uint8_t>> decompressed_columns;
    size_t col_idx = 0;

    for (const auto& field : schema.fields) {
        if (field.type == FieldType::FLOAT32) {
            // 4 planes for float
            std::vector<std::vector<uint8_t>> planes(4);
            for (size_t p = 0; p < 4; p++) {
                uint32_t orig_size, comp_size;
                std::memcpy(&orig_size, ptr, 4); ptr += 4;
                std::memcpy(&comp_size, ptr, 4); ptr += 4;

                planes[p].resize(orig_size);
                int result = LZ4_decompress_safe(
                    reinterpret_cast<const char*>(ptr),
                    reinterpret_cast<char*>(planes[p].data()),
                    comp_size,
                    orig_size
                );
                if (result < 0) return false;
                ptr += comp_size;
            }

            // Plane unsplit
            auto col = plane_unsplit_float(planes);
            delta_decode_uint32(col);
            decompressed_columns.push_back(col);
        } else {
            uint32_t orig_size, comp_size;
            std::memcpy(&orig_size, ptr, 4); ptr += 4;
            std::memcpy(&comp_size, ptr, 4); ptr += 4;

            std::vector<uint8_t> col(orig_size);
            int result = LZ4_decompress_safe(
                reinterpret_cast<const char*>(ptr),
                reinterpret_cast<char*>(col.data()),
                comp_size,
                orig_size
            );
            if (result < 0) return false;
            ptr += comp_size;

            // Delta decode based on type (reverse order: zigzag first, then delta)
            switch (field.type) {
                case FieldType::UINT32:
                    delta_decode_uint32(col);
                    break;
                case FieldType::INT32:
                    zigzag_decode_int32(col);
                    delta_decode_uint32(col);
                    break;
                case FieldType::INT16:
                    zigzag_decode_int16(col);
                    delta_decode_uint16(col);
                    break;
                case FieldType::UINT8:
                    delta_decode_uint8(col);
                    break;
                default:
                    break;
            }

            decompressed_columns.push_back(col);
        }
        col_idx++;
    }

    // Reconstruct records (SoA → AoS)
    reconstruct_records(decompressed_columns, output, n_records, schema);

    return true;
}

std::vector<std::vector<uint8_t>> NSComp::extract_columns(const uint8_t* data, size_t n_records, const Schema& schema) {
    std::vector<std::vector<uint8_t>> columns;

    for (const auto& field : schema.fields) {
        size_t elem_size = (field.type == FieldType::FLOAT32 || field.type == FieldType::UINT32 || field.type == FieldType::INT32) ? 4 :
                          (field.type == FieldType::INT16) ? 2 : 1;
        size_t col_size = field.count * n_records * elem_size;
        std::vector<uint8_t> col(col_size);

        for (size_t rec = 0; rec < n_records; rec++) {
            const uint8_t* rec_ptr = data + rec * schema.struct_size + field.offset;
            uint8_t* col_ptr = col.data() + rec * field.count * elem_size;
            std::memcpy(col_ptr, rec_ptr, field.count * elem_size);
        }

        columns.push_back(col);
    }

    return columns;
}

void NSComp::delta_encode_uint32(std::vector<uint8_t>& data) {
    if (data.size() < 4) return;
    size_t n = data.size() / 4;
    uint32_t* words = reinterpret_cast<uint32_t*>(data.data());
    for (size_t i = n - 1; i > 0; i--) {
        words[i] -= words[i-1];
    }
}

void NSComp::delta_encode_uint16(std::vector<uint8_t>& data) {
    if (data.size() < 2) return;
    size_t n = data.size() / 2;
    uint16_t* words = reinterpret_cast<uint16_t*>(data.data());
    for (size_t i = n - 1; i > 0; i--) {
        words[i] -= words[i-1];
    }
}

void NSComp::delta_encode_uint8(std::vector<uint8_t>& data) {
    if (data.size() < 1) return;
    for (size_t i = data.size() - 1; i > 0; i--) {
        data[i] -= data[i-1];
    }
}

void NSComp::delta_decode_uint32(std::vector<uint8_t>& data) {
    if (data.size() < 4) return;
    size_t n = data.size() / 4;
    uint32_t* words = reinterpret_cast<uint32_t*>(data.data());
    for (size_t i = 1; i < n; i++) {
        words[i] += words[i-1];
    }
}

void NSComp::delta_decode_uint16(std::vector<uint8_t>& data) {
    if (data.size() < 2) return;
    size_t n = data.size() / 2;
    uint16_t* words = reinterpret_cast<uint16_t*>(data.data());
    for (size_t i = 1; i < n; i++) {
        words[i] += words[i-1];
    }
}

void NSComp::delta_decode_uint8(std::vector<uint8_t>& data) {
    if (data.size() < 1) return;
    for (size_t i = 1; i < data.size(); i++) {
        data[i] += data[i-1];
    }
}

void NSComp::zigzag_encode_int32(std::vector<uint8_t>& data) {
    if (data.size() < 4) return;
    size_t n = data.size() / 4;
    int32_t* words = reinterpret_cast<int32_t*>(data.data());
    for (size_t i = 0; i < n; i++) {
        int32_t value = words[i];
        uint32_t encoded = static_cast<uint32_t>((value << 1) ^ (value >> 31));
        words[i] = static_cast<int32_t>(encoded);
    }
}

void NSComp::zigzag_decode_int32(std::vector<uint8_t>& data) {
    if (data.size() < 4) return;
    size_t n = data.size() / 4;
    uint32_t* words = reinterpret_cast<uint32_t*>(data.data());
    for (size_t i = 0; i < n; i++) {
        uint32_t encoded = words[i];
        int32_t value = static_cast<int32_t>((encoded >> 1) ^ -(encoded & 1));
        words[i] = static_cast<uint32_t>(value);
    }
}

void NSComp::zigzag_encode_int16(std::vector<uint8_t>& data) {
    if (data.size() < 2) return;
    size_t n = data.size() / 2;
    int16_t* words = reinterpret_cast<int16_t*>(data.data());
    for (size_t i = 0; i < n; i++) {
        int16_t value = words[i];
        uint16_t encoded = static_cast<uint16_t>((value << 1) ^ (value >> 15));
        words[i] = static_cast<int16_t>(encoded);
    }
}

void NSComp::zigzag_decode_int16(std::vector<uint8_t>& data) {
    if (data.size() < 2) return;
    size_t n = data.size() / 2;
    uint16_t* words = reinterpret_cast<uint16_t*>(data.data());
    for (size_t i = 0; i < n; i++) {
        uint16_t encoded = words[i];
        int16_t value = static_cast<int16_t>((encoded >> 1) ^ -(encoded & 1));
        words[i] = static_cast<uint16_t>(value);
    }
}

std::vector<std::vector<uint8_t>> NSComp::plane_split_float(const std::vector<uint8_t>& data) {
    size_t n = data.size();
    size_t n4 = n / 4;
    std::vector<std::vector<uint8_t>> planes(4);
    for (size_t p = 0; p < 4; p++) {
        planes[p].resize(n4);
    }

    for (size_t i = 0; i < n4; i++) {
        planes[0][i] = data[4*i+0];
        planes[1][i] = data[4*i+1];
        planes[2][i] = data[4*i+2];
        planes[3][i] = data[4*i+3];
    }

    return planes;
}

std::vector<uint8_t> NSComp::plane_unsplit_float(const std::vector<std::vector<uint8_t>>& planes) {
    if (planes.size() != 4) return {};
    size_t n4 = planes[0].size();
    std::vector<uint8_t> data(n4 * 4);

    for (size_t i = 0; i < n4; i++) {
        data[4*i+0] = planes[0][i];
        data[4*i+1] = planes[1][i];
        data[4*i+2] = planes[2][i];
        data[4*i+3] = planes[3][i];
    }

    return data;
}

void NSComp::reconstruct_records(const std::vector<std::vector<uint8_t>>& columns, uint8_t* output, size_t n_records, const Schema& schema) {
    // Initialize with original data (preserves padding)
    // We'll only overwrite the fields we have columns for
    
    for (size_t col_idx = 0; col_idx < columns.size(); col_idx++) {
        const auto& field = schema.fields[col_idx];
        const auto& col = columns[col_idx];
        size_t elem_size = (field.type == FieldType::FLOAT32 || field.type == FieldType::UINT32 || field.type == FieldType::INT32) ? 4 :
                          (field.type == FieldType::INT16) ? 2 : 1;

        for (size_t rec = 0; rec < n_records; rec++) {
            uint8_t* rec_ptr = output + rec * schema.struct_size + field.offset;
            const uint8_t* col_ptr = col.data() + rec * field.count * elem_size;
            std::memcpy(rec_ptr, col_ptr, field.count * elem_size);
        }
    }
}

} // namespace nscomp

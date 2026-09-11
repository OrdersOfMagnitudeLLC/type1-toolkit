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
#include "nscomp.hpp"

using json = nlohmann::json;

// Base64 encoding/decoding helpers
#include <openssl/evp.h>
#include <openssl/bio.h>
#include <openssl/buffer.h>

std::string base64_encode(const std::vector<uint8_t>& data) {
    BIO *bio, *b64;
    BUF_MEM *bufferPtr;

    b64 = BIO_new(BIO_f_base64());
    bio = BIO_new(BIO_s_mem());
    bio = BIO_push(b64, bio);

    BIO_set_flags(bio, BIO_FLAGS_BASE64_NO_NL);
    BIO_write(bio, data.data(), data.size());
    BIO_flush(bio);
    BIO_get_mem_ptr(bio, &bufferPtr);

    std::string result(bufferPtr->data, bufferPtr->length);
    BIO_free_all(bio);

    return result;
}

std::vector<uint8_t> base64_decode(const std::string& encoded) {
    BIO *bio, *b64;
    std::vector<uint8_t> result;

    b64 = BIO_new(BIO_f_base64());
    bio = BIO_new_mem_buf(encoded.data(), encoded.size());
    bio = BIO_push(b64, bio);

    BIO_set_flags(bio, BIO_FLAGS_BASE64_NO_NL);

    int bytes;
    uint8_t buffer[1024];
    while ((bytes = BIO_read(bio, buffer, sizeof(buffer))) > 0) {
        result.insert(result.end(), buffer, buffer + bytes);
    }

    BIO_free_all(bio);
    return result;
}

// Convert string to FieldType enum
nscomp::FieldType parse_field_type(const std::string& type_str) {
    if (type_str == "FLOAT32") return nscomp::FieldType::FLOAT32;
    if (type_str == "INT32") return nscomp::FieldType::INT32;
    if (type_str == "INT16") return nscomp::FieldType::INT16;
    if (type_str == "UINT8") return nscomp::FieldType::UINT8;
    if (type_str == "UINT32") return nscomp::FieldType::UINT32;
    throw std::runtime_error("Unknown field type: " + type_str);
}

// Parse schema from JSON
nscomp::Schema parse_schema(const json& schema_json) {
    nscomp::Schema schema;
    schema.struct_size = schema_json["struct_size"];
    
    for (const auto& field_json : schema_json["fields"]) {
        nscomp::FieldDef field;
        // Note: We need to store the name, but FieldDef uses const char*
        // For simplicity, we'll use a workaround with static storage
        // In production, you'd want a more robust solution
        static std::vector<std::string> name_storage;
        name_storage.push_back(field_json["name"]);
        field.name = name_storage.back().c_str();
        field.type = parse_field_type(field_json["type"]);
        field.offset = field_json["offset"];
        field.count = field_json["count"];
        schema.fields.push_back(field);
    }
    
    return schema;
}

int main() {
    try {
        // Read entire stdin
        std::string input((std::istreambuf_iterator<char>(std::cin)),
                          std::istreambuf_iterator<char>());

        json request = json::parse(input);
        std::string action = request["action"];
        std::string data_b64 = request["data"];
        size_t n_records = request["n_records"];
        
        if (!request.contains("schema")) {
            json error = {{"error", "Schema is required"}};
            std::cout << error.dump() << std::endl;
            return 1;
        }
        
        nscomp::Schema schema = parse_schema(request["schema"]);
        std::vector<uint8_t> input_data = base64_decode(data_b64);
        size_t original_size = input_data.size();

        auto start = std::chrono::high_resolution_clock::now();

        std::vector<uint8_t> output_data;
        nscomp::NSComp compressor;

        if (action == "compress") {
            output_data = compressor.compress_records(input_data.data(), n_records, schema);

        } else if (action == "decompress") {
            size_t expected_output_size = n_records * schema.struct_size;
            output_data.resize(expected_output_size);
            
            bool success = compressor.decompress_records(
                input_data.data(), 
                output_data.data(), 
                n_records, 
                schema
            );
            
            if (!success) {
                json error = {{"error", "Decompression failed"}};
                std::cout << error.dump() << std::endl;
                return 1;
            }

        } else {
            json error = {{"error", "Unknown action"}};
            std::cout << error.dump() << std::endl;
            return 1;
        }

        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        size_t output_size = output_data.size();
        double ratio = (action == "compress") ? 
                       static_cast<double>(output_size) / original_size : 
                       static_cast<double>(original_size) / output_size;

        json response = {
            {"result", base64_encode(output_data)},
            {"original_size", original_size},
            {"output_size", output_size},
            {"ratio", ratio},
            {"time_ms", time_ms}
        };

        std::cout << response.dump() << std::endl;

    } catch (const std::exception& e) {
        json error = {{"error", e.what()}};
        std::cout << error.dump() << std::endl;
        return 1;
    }

    return 0;
}

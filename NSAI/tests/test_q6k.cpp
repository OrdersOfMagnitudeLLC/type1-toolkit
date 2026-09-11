#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <string>
#include "../include/gguf_parser.h"
#include "../vendor/ns_dequant.h"

int main() {
    const char* home = getenv("HOME");
    std::string base = home ? std::string(home) : ".";
    std::string model_path = base + "/NS/NSRun/models/qwen2.5-7b-instruct-q4_k_m.gguf";
    
    // Parse GGUF file
    GGUFParser parser;
    if (!parser.load(model_path.c_str())) {
        std::cerr << "ERROR: Failed to load GGUF file" << std::endl;
        return 1;
    }
    
    // Find tensor "blk.0.attn_v.weight"
    const GGUFTensor* tensor = nullptr;
    for (const auto& t : parser.tensors()) {
        if (t.name == "blk.0.attn_v.weight") {
            tensor = &t;
            break;
        }
    }
    
    if (!tensor) {
        std::cerr << "ERROR: Tensor blk.0.attn_v.weight not found" << std::endl;
        return 1;
    }
    
    std::cout << "Found tensor: " << tensor->name << std::endl;
    std::cout << "Tensor type: " << static_cast<int>(tensor->type) << std::endl;
    std::cout << "Shape: ";
    for (auto dim : tensor->shape) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;
    std::cout << "Data offset: " << tensor->offset << std::endl;
    
    if (tensor->type != GGMLType::Q6_K) {
        std::cerr << "ERROR: Expected Q6_K tensor, got type " << static_cast<int>(tensor->type) << std::endl;
        return 1;
    }
    
    // Calculate total elements
    uint64_t total_elements = 1;
    for (auto dim : tensor->shape) {
        total_elements *= dim;
    }
    std::cout << "Total elements: " << total_elements << std::endl;
    
    // Get pointer to tensor data
    const uint8_t* data_ptr = parser.data_ptr();
    const uint8_t* tensor_data = data_ptr + tensor->offset;
    
    // Calculate number of blocks
    const int64_t k = total_elements;
    const int64_t nb = k / QK_K;
    std::cout << "Number of blocks: " << nb << std::endl;
    
    // Allocate output buffer
    float* output = new float[total_elements];
    
    // Dequantize
    dequantize_row_q6_K(reinterpret_cast<const block_q6_K*>(tensor_data), output, k);
    
    // Write output to file
    std::ofstream out("/tmp/test_q6k.txt");
    if (!out) {
        std::cerr << "ERROR: Failed to open output file" << std::endl;
        delete[] output;
        return 1;
    }
    
    for (uint64_t i = 0; i < total_elements; i++) {
        out << output[i] << "\n";
    }
    
    out.close();
    delete[] output;
    
    std::cout << "Wrote test output to /tmp/test_q6k.txt" << std::endl;
    
    return 0;
}

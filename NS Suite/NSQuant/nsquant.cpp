// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#include "gguf_parser.h"
#include "tokenizer.h"
#include "ns_q4k_quant.h"
#include "ns_gguf_quant.h"

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <random>
#include <omp.h>
#include <chrono>
#include <regex>
#include <numeric>

#include <cassert>

static const size_t CLUSTER_SIZE = 256;
static const float HOT_THR = 0.8f;
static const float COLD_THR = 0.1f;

// --dequant-input: treat Q4_K/Q6_K/Q8_0 input tensors as dequantizable to float
static bool g_dequant_input = false;

static bool is_dequant_type(GGMLType ty) {
    return ty == GGMLType::Q4_K || ty == GGMLType::Q6_K || ty == GGMLType::Q8_0;
}

// Byte offset for a given number of elements, per on-disk type.
// Quantized types are block-packed: caller must keep element offsets block-aligned.
static uint64_t elem_offset_bytes(GGMLType ty, size_t elems) {
    switch (ty) {
        case GGMLType::F16:  return (uint64_t)elems * sizeof(uint16_t);
        case GGMLType::Q4_K: return (uint64_t)(elems / QK_K) * sizeof(block_q4_K);
        case GGMLType::Q6_K: return (uint64_t)(elems / QK_K) * sizeof(block_q6_K);
        case GGMLType::Q8_0: return (uint64_t)(elems / QK8_0) * sizeof(block_q8_0);
        default:             return (uint64_t)elems * sizeof(float);
    }
}
struct ClusterOut {
    float activation_freq;
    uint8_t quant_bits;
};

struct TensorOut {
    char name[64];
    uint32_t n_clusters;
    uint32_t quant_type;   // GGMLType value written to the GGUF tensor descriptor
    uint64_t data_offset;  // offset within the temp data stream
    uint64_t data_bytes;
    std::vector<ClusterOut> clusters;
};

static size_t file_size(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return 0;
    return f.tellg();
}

static std::vector<std::string> read_prompts(const std::string& path, int want) {
    std::vector<std::string> out;
    std::ifstream f(path);
    if (f) {
        std::string line;
        while (out.size() < (size_t)want && std::getline(f, line)) {
            if (!line.empty()) out.push_back(line);
        }
    }
    if (out.empty()) {
        std::mt19937 rng(123);
        for (int i = 0; i < want; ++i) {
            out.push_back("prompt " + std::to_string(i) + " placeholder text");
        }
    }
    if (out.size() > (size_t)want) out.resize(want);
    return out;
}

static size_t numel(const GGUFTensor& t) {
    size_t n = 1;
    for (auto d : t.shape) n *= (size_t)d;
    return n;
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp  = (h & 0x7C00u) >> 10;
    uint32_t mant =  h & 0x03FFu;

    if (exp == 0) {
        if (mant == 0) { uint32_t v = sign; float f; std::memcpy(&f, &v, sizeof(f)); return f; }
        float f = (float)mant * (1.0f / 1024.0f) * (1.0f / 16384.0f);
        return (h & 0x8000u) ? -f : f;
    }
    if (exp == 31) {
        uint32_t v = sign | (0xFFu << 23) | (mant << 13);
        float f; std::memcpy(&f, &v, sizeof(f)); return f;
    }
    uint32_t v = sign | ((exp + 112) << 23) | (mant << 13);
    float f; std::memcpy(&f, &v, sizeof(f)); return f;
}

static bool read_tensor_floats(const GGUFParser& parser, const GGUFTensor& t, std::vector<float>& out) {
    size_t n = numel(t);
    out.resize(n);
    if (t.type == GGMLType::F32) {
        return parser.read_tensor(t, (uint8_t*)out.data(), n * sizeof(float));
    } else if (t.type == GGMLType::F16) {
        std::vector<uint16_t> raw(n);
        if (!parser.read_tensor(t, (uint8_t*)raw.data(), n * sizeof(uint16_t))) return false;
        for (size_t i = 0; i < n; ++i) out[i] = f16_to_f32(raw[i]);
        return true;
    } else if (g_dequant_input && t.type == GGMLType::Q4_K) {
        if (n % QK_K != 0) {
            std::cerr << "read_tensor_floats: " << t.name << " Q4_K not block-aligned (n=" << n << ")" << std::endl;
            return false;
        }
        std::vector<uint8_t> raw((n / QK_K) * sizeof(block_q4_K));
        if (!parser.read_tensor(t, raw.data(), raw.size())) return false;
        dequantize_row_q4_K((const block_q4_K*)raw.data(), out.data(), (int64_t)n);
        return true;
    } else if (g_dequant_input && t.type == GGMLType::Q6_K) {
        if (n % QK_K != 0) {
            std::cerr << "read_tensor_floats: " << t.name << " Q6_K not block-aligned (n=" << n << ")" << std::endl;
            return false;
        }
        std::vector<uint8_t> raw((n / QK_K) * sizeof(block_q6_K));
        if (!parser.read_tensor(t, raw.data(), raw.size())) return false;
        dequantize_row_q6_K((const block_q6_K*)raw.data(), out.data(), (int64_t)n);
        return true;
    } else if (g_dequant_input && t.type == GGMLType::Q8_0) {
        if (n % QK8_0 != 0) {
            std::cerr << "read_tensor_floats: " << t.name << " Q8_0 not block-aligned (n=" << n << ")" << std::endl;
            return false;
        }
        std::vector<uint8_t> raw((n / QK8_0) * sizeof(block_q8_0));
        if (!parser.read_tensor(t, raw.data(), raw.size())) return false;
        dequantize_row_q8_0((const block_q8_0*)raw.data(), out.data(), (int64_t)n);
        return true;
    }
    std::cerr << "read_tensor_floats: unsupported type for " << t.name << std::endl;
    return false;
}

static bool is_preserve(const std::string& name) {
    std::regex re_preserve("^(blk\\.\\d+\\.attn_(q|k|v)\\.(bias))$");
    return std::regex_match(name, re_preserve);
}

// Tensors that must stay F32 in GGUF output: norms and biases are 1D and
// llama.cpp expects them unquantized.
static bool is_f32_tensor(const std::string& name) {
    if (is_preserve(name)) return false;
    static const char* suffixes[] = {
        "attn_norm.weight", "ffn_norm.weight", "output_norm.weight", ".bias"
    };
    for (const char* s : suffixes) {
        size_t sl = strlen(s);
        if (name.size() >= sl && name.compare(name.size() - sl, sl, s) == 0)
            return true;
    }
    return false;
}

// Quantize a float buffer to a GGUF-standard block type. Returns bytes written.
static size_t quantize_to_type(const std::vector<float>& w, GGMLType ty, std::vector<uint8_t>& out) {
    size_t n = w.size();
    switch (ty) {
        case GGMLType::F32:
            out.resize(n * sizeof(float));
            std::memcpy(out.data(), w.data(), n * sizeof(float));
            return out.size();
        case GGMLType::F16: {
            out.resize(n * sizeof(uint16_t));
            uint16_t* h = (uint16_t*)out.data();
            for (size_t i = 0; i < n; ++i) h[i] = f32_to_fp16(w[i]);
            return out.size();
        }
        case GGMLType::Q8_0: {
            out.resize((n / QK8_0) * sizeof(block_q8_0));
            quantize_row_q8_0(w.data(), (block_q8_0*)out.data(), (int64_t)n);
            return out.size();
        }
        case GGMLType::Q4_K: {
            out.resize((n / QK_K) * sizeof(block_q4_K));
            quantize_row_q4_K(w.data(), (block_q4_K*)out.data(), (int64_t)n);
            return out.size();
        }
        case GGMLType::Q2_K: {
            out.resize((n / QK_K) * sizeof(block_q2_K));
            quantize_row_q2_K(w.data(), (block_q2_K*)out.data(), (int64_t)n);
            return out.size();
        }
        default:
            return 0;
    }
}

static void profile_tensor(const float* w, int n_rows, int n_cols,
                           const std::vector<std::vector<float>>& xs,
                           std::vector<int>& activations) {
    size_t total = (size_t)n_rows * (size_t)n_cols;
    size_t n_clusters = (total + CLUSTER_SIZE - 1) / CLUSTER_SIZE;
    activations.assign(n_clusters, 0);
    if (n_rows <= 0 || n_cols <= 0 || xs.empty()) return;

    #pragma omp parallel
    {
        std::vector<int> la(n_clusters, 0);
        std::vector<float> y(n_rows);
        std::vector<float> absy(n_rows);
        #pragma omp for nowait
        for (size_t pi = 0; pi < xs.size(); ++pi) {
            const std::vector<float>& x = xs[pi];
            for (int r = 0; r < n_rows; ++r) {
                float s = 0.0f;
                const float* wr = w + (size_t)r * n_cols;
                for (int c = 0; c < n_cols; ++c) s += wr[c] * x[c];
                y[r] = s;
            }
            // Per-prompt threshold: 80th percentile of |y| for THIS prompt
            for (int r = 0; r < n_rows; ++r) absy[r] = std::fabs(y[r]);
            std::nth_element(absy.begin(), absy.begin() + (size_t)(absy.size() * 0.8f), absy.end());
            float thr = absy[(size_t)(absy.size() * 0.8f)];
            if (thr <= 0.0f) thr = 1e-6f;
            for (int r = 0; r < n_rows; ++r) {
                if (std::fabs(y[r]) > thr) {
                    size_t f0 = (size_t)r * n_cols;
                    size_t f1 = f0 + n_cols - 1;
                    size_t c0 = f0 / CLUSTER_SIZE;
                    size_t c1 = f1 / CLUSTER_SIZE;
                    for (size_t ci = c0; ci <= c1; ++ci) {
                        if (ci < n_clusters) la[ci]++;
                    }
                }
            }
        }
        #pragma omp critical
        {
            for (size_t i = 0; i < n_clusters; ++i) activations[i] += la[i];
        }
    }
}

static std::vector<std::vector<float>> build_prompt_inputs(const BPETokenizer& tok,
                                                           const std::vector<std::string>& prompts,
                                                           int n_embd,
                                                           const std::vector<float>& token_embd,
                                                           int n_vocab) {
    std::vector<std::vector<float>> out;
    out.reserve(prompts.size());

    for (const auto& p : prompts) {
        std::vector<float> x(n_embd, 0.0f);
        std::vector<int32_t> ids;
        bool ok = false;
        if (tok.vocab_size() > 0) {
            ids = tok.encode(p);
            ok = !ids.empty();
        }
        if (ok && !token_embd.empty()) {
            const float* te = token_embd.data();
            for (int id : ids) {
                if (id >= 0 && id < n_vocab) {
                    const float* row = te + (size_t)id * n_embd;
                    for (int i = 0; i < n_embd; ++i) x[i] += row[i];
                }
            }
            for (int i = 0; i < n_embd; ++i) x[i] /= (float)ids.size();
        } else if (!token_embd.empty()) {
            // Use a stable random input seeded by prompt
            std::mt19937 rng((unsigned)std::hash<std::string>{}(p));
            std::normal_distribution<float> dist(0.0f, 0.02f);
            for (int i = 0; i < n_embd; ++i) x[i] = dist(rng);
        } else {
            std::mt19937 rng((unsigned)std::hash<std::string>{}(p));
            std::normal_distribution<float> dist(0.0f, 0.02f);
            for (int i = 0; i < n_embd; ++i) x[i] = dist(rng);
        }
        out.push_back(std::move(x));
    }
    return out;
}

int main(int argc, char** argv) {
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--dequant-input") g_dequant_input = true;
        else pos.push_back(a);
    }
    if (pos.size() < 2) {
        std::cerr << "Usage: " << argv[0] << " [--dequant-input] <input.gguf> <output.gguf> [n_prompts [prompts.txt]]" << std::endl;
        return 1;
    }
    const char* in_path = pos[0].c_str();
    const char* out_path = pos[1].c_str();
    int n_prompts = (pos.size() >= 3) ? std::atoi(pos[2].c_str()) : 500;
    if (n_prompts < 0) n_prompts = 0;
    std::string prompts_path = (pos.size() >= 4) ? pos[3] : "prompts.txt";

    auto t0 = std::chrono::steady_clock::now();

    GGUFParser parser;
    if (!parser.load_pread(in_path)) {
        std::cerr << "Failed to load " << in_path << std::endl;
        return 1;
    }

    BPETokenizer tok;
    tok.load(parser);

    const ModelParams& mp = parser.params();
    int n_embd = (int)mp.n_embd;
    if (n_embd == 0) n_embd = 64;

    // --dequant-input: reject tensor types we cannot dequantize (Q2_K, Q1/IQ*, etc.)
    if (g_dequant_input) {
        for (const auto& t : parser.tensors()) {
            if (t.type == GGMLType::F32 || t.type == GGMLType::F16 || is_dequant_type(t.type)) continue;
            std::cerr << "Error: --dequant-input cannot dequantize tensor '" << t.name
                      << "' (type " << (uint32_t)t.type << "). Supported input types: F32, F16, Q4_K, Q6_K, Q8_0."
                      << std::endl;
            return 1;
        }
        std::cout << "--dequant-input: dequantizing Q4_K/Q6_K/Q8_0 tensors to float before profiling" << std::endl;
    }

    bool any_float = false;
    for (const auto& t : parser.tensors()) {
        if (t.type == GGMLType::F32 || t.type == GGMLType::F16 ||
            (g_dequant_input && is_dequant_type(t.type))) { any_float = true; break; }
    }
    bool structural_q4 = !any_float;

    // Read token_embd into memory once for prompt profiling, then release it
    // before the per-tensor loop so only one tensor is in RAM at a time.
    const GGUFTensor* et = nullptr;
    for (const auto& t : parser.tensors()) {
        if (t.name == "token_embd.weight" && (t.type == GGMLType::F32 || t.type == GGMLType::F16 ||
            (g_dequant_input && is_dequant_type(t.type)))) { et = &t; break; }
    }
    std::vector<float> token_embd;
    int n_vocab = 0;
    if (et && et->shape.size() >= 2) {
        size_t n = numel(*et);
        (void)n;
        if (!read_tensor_floats(parser, *et, token_embd)) {
            std::cerr << "Failed to read token_embd" << std::endl;
            return 1;
        }
        n_vocab = (int)et->shape[0];
    }

    std::vector<std::vector<float>> prompt_inputs;
    if (!structural_q4 && n_prompts > 0) {
        auto prompts = read_prompts(prompts_path, n_prompts);
        prompt_inputs = build_prompt_inputs(tok, prompts, n_embd, token_embd, n_vocab);
        std::cout << "Profiled with " << prompt_inputs.size() << " prompt samples" << std::endl;
    }

    // Token embedding is no longer needed after prompt inputs are built
    token_embd.clear();
    token_embd.shrink_to_fit();

    // RAM budget pass: largest float dequant buffer, current + next + 20% headroom
    size_t largest_bytes = 0;
    for (const auto& t : parser.tensors()) {
        size_t b = numel(t) * 4; // F32 dequant estimate
        if (b > largest_bytes) largest_bytes = b;
    }
    size_t working_budget = (size_t)(largest_bytes * 2.0 * 1.20);
    size_t chunk_floats = working_budget / 2 / sizeof(float);
    chunk_floats = (chunk_floats / 256) * 256;
    assert(chunk_floats >= 256);
    std::cout << "RAM budget: " << (working_budget / (1024 * 1024)) << " MB" << std::endl;

    // Stream quantized tensor data to a temporary file; only one tensor is held in RAM
    std::string data_tmp_path = std::string(out_path) + ".data.tmp";
    FILE* data_tmp = fopen(data_tmp_path.c_str(), "w+b");
    if (!data_tmp) {
        std::cerr << "Failed to open temporary data file " << data_tmp_path << std::endl;
        return 1;
    }

    std::vector<TensorOut> outs;
    outs.reserve(parser.tensors().size());

    size_t total_hot = 0, total_warm = 0, total_cold = 0;
    size_t n_tensors = parser.tensors().size();

    for (size_t ti = 0; ti < n_tensors; ++ti) {
        const GGUFTensor& t = parser.tensors()[ti];
        if ((ti % 10) == 0) {
            std::cout << "Processing [" << (ti + 1) << "/" << n_tensors << "] " << t.name << std::endl;
        }

        TensorOut to;
        std::memset(to.name, 0, sizeof(to.name));
        std::strncpy(to.name, t.name.c_str(), sizeof(to.name) - 1);
        to.n_clusters = 0;
        to.quant_type = 0;
        to.data_offset = 0;
        to.data_bytes = 0;

        bool float_path = (t.type == GGMLType::F32 || t.type == GGMLType::F16) ||
                          (g_dequant_input && is_dequant_type(t.type));
        if (float_path) {
            size_t n_full = numel(t);
            size_t nc = (n_full + CLUSTER_SIZE - 1) / CLUSTER_SIZE;

            // Flatten shape; treat shape[0] as input columns and shape[1] as
            // output rows, matching the GGUF layout and the NSRun kernels.
            int n_rows = 1, n_cols = 1;
            if (t.shape.size() >= 2) {
                n_cols = (int)t.shape[0];
                n_rows = (int)t.shape[1];
            } else if (!t.shape.empty()) {
                n_rows = (int)t.shape[0];
                n_cols = 1;
            }

            bool preserve = is_preserve(t.name);
            bool f32_tensor = is_f32_tensor(t.name);
            bool is2d = (t.shape.size() >= 2);
            // K-quants need ne[0] % 256; Q8_0 needs ne[0] % 32
            bool kquant_ok = is2d && (n_cols % (int)QK_K == 0);
            bool q8_ok = is2d && (n_cols % (int)QK8_0 == 0);
            bool quantizable = is2d && !preserve && !f32_tensor && (kquant_ok || q8_ok);

            to.n_clusters = (uint32_t)nc;
            to.clusters.resize(to.n_clusters);

            // Build prompt inputs projected to n_cols once per tensor
            std::vector<std::vector<float>> xs;
            bool do_profile = quantizable && n_prompts > 0 && (int)prompt_inputs.size() == n_prompts && n_rows > 1 && n_cols > 1;
            if (do_profile) {
                for (const auto& p : prompt_inputs) {
                    int dim = p.size();
                    std::vector<float> x(n_cols, 0.0f);
                    if (dim == n_cols) x = p;
                    else {
                        std::mt19937 rng((unsigned)(dim + n_cols));
                        std::normal_distribution<float> dist(0.0f, 0.02f);
                        for (float& v : x) v = dist(rng);
                    }
                    xs.push_back(std::move(x));
                }
            }

            // Decide chunking along rows to stay within the RAM budget
            size_t row_chunk = (size_t)n_rows;
            if (n_full > chunk_floats) {
                size_t g = std::gcd((size_t)n_cols, (size_t)CLUSTER_SIZE);
                size_t col_step = CLUSTER_SIZE / g;
                size_t rows_per_chunk = chunk_floats / (size_t)n_cols;
                size_t rc = (rows_per_chunk / col_step) * col_step;
                if (rc == 0) rc = col_step;
                row_chunk = rc;
                std::cout << "  chunking " << t.name << " rows " << n_rows
                          << " by " << row_chunk << " (cols " << n_cols << ")" << std::endl;
            }

            // Precision floor: attention and lm_head must never be Q2, and Q8 is
            // wasted because the loader requantizes to Q4_K, so keep them at Q4.
            int min_bits = 2;
            int max_bits = 8;
            if (t.name.find("token_embd") != std::string::npos ||
                t.name.find("blk.0.") != std::string::npos ||
                t.name.find(".attn_q.") != std::string::npos ||
                t.name.find(".attn_k.") != std::string::npos ||
                t.name.find(".attn_v.") != std::string::npos ||
                t.name.find(".attn_output.") != std::string::npos) {
                min_bits = 8;
                max_bits = 8;
            } else if (t.name.find("attn_q") != std::string::npos ||
                       t.name.find("attn_k") != std::string::npos ||
                       t.name.find("attn_v") != std::string::npos ||
                       t.name.find("attn_output") != std::string::npos ||
                       t.name.find("output.weight") != std::string::npos) {
                min_bits = 4;
                max_bits = 4;
            }

            // Pass A: dequantize each chunk and profile per-cluster activations
            for (int row_start = 0; row_start < n_rows; row_start += (int)row_chunk) {
                int row_end = std::min(row_start + (int)row_chunk, n_rows);
                int chunk_n_rows = row_end - row_start;
                size_t chunk_n = (size_t)chunk_n_rows * (size_t)n_cols;
                size_t chunk_offset_elems = (size_t)row_start * (size_t)n_cols;
                size_t cluster_offset = chunk_offset_elems / CLUSTER_SIZE;
                size_t chunk_nc = (chunk_n + CLUSTER_SIZE - 1) / CLUSTER_SIZE;

                GGUFTensor t_chunk = t;
                t_chunk.offset += elem_offset_bytes(t.type, chunk_offset_elems);
                t_chunk.shape = { (uint64_t)chunk_n_rows, (uint64_t)n_cols };

                std::vector<float> w;
                if (!read_tensor_floats(parser, t_chunk, w)) {
                    std::cerr << "Failed to read " << t.name << " chunk at row " << row_start << std::endl;
                    return 1;
                }

                if (do_profile) {
                    std::vector<int> acts;
                    profile_tensor(w.data(), chunk_n_rows, n_cols, xs, acts);
                    for (size_t i = 0; i < acts.size(); ++i) {
                        to.clusters[cluster_offset + i].activation_freq = (float)acts[i] / (float)n_prompts;
                    }
                } else {
                    for (size_t i = 0; i < chunk_nc; ++i) {
                        to.clusters[cluster_offset + i].activation_freq = 0.5f;
                    }
                }
            }

            // Per-cluster bit widths (distribution report) + dominant class → tensor type
            int n_cold = 0, n_warm = 0, n_hot = 0;
            if (quantizable) {
                for (size_t c = 0; c < nc; ++c) {
                    float f = to.clusters[c].activation_freq;
                    int bits;
                    if (f < COLD_THR) bits = 2;
                    else if (f > HOT_THR) bits = 8;
                    else bits = 4;
                    if (bits < min_bits) bits = min_bits;
                    if (bits > max_bits) bits = max_bits;
                    to.clusters[c].quant_bits = (uint8_t)bits;
                    if (bits == 2) { n_cold++; total_cold++; }
                    else if (bits == 8) { n_hot++; total_hot++; }
                    else { n_warm++; total_warm++; }
                }
            }

            GGMLType out_type;
            if (!quantizable) {
                // Keep F32 source tensors (biases, norms) as F32 — llama.cpp's
                // element-wise add path requires matching operand types/shapes.
                out_type = (t.type == GGMLType::F32 || f32_tensor) ? GGMLType::F32 : GGMLType::F16;
            } else {
                int bits = 4; // warm default
                if (n_hot >= n_warm && n_hot >= n_cold) bits = 8;
                else if (n_cold > n_warm && n_cold > n_hot) bits = 2;
                if (bits < min_bits) bits = min_bits;
                if (bits > max_bits) bits = max_bits;
                out_type = (bits == 8) ? GGMLType::Q8_0 : (bits == 2) ? GGMLType::Q2_K : GGMLType::Q4_K;
                if (out_type == GGMLType::Q8_0 && !q8_ok) out_type = GGMLType::F16;
                if ((out_type == GGMLType::Q4_K || out_type == GGMLType::Q2_K) && !kquant_ok)
                    out_type = q8_ok ? GGMLType::Q8_0 : GGMLType::F16;
            }
            to.quant_type = (uint32_t)out_type;

            // Pass B: dequantize each chunk again and quantize to the decided GGUF type
            to.data_offset = (uint64_t)ftell(data_tmp);
            uint64_t tensor_bytes = 0;
            for (int row_start = 0; row_start < n_rows; row_start += (int)row_chunk) {
                int row_end = std::min(row_start + (int)row_chunk, n_rows);
                int chunk_n_rows = row_end - row_start;
                size_t chunk_offset_elems = (size_t)row_start * (size_t)n_cols;

                GGUFTensor t_chunk = t;
                t_chunk.offset += elem_offset_bytes(t.type, chunk_offset_elems);
                t_chunk.shape = { (uint64_t)chunk_n_rows, (uint64_t)n_cols };

                std::vector<float> w;
                if (!read_tensor_floats(parser, t_chunk, w)) {
                    std::cerr << "Failed to read " << t.name << " chunk at row " << row_start << std::endl;
                    return 1;
                }
                std::vector<uint8_t> qbuf;
                size_t wrote = quantize_to_type(w, out_type, qbuf);
                if (wrote == 0 && !w.empty()) {
                    std::cerr << "Failed to quantize " << t.name << std::endl;
                    return 1;
                }
                if (wrote > 0) fwrite(qbuf.data(), 1, wrote, data_tmp);
                tensor_bytes += wrote;
            }
            to.data_bytes = tensor_bytes;

        } else {
            // Passthrough: copy raw tensor bytes, keep original GGUF type
            to.quant_type = (uint32_t)t.type;
            to.data_offset = (uint64_t)ftell(data_tmp);

            size_t data_len = 0;
            if (ti + 1 < parser.tensors().size()) {
                const GGUFTensor& next_t = parser.tensors()[ti + 1];
                data_len = (size_t)(next_t.offset - t.offset);
            } else {
                data_len = parser.full_file_size() - parser.data_offset() - t.offset;
            }

            std::vector<uint8_t> block(data_len);
            if (data_len > 0) {
                if (!parser.read_tensor(t, block.data(), data_len)) {
                    std::cerr << "Failed to read " << t.name << std::endl;
                    return 1;
                }
                fwrite(block.data(), 1, data_len, data_tmp);
            }
            to.data_bytes = data_len;
        }

        outs.push_back(std::move(to));
    }

    // Assemble GGUF: KV section copied verbatim from source, new tensor
    // descriptors, then tensor data already in final block layout in the
    // temp stream.
    fflush(data_tmp);
    fseek(data_tmp, 0, SEEK_SET);

    const auto& src_buf = parser.head_buffer();

    // Walk source GGUF header to find where the KV section ends
    size_t off = 0;
    off += 4; // magic
    off += 4; // version
    off += 8; // tensor_count
    uint64_t src_kv = *(const uint64_t*)(src_buf.data() + off); off += 8;
    for (uint64_t i = 0; i < src_kv; ++i) {
        uint64_t klen = *(const uint64_t*)(src_buf.data() + off); off += 8 + klen;
        uint32_t vt = *(const uint32_t*)(src_buf.data() + off); off += 4;
        switch (vt) {
            case 0: case 1: case 7: off += 1; break;
            case 2: case 3: off += 2; break;
            case 4: case 5: case 6: off += 4; break;
            case 10: case 11: case 12: off += 8; break;
            case 8: { uint64_t sl = *(const uint64_t*)(src_buf.data() + off); off += 8 + sl; } break;
            case 9: {
                uint32_t at = *(const uint32_t*)(src_buf.data() + off); off += 4;
                uint64_t al = *(const uint64_t*)(src_buf.data() + off); off += 8;
                if (at == 8) {
                    for (uint64_t j = 0; j < al; ++j) {
                        uint64_t sl = *(const uint64_t*)(src_buf.data() + off); off += 8 + sl;
                    }
                } else {
                    size_t es = (at <= 1 || at == 7) ? 1 : (at <= 3) ? 2 : (at <= 6) ? 4 : 8;
                    off += al * es;
                }
            } break;
            default: break;
        }
    }
    size_t kv_end = off;

    // Tensor descriptor section size
    size_t desc_size = 0;
    for (size_t i = 0; i < outs.size(); ++i) {
        const GGUFTensor& st = parser.tensors()[i];
        desc_size += 8 + st.name.size() + 4 + st.shape.size() * 8 + 4 + 8;
    }
    size_t header_size = kv_end + desc_size;
    size_t data_start = (header_size + 31) & ~size_t(31);

    // Per-tensor data offsets relative to data section start, 32-aligned
    std::vector<uint64_t> data_offs(outs.size());
    uint64_t cur = 0;
    for (size_t i = 0; i < outs.size(); ++i) {
        data_offs[i] = cur;
        cur += outs[i].data_bytes;
        cur = (cur + 31) & ~uint64_t(31);
    }

    FILE* f = fopen(out_path, "wb");
    if (!f) {
        std::cerr << "Failed to open output " << out_path << std::endl;
        fclose(data_tmp);
        std::remove(data_tmp_path.c_str());
        return 1;
    }

    // KV section verbatim from source (magic, version, tensor_count, kv_count, KVs)
    fwrite(src_buf.data(), 1, kv_end, f);

    // Tensor descriptors
    for (size_t i = 0; i < outs.size(); ++i) {
        const GGUFTensor& st = parser.tensors()[i];
        uint64_t nl = st.name.size();
        fwrite(&nl, 8, 1, f);
        fwrite(st.name.data(), 1, nl, f);
        uint32_t nd = (uint32_t)st.shape.size();
        fwrite(&nd, 4, 1, f);
        for (auto dd : st.shape) { fwrite(&dd, 8, 1, f); }
        fwrite(&outs[i].quant_type, 4, 1, f);
        fwrite(&data_offs[i], 8, 1, f);
    }

    // Pad to data_start
    {
        long pos = ftell(f);
        long pad = (long)data_start - pos;
        if (pad > 0) {
            std::vector<uint8_t> zeros(pad, 0);
            fwrite(zeros.data(), 1, pad, f);
        }
    }

    // Tensor data: stream each tensor's span from temp, pad to 32
    const size_t COPY_BUF = 8 * 1024 * 1024;
    std::vector<uint8_t> copy_buf(COPY_BUF);
    for (size_t i = 0; i < outs.size(); ++i) {
        fseek(data_tmp, (long)outs[i].data_offset, SEEK_SET);
        uint64_t remaining = outs[i].data_bytes;
        while (remaining > 0) {
            size_t want = remaining < COPY_BUF ? (size_t)remaining : COPY_BUF;
            size_t got = fread(copy_buf.data(), 1, want, data_tmp);
            if (got == 0) break;
            fwrite(copy_buf.data(), 1, got, f);
            remaining -= got;
        }
        long p = ftell(f);
        long npad = (32 - (p % 32)) % 32;
        if (npad > 0) {
            std::vector<uint8_t> zeros(npad, 0);
            fwrite(zeros.data(), 1, npad, f);
        }
    }

    fclose(data_tmp);
    fclose(f);
    std::remove(data_tmp_path.c_str());

    std::cout << "GGUF output written: " << out_path << std::endl;

    // Validation: check GGUF magic
    FILE* v = fopen(out_path, "rb");
    if (v) {
        char magic[4] = {0};
        if (fread(magic, 1, 4, v) == 4 && std::memcmp(magic, "GGUF", 4) == 0) {
            std::cout << "Validation: GGUF magic OK, tensors=" << outs.size() << std::endl;
        } else {
            std::cerr << "Validation failed: bad GGUF magic" << std::endl;
        }
        fclose(v);
    }

    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();

    size_t in_bytes = file_size(in_path);
    size_t out_bytes = file_size(out_path);
    std::cout << "Input:  " << in_bytes << " bytes" << std::endl;
    std::cout << "Output: " << out_bytes << " bytes" << std::endl;
    size_t total_clusters = total_hot + total_warm + total_cold;
    if (total_clusters > 0) {
        std::cout << "Cluster distribution:" << std::endl;
        std::cout << "  hot  (8-bit): " << total_hot  << " (" << (100.0 * total_hot  / total_clusters) << "%)" << std::endl;
        std::cout << "  warm (4-bit): " << total_warm << " (" << (100.0 * total_warm / total_clusters) << "%)" << std::endl;
        std::cout << "  cold (2-bit): " << total_cold << " (" << (100.0 * total_cold / total_clusters) << "%)" << std::endl;
    } else if (structural_q4) {
        std::cout << "Structural Q4 pass-through; no variable-rate quantization performed." << std::endl;
    }
    std::cout << "Compile time: " << sec << " s" << std::endl;
    std::cout << (structural_q4 ? "Note: input was not F32; structural Q4 pass-through mode." : "") << std::endl;

    return 0;
}

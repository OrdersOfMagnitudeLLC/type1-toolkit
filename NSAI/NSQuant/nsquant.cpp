#include "nsm.h"
#include "../include/gguf_parser.h"
#include "../include/tokenizer.h"
#include "../src/ns_q4k_quant.h"
#include "../src/ns_repack.h"

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
#include <atomic>

#include <zlib.h>
#include <cassert>

static const size_t CLUSTER_SIZE = 256;
static const float HOT_THR = 0.8f;
static const float COLD_THR = 0.1f;
std::atomic<uint64_t> n_d_zero(0);

struct ClusterOut {
    float activation_freq;
    uint8_t quant_bits;
    uint32_t data_offset;
    uint32_t data_bytes;
    float scale;
    uint8_t compressed;
    std::vector<uint8_t> packed;
};

struct TensorOut {
    char name[64];
    uint32_t n_clusters;
    uint32_t quant_type;
    uint64_t data_offset;
    uint64_t data_bytes;
    std::vector<ClusterOut> clusters;
    std::vector<uint8_t> blob;
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
    }
    std::cerr << "read_tensor_floats: unsupported type for " << t.name << std::endl;
    return false;
}

static void pack_q2(const std::vector<float>& w, const std::vector<int8_t>& q, std::vector<uint8_t>& out) {
    out.assign((w.size() + 3) / 4, 0);
    for (size_t i = 0; i < w.size(); ++i) {
        uint8_t n = (uint8_t)(q[i] + 1); // q in [-1,1], n in [0,2]
        size_t b = i / 4;
        size_t s = (i % 4) * 2;
        out[b] |= (n & 0x3) << s;
    }
}

static void pack_q4(const std::vector<float>& w, const std::vector<int8_t>& q, std::vector<uint8_t>& out) {
    out.assign((w.size() + 1) / 2, 0);
    for (size_t i = 0; i < w.size(); ++i) {
        uint8_t n = (uint8_t)(q[i] + 7); // q in [-7,7], n in [0,14]
        size_t b = i / 2;
        if (i % 2 == 0) out[b] |= (n & 0xF);
        else out[b] |= (n & 0xF) << 4;
    }
}

static void pack_q8(const std::vector<float>& w, const std::vector<int8_t>& q, std::vector<uint8_t>& out) {
    out.assign(w.size(), 0);
    for (size_t i = 0; i < w.size(); ++i) out[i] = (uint8_t)q[i];
}

static std::vector<uint8_t> zlib_compress(const std::vector<uint8_t>& in) {
    std::vector<uint8_t> out;
    if (in.empty()) return out;
    z_stream s;
    memset(&s, 0, sizeof(s));
    deflateInit(&s, Z_DEFAULT_COMPRESSION);
    s.avail_in = (uInt)in.size();
    s.next_in = (Bytef*)in.data();
    out.resize(deflateBound(&s, (uLong)in.size()));
    s.avail_out = (uInt)out.size();
    s.next_out = out.data();
    int r = deflate(&s, Z_FINISH);
    (void)r;
    deflateEnd(&s);
    out.resize(s.total_out);
    return out;
}

static std::vector<uint8_t> zlib_decompress(const std::vector<uint8_t>& in) {
    std::vector<uint8_t> out;
    if (in.empty()) return out;
    z_stream s;
    memset(&s, 0, sizeof(s));
    inflateInit(&s);
    s.avail_in = (uInt)in.size();
    s.next_in = (Bytef*)in.data();
    out.resize(in.size() * 4);
    s.avail_out = (uInt)out.size();
    s.next_out = out.data();
    int r = inflate(&s, Z_FINISH);
    while (r == Z_BUF_ERROR) {
        size_t old = out.size();
        out.resize(old * 2);
        s.next_out = out.data() + old;
        s.avail_out = (uInt)(out.size() - old);
        r = inflate(&s, Z_FINISH);
    }
    inflateEnd(&s);
    out.resize(s.total_out);
    return out;
}

static std::vector<int8_t> quant_floats(const std::vector<float>& w, int bits, float& scale) {
    float maxabs = 0.0f;
    for (float v : w) maxabs = std::max(maxabs, std::fabs(v));
    int max_q = 1;
    if (bits == 8) max_q = 127;
    else if (bits == 4) max_q = 7;
    scale = (maxabs > 0.0f) ? (maxabs / (float)max_q) : 1.0f;
    std::vector<int8_t> q(w.size());
    for (size_t i = 0; i < w.size(); ++i) {
        float v = w[i] / scale;
        v = std::round(v);
        v = std::max(-(float)max_q, std::min((float)max_q, v));
        q[i] = (int8_t)v;
    }
    return q;
}

static bool parse_layer_base(const std::string& name, std::string& base, int& layer) {
    std::regex re("^blk\\.(\\d+)\\.(.+)$");
    std::smatch m;
    if (std::regex_match(name, m, re)) {
        layer = std::stoi(m[1].str());
        base = "blk." + m[2].str();
        return true;
    }
    base = name;
    layer = 0;
    return false;
}

static std::string prev_name(const std::string& name) {
    std::regex re("^blk\\.(\\d+)\\.(.+)$");
    std::smatch m;
    if (std::regex_match(name, m, re)) {
        int l = std::stoi(m[1].str());
        if (l > 0) return "blk." + std::to_string(l - 1) + "." + m[2].str();
    }
    return "";
}

static bool is_preserve(const std::string& name) {
    std::regex re_preserve("^(blk\\.\\d+\\.attn_(q|k|v)\\.(bias))$");
    return std::regex_match(name, re_preserve);
}

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

static uint16_t fp32_to_fp16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint16_t sign = (uint16_t)((x >> 16) & 0x8000);
    int32_t exp = (x >> 23) & 0xff;
    uint32_t frac = x & 0x7fffff;
    if (exp == 0xff) return sign | 0x7c00 | (frac ? 0x200 : 0);
    exp -= 127 - 15;
    if (exp >= 31) return sign | 0x7c00;
    if (exp <= 0) {
        if (exp < -10) return sign;
        frac |= 0x800000;
        uint32_t shift = 14 - exp;
        uint32_t rounded = frac >> shift;
        if ((frac >> (shift - 1)) & 1) rounded++;
        return sign | (uint16_t)rounded;
    }
    return sign | (uint16_t)(exp << 10) | (uint16_t)(frac >> 13);
}

static void profile_tensor(const float* w, int n_rows, int n_cols,
                           const std::vector<std::vector<float>>& xs,
                           std::vector<int>& activations) {
    size_t total = (size_t)n_rows * (size_t)n_cols;
    size_t n_clusters = (total + CLUSTER_SIZE - 1) / CLUSTER_SIZE;
    activations.assign(n_clusters, 0);
    if (n_rows <= 0 || n_cols <= 0 || xs.empty()) return;

    std::vector<float> y(n_rows);
    // Threshold from first prompt (80th percentile of |y|)
    const std::vector<float>& x0 = xs[0];
    for (int r = 0; r < n_rows; ++r) {
        float s = 0.0f;
        const float* wr = w + (size_t)r * n_cols;
        for (int c = 0; c < n_cols; ++c) s += wr[c] * x0[c];
        y[r] = s;
    }
    std::vector<float> absy = y;
    for (float& v : absy) v = std::fabs(v);
    std::nth_element(absy.begin(), absy.begin() + (size_t)(absy.size() * 0.8f), absy.end());
    float thr = absy[(size_t)(absy.size() * 0.8f)];
    if (thr <= 0.0f) thr = 1e-6f;

    // Count x0 directly, then parallelize the remaining prompts
    for (int r = 0; r < n_rows; ++r) {
        if (std::fabs(y[r]) > thr) {
            size_t f0 = (size_t)r * n_cols;
            size_t f1 = f0 + n_cols - 1;
            size_t c0 = f0 / CLUSTER_SIZE;
            size_t c1 = f1 / CLUSTER_SIZE;
            for (size_t ci = c0; ci <= c1; ++ci) {
                if (ci < n_clusters) activations[ci]++;
            }
        }
    }

    #pragma omp parallel
    {
        std::vector<int> la(n_clusters, 0);
        #pragma omp for nowait
        for (size_t pi = 1; pi < xs.size(); ++pi) {
            const std::vector<float>& x = xs[pi];
            for (int r = 0; r < n_rows; ++r) {
                float s = 0.0f;
                const float* wr = w + (size_t)r * n_cols;
                for (int c = 0; c < n_cols; ++c) s += wr[c] * x[c];
                if (std::fabs(s) > thr) {
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
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.gguf> <output.nsm> [n_prompts [prompts.txt]]" << std::endl;
        return 1;
    }
    const char* in_path = argv[1];
    std::string eff_out_path = argv[2];
    int n_prompts = 500;
    std::string prompts_path = "prompts.txt";
    bool output_gguf = false;

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--output-gguf") {
            output_gguf = true;
        } else if (n_prompts == 500 && !arg.empty() && arg[0] != '-') {
            n_prompts = std::atoi(arg.c_str());
            if (i + 1 < argc) prompts_path = argv[i + 1];
        }
    }
    if (n_prompts < 0) n_prompts = 0;

    if (output_gguf) {
        size_t dot = eff_out_path.rfind('.');
        if (dot != std::string::npos) eff_out_path = eff_out_path.substr(0, dot);
        eff_out_path += ".gguf";
    }
    const char* out_path = eff_out_path.c_str();

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

    bool any_float = false;
    for (const auto& t : parser.tensors()) {
        if (t.type == GGMLType::F32 || t.type == GGMLType::F16) { any_float = true; break; }
    }
    bool structural_q4 = !any_float;

    // Read token_embd into memory once for prompt profiling, then release it
    // before the per-tensor loop so only one tensor is in RAM at a time.
    const GGUFTensor* et = nullptr;
    for (const auto& t : parser.tensors()) {
        if (t.name == "token_embd.weight" && (t.type == GGMLType::F32 || t.type == GGMLType::F16)) { et = &t; break; }
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
    // Only re-quantized tensors (F32/F16 except token_embd.weight) use the chunk buffer;
    // passthrough tensors (already-quantized or token_embd.weight) copy raw and need no budget.
    size_t largest_bytes = 0;
    for (const auto& t : parser.tensors()) {
        if (!((t.type == GGMLType::F32 || t.type == GGMLType::F16) && t.name != "token_embd.weight"))
            continue;
        size_t b = numel(t) * 4; // F32 dequant estimate
        if (b > largest_bytes) largest_bytes = b;
    }
    size_t working_budget = (size_t)(largest_bytes * 2.0 * 1.20);
    size_t chunk_floats = working_budget / 2 / sizeof(float);
    chunk_floats = (chunk_floats / 256) * 256;
    assert(chunk_floats >= 256);
    std::cout << "RAM budget: " << (working_budget / (1024 * 1024)) << " MB" << std::endl;

    // Map tensor names to their parser index for previous-layer lookups
    std::map<std::string, int> name_to_parser_idx;
    for (int i = 0; i < (int)parser.tensors().size(); ++i) {
        name_to_parser_idx[parser.tensors()[i].name] = i;
    }

    // Stream packed tensor data to a temporary file; only one tensor is held in RAM
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

        if ((t.type == GGMLType::F32 || t.type == GGMLType::F16) && t.name != "token_embd.weight") {
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
            bool no_panel = (t.name == "token_embd.weight");
            bool panel_mode = (t.shape.size() >= 2 && !preserve && !no_panel);
            size_t n_blocks_per_row = (n_cols + CLUSTER_SIZE - 1) / CLUSTER_SIZE;
            size_t n_row_groups = (n_rows + 7) / 8;
            size_t n_panels_total = n_row_groups * n_blocks_per_row;
            to.n_clusters = panel_mode ? (uint32_t)n_panels_total : (uint32_t)nc;
            to.clusters.resize(to.n_clusters);

            // Build prompt inputs projected to n_cols once per tensor
            std::vector<std::vector<float>> xs;
            bool do_profile = (n_prompts > 0 && (int)prompt_inputs.size() == n_prompts && n_rows > 1 && n_cols > 1);
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
                if (panel_mode) rc = (rc / 8) * 8;
                if (rc == 0 && panel_mode) rc = 8;
                row_chunk = rc;
                std::cout << "  chunking " << t.name << " rows " << n_rows
                          << " by " << row_chunk << " (cols " << n_cols << ")" << std::endl;
            }

            std::string base;
            int layer = 0;
            parse_layer_base(t.name, base, layer);
            std::string pn = prev_name(t.name);
            bool has_prev = !pn.empty() && name_to_parser_idx.count(pn);

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

            std::vector<float> prev_w;
            size_t prev_n = 0;
            uint64_t tensor_bytes = 0;

            for (int row_start = 0; row_start < n_rows; row_start += (int)row_chunk) {
                int row_end = std::min(row_start + (int)row_chunk, n_rows);
                int chunk_n_rows = row_end - row_start;
                size_t chunk_n = (size_t)chunk_n_rows * (size_t)n_cols;
                size_t chunk_offset_elems = (size_t)row_start * (size_t)n_cols;
                size_t cluster_offset;
                size_t chunk_nc;
                if (panel_mode) {
                    cluster_offset = ((size_t)row_start / 8) * n_blocks_per_row;
                    chunk_nc = ((chunk_n_rows + 7) / 8) * n_blocks_per_row;
                } else {
                    cluster_offset = chunk_offset_elems / CLUSTER_SIZE;
                    chunk_nc = (chunk_n + CLUSTER_SIZE - 1) / CLUSTER_SIZE;
                }

                GGUFTensor t_chunk = t;
                t_chunk.offset += chunk_offset_elems * (t.type == GGMLType::F16 ? sizeof(uint16_t) : sizeof(float));
                t_chunk.shape = { (uint64_t)chunk_n_rows, (uint64_t)n_cols };

                std::vector<float> w;
                if (!read_tensor_floats(parser, t_chunk, w)) {
                    std::cerr << "Failed to read " << t.name << " chunk at row " << row_start << std::endl;
                    return 1;
                }
                const float* w_ptr = w.data();

                // Profile for weight matrices only
                if (do_profile) {
                    std::vector<int> acts;
                    profile_tensor(w_ptr, chunk_n_rows, n_cols, xs, acts);
                    if (panel_mode) {
                        std::vector<int> panel_acts(chunk_nc, 0);
                        for (size_t i = 0; i < acts.size(); ++i) {
                            size_t r = i / n_blocks_per_row;
                            size_t b = i % n_blocks_per_row;
                            size_t p = (r / 8) * n_blocks_per_row + b;
                            panel_acts[p] = std::max(panel_acts[p], acts[i]);
                        }
                        for (size_t i = 0; i < chunk_nc; ++i) {
                            to.clusters[cluster_offset + i].activation_freq = (float)panel_acts[i] / (float)n_prompts;
                        }
                    } else {
                        for (size_t i = 0; i < acts.size(); ++i) {
                            to.clusters[cluster_offset + i].activation_freq = (float)acts[i] / (float)n_prompts;
                        }
                    }
                } else {
                    for (size_t i = 0; i < chunk_nc; ++i) {
                        to.clusters[cluster_offset + i].activation_freq = 0.5f;
                    }
                }

                // Decide per-cluster bit width using activation frequencies
                bool any_q2 = false;
                for (size_t c = 0; c < chunk_nc; ++c) {
                    size_t c_global = cluster_offset + c;
                    float f = to.clusters[c_global].activation_freq;
                    int bits;
                    if (!panel_mode && f < COLD_THR) bits = 2;
                    else if (f > HOT_THR) bits = 8;
                    else bits = 4;
                    if (bits < min_bits) bits = min_bits;
                    if (bits > max_bits) bits = max_bits;
                    to.clusters[c_global].quant_bits = (uint8_t)(bits == 8 ? 4 : bits);
                    if (bits == 2) total_cold++;
                    else if (bits == 8) total_hot++;
                    else total_warm++;
                    if (bits == 2) any_q2 = true;
                }

                // Load the previous layer's weights only if this chunk uses Q2 residuals
                if (any_q2 && has_prev && prev_w.empty()) {
                    int prev_pi = name_to_parser_idx[pn];
                    const GGUFTensor& prev_t = parser.tensors()[prev_pi];
                    if (!read_tensor_floats(parser, prev_t, prev_w)) {
                        std::cerr << "Failed to read prev " << prev_t.name << std::endl;
                        return 1;
                    }
                    prev_n = prev_w.size();
                }

                // Pack this chunk's clusters; store the compressed blob temporarily in the
                // ClusterOut, then flush it to the data stream and free the RAM
                #pragma omp parallel for schedule(dynamic)
                for (size_t c = 0; c < chunk_nc; ++c) {
                    size_t c_global = cluster_offset + c;
                    int bits = to.clusters[c_global].quant_bits;

                    std::vector<uint8_t> packed;

                    if (panel_mode) {
                        size_t panel_b = c % n_blocks_per_row;
                        size_t panel_g = c / n_blocks_per_row;
                        size_t row0 = panel_g * 8;
                        size_t rows_in_panel = std::min<size_t>(8, (size_t)chunk_n_rows - row0);
                        size_t col_off = (size_t)panel_b * CLUSTER_SIZE;

                        if (bits == 2) {
                            size_t panel_len = rows_in_panel * CLUSTER_SIZE;
                            std::vector<float> residual(panel_len, 0.0f);
                            for (size_t r = 0; r < rows_in_panel; ++r) {
                                size_t prev_pos = (size_t)c_global * (8 * CLUSTER_SIZE) + r * CLUSTER_SIZE;
                                float prev_f32[CLUSTER_SIZE];
                                std::memset(prev_f32, 0, sizeof(prev_f32));
                                if (has_prev && !prev_w.empty() && prev_pos < prev_n) {
                                    size_t block_len = std::min<size_t>(CLUSTER_SIZE, prev_n - prev_pos);
                                    block_q4_K prev_q4k;
                                    quantize_block_q4_K(prev_w.data() + prev_pos, &prev_q4k, block_len);
                                    if (*(const uint16_t*)&prev_q4k.d != 0) {
                                        dequantize_row_q4_K(&prev_q4k, prev_f32, (int64_t)CLUSTER_SIZE);
                                    }
                                }
                                for (size_t k = 0; k < CLUSTER_SIZE; ++k) {
                                    size_t cur_pos = (row0 + r) * (size_t)n_cols + col_off + k;
                                    float cur = (cur_pos < chunk_n) ? w_ptr[cur_pos] : 0.0f;
                                    residual[r * CLUSTER_SIZE + k] = cur - prev_f32[k];
                                }
                            }
                            float scale = 0.0f;
                            std::vector<int8_t> q = quant_floats(residual, bits, scale);
                            to.clusters[c_global].scale = scale;
                            pack_q2(residual, q, packed);
                        } else {
                            block_q4_K in[8];
                            std::memset(in, 0, sizeof(in));
                            for (size_t r = 0; r < rows_in_panel; ++r) {
                                size_t col_len = std::min<size_t>(CLUSTER_SIZE, (size_t)n_cols - col_off);
                                float v256[CLUSTER_SIZE];
                                std::memset(v256, 0, sizeof(v256));
                                for (size_t k = 0; k < col_len; ++k) {
                                    v256[k] = w_ptr[(row0 + r) * (size_t)n_cols + col_off + k];
                                }
                                quantize_row_q4_K_ref(v256, &in[r], (int64_t)CLUSTER_SIZE);
                                if (in[r].d == 0) std::memset(&in[r], 0, sizeof(in[r]));
                            }

                            if (output_gguf) {
                                packed.resize(rows_in_panel * sizeof(block_q4_K));
                                std::memcpy(packed.data(), in, rows_in_panel * sizeof(block_q4_K));
                            } else {
                                ns_q4_Kx8 panel;
                                repack_q4_K_row_panel_llama(in, (int)rows_in_panel, 1, &panel, 8);
                                packed.resize(sizeof(panel));
                                std::memcpy(packed.data(), &panel, sizeof(panel));
                            }
                            to.clusters[c_global].scale = 1.0f;
                        }
                    } else {
                        size_t c_start = c * CLUSTER_SIZE;
                        size_t c_end = std::min((c + 1) * CLUSTER_SIZE, chunk_n);
                        std::vector<float> values(w_ptr + c_start, w_ptr + c_end);

                        if (bits == 2 && has_prev && !prev_w.empty()) {
                            size_t prev_nc = (prev_n + CLUSTER_SIZE - 1) / CLUSTER_SIZE;
                            if (c_global < prev_nc) {
                                size_t off = c_global * CLUSTER_SIZE;
                                const float* prev_w_ptr = prev_w.data();
                                std::vector<float> prev_values(prev_w_ptr + off,
                                                               prev_w_ptr + std::min(off + CLUSTER_SIZE, prev_n));
                                if (prev_values.size() == values.size()) {
                                    block_q4_K prev_q4k;
                                    quantize_block_q4_K(prev_values.data(), &prev_q4k, prev_values.size());
                                    if (*(const uint16_t*)&prev_q4k.d != 0) {
                                        float prev_q4k_f32[CLUSTER_SIZE];
                                        dequantize_row_q4_K(&prev_q4k, prev_q4k_f32, (int64_t)CLUSTER_SIZE);
                                        for (size_t i = 0; i < values.size(); ++i)
                                            values[i] = values[i] - prev_q4k_f32[i];
                                    }
                                }
                            }
                        }

                        if (bits == 4 || bits == 8) {
                            block_q4_K b;
                            float v256[CLUSTER_SIZE];
                            std::memset(v256, 0, sizeof(v256));
                            std::memcpy(v256, values.data(), values.size() * sizeof(float));
                            quantize_row_q4_K_ref(v256, &b, (int64_t)CLUSTER_SIZE);
                            if (b.d == 0) {
                                std::memset(&b, 0, sizeof(b));
                                n_d_zero.fetch_add(1, std::memory_order_relaxed);
                            }
                            packed.resize(sizeof(b));
                            std::memcpy(packed.data(), &b, sizeof(b));
                            to.clusters[c_global].scale = 1.0f;
                        } else {
                            float scale = 0.0f;
                            std::vector<int8_t> q = quant_floats(values, bits, scale);
                            to.clusters[c_global].scale = scale;
                            pack_q2(values, q, packed);
                        }
                    }

                    to.clusters[c_global].packed = zlib_compress(packed);
                    to.clusters[c_global].compressed = 1;
                }

                for (size_t c = 0; c < chunk_nc; ++c) {
                    size_t c_global = cluster_offset + c;
                    to.clusters[c_global].data_offset = (uint32_t)tensor_bytes;
                    to.clusters[c_global].data_bytes = (uint32_t)to.clusters[c_global].packed.size();
                    if (to.clusters[c_global].data_bytes > 0) {
                        fwrite(to.clusters[c_global].packed.data(), 1, to.clusters[c_global].data_bytes, data_tmp);
                    }
                    tensor_bytes += to.clusters[c_global].data_bytes;
                    to.clusters[c_global].packed.clear();
                    to.clusters[c_global].packed.shrink_to_fit();
                }
            }
            to.data_bytes = tensor_bytes;
            to.quant_type = (uint32_t)GGMLType::Q4_K;
            if (std::string(to.name) == "blk.1.ffn_gate.weight") {
                size_t sum_cluster_data_bytes = 0;
                for (const auto& c : to.clusters) sum_cluster_data_bytes += c.data_bytes;
                std::cout << "[NSQ CHECK] " << to.name
                          << " n_clusters=" << to.n_clusters
                          << " sum_cluster_data_bytes=" << sum_cluster_data_bytes
                          << " tensor_data_bytes=" << to.data_bytes
                          << " expected_raw_q4k=" << (to.n_clusters * sizeof(block_q4_K))
                          << std::endl;
            }

        } else {
            // Already-quantized tensor: one raw passthrough cluster, no re-quantization
            to.n_clusters = 1;
            to.clusters.resize(1);
            to.clusters[0].activation_freq = 0.5f;

            size_t data_len = 0;
            if (ti + 1 < parser.tensors().size()) {
                const GGUFTensor& next_t = parser.tensors()[ti + 1];
                data_len = (size_t)(next_t.offset - t.offset);
            } else {
                data_len = parser.full_file_size() - parser.data_offset() - t.offset;
            }

            int bits = 8;
            switch (t.type) {
                case GGMLType::Q2_K: bits = 2; break;
                case GGMLType::Q3_K: bits = 3; break;
                case GGMLType::Q4_0:
                case GGMLType::Q4_1:
                case GGMLType::Q4_K: bits = 4; break;
                case GGMLType::Q5_0:
                case GGMLType::Q5_1:
                case GGMLType::Q5_K: bits = 5; break;
                case GGMLType::Q6_K: bits = 6; break;
                case GGMLType::Q8_0:
                case GGMLType::Q8_1: bits = 8; break;
                default: bits = 8; break;
            }
            to.clusters[0].quant_bits = (uint8_t)bits;
            to.clusters[0].scale = 0.0f;
            to.clusters[0].compressed = 0;

            std::vector<uint8_t> block(data_len);
            if (data_len > 0) {
                if (!parser.read_tensor(t, block.data(), data_len)) {
                    std::cerr << "Failed to read " << t.name << std::endl;
                    return 1;
                }
            }

            to.clusters[0].data_offset = 0;
            to.clusters[0].data_bytes = (uint32_t)data_len;
            to.data_bytes = data_len;
            to.quant_type = (uint32_t)t.type;

            if (data_len > 0) {
                fwrite(block.data(), 1, data_len, data_tmp);
            }
        }

        outs.push_back(std::move(to));
    }

    fclose(data_tmp);

    if (output_gguf) {
        // --- GGUF output mode ---
        const auto& src_buf = parser.head_buffer();

        // Walk source GGUF header to find where KV section ends
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

        // Compute tensor types for output
        std::vector<uint32_t> gguf_types(outs.size());
        for (size_t i = 0; i < outs.size(); ++i) {
            if (outs[i].quant_type == (uint32_t)GGMLType::Q4_K) {
                if (is_f32_tensor(parser.tensors()[i].name)) {
                    gguf_types[i] = (uint32_t)GGMLType::F32;
                } else if (is_preserve(parser.tensors()[i].name)) {
                    gguf_types[i] = (uint32_t)parser.tensors()[i].type; // keep F16 for bias
                } else {
                    gguf_types[i] = (uint32_t)GGMLType::Q4_K;
                }
            } else {
                gguf_types[i] = (uint32_t)parser.tensors()[i].type;
            }
        }

        // Compute tensor descriptor section size
        size_t desc_size = 0;
        for (size_t i = 0; i < outs.size(); ++i) {
            const GGUFTensor& st = parser.tensors()[i];
            desc_size += 8 + st.name.size() + 4 + st.shape.size() * 8 + 4 + 8;
        }
        size_t header_size = kv_end + desc_size;
        size_t data_start = (header_size + 31) & ~size_t(31);

        // Compute per-tensor data sizes and offsets (relative to data section start)
        std::vector<uint64_t> data_offs(outs.size());
        uint64_t cur = 0;
        for (size_t i = 0; i < outs.size(); ++i) {
            data_offs[i] = cur;
            const GGUFTensor& st = parser.tensors()[i];
            if (gguf_types[i] == (uint32_t)GGMLType::F32) {
                cur += numel(st) * 4;
            } else if (gguf_types[i] == (uint32_t)GGMLType::F16) {
                cur += numel(st) * 2;
            } else if (gguf_types[i] == (uint32_t)GGMLType::Q4_K) {
                cur += (numel(st) / 256) * sizeof(block_q4_K);
            } else {
                cur += outs[i].data_bytes;
            }
            cur = (cur + 31) & ~uint64_t(31);
        }

        // Compute temp file offsets per tensor
        std::vector<uint64_t> tmp_start(outs.size());
        uint64_t acc = 0;
        for (size_t i = 0; i < outs.size(); ++i) {
            tmp_start[i] = acc;
            acc += outs[i].data_bytes;
        }

        FILE* f = fopen(out_path, "wb");
        if (!f) {
            std::cerr << "Failed to open output " << out_path << std::endl;
            std::remove(data_tmp_path.c_str());
            return 1;
        }

        // Copy KV section from source (magic, version, tensor_count, kv_count, all KVs)
        fwrite(src_buf.data(), 1, kv_end, f);

        // Write tensor descriptors
        for (size_t i = 0; i < outs.size(); ++i) {
            const GGUFTensor& st = parser.tensors()[i];
            uint64_t nl = st.name.size();
            fwrite(&nl, 8, 1, f);
            fwrite(st.name.data(), 1, nl, f);
            uint32_t nd = (uint32_t)st.shape.size();
            fwrite(&nd, 4, 1, f);
            for (auto d : st.shape) { fwrite(&d, 8, 1, f); }
            fwrite(&gguf_types[i], 4, 1, f);
            fwrite(&data_offs[i], 8, 1, f);
        }

        // Pad to 32-byte alignment
        long pos = ftell(f);
        long pad = (long)data_start - pos;
        if (pad > 0) {
            std::vector<uint8_t> zeros(pad, 0);
            fwrite(zeros.data(), 1, pad, f);
        }

        // Write tensor data
        FILE* d = fopen(data_tmp_path.c_str(), "rb");
        if (!d) {
            std::cerr << "Failed to reopen temp data file" << std::endl;
            fclose(f);
            std::remove(data_tmp_path.c_str());
            return 1;
        }

        for (size_t i = 0; i < outs.size(); ++i) {
            const GGUFTensor& st = parser.tensors()[i];

            if (gguf_types[i] == (uint32_t)GGMLType::F32 && outs[i].quant_type == (uint32_t)GGMLType::Q4_K) {
                // Norm/bias tensor: dequantize Q4_K → F32 for ik_llama
                size_t n = numel(st);
                size_t n_blocks = n / 256;
                std::vector<uint8_t> qbuf(n_blocks * sizeof(block_q4_K), 0);

                for (uint32_t c = 0; c < outs[i].n_clusters; ++c) {
                    const auto& cl = outs[i].clusters[c];
                    fseek(d, (long)(tmp_start[i] + cl.data_offset), SEEK_SET);
                    std::vector<uint8_t> raw(cl.data_bytes);
                    size_t got = fread(raw.data(), 1, cl.data_bytes, d);
                    (void)got;
                    if (cl.compressed) raw = zlib_decompress(raw);
                    size_t boff = (size_t)c * sizeof(block_q4_K);
                    if (boff + sizeof(block_q4_K) <= qbuf.size() && raw.size() >= sizeof(block_q4_K)) {
                        memcpy(qbuf.data() + boff, raw.data(), sizeof(block_q4_K));
                    }
                }

                std::vector<float> fbuf(n, 0.0f);
                dequantize_row_q4_K((const block_q4_K*)qbuf.data(), fbuf.data(), (int64_t)n);
                fwrite(fbuf.data(), 1, n * 4, f);
            } else if (gguf_types[i] == (uint32_t)GGMLType::F16 && outs[i].quant_type == (uint32_t)GGMLType::Q4_K) {
                // Preserved bias tensor: dequantize Q4_K → F16 for ik_llama
                size_t n = numel(st);
                size_t n_blocks = n / 256;
                std::vector<uint8_t> qbuf(n_blocks * sizeof(block_q4_K), 0);

                for (uint32_t c = 0; c < outs[i].n_clusters; ++c) {
                    const auto& cl = outs[i].clusters[c];
                    fseek(d, (long)(tmp_start[i] + cl.data_offset), SEEK_SET);
                    std::vector<uint8_t> raw(cl.data_bytes);
                    size_t got = fread(raw.data(), 1, cl.data_bytes, d);
                    (void)got;
                    if (cl.compressed) raw = zlib_decompress(raw);
                    size_t boff = (size_t)c * sizeof(block_q4_K);
                    if (boff + sizeof(block_q4_K) <= qbuf.size() && raw.size() >= sizeof(block_q4_K)) {
                        memcpy(qbuf.data() + boff, raw.data(), sizeof(block_q4_K));
                    }
                }

                std::vector<float> fbuf(n, 0.0f);
                dequantize_row_q4_K((const block_q4_K*)qbuf.data(), fbuf.data(), (int64_t)n);
                std::vector<uint16_t> hbuf(n);
                for (size_t j = 0; j < n; ++j) {
                    hbuf[j] = fp32_to_fp16(fbuf[j]);
                }
                fwrite(hbuf.data(), 1, n * 2, f);
            } else if (gguf_types[i] == (uint32_t)GGMLType::Q4_K && outs[i].n_clusters > 1) {
                // Re-quantized Q4_K tensor: rearrange clusters to row-major
                size_t n = numel(st);
                bool is_2d = st.shape.size() >= 2;
                size_t n_cols = is_2d ? st.shape[0] : n;
                size_t n_rows = is_2d ? st.shape[1] : 1;
                size_t n_blocks_per_row = (n_cols + CLUSTER_SIZE - 1) / CLUSTER_SIZE;
                size_t n_row_groups = (n_rows + 7) / 8;
                size_t n_panels = n_row_groups * n_blocks_per_row;
                bool was_panel = (outs[i].n_clusters == n_panels);

                size_t total_blocks = n / 256;
                size_t buf_size = total_blocks * sizeof(block_q4_K);
                std::vector<uint8_t> tbuf(buf_size, 0);

                for (uint32_t c = 0; c < outs[i].n_clusters; ++c) {
                    const auto& cl = outs[i].clusters[c];
                    fseek(d, (long)(tmp_start[i] + cl.data_offset), SEEK_SET);
                    std::vector<uint8_t> raw(cl.data_bytes);
                    size_t got = fread(raw.data(), 1, cl.data_bytes, d);
                    (void)got;
                    if (cl.compressed) {
                        raw = zlib_decompress(raw);
                    }

                    if (was_panel) {
                        size_t rg = c / n_blocks_per_row;
                        size_t cb = c % n_blocks_per_row;
                        size_t rows_in_panel = std::min<size_t>(8, n_rows - rg * 8);
                        for (size_t r = 0; r < rows_in_panel; ++r) {
                            size_t block_idx = (rg * 8 + r) * n_blocks_per_row + cb;
                            size_t boff = block_idx * sizeof(block_q4_K);
                            if (boff + sizeof(block_q4_K) <= buf_size && (r + 1) * sizeof(block_q4_K) <= raw.size()) {
                                memcpy(tbuf.data() + boff, raw.data() + r * sizeof(block_q4_K), sizeof(block_q4_K));
                            }
                        }
                    } else {
                        size_t boff = (size_t)c * sizeof(block_q4_K);
                        if (boff + sizeof(block_q4_K) <= buf_size && raw.size() >= sizeof(block_q4_K)) {
                            memcpy(tbuf.data() + boff, raw.data(), sizeof(block_q4_K));
                        }
                    }
                }
                fwrite(tbuf.data(), 1, buf_size, f);
            } else {
                // Passthrough: copy raw data from temp file
                fseek(d, (long)tmp_start[i], SEEK_SET);
                std::vector<uint8_t> raw(outs[i].data_bytes);
                size_t got = fread(raw.data(), 1, outs[i].data_bytes, d);
                (void)got;
                fwrite(raw.data(), 1, outs[i].data_bytes, f);
            }

            // Pad to 32-byte alignment
            long p = ftell(f);
            long npad = (32 - (p % 32)) % 32;
            if (npad > 0) {
                std::vector<uint8_t> zeros(npad, 0);
                fwrite(zeros.data(), 1, npad, f);
            }
        }

        fclose(d);
        fclose(f);
        std::remove(data_tmp_path.c_str());

        std::cout << "GGUF output written: " << out_path << std::endl;
    } else {
    // Write final .nsm file: header, tensor table, cluster table, then packed data
    FILE* f = fopen(out_path, "wb+");
    if (!f) {
        std::cerr << "Failed to open output " << out_path << std::endl;
        std::remove(data_tmp_path.c_str());
        return 1;
    }

    NSMHeader hdr;
    hdr.magic = NSM_MAGIC;
    hdr.version = 2;
    hdr.n_layers = mp.n_layers;
    hdr.n_tensors = (uint32_t)outs.size();
    hdr.data_offset = 0;
    hdr.layer_table_offset = 0;
    hdr.gguf_metadata_offset = 0;
    hdr.gguf_metadata_bytes = 0;
    fwrite(&hdr, sizeof(hdr), 1, f);

    size_t cluster_map_start = sizeof(hdr) + outs.size() * sizeof(NSMTensor);
    uint64_t data_start = cluster_map_start;
    for (const auto& to : outs) data_start += to.n_clusters * sizeof(NSMCluster);

    uint64_t cursor = data_start;
    for (auto& to : outs) {
        to.data_offset = cursor;
        cursor += to.data_bytes;
    }

    hdr.data_offset = data_start;
    fseek(f, 0, SEEK_SET);
    fwrite(&hdr, sizeof(hdr), 1, f);

    for (const auto& to : outs) {
        NSMTensor nt;
        std::memset(&nt, 0, sizeof(nt));
        std::memcpy(nt.name, to.name, sizeof(nt.name));
        nt.n_clusters = to.n_clusters;
        nt.quant_type = to.quant_type;
        nt.data_offset = to.data_offset;
        nt.data_bytes = to.data_bytes;
        fwrite(&nt, sizeof(nt), 1, f);
    }

    for (const auto& to : outs) {
        for (const auto& c : to.clusters) {
            NSMCluster cl;
            cl.activation_freq = c.activation_freq;
            cl.quant_bits = c.quant_bits;
            cl.data_offset = c.data_offset;
            cl.data_bytes = c.data_bytes;
            cl.scale = c.scale;
            cl.compressed = c.compressed;
            fwrite(&cl, sizeof(cl), 1, f);
        }
    }

    FILE* d = fopen(data_tmp_path.c_str(), "rb");
    if (!d) {
        std::cerr << "Failed to reopen temporary data file " << data_tmp_path << std::endl;
        fclose(f);
        std::remove(data_tmp_path.c_str());
        return 1;
    }

    const size_t COPY_BUF = 8 * 1024 * 1024;
    std::vector<uint8_t> copy_buf(COPY_BUF);
    size_t got;
    while ((got = fread(copy_buf.data(), 1, COPY_BUF, d)) > 0) {
        if (fwrite(copy_buf.data(), 1, got, f) != got) {
            std::cerr << "Failed to write output data" << std::endl;
            break;
        }
    }
    fclose(d);
    std::remove(data_tmp_path.c_str());

    // Append embedded GGUF metadata+tensor-descriptor section (v2)
    size_t gguf_meta_offset = (size_t)ftell(f);
    const std::vector<uint8_t>& gguf_meta = parser.head_buffer();
    size_t gguf_meta_bytes = parser.head_data_offset(); // bytes up to the GGUF data section
    if (fwrite(gguf_meta.data(), 1, gguf_meta_bytes, f) != gguf_meta_bytes) {
        std::cerr << "Failed to write embedded GGUF metadata" << std::endl;
        fclose(f);
        std::remove(out_path);
        return 1;
    }

    // Patch header to v2 with metadata location
    fseek(f, 0, SEEK_SET);
    NSMHeader hdr_patch;
    if (fread(&hdr_patch, sizeof(hdr_patch), 1, f) == 1) {
        hdr_patch.version = 2;
        hdr_patch.gguf_metadata_offset = gguf_meta_offset;
        hdr_patch.gguf_metadata_bytes = gguf_meta_bytes;
        fseek(f, 0, SEEK_SET);
        fwrite(&hdr_patch, sizeof(hdr_patch), 1, f);
    }
    fclose(f);

    // Validation pass
    FILE* v = fopen(out_path, "rb");
    if (v) {
        NSMHeader vh;
        if (fread(&vh, sizeof(vh), 1, v) == 1 && vh.magic == NSM_MAGIC) {
            uint64_t total = 0;
            fseek(v, sizeof(NSMHeader), SEEK_SET);
            for (uint32_t i = 0; i < vh.n_tensors; ++i) {
                NSMTensor vt;
                if (fread(&vt, sizeof(vt), 1, v) != 1) break;
                total += vt.n_clusters;
            }
            std::cout << "Validation: magic OK, tensors=" << vh.n_tensors
                      << ", layers=" << vh.n_layers
                      << ", clusters=" << total << std::endl;
        } else {
            std::cerr << "Validation failed: bad header" << std::endl;
        }
        fclose(v);
    }
    } // end else (NSM write mode)

    uint64_t total_clusters_val = 0;
    for (const auto& to : outs) total_clusters_val += to.n_clusters;

    std::cout << "[NSQuant] d=0 clusters: " << n_d_zero.load()
              << " / " << total_clusters_val
              << " (" << (100.0 * (double)n_d_zero.load() / (double)total_clusters_val) << "%)" << std::endl;

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

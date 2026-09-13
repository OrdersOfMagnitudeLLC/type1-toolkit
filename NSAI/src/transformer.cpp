#include "transformer.h"
#include "nsm.h"
#include "nsm_loader.h"
#include "../vendor/ns_dequant.h"
#include "ns_gemm_q4k.h"
#include "ns_attend.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <cblas.h>
#include <immintrin.h>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unistd.h>
#include <sys/mman.h>
#include "hwy/highway.h"
#include "ns_quant_fused.h"
#include "ns_threadpool.h"

namespace hn = hwy::HWY_NAMESPACE;

extern "C" void openblas_set_num_threads(int num_threads);

// ---- Profiling instrumentation (CLOCK_MONOTONIC) ----
#include <time.h>
#include <atomic>

struct ProfileTimers {
    std::atomic<uint64_t> kv_write_ns{0};
    std::atomic<uint64_t> kv_read_ns{0};
    std::atomic<uint64_t> attention_ns{0};
    std::atomic<uint64_t> matmul_ns{0};
    std::atomic<uint64_t> layer_load_ns{0};
    std::atomic<uint64_t> sampling_ns{0};
    std::atomic<uint64_t> disk_preads{0};
    std::atomic<uint64_t> buffer_hits{0};
    std::atomic<uint64_t> layers_pread{0};
    std::atomic<uint64_t> layers_buffer{0};
    std::atomic<int> enabled{0};
};
ProfileTimers g_prof;

static inline int64_t prof_now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

struct ProfScope {
    int64_t t0;
    std::atomic<uint64_t>& c;
    bool on;
    ProfScope(std::atomic<uint64_t>& c_, bool on_) : t0(0), c(c_), on(on_) { if (on) t0 = prof_now_ns(); }
    ~ProfScope() { if (on) c.fetch_add(prof_now_ns() - t0, std::memory_order_relaxed); }
};

static ssize_t profiled_pread(int fd, void* buf, size_t count, off_t offset) {
    int64_t t0 = 0;
    bool on = g_prof.enabled.load(std::memory_order_relaxed);
    if (on) t0 = prof_now_ns();
    ssize_t r = ::pread(fd, buf, count, offset);
    if (on) {
        g_prof.layer_load_ns.fetch_add(prof_now_ns() - t0, std::memory_order_relaxed);
        g_prof.disk_preads.fetch_add(1, std::memory_order_relaxed);
        g_prof.layers_pread.fetch_add(1, std::memory_order_relaxed);
    }
    return r;
}

extern "C" void profile_add_sampling(uint64_t ns) {
    g_prof.sampling_ns.fetch_add(ns, std::memory_order_relaxed);
}

extern "C" void profile_print_report(double total_ms) {
    auto l = [](std::atomic<uint64_t>& a) -> double { return a.load(std::memory_order_relaxed) / 1e6; };
    double t_kv_write = l(g_prof.kv_write_ns);
    double t_kv_read  = l(g_prof.kv_read_ns);
    double t_attn     = l(g_prof.attention_ns);
    double t_matmul   = l(g_prof.matmul_ns);
    double t_load     = l(g_prof.layer_load_ns);
    double t_sampling = l(g_prof.sampling_ns);
    double measured   = t_kv_write + t_kv_read + t_attn + t_matmul + t_load + t_sampling;
    auto pct = [&](double v) { return total_ms > 0 ? (v / total_ms) * 100.0 : 0.0; };
    printf("\n=== Generate Profile Breakdown ===\n");
    printf("%-18s %10s %10s\n", "Section", "ms", "% wall");
    printf("%-18s %10.2f %10.1f\n", "KV write", t_kv_write, pct(t_kv_write));
    printf("%-18s %10.2f %10.1f\n", "KV read", t_kv_read, pct(t_kv_read));
    printf("%-18s %10.2f %10.1f\n", "Attention", t_attn, pct(t_attn));
    printf("%-18s %10.2f %10.1f\n", "Matmul", t_matmul, pct(t_matmul));
    printf("%-18s %10.2f %10.1f\n", "Layer load/pread", t_load, pct(t_load));
    printf("%-18s %10.2f %10.1f\n", "Sampling", t_sampling, pct(t_sampling));
    printf("%-18s %10.2f\n", "Sum (measured)", measured);
    printf("%-18s %10.2f\n", "Wall (measured)", total_ms);
    printf("Disk preads: %lu, buffer hits: %lu, layers from pread: %lu, layers from buffer: %lu\n",
           (unsigned long)g_prof.disk_preads.load(),
           (unsigned long)g_prof.buffer_hits.load(),
           (unsigned long)g_prof.layers_pread.load(),
           (unsigned long)g_prof.layers_buffer.load());
}

// Raw free helpers used by the per-layer NSM streaming path.
static const float* get_weight_f32_raw(const uint8_t* data, GGMLType type, const std::vector<uint64_t>& shape, std::vector<float>& buf) {
    if (type == GGMLType::F32) {
        size_t num_values = 1;
        for (auto d : shape) { num_values *= d; }
        buf.resize(num_values);
        std::memcpy(buf.data(), data, num_values * sizeof(float));
        return buf.data();
    } else if (type == GGMLType::Q4_K) {
        size_t num_values = 1;
        for (uint64_t dim : shape) { num_values *= dim; }
        if (num_values % QK_K != 0) {
            std::cerr << "get_weight_f32_raw: Q4_K tensor size not a multiple of " << QK_K << std::endl;
            return nullptr;
        }
        buf.resize(num_values);
        dequantize_row_q4_K((const block_q4_K*)data, buf.data(), (int)num_values);
        return buf.data();
    } else if (type == GGMLType::Q6_K) {
        size_t num_values = 1;
        for (uint64_t dim : shape) { num_values *= dim; }
        buf.resize(num_values);
        size_t num_blocks = (num_values + 255) / 256;
        for (size_t block = 0; block < num_blocks; ++block) {
            const uint8_t* block_data = data + block * 210;
            uint16_t d_f16 = *reinterpret_cast<const uint16_t*>(block_data + 208);
            float d = Transformer::f16_to_f32(d_f16);
            const int8_t* scales = reinterpret_cast<const int8_t*>(block_data + 192);
            const uint8_t* ql = block_data;
            const uint8_t* qh = block_data + 128;
            for (int i = 0; i < 256; ++i) {
                uint8_t ql_byte = ql[i / 2];
                uint8_t ql_bits = (i % 2 == 0) ? (ql_byte & 0x0F) : ((ql_byte >> 4) & 0x0F);
                uint8_t qh_byte = qh[i / 4];
                uint8_t qh_bits = (qh_byte >> (2 * (i % 4))) & 0x03;
                uint8_t q = ql_bits | (qh_bits << 4);
                int8_t scale = scales[i / 16];
                float value = d * scale * static_cast<float>(q - 32);
                size_t out_idx = block * 256 + i;
                if (out_idx < num_values) { buf[out_idx] = value; }
            }
        }
        return buf.data();
    } else if (type == GGMLType::Q8_0) {
        size_t num_values = 1;
        for (uint64_t dim : shape) { num_values *= dim; }
        buf.resize(num_values);
        size_t num_blocks = (num_values + 31) / 32;
        for (size_t block = 0; block < num_blocks; ++block) {
            const uint8_t* block_data = data + block * 34;
            uint16_t d_f16 = *reinterpret_cast<const uint16_t*>(block_data);
            float d = Transformer::f16_to_f32(d_f16);
            const int8_t* q = reinterpret_cast<const int8_t*>(block_data + 2);
            for (int i = 0; i < 32; ++i) {
                float value = d * q[i];
                size_t out_idx = block * 32 + i;
                if (out_idx < num_values) { buf[out_idx] = value; }
            }
        }
        return buf.data();
    } else {
        std::cerr << "Unsupported tensor type: " << static_cast<int>(type) << std::endl;
        return nullptr;
    }
}

static size_t packed_unit_size(GGMLType type) {
    switch (type) {
        case GGMLType::Q4_K: return sizeof(block_q4_K);
        case GGMLType::Q6_K: return sizeof(block_q6_K);
        case GGMLType::Q8_0: return 8 * sizeof(block_q8_0);
        case GGMLType::F16:   return 256 * sizeof(uint16_t);
        case GGMLType::F32:   return 256 * sizeof(float);
        default:              return 0;
    }
}

// Per-tensor NSM streaming helper for forward_batch().
static const uint8_t* get_nsm_tensor_raw(const GGUFParser* parser, const GGUFTensor* t,
                                          std::vector<uint8_t>& packed,
                                          const uint8_t* layer_base = nullptr,
                                          size_t layer_base_offset = 0) {
    ProfScope ps(g_prof.layer_load_ns, g_prof.enabled.load());
    if (!t || parser->nsm_fd < 0) {
        packed.clear();
        return t ? parser->data_ptr() + t->offset : nullptr;
    }

    // keep_in_ram tensors (norms, biases, token_embd, lm_head, output_norm) are already
    // dequantized into parser->data_ptr(); just return the packed pointer.
    if (!t->on_disk) {
        packed.clear();
        return parser->data_ptr() + t->offset;
    }

    size_t n = 1;
    for (uint64_t dim : t->shape) n *= (size_t)dim;
    size_t unit_size = packed_unit_size(t->type);
    if (unit_size == 0) {
        std::cerr << "get_nsm_tensor_raw: unsupported type " << static_cast<int>(t->type)
                  << " for " << t->name << std::endl;
        packed.clear();
        return nullptr;
    }
    size_t full_blocks = (n + 256 - 1) / 256;
    size_t out_bytes = full_blocks * unit_size;

    packed.resize(out_bytes);
    NSMTensor nt;
    nt.n_clusters = t->nsm_n_clusters;
    nt.data_offset = t->offset;
    nt.data_bytes = t->nsm_data_bytes;
    std::memset(nt.name, 0, sizeof(nt.name));
    nt.quant_type = (uint32_t)t->type;
    if (!nsm::nsm_dequant_tensor(parser->nsm_fd, nt, t->cl_offset, *t,
                                  nullptr, GGMLType::F32, 0,
                                  packed.data(), out_bytes,
                                  layer_base, layer_base_offset)) {
        std::cerr << "get_nsm_tensor_raw: failed for " << t->name << std::endl;
        packed.clear();
        return nullptr;
    }
    return packed.data();
}

static void add_bias_raw(float* out, const float* b, int n) {
    if (!b) return;
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vo = _mm256_loadu_ps(out + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_add_ps(vo, vb));
    }
    for (; i < n; i++) out[i] += b[i];
}

static void weight_matmul_batch_raw(const Transformer* self, const uint8_t* raw, int weight_type,
                                     const float* x_batch, int n_cols,
                                     float* out_batch, int n_rows,
                                     int batch_size) {
    if (weight_type == static_cast<int>(GGMLType::F32)) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    batch_size, n_rows, n_cols,
                    1.0f, x_batch, n_cols,
                    (const float*)raw, n_cols,
                    0.0f, out_batch, n_rows);
        return;
    }

    int n_blocks_per_row = n_cols / 256;
    int n_blocks_q8 = n_cols / 32;
    (void)n_blocks_q8;
    std::vector<block_q8_K> x_q8;
    if (weight_type == static_cast<int>(GGMLType::Q4_K) || weight_type == static_cast<int>(GGMLType::Q6_K)) {
        x_q8.resize((size_t)batch_size * n_blocks_per_row);
        for (int b = 0; b < batch_size; b++) {
            quantize_row_q8k(x_batch + (size_t)b * n_cols,
                              x_q8.data() + (size_t)b * n_blocks_per_row,
                              n_blocks_per_row);
        }
    }

    auto row_loop = [&](int r0, int r1) {
        for (int r = r0; r < r1; ++r) {
            switch (static_cast<GGMLType>(weight_type)) {
                case GGMLType::Q4_K: {
                    const block_q4_K* row = (const block_q4_K*)(raw + (size_t)r * n_blocks_per_row * 144);
                    for (int b = 0; b < batch_size; b++)
                        out_batch[(size_t)b * n_rows + r] = vec_dot_q4k_q8k(
                            row, x_q8.data() + (size_t)b * n_blocks_per_row, n_blocks_per_row);
                    break;
                }
                case GGMLType::Q6_K: {
                    const block_q6_K* row = (const block_q6_K*)(raw + (size_t)r * n_blocks_per_row * 210);
                    for (int b = 0; b < batch_size; b++)
                        out_batch[(size_t)b * n_rows + r] = ns_vec_dot_q6k_q8k(
                            row, x_q8.data() + (size_t)b * n_blocks_per_row, n_blocks_per_row);
                    break;
                }
                case GGMLType::Q8_0: {
                    int n_blocks_q8 = n_cols / 32;
                    const uint8_t* row_data = raw + (size_t)r * n_blocks_q8 * 34;
                    for (int b = 0; b < batch_size; b++)
                        out_batch[(size_t)b * n_rows + r] = self->vec_dot_q8_0(
                            row_data, x_batch + (size_t)b * n_cols, n_blocks_q8);
                    break;
                }
                default: break;
            }
        }
    };
    self->pool_->parallel_for(0, n_rows, row_loop);
}

Transformer::Transformer(size_t num_threads)
    : parser_(nullptr), kv_cache_capacity_(0),
      pool_(std::make_unique<ThreadPool>(num_threads)) {
}

Transformer::~Transformer() {
}

bool Transformer::load(const GGUFParser& parser, int context_len) {
    if (&parser != parser_owned_.get()) {
        parser_owned_.reset();
    }
    parser_ = &parser;
    const ModelParams& p = parser.params();
    params_.n_layers = p.n_layers;
    params_.n_heads = p.n_heads;
    params_.n_kv_heads = p.n_kv_heads;
    params_.n_embd = p.n_embd;
    params_.n_ff = p.n_ff;
    params_.n_vocab = p.n_vocab;
    params_.head_dim = p.head_dim;
    params_.max_seq_len = p.max_seq_len;
    params_.rope_freq_base = p.rope_freq_base;
    params_.rope_freq_scale = p.rope_freq_scale;

    // Precompute RoPE cos/sin tables for all positions up to max_seq_len
    int half_dim = params_.head_dim / 2;
    rope_cos_table_.resize((size_t)params_.max_seq_len * half_dim);
    rope_sin_table_.resize((size_t)params_.max_seq_len * half_dim);
    for (int pos = 0; pos < params_.max_seq_len; ++pos) {
        for (int i = 0; i < half_dim; ++i) {
            float val = (float)pos / powf(params_.rope_freq_base, 
                                          (2.0f * i) / params_.head_dim);
            rope_cos_table_[(size_t)pos * half_dim + i] = cosf(val);
            rope_sin_table_[(size_t)pos * half_dim + i] = sinf(val);
        }
    }

    int initial_capacity = context_len;
    if (initial_capacity < 1) initial_capacity = 1;
    kv_cache_capacity_ = initial_capacity;

    // Allocate KV cache with initial capacity
    size_t kv_size = (size_t)params_.n_layers * initial_capacity
                     * params_.n_kv_heads * params_.head_dim;
    (void)kv_size;
    k_cache_.resize(0);  // NS_KV_DISABLED: Zero old float32 cache (replaced by NSKVStore)
    v_cache_.resize(0);  // NS_KV_DISABLED: Zero old float32 cache (replaced by NSKVStore)
    // Initialize NSKVStore (INT4 quantized KV cache with eviction)
    NSKVConfig kv_cfg;
    kv_cfg.n_layers   = params_.n_layers;
    kv_cfg.n_kv_heads = params_.n_kv_heads;
    kv_cfg.head_dim   = params_.head_dim;
    kv_cfg.capacity   = (size_t)kv_cache_capacity_;
    kv_cfg.latent_dim = 0;
    kv_cfg.debug_float32 = false;
    kv_store_ = std::make_unique<NSKVStore>(kv_cfg);

    // Build O(1) name -> tensor index once. weight_matmul()/get_weight_f32()
    // used to do a linear scan with string comparisons over every tensor
    // (hundreds of them) on every single call, every layer, every token.
    tensor_index_.clear();
    tensor_index_.reserve(parser.tensors().size());
    for (size_t i = 0; i < parser.tensors().size(); ++i) {
        tensor_index_[parser.tensors()[i].name] = &parser.tensors()[i];
    }

    // Lazy per-tensor repack state.  token_embd is special: it is used for
    // embedding lookups, so its original mmap pages are kept even after repack.
    repacked_weights_llama_.clear();
    repacked_weights_llama_.reserve(parser.tensors().size());
    keep_original_names_.clear();
    keep_original_names_.insert("token_embd.weight");

    // Persistent scratch buffers, sized once instead of heap-allocated on
    // every call to forward() (i.e. every generated token).
    buf_x_norm_.resize(params_.n_embd);
    buf_q_.resize((size_t)params_.n_heads * params_.head_dim);
    buf_k_.resize((size_t)params_.n_kv_heads * params_.head_dim);
    buf_v_.resize((size_t)params_.n_kv_heads * params_.head_dim);
    buf_attn_out_.resize((size_t)params_.n_heads * params_.head_dim);
    buf_gate_.resize(params_.n_ff);
    buf_up_.resize(params_.n_ff);
    buf_ffn_mid_.resize(params_.n_ff);
    buf_attn_proj_.resize(params_.n_embd);
    buf_down_.resize(params_.n_embd);

    size_t n_blocks_attn_x   = (size_t)params_.n_embd / 256;
    size_t n_blocks_ffn_x    = (size_t)params_.n_embd / 256;
    size_t n_blocks_attn_out = (size_t)params_.n_heads * params_.head_dim / 256;
    size_t n_blocks_ffn_mid  = (size_t)params_.n_ff / 256;
    size_t n_blocks_logits   = (size_t)params_.n_embd / 256;
    size_t q8_sz = sizeof(block_q8_K);
    buf_attn_x_q8_.resize(n_blocks_attn_x * q8_sz);
    buf_ffn_x_q8_.resize(n_blocks_ffn_x * q8_sz);
    buf_attn_out_q8_.resize(n_blocks_attn_out * q8_sz);
    buf_ffn_mid_q8_.resize(n_blocks_ffn_mid * q8_sz);
    buf_logits_x_q8_.resize(n_blocks_logits * q8_sz);
    buf_scores_.resize((size_t)params_.n_heads * params_.max_seq_len);
    buf_v_cache_.resize((size_t)params_.max_seq_len * params_.head_dim);

    // Cache all norm weights: dequantize once at load time instead of every token.
    cached_norm_weights_.resize(params_.n_layers);
    for (int l = 0; l < params_.n_layers; ++l) {
        std::vector<float> buf;
        const float* w = get_weight_f32("blk." + std::to_string(l) + ".attn_norm.weight", buf);
        if (w) cached_norm_weights_[l].attn_norm_w.assign(w, w + params_.n_embd);
        w = get_weight_f32("blk." + std::to_string(l) + ".ffn_norm.weight", buf);
        if (w) cached_norm_weights_[l].ffn_norm_w.assign(w, w + params_.n_embd);
    }
    {
        std::vector<float> buf;
        const float* w = get_weight_f32("output_norm.weight", buf);
        if (w) cached_output_norm_w_.assign(w, w + params_.n_embd);
    }

    // Quantized weight_matmul() paths use our own OpenMP row-parallel loop.
    // Prevent OpenBLAS from *also* spawning its own thread pool for the F32
    // sgemv/sgemm branches, which would oversubscribe the CPU when nested
    // inside (or alongside) our omp parallel regions.
    openblas_set_num_threads(1);

    compile_schedule();

    std::cout << "Loaded transformer: " << params_.n_layers << " layers, "
              << params_.n_heads << " heads, " << params_.n_embd << " embedding dim" << std::endl;

    return true;
}

bool Transformer::load(const std::string& model_path, int context_len) {
    // Detect format by magic or extension
    bool is_nsm = false;
    if (model_path.size() >= 4 && model_path.compare(model_path.size() - 4, 4, ".nsm") == 0) {
        is_nsm = true;
    } else {
        FILE* f = std::fopen(model_path.c_str(), "rb");
        if (f) {
            uint32_t magic = 0;
            if (std::fread(&magic, sizeof(magic), 1, f) == 1 && magic == NSM_MAGIC) {
                is_nsm = true;
            }
            std::fclose(f);
        }
    }

    parser_owned_ = std::make_unique<GGUFParser>();
    bool ok = false;
    if (is_nsm) {
        // Look for a .gguf companion of the same stem for metadata/shapes
        std::string gguf_path = model_path;
        size_t dot = gguf_path.rfind('.');
        if (dot != std::string::npos) {
            gguf_path = gguf_path.substr(0, dot) + ".gguf";
        }
        ok = parser_owned_->load_nsm(model_path, gguf_path);
    } else {
        ok = parser_owned_->load(model_path);
    }
    if (!ok) {
        parser_owned_.reset();
        parser_ = nullptr;
        return false;
    }
    return load(*parser_owned_, context_len);
}

const float* Transformer::get_weight_f32(const std::string& name, 
                                          std::vector<float>& buf) const {
    const GGUFTensor* t = find_tensor(name);
    if (!t) {
        std::cerr << "Tensor not found: " << name << std::endl;
        return nullptr;
    }
    return get_weight_f32_raw(parser_->data_ptr() + t->offset, t->type, t->shape, buf);
}

void Transformer::rms_norm(float* out, const float* x, const float* w, int n) const {
    float sum_sq = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum_sq += x[i] * x[i];
    }
    float rms = std::sqrt(sum_sq / n + 1e-6f);
    for (int i = 0; i < n; ++i) {
        out[i] = x[i] / rms * w[i];
    }
}

void Transformer::rope(float* q, float* k, int pos, int n_heads, int n_kv_heads,
                        int head_dim, float freq_base) const {
    int half_dim = head_dim / 2;
    const float* cos_row = rope_cos_table_.data() + (size_t)pos * half_dim;
    const float* sin_row = rope_sin_table_.data() + (size_t)pos * half_dim;

    // For each head, rotate pairs (q[2i], q[2i+1]) by angle theta_i
    // theta_i = pos / (freq_base ^ (2i / head_dim))
    for (int h = 0; h < n_heads; h++) {
        float* qh = q + h * head_dim;
        for (int i = 0; i < head_dim/2; i++) {
            float q0 = qh[2*i];
            float q1 = qh[2*i+1];
            qh[2*i]   = q0 * cos_row[i] - q1 * sin_row[i];
            qh[2*i+1] = q0 * sin_row[i] + q1 * cos_row[i];
        }
    }
    // Same for k with n_kv_heads
    for (int h = 0; h < n_kv_heads; h++) {
        float* kh = k + h * head_dim;
        for (int i = 0; i < head_dim/2; i++) {
            float k0 = kh[2*i];
            float k1 = kh[2*i+1];
            kh[2*i]   = k0 * cos_row[i] - k1 * sin_row[i];
            kh[2*i+1] = k0 * sin_row[i] + k1 * cos_row[i];
        }
    }
}

void Transformer::softmax(float* x, int n) const {
    float max_val = x[0];
    for (int i = 1; i < n; ++i) {
        max_val = std::max(max_val, x[i]);
    }

    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    for (int i = 0; i < n; ++i) {
        x[i] /= sum;
    }
}

void Transformer::matmul(float* out, const float* x, const float* w, int n, int d) const {
    // out[d] = W[d,n] @ x[n]
    // cblas_sgemv: out = alpha * A * B + beta * C
    // A = w (d×n, row-major), B = x (n×1), C = out (d×1)
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                d, n, 1.0f, w, n, x, 1, 0.0f, out, 1);
}

float Transformer::max_abs(const float* x, int n) const {
    float max_val = 0.0f;
    for (int i = 0; i < n; ++i) max_val = std::max(max_val, std::abs(x[i]));
    return max_val;
}

// [PROF] lightweight per-section wall-clock timing, layer 0 only, fires
// once (first call to forward()) since pos==0 never occurs during generate
// (prefill uses forward_batch, generate starts at pos == prompt_len).
// clock_gettime(CLOCK_MONOTONIC) chosen over std::chrono here per request;
// existing [TIMING] std::chrono blocks below are untouched.
static inline double prof_ms(const struct timespec& t0, const struct timespec& t1) {
    return (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
}

// Specialized matmul implementations for each weight type.
// These are set once at compile_schedule() and called with zero runtime branches.
static void matmul_q4k(const MatmulArgs& a, const float* x, float* out, int pos) {
    (void)pos;
    const block_q8_K* q8 = (const block_q8_K*)a.x_q8;
    const Transformer::RepackedBuffer* repacked_buf = static_cast<const Transformer::RepackedBuffer*>(a.repacked);

    int n_panels = a.n_rows / 8;
    int n_panel_rows = n_panels * 8;
    if (n_panels > 0) {
        auto panel_loop = [&](int p0, int p1) {
            int rows = (p1 - p0) * 8;
            const uint8_t* panel_base =
                (const uint8_t*)repacked_buf->data_ptr + (size_t)p0 * a.n_blocks_per_row * sizeof(ns_q4_Kx8);
            ns_gemv_q4k(panel_base, q8, rows, a.n_blocks_per_row, out + p0 * 8);
        };
        a.run->self->pool_->parallel_for(0, n_panels, panel_loop);
    }
    if (n_panel_rows < a.n_rows && a.repacked) {
        if (repacked_buf->n_tail_rows > 0) {
            for (int r = n_panel_rows; r < a.n_rows; ++r) {
                size_t tail_r = r - n_panel_rows;
                out[r] = vec_dot_q4k_q8k(
                    (const block_q4_K*)(repacked_buf->tail_data.data() + tail_r * a.n_blocks_per_row * 144),
                    q8, a.n_blocks_per_row);
            }
        }
    }
}

static void matmul_f32(const MatmulArgs& a, const float* x, float* out, int pos) {
    (void)pos;
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                a.n_rows, a.n_cols, 1.0f,
                (const float*)a.raw, a.n_cols, x, 1, 0.0f, out, 1);
}

static void matmul_other(const MatmulArgs& a, const float* x, float* out, int pos) {
    (void)pos;
    const block_q8_K* q8 = (const block_q8_K*)a.x_q8;

    auto row_loop = [&](int r0, int r1) {
        for (int r = r0; r < r1; ++r) {
            switch ((GGMLType)a.weight_type) {
                case GGMLType::Q4_K:
                    out[r] = vec_dot_q4k_q8k(
                        (const block_q4_K*)(a.raw + (size_t)r * a.n_blocks_per_row * 144),
                        q8, a.n_blocks_per_row);
                    continue;
                case GGMLType::Q6_K:
                    out[r] = ns_vec_dot_q6k_q8k(
                        (const block_q6_K*)(a.raw + (size_t)r * a.n_blocks_per_row * 210),
                        q8, a.n_blocks_per_row);
                    continue;
                case GGMLType::Q8_0:
                    out[r] = a.run->self->vec_dot_q8_0(
                        a.raw + (size_t)r * a.n_blocks_q8 * 34, x, a.n_blocks_q8);
                    continue;
                default:
                    out[r] = 0.0f;
                    continue;
            }
        }
    };
    a.run->self->pool_->parallel_for(0, a.n_rows, row_loop);
}


bool Transformer::forward(int32_t token_id, int pos, InferenceState& state, bool compute_logits_flag) {
    if (parser_->nsm_fd >= 0) {
        // NSM streaming: reuse forward_batch for single token
        std::vector<int32_t> single = {token_id};
        return forward_batch(single, pos, state, compute_logits_flag);
    }

    if (pos >= kv_cache_capacity_) {
        grow_kv_cache(kv_cache_capacity_ * 2);
    }

    state.hidden.resize(params_.n_embd);
    state.logits.resize(params_.n_vocab);

    run_ctx_.self = this;
    run_ctx_.pos = pos;
    run_ctx_.compute_logits = compute_logits_flag;
    run_ctx_.hidden = state.hidden.data();
    run_ctx_.logits = state.logits.data();

    embed_args_->token_id = token_id;

    pool_->enter_team();
    schedule_.run();
    pool_->leave_team();
    return true;
}

void Transformer::compile_schedule() {
    run_ctx_.self = this;
    schedule_.ops.clear();
    schedule_.storage.clear();
    schedule_.ops.reserve(params_.n_layers * 24 + 3);

    int n_embd = params_.n_embd;
    int n_ff = params_.n_ff;
    int n_heads = params_.n_heads;
    int n_kv_heads = params_.n_kv_heads;
    int head_dim = params_.head_dim;
    int q_rows = n_heads * head_dim;
    int kv_rows = n_kv_heads * head_dim;
    int n_blocks_attn_x = n_embd / 256;
    int n_blocks_ffn_x = n_embd / 256;
    int n_blocks_attn_out = q_rows / 256;
    int n_blocks_ffn_mid = n_ff / 256;
    int n_blocks_logits = n_embd / 256;

    auto push_rms_norm = [&](const std::string& name, float* out, int n) {
        NormArgs* a = new NormArgs;
        a->run = &run_ctx_;
        a->out = out;
        a->w = get_weight_f32(name, a->weight_buf);
        a->n = n;
        schedule_.storage.emplace_back(a);
        schedule_.ops.push_back({&Transformer::op_rms_norm, a});
    };

    auto push_quantize = [&](float* x, void* x_q8, int n_blocks) {
        QuantizeArgs* a = new QuantizeArgs;
        a->run = &run_ctx_;
        a->x = x;
        a->x_q8 = x_q8;
        a->n_blocks = n_blocks;
        schedule_.storage.emplace_back(a);
        schedule_.ops.push_back({&Transformer::op_quantize_q8, a});
    };

    auto push_matmul = [&](const std::string& name, float* x, void* x_q8, float* out,
                           int n_cols, int n_rows, int n_blocks_per_row) {
        MatmulArgs* a = new MatmulArgs;
        const GGUFTensor* t = find_tensor(name);
        if (t) {
            a->raw = parser_->data_ptr() + t->offset;
            a->weight_type = static_cast<int>(t->type);
            if (t->type == GGMLType::Q4_K) {
                const RepackedBuffer* buf = get_repacked_buffer_llama(t);
                if (buf) {
                    a->repacked = buf;
                    a->repacked_R = buf->R;
                    a->matmul_fn = &matmul_q4k;
                } else {
                    a->matmul_fn = &matmul_other;
                }
            } else if (t->type == GGMLType::F32) {
                a->matmul_fn = &matmul_f32;
            } else if (t->type == GGMLType::Q6_K) {
                a->matmul_fn = &matmul_other;
            } else {
                a->matmul_fn = &matmul_other;
            }
        } else {
            a->raw = nullptr;
            a->weight_type = -1;
            a->matmul_fn = &matmul_other;
        }
        a->x = x;
        a->x_q8 = x_q8;
        a->out = out;
        a->n_cols = n_cols;
        a->n_rows = n_rows;
        a->n_blocks_per_row = n_blocks_per_row;
        a->n_blocks_q8 = n_cols / 32;
        a->is_logits = false;
        a->run = &run_ctx_;
        schedule_.storage.emplace_back(a);
        schedule_.ops.push_back({&Transformer::op_matmul_q8, a});
    };

    auto push_bias = [&](const std::string& name, float* out, int n) {
        BiasArgs* a = new BiasArgs;
        const GGUFTensor* t = find_tensor(name);
        if (t && t->type == GGMLType::F32) {
            a->b = reinterpret_cast<const float*>(parser_->data_ptr() + t->offset);
        } else {
            a->b = nullptr;
        }
        a->out = out;
        a->n = n;
        a->run = &run_ctx_;
        schedule_.storage.emplace_back(a);
        schedule_.ops.push_back({&Transformer::op_add_bias, a});
    };

    auto push_rope = [&]() {
        RopeArgs* a = new RopeArgs;
        a->q = buf_q_.data();
        a->k = buf_k_.data();
        a->n_heads = n_heads;
        a->n_kv_heads = n_kv_heads;
        a->head_dim = head_dim;
        a->freq_base = params_.rope_freq_base;
        a->run = &run_ctx_;
        schedule_.storage.emplace_back(a);
        schedule_.ops.push_back({&Transformer::op_rope, a});
    };

    auto push_kv_write = [&](int l) {
        KVWriteArgs* a = new KVWriteArgs;
        a->layer = l;
        a->k = buf_k_.data();
        a->v = buf_v_.data();
        a->n = kv_rows;
        a->run = &run_ctx_;
        schedule_.storage.emplace_back(a);
        schedule_.ops.push_back({&Transformer::op_kv_write, a});
    };

    auto push_attention = [&](int l) {
        AttentionArgs* a = new AttentionArgs;
        a->layer = l;
        a->q = buf_q_.data();
        a->out = buf_attn_out_.data();
        a->n = q_rows;
        a->run = &run_ctx_;
        schedule_.storage.emplace_back(a);
        schedule_.ops.push_back({&Transformer::op_attention, a});
    };

    auto push_residual = [&](const float* b, int n) {
        ResidualArgs* a = new ResidualArgs;
        a->b = b;
        a->n = n;
        a->run = &run_ctx_;
        schedule_.storage.emplace_back(a);
        schedule_.ops.push_back({&Transformer::op_residual, a});
    };

    auto push_swiglu = [&]() {
        SwiGLUArgs* a = new SwiGLUArgs;
        a->gate = buf_gate_.data();
        a->up = buf_up_.data();
        a->mid = buf_ffn_mid_.data();
        a->n = n_ff;
        a->run = &run_ctx_;
        schedule_.storage.emplace_back(a);
        schedule_.ops.push_back({&Transformer::op_swiglu, a});
    };

    // EMBED
    {
        EmbedArgs* a = new EmbedArgs;
        const GGUFTensor* t = find_tensor("token_embd.weight");
        if (t) {
            a->raw = parser_->data_ptr() + t->offset;
            a->weight_type = static_cast<int>(t->type);
        } else {
            a->raw = nullptr;
            a->weight_type = -1;
        }
        a->n = n_embd;
        a->out = nullptr;
        a->run = &run_ctx_;
        schedule_.storage.emplace_back(a);
        schedule_.ops.push_back({&Transformer::op_embed, a});
        embed_args_ = a;
    }

    for (int l = 0; l < params_.n_layers; ++l) {
        push_rms_norm("blk." + std::to_string(l) + ".attn_norm.weight", buf_x_norm_.data(), n_embd);
        push_quantize(buf_x_norm_.data(), buf_attn_x_q8_.data(), n_blocks_attn_x);

        push_matmul("blk." + std::to_string(l) + ".attn_q.weight",
                    buf_x_norm_.data(), buf_attn_x_q8_.data(), buf_q_.data(),
                    n_embd, q_rows, n_blocks_attn_x);
        push_bias("blk." + std::to_string(l) + ".attn_q.bias", buf_q_.data(), q_rows);

        push_matmul("blk." + std::to_string(l) + ".attn_k.weight",
                    buf_x_norm_.data(), buf_attn_x_q8_.data(), buf_k_.data(),
                    n_embd, kv_rows, n_blocks_attn_x);
        push_bias("blk." + std::to_string(l) + ".attn_k.bias", buf_k_.data(), kv_rows);

        push_matmul("blk." + std::to_string(l) + ".attn_v.weight",
                    buf_x_norm_.data(), buf_attn_x_q8_.data(), buf_v_.data(),
                    n_embd, kv_rows, n_blocks_attn_x);
        push_bias("blk." + std::to_string(l) + ".attn_v.bias", buf_v_.data(), kv_rows);

        push_rope();
        push_kv_write(l);
        push_attention(l);

        push_quantize(buf_attn_out_.data(), buf_attn_out_q8_.data(), n_blocks_attn_out);
        push_matmul("blk." + std::to_string(l) + ".attn_output.weight",
                    buf_attn_out_.data(), buf_attn_out_q8_.data(), buf_attn_proj_.data(),
                    q_rows, n_embd, n_blocks_attn_out);
        push_bias("blk." + std::to_string(l) + ".attn_output.bias", buf_attn_proj_.data(), n_embd);
        push_residual(buf_attn_proj_.data(), n_embd);

        push_rms_norm("blk." + std::to_string(l) + ".ffn_norm.weight", buf_x_norm_.data(), n_embd);
        push_quantize(buf_x_norm_.data(), buf_ffn_x_q8_.data(), n_blocks_ffn_x);

        push_matmul("blk." + std::to_string(l) + ".ffn_gate.weight",
                    buf_x_norm_.data(), buf_ffn_x_q8_.data(), buf_gate_.data(),
                    n_embd, n_ff, n_blocks_ffn_x);
        push_matmul("blk." + std::to_string(l) + ".ffn_up.weight",
                    buf_x_norm_.data(), buf_ffn_x_q8_.data(), buf_up_.data(),
                    n_embd, n_ff, n_blocks_ffn_x);

        push_swiglu();

        push_quantize(buf_ffn_mid_.data(), buf_ffn_mid_q8_.data(), n_blocks_ffn_mid);
        push_matmul("blk." + std::to_string(l) + ".ffn_down.weight",
                    buf_ffn_mid_.data(), buf_ffn_mid_q8_.data(), buf_down_.data(),
                    n_ff, n_embd, n_blocks_ffn_mid);
        push_residual(buf_down_.data(), n_embd);
    }

    push_rms_norm("output_norm.weight", nullptr, n_embd);
    push_quantize(nullptr, buf_logits_x_q8_.data(), n_blocks_logits);

    {
        MatmulArgs* a = new MatmulArgs;
        const GGUFTensor* t = find_tensor("output.weight");
        if (!t) t = find_tensor("token_embd.weight");
        if (t) {
            a->raw = parser_->data_ptr() + t->offset;
            std::cout << "[LM HEAD] " << t->name
                      << " raw=" << (void*)a->raw
                      << " offset=" << t->offset
                      << std::endl;
            a->weight_type = static_cast<int>(t->type);
            if (t->type == GGMLType::Q4_K) {
                const RepackedBuffer* buf = get_repacked_buffer_llama(t);
                if (buf) {
                    a->repacked = buf;
                    a->repacked_R = buf->R;
                    a->matmul_fn = &matmul_q4k;
                } else {
                    a->matmul_fn = &matmul_other;
                }
            } else if (t->type == GGMLType::F32) {
                a->matmul_fn = &matmul_f32;
            } else if (t->type == GGMLType::Q6_K) {
                a->matmul_fn = &matmul_other;
            } else {
                a->matmul_fn = &matmul_other;
            }
        } else {
            a->raw = nullptr;
            a->weight_type = -1;
            a->matmul_fn = &matmul_other;
        }
        a->x = nullptr;
        a->x_q8 = buf_logits_x_q8_.data();
        a->out = nullptr;
        a->n_cols = n_embd;
        a->n_rows = params_.n_vocab;
        a->n_blocks_per_row = n_blocks_logits;
        a->n_blocks_q8 = n_embd / 32;
        a->is_logits = true;
        a->run = &run_ctx_;
        schedule_.storage.emplace_back(a);
        schedule_.ops.push_back({&Transformer::op_matmul_q8, a});
    }
}

void Transformer::op_embed(void* ctx) {
    EmbedArgs* a = (EmbedArgs*)ctx;
    if (a->weight_type < 0) return;
    float* out = a->out ? a->out : a->run->hidden;
    a->run->self->get_embedding_row_impl(a->raw, a->weight_type, a->token_id, out, a->n);
}

void Transformer::op_rms_norm(void* ctx) {
    NormArgs* a = (NormArgs*)ctx;
    if (!a->w) return;
    float* out = a->out ? a->out : a->run->hidden;
    a->run->self->rms_norm(out, a->run->hidden, a->w, a->n);
}

void Transformer::op_quantize_q8(void* ctx) {
    QuantizeArgs* a = (QuantizeArgs*)ctx;
    if (!a->x_q8) return;
    const float* x = a->x ? a->x : a->run->hidden;
    quantize_row_q8k(x, (block_q8_K*)a->x_q8, a->n_blocks);
}

void Transformer::op_matmul_q8(void* ctx) {
    MatmulArgs* a = (MatmulArgs*)ctx;
    if (a->weight_type < 0) return;
    if (a->is_logits && !a->run->compute_logits) return;
    if (!a->matmul_fn) {
        std::cerr << "ERROR: matmul_fn is null for weight_type " << a->weight_type << std::endl;
        return;
    }
    const float* x = a->x ? a->x : a->run->hidden;
    float* out = a->out ? a->out : a->run->logits;
    a->matmul_fn(*a, x, out, a->run->pos);
}

void Transformer::op_add_bias(void* ctx) {
    BiasArgs* a = (BiasArgs*)ctx;
    if (!a->b) return;
    int i = 0;
    for (; i + 8 <= a->n; i += 8) {
        __m256 vo = _mm256_loadu_ps(a->out + i);
        __m256 vb = _mm256_loadu_ps(a->b + i);
        _mm256_storeu_ps(a->out + i, _mm256_add_ps(vo, vb));
    }
    for (; i < a->n; ++i) a->out[i] += a->b[i];
}

void Transformer::op_rope(void* ctx) {
    RopeArgs* a = (RopeArgs*)ctx;
    a->run->self->rope(a->q, a->k, a->run->pos,
                       a->n_heads, a->n_kv_heads, a->head_dim, a->freq_base);
}

void Transformer::op_kv_write(void* ctx) {
    KVWriteArgs* a = (KVWriteArgs*)ctx;
    Transformer* s = a->run->self;
    int pos = a->run->pos;
    
    // NSKVStore: INT4 quantized write with eviction
    if (s->kv_store_->needs_eviction()) s->kv_store_->evict();
    s->kv_store_->write(a->layer, pos,
                        a->k, a->v,
                        s->params_.n_kv_heads * s->params_.head_dim);
    
    // NS_KV_DISABLED: Old float32 memcpy (kept for A/B comparison)
    // size_t offset = (size_t)a->layer * s->kv_cache_capacity_ * s->params_.n_kv_heads * s->params_.head_dim
    //                 + (size_t)pos * s->params_.n_kv_heads * s->params_.head_dim;
    // std::memcpy(s->k_cache_.data() + offset, a->k, a->n * sizeof(float));
    // std::memcpy(s->v_cache_.data() + offset, a->v, a->n * sizeof(float));
}

void Transformer::op_attention(void* ctx) {
    AttentionArgs* a = (AttentionArgs*)ctx;
    a->run->self->attention_impl(a->layer, a->q, a->out, a->run->pos);
}

void Transformer::op_residual(void* ctx) {
    ResidualArgs* r = (ResidualArgs*)ctx;
    float* a = r->run->hidden;
    int i = 0;
    for (; i + 8 <= r->n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(r->b + i);
        _mm256_storeu_ps(a + i, _mm256_add_ps(va, vb));
    }
    for (; i < r->n; ++i) a[i] += r->b[i];
}

void Transformer::op_swiglu(void* ctx) {
    SwiGLUArgs* a = (SwiGLUArgs*)ctx;
    for (int i = 0; i < a->n; ++i) {
        a->gate[i] = a->gate[i] / (1.0f + expf(-a->gate[i]));
        a->mid[i] = a->gate[i] * a->up[i];
    }
}

void Transformer::weight_matmul_q8_tensor(const uint8_t* raw, int weight_type,
                                          const float* x, int n_cols,
                                          const void* x_q8, int n_blocks_per_row,
                                          float* out, int n_rows,
                                          int n_blocks_q8,
                                          const void* repacked,
                                          int repacked_R) const {
    (void)x; (void)repacked_R;
    if (weight_type == (int)GGMLType::F32) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    n_rows, n_cols, 1.0f,
                    (const float*)raw, n_cols, x, 1, 0.0f, out, 1);
        return;
    }
    const block_q8_K* q8 = (const block_q8_K*)x_q8;

    // Q4_K: use the llama.cpp-ported pure-AVX2/FMA panel GEMV kernel
    // (ns_gemv_q4k, ns_llama_gemv.cpp) for whole 8-row panels of the
    // repacked buffer. This avoids the sustained-ZMM downclocking of
    // vec_dot_q4k_q8k. Tail rows (n_rows % 8) fall back to the original
    // per-row AVX-512 kernel.
    if ((GGMLType)weight_type == GGMLType::Q4_K && repacked != nullptr) {
        int n_panels = n_rows / 8;
        int n_panel_rows = n_panels * 8;
        if (n_panels > 0) {
            auto panel_loop = [&](int p0, int p1) {
                int rows = (p1 - p0) * 8;
                const uint8_t* panel_base =
                    (const uint8_t*)repacked + (size_t)p0 * n_blocks_per_row * sizeof(ns_q4_Kx8);
                ns_gemv_q4k(panel_base, q8, rows, n_blocks_per_row, out + p0 * 8);
            };
            pool_->parallel_for(0, n_panels, panel_loop);
        }
        if (n_panel_rows < n_rows && repacked) {
            const RepackedBuffer* repacked_buf = static_cast<const RepackedBuffer*>(repacked);
            if (repacked_buf->n_tail_rows > 0) {
                for (int r = n_panel_rows; r < n_rows; ++r) {
                    size_t tail_r = r - n_panel_rows;
                    out[r] = vec_dot_q4k_q8k(
                        (const block_q4_K*)(repacked_buf->tail_data.data() + tail_r * n_blocks_per_row * 144),
                        q8, n_blocks_per_row);
                }
            }
        }
        return;
    }

    auto row_loop = [&](int r0, int r1) {
        for (int r = r0; r < r1; ++r) {
            switch ((GGMLType)weight_type) {
                case GGMLType::Q4_K:
                    out[r] = vec_dot_q4k_q8k(
                        (const block_q4_K*)(raw + (size_t)r * n_blocks_per_row * 144),
                        q8, n_blocks_per_row);
                    continue;
                case GGMLType::Q6_K:
                    // ns_vec_dot_q6k_q8k: pure AVX2/FMA port (ns_llama_gemv.cpp),
                    // avoids the sustained-ZMM vec_dot_q6k_q8k above so Q6_K rows
                    // (token_embd/output + some attn_v/ffn_down layers) don't
                    // downclock the CPU during the hot per-token matmul path.
                    out[r] = ns_vec_dot_q6k_q8k(
                        (const block_q6_K*)(raw + (size_t)r * n_blocks_per_row * 210),
                        q8, n_blocks_per_row);
                    continue;
                case GGMLType::Q8_0:
                    out[r] = vec_dot_q8_0(
                        raw + (size_t)r * n_blocks_q8 * 34, x, n_blocks_q8);
                    continue;
                default:
                    out[r] = 0.0f;
                    continue;
            }
        }
    };
    pool_->parallel_for(0, n_rows, row_loop);
}

void Transformer::attention_impl(int layer, const float* q, float* out, int pos) {
    int q_rows = params_.n_heads * params_.head_dim;
    std::fill(out, out + q_rows, 0.0f);
    int n_q_heads_per_kv = params_.n_heads / params_.n_kv_heads;

    auto head_loop = [&](int h0, int h1) {
        for (int h = h0; h < h1; ++h) {
            int kv_h = h / n_q_heads_per_kv;
            float* scores = buf_scores_.data() + h * params_.max_seq_len;
            std::fill(scores, scores + pos + 1, -1e30f);
            // Fused K+V read: dequant both in one call, cache V for second pass.
            static thread_local float v_cache[8192];  // max_seq_len * head_dim (safe for 2048*128/256=1024)
            for (int p = 0; p <= pos; ++p) {
                static thread_local float k_tmp[256];
                float* vp_cache = v_cache + (size_t)p * params_.head_dim;
                bool valid = kv_store_->read_kv_head(layer, p, kv_h, k_tmp, vp_cache);
                if (!valid) continue;
                const float* kp = k_tmp;

                float dot = 0.0f;
                const float* qh = q + h * params_.head_dim;
                for (int d = 0; d < params_.head_dim; ++d) dot += qh[d] * kp[d];
                scores[p] = dot / std::sqrt(static_cast<float>(params_.head_dim));
            }
            softmax(scores, pos + 1);
            float* oh = out + h * params_.head_dim;
            for (int p = 0; p <= pos; ++p) {
                if (scores[p] <= -1e29f) continue;
                const float* vp = v_cache + (size_t)p * params_.head_dim;
                for (int d = 0; d < params_.head_dim; ++d) oh[d] += scores[p] * vp[d];
            }
        }
    };
    pool_->parallel_for(0, params_.n_heads, head_loop);
}

void Transformer::get_embedding_row_impl(const uint8_t* raw, int weight_type,
                                         int32_t token_id, float* out, int n) const {
    size_t row_start_elem = (size_t)token_id * (size_t)n;
    if ((GGMLType)weight_type == GGMLType::F32) {
        const float* f32_data = (const float*)raw;
        std::memcpy(out, f32_data + row_start_elem, n * sizeof(float));
    } else if ((GGMLType)weight_type == GGMLType::Q4_K) {
        size_t row_start_block = row_start_elem / 256;
        const uint8_t* row_ptr = raw + row_start_block * 144;
        dequantize_row_q4_K((const block_q4_K*)row_ptr, out, n);
    } else {
        std::cerr << "Unsupported embedding type: " << weight_type << std::endl;
    }
}

bool Transformer::forward_batch(const std::vector<int32_t>& token_ids, int start_pos,
                                 InferenceState& state, bool compute_logits_last_only,
                                 std::vector<float>* all_hidden,
                                 std::vector<NSKVStore*>* per_token_kv_stores,
                                 const std::vector<int>* per_token_positions) {
    int batch_size = static_cast<int>(token_ids.size());
    if (batch_size == 0) return true;
    bool prof = (batch_size == 1);
    if (prof) g_prof.enabled.store(1, std::memory_order_relaxed);

    int end_pos;
    if (per_token_positions && !per_token_positions->empty()) {
        end_pos = *std::max_element(per_token_positions->begin(), per_token_positions->end());
    } else {
        end_pos = start_pos + batch_size - 1;
    }
    while (end_pos >= kv_cache_capacity_) {
        grow_kv_cache(kv_cache_capacity_ * 2);
    }

    int n_embd = params_.n_embd;
    int q_rows = params_.n_heads * params_.head_dim;
    int kv_rows = params_.n_kv_heads * params_.head_dim;
    int n_ff = params_.n_ff;
    int n_q_heads_per_kv = params_.n_heads / params_.n_kv_heads;

    // Layer-wise double-buffered pread from the NSM file.
    std::vector<uint8_t> layer_buf[2];
    std::thread prefetch;
    int cur = 0;
    size_t layer_off = 0;
    if (parser_->nsm_fd >= 0 && !parser_->nsm_layer_index.empty()) {
        size_t max_layer_bytes = 0;
        for (const auto& li : parser_->nsm_layer_index) {
            max_layer_bytes = std::max(max_layer_bytes, li.data_bytes);
        }
        layer_buf[0].resize(max_layer_bytes);
        layer_buf[1].resize(max_layer_bytes);
        if (params_.n_layers > 0) {
            (void)profiled_pread(parser_->nsm_fd, layer_buf[cur].data(), parser_->nsm_layer_index[0].data_bytes, parser_->nsm_layer_index[0].data_offset);
        }
    }

    std::vector<float> hidden_batch((size_t)batch_size * n_embd);
    std::vector<float> x_norm_batch((size_t)batch_size * n_embd);
    std::vector<float> q_batch((size_t)batch_size * q_rows);
    std::vector<float> k_batch((size_t)batch_size * kv_rows);
    std::vector<float> v_batch((size_t)batch_size * kv_rows);
    std::vector<float> attn_out_batch((size_t)batch_size * q_rows);
    std::vector<float> attn_proj_batch((size_t)batch_size * n_embd);
    std::vector<float> gate_batch((size_t)batch_size * n_ff);
    std::vector<float> up_batch((size_t)batch_size * n_ff);
    std::vector<float> ffn_mid_batch((size_t)batch_size * n_ff);
    std::vector<float> down_batch((size_t)batch_size * n_embd);

    // 1. Embed every token in the batch.
    for (int b = 0; b < batch_size; b++) {
        std::vector<float> row;
        get_embedding_row(token_ids[b], row);
        std::memcpy(hidden_batch.data() + (size_t)b * n_embd, row.data(), n_embd * sizeof(float));
    }

    pool_->enter_team();
    for (int l = 0; l < static_cast<int>(params_.n_layers); ++l) {
        if (parser_->nsm_fd >= 0 && l < static_cast<int>(parser_->nsm_layer_index.size())) {
            layer_off = parser_->nsm_layer_index[l].data_offset;
        }
        auto get_layer = [&](const GGUFTensor* t, std::vector<uint8_t>& packed) {
            return get_nsm_tensor_raw(parser_, t, packed, layer_buf[cur].data(), layer_off);
        };

        // Fast path: for an on-disk tensor whose clusters are stored raw (no zlib),
        // point directly into the prefetched layer buffer and skip nsm_dequant_tensor.
        auto get_weight_ptr = [&](const GGUFTensor* t, std::vector<uint8_t>& packed) -> const uint8_t* {
            if (t->on_disk && !t->clusters.empty() && !t->clusters[0].compressed) {
                packed.clear();
                if (prof) {
                    g_prof.buffer_hits.fetch_add(1, std::memory_order_relaxed);
                    g_prof.layers_buffer.fetch_add(1, std::memory_order_relaxed);
                }
                return layer_buf[cur].data() + (t->offset - layer_off);
            }
            return get_nsm_tensor_raw(parser_, t, packed, layer_buf[cur].data(), layer_off);
        };

        auto do_matmul = [&](const uint8_t* raw, int weight_type,
                             const float* x, int n_cols, float* out, int n_rows, int) {
            if (!prof) {
                (weight_matmul_batch_raw)(this, raw, weight_type, x, n_cols, out, n_rows, batch_size);
                return;
            }
            int64_t t0 = prof_now_ns();
            (weight_matmul_batch_raw)(this, raw, weight_type, x, n_cols, out, n_rows, batch_size);
            g_prof.matmul_ns.fetch_add(prof_now_ns() - t0, std::memory_order_relaxed);
        };

        // Prefetch next layer into the alternate buffer while we compute this one.
        if (parser_->nsm_fd >= 0 && l + 1 < static_cast<int>(parser_->nsm_layer_index.size())) {
            int next_cur = 1 - cur;
            prefetch = std::thread([&, next = l + 1, next_cur]() {
                (void)profiled_pread(parser_->nsm_fd, layer_buf[next_cur].data(), parser_->nsm_layer_index[next].data_bytes, parser_->nsm_layer_index[next].data_offset);
            });
        }

        // a. attn RMS norm (per token: cheap, O(batch * n_embd))
        const float* attn_norm_w = cached_norm_weights_[l].attn_norm_w.data();
        for (int b = 0; b < batch_size; b++) {
            rms_norm(x_norm_batch.data() + (size_t)b * n_embd,
                     hidden_batch.data() + (size_t)b * n_embd,
                     attn_norm_w, n_embd);
        }

        // b/c/d. Q/K/V projections: each weight row is dequantized ONCE for
        // the whole batch here, instead of once per token like weight_matmul().
        std::string q_name = "blk." + std::to_string(l) + ".attn_q.weight";
        const GGUFTensor* t_q = find_tensor(q_name);
        if (!t_q) { std::cerr << "weight_matmul_batch: tensor not found: " << q_name << std::endl; return false; }
        std::vector<uint8_t> packed_q;
        do_matmul(get_weight_ptr( t_q, packed_q), (int)t_q->type, x_norm_batch.data(), n_embd, q_batch.data(), q_rows, batch_size);
        std::string k_name = "blk." + std::to_string(l) + ".attn_k.weight";
        const GGUFTensor* t_k = find_tensor(k_name);
        if (!t_k) { std::cerr << "weight_matmul_batch: tensor not found: " << k_name << std::endl; return false; }
        std::vector<uint8_t> packed_k;
        do_matmul(get_weight_ptr( t_k, packed_k), (int)t_k->type, x_norm_batch.data(), n_embd, k_batch.data(), kv_rows, batch_size);
        std::string v_name = "blk." + std::to_string(l) + ".attn_v.weight";
        const GGUFTensor* t_v = find_tensor(v_name);
        if (!t_v) { std::cerr << "weight_matmul_batch: tensor not found: " << v_name << std::endl; return false; }
        std::vector<uint8_t> packed_v;
        do_matmul(get_weight_ptr( t_v, packed_v), (int)t_v->type, x_norm_batch.data(), n_embd, v_batch.data(), kv_rows, batch_size);

        std::string q_bias_name = "blk." + std::to_string(l) + ".attn_q.bias";
        std::string k_bias_name = "blk." + std::to_string(l) + ".attn_k.bias";
        std::string v_bias_name = "blk." + std::to_string(l) + ".attn_v.bias";
        const GGUFTensor* t_qb = find_tensor(q_bias_name);
        const GGUFTensor* t_kb = find_tensor(k_bias_name);
        const GGUFTensor* t_vb = find_tensor(v_bias_name);
        std::vector<float> q_bias_buf, k_bias_buf, v_bias_buf;
        std::vector<uint8_t> packed_qb, packed_kb, packed_vb;
        const float* q_b = t_qb ? get_weight_f32_raw(get_layer( t_qb, packed_qb), t_qb->type, t_qb->shape, q_bias_buf) : nullptr;
        const float* k_b = t_kb ? get_weight_f32_raw(get_layer( t_kb, packed_kb), t_kb->type, t_kb->shape, k_bias_buf) : nullptr;
        const float* v_b = t_vb ? get_weight_f32_raw(get_layer( t_vb, packed_vb), t_vb->type, t_vb->shape, v_bias_buf) : nullptr;
        for (int b = 0; b < batch_size; b++) {
            add_bias_raw(q_batch.data() + (size_t)b * q_rows, q_b, q_rows);
            add_bias_raw(k_batch.data() + (size_t)b * kv_rows, k_b, kv_rows);
            add_bias_raw(v_batch.data() + (size_t)b * kv_rows, v_b, kv_rows);
        }

        // e/f. RoPE + KV cache store (inherently per-token; cheap).
        for (int b = 0; b < batch_size; b++) {
            int pos = per_token_positions ? (*per_token_positions)[b] : start_pos + b;
            rope(q_batch.data() + (size_t)b * q_rows, k_batch.data() + (size_t)b * kv_rows,
                 pos, params_.n_heads, params_.n_kv_heads, params_.head_dim, params_.rope_freq_base);

            // KV store selection: per-token override, then state, then Transformer default.
            NSKVStore* kv_store = kv_store_.get();
            if (per_token_kv_stores && b < (int)per_token_kv_stores->size() && (*per_token_kv_stores)[b]) {
                kv_store = (*per_token_kv_stores)[b];
            } else if (state.kv_store) {
                kv_store = state.kv_store.get();
            }

            // NSKVStore: INT4 quantized write with eviction
            {
                ProfScope ps(g_prof.kv_write_ns, prof);
                if (kv_store->needs_eviction()) kv_store->evict();
                kv_store->write(l, pos,
                                k_batch.data() + (size_t)b * kv_rows,
                                v_batch.data() + (size_t)b * kv_rows,
                                kv_rows);
            }

            // NS_KV_DISABLED: Old float32 memcpy (kept for A/B comparison)
            // size_t cache_offset = (size_t)l * kv_cache_capacity_ * params_.n_kv_heads * params_.head_dim +
            //                        (size_t)pos * params_.n_kv_heads * params_.head_dim;
            // std::memcpy(k_cache_.data() + cache_offset, k_batch.data() + (size_t)b * kv_rows,
            //             kv_rows * sizeof(float));
            // std::memcpy(v_cache_.data() + cache_offset, v_batch.data() + (size_t)b * kv_rows,
            //             kv_rows * sizeof(float));
        }

        // g. Causal attention. Independent per token (each only reads KV up
        // to its own position and writes its own output row) so it is safe
        // and cheap to parallelize over the batch dimension.
        std::fill(attn_out_batch.begin(), attn_out_batch.end(), 0.0f);
        uint64_t t_kv_read_local = 0;
        auto batch_loop = [&](int b0, int b1) {
            for (int b = b0; b < b1; ++b) {
                int pos = per_token_positions ? (*per_token_positions)[b] : start_pos + b;
                const float* qb = q_batch.data() + (size_t)b * q_rows;
                float* ob = attn_out_batch.data() + (size_t)b * q_rows;
                float* scores = buf_scores_.data() + (size_t)b * params_.max_seq_len;
                std::fill(scores, scores + pos + 1, -1e30f);

                // KV store selection for this token.
                NSKVStore* kv_store = kv_store_.get();
                if (per_token_kv_stores && b < (int)per_token_kv_stores->size() && (*per_token_kv_stores)[b]) {
                    kv_store = (*per_token_kv_stores)[b];
                } else if (state.kv_store) {
                    kv_store = state.kv_store.get();
                }

                for (int h = 0; h < params_.n_heads; ++h) {
                    int kv_h = h / n_q_heads_per_kv;
                    const float* qh = qb + h * params_.head_dim;
                    float* v_cache = buf_v_cache_.data();
                    for (int p = 0; p <= pos; ++p) {
                        // Fused K+V read: dequant both in one call, cache V.
                        float k_tmp[256];
                        float* vp_cache = v_cache + (size_t)p * params_.head_dim;
                        int64_t t0 = prof ? prof_now_ns() : 0;
                        bool valid = kv_store->read_kv_head(l, p, kv_h, k_tmp, vp_cache);
                        if (prof) t_kv_read_local += prof_now_ns() - t0;
                        if (!valid) continue;
                        const float* kp = k_tmp;

                        float dot = 0.0f;
                        const hn::ScalableTag<float> df;
                        auto vacc = hn::Zero(df);
                        int lanes = hn::Lanes(df);
                        int d = 0;
                        for (; d + lanes <= params_.head_dim; d += lanes)
                            vacc = hn::MulAdd(hn::LoadU(df, qh+d), hn::LoadU(df, kp+d), vacc);
                        dot = hn::ReduceSum(df, vacc);
                        for (; d < params_.head_dim; ++d) dot += qh[d] * kp[d];
                        scores[p] = dot / std::sqrt(static_cast<float>(params_.head_dim));
                    }
                    softmax(scores, pos + 1);
                    float* oh = ob + h * params_.head_dim;
                    for (int p = 0; p <= pos; ++p) {
                        const float* vp = v_cache + (size_t)p * params_.head_dim;
                        if (scores[p] <= -1e29f) continue;  // evicted token

                        const hn::ScalableTag<float> df;
                        auto vs = hn::Set(df, scores[p]);
                        int lanes = hn::Lanes(df);
                        int d = 0;
                        for (; d + lanes <= params_.head_dim; d += lanes)
                            hn::StoreU(hn::MulAdd(vs, hn::LoadU(df, vp+d),
                                       hn::LoadU(df, oh+d)), df, oh+d);
                        for (; d < params_.head_dim; ++d) oh[d] += scores[p] * vp[d];
                    }
                }
            }
        };
        int64_t t0_attn = 0;
        if (prof) t0_attn = prof_now_ns();
        pool_->parallel_for(0, batch_size, batch_loop);
        if (prof) {
            int64_t t_attn = prof_now_ns() - t0_attn;
            int64_t t_kv = (int64_t)t_kv_read_local;
            if (t_attn > t_kv) g_prof.attention_ns.fetch_add(t_attn - t_kv, std::memory_order_relaxed);
            g_prof.kv_read_ns.fetch_add((uint64_t)t_kv, std::memory_order_relaxed);
        }

        // h. O projection
        std::string o_name = "blk." + std::to_string(l) + ".attn_output.weight";
        const GGUFTensor* t_o = find_tensor(o_name);
        if (!t_o) { std::cerr << "weight_matmul_batch: tensor not found: " << o_name << std::endl; return false; }
        std::vector<uint8_t> packed_o;
        do_matmul(get_weight_ptr( t_o, packed_o), (int)t_o->type, attn_out_batch.data(), q_rows, attn_proj_batch.data(), n_embd, batch_size);
        std::string o_bias_name = "blk." + std::to_string(l) + ".attn_output.bias";
        const GGUFTensor* t_ob = find_tensor(o_bias_name);
        std::vector<float> o_bias_buf;
        std::vector<uint8_t> packed_ob;
        const float* o_b = t_ob ? get_weight_f32_raw(get_layer( t_ob, packed_ob), t_ob->type, t_ob->shape, o_bias_buf) : nullptr;
        for (int b = 0; b < batch_size; b++) {
            add_bias_raw(attn_proj_batch.data() + (size_t)b * n_embd, o_b, n_embd);
        }

        // i. Residual
        for (size_t i = 0; i + 8 <= hidden_batch.size(); i += 8) {
            __m256 vh = _mm256_loadu_ps(hidden_batch.data() + i);
            __m256 va = _mm256_loadu_ps(attn_proj_batch.data() + i);
            _mm256_storeu_ps(hidden_batch.data() + i, _mm256_add_ps(vh, va));
        }
        for (size_t i = (hidden_batch.size() / 8) * 8; i < hidden_batch.size(); i++) hidden_batch[i] += attn_proj_batch[i];

        // j. ffn RMS norm
        const float* ffn_norm_w = cached_norm_weights_[l].ffn_norm_w.data();
        for (int b = 0; b < batch_size; b++) {
            rms_norm(x_norm_batch.data() + (size_t)b * n_embd,
                     hidden_batch.data() + (size_t)b * n_embd,
                     ffn_norm_w, n_embd);
        }

        // k. FFN (SwiGLU): same batched dequant-once-per-row treatment.
        std::string gate_name = "blk." + std::to_string(l) + ".ffn_gate.weight";
        const GGUFTensor* t_gate = find_tensor(gate_name);
        if (!t_gate) { std::cerr << "weight_matmul_batch: tensor not found: " << gate_name << std::endl; return false; }
        std::vector<uint8_t> packed_gate;
        do_matmul(get_weight_ptr( t_gate, packed_gate), (int)t_gate->type, x_norm_batch.data(), n_embd, gate_batch.data(), n_ff, batch_size);
        std::string up_name = "blk." + std::to_string(l) + ".ffn_up.weight";
        const GGUFTensor* t_up = find_tensor(up_name);
        if (!t_up) { std::cerr << "weight_matmul_batch: tensor not found: " << up_name << std::endl; return false; }
        std::vector<uint8_t> packed_up;
        do_matmul(get_weight_ptr( t_up, packed_up), (int)t_up->type, x_norm_batch.data(), n_embd, up_batch.data(), n_ff, batch_size);

        for (size_t i = 0; i < gate_batch.size(); i++) {
            gate_batch[i] = gate_batch[i] / (1.0f + expf(-gate_batch[i]));
            ffn_mid_batch[i] = gate_batch[i] * up_batch[i];
        }

        std::string down_name = "blk." + std::to_string(l) + ".ffn_down.weight";
        const GGUFTensor* t_down = find_tensor(down_name);
        if (!t_down) { std::cerr << "weight_matmul_batch: tensor not found: " << down_name << std::endl; return false; }
        std::vector<uint8_t> packed_down;
        do_matmul(get_weight_ptr( t_down, packed_down), (int)t_down->type, ffn_mid_batch.data(), n_ff, down_batch.data(), n_embd, batch_size);

        // l. Residual
        for (size_t i = 0; i + 8 <= hidden_batch.size(); i += 8) {
            __m256 vh = _mm256_loadu_ps(hidden_batch.data() + i);
            __m256 vd = _mm256_loadu_ps(down_batch.data() + i);
            _mm256_storeu_ps(hidden_batch.data() + i, _mm256_add_ps(vh, vd));
        }
        for (size_t i = (hidden_batch.size() / 8) * 8; i < hidden_batch.size(); i++) hidden_batch[i] += down_batch[i];

        if (debug_layers_) {
            double norm2 = 0.0;
            double max_abs = 0.0;
            for (float v : hidden_batch) {
                double d = static_cast<double>(v);
                norm2 += d * d;
                if (std::abs(d) > max_abs) max_abs = std::abs(d);
            }
            std::cout << "blk " << std::setw(2) << l
                      << "  norm=" << std::fixed << std::setprecision(3) << std::sqrt(norm2)
                      << "  max_abs=" << std::fixed << std::setprecision(3) << max_abs
                      << std::endl;
        }

        if (prefetch.joinable()) prefetch.join();
        cur = 1 - cur;

    }

    // Final RMS norm + logits: only the last token's logits are needed by
    // the caller (the prompt tokens' logits are never sampled from).
    const float* output_norm_w = cached_output_norm_w_.data();

    state.hidden.resize(n_embd);
    rms_norm(state.hidden.data(), hidden_batch.data() + (size_t)(batch_size - 1) * n_embd,
             output_norm_w, n_embd);

    if (all_hidden) {
        all_hidden->resize((size_t)batch_size * n_embd);
        for (int b = 0; b < batch_size; ++b) {
            rms_norm(all_hidden->data() + (size_t)b * n_embd,
                     hidden_batch.data() + (size_t)b * n_embd,
                     output_norm_w, n_embd);
        }
    }

    if (compute_logits_last_only) {
        {
            ProfScope ps(g_prof.matmul_ns, prof);
            compute_logits(state.hidden.data(), state);
        }
    }
    pool_->leave_team();

    if (prof) g_prof.enabled.store(0, std::memory_order_relaxed);
    return true;
}

const GGUFTensor* Transformer::find_tensor(const std::string& name) const {
    auto it = tensor_index_.find(name);
    return it != tensor_index_.end() ? it->second : nullptr;
}

void Transformer::weight_matmul(const std::string& name,
                                const float* x, int n_cols,
                                float* out, int n_rows) const {
    const GGUFTensor* t = find_tensor(name);
    if (!t) {
        std::cerr << "weight_matmul: tensor not found: " << name << std::endl;
        return;
    }

    const uint8_t* raw = parser_->data_ptr() + t->offset;

    if (t->type == GGMLType::F32) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    n_rows, n_cols, 1.0f,
                    (const float*)raw, n_cols,
                    x, 1, 0.0f, out, 1);
        return;
    }

    // Use ggml dequantization for quantized types.
    // Always parallelize the row loop: K/Q/V/O projections (512-3584 rows)
    // used to fall below an arbitrary 1024-row threshold and run single
    // threaded, wasting 7 of 8 cores on those matmuls every single token.
    int n_blocks_per_row = n_cols / 256;  // Q4_K and Q6_K: 256 values/block
    int n_blocks_q8 = n_cols / 32;        // Q8_0: 32 values/block

    // Quantize activations ONCE per matmul call, outside the row loop.
    std::vector<block_q8_K> x_q8(n_blocks_per_row);
    if (t->type == GGMLType::Q4_K || t->type == GGMLType::Q6_K) {
        quantize_row_q8k(x, x_q8.data(), n_blocks_per_row);
    }

    auto row_loop = [&](int r0, int r1) {
        for (int r = r0; r < r1; ++r) {
            switch (t->type) {
                case GGMLType::Q4_K:
                    out[r] = vec_dot_q4k_q8k(
                        (const block_q4_K*)(raw + r * n_blocks_per_row * 144),
                        x_q8.data(), n_blocks_per_row);
                    continue;
                case GGMLType::Q6_K:
                    out[r] = vec_dot_q6k_q8k(
                        (const block_q6_K*)(raw + r * n_blocks_per_row * 210),
                        x_q8.data(), n_blocks_per_row);
                    continue;
                case GGMLType::Q8_0:
                    out[r] = vec_dot_q8_0(raw + r * n_blocks_q8 * 34, x, n_blocks_q8);
                    continue;
                default:
                    out[r] = 0.0f;
                    continue;
            }
        }
    };
    pool_->parallel_for(0, n_rows, row_loop);
}

void Transformer::weight_matmul_q8(const std::string& name,
                                    const float* x, int n_cols,
                                    const void* x_q8, int n_blocks_per_row,
                                    float* out, int n_rows) const {
    const GGUFTensor* t = find_tensor(name);
    if (!t) {
        std::cerr << "weight_matmul_q8: tensor not found: " << name << std::endl;
        return;
    }

    const uint8_t* raw = parser_->data_ptr() + t->offset;

    if (t->type == GGMLType::F32) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    n_rows, n_cols, 1.0f,
                    (const float*)raw, n_cols,
                    x, 1, 0.0f, out, 1);
        return;
    }

    int n_blocks_q8 = n_cols / 32;
    const block_q8_K* q8 = (const block_q8_K*)x_q8;

    auto row_loop = [&](int r0, int r1) {
        for (int r = r0; r < r1; ++r) {
            switch (t->type) {
                case GGMLType::Q4_K:
                    out[r] = vec_dot_q4k_q8k(
                        (const block_q4_K*)(raw + (size_t)r * n_blocks_per_row * 144),
                        q8, n_blocks_per_row);
                    continue;
                case GGMLType::Q6_K:
                    out[r] = vec_dot_q6k_q8k(
                        (const block_q6_K*)(raw + (size_t)r * n_blocks_per_row * 210),
                        q8, n_blocks_per_row);
                    continue;
                case GGMLType::Q8_0:
                    out[r] = vec_dot_q8_0(raw + (size_t)r * n_blocks_q8 * 34, x, n_blocks_q8);
                    continue;
                default:
                    out[r] = 0.0f;
                    continue;
            }
        }
    };
    pool_->parallel_for(0, n_rows, row_loop);
}

void Transformer::weight_matmul_batch(const std::string& name,
                                       const float* x_batch, int n_cols,
                                       float* out_batch, int n_rows,
                                       int batch_size) const {
    const GGUFTensor* t = find_tensor(name);
    if (!t) {
        std::cerr << "weight_matmul_batch: tensor not found: " << name << std::endl;
        return;
    }
    weight_matmul_batch_raw(this, parser_->data_ptr() + t->offset, (int)t->type,
                            x_batch, n_cols, out_batch, n_rows, batch_size);
}

float Transformer::vec_dot_q4k(const uint8_t* data,
                                const float* x, int n_blocks) const {
    float buf[256];
    float acc = 0.0f;
    for (int b = 0; b < n_blocks; b++) {
        dequantize_row_q4_K((const block_q4_K*)(data + b * 144), buf, 256);
        const hn::ScalableTag<float> df;
        auto vacc = hn::Zero(df);
        int lanes = hn::Lanes(df);
        int i = 0;
        for (; i + lanes <= 256; i += lanes) {
            auto vb = hn::LoadU(df, buf + i);
            auto vx = hn::LoadU(df, x + b * 256 + i);
            vacc = hn::MulAdd(vb, vx, vacc);
        }
        acc += hn::ReduceSum(df, vacc);
        for (; i < 256; i++) acc += buf[i] * x[b * 256 + i];
    }
    return acc;
}

float Transformer::vec_dot_q6k(const uint8_t* data,
                                const float* x, int n_blocks) const {
    float buf[256];
    float acc = 0.0f;
    for (int b = 0; b < n_blocks; b++) {
        dequantize_row_q6_K((const block_q6_K*)(data + b * 210), buf, 256);
        const hn::ScalableTag<float> df;
        auto vacc = hn::Zero(df);
        int lanes = hn::Lanes(df);
        int i = 0;
        for (; i + lanes <= 256; i += lanes) {
            auto vb = hn::LoadU(df, buf + i);
            auto vx = hn::LoadU(df, x + b * 256 + i);
            vacc = hn::MulAdd(vb, vx, vacc);
        }
        acc += hn::ReduceSum(df, vacc);
        for (; i < 256; i++) acc += buf[i] * x[b * 256 + i];
    }
    return acc;
}

float Transformer::vec_dot_q8_0(const uint8_t* data,
                                 const float* x, int n_blocks) const {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    for (int b = 0; b < n_blocks; b++) {
        float d = f16_to_f32(*(const uint16_t*)(data + b*34));
        const int8_t* qs = (const int8_t*)(data + b*34 + 2);

        // Load 32 int8 quants: no scalar loop
        __m128i q_lo = _mm_loadu_si128((__m128i*)(qs));
        __m128i q_hi = _mm_loadu_si128((__m128i*)(qs + 16));

        // Sign-extend int8 → int32 in 256-bit registers
        __m256i qi_lo = _mm256_cvtepi8_epi32(q_lo);
        __m256i qi_hi = _mm256_cvtepi8_epi32(q_hi);

        // Convert to float32
        __m256 qf_lo = _mm256_cvtepi32_ps(qi_lo);
        __m256 qf_hi = _mm256_cvtepi32_ps(qi_hi);

        // Load x values
        __m256 xf_lo = _mm256_loadu_ps(x + b*32);
        __m256 xf_hi = _mm256_loadu_ps(x + b*32 + 16);

        __m256 vd = _mm256_set1_ps(d);
        acc0 = _mm256_fmadd_ps(vd, _mm256_mul_ps(qf_lo, xf_lo), acc0);
        acc1 = _mm256_fmadd_ps(vd, _mm256_mul_ps(qf_hi, xf_hi), acc1);
    }
    __m256 acc = _mm256_add_ps(acc0, acc1);
    // Horizontal sum of 8 floats
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

void Transformer::get_embedding_row(int32_t token_id, std::vector<float>& out) const {
    const GGUFTensor* tensor = find_tensor("token_embd.weight");
    if (!tensor) {
        std::cerr << "token_embd.weight not found" << std::endl;
        return;
    }

    const uint8_t* data = parser_->data_ptr() + tensor->offset;
    size_t n_embd = params_.n_embd;
    size_t row_start_elem = static_cast<size_t>(token_id) * n_embd;

    out.resize(n_embd);

    if (tensor->type == GGMLType::F32) {
        // Direct mmap access
        const float* f32_data = reinterpret_cast<const float*>(data);
        std::memcpy(out.data(), f32_data + row_start_elem, n_embd * sizeof(float));
    } else if (tensor->type == GGMLType::Q4_K) {
        // Use ggml dequantization
        size_t row_start_block = row_start_elem / 256;
        const uint8_t* row_ptr = data + row_start_block * 144;
        dequantize_row_q4_K((const block_q4_K*)row_ptr, out.data(), n_embd);
    } else if (tensor->type == GGMLType::F16) {
        const uint16_t* f16_data = reinterpret_cast<const uint16_t*>(data);
        const uint16_t* row_ptr = f16_data + row_start_elem;
        for (size_t i = 0; i < n_embd; ++i) {
            out[i] = GGML_FP16_TO_FP32(row_ptr[i]);
        }
    } else {
        std::cerr << "Unsupported tensor type for embedding: " << static_cast<int>(tensor->type) << std::endl;
    }
}

void Transformer::compute_logits(const float* hidden, InferenceState& state) const {
    const GGUFTensor* t = find_tensor("output.weight");
    if (!t) t = find_tensor("token_embd.weight");
    if (!t) {
        std::cerr << "output.weight / token_embd.weight not found" << std::endl;
        return;
    }

    const uint8_t* raw = parser_->data_ptr() + t->offset;
    int n_cols = params_.n_embd;
    int n_rows = params_.n_vocab;
    int n_blocks_per_row = n_cols / 256;
    int n_blocks_q8 = n_cols / 32;

    std::vector<block_q8_K> x_q8;
    if (t->type == GGMLType::Q4_K || t->type == GGMLType::Q6_K) {
        x_q8.resize(n_blocks_per_row);
        quantize_row_q8k(hidden, x_q8.data(), n_blocks_per_row);
    }

    // Per-thread local max, merged under mutex after each thread's range.
    // Mutex acquired once per thread, not per row: contention is negligible.
    std::mutex best_mutex;
    float best_val = -INFINITY;
    int32_t best_idx = -1;

    auto row_loop = [&](int r0, int r1) {
        float local_best_val = -INFINITY;
        int32_t local_best_idx = -1;
        for (int r = r0; r < r1; ++r) {
            float val;
            switch (t->type) {
                case GGMLType::Q6_K:
                    val = ns_vec_dot_q6k_q8k(
                        (const block_q6_K*)(raw + (size_t)r * n_blocks_per_row * 210),
                        x_q8.data(), n_blocks_per_row);
                    break;
                case GGMLType::Q4_K:
                    val = vec_dot_q4k_q8k(
                        (const block_q4_K*)(raw + (size_t)r * n_blocks_per_row * 144),
                        x_q8.data(), n_blocks_per_row);
                    break;
                case GGMLType::Q8_0:
                    val = vec_dot_q8_0(
                        raw + (size_t)r * n_blocks_q8 * 34, hidden, n_blocks_q8);
                    break;
                case GGMLType::F32:
                    val = cblas_sdot(n_cols,
                        (const float*)raw + r * n_cols, 1, hidden, 1);
                    break;
                default:
                    val = 0.0f;
            }
            if (val > local_best_val) {
                local_best_val = val;
                local_best_idx = r;
            }
        }
        std::lock_guard<std::mutex> lock(best_mutex);
        if (local_best_val > best_val) {
            best_val = local_best_val;
            best_idx = local_best_idx;
        }
    };
    pool_->parallel_for(0, n_rows, row_loop);

    state.sampled_token = best_idx;
}

void Transformer::compute_logits_full(const float* hidden, InferenceState& state) const {
    const GGUFTensor* t = find_tensor("output.weight");
    if (!t) t = find_tensor("token_embd.weight");
    if (!t) {
        std::cerr << "output.weight / token_embd.weight not found" << std::endl;
        return;
    }

    const uint8_t* raw = parser_->data_ptr() + t->offset;
    state.logits.resize(params_.n_vocab);
    int n_cols = params_.n_embd;
    int n_rows = params_.n_vocab;
    int n_blocks_per_row = n_cols / 256;
    int n_blocks_q8 = n_cols / 32;

    std::vector<block_q8_K> x_q8;
    if (t->type == GGMLType::Q4_K || t->type == GGMLType::Q6_K) {
        x_q8.resize(n_blocks_per_row);
        quantize_row_q8k(hidden, x_q8.data(), n_blocks_per_row);
    }

    auto row_loop = [&](int r0, int r1) {
        for (int r = r0; r < r1; ++r) {
            switch (t->type) {
                case GGMLType::Q6_K:
                    state.logits[r] = ns_vec_dot_q6k_q8k(
                        (const block_q6_K*)(raw + (size_t)r * n_blocks_per_row * 210),
                        x_q8.data(), n_blocks_per_row);
                    break;
                case GGMLType::Q4_K:
                    state.logits[r] = vec_dot_q4k_q8k(
                        (const block_q4_K*)(raw + (size_t)r * n_blocks_per_row * 144),
                        x_q8.data(), n_blocks_per_row);
                    break;
                case GGMLType::Q8_0:
                    state.logits[r] = vec_dot_q8_0(
                        raw + (size_t)r * n_blocks_q8 * 34, hidden, n_blocks_q8);
                    break;
                case GGMLType::F32:
                    state.logits[r] = cblas_sdot(n_cols,
                        (const float*)raw + r * n_cols, 1, hidden, 1);
                    break;
                default:
                    state.logits[r] = 0.0f;
            }
        }
    };
    pool_->parallel_for(0, n_rows, row_loop);
}

void Transformer::compute_logits_batch(
    const float* all_hidden,
    int n_agents,
    std::vector<std::vector<float>>& all_logits) const {
    const GGUFTensor* t = find_tensor("output.weight");
    if (!t) t = find_tensor("token_embd.weight");
    if (!t) {
        std::cerr << "output.weight / token_embd.weight not found" << std::endl;
        return;
    }

    const uint8_t* raw = parser_->data_ptr() + t->offset;
    int n_cols = params_.n_embd;
    int n_rows = params_.n_vocab;
    int n_blocks_per_row = n_cols / 256;
    int n_blocks_q8 = n_cols / 32;

    all_logits.assign(n_agents, std::vector<float>(n_rows));

    std::vector<block_q8_K> x_q8_batch;
    if (t->type == GGMLType::Q4_K || t->type == GGMLType::Q6_K) {
        x_q8_batch.resize((size_t)n_agents * n_blocks_per_row);
        for (int i = 0; i < n_agents; ++i) {
            quantize_row_q8k(all_hidden + (size_t)i * n_cols,
                             x_q8_batch.data() + (size_t)i * n_blocks_per_row,
                             n_blocks_per_row);
        }
    }

    auto row_loop = [&](int r0, int r1) {
        for (int r = r0; r < r1; ++r) {
            switch (t->type) {
                case GGMLType::Q6_K: {
                    const block_q6_K* row = (const block_q6_K*)(raw + (size_t)r * n_blocks_per_row * 210);
                    for (int i = 0; i < n_agents; ++i)
                        all_logits[i][r] = ns_vec_dot_q6k_q8k(
                            row, x_q8_batch.data() + (size_t)i * n_blocks_per_row, n_blocks_per_row);
                    break;
                }
                case GGMLType::Q4_K: {
                    const block_q4_K* row = (const block_q4_K*)(raw + (size_t)r * n_blocks_per_row * 144);
                    for (int i = 0; i < n_agents; ++i)
                        all_logits[i][r] = vec_dot_q4k_q8k(
                            row, x_q8_batch.data() + (size_t)i * n_blocks_per_row, n_blocks_per_row);
                    break;
                }
                case GGMLType::Q8_0: {
                    const uint8_t* row_data = raw + (size_t)r * n_blocks_q8 * 34;
                    for (int i = 0; i < n_agents; ++i)
                        all_logits[i][r] = vec_dot_q8_0(
                            row_data, all_hidden + (size_t)i * n_cols, n_blocks_q8);
                    break;
                }
                case GGMLType::F32: {
                    const float* row = (const float*)raw + (size_t)r * n_cols;
                    for (int i = 0; i < n_agents; ++i)
                        all_logits[i][r] = cblas_sdot(n_cols, row, 1,
                                                      all_hidden + (size_t)i * n_cols, 1);
                    break;
                }
                default:
                    for (int i = 0; i < n_agents; ++i)
                        all_logits[i][r] = 0.0f;
            }
        }
    };
    pool_->parallel_for(0, n_rows, row_loop);
}

void Transformer::grow_kv_cache(int new_capacity) {
    new_capacity = std::min(new_capacity, (int)params_.max_seq_len);
    size_t new_size = (size_t)params_.n_layers * new_capacity
                      * params_.n_kv_heads * params_.head_dim;
    k_cache_.resize(new_size, 0.0f);
    v_cache_.resize(new_size, 0.0f);
    kv_cache_capacity_ = new_capacity;
}

void Transformer::add_bias(float* out, const std::string& bias_name, int n) const {
    const GGUFTensor* t = find_tensor(bias_name);
    if (!t) return;  // no bias for this layer: skip silently
    if (t->type != GGMLType::F32) {
        fprintf(stderr, "Warning: bias %s is not F32\n", bias_name.c_str());
        return;
    }
    const float* b = (const float*)(parser_->data_ptr() + t->offset);
    for (int i = 0; i < n; i++) out[i] += b[i];
}

bool Transformer::validate_vec_dot_q6k() const {
    const GGUFTensor* t = find_tensor("blk.0.ffn_down.weight");
    if (!t || t->type != GGMLType::Q6_K) {
        std::cerr << "[VALIDATE] blk.0.ffn_down.weight not found or not Q6_K" << std::endl;
        return false;
    }

    const uint8_t* raw = parser_->data_ptr() + t->offset;
    int n_embd = params_.n_embd;
    int n_blocks_per_row = n_embd / 256;

    // Create test vector filled with 1.0f
    std::vector<float> x(n_embd, 1.0f);

    // Compute AVX-512 vec_dot_q6k for row 0
    float avx_result = vec_dot_q6k(raw, x.data(), n_blocks_per_row);

    // Compute scalar reference by dequantizing row 0
    std::vector<float> dequant_row(n_embd);
    const uint8_t* row_data = raw;

    for (int block = 0; block < n_blocks_per_row; block++) {
        dequantize_row_q6_K((const block_q6_K*)(row_data + block * 210),
                            dequant_row.data() + block * 256, 256);
    }

    // Scalar dot product
    float scalar_result = 0.0f;
    for (int i = 0; i < n_embd; i++) {
        scalar_result += dequant_row[i] * x[i];
    }

    // Check match with tolerance
    bool match = std::abs(avx_result - scalar_result) < 1e-2f;
    printf("[VALIDATE] AVX-512: %.8f  Scalar: %.8f  Match: %s\n",
           avx_result, scalar_result, match ? "YES" : "NO");

    return match;
}

bool Transformer::validate_vec_dot_q4k() const {
    const GGUFTensor* t = find_tensor("blk.0.ffn_gate.weight");
    if (!t || t->type != GGMLType::Q4_K) {
        std::cerr << "[VALIDATE] blk.0.ffn_gate.weight not found or not Q4_K" << std::endl;
        return false;
    }

    const uint8_t* raw = parser_->data_ptr() + t->offset;
    int n_embd = params_.n_embd;

    // Create test vector filled with 1.0f
    std::vector<float> x(n_embd, 1.0f);

    // Dequantize row 0 using ggml
    std::vector<float> dequant_row(n_embd);
    dequantize_row_q4_K((const block_q4_K*)raw, dequant_row.data(), n_embd);

    // Compute dot product
    float result = 0.0f;
    for (int i = 0; i < n_embd; i++) {
        result += dequant_row[i] * x[i];
    }

    // Compute the same dot using the fused q4k/q8k kernel
    int n_blocks_per_row = n_embd / 256;
    std::vector<block_q8_K> x_q8(n_blocks_per_row);
    quantize_row_q8k(x.data(), x_q8.data(), n_blocks_per_row);
    float avx_result = vec_dot_q4k_q8k((const block_q4_K*)raw, x_q8.data(), n_blocks_per_row);

    bool match = std::abs(avx_result - result) < 1e-2f;
    printf("[VALIDATE] AVX-512: %.8f  Scalar: %.8f  Match: %s\n",
           avx_result, result, match ? "YES" : "NO");

    return match;
}

// -----------------------------------------------------------------------------
// Weight-repack support (Q4_Kx8 panels, lazy per-tensor with original-page drop)
// -----------------------------------------------------------------------------

void Transformer::drop_original_pages(const GGUFTensor* t, size_t original_size) const {
    if (!parser_ || !t || original_size == 0) return;
    if (keep_original_names_.find(t->name) != keep_original_names_.end()) return;
    if (parser_->is_nsm()) return;  // q4k_data_ is heap-owned; MADV_DONTNEED would zero it permanently

    const uint8_t* ptr = parser_->data_ptr() + t->offset;
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) return;

    uintptr_t start = (uintptr_t)ptr;
    uintptr_t end   = start + original_size;
    uintptr_t aligned_start = (start + (uintptr_t)page_size - 1) & ~((uintptr_t)page_size - 1);
    uintptr_t aligned_end   = end & ~((uintptr_t)page_size - 1);

    if (aligned_start < aligned_end) {
        uint8_t* base = const_cast<uint8_t*>(ptr);
        madvise(base + (aligned_start - start), aligned_end - aligned_start, MADV_DONTNEED);
    }
}

const Transformer::RepackedBuffer* Transformer::get_repacked_buffer_llama(const GGUFTensor* t) const {
    if (!t) return nullptr;

    auto it = repacked_weights_llama_.find(t->name);
    if (it != repacked_weights_llama_.end()) {
        return &it->second;
    }

    if (t->type != GGMLType::Q4_K && t->type != GGMLType::Q6_K) return nullptr;
    if (t->type == GGMLType::Q6_K) return nullptr; // Skip Q6_K repack (scalar panel kernel is too slow)
    if (t->shape.size() != 2) return nullptr;

    uint64_t n_cols = t->shape[0];
    uint64_t n_rows = t->shape[1];
    if (n_cols % QK_K != 0) return nullptr;

    size_t n_blocks_per_row = (size_t)(n_cols / QK_K);
    size_t n_panels = ((size_t)n_rows + 7) / 8;
    size_t panel_bytes = n_panels * n_blocks_per_row * sizeof(ns_q4_Kx8);

    RepackedBuffer buf;
    buf.n_rows = n_rows;
    buf.n_blocks_per_row = n_blocks_per_row;
    buf.R = 8;
    buf.weight_type = (int)t->type;

    const uint8_t* original_ptr = parser_->data_ptr() + t->offset;
    size_t n_panel_rows = n_panels * 8;
    size_t n_tail = (n_panel_rows > n_rows) ? 0 : (n_rows - n_panel_rows); // guard underflow

    if (parser_->is_nsm() && t->name != "token_embd.weight") {
        // .nsm already stores Q4_K panels in the loader's aligned buffer.
        buf.data_ptr = original_ptr;
        buf.data_size = panel_bytes;
    } else if (t->type == GGMLType::Q4_K) {
        buf.data.resize(panel_bytes, 0);
        buf.data_ptr = buf.data.data();
        buf.data_size = buf.data.size();

        const block_q4_K* src = (const block_q4_K*)original_ptr;
        ns_q4_Kx8* dst = (ns_q4_Kx8*)buf.data.data();

        for (size_t p = 0; p < n_panels; p++) {
            size_t r0 = p * 8;
            size_t rows_in_panel = std::min<size_t>(8, n_rows - r0);
            const block_q4_K* panel_src = src + r0 * n_blocks_per_row;
            repack_q4_K_row_panel_llama(panel_src, (int)rows_in_panel, (int)n_blocks_per_row,
                                        dst + p * n_blocks_per_row, 8);
        }

        if (n_tail > 0) {
            buf.n_tail_rows = n_tail;
            buf.tail_data.resize(n_tail * n_blocks_per_row * sizeof(block_q4_K));
            const uint8_t* tail_src = original_ptr + (size_t)n_panel_rows * n_blocks_per_row * sizeof(block_q4_K);
            memcpy(buf.tail_data.data(), tail_src, n_tail * n_blocks_per_row * sizeof(block_q4_K));
        }
    } else { // Q6_K
        buf.data.resize(panel_bytes, 0);
        buf.data_ptr = buf.data.data();
        buf.data_size = buf.data.size();

        const block_q6_K* src = (const block_q6_K*)original_ptr;
        ns_q6_Kx8* dst = (ns_q6_Kx8*)buf.data.data();

        for (size_t p = 0; p < n_panels; p++) {
            size_t r0 = p * 8;
            size_t rows_in_panel = std::min<size_t>(8, n_rows - r0);
            const block_q6_K* panel_src = src + r0 * n_blocks_per_row;
            repack_q6_K_row_panel(panel_src, (int)rows_in_panel, (int)n_blocks_per_row,
                                  dst + p * n_blocks_per_row, 8);
        }

        if (n_tail > 0) {
            buf.n_tail_rows = n_tail;
            buf.tail_data.resize(n_tail * n_blocks_per_row * sizeof(block_q6_K));
            const uint8_t* tail_src = original_ptr + (size_t)n_panel_rows * n_blocks_per_row * sizeof(block_q6_K);
            memcpy(buf.tail_data.data(), tail_src, n_tail * n_blocks_per_row * sizeof(block_q6_K));
        }
    }

    auto emplace_res = repacked_weights_llama_.emplace(t->name, std::move(buf));
    const RepackedBuffer* result = &emplace_res.first->second;

    size_t elem_size = (t->type == GGMLType::Q4_K) ? sizeof(block_q4_K) : sizeof(block_q6_K);
    size_t original_size = n_rows * n_blocks_per_row * elem_size;

    if (keep_original_names_.find(t->name) == keep_original_names_.end() && !parser_->is_nsm()) {
        long page_size = sysconf(_SC_PAGESIZE);
        uintptr_t start = (uintptr_t)original_ptr;
        uintptr_t end   = start + original_size;
        uintptr_t aligned_start = (start + (uintptr_t)page_size - 1) & ~((uintptr_t)page_size - 1);
        uintptr_t aligned_end   = end & ~((uintptr_t)page_size - 1);
        if (aligned_start < aligned_end)
            madvise((void*)aligned_start, aligned_end - aligned_start, MADV_DONTNEED);
    }
    return result;
}

void Transformer::repack_all_weights() const {
    if (!parser_) return;

    size_t total_original = 0;
    size_t total_repacked = 0;
    size_t count = 0;

    for (const auto& t : parser_->tensors()) {
        if (t.name == "blk.0.ffn_gate.weight") {
            const uint8_t* raw = parser_->data_ptr() + t.offset;
            std::cout << "[REPACK RAW] " << t.name
                      << " out_q4k_base=" << (void*)parser_->data_ptr()
                      << " raw=" << (void*)raw
                      << " raw_offset=" << t.offset
                      << " expected_raw=" << (void*)(parser_->data_ptr() + 1257343232)
                      << std::endl;
        }
        const RepackedBuffer* buf = get_repacked_buffer_llama(&t);
        if (t.name == "blk.0.ffn_gate.weight" && buf) {
            std::cout << "[NS_GEMV_Q4K PANEL0 PTR] " << t.name
                      << " panel_base=" << (void*)buf->data_ptr
                      << " n_rows=" << buf->n_rows
                      << " n_blocks_per_row=" << buf->n_blocks_per_row
                      << std::endl;
        }
        if (buf) {
            size_t orig = buf->n_rows * buf->n_blocks_per_row *
                          (buf->weight_type == (int)GGMLType::Q4_K ? sizeof(block_q4_K) : sizeof(block_q6_K));
            total_original += orig;
            total_repacked += buf->data_size;
            count++;
        }
    }

}

bool Transformer::validate_repack_q4k() const {
    const GGUFTensor* t = find_tensor("blk.0.ffn_gate.weight");
    if (!t || t->type != GGMLType::Q4_K) {
        for (const auto& candidate : parser_->tensors()) {
            if (candidate.type == GGMLType::Q4_K &&
                candidate.shape.size() == 2 &&
                candidate.shape[1] >= 8) {
                t = &candidate;
                break;
            }
        }
    }
    if (!t) {
        std::cerr << "[VALIDATE] no Q4_K tensor found for repack validation" << std::endl;
        return false;
    }
    if (t->shape.size() != 2 || t->shape[0] % QK_K != 0 || t->shape[1] < 8) {
        std::cerr << "[VALIDATE] invalid Q4_K tensor shape" << std::endl;
        return false;
    }

    uint64_t n_cols = t->shape[0];
    size_t n_blocks_per_row = (size_t)(n_cols / QK_K);

    // Copy the first 8 rows out of the mmap so the validator does not touch
    // or free the live model weights.
    std::vector<block_q4_K> src(8 * n_blocks_per_row);
    const block_q4_K* raw = (const block_q4_K*)(parser_->data_ptr() + t->offset);
    memcpy(src.data(), raw, 8 * n_blocks_per_row * sizeof(block_q4_K));

    std::vector<ns_q4_Kx8, AlignedAllocator<ns_q4_Kx8, 64>> dst(n_blocks_per_row);
    repack_q4_K_row_panel_llama(src.data(), 8, (int)n_blocks_per_row, dst.data(), 8);

    // Deterministic test activation.
    std::vector<float> x(n_cols);
    for (size_t i = 0; i < n_cols; i++) {
        x[i] = (float)((i % 31) - 15) * 0.05f;
    }
    std::vector<block_q8_K> x_q8(n_blocks_per_row);
    quantize_row_q8k(x.data(), x_q8.data(), (int)n_blocks_per_row);

    // 8 independent reference dot products from the original blocks.
    float expected[8];
    for (int r = 0; r < 8; r++) {
        expected[r] = vec_dot_q4k_q8k(src.data() + (size_t)r * n_blocks_per_row,
                                      x_q8.data(), (int)n_blocks_per_row);
    }

    // Panel dot product from the repacked buffer using the llama-layout AVX2 kernel.
    float got[8] = {0};
    ns_gemv_q4k(dst.data(), x_q8.data(), 8, (int)n_blocks_per_row, got);

    bool all_match = true;
    float max_rel_err = 0.0f;
    for (int r = 0; r < 8; r++) {
        float err = std::abs(expected[r] - got[r]);
        float ref = std::max(1.0f, std::abs(expected[r]));
        float rel_err = err / ref;
        if (rel_err > max_rel_err) max_rel_err = rel_err;
        if (rel_err > 1e-2f) {
            all_match = false;
            printf("[VALIDATE] row %d expected %.8f got %.8f err %.8f rel_err %.8f\n",
                   r, expected[r], got[r], err, rel_err);
        }
    }
    printf("[VALIDATE] repack_q4k (llama layout + ns_gemv_q4k): %s (max rel err %.8f, orig %zu bytes, repacked %zu bytes)\n",
           all_match ? "PASS" : "FAIL", max_rel_err, 8 * n_blocks_per_row * sizeof(block_q4_K),
           n_blocks_per_row * sizeof(ns_q4_Kx8));

    return all_match;
}

bool Transformer::forward_hive_batch(
    const std::vector<std::vector<int32_t>>& agent_tokens,
    const std::vector<int>& agent_positions,
    std::vector<InferenceState*>& agent_states,
    std::vector<std::vector<float>>& agent_logits) {
    int n_agents = static_cast<int>(agent_tokens.size());
    if (n_agents == 0) return true;

    // 1. Flatten one token per agent into a single batch vector.
    std::vector<int32_t> flat;
    flat.reserve(n_agents);
    for (const auto& toks : agent_tokens) {
        if (toks.empty()) return false;
        flat.push_back(toks[0]);
    }

    // 2. Ensure each agent has its own per-call KV store.
    std::vector<NSKVStore*> per_token_kv_stores;
    std::vector<int> per_token_positions;
    per_token_kv_stores.reserve(n_agents);
    per_token_positions.reserve(n_agents);
    NSKVConfig kv_cfg;
    kv_cfg.n_layers   = params_.n_layers;
    kv_cfg.n_kv_heads = params_.n_kv_heads;
    kv_cfg.head_dim   = params_.head_dim;
    kv_cfg.capacity   = (size_t)kv_cache_capacity_;
    kv_cfg.latent_dim = 0;
    kv_cfg.debug_float32 = false;
    for (int i = 0; i < n_agents; ++i) {
        if (i < (int)agent_states.size() && agent_states[i]) {
            if (!agent_states[i]->kv_store) {
                agent_states[i]->kv_store = std::make_shared<NSKVStore>(kv_cfg);
            }
            per_token_kv_stores.push_back(agent_states[i]->kv_store.get());
        } else {
            per_token_kv_stores.push_back(nullptr);
        }
        per_token_positions.push_back(i < (int)agent_positions.size() ? agent_positions[i] : 0);
    }

    // 3. Run one combined forward pass.
    int start_pos = *std::min_element(agent_positions.begin(), agent_positions.end());
    std::vector<float> all_hidden;
    InferenceState combined_state;
    if (!forward_batch(flat, start_pos, combined_state, false, &all_hidden,
                       &per_token_kv_stores, &per_token_positions)) {
        return false;
    }

    // 3. Compute per-agent logits from the shared pass.
    compute_logits_batch(all_hidden.data(), n_agents, agent_logits);
    for (int i = 0; i < n_agents; ++i) {
        if (agent_states[i]) agent_states[i]->logits = agent_logits[i];
    }

    // KV cache is updated by forward_batch() in the single shared k/v cache
    // (valid only while all agents share a position).
    return true;
}

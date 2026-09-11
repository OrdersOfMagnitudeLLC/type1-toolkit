#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "gguf_parser.h"
#include "ns_schedule.h"
#include "ns_repack.h"
#include "ns_kv_store.h"
#include <vector>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <string>

struct InferenceState {
    std::vector<float> logits;       // [n_vocab]
    std::vector<float> hidden;       // [n_embd] current hidden state
    int32_t sampled_token = -1;      // set by compute_logits() argmax-only path
    // Per-call / per-agent KV store. forward_batch() uses this before
    // falling back to Transformer::kv_store_.
    std::shared_ptr<NSKVStore> kv_store;
};

class ThreadPool;

enum class ActivationType { RELU, SILU_SWIGLU, GELU };

class Transformer {
public:
    explicit Transformer(size_t num_threads = 0);
    ~Transformer();

    const char* activation_name() const;

    bool load(const GGUFParser& parser, int context_len = 2048);
    bool load(const std::string& model_path, int context_len = 2048);
    bool forward(int32_t token_id, int pos, InferenceState& state, bool compute_logits_flag = true);
    // Batched forward for prefill: dequantizes each weight row ONCE per layer
    // and reuses it across every position in the batch instead of once per token.
    // If all_hidden is non-null, fills it with token-major [batch_size * n_embd].
    // Optional per-token kv_stores and positions for HIVE-style independent agents.
    bool forward_batch(const std::vector<int32_t>& token_ids, int start_pos,
                        InferenceState& state, bool compute_logits_last_only = true,
                        std::vector<float>* all_hidden = nullptr,
                        std::vector<NSKVStore*>* per_token_kv_stores = nullptr,
                        const std::vector<int>* per_token_positions = nullptr);
    // HIVE batched inference: N agents, one token/position/KV-cache each.
    // One combined forward_batch() weight read -> N per-agent logits.
    bool forward_hive_batch(const std::vector<std::vector<int32_t>>& agent_tokens,
                            const std::vector<int>& agent_positions,
                            std::vector<InferenceState*>& agent_states,
                            std::vector<std::vector<float>>& agent_logits);
    bool validate_vec_dot_q6k() const;
    bool validate_vec_dot_q4k() const;
    bool validate_repack_q4k() const;
    void repack_all_weights() const;
    int get_n_embd() const { return params_.n_embd; }
    void set_debug_layers(bool v) { debug_layers_ = v; }
    const GGUFTensor* find_tensor(const std::string& name) const;
    float vec_dot_q4k(const uint8_t* block_ptr, const float* x, int n_blocks) const;
    float vec_dot_q6k(const uint8_t* block_ptr, const float* x, int n_blocks) const;
    float vec_dot_q8_0(const uint8_t* block_ptr, const float* x, int n_blocks) const;

private:
    void add_bias(float* out, const std::string& bias_name, int n) const;

private:
    const GGUFParser* parser_;
    struct Params {
        int n_layers;
        int n_heads;
        int n_kv_heads;
        int n_embd;
        int n_ff;
        int n_vocab;
        int head_dim;
        int max_seq_len;
        float rope_freq_base;
        float rope_freq_scale;
    } params_;

    // KV cache (dynamic allocation)
    std::vector<float> k_cache_;
    std::vector<float> v_cache_;
    int kv_cache_capacity_;

    // NSKVStore: INT4 quantized KV cache with eviction
    std::unique_ptr<NSKVStore> kv_store_;

    // O(1) tensor lookup by name, built once in load() instead of the
    // O(n) linear scan that used to run on every weight access.
    std::unordered_map<std::string, const GGUFTensor*> tensor_index_;

    // Owned parser when loaded from a file path; parser_ points to this.
    std::unique_ptr<GGUFParser> parser_owned_;

public:
    // Lazy per-tensor repacked Q4_K/Q6_K weight panels.  mutable because
    // get_repacked_buffer() may be called from const matmul paths.
    struct RepackedBuffer {
        std::vector<uint8_t, AlignedAllocator<uint8_t, 64>> data;
        const uint8_t* data_ptr = nullptr;
        size_t data_size = 0;
        size_t n_rows = 0;
        size_t n_blocks_per_row = 0;
        int R = 0;
        int weight_type = 0; // GGMLType cast to int
        std::vector<uint8_t, AlignedAllocator<uint8_t, 64>> tail_data;
        size_t n_tail_rows = 0;
    };
    // Cache of Q4_K panels repacked with llama.cpp's raw byte
    // interleave (make_block_q4_Kx8 / blck_size_interleave=8), required by
    // the ported ggml_gemv_q4_K_8x8_q8_K AVX2 kernel (ns_gemv_q4k). This is
    // NOT byte-compatible with repacked_weights_ above (see ns_repack.h).
    mutable std::unordered_map<std::string, RepackedBuffer> repacked_weights_llama_;
    const RepackedBuffer* get_repacked_buffer_llama(const GGUFTensor* t) const;

    // Tensor names whose original mmap pages must stay resident (e.g. token_embd
    // is needed for get_embedding_row_impl() even when it is also repacked).
    std::unordered_set<std::string> keep_original_names_;
    bool debug_layers_ = false;
    float nsinfer_threshold_ = 0.0f;
    ActivationType activation_ = ActivationType::SILU_SWIGLU;

    // Lazy per-tensor repack.  Returns cached buffer or builds it, optionally
    // freeing the original mmap pages (unless in keep_original_names_).
    void drop_original_pages(const GGUFTensor* t, size_t original_size) const;

    // Persistent per-token scratch buffers for forward(), sized once in
    // load() instead of being heap-allocated on every single token.
    std::vector<float> buf_x_norm_;
    std::vector<float> buf_q_;
    std::vector<float> buf_k_;
    std::vector<float> buf_v_;
    std::vector<float> buf_attn_out_;
    std::vector<float> buf_gate_;
    std::vector<float> buf_up_;
    std::vector<float> buf_ffn_mid_;
    std::vector<float> buf_attn_proj_;
    std::vector<float> buf_down_;

    // Flat dispatch schedule built once at load time.
    Schedule schedule_;
    RunCtx run_ctx_;
    EmbedArgs* embed_args_ = nullptr;

    // Extra q8 activation buffers and attention scratch space.
    std::vector<uint8_t> buf_attn_x_q8_;
    std::vector<uint8_t> buf_ffn_x_q8_;
    std::vector<uint8_t> buf_attn_out_q8_;
    std::vector<uint8_t> buf_ffn_mid_q8_;
    std::vector<uint8_t> buf_logits_x_q8_;
    std::vector<float> buf_scores_;
    std::vector<float> buf_v_cache_;  // [max_seq_len * head_dim] for fused KV attention reads

    // Cached norm weights — dequantized once at load time instead of every token.
    struct CachedNormWeights {
        std::vector<float> attn_norm_w;
        std::vector<float> ffn_norm_w;
    };
    std::vector<CachedNormWeights> cached_norm_weights_;
    std::vector<float> cached_output_norm_w_;

    // RoPE precomputed tables for fast position encoding
    std::vector<float> rope_cos_table_;  // [max_seq_len * head_dim/2]
    std::vector<float> rope_sin_table_;  // [max_seq_len * head_dim/2]

    mutable std::unique_ptr<ThreadPool> pool_;
    // Helper: float16 to float32 conversion
    static inline float f16_to_f32(uint16_t f16) {
        uint32_t f32;
        uint32_t sign = (f16 >> 15) & 0x1;
        uint32_t exp = (f16 >> 10) & 0x1F;
        uint32_t mant = f16 & 0x3FF;
        if (exp == 0) {
            if (mant == 0) {
                f32 = sign << 31;
            } else {
                exp = 1;
                while (!(mant & 0x400)) {
                    mant <<= 1;
                    exp--;
                }
                mant &= 0x3FF;
                f32 = (sign << 31) | ((exp - 1 + 127) << 23) | (mant << 13);
            }
        } else if (exp == 31) {
            f32 = (sign << 31) | 0x7F800000 | (mant << 13);
        } else {
            f32 = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
        }
        float result;
        std::memcpy(&result, &f32, sizeof(float));
        return result;
    }

    // Weight access — returns float ptr into mmap'd data, dequantized on demand
    const float* get_weight_f32(const std::string& name,
                                 std::vector<float>& buf) const;

    // Primitive ops
    void rms_norm(float* out, const float* x, const float* w, int n) const;
    void rope(float* q, float* k, int pos, int n_heads, int n_kv_heads,
              int head_dim, float freq_base) const;
    void softmax(float* x, int n) const;
    void matmul(float* out, const float* x, const float* w,
                int n, int d) const;
    float max_abs(const float* x, int n) const;

    // Single-pass matmul: dequantizes full weight, one sgemv
    void weight_matmul(const std::string& name,
                       const float* x, int n_cols,
                       float* out, int n_rows) const;

    // Same as weight_matmul, but takes an already-quantized activation
    // vector (block_q8_K*, passed as void* to avoid pulling ns_quant_fused.h
    // into this header) instead of quantizing x internally. Used when the
    // same activation vector feeds multiple weight matmuls in a row (Q/K/V
    // share one, gate/up share another) so quantize_row_q8k runs once
    // instead of once per matmul. x is still needed for the F32/Q8_0
    // fallback branches, which don't use x_q8.
    void weight_matmul_q8(const std::string& name,
                          const float* x, int n_cols,
                          const void* x_q8, int n_blocks_per_row,
                          float* out, int n_rows) const;

    // Batched matmul used by forward_batch(): dequantizes each weight row
    // ONCE and reuses it against every vector in x_batch, instead of paying
    // the dequant cost once per batch element. x_batch is [batch_size][n_cols]
    // row-major, out_batch is [batch_size][n_rows] row-major.
    void weight_matmul_batch(const std::string& name,
                              const float* x_batch, int n_cols,
                              float* out_batch, int n_rows,
                              int batch_size) const;

    // Get a single embedding row (dequantizes only needed blocks)
    void get_embedding_row(int32_t token_id, std::vector<float>& out) const;

    // Compute argmax token only (no logits vector fill). Stores result in state.sampled_token.
    void compute_logits(const float* hidden, InferenceState& state) const;

    // Compute full logits vector (for beam search, hive_test, debugging).
    void compute_logits_full(const float* hidden, InferenceState& state) const;

    // Compute logits for N agents in one parallel row pass.
    void compute_logits_batch(const float* all_hidden,  // [n_agents * n_embd]
                              int n_agents,
                              std::vector<std::vector<float>>& all_logits) const;

    // Build flat op schedule once at load time.
    void compile_schedule();

    // Op dispatchers for the flat schedule.
    static void op_embed(void* ctx);
    static void op_rms_norm(void* ctx);
    static void op_quantize_q8(void* ctx);
    static void op_matmul_q8(void* ctx);
    static void op_add_bias(void* ctx);
    static void op_rope(void* ctx);
    static void op_kv_write(void* ctx);
    static void op_attention(void* ctx);
    static void op_residual(void* ctx);
    static void op_swiglu(void* ctx);

    // Pre-resolved implementations used by the op dispatchers.
    void weight_matmul_q8_tensor(const uint8_t* raw, int weight_type,
                                  const float* x, int n_cols,
                                  const void* x_q8, int n_blocks_per_row,
                                  float* out, int n_rows,
                                  int n_blocks_q8,
                                  const void* repacked = nullptr,
                                  int repacked_R = 0) const;
    void attention_impl(int layer, const float* q, float* out, int pos);
    void get_embedding_row_impl(const uint8_t* raw, int weight_type,
                                int32_t token_id, float* out, int n) const;

    // Grow KV cache capacity dynamically
    void grow_kv_cache(int new_capacity);
};

#endif // TRANSFORMER_H

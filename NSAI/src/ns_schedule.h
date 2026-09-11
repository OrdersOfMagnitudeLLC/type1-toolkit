#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

// Forward declarations to keep this header lightweight.
struct GGUFTensor;
struct InferenceState;
class Transformer;

// Per-token state shared across every op in the schedule.
// Filled by Transformer::forward() before schedule_.run() is called.
struct RunCtx {
    Transformer* self = nullptr;
    int pos = 0;
    bool compute_logits = true;
    float* hidden = nullptr;   // points to state.hidden.data()
    float* logits = nullptr;   // points to state.logits.data()
};

// Base for all op-specific argument bundles.  Each bundle is heap-allocated
// once in compile_schedule() and owned by Schedule::storage.
struct ArgsBase {
    RunCtx* run = nullptr;
    virtual ~ArgsBase() = default;
};

// One dispatch entry: function pointer + generic context pointer.
struct Op {
    void (*fn)(void* ctx);
    void* ctx = nullptr;
};

// Pre-compiled flat op sequence.  Built once in Transformer::load() and
// executed as a single tight loop in Transformer::forward().
struct Schedule {
    std::vector<Op> ops;
    std::vector<std::unique_ptr<ArgsBase>> storage;

    void run() const {
        for (const auto& op : ops) op.fn(op.ctx);
    }
};

// ---------------------------------------------------------------------------
// Concrete args structs, one per op type.  All pointers are persistent
// (member buffers or mmap'd data); the only values that change per token are
// the ones carried in RunCtx and a few fields set by forward() (token_id).
// ---------------------------------------------------------------------------

struct EmbedArgs : ArgsBase {
    const uint8_t* raw = nullptr;
    int weight_type = 0;
    int32_t token_id = 0;
    float* out = nullptr;     // nullptr -> resolved to RunCtx::hidden
    int n = 0;
};

struct NormArgs : ArgsBase {
    const float* w = nullptr;     // pre-dequantized at load time
    std::vector<float> weight_buf;
    float* out = nullptr;         // nullptr -> resolved to RunCtx::hidden
    int n = 0;
};

struct QuantizeArgs : ArgsBase {
    float* x = nullptr;           // nullptr -> resolved to RunCtx::hidden
    void* x_q8 = nullptr;
    int n_blocks = 0;
};

struct MatmulArgs : ArgsBase {
    const uint8_t* raw = nullptr; // weight tensor data pointer
    int weight_type = 0;          // GGMLType cast to int
    float* x = nullptr;           // nullptr -> resolved to RunCtx::hidden
    void* x_q8 = nullptr;
    float* out = nullptr;         // nullptr -> resolved to RunCtx::logits
    int n_cols = 0;
    int n_rows = 0;
    int n_blocks_per_row = 0;
    int n_blocks_q8 = 0;
    bool is_logits = false;       // allows forward() to skip final matmul

    // Repacked Q4_K/Q6_K panel data (8-row interleaved).  Non-null once the
    // lazy per-tensor repack has been completed for this weight.
    const void* repacked = nullptr;
    int repacked_R = 0;

    // Function pointer to specialized matmul implementation, set once at load time
    using MatmulFn = void(*)(const MatmulArgs&, const float*, float*, int);
    MatmulFn matmul_fn = nullptr;
};

struct BiasArgs : ArgsBase {
    const float* b = nullptr;
    float* out = nullptr;
    int n = 0;
};

struct RopeArgs : ArgsBase {
    float* q = nullptr;
    float* k = nullptr;
    int n_heads = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
    float freq_base = 0.0f;
};

struct KVWriteArgs : ArgsBase {
    int layer = 0;
    const float* k = nullptr;
    const float* v = nullptr;
    int n = 0;
};

struct AttentionArgs : ArgsBase {
    int layer = 0;
    const float* q = nullptr;
    float* out = nullptr;
    int n = 0;
};

struct ResidualArgs : ArgsBase {
    const float* b = nullptr;     // a is always RunCtx::hidden
    int n = 0;
};

struct SwiGLUArgs : ArgsBase {
    float* gate = nullptr;
    const float* up = nullptr;
    float* mid = nullptr;
    int n = 0;
};

// NSMace: OOM Proprietary, All Rights Reserved, Orders of Magnitude LLC
// Node embedding: element one-hot -> 128 scalar features via e3nn Linear
// Radial MLP: e3nn FullyConnectedNet [10 -> 64 -> 64 -> 64 -> 512]
#pragma once
#include "WeightLoader.h"
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include "hwy/highway.h"

namespace NSMace {

// e3nn normalize2mom(silu) constant: scales raw SiLU output so that
// E[act(z)^2] = 1 for z ~ N(0,1). Verified numerically against e3nn.
constexpr Real SILU_NORM2MOM = Real(1.6791767923989418);

// Raw SiLU: x * sigmoid(x)
inline Real silu_raw(Real x) {
    return x / (Real(1.0) + std::exp(-x));
}

// e3nn-normalized SiLU, as used inside FullyConnectedNet / Activation
inline Real silu(Real x) {
    return silu_raw(x) * SILU_NORM2MOM;
}

// General vectorized GEMV: y = pw * (W * x), W row-major with leading dimension ld.
template<int MAX_OUT>
inline void vec_gemv_pw(const Real* W, int in_dim, int out_dim, int ld,
                        const Real* x, Real* out, Real pw) {
    namespace hn = hwy::HWY_NAMESPACE;
    const hn::ScalableTag<Real> d;
    const size_t N = hn::Lanes(d);
    const int MAX_CHUNKS = (MAX_OUT + (int)N - 1) / (int)N;
    using Vec = decltype(hn::Zero(d));
    Vec acc[MAX_CHUNKS];
    for (int k = 0; k < MAX_CHUNKS; k++) acc[k] = hn::Zero(d);

    int n_chunks = out_dim / (int)N;
    for (int i = 0; i < in_dim; i++) {
        auto xv = hn::Set(d, x[i] * pw);
        for (int k = 0; k < n_chunks; k++) {
            auto w = hn::LoadU(d, W + (size_t)i * ld + (size_t)k * N);
            acc[k] = hn::MulAdd(xv, w, acc[k]);
        }
    }

    for (int k = 0; k < n_chunks; k++) {
        hn::StoreU(acc[k], d, out + (size_t)k * N);
    }

    int tail = out_dim - n_chunks * (int)N;
    for (int o = out_dim - tail; o < out_dim; o++) {
        Real s = Real(0.0);
        for (int i = 0; i < in_dim; i++)
            s += x[i] * W[(size_t)i * ld + o];
        out[o] = s * pw;
    }
}

struct NodeEmbedding {
    // weight shape: [89, 128] row-major (e3nn o3.Linear path_shape=(mul_in, mul_out))
    // out[o] = path_weight * sum_i x[i] * W[i,o]; with x one-hot on atom_type,
    // this reduces to out[o] = path_weight * W[atom_type, o]
    std::vector<Real> W;  // [89 * 128]
    int num_elements = 89;
    int hidden_dim   = 128;
    Real path_weight = 0.0;  // 1/sqrt(fan_in)

    void load(const WeightLoader& wl) {
        const auto& t = wl.get("node_embedding.linear.weight");
        W = t.data;  // flat [89*128], row-major (mul_in, mul_out)
        path_weight = 1.0 / std::sqrt((Real)num_elements);
    }

    // atom_type: index 0..88 (atomic number mapped to model index)
    // out: pointer to 128 Reals
    void embed(int atom_type, Real* out) const {
        const Real* row = W.data() + (size_t)atom_type * hidden_dim;
        namespace hn = hwy::HWY_NAMESPACE;
        const hn::ScalableTag<Real> d;
        const size_t N = hn::Lanes(d);
        auto pw = hn::Set(d, path_weight);
        int i = 0;
        for (; i + (int)N <= hidden_dim; i += N) {
            auto v = hn::LoadU(d, row + i);
            hn::StoreU(hn::Mul(v, pw), d, out + i);
        }
        for (; i < hidden_dim; i++) out[i] = row[i] * path_weight;
    }
};

struct RadialMLP {
    // Architecture: [10 -> 64 -> 64 -> 64 -> 512]
    // e3nn FullyConnectedNet: weight shape [h_in, h_out] (row-major), no bias.
    // forward per layer: y = x @ (W / sqrt(h_in))
    // SiLU (normalize2mom) applied on layers 0,1,2; layer 3 (last) has no activation.
    static const int N_LAYERS = 4;

    struct Layer {
        std::vector<Real> W;  // [in_dim * out_dim], row-major (in, out)
        int in_dim = 0, out_dim = 0;
        Real inv_sqrt_in = 1.0;

        void forward(const Real* x, Real* y) const {
            vec_gemv_pw<512>(W.data(), in_dim, out_dim, out_dim, x, y, inv_sqrt_in);
        }
    };

    std::vector<Layer> layers;

    void load(const WeightLoader& wl, int interaction_idx) {
        const int arch[] = {10, 64, 64, 64, 512};
        layers.resize(4);
        for (int k = 0; k < 4; k++) {
            std::string wname = "interactions." + std::to_string(interaction_idx)
                              + ".conv_tp_weights.layer" + std::to_string(k) + ".weight";
            layers[k].in_dim  = arch[k];
            layers[k].out_dim = arch[k + 1];
            layers[k].W = wl.get(wname).data;
            layers[k].inv_sqrt_in = 1.0 / std::sqrt((Real)arch[k]);
        }
    }

    // x: [10] bessel features -> out: [512] radial weights
    void forward(const Real* x, Real* out, Real threshold = 0.0, bool* pruned_stats = nullptr) const {
        // Ping-pong buffers: max layer size is 512, now on the stack
        Real buf0[512], buf1[512];
        std::memcpy(buf0, x, 10 * sizeof(Real));
        Real *src = buf0, *dst = buf1;
        for (int k = 0; k < (int)layers.size(); k++) {
            layers[k].forward(src, dst);
            // Apply normalized SiLU on all but last layer
            if (k < (int)layers.size() - 1) {
                for (int i = 0; i < layers[k].out_dim; i++) dst[i] = silu(dst[i]);
                
                // TASK 2: NSInfer-style pruning - compute L2 norm and zero if below threshold
                if (threshold > 0.0) {
                    Real norm = 0.0;
                    for (int i = 0; i < layers[k].out_dim; i++) {
                        norm += dst[i] * dst[i];
                    }
                    norm = std::sqrt(norm);
                    
                    if (norm < threshold) {
                        // Zero the vector
                        for (int i = 0; i < layers[k].out_dim; i++) {
                            dst[i] = 0.0;
                        }
                        if (pruned_stats) pruned_stats[k] = true;
                    }
                }
            }
            std::swap(src, dst);
        }
        std::memcpy(out, src, 512 * sizeof(Real));
    }
};

} // namespace NSMace

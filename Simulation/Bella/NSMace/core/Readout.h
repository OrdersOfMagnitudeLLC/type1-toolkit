// NSMace: OOM Proprietary, All Rights Reserved, Orders of Magnitude LLC
// Two readout heads: linear (interaction 0) + nonlinear (interaction 1)
// Both use e3nn o3.Linear (no bias, path_weight = 1/sqrt(fan_in)).
#pragma once
#include "WeightLoader.h"
#include "NodeEmbedding.h"  // for silu()
#include <cmath>
#include <string>
#include <vector>

namespace NSMace {

struct LinearReadout {
    // readouts.0.linear.weight: flat [128], path_shape=(128,1)
    std::vector<Real> W;
    Real path_weight = 0.0;  // 1/sqrt(128)

    void load(const WeightLoader& wl, int idx) {
        W = wl.get("readouts." + std::to_string(idx) + ".linear.weight").data;
        path_weight = 1.0 / std::sqrt((Real)W.size());
    }

    Real forward(const Real* h) const {
        namespace hn = hwy::HWY_NAMESPACE;
        const hn::ScalableTag<Real> d;
        const size_t N = hn::Lanes(d);
        auto vsum = hn::Zero(d);
        int i = 0;
        for (; i + (int)N <= (int)W.size(); i += N) {
            auto a = hn::LoadU(d, h + i);
            auto b = hn::LoadU(d, W.data() + i);
            vsum = hn::MulAdd(a, b, vsum);
        }
        Real s = hn::ReduceSum(d, vsum);
        for (; i < (int)W.size(); i++) s += W[i] * h[i];
        return s * path_weight;
    }
};

struct NonLinearReadout {
    // readouts.1: linear_1 128->16 (flat [2048], path_shape=(128,16)),
    // SiLU (normalize2mom), linear_2 16->1 (flat [16], path_shape=(16,1))
    std::vector<Real> W1;  // [128*16] row-major (in, out)
    std::vector<Real> W2;  // [16*1]
    Real pw1 = 0.0;  // 1/sqrt(128)
    Real pw2 = 0.0;  // 1/sqrt(16)

    void load(const WeightLoader& wl, int idx) {
        std::string base = "readouts." + std::to_string(idx);
        W1 = wl.get(base + ".linear_1.weight").data;
        W2 = wl.get(base + ".linear_2.weight").data;
        pw1 = 1.0 / std::sqrt(128.0);
        pw2 = 1.0 / std::sqrt(16.0);
    }

    Real forward(const Real* h) const {
        namespace hn = hwy::HWY_NAMESPACE;
        const hn::ScalableTag<Real> d;
        const size_t N = hn::Lanes(d);
        Real mid[16];
        vec_gemv_pw<16>(W1.data(), 128, 16, 16, h, mid, pw1);
        for (int o = 0; o < 16; o++) mid[o] = silu(mid[o]);
        auto vsum = hn::Zero(d);
        int i = 0;
        for (; i + (int)N <= 16; i += N) {
            auto a = hn::LoadU(d, mid + i);
            auto b = hn::LoadU(d, W2.data() + i);
            vsum = hn::MulAdd(a, b, vsum);
        }
        Real s = hn::ReduceSum(d, vsum);
        for (; i < 16; i++) s += mid[i] * W2[i];
        return s * pw2;
    }
};

} // namespace NSMace

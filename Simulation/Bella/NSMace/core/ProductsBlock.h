// NSMace — OOM Proprietary, All Rights Reserved, Orders of Magnitude LLC
// Session 4: symmetric-contraction "products" block (EquivariantProductBasisBlock).
// Implements MACE paper Eq. 10-11 for correlation=3, scalar (0e) output only,
// using precomputed Clebsch-Gordan-derived U-tensors (weights/cg_tensors.json)
// and the trained per-element contraction weights (weights_max, weights.0/1).
// Formula verified numerically against live e3nn/MACE (reference/numpy_prototype.py).
#pragma once
#include "WeightLoader.h"
#include "NodeEmbedding.h"  // for vec_gemv_pw
#include "hwy/highway.h"
#include <string>
#include <vector>
#include <cmath>
#include <cstring>
#include <omp.h>

namespace NSMace {

constexpr int ELL_DIM = 16;      // flattened l=0..3 coupling dimension (1+3+5+7)
constexpr int NU3_PARAMS = 23;
constexpr int NU2_PARAMS = 4;
constexpr int NU1_PARAMS = 1;

struct ProductsBlock {
    // Shared CG tensors (identical across all interactions/elements)
    static inline std::vector<Real> U1;  // [16,1]        -- ell(w) x params
    static inline std::vector<Real> U2;  // [16,16,4]     -- ell(w,x) x params
    static inline std::vector<Real> U3;  // [16,16,16,23] -- ell(w,x,i) x params
    static inline bool cg_loaded = false;

    // Per-element trained contraction weights
    std::vector<Real> weights_max;  // [89,23,128]
    std::vector<Real> weights_nu2;  // [89,4,128]
    std::vector<Real> weights_nu1;  // [89,1,128]
    // products.N.linear: 128x0e -> 128x0e, path_weight 1/sqrt(128)
    std::vector<Real> linear_W;

    static void load_cg(const WeightLoader& wl) {
        if (cg_loaded) return;
        U1 = wl.get("cg.U1").data;
        U2 = wl.get("cg.U2").data;
        U3 = wl.get("cg.U3").data;
        cg_loaded = true;
    }

    void load(const WeightLoader& wl, int idx) {
        load_cg(wl);
        std::string base = "products." + std::to_string(idx) + ".symmetric_contractions.contractions.0";
        weights_max = wl.get(base + ".weights_max").data;
        weights_nu2 = wl.get(base + ".weights.0").data;
        weights_nu1 = wl.get(base + ".weights.1").data;
        linear_W = wl.get("products." + std::to_string(idx) + ".linear.weight").data;
    }

    // feat: [128][16] (channel-major, c*16+i), elem_idx: element index -> out: [128]
    void symmetric_contraction(const Real* feat, int elem_idx, Real* out) const {
        // Per-element weight slices
        const Real* Wmax_e = weights_max.data() + (size_t)elem_idx * NU3_PARAMS * 128;  // [23,128]
        const Real* Wnu2_e = weights_nu2.data() + (size_t)elem_idx * NU2_PARAMS * 128;  // [4,128]
        const Real* Wnu1_e = weights_nu1.data() + (size_t)elem_idx * NU1_PARAMS * 128;  // [1,128]

        for (int c = 0; c < 128; c++) {
            const Real* feat_c = feat + c * ELL_DIM;  // [16]

            // Precompute per-channel weight slices — eliminates k*128 stride
            // (was: 94K scattered 1KB-stride accesses per atom; now: 28 contiguous reads)
            Real wc_max[NU3_PARAMS], wc_nu2[NU2_PARAMS], wc_nu1[NU1_PARAMS];
            for (int k = 0; k < NU3_PARAMS; k++) wc_max[k] = Wmax_e[k * 128 + c];
            for (int k = 0; k < NU2_PARAMS; k++) wc_nu2[k] = Wnu2_e[k * 128 + c];
            for (int k = 0; k < NU1_PARAMS; k++) wc_nu1[k] = Wnu1_e[k * 128 + c];

            // out3[w,x] = sum_{i,k} U3[w,x,i,k] * Wmax_e[k,c] * feat_c[i]
            Real out3[ELL_DIM][ELL_DIM];
            for (int w = 0; w < ELL_DIM; w++) {
                for (int x = 0; x < ELL_DIM; x++) {
                    Real s = 0.0;
                    for (int i = 0; i < ELL_DIM; i++) {
                        Real fi = feat_c[i];
                        if (fi == 0.0) continue;
                        const Real* U3_wxi = U3.data() + (((size_t)w * ELL_DIM + x) * ELL_DIM + i) * NU3_PARAMS;
                        namespace hn = hwy::HWY_NAMESPACE;
                        const hn::ScalableTag<Real> d;
                        auto vsum = hn::Zero(d);
                        size_t N = hn::Lanes(d);
                        size_t k = 0;
                        for (; k + N <= NU3_PARAMS; k += N)
                            vsum = hn::MulAdd(hn::LoadU(d, U3_wxi+k), hn::LoadU(d, wc_max+k), vsum);
                        Real acc = hn::ReduceSum(d, vsum);
                        for (; k < NU3_PARAMS; k++) acc += U3_wxi[k] * wc_max[k];
                        s += acc * fi;
                    }
                    out3[w][x] = s;
                }
            }

            // c_tensor2[w,x] = sum_k U2[w,x,k]*Wnu2_e[k,c] + out3[w,x]
            // out2[w] = sum_i c_tensor2[w,i] * feat_c[i]
            Real out2[ELL_DIM];
            for (int w = 0; w < ELL_DIM; w++) {
                Real s = 0.0;
                for (int x = 0; x < ELL_DIM; x++) {
                    const Real* U2_wx = U2.data() + ((size_t)w * ELL_DIM + x) * NU2_PARAMS;
                    Real ctensor = 0.0;
                    for (int k = 0; k < NU2_PARAMS; k++) ctensor += U2_wx[k] * wc_nu2[k];
                    ctensor += out3[w][x];
                    s += ctensor * feat_c[x];
                }
                out2[w] = s;
            }

            // out1 = sum_w (U1[w,0]*wc_nu1[0] + out2[w]) * feat_c[w]
            Real s = 0.0;
            for (int w = 0; w < ELL_DIM; w++) {
                Real ctensor = U1.data()[w * NU1_PARAMS] * wc_nu1[0];
                ctensor += out2[w];
                s += ctensor * feat_c[w];
            }
            out[c] = s;
        }
    }

    // x: [128] -> out: [128], e3nn Linear 128x0e->128x0e, path_weight 1/sqrt(128)
    void linear(const Real* x, Real* out) const {
        const Real pw = Real(1.0) / std::sqrt(128.0);
        vec_gemv_pw<128>(linear_W.data(), 128, 128, 128, x, out, pw);
    }
};

} // namespace NSMace

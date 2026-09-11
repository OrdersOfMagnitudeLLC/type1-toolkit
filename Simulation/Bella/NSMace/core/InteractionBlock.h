// NSMace — OOM Proprietary, All Rights Reserved, Orders of Magnitude LLC
// Session 4: correct message-passing interaction block.
// conv_tp: 128x0e x (0e+1o+2e+3o) -> 128x0e+128x1o+128x2e+128x3o (uvu, per-l scalar mul)
// linear:  per-l 128x128 mix (irreps_mid -> irreps_mid), path_weight=1/sqrt(128)
// skip_tp: 128x0e x 89x0e -> 128x0e (uvw fully-connected), path_weight=1/sqrt(128*89)
// All formulas verified numerically against live e3nn/MACE modules
// (see reference/numpy_prototype.py) before porting.
#pragma once
#include "WeightLoader.h"
#include "NodeEmbedding.h"  // for silu()/e3nn linear conventions
#include "RadialBasis.h"
#include <algorithm>
#include <string>
#include <vector>
#include <cmath>
#include <cstring>
#include "hwy/highway.h"

namespace NSMace {

// Compile-time constants for MACE-MP-0 fixed architecture
#define HIDDEN 128
#define LAYERS 2
#define MAX_ELL 3
#define NUM_ELEMENTS 89
#define MID_TOTAL 2048

// l-block layout inside the 2048-d irreps_mid = 128x0e+128x1o+128x2e+128x3o
constexpr int L_DIM[4]    = {1, 3, 5, 7};
constexpr int L_OFFSET[4] = {0, 128, 512, 1152};   // offset into 2048-d flat vector
constexpr int L_SHOFF[4]  = {0, 1, 4, 9};          // offset into 16-d spherical harmonics

struct InteractionBlock {
    // linear_up: 128x0e -> 128x0e, flat [128*128], path_weight 1/sqrt(128)
    std::vector<Real> linear_up_W;
    // linear: irreps_mid -> irreps_mid, flat [4*128*128], per-l block, path_weight 1/sqrt(128)
    std::vector<Real> linear_W;
    // skip_tp: 128x0e x 89x0e -> 128x0e, flat [128*89*128], path_weight 1/sqrt(128*89)
    std::vector<Real> skip_tp_W;

    void load(const WeightLoader& wl, int idx) {
        std::string base = "interactions." + std::to_string(idx);
        linear_up_W = wl.get(base + ".linear_up.weight").data;
        linear_W    = wl.get(base + ".linear.weight").data;
        skip_tp_W   = wl.get(base + ".skip_tp.weight").data;
    }

    // node_feats: [128] -> out: [128]  (e3nn Linear 128x0e->128x0e)
    void linear_up(const Real* x, Real* out) const {
        const Real pw = Real(1.0) / std::sqrt((Real)HIDDEN);
        vec_gemv_pw<128>(linear_up_W.data(), HIDDEN, HIDDEN, HIDDEN, x, out, pw);
    }

    // Compute one edge's message contribution mji [2048] and accumulate into acc[2048].
    // node_feats_up_j: [128] (already passed through linear_up for the source atom)
    // tp_weights: [512] (output of the radial MLP for this edge)
    // sh: [16] (spherical harmonics of the edge direction, l=0..3 concatenated)
    void conv_tp_accumulate(const Real* node_feats_up_j, const Real* tp_weights,
                             const Real* sh, Real* acc) const {
        for (int l = 0; l < 4; l++) {
            int dim = L_DIM[l];
            int off = L_OFFSET[l];
            const Real* w_block = tp_weights + l * HIDDEN;   // [128]
            const Real* sh_block = sh + L_SHOFF[l];          // [dim]
            for (int c = 0; c < HIDDEN; c++) {
                Real wf = w_block[c] * node_feats_up_j[c];
                Real* dst = acc + off + c * dim;
                for (int m = 0; m < dim; m++) dst[m] += wf * sh_block[m];
            }
        }
    }

    // Fused per-edge: radial basis + MLP + spherical harmonics + conv_tp in one call.
    void conv_tp_full(const Neighbor& nb, const Real* node_feats_up_j,
                      const RadialMLP& rmlp, Real* acc) const {
        Real bessel[10];
        bessel_basis((Real)nb.r, Real(6.0), 10, bessel);

        Real tp_weights[512];
        rmlp.forward(bessel, tp_weights);

        Real sh[16];
        spherical_harmonics(nb.dx, nb.dy, nb.dz, nb.r, 3, sh);

        conv_tp_accumulate(node_feats_up_j, tp_weights, sh, acc);
    }

    // Version with pruning: returns message norm, caller can skip if below threshold
    Real conv_tp_full_with_norm(const Neighbor& nb, const Real* node_feats_up_j,
                                const RadialMLP& rmlp, Real* acc) const {
        Real bessel[10];
        bessel_basis((Real)nb.r, Real(6.0), 10, bessel);

        Real tp_weights[512];
        rmlp.forward(bessel, tp_weights);

        // Compute message norm from tp_weights
        Real norm = Real(0.0);
        for (int i = 0; i < 512; i++) {
            norm += tp_weights[i] * tp_weights[i];
        }
        norm = std::sqrt(norm);

        if (norm > 0.0) {
            Real sh[16];
            spherical_harmonics(nb.dx, nb.dy, nb.dz, nb.r, 3, sh);
            conv_tp_accumulate(node_feats_up_j, tp_weights, sh, acc);
        }

        return norm;
    }

    // message: [N, 2048] -> out: [N, 2048], per-l 128x128 linear, path_weight 1/sqrt(128)
    // tmp must be at least N * HIDDEN floats.
    void linear(const Real* message, Real* out, int N, Real* tmp) const {
        namespace hn = hwy::HWY_NAMESPACE;
        const hn::ScalableTag<Real> d;
        const size_t NL = hn::Lanes(d);
        const Real pw = Real(1.0) / std::sqrt((Real)HIDDEN);
        constexpr int BS = 8;
        using Vec = decltype(hn::Zero(d));

        for (int l = 0; l < 4; l++) {
            int dim = L_DIM[l];
            int off = L_OFFSET[l];
            const Real* Wl = linear_W.data() + (size_t)l * HIDDEN * HIDDEN;

            for (int m = 0; m < dim; m++) {
                for (size_t co = 0; co + NL <= (size_t)HIDDEN; co += NL) {
                    for (int i_base = 0; i_base < N; i_base += BS) {
                        int i_end = std::min(i_base + BS, N);
                        int active = i_end - i_base;

                        Vec acc[BS];
                        for (int a = 0; a < active; a++) acc[a] = hn::Zero(d);

                        for (int ci = 0; ci < HIDDEN; ci++) {
                            auto w = hn::LoadU(d, Wl + (size_t)ci * HIDDEN + co);
                            for (int a = 0; a < active; a++) {
                                int i = i_base + a;
                                Real x_val = message[(size_t)i * MID_TOTAL + off + (size_t)ci * dim + m];
                                auto xv = hn::Set(d, x_val);
                                acc[a] = hn::MulAdd(xv, w, acc[a]);
                            }
                        }

                        auto vpw = hn::Set(d, pw);
                        for (int a = 0; a < active; a++) {
                            int i = i_base + a;
                            auto v = hn::Mul(acc[a], vpw);
                            hn::StoreU(v, d, tmp + (size_t)i * HIDDEN + co);
                        }
                    }
                }

                for (int i = 0; i < N; i++) {
                    const Real* t = tmp + (size_t)i * HIDDEN;
                    Real* out_i = out + (size_t)i * MID_TOTAL + off + m;
                    for (int co = 0; co < HIDDEN; co++)
                        out_i[(size_t)co * dim] = t[co];
                }
            }
        }
    }

    // node_feats: [128] (pre-update), elem_idx: element index (0..88) -> out: [128]
    void skip_tp(const Real* node_feats, int elem_idx, Real* out) const {
        const Real pw = Real(1.0) / std::sqrt((Real)(HIDDEN * NUM_ELEMENTS));
        vec_gemv_pw<128>(skip_tp_W.data() + (size_t)elem_idx * HIDDEN,
                         HIDDEN, HIDDEN, NUM_ELEMENTS * HIDDEN,
                         node_feats, out, pw);
    }
};

} // namespace NSMace

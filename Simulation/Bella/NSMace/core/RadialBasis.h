// NSMace: OOM Proprietary, All Rights Reserved, Orders of Magnitude LLC
// Bessel radial basis + polynomial envelope: matches MACE-MP-0 radial embedding
#pragma once
#include "NSMace.h"
#include <cmath>
#include <cstring>

namespace NSMace {

// Polynomial envelope: p(x) = 1 - ((p+1)(p+2)/2)*x^p + p(p+2)*x^(p+1) - (p*(p+1)/2)*x^(p+2)
// where x = r / r_max, p = 5 (MACE default)
// Ensures smooth cutoff to 0 at r_max
inline Real poly_envelope(Real r, Real r_max, int p=5) {
    if(r >= r_max) return 0.0;
    Real x = r / r_max;
    Real pp1 = p+1, pp2 = p+2;
    return 1.0 
        - (pp1*pp2/2.0) * std::pow(x, p)
        + p * pp2        * std::pow(x, p+1)
        - (p*pp1/2.0)   * std::pow(x, p+2);
}

// Bessel basis: e_n(r) = sqrt(2/r_max) * sin(n*pi*r/r_max) / r * envelope(r)
// n = 1..num_bessel
inline void bessel_basis(Real r, Real r_max, int num_bessel, Real* out) {
    std::memset(out, 0, num_bessel * sizeof(Real));
    if(r < 1e-8 || r >= r_max) return;
    Real env = poly_envelope(r, r_max);
    Real norm = std::sqrt(Real(2.0) / r_max);
    const Real pi = Real(M_PI);
    for(int n=1; n<=num_bessel; n++) {
        out[n-1] = norm * std::sin(Real(n) * pi * r / r_max) / r * env;
    }
}

// TASK 3: Fused bessel + spherical harmonics - single memory read, both outputs
// Computes both bessel basis and spherical harmonics in one pass
// Pure memory bandwidth optimization - no math changes
inline void compute_edge_features(Real dx, Real dy, Real dz, Real r,
                                  Real r_max, int num_bessel, Real* bessel_out,
                                  int max_ell, Real* sh_out) {
    // Compute bessel basis
    std::memset(bessel_out, 0, num_bessel * sizeof(Real));
    if(r >= 1e-8 && r < r_max) {
        Real env = poly_envelope(r, r_max);
        Real norm = std::sqrt(Real(2.0) / r_max);
        const Real pi = Real(M_PI);
        for(int n=1; n<=num_bessel; n++) {
            bessel_out[n-1] = norm * std::sin(Real(n) * pi * r / r_max) / r * env;
        }
    }
    
    // Compute spherical harmonics
    Real ir = (r > 1e-10) ? Real(1.0)/r : Real(0.0);
    Real x=dx*ir, y=dy*ir, z=dz*ir;
    int idx = 0;

    Real sh_0_0 = Real(1.0);
    sh_out[idx++] = sh_0_0;
    if(max_ell < 1) return;

    Real sh_1_0 = x, sh_1_1 = y, sh_1_2 = z;
    const Real norm1 = std::sqrt(Real(3.0));
    sh_out[idx++] = sh_1_0 * norm1;
    sh_out[idx++] = sh_1_1 * norm1;
    sh_out[idx++] = sh_1_2 * norm1;
    if(max_ell < 2) return;

    Real sh_2_0 = std::sqrt(Real(3.0)) * x * z;
    Real sh_2_1 = std::sqrt(Real(3.0)) * x * y;
    Real y2 = y*y;
    Real x2z2 = x*x + z*z;
    Real sh_2_2 = y2 - Real(0.5)*x2z2;
    Real sh_2_3 = std::sqrt(Real(3.0)) * y * z;
    Real sh_2_4 = std::sqrt(Real(3.0)/Real(4.0)) * (x*x - z*z);
    const Real norm2 = std::sqrt(Real(5.0)) / Real(2.0);
    sh_out[idx++] = sh_2_0 * norm2;
    sh_out[idx++] = sh_2_1 * norm2;
    sh_out[idx++] = sh_2_2 * norm2;
    sh_out[idx++] = sh_2_3 * norm2;
    sh_out[idx++] = sh_2_4 * norm2;
    if(max_ell < 3) return;

    Real sh_3_0 = std::sqrt(Real(2.5)) * x * (x*x - Real(3.0)*(y*y+z*z));
    Real sh_3_1 = std::sqrt(Real(7.5)) * x * (y*y - z*z);
    Real sh_3_2 = std::sqrt(Real(15.0)) * x * y * z;
    Real sh_3_3 = std::sqrt(Real(1.25)) * y * (Real(4.0)*x*x - y*y - z*z);
    Real sh_3_4 = std::sqrt(Real(1.25)) * z * (Real(2.0)*x*x - Real(3.0)*y*y - z*z);
    Real sh_3_5 = std::sqrt(Real(7.5)) * y * (y*y - Real(3.0)*z*z);
    Real sh_3_6 = std::sqrt(Real(2.5)) * z * (Real(2.0)*z*z - Real(3.0)*(x*x+y*y));
    const Real norm3 = std::sqrt(Real(7.0)) / Real(2.0);
    sh_out[idx++] = sh_3_0 * norm3;
    sh_out[idx++] = sh_3_1 * norm3;
    sh_out[idx++] = sh_3_2 * norm3;
    sh_out[idx++] = sh_3_3 * norm3;
    sh_out[idx++] = sh_3_4 * norm3;
    sh_out[idx++] = sh_3_5 * norm3;
    sh_out[idx++] = sh_3_6 * norm3;
}

// Real spherical harmonics, matching e3nn's o3.spherical_harmonics() exactly
// (normalize=True, normalization='component': this is what MACE's
// model.spherical_harmonics module uses internally, verified against the
// live model). NOTE: e3nn's l=1 component order is (x, y, z) directly :
// NOT the (y, z, x) Condon-Shortley convention found in most QM textbooks.
// Higher l blocks are built recursively via Clebsch-Gordan-derived
// polynomials (see e3nn/o3/_spherical_harmonics.py::_spherical_harmonics),
// reproduced here verbatim rather than from generic closed-form formulas.
// Returns flat vector of length (max_ell+1)^2:
//   [Y0_0, Y1_0,Y1_1,Y1_2, Y2_0..Y2_4, Y3_0..Y3_6]
// Input: displacement vector (dx,dy,dz), its length r.
inline void spherical_harmonics(Real dx, Real dy, Real dz,
                                Real r, int max_ell, Real* out) {
    Real ir = (r > 1e-10) ? Real(1.0)/r : Real(0.0);
    Real x=dx*ir, y=dy*ir, z=dz*ir;
    int idx = 0;

    Real sh_0_0 = Real(1.0);
    out[idx++] = sh_0_0;
    if(max_ell < 1) return;

    Real sh_1_0 = x, sh_1_1 = y, sh_1_2 = z;
    const Real norm1 = std::sqrt(Real(3.0));
    out[idx++] = sh_1_0 * norm1;
    out[idx++] = sh_1_1 * norm1;
    out[idx++] = sh_1_2 * norm1;
    if(max_ell < 2) return;

    Real sh_2_0 = std::sqrt(Real(3.0)) * x * z;
    Real sh_2_1 = std::sqrt(Real(3.0)) * x * y;
    Real y2 = y*y;
    Real x2z2 = x*x + z*z;
    Real sh_2_2 = y2 - Real(0.5)*x2z2;
    Real sh_2_3 = std::sqrt(Real(3.0)) * y * z;
    Real sh_2_4 = std::sqrt(Real(3.0))/Real(2.0) * (z*z - x*x);
    const Real norm2 = std::sqrt(Real(5.0));
    out[idx++] = sh_2_0 * norm2;
    out[idx++] = sh_2_1 * norm2;
    out[idx++] = sh_2_2 * norm2;
    out[idx++] = sh_2_3 * norm2;
    out[idx++] = sh_2_4 * norm2;
    if(max_ell < 3) return;

    Real sh_3_0 = std::sqrt(Real(5.0)/Real(6.0)) * (sh_2_0*z + sh_2_4*x);
    Real sh_3_1 = std::sqrt(Real(5.0)) * sh_2_0 * y;
    Real sh_3_2 = std::sqrt(Real(3.0)/Real(8.0)) * (Real(4.0)*y2 - x2z2) * x;
    Real sh_3_3 = Real(0.5) * y * (Real(2.0)*y2 - Real(3.0)*x2z2);
    Real sh_3_4 = std::sqrt(Real(3.0)/Real(8.0)) * z * (Real(4.0)*y2 - x2z2);
    Real sh_3_5 = std::sqrt(Real(5.0)) * sh_2_4 * y;
    Real sh_3_6 = std::sqrt(Real(5.0)/Real(6.0)) * (sh_2_4*z - sh_2_0*x);
    const Real norm3 = std::sqrt(Real(7.0));
    out[idx++] = sh_3_0 * norm3;
    out[idx++] = sh_3_1 * norm3;
    out[idx++] = sh_3_2 * norm3;
    out[idx++] = sh_3_3 * norm3;
    out[idx++] = sh_3_4 * norm3;
    out[idx++] = sh_3_5 * norm3;
    out[idx++] = sh_3_6 * norm3;
}

} // namespace NSMace

#!/usr/bin/env python3
"""Muon g-2 hadronic vacuum polarization from foam inputs.
ALL RESULTS SPECULATION until verified vs HVP literature.
PASS 2: adds omega, phi, KKbar threshold, rho-omega interference.

a_mu^HVP = (alpha^2/3 pi^2) int ds K(s) R(s) / s
  K(s) = int_0^1 dx x^2 (1 - x) / (x^2 + (1-x) s/m_mu^2)   [standard kernel]
  R(s) = sigma(e+e- -> hadrons)/sigma(e+e- -> mu+mu-)

Foam inputs (fixed, not refit):
  sigma_str = 0.289 GeV^2 (T32, DERIVED), m_q = 0.303 GeV (T61, DERIVED),
  alpha_s = 4/pi^2 = 0.405 (T76, DERIVED), m_rho = 0.760 GeV (DERIVED),
  m_pi = 0.135 GeV, f_pi = 0.093 GeV (inputs).
Pass-2 inputs: omega/phi masses, widths, BR_ee (MEASURED);
  m_s = 0.495 GeV constituent fallback (MEASURED, flagged — no T80 m_s);
  delta_rho-omega = (-3.5+0.2i) MeV (MEASURED).
Regimes matched to Gamma(2) cusps:
  R1 IR s < m_rho^2       <-> r_dS -> inf cusp
  R2 mid m_rho^2..4 GeV^2 <-> r_reset/r_dS -> 1 cusp
  R3 UV s > 4 GeV^2       <-> r_Planck -> 0 cusp
"""
import numpy as np
from scipy.integrate import quad

# ---------- constants ----------
ALPHA = 1.0 / 137.035999084
M_MU = 0.1056583755           # GeV
M_RHO = 0.760                 # foam-derived (DERIVED)
M_PI = 0.135                  # input
SIGMA_STR = 0.289             # GeV^2, T32 (unused here, kept for record)
M_Q = 0.303                   # GeV, T61 (DERIVED)
ALPHA_S = 4.0 / np.pi**2      # 0.405, T76 (DERIVED)
F_PI = 0.093                  # GeV, input (pion decay constant)
NC = 3

# rho width from KSRF relation g_rhopipi = m_rho/(sqrt(2) f_pi):
#   Gamma_rho = g^2 m_rho/(48 pi) (1 - 4 m_pi^2/m_rho^2)^{3/2}
g_rhopipi = M_RHO / (np.sqrt(2) * F_PI)
phase = (1 - 4 * M_PI**2 / M_RHO**2) ** 1.5
G_RHO = g_rhopipi**2 * M_RHO / (48 * np.pi) * phase
print(f"g_rhopipi(KSRF) = {g_rhopipi:.3f}  (expt ~6.0)")
print(f"Gamma_rho(KSRF) = {G_RHO*1e3:.1f} MeV (expt ~149)")

# ---------- pass-2 channels ----------
# omega(782), phi(1020): MEASURED inputs
M_OM, G_OM, BR_OM = 0.782, 8.68e-3, 7.38e-5      # GeV
M_PHI, G_PHI, BR_PHI = 1.020, 4.27e-3, 2.97e-4   # GeV
GEE_OM = G_OM * BR_OM                            # 6.41e-7 GeV (~0.64 keV)
GEE_PHI = G_PHI * BR_PHI                         # 1.27e-6 GeV (~1.27 keV)
# KKbar: m_K = m_q + m_s. No foam m_s from T80 (T80 = CKM angles);
# T66 gives m_s_eff = 0.6705 GeV (strange-baryon effective, DERIVED).
# Per spec, fallback m_s = 0.495 GeV constituent (MEASURED, FLAGGED).
M_S = 0.495                                      # MEASURED fallback
M_S_T66 = 0.6705                                 # T66 alternative (DERIVED)
M_K = M_Q + M_S                                  # = 0.798 GeV (foam constituent)
# rho-omega mixing, GS convention (MEASURED)
DELTA_RW = (-3.5e-3 + 0.2e-3j)                   # GeV


# ---------- dispersion kernel ----------
def K(s):
    f = lambda x: x**2 * (1 - x) / (x**2 + (1 - x) * s / M_MU**2)
    return quad(f, 0, 1, epsabs=1e-12)[0]


# ---------- R(s) pieces ----------
def rho_width(s):
    """Energy-dependent p-wave width (Gounaris-Sakurai kinematics)."""
    if s <= 4 * M_PI**2:
        return 0.0
    return G_RHO * (M_RHO**2 / s) * \
        ((s - 4 * M_PI**2) / (M_RHO**2 - 4 * M_PI**2)) ** 1.5


def R_pipi(s, mix=True):
    """VDM pion form factor, running-width rho pole; optional standard
    rho-omega mixing term 1 + delta_rw*s/(m_om*(m_om^2 - s - i m_om G_om)).
    delta_rw MEASURED."""
    if s <= 4 * M_PI**2:
        return 0.0
    beta3 = (1 - 4 * M_PI**2 / s) ** 1.5
    G = rho_width(s)
    Fc = M_RHO**2 / (M_RHO**2 - s - 1j * M_RHO * G)
    if mix:
        Fc *= (1 + DELTA_RW * s /
               (M_OM * (M_OM**2 - s - 1j * M_OM * G_OM)))
    return 0.25 * beta3 * abs(Fc) ** 2


def R_regge(s):
    """Foam Regge tower n = 4, 9 (m_n = sqrt(n) m_rho = 1.52, 2.28 GeV;
    n=1 rho tail is carried by R_pipi).  BW resonances, Gamma_ee via VDM
    alpha^2 m/(3 g^2) * 4pi; Gamma_n = Gamma_rho m/m_rho.  SPECULATION."""
    tot = 0.0
    for n in (4, 9):
        m = np.sqrt(n) * M_RHO
        G = G_RHO * m / M_RHO
        gee = ALPHA**2 * m / (3 * g_rhopipi**2) * 4 * np.pi
        amp = 12 * np.pi**2 * gee / m          # integral weight
        tot += amp * (m * G / np.pi) / ((s - m**2) ** 2 + m**2 * G**2)
    # continuum floor under resonances, foam-quark threshold 2 m_q
    if s > 4 * M_Q**2:
        tot += NC * (4/9 + 1/9 + 1/9) * (1 + ALPHA_S / np.pi) * \
               np.sqrt(1 - 4 * M_Q**2 / s)
    return tot


def R_narrow(s, m, G, gee):
    """Narrow vector resonance: R(s) = 9 Gee G s^2 /
    (alpha^2 m^2 ((s-m^2)^2 + m^2 G^2)); integrates to 9 pi Gee/alpha^2."""
    return 9 * gee * G * s**2 / \
        (ALPHA**2 * m**2 * ((s - m**2) ** 2 + m**2 * G**2))


def R_omega(s):
    return R_narrow(s, M_OM, G_OM, GEE_OM)


def R_phi(s):
    return R_narrow(s, M_PHI, G_PHI, GEE_PHI)


def R_KK(s, m_k=M_K):
    """KKbar threshold: R = (1/4)(beta+^3 + beta0^3) ~ (1/2) beta^3
    pointlike.  m_K = m_q + m_s = 0.798 GeV (foam constituent with
    MEASURED m_s fallback).  Threshold 2 m_K = 1.596 GeV vs physical
    0.988 GeV -- FLAGGED."""
    if s <= 4 * m_k**2:
        return 0.0
    return 0.5 * (1 - 4 * m_k**2 / s) ** 1.5


def R_mid(s):
    """Regime 2: pipi tail + rho-omega interference, Regge tower n=4,9 +
    continuum floor, omega, phi, KKbar."""
    return R_pipi(s) + R_regge(s) + R_omega(s) + R_phi(s) + R_KK(s)


def R_pqcd(s):
    """R = Nc sum Q_f^2 (1 + alpha_s/pi); u,d,s,c active above 2 GeV."""
    q2 = 4/9 + 1/9 + 1/9 + 4/9
    return NC * q2 * (1 + ALPHA_S / np.pi)


def integrand(s):
    if s < 4 * M_PI**2:
        return 0.0
    r = R_pipi(s) if s < M_RHO**2 else \
        (R_mid(s) if s < 4.0 else R_pqcd(s))
    return K(s) * r / s


# ---------- integrate ----------
s1, s2 = 4 * M_PI**2, M_RHO**2          # 0.0729 -> 0.578 GeV^2
s3 = 4.0                                 # (2 GeV)^2
S_TOP = 1e4
pref = ALPHA**2 / (3 * np.pi**2)

I1, _ = quad(integrand, s1, s2, epsabs=1e-13, limit=400)
I2, _ = quad(integrand, s2, s3, epsabs=1e-13, limit=400,
             points=[M_OM**2, M_PHI**2, 4 * M_K**2])
I3, _ = quad(integrand, s3, S_TOP, epsabs=1e-13, limit=400)
a1, a2, a3 = pref * I1, pref * I2, pref * I3
tot = a1 + a2 + a3

print("\n===== RESULTS pass 2 (x1e-11) =====")
print(f"Regime 1 IR  (s < m_rho^2)        : {a1*1e11:9.1f}  "
      f"({100*a1/tot:5.1f}% of total)")
print(f"Regime 2 mid (m_rho^2 < s < 4)    : {a2*1e11:9.1f}  "
      f"({100*a2/tot:5.1f}%)")
print(f"Regime 3 UV  (s > 4 GeV^2)        : {a3*1e11:9.1f}  "
      f"({100*a3/tot:5.1f}%)")
print(f"TOTAL a_mu^HVP (foam)             : {tot*1e11:9.1f}")
print(f"  known HVP (data)                : ~6900")
print(f"  observed gap exp-SM             : ~249")
print(f"  foam/total ratio                : {tot*1e11/6900:.3f}")

# ---------- per-channel breakdown ----------
def chan(f, a, b):
    v, _ = quad(lambda s: K(s) * f(s) / s, a, b, epsabs=1e-13, limit=400,
                points=[x for x in (M_OM**2, M_PHI**2, 4*M_K**2)
                        if a < x < b])
    return pref * v * 1e11

print("\n===== per-channel (x1e-11) =====")
c = {}
c['pipi pure (rho, no omega mix)'] = chan(
    lambda s: R_pipi(s, mix=False), s1, s3)
c['pipi with rho-omega mix'] = chan(R_pipi, s1, s3)
c['rho-omega interference alone'] = c['pipi with rho-omega mix'] - \
    c['pipi pure (rho, no omega mix)']
c['omega narrow BW'] = chan(R_omega, s2, s3)
c['phi narrow BW'] = chan(R_phi, s2, s3)
c['KKbar (m_K=0.798 foam)'] = chan(R_KK, s2, s3)
c['KKbar (m_K=0.495 physical)'] = chan(
    lambda s: R_KK(s, 0.495), s2, s3)
c['Regge n=4,9 + continuum'] = chan(R_regge, s2, s3)
c['UV pQCD'] = a3 * 1e11
for k, v in c.items():
    print(f"  {k:38s}: {v:9.1f}")

print("\nstandard dispersive reference: pipi ~500-540 x1e-10 total,")
print("  omega ~36-38, phi ~34-37, KKbar ~21-23, rho-omega mix ~+2-3")
print(f"within 3% of 6900? {abs(tot*1e11-6900)/6900 < 0.03} "
      f"(deficit {6900-tot*1e11:.0f})")

# ================= PASS 3: foam m_s, m_K, running alpha_s =================
print("\n" + "="*60)
print("PASS 3 — foam m_s, m_K, running alpha_s")
print("="*60)

# --- Task 1: m_s = sqrt(m_q^2 + sigma*Delta_n), Delta_n = 1 ---
M_Q_F = np.sqrt(SIGMA_STR / np.pi)              # 0.3033 (T61 check)
M_S_FOAM = np.sqrt(M_Q_F**2 + SIGMA_STR * 1)   # one radial node = one sigma of m^2
M_S_FOAM3 = np.sqrt(M_Q_F**2 + SIGMA_STR * 3)  # alt: first tower step n=1->4
print(f"m_q check = {M_Q_F*1e3:.1f} MeV")
print(f"m_s foam (Delta_n=1) = {M_S_FOAM*1e3:.0f} MeV  [DERIVED]")
print(f"   vs constituent ~500 (23% high), T66 m_s_eff 670 (8% low),")
print(f"   vs current m_s = 93 MeV (constituent != current, expected)")
print(f"   alt Delta_n=3 (tower n=1->4): {M_S_FOAM3*1e3:.0f} MeV -- too heavy")

# --- Task 2: foam m_K = (m_q + m_s)/2 (half-sum constituent) ---
M_K_FOAM = (M_Q_F + M_S_FOAM) / 2              # 0.460 GeV
# hyperfine alternative (De Rujula), A fixed by the pion:
A_hyp = M_Q_F**2 * (2*M_Q_F - M_PI)
M_K_HYP = M_Q_F + M_S_FOAM - A_hyp/(M_Q_F*M_S_FOAM)   # 0.689, overshoots
print(f"m_K foam half-sum = {M_K_FOAM*1e3:.0f} MeV vs physical 495 "
      f"({100*(M_K_FOAM-0.495)/0.495:+.0f}%)  [DERIVED]")
print(f"   hyperfine alt = {M_K_HYP*1e3:.0f} MeV (39% high, rejected)")
print(f"   KKbar threshold 2 m_K = {2*M_K_FOAM:.3f} GeV vs physical 0.988")

# --- Task 3: alpha_s(2 GeV) via NSQCD 3-loop beta, nf=3 ---
def run_as(a0, mu0, mu1, nf=3):
    b0, b1, b2 = 9.0, 64.0, 3863.0/6.0
    def beta(a, t):
        return (-(b0/(2*np.pi))*a**2 - (b1/(4*np.pi**2))*a**3
                - (b2/(64*np.pi**3))*a**4)
    from scipy.integrate import odeint
    t = np.linspace(np.log(mu0), np.log(mu1), 2000)
    return float(odeint(beta, a0, t)[-1, 0])

MU_CONF = np.sqrt(SIGMA_STR)                  # 0.5376 GeV confinement scale
A_S_2GEV = run_as(ALPHA_S, MU_CONF, 2.0, 3)
print(f"alpha_s(2 GeV) = {A_S_2GEV:.4f}  [DERIVED: 0.405 @ sqrt(sigma)"
      f" = {MU_CONF:.3f} GeV, 3-loop nf=3]")
print(f"   (machinery check: PDG anchor gives alpha_s(2 GeV) = 0.324)")

# --- Task 4: rerun with foam values ---
def R_KK_foam(s):
    if s <= 4 * M_K_FOAM**2:
        return 0.0
    return 0.5 * (1 - 4 * M_K_FOAM**2 / s) ** 1.5

def R_pqcd_run(s):
    q2 = 4/9 + 1/9 + 1/9 + 4/9
    return NC * q2 * (1 + A_S_2GEV / np.pi)

def integrand_p3(s):
    if s < 4 * M_PI**2:
        return 0.0
    if s < M_RHO**2:
        r = R_pipi(s)
    elif s < 4.0:
        r = (R_pipi(s) + R_regge(s) + R_omega(s) + R_phi(s)
             + R_KK_foam(s))
    else:
        r = R_pqcd_run(s)
    return K(s) * r / s

J1, _ = quad(integrand_p3, s1, s2, epsabs=1e-13, limit=400)
J2, _ = quad(integrand_p3, s2, s3, epsabs=1e-13, limit=400,
             points=[M_OM**2, M_PHI**2, 4*M_K_FOAM**2])
J3, _ = quad(integrand_p3, s3, S_TOP, epsabs=1e-13, limit=400)
b1_, b2_, b3_ = pref*J1, pref*J2, pref*J3
tot3 = b1_ + b2_ + b3_
kk_f = pref * quad(lambda s: K(s)*R_KK_foam(s)/s, s2, s3,
                   epsabs=1e-13, limit=400, points=[4*M_K_FOAM**2])[0]
print(f"\nPASS 3 RESULTS (x1e-11):")
print(f"  IR {b1_*1e11:8.1f} | mid {b2_*1e11:8.1f} | UV {b3_*1e11:8.1f}")
print(f"  KKbar (foam m_K={M_K_FOAM:.3f}) = {kk_f*1e11:.1f} "
      f"(vs 4.0 with m_K=0.798, ~220 physical)")
print(f"  UV with alpha_s(2 GeV)={A_S_2GEV:.3f} = {b3_*1e11:.1f} "
      f"(was 617.9 with 0.405)")
print(f"  TOTAL = {tot3*1e11:.1f}  (ratio {tot3*1e11/6900:.3f})")
print(f"  within 2% of 6900? {abs(tot3*1e11-6900)/6900 < 0.02}")

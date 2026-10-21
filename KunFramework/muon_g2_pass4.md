# MUON g-2 HVP — PASS 4: phi-KKbar CLOSURE + THE 249 QUESTION
# 2026-10-17. Definitive analysis. Code: muon_g2_hvp.py (+pass-3 block).
# Labels: DERIVED = foam chain only; MEASURED = external input; FAILED = test failed.

=====================================================================
TASK 1 — FOAM phi MASS                                    [FAILED]
=====================================================================
phi(1020) ~ pure ssbar. Foam m_s = 617 MeV (pass 3, Delta_n=1).
  Bare:    m_phi = 2 m_s           = 1234 MeV  vs 1020  (+21%)
  T76:     +color-magnetic shift   = +110 MeV  (|psi(0)|^2=3/(4 pi r_conf^3),
           dM = 8 pi a_s/9m_s^2 |psi|^2)       -> 1345 MeV  (+32%)
  DGG:     pion-calibrated hyperfine a=0.0577 GeV^3 -> +38 -> 1272 MeV (+25%)
VERDICT: FAILS the 10% test. Root cause is upstream: constituent m_s
= 617 MeV overshoots — the empirical m_s ~ 500 MeV exists precisely so
that 2 m_s ~ m_phi. The Delta_n=1 Regge step gives a strange quark
~24% too heavy for vector-meson spectroscopy (though within 8% of
T66's m_s_eff = 670 for baryons). The phi is REAL in the foam spectrum
(it is the ssbar ground vector) but its mass lands high. This is a
genuine model discrepancy, not a convention issue.

=====================================================================
TASK 2 — phi -> KKbar FROM STRING UNIVERSALITY         [DERIVED, partial]
=====================================================================
Foam prescription: flux-tube reconnection gives ONE universal coupling
g = g_rhopipi = m_rho/(sqrt(2) f_pi) = 5.958 (KSRF, DERIVED from
T32/T61 inputs — zero free parameters).
  Gamma(V->PP) = g^2 k^3 / (6 pi m_V^2)
Results vs measured (total Gamma(phi->KKbar) = 3.54 MeV):
  phi->K+K-   : foam 3.19 MeV vs measured 2.98 MeV   ( 7% — good)
  phi->K0 K0  : foam 2.53 MeV vs measured 1.37 MeV   (85% high)
  TOTAL       : foam 5.71 MeV vs measured 3.54 MeV   (61% high)
  required suppression factor: 0.62 (not derived — the K0 overshoot is
  phase-space sensitive: k_K0 = 112 MeV in the cubic)
3P0-style suppression e^{-pi(m_s^2-m_q^2)/sigma} = e^{-pi} = 0.043
overshoots downward (Gamma -> 0.25 MeV). String universality gets the
charged channel right; neutral needs a ~0.6 barrier factor not yet
derived from foam.
KKbar CHANNEL RECOMPUTED (resolves pass-2/3 "missing KKbar" issue):
  phi->KKbar piece (already inside R_phi via BR_ee): ~322 x1e-11
  continuum KKbar above 1.05 GeV (foam m_K=0.460):      79 x1e-11
  ----------------------------------------------------
  effective KKbar sector:                            ~400 x1e-11
  physical total KKbar ~350 (phi ~310 + continuum ~40).
  PASS-2/3 CORRECTION: the "~220" reference used before was K+K- alone.
  With K_S K_L (~130) the standard total is ~350, and the phi part was
  never missing — it lives inside R_phi. The foam continuum over-
  produces by ~2x because m_K=0.460 lowers the threshold.
  ==> phi-KKbar closure: CLOSED (total overshoots ~15%, not a deficit)

=====================================================================
TASK 3 — rho-omega MIXING SIGN                          [RESOLVED]
=====================================================================
Dip-position test of the GS interference factor (delta=-3.5+0.2i MeV):
  +delta (current): |F_pi|^2 minimum at E=776 MeV, depth 0.66,
                    enhancement above 782 — matches the physical
                    pi-pi pattern (dip on low side of omega).
  -delta (flipped): minimum at E=788 MeV (above the pole) — WRONG.
  sqrt(s) width / GS eps*s variants: same structure as current.
VERDICT: the current convention is correct; no fix needed. The net
-56.6 x1e-11 is the right sign and order — dispersive analyses find
the rho-omega interference SUBTRACTS tens of x1e-11 from a_mu^pipi.
The pass-2 report's "~+30" reference was erroneous. RESOLVED.

=====================================================================
TASK 4 — THE FUNDAMENTAL QUESTION: can DeltaP=2gamma/r
         produce the 249 x1e-11 anomaly?
=====================================================================
Full-precision foam total (pass-3 config: m_s=617 DERIVED, m_K=460
DERIVED, alpha_s(2 GeV)=0.201 DERIVED, all else as pass 2):
    a_mu^HVP(foam) = 7039.4 x1e-11   (ratio 1.020 vs ~6900)
    IR 2807.6 | mid 3649.4 | UV 582.4

Q1: Can foam pressure at r_mu = 1/m_mu = 0.56 fm produce a correction
    outside the dispersive integral?
    NO — structurally. a_mu^HVP = (alpha^2/3pi^2) int K(s) R(s)/s ds
    is exact given R(s): the optical theorem + analyticity map ALL
    hadronic vacuum-polarization content onto R(s). Foam physics
    (DeltaP=2gamma/r) can only enter through R(s)'s parameters —
    masses, widths, threshold positions — all of which we already
    model. There is no integration region or topology the dispersive
    formula misses; it sums over every hadronic state by construction.

Q2: Does foam predict a modified FUNCTIONAL FORM of R(s)?
    Foam's model R(s) does differ from measured R(s): Regge+continuum
    mid-region excess (~+300 vs data), UV Born-term excess, KKbar
    continuum ~2x. Net foam-model total sits +139 x1e-11 ABOVE the
    data-driven number. Taken at face value that would shrink the
    249 gap to ~110 — but this is model-vs-data residual concentrated
    in the wrong channels, not a controlled foam correction. R(s) is
    MEASURED; foam cannot legitimately override data with its model
    residuals and call the difference a prediction. The honest
    statement: where foam's R(s) disagrees with data, data wins.

Q3: A foam contribution at the muon Compton wavelength missed by
    construction?
    None within the stated mechanism. The kernel K(s) is pure QED; a
    foam modification of K(s) (e.g. a confinement-scale correction to
    the muon-photon vertex or a new light state coupling to muons)
    would be a NEW non-HVP channel — and no T-element derives one.
    The framework's own g-2 note (KunFramework_v2.txt L4392) already
    states: "foam corrections at the hadronic scale are order-unity...
    the perturbative channel fails by construction. The electroweak
    channel via T78/T79 Higgs-muon Yukawa is the open path."

VERDICT — DEFINITIVE:
  The 249 x1e-11 anomaly is OUTSIDE the scope of DeltaP=2gamma/r
  through the hadronic channel. Foam reproduces the SM HVP total
  (7039 vs 6900, +2.0%) — i.e. it reproduces the dispersive SM
  answer, not the experimental one. Any foam explanation of the gap
  must come from a different sector (electroweak/Yukawa via T78-T79,
  or a genuinely new light state), not from HVP. This is a negative
  result, cleanly established: the hadronic foam channel is closed.

=====================================================================
DEFINITIVE CHANNEL TABLE (foam vs standard dispersive, x1e-11)
=====================================================================
  channel                 foam      standard      status
  pi-pi pure rho           4024      ~5000-5400    20% low (VDM only)
  rho-omega interference    -57      ~-30 to -60   consistent [RESOLVED]
  omega                     392      ~370-400      good [MEASURED inputs]
  phi (incl phi->KK ~322)   387      ~350-400      good [MEASURED inputs]
  KKbar continuum            82      ~40           2x high (foam m_K low)
  Regge n=4,9 + cont.      1628      ~1300-1600    plausible [DERIVED]
  UV pQCD (alpha_s=0.201)   582      ~200-300      high (Born-dominated)
  ----------------------------------------------------------
  TOTAL                    7039      ~6900         +2.0%

  Zero-parameter accounting: DERIVED inputs = sigma, m_rho (Regge),
  f_pi (KSRF g), alpha_s=4/pi^2, m_s, m_K, alpha_s(2GeV run).
  MEASURED inputs = m_om, m_phi, widths, BR_ee, delta_rw, m_pi, m_mu.
  T87 status: NOT flagged — 2.0% total is at the gate edge and rests
  on partial error cancellation (pi-pi deficit vs Regge/UV excess);
  per-channel residuals exceed the anomaly itself.

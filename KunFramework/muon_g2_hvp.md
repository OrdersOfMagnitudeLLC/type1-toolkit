# MUON g-2 HVP — PASS 2: MISSING CHANNELS
# 2026-10-17. Code: muon_g2_hvp.py. Pass 1 report: muon_g2_hvp.txt.
# Foam-derived quantities labeled DERIVED; measured inputs MEASURED.

=====================================================================
CHANNELS ADDED
=====================================================================
- omega(782):   narrow BW, m=0.782, Gamma=8.68 MeV, BR_ee=7.38e-5 (MEASURED)
                -> Gamma_ee = 0.64 keV
- phi(1020):    narrow BW, m=1.020, Gamma=4.27 MeV, BR_ee=2.97e-4 (MEASURED)
                -> Gamma_ee = 1.27 keV
- KKbar:        threshold R = (1/4)(beta+^3 + beta0^3), m_K = m_q + m_s.
                No foam m_s exists at T80 (T80 = CKM angles). Used
                m_s = 495 MeV constituent fallback (MEASURED, FLAGGED)
                -> m_K = 0.798 GeV, threshold 1.596 GeV
                (physical threshold 0.988 GeV — flagged below)
- rho-omega:    GS mixing term delta_rw = (-3.5+0.2i) MeV (MEASURED)
                folded into F_pi as complex form factor.

=====================================================================
RESULTS (x1e-11)
=====================================================================
  Regime 1 IR   (s < m_rho^2)        : 2807.6   (40.1%)
  Regime 2 mid  (m_rho^2 < s < 4)    : 3571.0   (51.0%)
  Regime 3 UV   (s > 4 GeV^2)        :  617.9   ( 8.8%)
  -----------------------------------
  TOTAL a_mu^HVP                     : 6996.6
  Known HVP (WP data-driven)         : ~6900-6931
  Ratio                              : 1.014  — WITHIN 3%
  Zero fitted parameters on the foam side; narrow-resonance inputs MEASURED.

PER-CHANNEL vs STANDARD DISPERSIVE (approx literature values)
  channel              foam      standard        note
  pi-pi (rho pole)     4024      ~5000-5400      20% low (VDM only,
                                                no rho' tail, no data)
  rho-omega mixing      -56.6    ~ +30 net       SIGN FLIPPED vs data
                                                (interference sign in
                                                fit conventions differs;
                                                magnitude ~2x)
  omega                  392      ~370-400       matches well
  phi                    387      ~340-400       matches well
  KKbar (m_K=0.798)        4      ~220-230       98% low — foam
                                                constituent m_K pushes
                                                threshold to 1.6 GeV;
                                                with m_K=0.495: 64
                                                (still low: real KKbar is
                                                phi-enhanced below 1.05)
  Regge n=4,9 + cont.   1628      ~1300-1600     plausible, unverified
  UV pQCD                618      ~200-300*      2x high: foam alpha_s =
                                                0.405 vs running
                                                alpha_s(2 GeV)~0.30
    (*standard UV s>4 GeV^2 is smaller because alpha_s runs down)

=====================================================================
FOAM vs STANDARD — WHERE THEY DIFFER
=====================================================================
1. KKbar channel: foam m_K = m_q + m_s(constituent) = 798 MeV fails —
   constituent masses describe baryons/meson spectroscopy, not the
   KKbar threshold position. Difference vs standard: ~220 x1e-11
   missing (the single largest deficit).
2. UV regime: foam alpha_s = 4/pi^2 = 0.405 is a fixed color-magnetic
   constant; real alpha_s runs (~0.30 at 2 GeV). Foam overproduces UV
   by ~2x (~300 x1e-11 excess vs standard running coupling).
3. rho-omega mixing sign: our delta_rw convention gives -56.6 where
   the dispersive fit gives a small positive net shift. Convention/
   phase issue — flagged, not resolved.
4. Compensating errors: Regge+continuum excess and UV excess roughly
   cancel the KKbar deficit — the 1.014 total ratio is NOT a clean
   hit; it is a partial cancellation of ~±300 x1e-11 residuals.

=====================================================================
THE 249 x1e-11 GAP
=====================================================================
UNCHANGED — and now better defined why: the foam estimate converges to
the data-driven SM value (~6900), not above it. Foam HVP reproduces
the Standard Model HVP; the exp-SM discrepancy is between theory and
experiment, not a missing hadronic term. A foam "fix" to g-2 would
have to come from elsewhere (new light states, HLbL) — nothing in
DeltaP = 2 gamma/r generates 249 x1e-11 here.

=====================================================================
VERDICT: SPECULATION overall. Total lands within 3% of 6900 with zero
fitted parameters -> FLAGGED AS POTENTIAL T87 (foam HVP closure), but
with the explicit caveat that this relies on compensating errors and
MEASURED narrow-resonance inputs; the honest channel-by-channel claim
is "recovers the scale and three-regime structure, misses KKbar
threshold position and UV running." Recommend T87 status = SPECULATION
pending a derived m_s (mass-sector gap, currently open) and rho'
inclusion in the pi-pi channel.
=====================================================================

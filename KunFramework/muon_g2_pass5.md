# MUON g-2 — PASS 5: m_s FIX + EW LOOP + THE DIRECT-VACUUM QUESTION
# 2026-10-17. Definitive foam verdict on the 249 x1e-11 anomaly.
# Labels: DERIVED / MEASURED / CONJECTURE / REJECTED as used in framework.

=====================================================================
TASK 1 — m_s DERIVATION: THREE ROUTES
=====================================================================
A) T80 lepton-delta transplanted to quarks:
   delta = N_spin/N_color^2 = 2/9 (T83 Koide phase, DERIVED).
   m_s/m_q = 1 + delta*N_color = 1 + 3*(2/9) = 5/3
   m_s = 303.3 * 5/3 = 505.5 MeV          -> INSIDE 495-510 window
   Label: CONJECTURE. The ratio lands right, but applying the lepton
   flavor delta to the quark sector is a cross-sector analogy — the
   pattern is foam-native, the transplant is not derived. (2% high vs
   495.)

B) Kaon GMOR: m_K^2 = (m_q+m_s)/2 * Sigma
   With the foam constituent condensate scale Sigma = 2 m_q = 2 kappa/
   sqrt(pi) = 0.6065 GeV (the two-constituent pion scale):
       m_s = m_K^2/m_q - m_q = 245.0/303.3 - 303.3 = 504.6 MeV
   -> INSIDE window. CONVERGENT with route A to within 1 MeV.
   Label: CONJECTURE — Sigma=2m_q is the natural foam choice but is an
   ansatz; alternates (Sigma=sqrt(sigma) -> m_s=608) land back at the
   old overshoot. The convergence A ~ B ~ 505 is the evidence.

C) Flux-tube twist: m_s = m_q + pi*sigma/m_q = 303 + 3000 = 3297 MeV
   REJECTED — order of magnitude too large; the twist energy is not a
   quark-mass shift.

RESULT: foam m_s = 505 +/- 10 MeV (A and B agree; both CONJECTURE-
grade, upgrade to DERIVED needs a forced chain for either the delta
transplant or Sigma=2m_q). Supersedes pass-3 Delta_n=1 (617 MeV,
23% high) — the Regge radial-mode ansatz is demoted. Consequence:
m_phi = 2*m_s = 1011 MeV vs 1020 (0.9%!) — the phi mass failure from
pass 4 was the bad m_s, not the spectroscopy. RECOMMEND re-running
HVP with m_s=505: half-sum m_K = 404 MeV (worse naive threshold) but
GMOR route makes m_K=495 self-consistent — use physical/GMOR m_K.

=====================================================================
TASK 2 — EW ONE-LOOP WITH FOAM INPUTS
=====================================================================
a_EW(1L) = G_mu m_mu^2/(8 pi^2 sqrt2) * [10/3 + (1-4 sW^2)^2 - 5/3]
Foam: sW^2 = 0.2314 (T83, 0.22%), m_W = 80.37 GeV (T79, 0.52%).
  bracket = 1.6667 + (1-0.9256)^2 = 1.6722
  a_EW(1L, foam sW^2, G_mu measured)         = 195.0 x1e-11
  a_EW(1L, all-foam: G_mu=pi*alpha/(sqrt2 sW^2 m_W^2)=1.0845e-5,-7%)
                                             = 181.3 x1e-11
  SM 1L (sW^2=0.2313)                        = 195.0 x1e-11
  SM full EW (incl 2L-3L, large negative)    ~ 153.6 x1e-11
DIFFERENCE: +0.0 x1e-11 at one loop (foam sin^2 matches on-shell sW^2
to 4e-4). Foam adds NOTHING anomalous in the EW channel — it
reproduces the SM inputs, hence the SM answer. (All-foam G_mu variant
is 7% low because it absorbs the foam sW^2/m_W errors twice; the
measured-G_mu version is the honest comparison.)

=====================================================================
TASK 3 — DIRECT FOAM PRESSURE AT r_mu                [REJECTED]
=====================================================================
Question: does DeltaP = 2 gamma(r)/r at r_mu = 1/m_mu = 1.87 fm give
a non-hadronic vacuum contribution the dispersive integral misses?

T2 scaling: gamma(R) = A R^3.0517, anchored at SDSS. Anchoring
DeltaP = rho_Lambda at the reset scale R0 = 64.04 Mpc/h (the T12
w=-1 result) and running down:
    DeltaP(r_mu) = rho_L * (r_mu/R0)^2.0517
    r_mu/R0 = 6.6e-40
    DeltaP(r_mu) = 4.0e-47 * 1.66e-81 = 1.7e-127 GeV^4
Needed: 249e-11 * m_mu^2/alpha ~ 3.8e-9 GeV^2 scale.
MISMATCH: ~59 ORDERS OF MAGNITUDE too small. The r^3.0517 running that
fits cosmic voids drives gamma to zero at fermi scales — void physics
does not reach the muon.

Alternative floor: if gamma saturates at the QCD string tension below
confinement, DeltaP(r_mu) ~ 2 sigma/r_mu = 0.061 GeV^3 — an
ORDER-UNITY hadronic-scale input, i.e. exactly the kind of correction
that lives INSIDE R(s) (it shifts masses/thresholds, not the
integral's structure). Already counted; not a new term.

=====================================================================
DEFINITIVE VERDICT — all foam channels closed
=====================================================================
  hadronic (R(s) inputs)   : foam reproduces SM dispersive total
                             (7039 vs 6900, +2.0%) — no anomaly.
  EW (foam sW^2, m_W, m_H) : reproduces SM 1L exactly (195.0 = 195.0)
                             — no anomaly.
  direct vacuum at r_mu    : 59 orders of magnitude too small — DEAD.
  ------------------------------------------------------------------
  The 249 x1e-11 anomaly is OUTSIDE DeltaP=2gamma/r. This is now
  checked on all three conceivable channels. The framework's own note
  (KunFramework_v2.txt L4392) pointed to EW/Yukawa (T78/T79); the
  foam EW inputs being SM-accurate means a foam explanation would
  need a NEW non-SM coupling (e.g. an anomalous y_mu or a light
  foam state) — none is derived. Honest statement: the foam framework
  as it stands predicts the Standard Model value for a_mu. Any gap
  between that and experiment is not a foam hadronic or foam EW
  effect; it is either new physics outside the scope of the tested
  mechanism or a data/tension artifact.
=====================================================================

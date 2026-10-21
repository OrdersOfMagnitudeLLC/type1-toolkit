# MUON ANOMALY HUNT — foam mechanisms for the 249 x1e-11 gap
# 2026-10-17. Every line SPECULATION unless stated otherwise.
# Criterion: flag anything within 1 order of magnitude of 249 x1e-11.
# Prior context: muon_g2_pass5.md closed HVP, EW, and vacuum channels.

=====================================================================
TEST 1 — LIGHT Z' / RESONANCE 10-200 MeV               [SPECULATION]
=====================================================================
T32 Regge tower (n=1,4,9, sigma=0.289):
    masses: 776, 1552, 2328 MeV — NOTHING below 500 MeV except rho.
    No tower mode lands in 10-200 MeV.
Delta=2/9-scaled candidates (ad hoc products, NOT framework modes):
    m_q*delta = 67.4 | m_q*delta/2 = 33.7 | kappa*delta = 119.5 |
    kappa/(2pi)*delta = 19.0 | m_pi*delta = 30.0 | m_pi = 135 |
    m_q/pi = 96.5 | sqrt(sigma)/4 = 134.4   [all MeV]
    Several land in-window but are numerology — foam produces no
    discrete 10-200 MeV state; no muon-specific coupling derived.
VERDICT: NO candidate. A foam Z' in this window would also be
strongly constrained by BaBar/NA64/KLOE — dead on experiment too.

=====================================================================
TEST 2 — SECOND GEOMETRIC-MEAN SCALAR                  [FLAGGED]
=====================================================================
T78 pattern m_H = sqrt(m_top m_Z) = 125.6 GeV check: PASSES
    (125.6 vs 125.2 — the anchor works).
Scan sqrt(m_i m_j):
    sqrt(m_mu m_tau)  = 433.3 MeV   <-- IN RANGE (10 MeV - 1 GeV)
    sqrt(m_e m_tau)   =  30.1 MeV   <-- in range
    sqrt(m_e m_mu)    =   7.3 MeV   (below window edge)
    sqrt(m_W m_mu)    = 2914  MeV   (over)
    sqrt(m_mu m_Z)    = 3104  MeV   (over)
S = sqrt(m_mu m_tau) = 433 MeV is the only leptonic combination that
lands in-window AND carries a muon leg (muon-philic by construction).
Required coupling to produce delta_a_mu = 249e-11 via scalar loop
    delta_a ~ g^2 m_mu^2/(8 pi^2 m_S^2) -> g = 1.8e-3  (g^2=3.3e-6)
Muon Yukawa y_mu = m_mu/v = 4.3e-4 -> required g is ~4.2x y_mu.
If foam coupling is y_mu*(1+delta*N_c)=y_mu*5/3=7.1e-4: gives ~40e-11.
If g ~ m_mu/r_conf = 5.7e-2: gives ~2.5e-8 (way over).
==> FLAGGED: the required coupling bracket straddles 249e-11 within
one order for foam-plausible y_mu scalings. This is the ONLY channel
in the hunt not excluded by orders of magnitude. NOTE: a 433 MeV
muon-coupled scalar is heavily constrained (e+e- -> mu+mu- + S,
KLOE, BABAR, LHCb); survival requires ~muon-specific coupling only.

=====================================================================
TEST 3 — LEPTON-RATIO MODE r_e vs r_mu                 [DEAD]
=====================================================================
gamma(r) = sigma*(r/1 fm)^3.0517 (nuclear anchor, user prescription):
    r_e = 386 fm   -> gamma(r_e) = 2.3e7 GeV^2
    r_mu = 1.87 fm -> gamma(r_mu) = 1.95 GeV^2
DeltaP between the two modes ~ 2 gamma/r differences ~ 10^4-10^8 GeV^3
vs needed 249e-11*m_mu^2/alpha = 3.8e-9 GeV^2 — off by ~9+ orders
MISMATCH. The lepton ratio produces no resonance; m_mu/m_e = 206.77
is a mass eigenvalue (T80), not a radial foam mode with its own wall.
DEAD.

=====================================================================
TEST 4 — MUON MICRO-PRESSURE FIELD AT r_mu             [DEAD]
=====================================================================
gamma(r_mu) = sigma*(1.87)^3.0517 = 1.95 GeV^2 (nuclear-anchored T2)
DeltaP = 2 gamma/r_mu = 0.41 GeV^3 -> normalized DeltaP/m_mu = 3.9 GeV^2
Needed: 3.8e-9 GeV^2  -> OVERSHOOTS by ~1e9 (nine orders too big).
Cross-check with pass 5: the same quantity anchored at SDSS/voids
UNDERSHOT by 59 orders. The two anchors bracket ~140 orders of
uncertainty — DeltaP at muon scale is anchor-undefined; neither lands
near the anomaly. Mediator: none derived — DeltaP=2gamma/r is a
vacuum-stress background, not a particle exchange; it would act as a
scalar vacuum expectation (Higgs-like shift of m_mu), not a loop.
VERDICT: DEAD on magnitude; mechanism type (direct vacuum stress)
would also shift m_mu itself — incompatible with measured m_mu
unless exactly tuned, i.e. not a prediction.

=====================================================================
HUNT SUMMARY
=====================================================================
  Test 1 light Z'           : NO foam state 10-200 MeV        DEAD
  Test 2 sqrt(m_mu m_tau)   : 433 MeV scalar, coupling g~1.8e-3
                              needed (~4x y_mu)               FLAGGED
  Test 3 lepton-ratio mode  : ~9 orders off                   DEAD
  Test 4 micro-pressure     : ~9 orders over (anchor-undef)   DEAD
  ------------------------------------------------------------------
  ONE surviving lead: a muonphilic scalar S = sqrt(m_mu m_tau) =
  433 MeV. It is SPECULATION — the geometric-mean pattern is T78's,
  but no foam theorem derives this second scalar or its coupling.
  Experimental status: likely excluded by KLOE/BaBar unless it
  couples ONLY to muons; worth one line in the framework as an
  open CONJECTURE, nothing more. Everything else in the hunt is
  dead by orders of magnitude. The 249 x1e-11 anomaly remains
  unexplained by DeltaP = 2 gamma/r.
=====================================================================

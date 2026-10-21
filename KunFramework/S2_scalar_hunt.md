# S2 = sqrt(m_mu m_tau) = 433 MeV — FULL CHARACTERIZATION
# 2026-10-17. Verdict: CONJECTURE (not foam-forced; coupling is a fit;
# parameter space essentially excluded). Labels per line.

=====================================================================
TASK 1 — DOES IT EXIST IN DATA?                        [EXCLUDED-ish]
=====================================================================
No observed resonance anywhere near 433 MeV in muon channels.
Closest experimental programs:
  - BaBar e+e- -> mu+mu- + Z' (4-muon final state), Phys.Rev.D 94,
    011102 (2016): covered m_Z' = 0.212-10.27 GeV — NULL result. The
    g-2-favored L_mu-L_tau band at 433 MeV is excluded by this search.
  - KLOE phi -> mu+mu-Z', e+e- -> hZ': covered ~GeV-scale, null.
  - NA64-mu (2404.06982 + pilot): excluded the L_mu-L_tau g-2 band
    for m_Z' >~ 20-40 MeV at g_Z' >~ 6e-4; muonphilic scalar reach
    projected to m_S ~< 300 MeV at g_S ~ 1e-3.
  - Neutrino trident (CCFR) bounds on muon-coupled bosons exclude
    most of the (g, m_S) region that would produce delta a_mu ~ 249.
  - LHCb B -> K mu+mu-: no 433 MeV structure.
No anomaly has ever been reported at 433 MeV. A prompt mu+mu- scalar
at this mass with g ~ 2.5e-3 would have appeared in BaBar's 4-muon
spectrum. STATUS: effectively excluded as a mu+mu- resonance; only a
muonphilic INVISIBLE/decay-suppressed variant could evade, which is
exactly the class NA64 constrains.

=====================================================================
TASK 2 — FOAM PROPERTIES IF IT EXISTED                 [SPECULATION]
=====================================================================
Spin: scalar (spin-0) — inherits T78 scalar-pressure character; the
    muon-tau geometric mean is a scalar pairing, not a current.
Width (g = 1.8e-3):
    Gamma(S -> mu+mu-) = g^2 m_S/(8 pi) (1-4 m_mu^2/m_S^2)^{3/2}
                       = 3.7e-8 GeV = 37 meV
    Gamma(S -> e+e-)   = 56 meV (phase space ~1; only if it couples
                       to electrons — muonphilic-by-construction: 0)
    Gamma(S -> tau+tau-) = 0 (kinematically forbidden, 433 < 3553)
    Gamma(S -> gamma gamma) ~ alpha^2 g^2 m_S^3/(64 pi^3 m_mu^2)
                       ~ 1 eV-scale... BR(gg) ~ 3e-5
    Gamma_tot ~ 0.04-0.1 eV — extremely narrow.
Lifetime: c tau ~ 5.3e-6 mm (5 nm) — PROMPT, decays at production.
Production: e+e- -> mu+mu- S (radiative return / 4-muon), muon
    bremsstrahlung mu N -> mu N S (NA64-mu channel), K+ -> mu+ nu S
    only if quark-coupled (muonphilic: no).

=====================================================================
TASK 3 — DOES g = 1.8e-3 HIT 249?                      [NO — FIT]
=====================================================================
Delta a_mu = g^2/(8 pi^2) * I,   I = int_0^1 2x^2(1-x)/(x^2+(1-x)(m_S/m_mu)^2) dx
  I = 0.030108
  g = 1.8e-3   -> Delta a_mu = 123.5 x1e-11   (HALF the target)
  g needed     = 2.56e-3  (g^2 = 6.5e-6)
Foam-plausible couplings (DERIVED scale references):
  y_mu = m_mu/v = 4.3e-4        -> 7.0 x1e-11   (36x short)
  y_mu*(1+delta N_c)=y_mu*5/3   -> 19.5 x1e-11  (13x short)
VERDICT: g is a FREE PARAMETER tuned to hit 249. No foam-derived
coupling lands within an order of magnitude of the required 2.6e-3.
The scalar loop integral at m_S = 433 MeV is too small (suppression
~ (m_mu/m_S)^2 * ln-ish) — this is why real "muonphilic scalar"
solutions to g-2 sit at m_S ~< 30-100 MeV, not 433.

=====================================================================
TASK 4 — IS sqrt(m_mu m_tau) FOAM-FORCED?              [NO]
=====================================================================
T78 in the framework is VELTMAN'S RELATION: m_H = sqrt(m_top m_Z)
arises from the quadratic-divergence cancellation condition in the
electroweak sector (2 m_t^2 + m_H^2 = 4 m_W^2 + 2 m_Z^2 structure);
it is specific to the Higgs potential's radiative stability. There
is NO leptonic analog of the Veltman condition — charged leptons are
Dirac fermions with no self-energy minimization that would pin a
scalar mass to sqrt(m_mu m_tau).
sqrt(m_mu m_tau) = 433 MeV is geometric-mean PATTERN MATCHING on
T78's motif. It also lands suspiciously close to m_K = 495 and
m_eta = 547 — the window is crowded; a sqrt-product landing in it is
not evidence.
VERDICT: NOT foam-forced. CONJECTURE at best.

=====================================================================
FINAL STATUS OF S2
=====================================================================
  mass formula      : sqrt(m_mu m_tau) — CONJECTURE (T78 motif, no
                      foam mechanism forces it)
  required coupling : g = 2.56e-3 — FIT, ~36x above y_mu^2
  a_mu contribution : 123.5 x1e-11 at the already-optimistic g=1.8e-3
  experimental      : BaBar 4-muon + trident + NA64 -> essentially
                      excluded in the needed coupling range
  ----------------------------------------------------------------
  CLASSIFICATION: CONJECTURE — and as an anomaly explanation, DEAD.
  The anomaly's remaining foam-explanation space is now: (a) lattice/
  dispersive resolution (gap is a data artifact), or (b) a foam
  degree of freedom not yet derived (e.g. invisible-sector scalar
  with forced coupling). S2 as constructed is neither DERIVED nor
  viable.
=====================================================================

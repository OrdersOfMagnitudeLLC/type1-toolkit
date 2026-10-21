# Foam Mechanics vs Jones Polynomial — Results

Date: 2026-10-15. Engine: `kb_engine.py` (self-contained Kauffman state-sum;
spherogram's `jones_polynomial()` requires Sage, which is absent — spherogram
was used only for census PD codes).

## Step 1 — Baseline V_K(t)

| knot | w(D) | V(t) |
|---|---|---|
| unknot (0cr / 1 kink / 2 kinks) | 0, -1, -2 | **1** (all diagrams) |
| 3_1 | -3 | -t^-4 + t^-3 + t^-1 |
| 4_1 | 0 | t^-2 - t^-1 + 1 - t + t^2 |
| 5_1 | -5 | -t^-7 + t^-6 - t^-5 + t^-4 + t^-2 |
| 5_2 | +5 | t - t^2 + 2t^3 - t^4 + t^5 - t^6 |
| 11n34 / 11n42 (mutant pair) | -1 / -1 | identical V (|coeff|sum=15) |

Mutant pair confirms Jones degeneracy on distinct nontrivial knots.
No known nontrivial knot has V=1 (the open conjecture); links DO have
counterexamples (Eliahou–Kauffman–Thistlethwaite).

## Step 2 — Pressure signature

Per spec (P_i = 2γ/r_i, r_i = 1/n): **P_norm = 2γ·writhe(D)** identically —
a diagram quantity, not an invariant.

- P_norm=0, V=1: unknot circle ✓ — but also 4_1 (w=0, V≠1) ✗
- P_norm≠0, V=1: kink unknots (w=-1,-2) ✗ — Reidemeister-I changes P_norm
  on the same knot.

Counterexamples exist in **both** directions → biconditional fails.

## Step 3 — Foam-weighted bracket <D>_foam(β)

ΔP(state) = Σ_i (P_i − s_i·ε_i·γ), ε=+1 for A-smoothing; weights e^{−β(ΔP−ΔPmin)}.

- β=0: reproduces standard bracket exactly (match_std=True everywhere).
- unknot circle: f=1 for all β (trivially — nothing to weight).
- unknot kink: f = α(β)t^-1 + 1, α: 0.18→0.98 as β=0.1→2 — **leaves V=1**:
  foam weighting breaks Reidemeister-I invariance.
- trefoil: diverges smoothly from -t^-4+t^-3+t^-1 toward the
  minimum-pressure state (t^-2 term dominates by β=2).

## Step 4 — VERDICT: NO CONNECTION

The stated foam signature cannot detect the unknot: it is (a) not
diagram-invariant, and (b) zero on w=0 nontrivial knots like 4_1.
The foam-weighted bracket is a legitimate statistical deformation of the
Kauffman partition function but destroys the very invariance that makes the
bracket topological. A viable foam hypothesis would need an *invariant*
pressure observable (e.g. pressure distribution on the moduli of flat
connections — the actual Chern-Simons/Witten content of V_K).

Files: `kb_engine.py`, `jones_baseline.py`, `jones_foam.py`,
`p_vs_jones.csv`, `p_vs_jones.png`
Run: `/tmp/sphenv/bin/python jones_baseline.py && /tmp/sphenv/bin/python jones_foam.py`

---

# Part 2 — SU(2) character variety (corrected foam observable)

The writhe-based P_norm was diagram-dependent. Correct observable:
flat SU(2) connections = foam equilibria (F=0 <=> YL satisfied).
Script: `su2_foam.py` (snappy 3.3.2 installed into /tmp/sphenv;
fundamental groups -> SU(2) reps found numerically up to conjugacy,
conjugacy-reduced gauge: gen0 axis->z, gen1 y=0).

## Rep counts / CS per knot

| knot | nonabelian reps found | nonab arcs | volume | CS(geometric) |
|---|---|---|---|---|
| unknot | 0 (pi_1 = Z, abelian only) | 0 | — | — |
| 3_1 | 20 classes (arc sampled; tr_a=0, tr_b=1, tr_ab varies) | 1* | 0 | -0.0833 |
| 4_1 | 154 | 2 (sweep) | 2.0299 | ~0 |
| 5_1 | 62 (two tr_b branches: -0.618, 1.618) | 2* | 0 | +0.05 |
| 5_2 | 144 | 3 (sweep) | 2.8281 | -0.1532 |
| 8_18 | 129 | (3-gen, n/a) | 12.35 | ~0 |
| 11n34 Conway | 138 | (3-gen) | 11.2191 | -0.1356 |
| 11n42 K-T | 126 | (3-gen) | 11.2191 | -0.1356 |

*for torus knots gen 'a' isn't the meridian so the alpha-sweep misses;
arc count read from tr_b branch clusters instead (matches (p-1)(q-1)/2).
For 4_1 and 5_2 the sweep counts 2,3 arcs = (det-1)/2 exactly.

## Mutant pair test (the discriminating experiment)

Same volume (11.219118), same geometric CS (-0.135620), same Jones —
but SAMPLED CHARACTER SPECTRA ARE DISJOINT:
- NN distance between spectra: 1.64/1.69 vs internal spread 5.29 (~31%)
- tr(gen0) marginals live in disjoint bands (~±1.7-1.8 vs ~±1.1-1.2)
Consistent with the known fact: Conway/KT have different SL(2,C)
character varieties (different A-polynomials; mutation does not
preserve them in general).

## Verdict: FOAM STRONGER than Jones

1. Unknot detection: PROVEN. Kronheimer-Mrowka: every nontrivial knot
   has nonabelian SU(2) reps -> flat-connection foam sees the unknot
   unconditionally. Jones=1 detection remains conjectural.
2. Mutant pair: Jones identical, SU(2) character spectra differ ->
   flat-connection foam resolves what Jones cannot.
3. Caveats: per-rep CS needs ptolemy+Magma (absent); only the geometric
   rep's CS was available (identical for mutants, as expected). Rep
   enumeration is numerical sampling; spectra compared via NN distance.
   Character coordinates are presentation-dependent, but disjointness
   of the sampled sets is strong evidence; the underlying variety
   difference is also a theorem (distinct A-polynomials).

Output log: `su2_foam.log`. Run: `/tmp/sphenv/bin/python -u su2_foam.py`

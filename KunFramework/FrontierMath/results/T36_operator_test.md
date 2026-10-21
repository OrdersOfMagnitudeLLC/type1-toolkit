# T36 OPERATOR TEST — H_foam constructed explicitly
# 2026-10-15. File: /mnt/HIVE/FrontierMath/H_foam_operator.py
# Exploratory run — numbers, not a forced conclusion.

=====================================================================
CONSTRUCTION
=====================================================================
Foam configuration space: radial interval x = r/r_P in [1, X_max],
X_max = r_dS/r_P = 1.0173e61 (r_P = 1.616e-35 m, r_dS = sqrt(3/Lambda)
= 1.644e26 m). Dirichlet walls both ends.

  H psi = -d2psi/dx2 + (gamma/x^2) psi,   gamma = 0.3345 (T2)

In u = ln x:  -(e^{-u} phi')' + gamma e^{-u} phi = E e^{u} phi
Discretized as generalized symmetric tridiagonal K phi = E M phi,
LAPACK eigh_tridiagonal. Numeric run at U_c = 10 (x <= 2e4, 6000
nodes); full U_max = 140.5 spans 61 decades — beyond float64 for the
lowest eigenvalues, but the counting LAW is U-independent (Weyl-type),
continued analytically.

=====================================================================
RESULT 1 — SPECTRUM IS NOT n^2 AND NOT PRIMES
=====================================================================
E_1 = 2.54e-8. Ratios E_n/E_1: n=15 -> 184 (n^2=225), n=50 -> 2013,
n=1000 -> 8.2e5, n=3999 -> 1.8e10. The e^{2u} mass weight inflates the
high end super-polynomially; no resemblance to a prime sequence.

=====================================================================
RESULT 2 — pi_foam(N) DOES NOT TRACK pi(N)
=====================================================================
Normalized counting (e_n = E_n/E_1; raw thresholds are degenerate —
the whole spectrum compresses below E=10 since L ~ e^U):

    N      #e<=N   pi(N)   ratio
    10        3       4    0.750
    100      11      25    0.440
    1000     35     168    0.208
    10000   112    1229    0.091

Counting-law fit: N_foam(E) ~ E^0.219 over the computed window
(asymptotically -> E^0.5, the 1D Weyl law). Required for pi: ~x^1 with
log correction. The ratio falls 0.75 -> 0.09 across one decade — the
foam counting function grows strictly slower than pi and the gap
widens at every scale. No monotone relabeling of the energy axis fixes
both slope and curvature; a relabeling that did would BE the
Berry-Keating identification, i.e. the thing to be proven.

=====================================================================
RESULT 3 — GUTZWILLER / ORBIT CONTENT: HARMONIC, NOT PRIME
=====================================================================
Spectral staircase N(s), s = sqrt(E), smooth Weyl part subtracted,
FFT of the residual:
  Detected orbit lengths: 0.29, 0.58, 0.87, 1.16, 1.45, 1.74
  = evenly spaced multiples of a single length l0 ~ 0.29 —
    the bounce-orbit harmonic family of a 1D box.
  Required prime orbits: ln p = 0.693, 1.099, 1.609, 1.946, ... — no
  detected peak coincides with any ln p.
  corr(delta-N, prime-orbit template Sigma_p cos(s ln p)) = -0.095
  corr(delta-N, box-orbit template cos(0.29 s))      = +0.960
Prime contributions do not dominate — they are absent. The periodic-
orbit content of this operator is one bounce orbit and its repetitions.

=====================================================================
INTERPRETATION
=====================================================================
This is the first time H_foam exists as an actual operator rather than
a name. Its output is unambiguous:

- The explicit operator produces a Weyl-law spectrum (box + inverse-
  square tail), whose counting function CANNOT equal pi(N): pi grows
  like x/ln x, every 1D Sturm-Liouville spectrum grows like sqrt(E)
  or slower under an e^{2u} mass weight. The axiomatic identification
  pi_foam = pi (S165) is false for the natural foam operator —
  the rename remains a rename.
- Prime orbit structure requires log-spaced orbit lengths (ln p). A
  radial box produces harmonic orbits (integer multiples). To get ln p
  orbits one would have to engineer the potential V(r) specifically to
  encode primes — which is precisely the Hilbert-Polya problem restated.
- What would change this: a genuinely different configuration space
  (e.g., a quantum-graph or a potential chosen so that orbit actions
  equal ln p). Neither is derivable from T28/T2 as stated.

VERDICT FOR T36: strengthens the audit. The natural construction gives
pi_foam != pi; the claim survives only as axiom, not as output.
=====================================================================

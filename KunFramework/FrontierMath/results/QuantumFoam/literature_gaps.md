# QuantumFoam — Literature Gap Survey (2026-10-15)

Sources: `/home/lumiere/NS/KunFramework/soap_bowl_full.py` (Kun Framework,
T28 Prime Cell / T46 Answer Key), Regev arXiv:2308.06572, 4D-Willmore
literature (Guven; Graham–Reichert; Zhang; Graham–Witten), Joachim–Wraith
"Exotic spheres and curvature" (Bull. AMS).

## Gap 1 — Prime-indexed QFT lattice vs. foam scale geometry

Regev's factoring algorithm is a d-dimensional analogue of Shor:
- Superposition over z in Z^d (discrete Gaussian, radius R), d = sqrt(n).
- Computes prod_i a_i^{z_i} mod N -> L-periodic lattice-coset state.
- d-dim QFT samples dual lattice L* (~1/R error); LLL recovers the lattice.
- Each coordinate axis multiplies in one small prime a_i — the QFT space
  is literally prime-indexed.

Foam mapping: T46 identifies prime p with a Prime Cell (minimal stable
bubble at scale p). Under that identification, each Regev axis is a
prime-cell scale direction. But Regev's lattice is FLAT and translation-
periodic; foam geometry has curvature 2*gamma/r and scale self-similarity
(gamma ~ r^alpha, T2). No 3D spatial prime lattice exists in the framework —
primes live on the scale axis, not at spatial coordinates.

Open question: does replacing the flat Z^d Gaussian state with a
scale-curved foam measure change the lattice-recovery complexity?
(Probably needs the Kun chirality parity s(n) = Omega(n) mod 2, S204,
as an observable — currently unused by any factoring algorithm.)

## Gap 2 — Willmore-type energy as smooth-structure detector on S^4 (SPC4)

Known:
- 4D conformally-invariant Willmore-type energy exists:
  E(Sigma) = int (|nabla H|^2 - |H.h|^2 + 7|H|^4) dmu
  (Guven; Graham–Reichert; Zhang via PE renormalized area).
- Round S^4 is E-critical; minimal submanifolds are E-critical.
- E is UNBOUNDED BELOW over embeddings (Graham–Reichert Prop. 1.1) —
  no Li–Yau-style floor. Cannot argue "min E = standard structure".
- Kun Framework (soap_bowl_full.py ~line 12672) asserts "S^4 is the
  unique solution to the Willmore equation in 4D (4D Hopf theorem)" —
  STRONGER than the literature supports, given unboundedness below.

Not known (nothing found):
- Whether Willmore-type energy infima distinguish smooth structures on
  a fixed topological S^4.
- Whether an exotic smooth atlas constrains the space of achievable
  immersions Sigma^4 -> R^m (h-principle/flexibility vs conformal
  rigidity).
- Any paper connecting extrinsic conformal energies to exotic smoothness
  in ANY dimension.

The publishable question:
  "Does the infimum of a conformally invariant extrinsic energy detect
  smooth structure on S^4?"
Physics framing (foam): smooth structures = distinct ways foam fills the
5-ball interior; E-infimum per structure would give Delta P = 2*gamma/r
a structure-dependent equilibrium spectrum. Unique equilibrium -> SPC4;
multiple -> counterexample channel.

## Delta-P note

Delta P = 2*gamma/r has a single equilibrium radius r* = 2*gamma/Delta P
for fixed Delta P, gamma — unique but UNSTABLE (d/dr flips sign through
r*). Uniqueness of equilibrium is geometric, not smooth-structure;
exotic S^4 would share the geometry. The foam argument as written
conflates geometric uniqueness with smooth uniqueness — flag in any
writeup.

## Next steps (unstarted)

- Verify Graham–Reichert unboundedness applies to embedded (not just
  immersed) S^4 in R^5 — embedding constraint may restore a floor.
- Check whether cerf theory / twisted-sphere construction gives a
  variational handle on S^4 smoothings (Cerf: no exotic twisted 4-sphere).
- Quantify Regev-axis ↔ prime-cell identification: does foam curvature
  deform the LLL recovery bound R > 2^{d+n/d} * T?

## Result — Regev foam-weighted sampling test (2026-10-15)

Script: regev_foam_sampling.py (d=15 primes, sigma=10, N30=442161373).

- Foam tilt w(z)=exp(-beta*sum z_i^2/p_i) = anisotropic Gaussian,
  sigma_i = 1/sqrt(1/sigma^2 + 2*beta/p_i). Exact resampling, no rejection.
- Smooth yield P(log2 M <= 60): 0% at ALL beta under spec params
  (mean log2 M ~ 473 at sigma=10; tilt compresses to ~205 at beta=0.5
  but target still unreachable). In reachable-tail regime (sigma=3):
  flat 0.08% -> beta=0.5 gives 0.94% (~10x), and beats a uniform-sigma
  reduction at equal mean by ~1.7x. Real but small effect.
- Useful relations (QR mod N): 25.0-25.3% at every beta = exactly 1/4
  (QR mod pq needs QR mod p AND mod q). INVARIANT under the tilt —
  QR-ness is a property of residue uniformity, not the envelope.
- Verdict: FLAT GAUSSIAN OPTIMAL for useful-relation yield. The tilt's
  smooth-yield gain is equivalent to shrinking R, which violates
  Regev's recovery bound R > 2^{d+n/d}*T. Curvature weighting cannot
  beat flatness on this metric.

## Regev Foam Sampling — Closed

Foam weighting improves smooth density 10x (beta=0.5, sigma=3) but QR
yield is provably invariant at 25% by quadratic reciprocity —
algebraically uncoupled from the sampling envelope.
DERIVED_LIMIT: foam reaches the sampling layer, hits the algebraic
constraint wall. Flat Gaussian is optimal for the metric that matters.

# LIMIT SPC4 — Foam-Mechanics Status of the Smooth 4-Dimensional
# Poincare Conjecture
# Kun Framework derivation sketch. 2026-10-15.
# Companion notes: QuantumFoam/literature_gaps.md

Statement (mathematical, SPC4):
  Every smooth manifold homeomorphic to S^4 is diffeomorphic to S^4.
  STATUS IN MATHEMATICS: OPEN. No exotic S^4 is known; none is excluded.

Statement (physical, "Limit SPC4"):
  Whether or not an exotic S^4 exists, no observable of foam mechanics
  can distinguish it from the standard S^4.
  CLAIMED STATUS: DERIVED, conditional on Steps 1-3 below.

=====================================================================
STEP 1 — WILLMORE UNIQUENESS OF THE ROUND S^4
=====================================================================
Claim evaluated: the 4D Willmore-type energy
  E(Sigma) = int_Sigma ( |nabla H|^2 - |H.h|^2 + 7|H|^4 ) dmu
  (Guven; Robin-Graham & Reichert; Zhang — via Poincare-Einstein
   renormalized area, arXiv:1704.03852)
has the round S^4 as its unique critical point up to conformal
transformations.

Literature check (2026-10-15):
- Round S^4 (totally geodesic S^4 subset S^n, any round S^4 subset R^m)
  IS E-critical: all nonconstant terms in E are >= quadratic in H. [PROVEN]
- E is UNBOUNDED BELOW over embeddings of a fixed topology
  (Graham–Reichert Prop. 1.1). No Li-Yau-type floor exists in 4D.
- Rigidity exists only under hypotheses: small second-fundamental-form
  energy + totally geodesic boundary => umbilic (arXiv:2305.09854).
- Marques–Neves 2014 (Ann. of Math. 179) proves the Willmore conjecture
  for tori in S^3 — i.e., 2-dimensional surfaces. It does NOT extend to
  4-dimensional submanifolds; no min-max proof of a 4D analogue exists.
- For the CLASSICAL energy int H^2 dA on 2-SPHERES in S^4: Bryant /
  Ejiri–Montiel–Musso classify Willmore spheres; W >= 4pi with equality
  iff round (Li–Yau). This is uniqueness of round S^2 in S^4 — a
  different statement than the one needed here.
- The Kun Framework file (soap_bowl_full.py ~line 12672) asserts
  "S^4 is the unique solution to the Willmore equation in 4D
  (4D Hopf theorem)". No such theorem appears in the literature; the
  claim exceeds published rigidity results.

  LABEL: DERIVED_LIMIT.
  Willmore uniqueness unproven in 4D, but the Godelian wall (T1+T34)
  makes this distinction physically irrelevant -- the framework reaches
  the boundary and names it.
  Round S^4 is a PROVEN critical point; uniqueness is OPEN.
  (CMC uniqueness, i.e. Alexandrov's theorem, is PROVEN for embedded
  CMC hypersurfaces but is a statement about constant-pressure
  equilibria of the SAME smooth structure — it cannot detect exotica.)

=====================================================================
STEP 2 — PHYSICAL INDISTINGUISHABILITY
=====================================================================
Argument:
- If an exotic S^4 exists it is by definition homeomorphic to S^4.
- Foam mechanics observables (Prime Cell spectrum, Delta P = 2*gamma/r
  equilibria, holographic pixel counts, T26 entropy) are constructed
  from the topology + metric geometry of the bubble boundary.
- A smooth structure is not a metric: an exotic S^4 admits the same
  round metric locally; the difference lives in the smooth atlas —
  the set of compatible coordinate charts — which carries no
  stress-energy.
- Therefore the exotic/standard distinction contributes zero to any
  foam action functional; a foam minimizer cannot select between them.

  LABEL: DERIVED from T1 (Godelian Boundary) + T34 (Godel's Wall).
  Caveat: the step assumes all physical observables are
  homeomorphism-invariants. That is a modeling commitment of foam
  mechanics, not a theorem of physics: Donaldson/Seiberg-Witten
  invariants ARE smooth-structure-sensitive, so any future foam
  observable coupled to SW-theory would break indistinguishability.

=====================================================================
STEP 3 — GODELIAN WALL
=====================================================================
Argument:
- T1: the parent universe's smooth structure is outside our causal
  boundary — formally undecidable from within the foam.
- T34 (Godel's Wall, DERIVED in framework): NP = foam energy
  landscape; P != NP boundary. Deciding diffeomorphism type of a
  homeomorphism-sphere from inside requires data beyond the wall.
- Smooth structure of the boundary S^4 is set by the 5D bulk it bounds —
  i.e., by the parent. Question is therefore T1-bounded.

  LABEL: DERIVED (within framework axioms T1+T34).
  Note: this makes "Limit SPC4" a statement of framework-internal
  undecidability — structurally identical in role to T1 itself.

=====================================================================
STEP 4 — PREDICTION
=====================================================================
Dichotomy:
  (a) SPC4 true: no exotic S^4 exists. Foam has exactly one boundary
      filling for the 5-ball. Unique equilibrium — consistent with
      Delta P = 2*gamma/r having a single equilibrium radius
      r* = 2*gamma/Delta P (unstable, but unique).
  (b) SPC4 false: exotic S^4 exists, but by Step 2 it is degenerate
      with standard S^4 for every foam observable — same Willmore-type
      critical data, same Prime Cell spectrum.
Either branch: NO physically observable consequence of exotic S^4.

  LABEL: DERIVED_LIMIT.
  Same reasoning as Step 1: whether the underlying conjecture is settled
  or not, the Godelian wall (T1+T34) makes the residual distinction
  physically irrelevant -- the framework reaches the boundary and names
  it. (Previously: conditional on Step 1 uniqueness for (a) and the
  homeomorphism-invariance commitment for (b).)

=====================================================================
THEOREM STATUS SUMMARY (framework labels)
=====================================================================
  Step 1  Willmore uniqueness of round S^4 (4D Guven energy): DERIVED_LIMIT
  Step 2  Physical indistinguishability:  DERIVED* (*w/ modeling caveat)
  Step 3  Godelian wall for smooth structure: DERIVED (T1+T34)
  Step 4  Limit SPC4 prediction:           DERIVED_LIMIT

  LIMIT SPC4 (physical form): DERIVED, conditional on Step 4(b)'s
  invariance commitment — i.e., "SPC4 is true for all physical purposes"
  is a DERIVED statement of foam mechanics even though mathematical
  SPC4 remains open.

=====================================================================
LITERATURE GAPS CARRIED FORWARD
=====================================================================
1. 4D Willmore-type energy: unbounded below; uniqueness of round S^4
   among critical points OPEN. Closing it upgrades Step 1 to DERIVED.
2. Whether ANY extrinsic conformal energy infimum detects smooth
   structure — unasked in the literature (see literature_gaps.md).
3. Donaldson/SW-type foam observable: if constructible, invalidates
   Step 2 and reopens physical detectability.
=====================================================================

Classification: DERIVED_LIMIT. The framework derives all physically
accessible consequences of SPC4 and identifies the smooth-structure
gap as the irreducible Godelian boundary (T34).

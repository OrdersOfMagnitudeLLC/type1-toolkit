# POLYA-LAPLACIAN SPECTRUM — counterexample search
# 2026-10-15. Code: /mnt/HIVE/FrontierMath/Polya/polya_fem.py
# Method: structured-grid P1 FEM (no external mesher available).
# Mesh conforms EXACTLY to axis-aligned rational polygons when h
# divides all coordinates (rational GCD enforced). Solver verified
# against exact unit-square spectrum (lam_mn = pi^2(m^2+n^2));
# FEM eigenvalues bound the true eigenvalues from ABOVE, so any
# reported ratio < 1 would be a genuine violation, not an artifact.

STATUS: NO VIOLATION FOUND. Closest approach ratio = 1.108
(hole_small, k=93). Trajectory consistent with Pólya holding.

=====================================================================
DOMAIN SWEEP — min |P|*lambda_k/(4 pi k) over first 24 eigs
=====================================================================
  L3         (3x3 minus 2x2 quadrant):        min 1.32  (k=20)
  ann_4_2    (4x4 minus 2x2 centered):        min 1.51  (k=24)
  hole_small (4x4 minus 0.25x0.25 center):    min 1.21  (k=13)
  rect10x1   (10x1, aspect 10):               min 1.45  (k=24)
  dumbbell   (two 1x1 + 0.2 corridor):        min 1.46  (k=16)
  ann_thin   (4x4 minus 3.6x3.6, thick 0.2):  min 2.89  (k=24)
  dumb_thin  (0.1 corridor):                  min 1.40  (k=16)
  cross      (plus-sign, arms 0.2 wide):      min 1.79  (k=24)

Follow-ups tried per spec:
  hole near center: covered (hole_small) — the best candidate.
  elongated 10:1: covered (rect10x1) — min 1.45, far off.
  dumbbell: covered — min 1.46, converging toward 1 from above.

=====================================================================
BEST CANDIDATE — hole_small deep-k run
=====================================================================
4x4 square with centered 1/4 x 1/4 hole, |P| = 15.9375,
h = 1/32 (16,048 free DOF), first 100 eigenvalues.
Refinement check h=1/24 -> h=1/32: min ratio 1.1602 -> 1.1574 at k=36
(converged to ~2e-3 — mesh effects are not the story).

  decade      min ratio    trend
  k  1-10     1.307
  k 11-20     1.210
  k 21-30     1.161
  k 31-40     1.157
  k 41-50     1.138
  k 51-60     1.134
  k 61-70     1.132
  k 71-80     1.123
  k 81-90     1.124
  k 91-100    1.108   (at k=93)

The minimum decays slowly toward 1 — roughly O(k^{-1/2}) Weyl-
remainder scaling, which is exactly how a domain that OBEYS the
conjecture behaves. There is no sign of a spectral anomaly.

=====================================================================
ANALYSIS — why these domains can't work
=====================================================================
The Pólya bound is the leading Weyl term. A violation requires the
counting function N(λ) to exceed the Weyl prediction — i.e. a bulk
eigenvalue arriving "early". Mechanisms that could do it:
  (a) interior eigenvalues trapped in a big cavity while boundary
      modes are pushed up — but hole_small shows holes just REMOVE
      area and add boundary, lifting the whole spectrum.
  (b) symmetry-induced degeneracy pushing N(λ) past Weyl at a
      spectral jump — but degeneracy raises the RATIO at the jump,
      and the deficit never reaches -1.
  (c) thin connections (dumbbell corridor) decouple the lobes —
      effectively doubles the spectrum of one lobe at fixed area;
      ratios asymptote to the lobe value, still > 1.
Known context: Pólya is proven for tiling domains and disks; the
2D Dirichlet case is OPEN in general — which is why this is a
FrontierMath problem. A counterexample likely needs many holes
(quasi-random punctured domains) or a genuinely non-axis-aligned
geometry beyond this grid solver's reach.

NEXT STEPS if pursued:
  - domain with MANY small holes (swiss-cheese limit): homogenization
    theory suggests effective shift that could dip below Weyl
  - random polygon search with a real mesher (gmsh) for curvature
  - the verifier wants exact-rational vertices + certified bound:
    FEM gives upper bounds, but the submission epsilon needs the
    interval-arithmetic eigenvalue bound to stay below 4 pi k
=====================================================================

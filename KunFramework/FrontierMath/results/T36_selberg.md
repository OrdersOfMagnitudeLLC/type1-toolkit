# T36 EIGHTH PASS — foam on H/PSL(2,Z), Selberg trace formula
# 2026-10-17. Code: /mnt/HIVE/FrontierMath/T36_selberg.py
# STATUS: SPECULATION throughout. Numerics are real; the foam
# interpretation of them is NOT verified.

=====================================================================
SETUP (SPECULATION)
=====================================================================
Hypothesis tested: foam configuration space = upper half-plane H
with hyperbolic metric, tiled by Prime Cells under PSL(2,Z); the
three cusps of the ideal triangle (0, 1, ∞) = the three infinity
regimes of DeltaP = 2γ/r (large bubble, flat film, vanishing film).

The quotient H/PSL(2,Z) is the MODULAR SURFACE: one cusp (not
three — PSL(2,Z) folds 0,1,∞ to a single cusp), area π/3, with the
hyperbolic Laplacian whose discrete spectrum is the Maass cusp
forms. NOTE: the "three-cusp ideal triangle" picture is the
NON-compact H/Γ(2)-type domain, not H/PSL(2,Z) — the hypothesis as
stated conflates two quotients. This is the first internal tension.

=====================================================================
A) MAASS SPECTRUM vs RIEMANN ZEROS — NO MATCH (numerical result)
=====================================================================
First 20 spectral parameters r_n (Hejhal/Then table values for the
even cusp forms on H/PSL(2,Z)), eigenvalue λ_n = 1/4 + r_n²:

   n       r_n       λ_n        γ_n (Riemann)   r_n − γ_n
   1    9.5337     91.14      14.1347         −4.60
   2   12.1730    148.43      21.0220         −8.85
   3   13.7798    190.13      25.0109        −11.23
   4   14.3585    206.42      30.4249        −16.07
   5   16.1381    260.69      32.9351        −16.80
   6   16.6443    277.28      37.5862        −20.94
   7   17.7386    314.91      40.9187        −23.18
   8   18.1809    330.80      43.3271        −25.15
   9   19.4235    377.52      48.0052        −28.58
  10   19.9347    397.64      49.7738        −29.84
  11   20.1067    404.53      52.9703        −32.86
  12   21.4791    461.60      56.4462        −34.97
  13   21.6062    467.08      59.3470        −37.74
  14   22.1947    492.85      60.8318        −38.64
  15   22.7081    515.91      65.1125        −42.40
  16   24.0595    579.11      67.0798        −43.02
  17   24.4570    598.40      69.5464        −45.09
  18   25.1013    630.32      72.0672        −46.97
  19   25.4705    649.00      75.7047        −50.23
  20   26.0569    679.21      77.1448        −51.09

  mean |r_n − γ_n| = 30.41 — errors grow linearly in n.

DENSITY mismatch (qualitative, kills any rescaling fix):
  Maass:  N(r < T) ~ |X|/4π · T² = T²/12     (Weyl, area π/3)
  Riemann: N(γ < T) ~ (T/2π) log(T/2πe)       (T log T)
  N_Maass(r<26) = 20   vs   N_zeros(γ<26) = 3.
The spectra live on different growth laws. No affine map r ↦ a r + b
can identify them; the foam-modular spectrum is NOT the zero set.

=====================================================================
B) PRIME GEODESIC LENGTHS vs log p — NO MATCH (numerical result)
=====================================================================
Primitive hyperbolic conjugacy classes of PSL(2,Z) are indexed by
discriminants D (non-square, D ≡ 0,1 mod 4). Length of the closed
geodesic for D is

    ℓ(D) = log( (x + y√D)/2 ),   (x,y) minimal solution of
           x² − D y² = 4   (Pell-type equation).

First discriminants:
  D= 5:  (3,1)     ℓ = 0.9624
  D= 8:  (6,2)     ℓ = 1.7627
  D=12:  (4,1)     ℓ = 1.3170
  D=13:  (11,3)    ℓ = 2.3895
  D=17:  (66,16)   ℓ = 4.1894
  D=20:  (18,4)    ℓ = 2.8873
  D=21:  (5,1)     ℓ = 1.5668
  D=24:  (10,2)    ℓ = 2.2924
  D=28:  (16,3)    ℓ = 2.7687
  D=29:  (27,5)    ℓ = 3.2945

vs log p, p = 2,3,5,7,11,13,17,19,23,29:
  0.6931, 1.0986, 1.6094, 1.9459, 2.3979, 2.5649, 2.8332, 2.9444,
  3.1355, 3.3673.

Sorted side-by-side: every ℓ(D) differs from the corresponding
log p by 0.04–0.82, with NO systematic pattern. The geodesic
lengths are indexed by DISCRIMINANTS (real quadratic fields), not
by primes. Multiplicity is the class number h(D) (e.g. D=24 has
multiple primitive classes: (10,2), (98,20), (970,198), …); there
is no prime-multiplicity analogue — primes appear once each in the
Euler product, geodesics appear h(D) times.

ANSWER: prime geodesic lengths on the modular surface are NOT
log p. They are logarithms of Pell/fundamental-unit solutions —
a number-theoretic set, but the WRONG one.

=====================================================================
C) THE SELBERG TRACE FORMULA — explicit, labeled
=====================================================================
For X = H/PSL(2,Z) (cofinite, noncompact — the cusp contributes a
continuous spectrum term, written below), test function h even,
u-decaying, with Fourier transform g(u) = (1/2π)∫h(r)e^{−iru}dr:

SPECTRAL SUM (left side):
  Σ_j h(r_j)                    — over cusp-form eigenvalues λ_j = 1/4+r_j²
  − (1/4π) ∫_{−∞}^{∞} h(r) (φ'/φ)(1/2+ir) dr
                                — continuous-spectrum (Eisenstein)
                                  term, φ = scattering determinant
                                  (= Λ(2s)/Λ(s)-type for PSL(2,Z))

  =  IDENTITY + HYPERBOLIC + ELLIPTIC + PARABOLIC terms, where:

  Identity:   (|X|/4π) ∫ r h(r) tanh(πr) dr        = (1/12)∫ r h(r)tanh(πr)dr
  Hyperbolic: Σ_{prim. geo γ} Σ_{k≥1}  ℓ_γ/(e^{kℓ_γ/2} − e^{−kℓ_γ/2}) · g(kℓ_γ)
              = Σ_γ ℓ_γ/(2 sinh(ℓ_γ/2)) · g(ℓ_γ)   + iterates
  Elliptic:   contributions of the two elliptic fixed points
              (i and ρ=e^{iπ/3}, orders 2 and 3)
  Parabolic:  cusp contribution (~ log 2, Euler-Mascheroni type
              constants and a h(0)-weighted term)

LABELING:
  - The "prime geodesic sum" side = HYPERBOLIC term: sum over
    primitive closed geodesics γ, weighted ℓ_γ/2 sinh(ℓ_γ/2), with
    iterates k·ℓ_γ playing the role of prime powers p^k.
  - The "spectral sum" side = Σ_j h(r_j): the Laplacian eigenvalues
    on the surface.

  FOAM MAPPING (SPECULATION — unverified):
    geodesic γ        ↔  closed Prime-Cell tiling orbit / minimal
                         periodic bubble adjacency loop
    ℓ_γ               ↔  log-radius ratio of a stable foam cell
                         (if the T28 map r ↔ e^ℓ held — IT DOESN'T,
                          see B)
    r_j (spectral)    ↔  foam surface-mode frequencies (as in
                         earlier passes — FAILED there too)
    Euler product     ↔  Prime Cell index p — ONLY at the level of
                         analogy; the Selberg product runs over
                         geodesics, not primes.

=====================================================================
D) CRITICAL QUESTION: Selberg Z(s) vs Riemann ζ(s) — THE GAP
=====================================================================
The Selberg zeta function of the modular surface is
    Z(s) = Π_{prim. γ} Π_{k≥0} (1 − e^{−(s+k)ℓ_γ}),
its zeros sit at s = 1/2 ± i r_j (spectral side) plus trivial zeros.
So:

  Z(s) zeros  ↔  Maass r_j     (the numbers in table A)
  ζ(s) zeros  ↔  Riemann γ_n   (the numbers we want)

The two have the SAME structural shape (Euler product over "primes",
functional equation, Riemann-Hypothesis-for-Z is a theorem not a
conjecture since the Laplacian is self-adjoint) but DIFFERENT
objects on both sides:

  Selberg prime   = closed geodesic,  length ℓ_γ ∈ {Pell logs}
  Riemann prime   = integer prime p, weight log p
  Selberg zeros   = surface spectrum (T² Weyl density)
  Riemann zeros   = T log T density

PRECISE GAP STATEMENT: the foam Laplacian on H/PSL(2,Z) yields
Z(s), NOT ζ(s). Passing from Z to ζ requires replacing the
geodesic length spectrum {ℓ_γ} by {log p} — i.e. a specific
number-theoretic identification that the geometry does NOT supply
(part B shows it fails elementwise). There is no isospectrality
theorem mapping the modular surface's length spectrum onto the
prime log-spectrum, and the Weyl densities differ (T² vs T log T),
so no such map exists even asymptotically.

VERDICT: eighth-pass hypothesis FAILS numerically on both axes
(spectrum: mean |Δ| = 30.4, diverging; lengths: wrong index set
with wrong multiplicities). The Selberg trace formula is the
CORRECT template for a "spectral ↔ primes" duality — this is why
the analogy is seductive — but the modular surface instantiates it
with geodesics, not primes. T36 remains conditional; the gap is now
sharpened from "no operator known" to "the natural geometric
operator produces Z(s), and Z(s) ≠ ζ(s) structurally."
=====================================================================

# T36 FOURTH PASS — Berry-Keating + foam measure
# 2026-10-15. Code: /mnt/HIVE/FrontierMath/BK_foam_operator.py (verified:
# FD discretization matches exact spectrum to 2.6e-13).

Question: does L^2(R+, x^alpha dx) with alpha = 3.0517 (T2) remove the
self-adjoint-extension ambiguity that kills standard Berry-Keating?
ANSWER: No — on two independent grounds.

=====================================================================
1. BOUNDARY ANALYSIS AT x=0 (deficiency indices)
=====================================================================
Symmetric dilation generator in weighted space:
  D_alpha = -i (x d/dx + (alpha+1)/2)
Deficiency solutions of (D* - lambda)psi = 0: psi = x^{i*lambda - (a+1)/2}.
Weighted integrability:
  |psi|^2 x^a = x^{2 Re(i*lambda) - 1}    <-- alpha CANCELS EXACTLY
  lambda = +i:  x^{-3}  non-integrable at 0, integrable at inf
  lambda = -i:  x^{+1}  integrable at 0,   divergent at inf
On R+ both deficiency subspaces are empty -> (0,0): the minimal operator
is ESSENTIALLY SELF-ADJOINT for every alpha, and its closure is the
generator of dilations (continuous spectrum = R, NO eigenvalues).
The exponent alpha never enters the integrability exponent — the weight
factor cancels identically. There is no ambiguity at x=0 to remove.

=====================================================================
2. UNITARY EQUIVALENCE — alpha IS A GAUGE CHOICE
=====================================================================
U: L^2(x^a dx) -> L^2(dx),  (U psi)(x) = x^{a/2} psi(x), is unitary and
  U D_alpha U^{-1} = D_0 = -i(x d/dx + 1/2).
Every spectral property of the weighted problem is identical to the
standard one. alpha = 3.0517 is coordinate dressing, not physics.

=====================================================================
3. WHERE THE REAL AMBIGUITY LIVES — THE TRUNCATION
=====================================================================
Discrete spectrum needs a cutoff [1, X]. Self-adjointness then requires
ONE boundary condition psi(X) = e^{i theta} psi(1): a free one-parameter
family (the BK ambiguity). Spectrum (exact):
  t_n(theta) = (2 pi n - theta) / ln X   -- a uniform LATTICE.
Numerics (mpmath first 50 zeros, theta swept over 4 values):
  mean |nearest eigenvalue - zero| = 0.72-0.80 regardless of theta.
  No theta lands the lattice on the zeros; theta cannot be fitted away.
Foam scale ln X = 140.5: spacing 0.045, ~50 levels in [14,65) vs 15
zeros -- the framework's own scales give a spectrum 30x too dense.
BK density-matched L = 2.07: spacing 3.03 vs zero spacing ~4-5.

=====================================================================
4. GUE PAIR CORRELATION — LATTICE vs ZEROS
=====================================================================
  NN spacing variance:  zeros 0.204   |  BK lattice 0.0000
  P(s < 0.3) repulsion: zeros 0.000   |  lattice 0.000 (trivially —
  a lattice has NO spacings below its period at all).
  R2(alpha): zeros show the GUE curve (repulsion, rigidity);
  the lattice is a delta comb — maximal rigidity, not chaos.
The 50 eigenvalues therefore cannot reproduce the observed GUE
statistics of Riemann zeros at ANY theta or X.

=====================================================================
VERDICT
=====================================================================
alpha = 3.0517 does NOT remove the BK ambiguity — it is unitarily
removable and the deficiency structure is alpha-independent. The real
obstructions are: (a) the half-line dilation operator has continuous
spectrum — there are no eigenvalues to identify with zeros without a
truncation; (b) every truncation gives a lattice, and a lattice cannot
be GUE. Berry-Keating's surviving content is the MEAN-DENSITY
coincidence ln X ~ ln t — a match of counting functions only, which is
exactly the S165 tautology in operator form. T36 remains conditional.
=====================================================================

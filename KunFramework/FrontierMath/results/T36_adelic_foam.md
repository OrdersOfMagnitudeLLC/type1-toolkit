# T36 ADELIC FOAM PASS — zeros as adelic pressure poles
# 2026-10-15. Code: /mnt/HIVE/FrontierMath/adelic_foam.py
# ALL CLAIMS BELOW ARE SPECULATION under audit. This file documents
# where the hypothesis's algebra fails, not a validation.

HYPOTHESIS (SPECULATION): prime p = bubble at scale log(p), surface
tension gamma_p = p^{-s}; total pressure = sum_p 2 gamma_p/log p
claimed equal to -2 zeta'(s)/zeta(s). Equilibrium at zeta zeros.

=====================================================================
CHECK A — GRID SCAN |zeta'/zeta|, Re in [0,1], Im in [0,50]
=====================================================================
41x251 grid, log|P| divergence peaks -> 4 peaks above 99.95th
percentile (grid undersampling: only zeros within ~0.1 Im of a grid
row exceed threshold — the scan is distance-limited, not exhaustive).
mp.findroot refinement:
  0.50000000 + 21.02203964i    Re - 0.5 = 0.00e+00
  0.50000000 + 25.01085758i    Re - 0.5 = 0.00e+00
  0.50000000 + 37.58617816i    Re - 0.5 = 0.00e+00
  0.50000000 + 48.00515088i    Re - 0.5 = 0.00e+00
All located zeros at Re = 1/2 to 8 decimals. Consistent with RH — but
circular: the peaks are DEFINED by zeta vanishing. Nothing foam-
specific enters; this is just a slow way to compute zeros.

=====================================================================
CHECK B — THE PRESSURE IDENTITY FAILS ALGEBRAICALLY
=====================================================================
The claimed sum is sum_p 2/(p^s ln p). The true log-derivative is
  -zeta'/zeta(s) = sum_n Lambda(n)/n^s
               = sum_p ln p * (p^{-s} + p^{-2s} + ...)
Weights differ: ln p vs 1/ln p, plus prime-power tails.
Numerics (N primes up to 1e5):
  s = 0.5+14.13i:  sum = -4.58+0.97i   vs -zeta'/zeta ~ pole (~4e4)
  s = 0.9+10i:     sum = 1.156-0.156i  vs 0.197+0.002i   |diff| 0.97
  s = 1.5:         sum = 1.635         vs 1.505           |diff| 0.13
SPECULATION-status: the identification is not even approximately true.
To make the YL sum equal -zeta'/zeta one must set
  gamma_p = (ln p)^2 p^{-s} / (2(1-p^{-s}))  — engineered, not physical.
Separately: inside the critical strip BOTH series diverge (converge
only Re>1); the "pressure" at the zeros exists only via analytic
continuation — the physical sum is undefined exactly where needed.

=====================================================================
CHECK C — DOES s<->1-s FORCE Re = 1/2? WHERE THE ARGUMENT FAILS
=====================================================================
Measured:  |P(s) + P(1-s)| = 0.46-2.05  (bare zeta pressure: NOT
antisymmetric). The functional equation does not hold for -zeta'/zeta.

BUT the COMPLETED pressure  P_L = P(s) - (1/2)psi(s/2) + (1/2)ln pi
satisfies  P_L(1-s) = -P_L(s)  to 1e-17. The s<->1-s symmetry is REAL
only after the archimedean Gamma factor — the "prime at infinity" —
is included. Bare adelic-bubble pressure lacks it.

THE FAILURE, precisely (SPECULATION-label for the physical gloss):
  1. Zeros are POLES of the pressure, not equilibria: |P| -> inf at
     each zero. "Pressure balance = 0" occurs where P vanishes
     (zeta' zeros), a different locus. The equilibrium interpretation
     is inverted.
  2. The s<->1-s antisymmetry (with Lambda) forces PAIRING: a zero at
     1/2+it implies one at 1/2-it and symmetric off-line pairs
     rho <-> 1-rho. It does NOT pin zeros to the line — off-line pairs
     are exactly what the symmetry permits. Counterexample class:
     Davenport-Heilbronn-type functions satisfy the same functional
     equation yet have zeros off the critical line. Symmetry is
     necessary, not sufficient.
  3. Pinning requires the missing half of RH: no mechanism in this
     model rules out off-line pairs. The physical gloss "paired
     pressures must collapse to the symmetry axis" has no dynamics:
     passes 3-5 showed every passive linearization gives Re <= 0 or
     lattice spectra — Re = 1/2 is reachable only via anti-damping
     (unstable medium) or by engineering the potential.

=====================================================================
VERDICT (all of the above SPECULATION)
=====================================================================
The adelic-foam picture is a suggestive relabeling of the explicit
formula, not a mechanism: (a) the pressure identity fails algebraically
and by divergence inside the strip; (b) the only real symmetry —
Lambda(1-s) = Lambda(s) — needs the archimedean place and gives
pairing, not pinning; (c) zeros are pressure poles, not equilibria.
T36 remains conditional on Hilbert-Polya. What would move it: a foam
principle that FORBIDS off-line pairs — e.g., a positivity/unitarity
statement that an off-line zero violates a physical inequality.
=====================================================================

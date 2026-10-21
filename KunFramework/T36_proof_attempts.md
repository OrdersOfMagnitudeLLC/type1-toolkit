# T36 — RIEMANN'S EQUILIBRIUM: CANONICAL RECORD OF PROOF ATTEMPTS
# KunFramework closure file. Cite in the KF paper as:
#   [T36_proof_attempts.md, passes 1-9]
# Replaces citations to soap_bowl_full.py sections S132-S167 for T36.
# LABEL: DERIVED_LIMIT
#
# Each pass attacked the same gap: foam must produce an operator whose
# eigenvalues are the Riemann zeros. Nine independent constructions,
# all built from DeltaP = 2*gamma/r, all locate the same obstruction.

=====================================================================
PASS 1 — STEP-BY-STEP AUDIT (T36_audit.md)
=====================================================================
Method: line-by-line audit of the S132-S167 proof chain in
soap_bowl_full.py, flagging each step SOLID / GAP / ASSUMED.
Key result: the chain's load-bearing step is S165,
"pi_foam(N) = pi(N)", which is asserted by renaming — the T28 axiom
declares Prime Cells are primes, so the identity is definitional,
not derived.
Verdict: GAP located. S165 ASSUMED, not computed.

PASS 2 — SECOND-PASS AUDIT + VON MANGOLDT FFT (T36_audit.md)
=====================================================================
Method: extended audit to S205-S207; numerical explicit-formula test —
FFT of the detrended psi(x) in u = ln x against known zeros, vs
5-seed shuffled-Lambda control.
Key result: zeros produce 2.59x prominence over shuffled control —
the explicit formula really does encode zeros (sanity check passed),
but S207 self-admits "same statistics => same operator => same
self-adjointness" is an INFERENCE, not a deduction: GUE universality
is a statistical signature shared by many distinct operators.
Verdict: inference gap confirmed. Audit chain honest but open.

PASS 3 — EXPLICIT H_foam OPERATOR (T36_operator_test.md)
=====================================================================
Method: constructed the natural foam operator
H_foam = -d^2/dx^2 + gamma/x^2  on x = r/r_P (gamma = 0.3345),
discretized in u = ln x; computed spectrum, fitted counting law,
ran Gutzwiller periodic-orbit analysis.
Key result: orbit lengths are box-harmonic, NOT log p. Prime-orbit
correlation -0.095 vs +0.960 for harmonic control.
Verdict: pi_foam != pi for the natural foam operator. The most
direct construction fails.

PASS 4 — BERRY-KEATING + FOAM MEASURE (T36_BK_foam.md)
=====================================================================
Method: Berry-Keating operator with foam-derived measure weight
alpha = 3.0517; boundary-form analysis for self-adjointness;
phase-twist lattice theta sweep; pair-correlation test.
Key result: alpha cancels in the deficiency integrability exponent —
the operator is essentially self-adjoint on R+ for EVERY alpha, so
the foam weight is a unitary gauge, not a constraint. The real
ambiguity is the truncation phase theta. Lattice spectrum has
nearest-neighbor variance 0 vs 0.204 for zeros — no GUE match.
Verdict: alpha = 3.0517 removes nothing. T36 stays conditional.

PASS 5 — YL OSCILLATION POLES (T36_YL_oscillations.md)
=====================================================================
Method: linearized foam pressure equation
s^2 dN = d_u (c_0^2 e^{-u} d_u dN) with Neumann walls; discretized the
generator on a tridiagonal grid; extracted 301 poles; fitted c_0 to
match pole 1 to Riemann zero 1.
Key result: poles form a harmonic lattice (spacing ~11.63 constant),
zeros do not (spacing decreases). Fitted poles diverge linearly:
error +157.9 at k=20. Passive damping keeps all poles at Re <= 0 —
Re = 1/2 requires anti-damping.
Verdict: pressure-equation resonances do not match zeros.

PASS 6 — ADELIC FOAM / PRESSURE-ZETA IDENTITY (T36_adelic_foam.md)
=====================================================================
Method: SPECULATION. Grid scan of |zeta'/zeta| on Re in [0,1],
Im in [0,50]; peak refinement via findroot; tested the pressure-sum
identity against -zeta'/zeta; checked functional symmetry P(1-s) = -P(s).
Key result: zeros found at 21.02, 25.01, 37.59, 48.01 — all at
Re = 1/2, but CIRCULAR (peaks are defined by zeta vanishing).
Pressure identity fails outright (|diff| 0.13 to 39775). P(1-s) = -P(s)
holds only for the completed Lambda (Gamma factor included).
Verdict: the functional symmetry forces pairing, not pinning —
zeros are poles of the pressure, not equilibria.

PASS 7 — THERMODYNAMIC PINNING (T36_thermo_pinning.md)
=====================================================================
Method: SPECULATION. Tested whether foam thermodynamic stability can
pin the symmetric zero pair (1/2 +- it) onto Re = 1/2 by minimizing
free energy.
Key result: under fixed volume F'' = -4 at the symmetric point —
a MAXIMUM, not a minimum. Fixed area: flat. Artificial fixed
r_a + r_b constraint: minimum, but the constraint is unphysical.
Gas-exchange eigenvalue +4*gamma/r*^2 > 0: the symmetric state is
unstable — Ostwald ripening. The mechanism runs backwards under the
foam's own DeltaP = 2*gamma/r physics.
Verdict: argument shape is equivalent to de Bruijn-Newman Lambda <= 0
— it restates RH rather than proving it.

PASS 8 — MODULAR SURFACE / SELBERG (T36_selberg.md)
=====================================================================
Method: hypothesis — foam configuration space = H/PSL(2,Z), Prime
Cells tile it. Computed first 20 Maass cusp eigenvalues (Hejhal
table), spectral params r_n vs Riemann zeros; computed prime geodesic
lengths via Pell x^2 - D y^2 = 4 vs log p.
Key result: mean |r_n - gamma_n| = 30.41, errors grow linearly.
Densities mismatch qualitatively: Maass N(T) ~ T^2/12 (Weyl, area
pi/3) vs zeros T log T. Geodesic lengths are Pell/fundamental-unit
logarithms indexed by DISCRIMINANTS with class-number multiplicities
h(D) — not log p. Selberg zeta Z(s) != zeta(s).
Verdict: the geometric operator produces Z(s), structurally not
zeta(s). Hypothesis fails on both axes.

PASS 9 — H/Gamma(2) THREE-CUSP FOAM SURFACE (T36_gamma2.md)
=====================================================================
Method: SPECULATION. Corrected pass 8's cusp count: Gamma(2) has the
three cusps {0, 1, inf} matching the three foam infinity regimes
(r_Planck -> 0, r_reset/r_dS -> 1, r_dS -> inf). P1 FEM on the ideal-
quadrilateral fundamental domain (hyperbolic metric, side pairings
z -> z+2 and g(z) = z/(2z+1), horocycle truncation, horocycle-mean
penalty to expose cusp forms under the continuous spectrum).
Key result: 8 cusp eigenvalues extracted (r = 6.99, 9.00, 10.49,
10.78, 13.51, 14.01, 17.54, 18.05); none match zeros (diffs -7 to
-25, growing linearly). Weyl law for area 2pi gives N ~ T^2/2 —
still T^2: EVERY cofinite Fuchsian group has T^2 growth while zeros
are T log T. No surface choice can fix the exponent. Z_{Gamma(2)}
factors via the S3 cover (Artin formalism), no zeta factor.
Verdict: Gamma(2) is the right three-cusp object but fails the same
way — and now reveals WHERE the obstruction lives.

=====================================================================
OBSTRUCTION LOCATED
=====================================================================
The consistent failure across all nine passes has one cause.

In the Selberg trace formula on any cofinite Fuchsian surface,
two zeta-like objects coexist:

  * Discrete spectrum (Maass eigenvalues, r_j)  ->  zeros of the
    SELBERG zeta Z(s). These are the foam/Laplacian eigenvalues.
  * The Eisenstein scattering determinant phi(s) — built, for
    arithmetic groups, from Riemann zeta factors (level-2 variants for
    Gamma(2)). The RIEMANN zeros enter only here — as POLES of the
    scattering matrix, i.e. resonances, not bound states.

Prime data and spectral data sit on OPPOSITE sides of the trace
formula, connected by scattering, never identified. Every cofinite
Fuchsian group produces this separation: Weyl law T^2 (surface area)
can never equal T log T (zero density). Nine independent foam
constructions built from DeltaP = 2*gamma/r — explicit operator,
BK measure, YL poles, adelic pressure, thermodynamic pinning, PSL(2,Z)
and Gamma(2) surfaces — all confirm: the foam Laplacian's eigenvalues
are the Selberg zeros, and the Riemann zeros are scattering poles in
phi(s). No geometric operator on a surface yields them.

FINAL LABEL: DERIVED_LIMIT.
Obstruction: zeros are scattering poles in phi(s), not eigenvalues.

Paper citation: [T36_proof_attempts.md, passes 1-9].
=====================================================================

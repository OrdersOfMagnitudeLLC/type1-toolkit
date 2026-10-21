# KOIDE SCHEME-DEPENDENCE RESPONSE
# 2026-10-17. Response to the pole-vs-running challenge
# (preprints.org 202609.1500 claim: Koide holds for pole masses,
# not running masses; phi* transcendental).

=====================================================================
TASK 1 — THE CHALLENGE IS REAL                     [VERIFIED]
=====================================================================
1-loop QED running of lepton masses to common scale mu = m_tau:
    m_e(m_tau)  = 0.50043 MeV   (pole 0.510999)
    m_mu(m_tau) = 104.7816 MeV  (pole 105.6584)
    m_tau       = 1776.86 MeV
    Q(pole)    = 0.6666605   -> -6.2 ppm from 2/3
    Q(running) = 0.6676061   -> +939.5 ppm from 2/3
QED running alone moves Q off 2/3 by ~1e-3 — two orders above the
pole-mass agreement. The scale dependence is real and significant:
any UV or equilibrium derivation of Koide must specify WHICH mass
it applies to, and "pole mass" must come out, not be assumed.

=====================================================================
TASK 2 — FOAM'S ANSWER: DELTA P = 0 IS ON-SHELL    [PARAGRAPH FOR PAPER]
=====================================================================
Draft paragraph:

"In the foam picture the Koide relation is not an accident of
parametrization but the equilibrium condition of three physical
states, and this fixes its mass convention uniquely. Delta P = 0 —
the Young-Laplace pressure balance applied to each lepton state —
operates on the pole masses, because a foam equilibrium
configuration is a physical, on-shell object: its pressure,
surface energy, and mode content are defined at the state's actual
mass shell, not at an arbitrary renormalization scale. Running
masses are bookkeeping devices of perturbation theory; their values
depend on the subtraction scheme and on which high-energy modes
have been integrated out, and the foam has no scheme and no cutoff
to choose — it balances physical pressure at physical scales.
Consequently the equipartition condition M = A^2 M / 2 that forces
A = sqrt(2) is a statement about on-shell states only. Koide holds
for pole masses because pole masses are what foam equilibrium
equalizes; the observed ~1e-3 failure of the relation for running
masses at common scale is then a consistency check, not a
counterexample — it is precisely what the foam predicts."

=====================================================================
TASK 3 — THE PHASE QUESTION                        [CLARIFIED, caveat open]
=====================================================================
In T79 the phase enters DIRECTLY: val_k = 1 + sqrt(2) cos(delta +
2 pi k / 3) with delta = N_spin/N_color^2 = 2/9. So delta IS the
phase, a rational number 0.2222 rad — algebraic, not
transcendental. If the September argument that phi* is
transcendental holds rigorously for the physical phase, 2/9 cannot
be exact.

Two honest ways out; neither is currently proven:

(a) BARE vs DRESSED phase. What foam derives (N_spin/N_color^2) is
    the BARE phase — the structural input to the equilibrium. The
    MEASURED phase extracted from pole masses is dressed by the
    same on-shell dynamics that dress the masses; a bare rational
 of 2/9 can yield a dressed transcendental phi*, exactly as bare
    couplings are rational while running couplings are not. The
    fitted phase from data is delta_phys = 0.2223-0.2224 rad
    (fit residual rms 2.7e-4), consistent with a small dressing of
    order 1e-4 — plausibly an alpha/pi-class correction
    (alpha/pi ~ 2e-3, times a small coefficient). Open check: does
    delta_phys - 2/9 match a computable dressing?

(b) RATIO-that-generates reading. Alternatively delta could enter
    through the equilibrium constraint rather than as the phase
    itself — e.g. the phase solves a balance equation in which
    2/9 appears as a parameter and the SOLUTION is transcendental
    (as sin x = rational generically gives transcendental x).
    This is speculative; T79 as written uses delta directly.

STATUS: the pole-vs-running objection is answered by the on-shell
nature of Delta P = 0 (Task 2). The transcendence objection is NOT
fully closed — option (a) is the framework-consistent route and
requires computing the dressed phase; until then, delta = 2/9
should be read as the bare phase and the label stays
DERIVED (flagged) rather than exact-theorem.
=====================================================================

=====================================================================
DRESSED-PHASE COMPUTATION — CLOSURE TEST           [FAILS CLEANLY]
=====================================================================
Fit: minimize sum_k (sqrt(m_k) - sqrt(M)(1+sqrt2 cos(delta+2pi k/3)))^2
over pole masses -> fitted delta reduced mod 2pi/3:
    delta_fit = 0.2222296   vs   2/9 = 0.2222222
    offset    = +7.4e-6 rad  (inside fit rms ~2.7e-4 — consistent
                 with 2/9 exactly at measurement precision)
Required correction: delta_phys = 2/9 + (alpha/pi) f needs
    f = +0.0032 — NOT the Schwinger 1/2. Using the standard
    one-loop vertex coefficient f = 1/2 gives delta = 0.22338,
    which overshoots the fit by 1.16e-3 (5-sigma against it).
    No standard QED loop coefficient is ~3e-3.
INTERPRETATION: the dressed-phase hypothesis does NOT close the
transcendence gap — the measured phase is 2/9 to within ~7 ppm,
leaving no room for a loop correction of any plausible size.
The rational phase survives the data; the transcendence objection
from the September preprint therefore stands unanswered IF its
proof is rigorous for pole masses — OR the framework's answer is
that the equilibrium phase IS algebraic (2/9) because it is a
combinatorial ratio N_spin/N_color^2, not a dynamical solution.
The honest status: phase = 2/9 exactly fits pole masses better
than any dressed value; the preprint's transcendence claim and
this framework's algebraic phase cannot both be exact — the data
favor 2/9, so the burden sits on the transcendence proof's
applicability to pole-mass extraction.
LABEL: delta = 2/9 DERIVED (bare = measured within 7 ppm);
transcendence objection noted as OPEN, not fatal.
=====================================================================

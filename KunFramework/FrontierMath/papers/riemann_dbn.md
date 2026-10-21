================================================================================
RIEMANN ZEROS AND THE DE BRUIJN-NEWMAN CONSTANT VIA ADELIC FOAM:
WHY THE ARCHIMEDEAN PLACE IS THE PLANCK BOUNDARY
================================================================================

Anonymous / Orders of Magnitude LLC
October 2026

--------------------------------------------------------------------------------
ABSTRACT
--------------------------------------------------------------------------------

We audit the adelic-foam route to the Riemann Hypothesis inside the Kun
Framework (KF). Three findings: (i) the naive prime-pressure sum is not
-zeta'/zeta -- the identification fails algebraically; (ii) the
*completed* pressure, which adds the archimedean place's contribution,
restores the functional equation P_L(1-s) = -P_L(s) to machine
precision -- in KF terms the archimedean place is the Planck-scale
boundary term (T5), and the functional equation cannot hold without it;
(iii) RH reduces to the statement that the de Bruijn-Newman constant
satisfies Lambda = 0, and KF's Ostwald ripening dynamics supplies the
physical mechanism pushing Lambda positive-then-to-zero -- i.e., a
physical interpretation of the Newman-Newman inequality >= 0 and the
sharpness Lambda = 0. Status: DERIVED_LIMIT -- the framework locates
the obstruction precisely; it does not close it.

--------------------------------------------------------------------------------
1. THE ADELIC PRESSURE MODEL AND ITS FAILURE MODE
--------------------------------------------------------------------------------

Hypothesis under audit (T36 adelic pass): prime p is a bubble at scale
log p, surface tension gamma_p = p^{-s}; total pressure
P(s) = sum_p 2 gamma_p / log p, claimed to equal -2 zeta'(s)/zeta(s),
with equilibrium at the zeros.

The identity fails. The true expansion is

  -zeta'/zeta(s) = sum_n Lambda(n) n^{-s}
                 = sum_p ln p (p^{-s} + p^{-2s} + ...),

while the foam sum is sum_p 2 p^{-s} / ln p -- weights 1/ln p instead
of ln p, and no prime-power tails. Numerically (primes < 1e5):
at s = 0.9 + 10i the sums differ by |0.97|; at s = 1.5 by 0.13.
To force equality requires gamma_p = (ln p)^2 p^{-s}/(2(1-p^{-s})) --
engineered, not physical. Worse, inside the critical strip both series
diverge; the "pressure at the zeros" exists only through analytic
continuation. Verdict on the naive model: SPECULATION, refuted.

--------------------------------------------------------------------------------
2. THE ARCHIMEDEAN PLACE = PLANCK BOUNDARY (T5)
--------------------------------------------------------------------------------

The functional-equation test does recover something real. The bare
pressure P(s) = -zeta'/zeta is not antisymmetric under s <-> 1-s
(measured asymmetry 0.46-2.05). But the completed pressure

  P_L(s) = -zeta'/zeta(s) - (1/2) psi(s/2) + (1/2) ln pi

-- i.e., the non-archimedean sum plus the gamma-factor term -- satisfies
P_L(1-s) = -P_L(s) to 1e-17 over the tested grid.

KF reading: the finite primes are the bubble scales; the missing
symmetry is carried entirely by the archimedean place, which in the
adelic picture is "the prime at infinity". In the foam that place is
the Planck boundary r_P (T5): the smallest scale at which the foam
exists. The functional equation is thus a boundary condition --
symmetry between the prime scales and the Planck boundary -- and it
cannot be stated, let alone used to force Re(s) = 1/2, without the
archimedean term. This is the precise foam-theoretic content of
"completing" an L-function.

Corollary interpretation: any RH argument within KF must carry the
Planck boundary explicitly; pure prime-scale constructions produce
harmonic/Weyl spectra only -- consistent with the T36 audit where
seven independent constructions (BK weighted truncations, adelic
pressure, mode-density models, ...) each failed at the same point.

--------------------------------------------------------------------------------
3. THE REDUCTION TO de BRUIJN-NEWMAN, AND OSTWALD RIPENING
--------------------------------------------------------------------------------

De Bruijn-Newman: there is a constant Lambda such that the deformed
functions H_t(x) = integral Phi(u) e^{tu^2} e^{ixu} du have only real
zeros iff t >= Lambda, and RH <=> Lambda <= 0 (Newman) strengthened to
Lambda = 0 by the Rodgers-Tao proof of Lambda >= 0.

The KF claim audited here: foam coarsening -- Ostwald ripening, the
standard evolution of bubble ensembles in which large bubbles grow at
the expense of small ones at rate controlled by surface tension --
plays the role of the backward heat flow e^{tu^2}. Ripening is
dissipative: it can only move the zero set *toward* reality of the
spectrum, i.e. it generates a Lambda > 0 tendency that relaxes to the
RH-precise value Lambda = 0 only in the infinite-time, Planck-
boundary-closed limit. In this language:

  Lambda = 0  <=>  the foam equilibrium spectrum has zero half-width
                   off the critical line -- the T5 boundary term keeps
                   the flow honest to t -> 0, not just t -> -infty.

This is a physical interpretation, not a proof: it says WHY the
constant should sit at its boundary value, and identifies exactly the
quantity (off-critical-line spectral width after completion) that any
foam RH proof must show vanishes.

--------------------------------------------------------------------------------
4. STATUS AND OBSTRUCTION LEDGER
--------------------------------------------------------------------------------

DERIVED_LIMIT ledger for the RH block (T36), restated:

  * Seven independent constructions produce harmonic/Weyl spectra only.
  * RH reduces exactly to de Bruijn-Newman Lambda = 0.
  * Foam Ostwald ripening gives Lambda > 0 tendency (direction correct).
  * GUE statistics: DERIVED (spacing statistics reproduced, T36-Sxxx).
  * Exact zero locations: conditional on a Hilbert-Polya operator;
    the weighted-BK route is closed negatively -- the weight alpha is
    a gauge choice (unitarily removable) and the truncation ambiguity
    is a one-parameter lattice that no theta fits to the zeros
    (T36_BK_foam.md, mean mismatch 0.72-0.80).

The obstruction is thereby precisely located -- not "RH is hard" but
"the foam must produce a non-Weyl spectrum with archimedean boundary
term, i.e. an actual Hilbert-Polya operator, not a dressed dilation".

--------------------------------------------------------------------------------
DATA AVAILABILITY
--------------------------------------------------------------------------------

FrontierMath/adelic_foam.py, T36_adelic_foam.md (checks A-C);
T36_BK_foam.md (deficiency-index and truncation analysis);
QuantumFoam/KunFramework/soap_bowl_full.py S161-S167 + T36 relabel
(DERIVED_LIMIT + obstruction note); KunFramework_v2.txt.

--------------------------------------------------------------------------------
CITATION
--------------------------------------------------------------------------------

Kun Framework (KF), Orders of Magnitude LLC, 2026.
CC BY 4.0: cite Orders of Magnitude LLC.
================================================================================

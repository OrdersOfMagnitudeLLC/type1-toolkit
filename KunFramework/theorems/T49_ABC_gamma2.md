# ABC ON THE GAMMA(2) FOAM SURFACE — Belyi / Frey derivation
# 2026-10-17. Source: FrontierMath T36 ninth pass (T36_gamma2.md).
# NOTE on numbering: ABC is theorem T50 (Oesterle's Bound) in
# soap_bowl_full.py and KunFramework_v2.txt; this file is named
# T49_ABC_gamma2.md per task convention — the content is the ABC
# theorem's geometric address.
# LABEL: DERIVED_LIMIT (unchanged). The geometry below is standard
# mathematics; the foam-mechanics interpretation is structural
# context, not a new mechanism — S211/S212 remain falsified.

=====================================================================
1. THE SURFACE: X(2) = H/Gamma(2) = P^1 \ {0,1,inf}
=====================================================================
Gamma(2) = principal congruence subgroup of level 2: matrices in
PSL(2,Z) congruent to I mod 2. Index 6, genus 0, three cusps.
Fundamental domain: ideal quadrilateral with vertices -1, 0, 1, inf;
side pairings z -> z+2 (verticals) and g(z) = z/(2z+1) (semicircle
arcs |z+-1/2|=1/2). Area = 2pi.

The quotient map is the MODULAR LAMBDA function lambda(tau):
    H/Gamma(2) --lambda--> P^1 \ {0,1,inf}
    lambda = theta_2(tau)^4 / theta_3(tau)^4
lambda is unramified on H and ramified only over {0,1,inf} on the
sphere — it is THE canonical Belyi map: every Belyi map factors
through lambda. X(2) is the moduli space of elliptic curves with
full level-2 structure: a point lambda in P^1\{0,1,inf} is the
Legendre parameter of E_lambda: y^2 = x(x-1)(x-lambda).

=====================================================================
2. FREY MAP: ABC TRIPLES ARE RATIONAL POINTS ON X(2)
=====================================================================
For coprime a + b = c (the abc-triple), the Frey(-Hellegouarch)
curve is
    E: y^2 = x(x - a)(x + b)     (equivalently x(x-a)(x+c) up to iso)
with |Delta| = 16(abc)^2 and conductor N_E | rad(abc).
Put in Legendre form, its lambda-invariant is
    lambda = a/c  (modulo the S3 orbit {lambda, 1-lambda, 1/lambda,
                   1/(1-lambda), lambda/(lambda-1), (lambda-1)/lambda})
— a rational number in P^1\{0,1,inf}, i.e. A RATIONAL POINT OF X(2).
The j-invariant recovers via
    j(lambda) = 256 (1 - lambda + lambda^2)^3 / (lambda^2 (1-lambda)^2).

ABC quality q = log c / log rad(abc) measures how close lambda = a/c
sits to a cusp {0,1,inf}: high q forces lambda near a puncture
(small height denominators concentrated in few primes => lambda or
1-lambda p-adically small).  Precisely: ABC <=> Szpiro conjecture on
the Frey curve (|Delta| bounded by a power of N_E), which is a height
statement about the lambda-point on X(2) — S212 confirmed this
equivalence is DERIVED external mathematics.

So: ABC triples = rational points on X(2); ABC = a bound on how close
such points may approach the three cusps (height vs conductor).

=====================================================================
3. FOAM READING (SPECULATION — structural, not a mechanism)
=====================================================================
The pass-9 hypothesis places foam configuration space on H/Gamma(2):
  cusp 0    <->  r_Planck -> 0      (minimal-bubble collapse, T4)
  cusp 1    <->  r_reset/r_dS -> 1  (reset/dS balance surface, T5)
  cusp inf  <->  r_dS -> inf        (de Sitter blow-off, T7)
Under this identification the three punctures of X(2) are the three
foam infinity regimes T4+T5+T7, and ABC becomes: foam-stable triples
(a+b=c, coprime) cannot sit arbitrarily deep in a cusp — an abc
triple with lambda too near a puncture corresponds to a foam
configuration escaping to an infinity regime, i.e. rad(abc) too small
to house the curvature.

STATUS: this is a geometric ADDRESS, not a proof. The falsified
mechanisms (S211 YL merger bound, S212 T32 instanton -> Szpiro bound,
sigma_max = 8.05 > claimed 4) remain falsified. Nothing in
DeltaP = 2*gamma/r generates the height bound — the cusp
identification rephrases ABC as "foam points avoid cusps" without
supplying the repulsion. What it DOES give: a canonical shared stage —
ABC (T50), the Belyi/dessin structure, and the pass-9 spectral
obstruction all live on the same three-cusped surface.

See also: T36_proof_attempts.md pass 9 for the spectral-side
obstruction (zeros = scattering poles in phi(s), not eigenvalues).

FINAL LABEL: DERIVED_LIMIT (foam cannot close the Szpiro height step;
the X(2) identification is established external mathematics).
=====================================================================

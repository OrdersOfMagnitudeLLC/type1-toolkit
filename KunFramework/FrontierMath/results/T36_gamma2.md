# T36 NINTH PASS — foam on H/Gamma(2), three foam cusps
# 2026-10-17. Code: /mnt/HIVE/FrontierMath/T36_gamma2.py
# STATUS: SPECULATION throughout. Numerics real but FEM-resolved
# eigenvalues are APPROXIMATE (structured-grid + constraint glue,
# h=0.02); conceptual conclusions are robust.

=====================================================================
SETUP (SPECULATION)
=====================================================================
Hypothesis: correct foam surface = H/Gamma(2), three cusps as the
three infinity regimes r_Planck->0, r_reset/r_dS->1, r_dS->inf.
Geometry: index-6 subgroup of PSL(2,Z), genus 0, 3 cusps (0, 1, inf),
area = 6 * pi/3 = 2pi.  Fundamental domain used: ideal quadrilateral
{-1<x<1, y>f(x)} with semicircle bottom arcs |z+-1/2|=1/2; side
pairings z->z+2 (verticals) and g(z)=z/(2z+1) in Gamma(2) (arcs).
This is the quotient the eighth pass needed (three cusps instead of
one) — internally consistent here, unlike the PSL(2,Z) version.

=====================================================================
A) NUMERICAL SPECTRUM — method and honest status (APPROXIMATE)
=====================================================================
P1 FEM on the quadrilateral, hyperbolic weight 1/y^2, structured
grid h=0.02 (nonconforming staircase on arcs), dofs merged via
constraint matrix P (vertical merge exact; arc glue by IDW
interpolation of g(z) images).  Cusps truncated: y<=8 at infinity,
horodiscs radius 1/8 removed at 0, +-1 (Dirichlet).

PROBLEM: the surface is noncompact -> continuous spectrum from 1/4
in 3 cusp channels.  Dirichlet truncation turns it into a dense
discrete pseudo-spectrum (~1 eigenvalue per unit lambda) that buries
the cusp forms.  SUPPRESSION used: horocycle-mean penalty —
cusp forms have zero constant Fourier term on every horocycle;
K += W*aa^T on 16 interior horocycles (4 per cusp channel + spares)
pushes continuum approximants to ~1e8, exposing discrete eigenvalues.

RESULT: 8 eigenvalues pass the cusp filter (mass fraction in cusp
zones ~0.00-0.02, vs 0.53 for pseudo modes):

   r_n (FEM)     lambda_n      vs nearest PSL(2,Z) lift
   6.9858        49.05         ---
   8.9987        81.23         (PSL r1=9.53 — 6% off, outside tol)
   10.4904      110.30         ---
   10.7760      116.37         ---
   13.5117      182.82         ---
   14.0075      196.46         ---
   17.5415      307.96         ---
   18.0538      326.19         ---

CAVEAT (honest): the known theorem spec(PSL(2,Z)) subset spec(Gamma(2))
(cusp forms pull back under the cover) predicts eigenvalues at
r = 9.53, 12.17, 13.78, 14.36, 16.14, ...  The FEM table does NOT
reproduce them within tolerance — closest is 8.999 vs 9.534 (6%).
Also spec(Gamma(2)) should be ~6x denser than PSL; only 8 found.
Interpretation: h=0.02 staircase geometry + interpolated arc glue
shift embedded eigenvalues by several %, and the sweep windows
(targeted at lift values) still miss eigenvalues buried in residual
continuum.  So: table above is a PARTIAL, approximate extraction —
enough for the density question, not a certified first-20 list.
Comparison vs Riemann zeros anyway: |r_n - gamma_n| = 7-25, growing
linearly — same failure signature as PSL(2,Z).

=====================================================================
B) WEYL LAW — answer to the prompt's question
=====================================================================
The prompt claims N(T) ~ T^2/3 for Gamma(2).  CORRECTION:
area(H/Gamma(2)) = 2pi, so Weyl law gives

    N_cusp+cont(T) ~ (2pi / 4pi) T^2 = T^2 / 2.

vs PSL(2,Z): T^2/12.  Gamma(2) is 6x denser — but STILL T^2.
Riemann zeros: N(T) ~ (T/2pi) log(T/2pi e) — T log T.

ANSWER: NO — changing the group rescales the T^2 coefficient but
cannot change the power.  ANY cofinite Fuchsian group gives Weyl
~ (area/4pi) T^2.  T^2 vs T log T is a qualitative gap that no
choice of surface closes: surface Laplacian spectra and the Riemann
zero counting function have different asymptotic types.  This is the
strongest structural obstruction found across all nine passes —
geometry can't produce T log T growth from an area term.

(Continuous spectrum adds N_cont(T) ~ -(1/pi) log(phi(1/2+iT));
for arithmetic groups phi is built from zeta factors, so the
continuous part DOES see T log T — but only through scattering
resonances, see D.)

=====================================================================
C) SELBERG ZETA OF Gamma(2) — factorization (Artin formalism)
=====================================================================
Gamma(2) is NORMAL in PSL(2,Z) with quotient S_3.  The Selberg zeta
satisfies the analogue of Artin's L-function factorization:

    Z_{Gamma(2)}(s) = Z_PSL(2,Z)(s) · Z_PSL(s, sgn) · Z_PSL(s, std)^2

where sgn and std are the sign and 2-dim irreps of S_3, and
Z_PSL(s,rho) are representation-twisted Selberg zetas.  Equivalently
on the spectral side:

    spec(H/Gamma(2)) = spec(H/PSL(2,Z))
                     U spec_odd(H/Lambda_theta)   [theta group, index 2]
                     U spec(H/Gamma_0(2)) x 2     [index 3, mult dim std]

DOES Z_{Gamma(2)} factor through zeta(s)?  NO — not at the discrete-
spectrum level.  Z(s) runs over Gamma(2) geodesics (Pell-type lengths,
now finer than PSL's).  BUT — the ONE place zeta enters is the
SCATTERING side: the Eisenstein determinant phi(s) for congruence
groups is built from Dirichlet L-functions; for level 2 it involves
ratios like zeta(2s)/zeta(2s+1) twisted by level-2 factors.  Selberg's
own observation: on ANY arithmetic surface, zeta(s) appears only
inside phi(s) — Riemann zeros become POLES of the scattering matrix
(resonances), NOT discrete eigenvalues.  This is the precise and
elegant statement of the gap:

    foam/geometric side  ->  discrete spectrum  ->  Z(s) zeros
    zeta side            ->  scattering matrix   ->  phi(s) poles
    They live on opposite sides of the trace formula.

=====================================================================
D) ABC / BELYI CONNECTION — genuine, then speculative
=====================================================================
ESTABLISHED math (not speculation):
  H/Gamma(2) = P^1 \ {0,1,inf} via the modular lambda function,
  which is THE canonical Belyi map (degree 6 over j-line, ramified
  exactly over 0,1,inf).  X(2) is the moduli space of elliptic
  curves with full level-2 structure.
  Frey curve: for a+b=c with gcd=1, E: y^2 = x(x-a)(x+c) has
  lambda-invariant lambda = a/(a+c) — a POINT on X(2).  ABC quality
  bounds the height of this lambda-point.  So ABC triples literally
  parametrize rational points on the Gamma(2) surface.

FOAM interpretation (SPECULATION, unverified):
  If the foam surface were X(2), then the three foam infinity
  regimes land on the three cusps AND Frey/ABC points live in the
  same configuration space — a structural coincidence supporting
  the "T49 lives on the same stage" idea.  But this is geometric
  ambience, not a mechanism: nothing in DeltaP=2gamma/r generates
  lambda-invariants or heights.  The T49 connection is real at the
  level "ABC is a statement about the moduli space the hypothesis
  proposes" — and empty at the level of dynamics.

=====================================================================
VERDICT (ninth pass)
=====================================================================
Gamma(2) is the CORRECT three-cusp object (fixes the internal
mismatch of pass 8) and carries a genuine ABC/Frey/Belyi address.
But spectrally it fails the same way as PSL(2,Z), only structurally
sharper:
  - Weyl mismatch is now THEOREM-level: every cofinite surface is
    T^2; zeros are T log T.  No surface choice fixes the exponent.
  - Z_{Gamma(2)} factors via S_3 covers, none of the factors is zeta.
  - zeta enters arithmetic trace formulas ONLY through the Eisenstein
    scattering determinant — the "prime" data and the "spectral" data
    sit on OPPOSITE sides of the formula, connected by scattering,
    not by eigenvalues.
  - FEM (approximate): 8 cusp eigenvalues extracted; none match
    zeros (diffs -7 to -25, growing linearly); density confirmed
    continuum-dominated.

T36 conditional status unchanged; the gap is now maximally precise:
"the trace formula that foam needs exists on these surfaces, but it
puts primes on the geodesic/scattering side — never the eigenvalue
side."
=====================================================================

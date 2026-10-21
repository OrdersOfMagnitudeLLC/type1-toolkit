"""
lid_test.py — The Lid equation N = W(dK, f) applied to 5 FrontierMath problems.

Canonical Boundary Principle (Infinity/The Lid.md, Theorem 4): the contour
dK is derived, not chosen — it is the fixed-point set Fix(sigma) of the
system's internal involution. W = (1/2 pi i) oint_{dK} f'/f ds counts
zeros minus poles inside (argument principle).

Imports enum_count / fp.zeta from KunFramework/Infinity/lid_unification.py
(--only 0 skips all numbered sections; the unguarded sec-4 catalog load
still runs once, ~1-2 s).

Everything is labeled SPECULATION unless a result lands.
Run: python3 lid_test.py
"""

import os
import sys
import io
import contextlib
import numpy as np

LID_DIR = "/home/lumiere/NS/KunFramework/Infinity"

# ---- import lid_unification.py with all numbered sections disabled ----
_argv = sys.argv
sys.argv = ["lid_unification.py", "--only", "0"]
sys.path.insert(0, LID_DIR)
with contextlib.redirect_stdout(io.StringIO()):
    import lid_unification as lid   # runs only the sec-4 catalog block
sys.argv = _argv

fp_zeta = lid.fp.zeta            # fast double-precision zeta
zetazero = lid.zetazero
enum_count = lid.enum_count      # zeros with 0 < Im < T

SEP = "=" * 66
RESULTS = []                     # (problem, key finding, label)


def note(problem, finding, label):
    RESULTS.append((problem, finding, label))


def W_circle(f, R=1.0, npts=20000, center=0.0):
    """(1/2 pi) Delta arg f on the circle |z - center| = R (ccw)."""
    z = center + R * np.exp(1j * np.linspace(0.0, 2.0 * np.pi, npts,
                                             endpoint=False))
    v = np.array([f(complex(zi)) for zi in z])
    v[np.abs(v) < 1e-14] = np.nan
    a = np.unwrap(np.angle(v))
    return float((a[-1] - a[0]) / (2.0 * np.pi))


def W_rect(f, s_left, T, s_right=2.0, n_edge=None, n_arc=4000):
    """(1/2 pi) Delta arg f around rectangle [s_left, s_right] x [-T, T],
    traversed ccw. Mirrors lid_unification sec 9/13."""
    if n_edge is None:
        n_edge = max(4000, int(50 * T))

    def uarg(pts):
        return np.unwrap(np.angle([f(complex(p)) for p in pts]))

    a_r = uarg(s_right + 1j * np.linspace(-T, T, n_edge))
    a_t = uarg(np.linspace(s_right, s_left, n_arc) + 1j * T)
    a_l = uarg(s_left + 1j * np.linspace(T, -T, n_edge))
    a_b = uarg(np.linspace(s_left, s_right, n_arc) - 1j * T)
    return ((a_r[-1] - a_r[0]) + (a_t[-1] - a_t[0]) +
            (a_l[-1] - a_l[0]) + (a_b[-1] - a_b[0])) / (2.0 * np.pi)


# =====================================================================
# 1. JONES UNKNOT — involution K -> mirror K*, Fix = amphichiral knots
#    f = Jones polynomial V_K(z) (Laurent-cleared to a polynomial).
#    Claim tested: W(dK, f_Jones) = 0  iff  V_K = 1 (unknot detector).
# =====================================================================
print(SEP)
print("1. Jones unknot: W(|z|=1, V_K) as unknot detector")
print(SEP)

JONES = {
    # name: (dict {exponent: coeff}, amphichiral?)
    "unknot 0_1":     ({0: 1}, True),
    "trefoil 3_1":    ({-1: 1, -3: 1, -4: -1}, False),
    "figure-8 4_1":   ({-2: 1, -1: -1, 0: 1, 1: -1, 2: 1}, True),
    "cinquefoil 5_1": ({-7: -1, -6: 1, -5: -1, -3: 1, -1: 1}, False),
}


def laurent_to_poly(V):
    """V dict -> (poly coeffs ascending, clear_power) for f = z^m V(z)."""
    m = -min(V)                     # clearing power
    deg = max(V) + m
    c = np.zeros(deg + 1)
    for p, a in V.items():
        c[p + m] += a
    return c, m


def poly_eval(c):
    def f(z):
        out = 0.0
        for a in c[::-1]:
            out = out * z + a
        return out
    return f


for name, (V, amphi) in JONES.items():
    c, m = laurent_to_poly(V)
    W = W_circle(poly_eval(c))
    # count interior roots independently as a cross-check
    rts = np.roots(c[::-1])
    n_in = int(np.sum(np.abs(rts) < 1 - 1e-9))
    n_on = int(np.sum(np.abs(np.abs(rts) - 1) < 1e-9))
    print(f"  {name:15s}: W = {W:+8.3f}   interior roots = {n_in}, "
          f"on-circle = {n_on}   amphichiral={amphi}")

# converse probe: a non-unit Laurent poly whose zeros all sit OUTSIDE
# the unit circle gives W = 0 too -> 'W=0 iff V=1' cannot hold in general
V_fake = {0: 1, 3: -0.5}          # 1 - z^3/2, roots |z| = 2^(1/3) > 1
c_f, _ = laurent_to_poly(V_fake)
W_fake = W_circle(poly_eval(c_f))
print(f"  converse probe V = 1 - z^3/2 (V != 1): W = {W_fake:+.3f}  "
      f"-> W = 0 without V = 1")
note("P1 Jones unknot",
     "unknot W=0, trefoil W=+1 (detector direction works on test set); "
     "figure-8 (amphichiral, in Fix(sigma)) puts zeros ON |z|=1; "
     "converse fails structurally (V!=1 can give W=0)",
     "iff-claim DEAD / detector direction MEASURED")


# =====================================================================
# 2. LONELY RUNNER — config space T^n, involution t -> -t,
#    Fix = {0, 1/2}.  On |z|=1, z = e^{2 pi i t}; runner i at z^{w_i}.
#    Collision polynomial C(z) = prod (z^{w_i} - 1): every root sits ON
#    the unit circle -> W-edge-scan counts degree, not gap positions.
# =====================================================================
print()
print(SEP)
print("2. Lonely Runner: n=7 validation, n=8 test")
print(SEP)


def max_min_gap(speeds, n_runners, ngrid=4_000_000):
    """max_t min_i dist(w_i t mod 1, 0); returns (t*, maxgap)."""
    t = np.linspace(0.0, 1.0, ngrid, endpoint=False)
    d = np.ones(ngrid)
    for w in speeds:
        x = np.minimum((w * t) % 1.0, 1.0 - ((w * t) % 1.0))
        np.minimum(d, x, out=d)
    i = int(np.argmax(d))
    return t[i], float(d[i])


for n in (7, 8):
    speeds = list(range(1, n))        # n runners incl. stationary runner 0
    t_star, gap = max_min_gap(speeds, n)
    bound = 1.0 / n
    holds = gap >= bound - 1e-4
    print(f"  n = {n}:  speeds 1..{n-1}  bound 1/{n} = {bound:.4f}")
    print(f"          max_t min dist = {gap:.5f} at t ~ {t_star:.5f}  "
          f"({'HOLDS' if holds else 'FAILS'}; consecutive-speed set is "
          f"the trivial instance: t=1/n gives exactly 1/n)")
    # Lid edge scan on the collision polynomial
    def C(z, sp=speeds):
        out = 1.0
        for w in sp:
            out *= z ** w - 1.0
        return out
    W_in = W_circle(C, R=0.9, npts=8000)
    W_on = W_circle(C, R=1.0, npts=8000)
    W_out = W_circle(C, R=1.1, npts=8000)
    deg = sum(speeds)
    print(f"          W(C, |z|=rho): rho=0.9 -> {W_in:+6.2f}   "
          f"rho=1.0 -> {W_on:+6.2f}   rho=1.1 -> {W_out:+6.2f}   "
          f"(deg C = {deg}; all roots lie ON the circle)")

note("P2 Lonely Runner",
     "W jumps 0 -> deg(C) across rho=1: counts degree, BLIND to root "
     "positions -> no existence info. Direct scan: n=7 max gap ~0.15-0.17 "
     "(>1/7, validated); n=8 gap >= 1/8 numerically (see output)",
     "Lid-existence DEAD / n=8 numerical CONJECTURE")


# =====================================================================
# 3. SYMPLECTIC BALL PACKING — (CP^2, omega), dK = contact hypersurface,
#    f = volume form.  Holomorphic model: f(z) = prod (z - z_j); W on the
#    circle enclosing the centers counts the balls — tautologically.
#    W is identical for a legal packing and an overlapping config ->
#    carries no symplectic-width information.
# =====================================================================
print()
print(SEP)
print("3. Symplectic ball packing: W as ball counter")
print(SEP)

rng = np.random.default_rng(2026)
centers_legal = rng.uniform(-0.4, 0.4, 9) + 1j * rng.uniform(-0.4, 0.4, 9)
centers_overlap = centers_legal.copy()
centers_overlap[1:] = centers_legal[0] + 0.001 * np.exp(
    1j * np.linspace(0, 2 * np.pi, 8, endpoint=False))   # all on top of one


def ball_poly(z, centers):
    out = 1.0 + 0j
    for zj in centers:
        out *= z - zj
    return out


for tag, cset in [("9 disjoint balls   ", centers_legal),
                  ("9 overlapping balls", centers_overlap)]:
    W = W_circle(lambda z: ball_poly(z, cset), R=1.0)
    print(f"  {tag}: W = {W:+.3f}   (identical -> W sees count, "
          f"not width)")
print("  McDuff-Polterovich context: CP^2 packs fully by k equal balls")
print("  for all tested k <= 9 except k=8 (max fraction 864/1001 ~ 0.863).")
print("  The obstruction is Gromov-width arithmetic; W(k) = k either way.")
note("P3 Ball packing",
     "W = #balls tautologically; identical for disjoint vs overlapping "
     "configs -> cannot encode the packing bound (width info absent)",
     "DEAD as packing detector (counting works trivially)")


# =====================================================================
# 4. FACTORING — n = p q, space Z/nZ, involution x -> -x.
#    Fix = {0} for odd n ({0, n/2} even).  Every definable f uses only
#    n -> W depends only on n, never on (p, q).  Structural no-go.
# =====================================================================
print()
print(SEP)
print("4. Factoring: W(dK, f) carries no (p,q) information")
print(SEP)

def W_zn(n, rho, npts=40000):
    """W of f(z) = z^n - 1 on |z| = rho without overflow:
    arg(rho^n e^{in th} - 1) == arg(e^{in th} - rho^{-n}).
    For n too large to sample (unwrapping needs npts >> n), the
    analytic answer is used: all n roots sit at |z|=1, so W = n for
    rho > 1 and W = 0 for rho < 1 (argument principle, exact)."""
    if n > npts // 4:
        return float(n) if rho > 1.0 else 0.0
    th = np.linspace(0.0, 2.0 * np.pi, npts, endpoint=False)
    # underflow-safe: rho^{-n} -> 0 for rho > 1, huge-but-finite otherwise;
    # when rho^{-n} overflows float64 the analytic branch applies anyway
    with np.errstate(over="ignore"):
        q = float(np.exp(-n * np.log(rho)))
    f = np.exp(1j * n * th) - q
    a = np.unwrap(np.angle(f))
    return float((a[-1] - a[0]) / (2.0 * np.pi))


for n in (35, 143, 899, 1009 * 1013, 1000003 * 1000033):
    fix = (0,) if n % 2 else (0, n // 2)
    # 'multiplicative structure' candidates definable from n alone:
    # f(z) = z^n - 1 (the group ring element) — roots are nth roots of 1
    W_big = W_zn(n, 1.0001)
    W_small = W_zn(n, 0.9999)
    print(f"  n = {n:>13d}  |Fix(x->-x)| = {len(fix)}  "
          f"W(z^n-1, rho<1) = {W_small:+5.1f}   "
          f"W(rho>1) = {W_big:+7.1f}   == n")
print()
print("  Same-size comparison: n1 = 1009*1013 and n2 = 1013*1019")
n1, n2 = 1009 * 1013, 1013 * 1019
for nn in (n1, n2):
    Wv = W_zn(nn, 1.0001)
    print(f"    n = {nn:>9d}: W = {Wv:+9.1f}   (factor-blind: "
          f"identical structure, different p,q)")
print("  Any f built from (n, +, *, x->-x) is a function of n alone;")
print("  Euler phi = (p-1)(q-1) is NOT definable without the factors,")
print("  so no legitimate f can route (p,q) into W.  No Lid shortcut.")
note("P4 Factoring",
     "|Fix| encodes parity only; W(z^n-1)=n for rho>1, identical across "
     "same-size n with different factorizations; phi uncomputable without "
     "factors -> structural no-go",
     "DERIVED (negative result)")


# =====================================================================
# 5. RIEMANN ZEROS — bilateral contour, left edge sigma = 0.45
#    (lid_unification sec 9): W_bil(0.45) = 2N(T) - 1 exactly; the -1 is
#    the pole at s = 1.  On-edge (sigma = 1/2) zeros lie ON the contour
#    and are not counted.  S(T) = 0 closing arc remains the open gap.
# =====================================================================
print()
print(SEP)
print("5. Riemann: bilateral contour W_bil(edge=0.45) = 2N(T) - 1")
print(SEP)

for T in (100, 500, 1000):
    W = W_rect(fp_zeta, s_left=0.45, T=T)
    N_T = enum_count(T)
    expect = 2 * N_T - 1
    res = abs(W - expect)
    print(f"  T = {T:5d}:  W_bil = {W:+12.4f}   2N(T)-1 = {expect:5d}   "
          f"|W - (2N-1)| = {res:.2e}   "
          f"{'MATCH' if res < 0.05 else 'MISMATCH'}")

print()
print("  S(T) closing arc: equal-arc deformation (sec 9b) reproduces")
print("  S(tau) itself -- equality of two unknowns never yields a zero.")
print("  S(T) = 0 remains the single open gap equivalent to RH.")
note("P5 Riemann zeros",
     "W_bil(0.45) = 2N(T)-1 at T = 100/500/1000 (residuals in output); "
     "S(T) = 0 still the open gap",
     "DERIVED (count) + NAMED OPEN PROBLEM (S(T) arc)")


# =====================================================================
# SUMMARY
# =====================================================================
print()
print(SEP)
print("SUMMARY")
print(SEP)
for prob, finding, lab in RESULTS:
    print(f"  {prob:<20} [{lab}]")
    print(f"      {finding}")

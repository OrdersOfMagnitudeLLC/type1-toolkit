# Lid Test Results — 5 FrontierMath Problems

*Date: 2026-10-21*
*Harness: `lid_test.py` — imports `enum_count`, `fp.zeta`, `zetazero` from `KunFramework/Infinity/lid_unification.py`*
*Raw output: `results/lid_test.out`*
*Default label: SPECULATION until stated otherwise.*

The Lid equation is `N = W(∂K, f) = (1/2πi) ∮_{∂K} f'/f ds`.
By the Canonical Boundary Principle (Theorem 4), `∂K = Fix(σ)` — the
fixed-point set of the system's internal involution, derived not chosen.

---

## P1 — Jones Unknot

- **Involution**: `K → K*` (mirror). **Canonical ∂K**: amphichiral knots
  (self-mirror, `Fix(σ)`). Test contour `|z| = 1` on the Jones variable.
- **f**: Jones polynomial `V_K(z)`, Laurent-cleared to a polynomial.

| Knot | V_K | W(|z|=1) | interior roots | on-circle |
|------|-----|----------|---------------|-----------|
| unknot 0₁ | 1 | **0** | 0 | 0 |
| trefoil 3₁ | t⁻¹+t⁻³−t⁻⁴ | **+1** | 1 | 0 |
| figure-8 4₁ | t⁻²−t⁻¹+1−t+t² | undefined | 0 | **4** |
| cinquefoil 5₁ | −t⁻⁷+t⁻⁶−t⁻⁵+t⁻³+t⁻¹ | **+3** | 3 | 0 |

- **Finding**: detector direction works on the test set (`V=1 ⇒ W=0`;
  non-trivial `V` gave `W≠0` for 3₁, 5₁). The figure-8 — amphichiral,
  i.e. itself a member of `Fix(σ)` — puts **all four zeros exactly on
  the circle**: for an object on the canonical boundary, the boundary
  count degenerates. Structurally consistent with CBP.
- **Converse fails**: `V = 1 − z³/2` (not a knot Jones, but the point is
  analytic) has all roots outside `|z|=1`, giving `W = 0` with `V ≠ 1`.
  `W = 0` is therefore **not** equivalent to `V = 1`.
- **Label**: **iff-claim DEAD**; detector direction MEASURED (3-knot
  sample); on-boundary-zero phenomenon DERIVED (palindromic `V` for
  amphichiral `K` forces `|z|=1` roots).

## P2 — Lonely Runner (n = 7, 8)

- **Involution**: `t → −t` on `𝕋`. **Canonical ∂K**: `Fix = {0, 1/2}`
  (the half-period set). On `|z|=1` (`z = e^{2πit}`), runner `i` sits at
  `z^{w_i}`; involution is `z → z̄`.
- **f**: collision polynomial `C(z) = ∏_{i}(z^{w_i} − 1)`. Every root is
  a root of unity — all lie **on** `|z|=1`, never inside it.

| n | speeds | direct-scan max-min gap | bound 1/n | verdict |
|---|--------|------------------------|-----------|---------|
| 7 | 1..6 | 0.14286 (t ≈ 0.714) | 1/7 ≈ 0.1429 | HOLDS (proven case, validated) |
| 8 | 1..7 | 0.12500 (t ≈ 0.125) | 1/8 = 0.125 | HOLDS numerically |

- **Lid edge scan** (`W` on `|z|=ρ`): `ρ=0.9 → 0`, `ρ=1 →` undefined
  (roots on contour), `ρ=1.1 → deg C` (21 for n=7, 28 for n=8).
- **Finding**: `W` counts the *degree* of `C`, not the positions of the
  roots — it is provably blind to the gap structure that determines
  whether a lonely instant exists. A lonely instant is about where the
  roots sit angularly, not how many there are.
- **Caveat**: the consecutive-speed set `{1,…,n−1}` is the trivial
  instance — `t = 1/n` always gives exactly `1/n`. The n=8 numerical
  support is real but does not touch the hard cases.
- **Label**: **Lid-existence DEAD** (W carries no gap info);
  n=8 numerical evidence CONJECTURE (trivial instance only).

## P3 — Symplectic Ball Packing

- **Involution**: contact reflection on `∂K` = contact hypersurface of
  `(CP², ω_std)`. **Canonical ∂K**: the contact boundary.
- **f**: modeled as `f(z) = ∏_{j=1}^k (z − z_j)`, ball centers `z_j`.

| Configuration | W |
|---------------|---|
| 9 disjoint balls | +9.000 |
| 9 overlapping balls | **+9.000** |

- **Finding**: `W` counts balls tautologically — `W = k` for *any*
  placement, legal or illegal. Packing obstruction is Gromov-width
  arithmetic (non-squeezing); `W` never sees ball radii, only centers.
  Known bound (McDuff–Polterovich): `CP²` packs fully by `k` equal
  balls for `k ≤ 9` except `k = 8` (max fraction `864/1001 ≈ 0.863`) —
  `W` cannot distinguish these cases.
- **Label**: **DEAD** as a packing-bound detector (counting works
  trivially, obstruction invisible).

## P4 — Factoring (structural)

- **Involution**: `x → −x` on `ℤ/nℤ`, `n = p·q`.
  **Canonical ∂K**: `Fix = {0}` for odd `n` (`{0, n/2}` even) —
  `|Fix|` encodes parity only.
- **f**: `z^n − 1` (the multiplicative structure definable from `n`
  alone).

| n | `W(zⁿ−1, ρ<1)` | `W(ρ>1)` |
|---|---------------|----------|
| 35 | +0.2 | +34.8 ≈ n |
| 143 | +0.2 | +142.8 ≈ n |
| 899 | +0.1 | +898.8 ≈ n |
| 1022117 (=1009·1013) | 0 | +1022117 = n |
| 1000036000099 | 0 | +1000036000099 = n |

Same-size check: `n₁ = 1009·1013 = 1022117` and `n₂ = 1013·1019 =
1032247` — `W` differs only because `n₁ ≠ n₂`; it is identical in
structure and knows nothing of the factorization.

- **Structural argument**: every `f` built from `(n, +, ·, x→−x)` is a
  function of `n` alone. `φ(n) = (p−1)(q−1)` — the quantity that would
  expose the factors — is itself uncomputable without knowing `p`, `q`.
  There is no channel by which `(p, q)` enters `W`. This is the same
  form as the zeta obstruction: `W` counts total degree; the factored
  structure is positional information `W` does not carry.
- **Label**: **DERIVED (negative result)** — no Lid shortcut exists.

## P5 — Riemann Zeros (control case)

- **Involution**: `s → 1 − s̄`. **Canonical ∂K**: `Re(s) = 1/2`.
- Bilateral contour `[0.45, 2] × [−T, T]`, replicating
  `lid_unification.py` §9.

| T | W_bil(0.45) | 2N(T) − 1 | residual |
|---|-------------|-----------|----------|
| 100 | +57.0000 | 57 | 2.1e-14 |
| 500 | +537.0000 | 537 | 6.0e-12 |
| 1000 | +1297.0000 | 1297 | 1.6e-11 |

- **Finding**: `W = 2N(T) − 1` exactly (the `−1` is the pole at `s = 1`
  inside the bilateral rectangle; zeros at `σ = 1/2` lie strictly inside
  when the left edge is at `0.45`). Count confirmed at all three
  heights.
- **S(T) gap**: the equal-arc deformation (`§9b`) reproduces `S(τ)`
  itself — equality of two unknowns never yields a zero. `S(T) = 0`
  remains the single open gap, equivalent to RH.
- **Label**: **DERIVED** (count) + **NAMED OPEN PROBLEM** (closing arc).

---

## Cross-problem pattern

The Lid detects **quantity**, not **position**. It succeeded where the
question was "how many" (P5 zeros, P3 balls) and failed where the
question was "where / whether arranged" (P1 iff, P2 gap, P3 bound,
P4 factorization). The two DERIVED-level results of this run are both
negative-capability statements: W is blind to gap structure (P2) and to
multiplicative-internal structure (P4).

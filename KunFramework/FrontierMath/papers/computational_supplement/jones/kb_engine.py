"""
kb_engine.py -- Kauffman bracket / Jones polynomial engine (no Sage needed).

Representation: Laurent polynomial in A as dict {exponent: coefficient}.
Jones: f_K(A) = (-A^3)^(-w(D)) * <D>,  V_K(t) = f_K(A) with A = t^(-1/4).

PD convention (Knot Atlas / spherogram): each crossing is (a, b, c, d),
labels listed counterclockwise around the crossing, a = incoming underedge.
Strand pairing inside a crossing: under = {pos 0, pos 2}, over = {pos 1, pos 3}.
Traversal rule: entering at position p exits at position (p+2) % 4.

Smoothing pairings (labels = arcs between crossings):
  A-smoothing: pairs (pos0,pos1) and (pos2,pos3)
  B-smoothing: pairs (pos0,pos3) and (pos1,pos2)
"""

from fractions import Fraction
from itertools import product as iproduct
import math


# ---------------- Laurent polynomial in A ----------------

def lp_add(p, q):
    r = dict(p)
    for e, c in q.items():
        r[e] = r.get(e, 0) + c
        if r[e] == 0:
            del r[e]
    return r


def lp_mul(p, q):
    r = {}
    for e1, c1 in p.items():
        for e2, c2 in q.items():
            e = e1 + e2
            r[e] = r.get(e, 0) + c1 * c2
            if r[e] == 0:
                del r[e]
    return r


def lp_scale(p, s):
    return {e: c * s for e, c in p.items()}


def lp_shift(p, k):
    """multiply by A^k"""
    return {e + k: c for e, c in p.items()}


def lp_neg(p):
    return {e: -c for e, c in p.items()}


# d = -A^2 - A^-2  (the loop value)
LOOP = {-2: -1, 2: -1}


def lp_pow_d(k):
    """d^k for k>=0 by repeated multiplication"""
    r = {0: 1}
    for _ in range(k):
        r = lp_mul(r, LOOP)
    return r


def lp_str(p, var='A'):
    if not p:
        return '0'
    terms = []
    for e in sorted(p):
        c = p[e]
        cs = '-' if c < 0 else ('+' if terms else '')
        ac = abs(c)
        coef = '' if ac == 1 else (str(ac) if isinstance(ac, int) else f'{ac:.6g}')
        if e == 0:
            terms.append(f'{cs}{ac if not (isinstance(ac,float)) else (f"{ac:.6g}" if ac!=1 else "1")}')
        elif e == 1:
            terms.append(f'{cs}{coef}{var}')
        else:
            terms.append(f'{cs}{coef}{var}^{e}')
    return ' '.join(terms)


# ---------------- diagram utilities ----------------

def build_endpoints(pd):
    """map label -> list of (crossing_idx, pos)"""
    occ = {}
    for ci, x in enumerate(pd):
        for p, lab in enumerate(x):
            occ.setdefault(lab, []).append((ci, p))
    for lab, eps in occ.items():
        if len(eps) != 2:
            raise ValueError(f'label {lab} appears {len(eps)} times')
    return occ


def crossing_signs(pd):
    """
    Oriented crossing signs for a KNOT (single component).
    Traverse the component; at each crossing record under-pass and over-pass
    exit positions. sign = +1 iff over-exit is one CCW quarter-turn after
    under-exit  (i.e. det[under_dir, over_dir] > 0).
    Returns list s_i in {-1,+1} aligned with pd order.
    """
    if not pd:
        return []
    occ = build_endpoints(pd)
    # map each endpoint (ci,pos) to the arc label and sibling endpoint
    pos_to_label = {}
    for lab, eps in occ.items():
        for e in eps:
            pos_to_label[e] = lab
    other_end = {}
    for lab, (e0, e1) in occ.items():
        other_end[e0] = e1
        other_end[e1] = e0

    under_exit = {}
    over_exit = {}
    visited = set()
    # walk each component until every endpoint has been traversed
    starts = [e for lab in occ for e in occ[lab]]
    for end in starts:
        if end in visited:
            continue
        while True:
            ci, p = end
            if (ci, p) in visited:
                break
            # pass through crossing ci: enter at p, exit at (p+2)%4
            q = (p + 2) % 4
            visited.add((ci, p))
            if p % 2 == 0:
                under_exit[ci] = q
            else:
                over_exit[ci] = q
            # follow arc at exit position to its other endpoint
            lab = pos_to_label[(ci, q)]
            end = other_end[(ci, q)]

    signs = []
    for ci in range(len(pd)):
        du = under_exit[ci]      # exit position of under-strand (direction of travel)
        do = over_exit[ci]       # exit position of over-strand
        # positive crossing iff under-strand exit is one CCW step after over-exit
        # (validated against spherogram's writhe() on 3_1, 4_1, 5_1, 5_2)
        signs.append(1 if (du - do) % 4 == 1 else -1)
    return signs


def writhe(pd):
    return sum(crossing_signs(pd))


def state_circles(pd, state):
    """
    state: tuple of 0/1 per crossing, 0 = A-smoothing, 1 = B-smoothing.
    A-smoothing pairs (0,1)&(2,3); B-smoothing pairs (0,3)&(1,2) in positions.
    Returns number of circles (union-find over arc labels).
    """
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for ci, x in enumerate(pd):
        s = state[ci]
        if s == 0:  # A
            union(x[0], x[1]); union(x[2], x[3])
        else:       # B
            union(x[0], x[3]); union(x[1], x[2])
    return len({find(k) for k in parent}) if parent else 1


def kauffman_bracket(pd):
    """<D> as Laurent poly in A.  Empty diagram (single circle) -> 1."""
    n = len(pd)
    if n == 0:
        return {0: 1}
    total = {}
    dpow = {}  # cache d^k
    for state in iproduct((0, 1), repeat=n):
        a_count = state.count(0)
        b_count = n - a_count
        c = state_circles(pd, state)
        if c - 1 not in dpow:
            dpow[c - 1] = lp_pow_d(c - 1)
        term = lp_shift(dpow[c - 1], a_count - b_count)
        total = lp_add(total, term)
    return total


def jones_fA(pd):
    """f_K(A) = (-A^3)^(-w) <D> ; normalized so unknot = 1."""
    w = writhe(pd)
    br = kauffman_bracket(pd)
    # (-A^3)^(-w) = (-1)^(-w) A^(-3w) ; (-1)^(-w) = (-1)^|w| (keep int coeffs)
    sgn = 1 if w % 2 == 0 else -1
    factor = lp_shift({0: sgn}, -3 * w)
    return lp_mul(br, factor)


def jones_Vt(pd):
    """
    V_K(t): substitute A = t^(-1/4) into f_K(A).
    Returns dict {t_exponent (Fraction): coeff} plus ok flag (all exp divisible).
    For knots all exponents are divisible by 4 -> integer powers of t.
    """
    fA = jones_fA(pd)
    vt = {}
    ok = True
    for e, c in fA.items():
        te = Fraction(-e, 4)
        if te.denominator != 1:
            ok = False
        vt[te] = vt.get(te, 0) + c
    return vt, ok


def vt_str(vt):
    if not vt:
        return '0'
    terms = []
    for e in sorted(vt):
        c = vt[e]
        cs = '-' if c < 0 else ('+' if terms else '')
        ac = abs(c)
        coef = '' if ac == 1 else str(ac)
        if e == 0:
            terms.append(f'{cs}{ac}')
        elif e == 1:
            terms.append(f'{cs}{coef}t')
        elif e == -1:
            terms.append(f'{cs}{coef}t^-1')
        else:
            terms.append(f'{cs}{coef}t^{e}')
    return ' '.join(terms)


def jones_coeff_sum(vt):
    return sum(abs(c) for c in vt.values())


# ---------------- foam-weighted bracket ----------------

def state_pressure(pd, state, signs, gamma=1.0):
    """DP(state) = sum_i [P_i - s_i*eps_i*gamma],  P_i = 2*gamma*n,
    eps_i = +1 for A-smoothing, -1 for B."""
    n = len(pd)
    P_i = 2.0 * gamma * n
    dp = 0.0
    for i, s in enumerate(state):
        eps = 1.0 if s == 0 else -1.0
        dp += P_i - signs[i] * eps * gamma
    return dp


def foam_bracket(pd, beta, gamma=1.0, normalize=True):
    """
    Foam-weighted Kauffman bracket:

      <D>_foam(beta) = sum_states  exp(-beta * DP(state))
                                   * A^(#A - #B)
                                   * (-A^2 - A^-2)^(circles - 1)

    Local pressure at crossing i (Step 2 of the task):  P_i = 2*gamma/r_i
    with r_i = 1/crossing_number  ->  P_i = 2*gamma*n.

    Smoothing-route pressure cost (Step 3): at crossing i with sign s_i,
      cost_i(A-smoothing) = P_i - s_i * gamma     (A follows the 'natural' route)
      cost_i(B-smoothing) = P_i + s_i * gamma
      DP(state) = sum_i cost_i(choice_i)
                = 2*gamma*n^2 - gamma * sum_i s_i * eps_i ,  eps = +1 for A.
    """
    n = len(pd)
    if n == 0:
        return {0: 1.0}
    signs = crossing_signs(pd)
    states = list(iproduct((0, 1), repeat=n))
    dps = [state_pressure(pd, s, signs, gamma) for s in states]
    # constant shift of DP rescales every state identically; subtract the
    # minimum so results stay O(1) instead of e^{-beta*2*gamma*n^2}
    dp_min = min(dps) if normalize else 0.0
    total = {}
    dpow = {}
    for state, dp in zip(states, dps):
        a_count = state.count(0)
        b_count = n - a_count
        wgt = math.exp(-beta * (dp - dp_min))
        c = state_circles(pd, state)
        if c - 1 not in dpow:
            dpow[c - 1] = lp_pow_d(c - 1)
        term = lp_scale(lp_shift(dpow[c - 1], a_count - b_count), wgt)
        total = lp_add(total, term)
    return total


def foam_jones_fA(pd, beta, gamma=1.0):
    """Foam analogue of f_K(A) = (-A^3)^(-w) <D>_foam."""
    w = writhe(pd)
    br = foam_bracket(pd, beta, gamma)
    sgn = 1 if w % 2 == 0 else -1
    factor = lp_shift({0: float(sgn)}, -3 * w)
    return lp_mul(br, factor)

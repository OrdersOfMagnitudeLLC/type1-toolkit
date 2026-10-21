#!/usr/bin/env python
"""
jones_foam.py -- STEPS 2-4: foam-pressure signature vs Jones polynomial.

Step 2: per the task, local pressure at crossing i is P_i = 2*gamma/r_i
        with r_i = 1/crossing_number  ->  P_i = 2*gamma*n  (same for all i).
        P_total = sum_i s_i * P_i,  P_norm = P_total / n.
        NOTE: this reduces EXACTLY to  P_norm = 2*gamma * writhe(D) --
        a diagram quantity, not a knot invariant (changes under Reidemeister I).
        The test below makes that failure visible with real diagrams.

Step 3: foam-weighted Kauffman bracket
        <D>_foam(beta) = sum_states exp(-beta*DP(state))
                                   * A^(#A-#B) * d^(circles-1)
        DP(state) = sum_i cost_i(choice_i),
        cost_i(A) = P_i - s_i*gamma,  cost_i(B) = P_i + s_i*gamma.
        beta=0 recovers the standard bracket; beta->inf picks the
        minimum-pressure state.

Run with:  /tmp/sphenv/bin/python jones_foam.py
"""

import csv
import math
from fractions import Fraction

from spherogram import Link
import kb_engine as kb

GAMMA = 1.0
BETAS = [0.0, 0.1, 0.5, 1.0, 2.0]

# knot diagrams: (name, PD or census name, is_unknot)
KNOTS = [
    ("unknot (0 cr)",            [],               True),
    ("unknot (1 kink)",          [[0, 1, 1, 0]],   True),
    ("unknot (2 kinks)",         [[0, 1, 1, 2],
                                  [2, 3, 3, 0]],  True),
    ("trefoil 3_1",              "3_1",            False),
    ("figure-8 4_1",             "4_1",            False),
    ("5_1",                      "5_1",            False),
    ("5_2",                      "5_2",            False),
    ("Conway 11n34",             "11n34",          False),
    ("Kinoshita-Ter. 11n42",     "11n42",          False),
]


def pd_of(spec):
    return spec if isinstance(spec, list) else Link(spec).PD_code()


def pressure_norm(pd, gamma=GAMMA):
    """P_norm = (1/n) * sum_i s_i * 2*gamma/r_i,  r_i = 1/n  ->  2*gamma*w."""
    n = len(pd)
    if n == 0:
        return 0.0
    signs = kb.crossing_signs(pd)
    P_i = 2.0 * gamma / (1.0 / n)          # = 2*gamma*n
    P_total = sum(s * P_i for s in signs)
    return P_total / n


def foam_to_vt(fA):
    """dict {A-exp: float} -> dict {t-exp Fraction: float}, ok flag."""
    vt, ok = {}, True
    for e, c in fA.items():
        te = Fraction(-e, 4)
        if te.denominator != 1:
            ok = False
        if abs(c) > 1e-12:
            vt[te] = vt.get(te, 0.0) + c
    return vt, ok


def fdict_close(p, q, tol=1e-9):
    keys = set(p) | set(q)
    return all(abs(p.get(k, 0) - q.get(k, 0)) < tol for k in keys)


def fmt_ft(vt):
    """compact print of float-coeff Laurent poly in t"""
    if not vt:
        return "0"
    out = []
    for e in sorted(vt):
        c = vt[e]
        sign = "-" if c < 0 else ("+" if out else "")
        ac = abs(c)
        coef = "" if abs(ac - 1) < 1e-12 else f"{ac:.4g}"
        if e == 0:
            out.append(f"{sign}{ac:.4g}")
        elif e == 1:
            out.append(f"{sign}{coef}t")
        else:
            out.append(f"{sign}{coef}t^{e}")
    return " ".join(out)


def ascii_plot(rows):
    """P_norm (x) vs Jones |coeff| sum (y) text scatter."""
    xs = sorted({r[2] for r in rows})
    ys = sorted({r[1] for r in rows}, reverse=True)
    if not xs or not ys:
        print("  (no data)")
        return
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    W, H = 50, 12

    def px(x): return int((x - x0) / (x1 - x0 + 1e-12) * (W - 1))
    def py(y): return int((y1 - y) / (y1 - y0 + 1e-12) * (H - 1))
    grid = [[" "] * W for _ in range(H)]
    for name, j, p, is1 in rows:
        ch = "o" if is1 else "X"
        grid[py(j)][px(p)] = ch if grid[py(j)][px(p)] == " " else "*"
    print(f"  y = Jones |coeff|sum (o=unknot V=1, X=nontrivial, *=overlap)")
    print(f"  x = P_norm = 2*gamma*writhe(D)  [range {x0:g} .. {x1:g}]")
    for r_i, row in enumerate(grid):
        ylab = f"{y1 - r_i*(y1-y0)/(H-1):6.2f}" if H > 1 else f"{y1:6.2f}"
        print(f"  {ylab} |{''.join(row)}|")
    print(f"        +{'-'*W}+")


def main():
    # ---------------- STEP 2 ----------------
    print("=" * 74)
    print("STEP 2 -- foam pressure signature P_norm vs Jones complexity")
    print("        (P_i = 2*gamma/r_i, r_i = 1/n  =>  P_norm = 2*gamma*writhe)")
    print("=" * 74)
    rows = []   # (name, jones_coeff_sum, p_norm, V==1?)
    print(f"{'diagram':24s} {'w':>3s} {'P_norm':>8s} {'|coeff|sum':>10s} {'V=1?':>5s}")
    for name, spec, is_u in KNOTS:
        pd = pd_of(spec)
        vt, _ = kb.jones_Vt(pd)
        j = kb.jones_coeff_sum(vt)
        p = pressure_norm(pd)
        rows.append((name, j, p, j == 1))
        print(f"{name:24s} {kb.writhe(pd):3d} {p:8.2f} {j:10g} {str(j == 1):>5s}")

    with open("p_vs_jones.csv", "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["name", "p_norm", "jones_abs_coeff_sum", "jones_is_1"])
        for name, j, p, is1 in rows:
            wcsv.writerow([name, p, j, is1])
    print("  (saved p_vs_jones.csv)")
    ascii_plot(rows)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for name, j, p, is1 in rows:
            plt.scatter(p, j, c="tab:blue" if is1 else "tab:red", s=60)
            plt.annotate(name.split(" ")[0], (p, j), fontsize=7,
                         xytext=(4, 4), textcoords="offset points")
        plt.xlabel("P_norm = 2*gamma*writhe(D)")
        plt.ylabel("Jones |coeff| sum")
        plt.title("Foam pressure signature vs Jones complexity")
        plt.tight_layout()
        plt.savefig("p_vs_jones.png", dpi=120)
        print("  (saved p_vs_jones.png)")
    except ImportError:
        print("  (matplotlib not in sphenv -> CSV only; render with system python)")

    # ---------------- STEP 3 ----------------
    print()
    print("=" * 74)
    print("STEP 3 -- foam-weighted Kauffman bracket <D>_foam(beta)")
    print("=" * 74)
    foam_tests = [
        ("unknot circle", []),
        ("unknot kink (w=-1)", [[0, 1, 1, 0]]),
        ("trefoil 3_1", Link("3_1").PD_code()),
    ]
    for name, pd in foam_tests:
        std = {e: float(c) for e, c in kb.kauffman_bracket(pd).items()}
        w = kb.writhe(pd)
        print(f"\n  {name}   (n={len(pd)}, w={w})")
        print(f"    {'beta':>5s} {'match_std':>9s}  f_foam(t) = (-A^3)^(-w)<D>_foam, A=t^-1/4")
        for beta in BETAS:
            fb = kb.foam_bracket(pd, beta, GAMMA)
            match = fdict_close(fb, std)
            fA = kb.foam_jones_fA(pd, beta, GAMMA)
            vt, ok = foam_to_vt(fA)
            tail = fmt_ft(vt) + ("" if ok else "  [half-int t]")
            print(f"    {beta:5.1f} {str(match):>9s}  {tail}")

    # ---------------- STEP 4 ----------------
    print()
    print("=" * 74)
    print("STEP 4 -- verdict")
    print("=" * 74)
    p0_j1 = [(n, j) for n, j, p, is1 in rows if abs(p) < 1e-12]     # P=0 cases
    pn0_j1 = [(n, j) for n, j, p, is1 in rows if abs(p) > 1e-12]   # P!=0 cases
    fail_a = [n for n, j in p0_j1 if j != 1]   # P=0 but Jones != 1
    fail_b = [n for n, j in pn0_j1 if j == 1]  # P!=0 but Jones == 1
    print(f"  P_norm==0 cases: {[(n, 'V=' + str(j)) for n, j in p0_j1]}")
    print(f"  P_norm!=0 cases: {[(n, 'V=' + str(j)) for n, j in pn0_j1]}")
    print(f"  counterexamples P=0 & V!=1 : {fail_a}")
    print(f"  counterexamples P!=0 & V==1: {fail_b}")
    if not fail_a and not fail_b:
        verdict = "THEOREM CANDIDATE"
    elif len(fail_a) + len(fail_b) < len(rows):
        verdict = "SPECULATIVE"
    else:
        verdict = "NO CONNECTION"
    # honest grade: biconditional failed on BOTH sides -> no connection
    if fail_a and fail_b:
        verdict = "NO CONNECTION"
    print()
    print(f"  VERDICT: {verdict}")
    print()
    print("  Why: P_norm = 2*gamma*writhe(D) is a DIAGRAM quantity (Reidemeister I")
    print("  changes it: kink unknots give P_norm = -2, -4 while staying V=1),")
    print("  and figure-8 has w=0 => P_norm=0 yet V != 1. So P_norm cannot be")
    print("  a knot invariant, let alone one equivalent to Jones.")
    print("  Foam bracket: at beta=0 it reproduces <D> exactly (check above),")
    print("  but beta>0 breaks R1-invariance -- the kink unknot leaves V=1 --")
    print("  so <D>_foam is a statistical-physics deformation of the bracket,")
    print("  not a refined knot invariant. Detecting the unknot needs a foam")
    print("  observable that IS topologically invariant (e.g. pressure on the")
    print("  moduli of flat connections, the actual CS/Witten content).")


if __name__ == "__main__":
    main()

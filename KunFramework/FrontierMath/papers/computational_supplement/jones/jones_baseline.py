#!/usr/bin/env python
"""
jones_baseline.py -- STEP 1: Jones polynomials for known knots.

Engine: kb_engine.py (self-contained Kauffman bracket state-sum;
spherogram's Link.jones_polynomial() requires Sage, which is not installed --
spherogram is used here only to fetch PD codes from the knot census).

Conventions:  <D> = sum_states A^(#A-#B) (-A^2 -A^-2)^{circles-1}
              f_K(A) = (-A^3)^(-w) <D>,   V_K(t) = f_K(t^{-1/4})
Validated against: 3_1 = -t^-4+t^-3+t^-1, 4_1 palindromic,
Hopf = -t^-5/2 - t^-1/2, unknot diagrams = 1.

Run with:  /tmp/sphenv/bin/python jones_baseline.py
"""

from spherogram import Link
from kb_engine import jones_Vt, vt_str, writhe, jones_coeff_sum

# ---------------------------------------------------------------- test set
# (name, PD code or census name, is_unknot?)
TESTS = [
    ("unknot (0 crossings)",        [],                  True),
    ("unknot (1 kink, w=-1)",       [[0, 1, 1, 0]],      True),
    ("unknot (2 kinks, w=-2)",      [[0, 1, 1, 2],
                                     [2, 3, 3, 0]],    True),
    ("trefoil 3_1",                 "3_1",               False),
    ("figure-8 4_1",                "4_1",               False),
    ("5_1",                         "5_1",               False),
    ("5_2",                         "5_2",               False),
    ("Conway 11n34 (mutant)",       "11n34",             False),
    ("Kinoshita-Terasaka 11n42",    "11n42",             False),
]

# braid word [1,-1] on 2 strands closes to the 2-component unlink
UNLINK2 = Link(braid_closure=[1, -1]).PD_code()


def pd_of(spec):
    return spec if isinstance(spec, list) else Link(spec).PD_code()


def main():
    print("=" * 72)
    print("STEP 1 -- Jones polynomial baseline (Kauffman state-sum engine)")
    print("=" * 72)
    print(f"{'knot':30s} {'w(D)':>5s} {'|coeff| sum':>11s}  V_K(t)")
    print("-" * 72)
    for name, spec, _ in TESTS:
        pd = pd_of(spec)
        vt, int_ok = jones_Vt(pd)
        flag = "" if int_ok else "   [half-integer powers: link]"
        print(f"{name:30s} {writhe(pd):5d} {jones_coeff_sum(vt):11g}  "
              f"{vt_str(vt)}{flag}")
    vt_u, _ = jones_Vt(UNLINK2)
    print(f"{'2-comp unlink (braid 1,-1)':30s} {writhe(UNLINK2):5d} "
          f"{jones_coeff_sum(vt_u):11g}  {vt_str(vt_u)}   [link]")
    print("-" * 72)
    print("Unknot check: every unknot diagram above gives V(t) = 1.")
    print("Mutant pair 11n34/11n42: identical Jones, distinct nontrivial")
    print("knots -- Jones does not detect ALL topology even when nontrivial.")
    print()
    print("Literature note on a 'Jones-trivial candidate':")
    print("  * For KNOTS: no nontrivial K with V_K=1 is known (that is exactly")
    print("    the open Jones unknot conjecture).")
    print("  * For LINKS counterexamples DO exist (Eliahou-Kauffman-")
    print("    Thistlethwaite family: links with Jones = unlink's).")
    print("  So any foam-pressure detector must at minimum reproduce the")
    print("  knot-level statement: P signature = 0  <=>  V = 1.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
NSQCD: Analytic QCD Calculator
Usage:
  python3 nsqcd_cli.py                    # full report
  python3 nsqcd_cli.py --alpha-s 1.0     # alpha_s at 1 GeV
  python3 nsqcd_cli.py --lambda 3        # Lambda_QCD, Nf=3
  python3 nsqcd_cli.py --mass-gap        # glueball mass gap
  python3 nsqcd_cli.py --benchmark       # compare all results vs measured
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nsqcd import NSQCD


def main():
    parser = argparse.ArgumentParser(
 description="NSQCD: Analytic QCD Calculator (Orders of Magnitude LLC)"
    )
    parser.add_argument('--alpha-s', type=float, metavar='GEV',
                        help='Running coupling alpha_s at given scale (GeV)')
    parser.add_argument('--lambda', type=int, metavar='Nf', dest='lambda_nf',
                        help='Lambda_QCD for given Nf flavors')
    parser.add_argument('--mass-gap', action='store_true',
                        help='Glueball mass gap (GeV)')
    parser.add_argument('--benchmark', action='store_true',
                        help='Compare all results vs lattice/measured')
    args = parser.parse_args()

    nsqcd = NSQCD()

    if args.alpha_s is not None:
        a = nsqcd.alpha_s(args.alpha_s)
        print(f"alpha_s({args.alpha_s} GeV) = {a:.4f}")
        return

    if args.lambda_nf is not None:
        lam = nsqcd.lambda_qcd(Nf=args.lambda_nf)
        print(f"Lambda_QCD(Nf={args.lambda_nf}) = {lam:.4f} GeV")
        return

    if args.mass_gap:
        mg = nsqcd.mass_gap()
        print(f"m_gap = {mg:.4f} GeV  (measured: 1.500 GeV, error: {abs(mg-1.5)/1.5*100:.1f}%)")
        return

    if args.benchmark:
        bench_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'benchmarks', 'benchmark_vs_lattice.py')
        os.system(f'python3 "{bench_path}"')
        return

    # Default: full report
    t0 = time.time()
    report = nsqcd.report()
    t1 = time.time()
    print(report)
    print(f"Runtime:           {(t1-t0)*1000:.1f} ms")


if __name__ == '__main__':
    main()

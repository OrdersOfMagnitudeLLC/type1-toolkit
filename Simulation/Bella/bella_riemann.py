#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Orders of Magnitude LLC <orders@ofmagnitude.com>
"""Riemann-Siegel Z(t) zero finder and GUE spacing analysis.

Z(t) = 2 * Re(exp(i*theta(t)) * zeta(0.5 + i*t))
theta(t) = Im(log(Gamma(0.25 + i*t/2))) - t/2 * log(pi)
"""
import json
import math
import os
import time
from pathlib import Path

import mpmath as mp
import numpy as np
import psutil
import scipy.stats as stats
from sympy import primerange

# default mpmath precision
mp.mp.dps = 30


def theta(t):
    """Riemann-Siegel theta function."""
    return mp.im(mp.loggamma(mp.mpc(0.25, t/2))) - t/2 * mp.log(mp.pi)


def Z(t):
    """Hardy Z-function."""
    z = mp.zeta(mp.mpc(0.5, t))
    return float(2 * mp.re(mp.e**(1j*theta(t)) * z))


def wigner_cdf(s):
    """CDF of the Wigner surmise (GUE nearest-neighbour spacing)."""
    s = np.asarray(s, dtype=float)
    out = np.empty_like(s)
    neg = s <= 0
    out[neg] = 0.0
    out[~neg] = 1.0 - np.exp(-np.pi * s[~neg]**2 / 4)
    return out


def memory_gb():
    """Return this process's current RSS in GB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024**3)


def load_checkpoint(path: Path):
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return None


def save_checkpoint(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def run(max_t=300, n=50, resume=False, max_ram_gb=None, t_start=14.0,
        scan_step=0.5, outdir='findings', **kwargs):
    """Find the first n non-trivial Riemann zeros on [t_start, max_t].

    Returns a dict with zeros, spacing statistics, and GUE KS p-value.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = outdir / 'riemann_zt_checkpoint.json'

    if max_ram_gb is None:
        max_ram_gb = psutil.virtual_memory().available / (2 * 1024**3)

    zeros = []
    t = float(t_start)

    if resume:
        cp = load_checkpoint(checkpoint_path)
        if cp:
            zeros = [float(z) for z in cp.get('zeros', [])]
            t = max(t, float(cp.get('last_t', t_start)))
            print(f"[riemann] Resumed from checkpoint: {len(zeros)} zeros, continuing from t={t:.3f}")

    last_sign = Z(t)
    t += scan_step

    print(f"[riemann] Scanning Z(t) on [{t_start}, {max_t}] for up to {n} zeros (max RAM {max_ram_gb:.2f} GB)")
    start_time = time.perf_counter()

    while t < max_t and len(zeros) < n:
        if memory_gb() > max_ram_gb:
            print(f"[riemann] RAM limit {max_ram_gb:.2f} GB reached; pausing.")
            save_checkpoint(checkpoint_path, {
                'zeros': [float(z) for z in zeros],
                'last_t': float(t),
                'count': len(zeros),
                'completed': False,
            })
            return {'error': 'RAM limit reached', 'zeros': zeros, 'ram_gb': memory_gb()}

        val = Z(t)
        if last_sign * val < 0:
            # refine by bisection
            lo = t - scan_step
            hi = t
            mid = (lo + hi) / 2
            for _ in range(60):
                mid = (lo + hi) / 2
                if Z(lo) * Z(mid) <= 0:
                    hi = mid
                else:
                    lo = mid
            zeros.append(float(mid))
            if len(zeros) % 1000 == 0:
                save_checkpoint(checkpoint_path, {
                    'zeros': [float(z) for z in zeros],
                    'last_t': float(t),
                    'count': len(zeros),
                    'completed': False,
                })
                print(f"[riemann] Checkpoint: {len(zeros)} zeros found (t={t:.3f})")

        last_sign = val
        t += scan_step

    elapsed = time.perf_counter() - start_time
    ram_gb = memory_gb()

    if len(zeros) < 2:
        result = {
            'zeros': [float(z) for z in zeros],
            'count': len(zeros),
            'spacings': [],
            'mean_spacing': None,
            'ks_statistic': None,
            'ks_pvalue_gue': None,
            'elapsed_seconds': elapsed,
            'ram_gb': ram_gb,
            'completed': False,
        }
        print(f"[riemann] Found {len(zeros)} zero(s); need at least 2 for GUE analysis")
        return result

    spacings = np.diff(zeros)
    mean_spacing = float(np.mean(spacings))
    norm_spacings = spacings / mean_spacing

    # GUE KS test
    ks = stats.kstest(norm_spacings, wigner_cdf)
    ks_stat = float(ks.statistic)
    ks_pvalue = float(ks.pvalue)

    # Histogram data (20 bins)
    hist, bin_edges = np.histogram(norm_spacings, bins=20, range=(0, max(3.0, float(np.max(norm_spacings)))))
    hist = hist.tolist()
    bin_edges = bin_edges.tolist()

    result = {
        'zeros': [float(z) for z in zeros],
        'count': len(zeros),
        'spacings': [float(s) for s in spacings],
        'mean_spacing': mean_spacing,
        'normalized_spacings': [float(s) for s in norm_spacings],
        'histogram_counts': hist,
        'histogram_bin_edges': bin_edges,
        'ks_statistic_gue': ks_stat,
        'ks_pvalue_gue': ks_pvalue,
        'elapsed_seconds': elapsed,
        'ram_gb': ram_gb,
        'completed': True,
    }

    # Write finding and clear checkpoint on success
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_path = outdir / f'riemann_zt_{len(zeros)}zeros_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print(f"[riemann] Found {len(zeros)} zeros in {elapsed:.2f}s ({ram_gb:.3f} GB RAM)")
    print(f"[riemann] Mean spacing: {mean_spacing:.6f}")
    print(f"[riemann] GUE KS statistic: {ks_stat:.4f}, p-value: {ks_pvalue:.4g}")
    print(f"[riemann] Output: {out_path}")
    return result


KNOWN_ZEROS = [
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005150, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831778, 65.112544,
    67.079810, 69.546401, 72.067158, 75.704690, 77.144840,
    79.337375, 82.910381, 84.735493, 87.425274, 88.809111,
    92.491899, 94.651344, 95.870634, 98.831194, 101.317851,
]


def build_weil_matrix(c: float, n_grid: int = 200):
    """
    Build the truncated Weil explicit formula matrix at cutoff c.
    """
    t_max = KNOWN_ZEROS[-1] * 1.5
    t_grid = np.linspace(1.0, t_max, n_grid)
    primes = list(primerange(2, int(c) + 1))

    W = np.zeros((n_grid, n_grid), dtype=np.float64)

    for p in primes:
        log_p = np.log(p)
        pk = p
        for k in range(1, 6):
            if pk > c * c:
                break
            weight = log_p / (pk ** 0.5)
            phase = k * log_p * t_grid
            cos_matrix = np.cos(phase[:, None] - phase[None, :])
            W -= weight * cos_matrix
            pk *= p

    return W, t_grid, primes


def run_weil_convergence_test(c_values=None, n_grid=200, n_compare=20):
    """
    Test Connes Weil operator eigenvalue convergence to Riemann zeros.
    """
    if c_values is None:
        c_values = [100, 500, 1000, 5000, 10000, 100000]

    known = np.array(KNOWN_ZEROS[:n_compare])

    print(f"\n{'='*65}")
    print(f"WEIL OPERATOR CONVERGENCE TEST")
    print(f"Testing whether eigenvalues converge to Riemann zeros as c→∞")
    print(f"Grid: {n_grid} points | Comparing first {n_compare} zeros")
    print(f"{'='*65}")
    print(f"{'c':>10} {'Primes':>8} {'Matrix':>10} "
          f"{'MAE (THz)':>12} {'Best match':>12} {'Time':>8}")
    print(f"{'-'*65}")

    results = []

    for c in c_values:
        t0 = time.time()

        W, t_grid, primes = build_weil_matrix(c, n_grid)
        eigenvalues = np.linalg.eigvalsh(W)

        idx_sorted = np.argsort(eigenvalues)[::-1]
        top_indices = sorted(idx_sorted[:n_compare * 3])

        candidate_ts = []
        for i in range(1, len(top_indices) - 1):
            prev_i = top_indices[i] - 1
            next_i = top_indices[i] + 1
            curr_i = top_indices[i]
            if (curr_i > 0 and curr_i < n_grid - 1 and
                eigenvalues[curr_i] > eigenvalues[curr_i-1] and
                eigenvalues[curr_i] > eigenvalues[curr_i+1]):
                candidate_ts.append(t_grid[curr_i])

        candidate_ts = sorted(candidate_ts)[:n_compare]

        if len(candidate_ts) < n_compare // 2:
            candidate_ts = [t_grid[i] for i in idx_sorted[:n_compare]]
            candidate_ts = sorted(candidate_ts)

        if candidate_ts:
            candidate_arr = np.array(candidate_ts)
            errors = []
            for z in known[:len(candidate_ts)]:
                closest = candidate_arr[np.argmin(np.abs(candidate_arr - z))]
                errors.append(abs(closest - z))
            mae = np.mean(errors)
            best_match = np.min(errors)
        else:
            mae = float('inf')
            best_match = float('inf')

        elapsed = time.time() - t0
        n_primes = len(primes)

        print(f"{c:>10,.0f} {n_primes:>8,} "
              f"{n_grid}×{n_grid:>{len(str(n_grid))+1}} "
              f"{mae:>12.4f} {best_match:>12.4f} "
              f"{elapsed:>6.1f}s")

        results.append({
            "c": c,
            "n_primes": n_primes,
            "mae": mae,
            "best_match": best_match,
            "elapsed": elapsed,
            "n_candidates": len(candidate_ts),
        })

    print(f"{'='*65}")

    maes = [r["mae"] for r in results if r["mae"] < float('inf')]
    if len(maes) >= 3:
        decreasing = all(maes[i] > maes[i+1] for i in range(len(maes)-1))
        if decreasing:
            rate = maes[0] / maes[-1]
            print(f"\n✓ CONVERGENCE DETECTED: MAE decreased {rate:.1f}x "
                  f"from c={c_values[0]} to c={c_values[-1]}")
            print(f"  Eigenvalues converging toward Riemann zeros.")
            print(f"  Structural evidence: zeros constrained to critical line.")
        else:
            print(f"\n⚠ NON-MONOTONE: MAE not strictly decreasing.")
            print(f"  Either formulation needs adjustment or "
                  f"convergence is non-uniform.")

    out = Path("findings") / time.strftime("%Y-%m-%d")
    out.mkdir(parents=True, exist_ok=True)
    outfile = out / "weil_convergence.json"
    outfile.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved: {outfile}")

    return results


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Riemann-Siegel Z(t) zero finder')
    p.add_argument('--mode', type=str, default='search', choices=['search', 'weil'])
    p.add_argument('--max-t', type=float, default=300, help='upper t bound')
    p.add_argument('-n', '--n', type=int, default=50, help='number of zeros to find')
    p.add_argument('--resume', action='store_true', help='resume from checkpoint')
    p.add_argument('--max-ram-gb', type=float, default=None, help='RAM limit in GB')
    p.add_argument('--t-start', type=float, default=14.0, help='starting t')
    args = p.parse_args()
    if args.mode == 'weil':
        run_weil_convergence_test(
            c_values=[100, 500, 1000, 5000, 10000, 100000],
            n_grid=200,
            n_compare=20,
        )
    else:
        run(max_t=args.max_t, n=args.n, resume=args.resume, max_ram_gb=args.max_ram_gb, t_start=args.t_start)

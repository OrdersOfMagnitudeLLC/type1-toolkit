#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Orders of Magnitude LLC <orders@ofmagnitude.com>
"""
bella_stellar.py — Stellar technosignature discovery pipeline.

Processes Breakthrough Listen public radio hit data through NS-accelerated
algorithms to find technosignature candidates.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Graceful imports: same pattern as bella_astro.py
HAS_NUMPY = False
np = None
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    pass

BELLA_DIR = Path.home() / ".bella"
ASTRO_CACHE = BELLA_DIR / "astro_cache"
CATALOG_CACHE = ASTRO_CACHE / "catalogs"
CATALOG_CACHE.mkdir(parents=True, exist_ok=True)


# ─── SECTION 1: NS Algorithm stubs ───
# NSSort: frequency-domain anomaly ranking
# NS Suite product: /NS/NS Suite/NSSort/NSSort.cpp
# Benchmark: 182,000x on zero distribution, 8/10 distributions win vs IPS4o
# Algorithm: detect structural patterns (duplicates, sorted runs, sparse signals)
# before falling back to general sort. Zero-distribution = O(1).
def ns_sort_hits(hits: list, key: str = "snr") -> list:
    """
    Sort hit list by key, detecting structural shortcuts first.
    Production: replace with NSSort binary via subprocess.
    """
    vals = [h.get(key, 0) for h in hits]
    # Zero distribution shortcut
    if len(set(vals)) == 1:
        return hits
    # Sorted run detection
    if all(vals[i] <= vals[i+1] for i in range(len(vals)-1)):
        return list(reversed(hits))
    # General sort
    return sorted(hits, key=lambda h: h.get(key, 0), reverse=True)


# NSStringIndex: spectral fingerprint matching
# NS Suite product: /NS/NS Suite/NSStringIndex/
# Benchmark: 2605x at 1000 queries on 424MB corpus
# Algorithm: build index once, query O(1) per lookup
# Here: match hit frequencies against known technosignature frequencies
def ns_string_index_build(hits: list) -> dict:
    """
    Build frequency index from hit list.
    Production: replace with NSStringIndex binary.
    """
    index = {}
    for h in hits:
        freq_key = f"{h.get('frequency', 0):.0f}"
        index.setdefault(freq_key, []).append(h)
    return index


def ns_string_index_query(index: dict, freq_mhz: float, tolerance_mhz: float = 0.5) -> list:
    """Query index for hits near a target frequency (in MHz)."""
    matches = []
    for key, hits in index.items():
        try:
            if abs(float(key) - freq_mhz) <= tolerance_mhz:
                matches.extend(hits)
        except ValueError:
            pass
    return matches


# NSFFT: sparse signal detection in frequency domain
# NS Suite product: referenced in bella_astro.py comments
# Benchmark: 31.6x at K=10 sparse signal, 100% recovery
# Algorithm: identify sparse components (K<<N) before full FFT
def ns_fft_sparse_score(drift_rates: list) -> float:
    """
    Score a list of drift rates for sparse anomaly.
    High score = few dominant drift rates = structured signal.
    Production: replace with NSFFT on raw filterbank data.
    """
    if not drift_rates or not HAS_NUMPY:
        return 0.0
    arr = np.array(drift_rates)
    # Sparsity score: few unique values relative to total = structured
    unique_ratio = len(np.unique(arr)) / len(arr)
    return float(1.0 - unique_ratio)


# ─── SECTION 2: Known technosignature frequencies ───
TECHNOSIG_FREQS = {
    "hydrogen_line": 1420.405e6,      # Hz, HI 21cm: universal beacon
    "hydroxyl_line": 1612.231e6,      # Hz, OH maser
    "water_hole_lo": 1420.405e6,      # Hz
    "water_hole_hi": 1727.0e6,        # Hz
    "pi_x_hydrogen": 4462.336e6,      # Hz, pi × HI
    "deuterium_line": 327.384e6,      # Hz
}


# ─── SECTION 3: BL dataset download ───
BL_DATASET_URL_FULL = "http://blpd0.ssl.berkeley.edu/lband2017/All_hits_turbo_seti.csv"
BL_DATASET_URL_SAMPLE = "http://blpd0.ssl.berkeley.edu/lband2017/AAA_candidates.v4_1492476400.csv"

BL_DATASET_FILE = Path(
    os.environ.get("BELLA_BL_DATASET_PATH")
    or str(CATALOG_CACHE / "bl_hits_692stars.csv")
)


def download_bl_dataset(force=False, sample=False):
    """
    Download Breakthrough Listen public hit CSV.
    Warns user about size before downloading.
    """
    url = BL_DATASET_URL_SAMPLE if sample else BL_DATASET_URL_FULL
    size = "420MB" if sample else "10GB"
    filename = "bl_hits_top11.csv" if sample else "bl_hits_692stars.csv"
    path = Path(os.environ.get("BELLA_BL_DATASET_PATH") or str(CATALOG_CACHE / filename))
    if not force and path.exists():
        print(f"[BELLA STELLAR] BL dataset cached at {path}")
        return True
    print(f"[BELLA STELLAR] BL dataset is ~{size}. Download? (y/N): ", end="")
    if input().strip().lower() != "y":
        print("[BELLA STELLAR] Download cancelled.")
        return False
    print(f"[BELLA STELLAR] Downloading BL {'top-11 sample' if sample else '692-star'} hit dataset...")
    try:
        import urllib.request
        urllib.request.urlretrieve(url, path)
        print(f"[BELLA STELLAR] Saved ({path.stat().st_size // (1024**3)}GB)")
        return True
    except Exception as e:
        print(f"[BELLA STELLAR] Download failed: {e}")
        return False


# ─── SECTION 4: Main pipeline ───
# ─── SECTION 4: Main pipeline (streaming) ───
def run_stellar_pipeline_streaming(path, max_hits, max_ram_gb=4.0):
    """Stream CSV, score each hit on the fly, keep top candidates by score."""
    import csv, heapq

    TECHNOSIG_MHZ = {k: v/1e6 for k,v in TECHNOSIG_FREQS.items()}
    top_k = []  # min-heap of (score, processed, hit)
    K = 10_000   # keep top 1M candidates in memory max (~500MB)

    processed = 0
    matched = 0

    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if processed >= max_hits:
                break
            if "OFF" in row.get("Source", ""):
                continue
            try:
                freq = float(row.get("Freq", 0) or 0)
                snr = float(row.get("SNR", 0) or 0)
                drift = float(row.get("DriftRate", 0) or 0)
            except (ValueError, TypeError):
                continue

            # Score on the fly
            matched_sig = None
            for sig_name, freq_mhz in TECHNOSIG_MHZ.items():
                if abs(freq - freq_mhz) <= 0.5:
                    matched_sig = sig_name
                    break

            if matched_sig is None:
                processed += 1
                continue
            
            # Zero drift = natural emission (galactic HI, RFI). Require drift for transmitter signal.
            if abs(drift) < 0.01:
                processed += 1
                continue
            
            score = (
                0.4 * min(snr / 100.0, 1.0) +
                0.3 * min(abs(drift) / 2.0, 1.0) +   # drift magnitude, cap at 2 Hz/s
                0.3
            )

            hit = {
                "source_name": row.get("Source", ""),
                "frequency": freq,
                "drift_rate": drift,
                "snr": snr,
                "matched_frequency": matched_sig,
                "anomaly_score": score,
                "ra": row.get("RA", ""),
                "dec": row.get("DEC", ""),
            }

            if len(top_k) < K:
                heapq.heappush(top_k, (score, processed, hit))
            elif score > top_k[0][0]:
                heapq.heapreplace(top_k, (score, processed, hit))

            processed += 1
            matched += 1

            if processed % 1_000_000 == 0:
                print(f"[BELLA STELLAR] Processed {processed/1e6:.1f}M hits, {matched} frequency matches...")

    candidates = [h for _, _, h in sorted(top_k, reverse=True)]
    print(f"[BELLA STELLAR] Processed {processed:,} hits | {matched} frequency matches | top {len(candidates)} candidates")
    return candidates


def run_stellar_pipeline(
    mode: str = "radio",
    dry_run: bool = False,
    dataset_path: str = None,
    use_cache: bool = True,
    max_hits: int = 500000,
) -> dict:
    """
    Process radio hit data for technosignature candidates.

    mode="radio": analyze BL hit CSV for narrowband anomalies
    dry_run=True: check dataset availability and print stats only
    dataset_path: custom CSV path (overrides BL default)
    """
    import psutil
    avail_gb = psutil.virtual_memory().available / (1024**3)
    print(f"[BELLA STELLAR] Available RAM: {avail_gb:.1f}GB")

    print(f"\n[BELLA STELLAR] Starting pipeline: mode={mode}, dry_run={dry_run}")

    path = Path(dataset_path) if dataset_path else BL_DATASET_FILE
    if not path.exists():
        print(f"[BELLA STELLAR] Dataset not found: {path}")
        print(f"[BELLA STELLAR] Run: bella signal --download to fetch BL dataset")
        return {}

    if dry_run:
        size = os.path.getsize(path) // (1024**2)
        print(f"[BELLA STELLAR] Dataset found: {path} ({size}MB)")
        return {"status": "dry_run_ok", "path": str(path)}

    # Load and score in a streaming fashion
    candidates = run_stellar_pipeline_streaming(path, max_hits=max_hits)

    # Rank and write report
    candidates = sorted(candidates, key=lambda x: -x.get("anomaly_score", 0))
    _write_stellar_report(candidates)

    top = candidates[0]["anomaly_score"] if candidates else 0.0
    print(f"[BELLA STELLAR] Pipeline complete. Top score: {top:.3f}")
    return {"candidates": candidates}


def _write_stellar_report(candidates: list) -> None:
    from datetime import datetime
    findings_dir = Path(__file__).parent / "findings" / datetime.now().strftime("%Y-%m-%d")
    findings_dir.mkdir(parents=True, exist_ok=True)

    json_path = findings_dir / "stellar_candidates.json"
    summary_path = findings_dir / "stellar_summary.txt"

    with open(json_path, "w") as f:
        json.dump({"candidates": candidates[:1000]}, f, indent=2)

    with open(summary_path, "w") as f:
        f.write(f"Bella Stellar Pipeline Summary\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Total candidates: {len(candidates)}\n\n")
        f.write("Top 10:\n")
        for i, c in enumerate(candidates[:10], 1):
            f.write(f"  {i}. {c.get('source_name','')} | "
                   f"freq={c.get('frequency',0):.3f}MHz | "
                   f"snr={c.get('snr',0):.1f} | "
                   f"match={c.get('matched_frequency','')} | "
                   f"score={c.get('anomaly_score',0):.3f}\n")

    print(f"[BELLA STELLAR] Wrote {json_path}")
    print(f"[BELLA STELLAR] Wrote {summary_path}")

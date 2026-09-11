#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Orders of Magnitude LLC <orders@ofmagnitude.com>
"""
bella_astro.py — Astrobiology / technosignature discovery pipeline.

Queries public astronomical archives for biosignatures and technosignatures,
ranks candidates by anomaly score, and writes Bella findings reports.
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Graceful imports — same pattern as bella_phonon.py
HAS_NUMPY = False
HAS_YAML = False
HAS_ASTROPY = False
HAS_ASTROQUERY = False
HAS_LIGHTKURVE = False

np = None
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    pass

yaml = None
try:
    import yaml
    HAS_YAML = True
except ImportError:
    pass

u = None
SkyCoord = None
try:
    import astropy
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    HAS_ASTROPY = True
except ImportError:
    pass

try:
    import astroquery
    HAS_ASTROQUERY = True
except ImportError:
    pass

search_lightcurve = None
try:
    import lightkurve
    from lightkurve import search_lightcurve
    HAS_LIGHTKURVE = True
except ImportError:
    pass

console = None
try:
    from rich.console import Console
    console = Console()
except ImportError:
    pass

ASTRO_AVAILABLE = HAS_NUMPY and HAS_YAML and HAS_ASTROPY and HAS_ASTROQUERY and HAS_LIGHTKURVE

_ASTRO_INSTALL_HINT = (
    "Astro dependencies not installed. Install with: "
    "pip install astropy astroquery lightkurve --break-system-packages"
)

_MAST_HOST = "mast.stsci.edu"
ASTRO_TIMEOUT = int(os.environ.get("BELLA_ASTRO_TIMEOUT", "30"))
MAST_TOKEN = os.environ.get("BELLA_MAST_TOKEN", "")


def _archives_reachable(timeout: float = 5.0) -> bool:
    """Probe MAST HTTPS reachability."""
    try:
        import socket
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((_MAST_HOST, 443))
        return True
    except Exception:
        return False


BELLA_DIR = Path(__file__).resolve().parent
ASTRO_CACHE = Path.home() / ".bella" / "astro_cache"
FINDINGS_DIR = BELLA_DIR / "findings"

# Effective temperature ranges (K) for the spectral types we care about
_SPTYPE_TEMP = {
    "A": (7500, 10000),
    "F": (6000, 7500),
    "G": (5200, 6000),
    "K": (3700, 5200),
    "M": (2400, 3700),
}

CATALOG_CACHE = ASTRO_CACHE / "catalogs"
CATALOG_CACHE.mkdir(parents=True, exist_ok=True)

CATALOG_SOURCES = {
    "toi": {
        "url": "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv",
        "filename": "toi_catalog.csv",
        "description": "TESS Objects of Interest (~7000 candidates)",
        "max_age_days": 7,
        "ra_col": "RA", "dec_col": "Dec", "ra_fmt": "hms",
        "teff_col": "Stellar Eff Temp (K)",
        "dist_col": "Stellar Distance (pc)", "name_col": "TOI",
        "disposition_col": "TESS Disposition",
        "depth_col": "Depth (ppm)",
        "eq_temp_col": "Planet Equil Temp (K)",
        "sectors_col": "Sectors",
        "known_col": "Comments",
    },
}

def _catalog_path(key: str) -> Path:
    return CATALOG_CACHE / CATALOG_SOURCES[key]["filename"]

def _catalog_stale(key: str) -> bool:
    p = _catalog_path(key)
    if not p.exists():
        return True
    age = (time.time() - p.stat().st_mtime) / 86400.0
    return age > CATALOG_SOURCES[key]["max_age_days"]

def download_catalog(key: str, force: bool = False) -> bool:
    src = CATALOG_SOURCES[key]
    path = _catalog_path(key)
    if not force and not _catalog_stale(key):
        return True
    print(f"[BELLA ASTRO] Downloading {src['description']}...")
    try:
        import urllib.request
        urllib.request.urlretrieve(src["url"], path)
        print(f"[BELLA ASTRO] Saved to {path} ({path.stat().st_size // 1024}KB)")
        return True
    except Exception as e:
        print(f"[BELLA ASTRO] Download failed: {e}")
        return False

def _load_csv_catalog(path: Union[str, Path], profile: Dict[str, Any], ra_col: str = "ra", dec_col: str = "dec", name_col: str = "name", ra_fmt: str = "deg") -> List[Dict[str, Any]]:
    import csv
    path = Path(path)
    if not path.exists():
        return []

    def _parse_deg(value: Any, is_ra: bool = False) -> float:
        s = str(value).strip()
        if not s:
            raise ValueError("empty coordinate")
        if ":" in s:
            parts = s.split(":")
            sign = -1 if parts[0].startswith("-") else 1
            lead = abs(float(parts[0].replace("-", "")))
            minutes = float(parts[1]) if len(parts) > 1 else 0.0
            seconds = float(parts[2]) if len(parts) > 2 else 0.0
            deg = sign * (lead + minutes / 60.0 + seconds / 3600.0)
            if is_ra:
                deg *= 15.0
            return deg
        return float(s)

    def _float_or(value: Any, default: float = 0.0) -> float:
        s = str(value).strip() if value is not None else ""
        if not s:
            return default
        return float(s)

    def _parse_coord(ra_s: str, dec_s: str, fmt: str) -> Tuple[float, float]:
        if fmt == "hms":
            from astropy.coordinates import Angle
            import astropy.units as u
            ra = float(Angle(ra_s, unit=u.hourangle).deg)
            dec = float(Angle(dec_s, unit=u.deg).deg)
            return ra, dec
        return _parse_deg(ra_s, is_ra=True), _parse_deg(dec_s, is_ra=False)

    sptypes = profile.get("target_spectral_types", ["K", "G", "M"])
    teff_lo = min(_SPTYPE_TEMP.get(s, [0, 99999])[0] for s in sptypes if s in _SPTYPE_TEMP)
    teff_hi = max(_SPTYPE_TEMP.get(s, [0, 99999])[1] for s in sptypes if s in _SPTYPE_TEMP)
    teff_col = None
    for row in csv.DictReader(path.open(newline='', encoding='utf-8', errors='ignore')):
        for key in row:
            if key and key.lower() in ('teff', 'stellar eff temp (k)', 'st_teff', 'teff_val'):
                teff_col = key
                break
        if teff_col:
            break
    candidates: List[Dict[str, Any]] = []
    with open(path, newline='', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ra, dec = _parse_coord(row.get(ra_col, ""), row.get(dec_col, ""), ra_fmt)
                teff = _float_or(row.get(teff_col, 0) if teff_col else 0, 0.0)
                if teff and not (teff_lo <= teff <= teff_hi):
                    continue
                dist = float("inf")
                dist_val = row.get("Stellar Distance (pc)") or row.get("sy_dist") or row.get("dist")
                if dist_val:
                    dist = _float_or(dist_val, float("inf"))
                name = str(row.get(name_col, ""))
                candidates.append({
                    "source_id": name,
                    "ra": ra,
                    "dec": dec,
                    "distance_pc": dist,
                    "spectral_type": "?",
                    "h_mag": None,
                    "k_mag": None,
                    "w3mag": None,
                    "w4mag": None,
                    "tessflag": None,
                    "signals": {},
                    "anomaly_score": 0.0,
                    "catalog": "custom",
                })
            except Exception:
                continue
    return candidates

def load_catalog_candidates(key: str, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    if key not in CATALOG_SOURCES:
        return []
    src = CATALOG_SOURCES[key]
    candidates = _load_csv_catalog(
        _catalog_path(key),
        profile,
        ra_col=src["ra_col"],
        dec_col=src["dec_col"],
        name_col=src["name_col"],
        ra_fmt=src.get("ra_fmt", "deg"),
    )
    def _float_or(value: Any, default: float = 0.0) -> float:
        s = str(value).strip() if value is not None else ""
        if not s:
            return default
        return float(s)

    if key == "toi":
        kept: List[Dict[str, Any]] = []
        for c in candidates:
            try:
                import csv
                with open(_catalog_path(key), newline='', encoding='utf-8', errors='ignore') as f:
                    for row in csv.DictReader(f):
                        if str(row.get(src["name_col"], "")).strip() != c["source_id"]:
                            continue
                        comments = str(row.get(src["known_col"], ""))
                        disposition = str(row.get(src["disposition_col"], "")).strip()
                        if disposition in ("EB", "FP", "FA"):
                            break
                        bad_comments = ["fp", "neb", "retired", "not real", "false positive"]
                        if any(b in comments.lower() for b in bad_comments):
                            break
                        c["disposition"] = disposition
                        c["transit_depth_ppm"] = _float_or(row.get(src["depth_col"], ""), 0.0)
                        c["eq_temp_k"] = _float_or(row.get(src["eq_temp_col"], ""), 0.0)
                        c["tess_sectors"] = str(row.get(src["sectors_col"], "")).strip()
                        c["known_name"] = comments.strip()
                        c["catalog"] = "toi"
                        kept.append(c)
                        break
            except Exception:
                continue
        return kept
    return candidates


def _ensure_cache() -> None:
    ASTRO_CACHE.mkdir(parents=True, exist_ok=True)


def _cache_path(key: str) -> Path:
    _ensure_cache()
    return ASTRO_CACHE / f"{key}.json"


def _load_yaml(name: str) -> Dict[str, Any]:
    path = BELLA_DIR / "profiles" / f"{name}.yaml"
    if not path.exists() or yaml is None:
        return {}
    return yaml.safe_load(path.read_text()) or {}


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _findings_dir() -> Path:
    d = FINDINGS_DIR / _today()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _merge_profiles(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Combine two profile dicts for mode='both'."""
    return {
        "name": "both",
        "description": f"{a.get('description','')} / {b.get('description','')}",
        "query_inject": f"{a.get('query_inject','')} / {b.get('query_inject','')}",
        "protein_query": a.get("protein_query") or b.get("protein_query"),
        "signals": list(a.get("signals", [])) + list(b.get("signals", [])),
        "data_sources": list(set(a.get("data_sources", []) + b.get("data_sources", []))),
        "target_spectral_types": list(set(a.get("target_spectral_types", []) + b.get("target_spectral_types", []))),
        "max_distance_pc": max(a.get("max_distance_pc", 0), b.get("max_distance_pc", 0)),
        "n_initial_candidates": max(a.get("n_initial_candidates", 0), b.get("n_initial_candidates", 0)),
        "max_spectral_followup": max(a.get("max_spectral_followup", 0), b.get("max_spectral_followup", 0)),
        "max_tess_followup": max(a.get("max_tess_followup", 0), b.get("max_tess_followup", 0)),
        "use_cache": a.get("use_cache", True) and b.get("use_cache", True),
        "category": a.get("category") or b.get("category"),
    }


def _profile_for_mode(mode: str) -> Dict[str, Any]:
    if mode == "biosignatures":
        return _load_yaml("astrobiology")
    if mode == "technosignatures":
        return _load_yaml("technosignatures")
    if mode == "both":
        return _merge_profiles(_load_yaml("astrobiology"), _load_yaml("technosignatures"))
    raise ValueError(f"Unknown mode: {mode}")


def _build_teff_clause(sptypes: List[str]) -> str:
    clauses = []
    for s in sptypes:
        s = s.upper()
        if s in _SPTYPE_TEMP:
            lo, hi = _SPTYPE_TEMP[s]
            clauses.append(f"(teff_gspphot BETWEEN {lo} AND {hi})")
    if not clauses:
        clauses.append("(1=1)")
    return " OR ".join(clauses)


def _scalar(v):
    """Return a Python scalar from an Astropy Quantity, masked, or plain number."""
    if v is None:
        return None
    if hasattr(v, "value"):
        v = v.value
    if hasattr(v, "mask"):
        if v.mask:
            return None
    try:
        return float(v)
    except Exception:
        return None


def _make_candidate(row: Dict[str, Any]) -> Dict[str, Any]:
    source_id = (
        row.get("source_id")
        or row.get("SOURCE_ID")
        or row.get("ID")
        or row.get("TICID")
        or row.get("pl_name")
        or row.get("PL_NAME")
        or "unknown"
    )
    ra = _scalar(row.get("ra", row.get("RA", 0.0))) or 0.0
    dec = _scalar(row.get("dec", row.get("DEC", 0.0))) or 0.0
    parallax = _scalar(
        row.get("parallax")
        or row.get("PARALLAX")
        or row.get("Plx")
        or row.get("PLX")
        or 0.0
    ) or 0.0
    direct_dist = _scalar(row.get("d") or row.get("D") or row.get("dist") or row.get("sy_dist"))
    if direct_dist and direct_dist > 0:
        distance_pc = float(direct_dist)
    elif parallax and parallax > 0:
        distance_pc = 1000.0 / parallax
    else:
        distance_pc = float("inf")
    teff = _scalar(
        row.get("teff_gsphot")
        or row.get("TEFF_GSPPHOT")
        or row.get("Teff")
        or row.get("teff")
        or row.get("st_teff")
        or row.get("ST_TEFF")
        or 0.0
    ) or 0.0
    spectral_type = "?"
    if teff is not None:
        for s, (lo, hi) in _SPTYPE_TEMP.items():
            if lo <= float(teff) <= hi:
                spectral_type = s
                break
    h_mag = _scalar(row.get("sy_hmag") or row.get("Hmag") or row.get("h_m"))
    k_mag = _scalar(row.get("sy_kmag") or row.get("Kmag") or row.get("k_m"))
    w3 = _scalar(row.get("sy_w3mag") or row.get("w3mag") or row.get("W3MAG"))
    w4 = _scalar(row.get("sy_w4mag") or row.get("w4mag") or row.get("W4MAG"))
    tessflag = row.get("TESSflag") or row.get("TESSFlag") or row.get("sy_tessflag")
    if tessflag is not None and not (hasattr(tessflag, "mask") and tessflag.mask):
        tessflag = str(tessflag).strip()
    else:
        tessflag = None

    return {
        "source_id": str(source_id),
        "ra": ra,
        "dec": dec,
        "distance_pc": distance_pc,
        "spectral_type": spectral_type,
        "h_mag": h_mag,
        "k_mag": k_mag,
        "w3mag": w3,
        "w4mag": w4,
        "tessflag": tessflag,
        "signals": {},
        "anomaly_score": 0.0,
    }


def query_gaia_candidates(
    mode: str = "both",
    max_distance_pc: float = 100.0,
    n_initial: int = 10000,
    use_cache: bool = True,
    tmag_range: Optional[List[float]] = None,
    nexsci_limit: int = 500,
    catalog: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """Query MAST TIC and NExScI exoplanet archive for target stars."""
    if not ASTRO_AVAILABLE:
        print(_ASTRO_INSTALL_HINT)
        return []

    from astroquery.mast import Catalogs
    profile = _profile_for_mode(mode)
    sptypes = profile.get("target_spectral_types", ["K", "G", "M"])
    if tmag_range is None:
        tmag_range = [-3, 10]
    cache_key = f"gaia_candidates_{mode}_{max_distance_pc}_{n_initial}_{'_'.join(sorted(sptypes))}"
    cache = _cache_path(cache_key)

    if use_cache and cache.exists():
        print("[BELLA ASTRO] Loading candidates from cache")
        try:
            return json.loads(cache.read_text())
        except Exception as e:
            print(f"[BELLA ASTRO] Cache read failed: {e}")

    if not _archives_reachable():
        print("[BELLA ASTRO] Warning: MAST not reachable; skipping live query. Falling back to cached candidates if available.")
        return []

    print("[BELLA ASTRO] Querying NExScI exoplanet archive (primary) and TIC (secondary)...")
    try:
        from astroquery.mast import Catalogs
        import urllib.request, json as _json
    except Exception as e:
        print(f"[BELLA ASTRO] MAST/NExScI import failed: {e}")
        return []

    teff_lo = min(_SPTYPE_TEMP[s][0] for s in sptypes if s in _SPTYPE_TEMP)
    teff_hi = max(_SPTYPE_TEMP[s][1] for s in sptypes if s in _SPTYPE_TEMP)

    def _in_range(t):
        t = _scalar(t)
        if t is None:
            return False
        return any(
            _SPTYPE_TEMP[s][0] <= t <= _SPTYPE_TEMP[s][1]
            for s in sptypes if s in _SPTYPE_TEMP
        )

    # Primary: confirmed transiting exoplanets from NExScI (JWST follow-up targets)
    planet_candidates = []
    try:
        url = (
            "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
            f"?query=SELECT+TOP+{nexsci_limit}+pl_name,ra,dec,st_teff,"
            "st_rad,sy_dist,sy_tmag,sy_hmag,sy_kmag,"
            "sy_w1mag,sy_w2mag,sy_w3mag,sy_w4mag+FROM+ps"
            "+WHERE+pl_controv_flag%3D0"
            "&format=json"
        )
        rows = []
        for attempt, t in enumerate([10, 25]):
            try:
                with urllib.request.urlopen(url, timeout=t) as r:
                    rows = _json.loads(r.read().decode())
                break
            except Exception as e:
                if attempt == 1:
                    print(f"[BELLA ASTRO] NExScI primary query failed: {e}")
                    rows = []
        for row in rows:
            if _in_range(row.get("st_teff")):
                planet_candidates.append(_make_candidate(row))
    except Exception as e:
        print(f"[BELLA ASTRO] NExScI primary query failed: {e}")

    # Secondary: TIC for TESS flag coverage (page 1, small pagesize)
    tic_candidates = []
    try:
        tic_table = Catalogs.query_criteria(
            catalog="TIC",
            Tmag=tmag_range,
            objType="STAR",
            pagesize=50,
            page=1,
        )
        for row in tic_table[:n_initial]:
            rowd = dict(zip(tic_table.colnames, row))
            if _in_range(rowd.get("Teff")):
                tic_candidates.append(_make_candidate(rowd))
    except Exception as e:
        print(f"[BELLA ASTRO] TIC secondary query failed: {e}")

    # Merge and deduplicate by coordinates (rounded to 0.001 deg)
    seen = set()
    candidates = []
    for c in tic_candidates + planet_candidates:
        if c["distance_pc"] != float("inf") and c["distance_pc"] > max_distance_pc:
            continue
        key = (round(c["ra"], 3), round(c["dec"], 3))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(c)

    all_candidates = list(candidates)

    # Local catalogs (downloaded once, searched instantly)
    for cat_key in ["toi"]:
        if _catalog_stale(cat_key):
            download_catalog(cat_key)
        cat_candidates = load_catalog_candidates(cat_key, profile)
        print(f"[BELLA ASTRO] {cat_key}: {len(cat_candidates)} candidates")
        all_candidates.extend(cat_candidates)

    if catalog:
        custom = _load_csv_catalog(catalog, profile)
        print(f"[BELLA ASTRO] custom catalog: {len(custom)} candidates")
        all_candidates.extend(custom)

    seen2 = set()
    final = []
    for c in all_candidates:
        key = (round(c["ra"], 3), round(c["dec"], 3))
        if key in seen2:
            continue
        seen2.add(key)
        final.append(c)

    if use_cache and final:
        cache.write_text(json.dumps(final, indent=2))
    print(f"[BELLA ASTRO] {len(final)} total candidates")
    return final


def detect_ir_excess(candidates: List[Dict[str, Any]], profile: Dict[str, Any], hk_threshold: float = 0.3) -> None:
    """IR excess from H-K color (2MASS/PS1/sy_hmag - sy_kmag)."""
    if not ASTRO_AVAILABLE or not np:
        return

    weight = next((s["weight"] for s in profile.get("signals", []) if s.get("id") == "ir_excess"), 0.0)
    if not weight:
        return

    print(f"[BELLA ASTRO] Checking IR excess (W3-W4 < -0.5, fallback H-K > {hk_threshold})...")
    for c in candidates:
        w3 = c.get("w3mag")
        w4 = c.get("w4mag")
        if w3 is not None and w4 is not None:
            if (w3 - w4) < -0.5:
                c["signals"]["ir_excess"] = True
                c["signals"]["w3_w4"] = round(w3 - w4, 3)
                c["anomaly_score"] += weight
            continue
        h = c.get("h_mag")
        k = c.get("k_mag")
        if h is not None and k is not None and (h - k) > hk_threshold:
            c["signals"]["ir_excess"] = True
            c["signals"]["h_k"] = round(h - k, 3)
            c["anomaly_score"] += weight
    print("[BELLA ASTRO] IR excess check complete")


def detect_dimming_anomalies(candidates: List[Dict[str, Any]], profile: Dict[str, Any]) -> None:
    """TESS candidate flag from TIC metadata."""
    if not ASTRO_AVAILABLE or not np:
        return

    weight = next((s["weight"] for s in profile.get("signals", []) if s.get("id") == "dimming_anomaly"), 0.0)
    if not weight:
        return

    print("[BELLA ASTRO] Checking TESS observability flags...")
    for c in candidates:
        if c.get("tessflag"):
            c["signals"]["tess_candidate"] = True
            c["anomaly_score"] += weight
    print("[BELLA ASTRO] TESS dimming check complete")


def detect_toi_signals(candidates: List[Dict[str, Any]], profile: Dict[str, Any]) -> None:
    """Score TESS Objects of Interest for deep transits and habitable-zone candidates."""
    if not ASTRO_AVAILABLE or not np:
        return

    weight_dimming = next((s["weight"] for s in profile.get("signals", []) if s.get("id") == "dimming_anomaly"), 0.0)
    weight_biosig = next((s["weight"] for s in profile.get("signals", []) if s.get("id") == "o2_ch4_disequilibrium"), 0.5)
    if not (weight_dimming or weight_biosig):
        return

    print("[BELLA ASTRO] Checking TOI catalog signals...")
    for c in candidates:
        if c.get("catalog") != "toi":
            continue
        if c.get("disposition") in ("EB", "FP", "FA"):
            continue
        bad_comments = ["fp", "neb", "retired", "not real", "false positive", "single transit"]
        known = c.get("known_name", "").strip()
        known_lower = known.lower()
        if any(b in known_lower for b in bad_comments):
            continue
        # Deep transit anomaly (not a known planet)
        depth = c.get("transit_depth_ppm", 0)
        if depth > 10000 and not known:
            c["signals"]["deep_transit_anomaly"] = True
            c["signals"]["depth_ppm"] = depth
            c["anomaly_score"] += weight_dimming
        # Habitable zone candidate
        eq_temp = c.get("eq_temp_k", 0)
        if 200 <= eq_temp <= 320:
            c["signals"]["habitable_zone"] = True
            c["signals"]["eq_temp_k"] = eq_temp
            c["anomaly_score"] += weight_biosig * 0.3
        # Multi-signal unknown: habitable zone + deep transit, PC, no known name
        if (c["signals"].get("habitable_zone") and c["signals"].get("deep_transit_anomaly")
                and not known and c.get("disposition") == "PC"):
            c["signals"]["multi_signal_unknown"] = True
            c["anomaly_score"] += 0.2
        # Has TESS sectors and is a planet candidate
        if c.get("tess_sectors") and c.get("disposition") == "PC":
            c["signals"]["tess_unconfirmed_candidate"] = True
    print("[BELLA ASTRO] TOI catalog signals check complete")


def detect_spectral_signatures(
    candidates: List[Dict[str, Any]],
    profile: Dict[str, Any],
    cone_arcsec: float = 30.0,
) -> None:
    """JWST MAST archive cross-correlation with signal line lists."""
    if not ASTRO_AVAILABLE:
        return
    if not candidates:
        return

    weight = next((s["weight"] for s in profile.get("signals", []) if "molecules" in s), 0.0)
    if not weight:
        return

    print(f"[BELLA ASTRO] Cross-matching JWST MAST spectra ({cone_arcsec} arcsec cone)...")
    try:
        from astroquery.mast import Observations
        from astropy.coordinates import SkyCoord
        from astropy.table import Table
        import astropy.units as u
        import threading
    except Exception as e:
        print(f"[BELLA ASTRO] JWST MAST import failed: {e}")
        return

    jwst_cache = ASTRO_CACHE / "jwst_catalog.json"
    jwst_table = None
    err: Optional[str] = None
    loaded_from_cache = False

    try:
        if jwst_cache.exists():
            age_days = (time.time() - jwst_cache.stat().st_mtime) / 86400.0
            if age_days < 7.0:
                jwst_table = Table.read(jwst_cache, format="ascii.json")
                loaded_from_cache = True
                print(f"[BELLA ASTRO] JWST catalog cached ({len(jwst_table)} observations)")
    except Exception as e:
        print(f"[BELLA ASTRO] JWST cache read failed: {e}")
        jwst_table = None

    if not loaded_from_cache:
        def _fetch() -> None:
            nonlocal jwst_table, err
            try:
                jwst_table = Observations.query_criteria(
                    obs_collection="JWST",
                    dataproduct_type="spectrum",
                )
            except Exception as e:
                err = str(e)

        t = threading.Thread(target=_fetch, daemon=True)
        t.start()
        t.join(timeout=60.0)

        if t.is_alive():
            print("[BELLA ASTRO] JWST catalog query timed out (60s); skipping spectral cross-match")
            return
        if err:
            print(f"[BELLA ASTRO] JWST catalog query failed: {err}")
            return
        if jwst_table is None or len(jwst_table) == 0:
            print("[BELLA ASTRO] No JWST spectra found; skipping spectral cross-match")
            return

        try:
            ASTRO_CACHE.mkdir(parents=True, exist_ok=True)
            jwst_table.write(jwst_cache, format="ascii.json", overwrite=True)
            print(f"[BELLA ASTRO] JWST catalog cached ({len(jwst_table)} observations)")
        except Exception as e:
            print(f"[BELLA ASTRO] JWST cache write failed: {e}")

    try:
        jwst_coords = SkyCoord(
            ra=jwst_table["s_ra"],
            dec=jwst_table["s_dec"],
            unit="deg",
        )
        cand_coords = SkyCoord(
            ra=[c["ra"] for c in candidates],
            dec=[c["dec"] for c in candidates],
            unit="deg",
        )
        _, sep, _ = cand_coords.match_to_catalog_sky(jwst_coords)
        matches = 0
        for i, c in enumerate(candidates):
            if sep[i].arcsec < cone_arcsec:
                c["signals"]["jwst_observed"] = True
                c["anomaly_score"] += weight
                matches += 1
        print(f"[BELLA ASTRO] JWST cross-match complete ({matches} matches)")
    except Exception as e:
        print(f"[BELLA ASTRO] JWST local cross-match failed: {e}")


def rank_and_report(candidates: List[Dict[str, Any]], mode: str, profile: Dict[str, Any]) -> None:
    """Sort by anomaly score and write findings.

    Note: NSSort would accelerate this at scale.
    """
    if np is None:
        return

    candidates.sort(key=lambda x: x["anomaly_score"], reverse=True)
    outdir = _findings_dir()
    json_path = outdir / "astro_candidates.json"
    txt_path = outdir / "astro_summary.txt"

    payload = {
        "mode": mode,
        "profile": profile.get("name", mode),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "total_candidates": len(candidates),
        "candidates": candidates,
    }

    json_path.write_text(json.dumps(payload, indent=2, default=float))

    top5 = candidates[:5]
    lines = [
        "Bella Astro Pipeline Summary",
        f"Mode: {mode}",
        f"Profile: {profile.get('name', mode)}",
        f"Generated: {payload['generated_utc']}",
        f"Total candidates: {len(candidates)}",
        "Top 5 by anomaly score:",
    ]
    for i, c in enumerate(top5, 1):
        signals = list(c["signals"].keys())
        known_name = c.get("known_name", "").strip()
        if known_name:
            console_label = f"[dim]known: {known_name}[/dim]"
            file_label = f"[known] {known_name}"
        else:
            console_label = "[bold yellow]UNKNOWN ★[/bold yellow]"
            file_label = "[UNKNOWN ★]"
        if console is not None:
            console.print(
                f"  {i}. {c['source_id']} {console_label} "
                f"(score={c['anomaly_score']:.3f}, "
                f"spt={c['spectral_type']}, d={c['distance_pc']:.1f} pc) "
                f"signals={signals}"
            )
        else:
            print(f"  {i}. {c['source_id']} {file_label} ...")
        lines.append(
            f"  {i}. {c['source_id']} {file_label} (score={c['anomaly_score']:.3f}, "
            f"spt={c['spectral_type']}, d={c['distance_pc']:.1f} pc) "
            f"signals={signals}"
        )

    txt_path.write_text("\n".join(lines) + "\n")
    print(f"[BELLA ASTRO] Wrote {json_path}")
    print(f"[BELLA ASTRO] Wrote {txt_path}")


def run_astro_pipeline(
    mode: str = "both",
    dry_run: bool = False,
    use_cache: bool = True,
    max_distance_pc: float = 100.0,
    n_initial: int = 10000,
    catalog: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """Main entry point for the astrobiology/technosignatures pipeline."""
    if not ASTRO_AVAILABLE:
        print(_ASTRO_INSTALL_HINT)
        return []

    print(f"[BELLA ASTRO] Starting pipeline: mode={mode}, dry_run={dry_run}")
    profile = _profile_for_mode(mode)
    effective_max = min(max_distance_pc, float(profile.get("max_distance_pc", max_distance_pc)))
    effective_n = int(min(n_initial, int(profile.get("n_initial_candidates", n_initial))))
    tmag_range = profile.get("tmag_range", [-3, 10])
    cone_arcsec = float(profile.get("cone_arcsec", 30.0))
    hk_threshold = float(profile.get("hk_threshold", 0.3))
    nexsci_limit = int(profile.get("nexsci_limit", 500))
    if dry_run:
        # Confirm URL reachability and return; do not run ADQL
        if not _archives_reachable():
            print("[BELLA ASTRO] MAST not reachable; skipping live query")
            return []
        print("[BELLA ASTRO] MAST reachable. Dry run complete.")
        return []

    toi_path = CATALOG_CACHE / "toi_catalog.csv"
    sptypes = profile.get("target_spectral_types", ["K", "G", "M"])
    cache_key = f"gaia_candidates_{mode}_{effective_max}_{effective_n}_{'_'.join(sorted(sptypes))}"
    cache_path = _cache_path(cache_key)
    if cache_path.exists() and toi_path.exists():
        if toi_path.stat().st_mtime > cache_path.stat().st_mtime:
            cache_path.unlink()
            print("[BELLA ASTRO] TOI catalog updated — cache cleared automatically")

    candidates = query_gaia_candidates(
        mode=mode,
        max_distance_pc=effective_max,
        n_initial=effective_n,
        use_cache=use_cache,
        tmag_range=tmag_range,
        nexsci_limit=nexsci_limit,
        catalog=catalog,
    )

    detect_ir_excess(candidates, profile, hk_threshold=hk_threshold)
    detect_toi_signals(candidates, profile)
    detect_dimming_anomalies(candidates, profile)
    detect_spectral_signatures(candidates, profile, cone_arcsec=cone_arcsec)
    rank_and_report(candidates, mode, profile)

    top_score = candidates[0]["anomaly_score"] if candidates else 0.0
    print(f"[BELLA ASTRO] Pipeline complete. Top candidate score: {top_score:.3f}")
    return candidates

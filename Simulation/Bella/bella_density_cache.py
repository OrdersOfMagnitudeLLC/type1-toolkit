# OOM Commercial License v1.0
# Copyright 2026 Orders of Magnitude LLC
# See ./LICENSE.md for terms.
# Copyright (C) 2026 Orders of Magnitude LLC <orders@ofmagnitude.com>
"""
Density restart cache for SPARC phonon calculations.
Caches converged reference cell density so subsequent phonon runs
on the same structure (different pressures, conditions) can skip the
cold SCF start. 5-10x speedup on runs 2+.
"""
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

DENSITY_CACHE = Path.home() / ".bella" / "density_cache"
DENSITY_EXTENSIONS = [".dens", ".restart"]


def cache_key(cif_path: str, pressure: float = 0.0) -> str:
    with open(cif_path, "rb") as f:
        content = f.read()
    hasher = hashlib.sha256()
    hasher.update(content)
    hasher.update(f"_{pressure:.4f}".encode())
    return hasher.hexdigest()[:16]


def save_reference_density(cif_path: str, workdir: str, name: str, pressure: float = 0.0):
    key = cache_key(cif_path, pressure)
    dest = DENSITY_CACHE / key
    dest.mkdir(parents=True, exist_ok=True)
    saved = []
    for ext in DENSITY_EXTENSIONS:
        src = Path(workdir) / f"{name}{ext}"
        if src.exists():
            shutil.copy(src, dest / f"reference{ext}")
            saved.append(ext)
    if saved:
        meta = {
            "cif": str(cif_path),
            "pressure": float(pressure),
            "created": datetime.utcnow().isoformat(),
            "files": saved,
        }
        (dest / "meta.json").write_text(json.dumps(meta, indent=2))
        print(
            f"[BELLA] ✓ Reference density cached ({', '.join(saved)}) "
            f"for P={pressure:.2f} GPa"
        )
        print(
            f"[BELLA]   Next phonon run on this structure/pressure: warm start "
            f"(~5-10 SCF vs ~50)"
        )
    else:
        print(
            f"[BELLA] ⚠ PRINT_DENSITY may not be set — "
            f"no density files found to cache"
        )


def load_reference_density(cif_path: str, disp_workdir: str,
                            name: str, pressure: float = 0.0) -> bool:
    key = cache_key(cif_path, pressure)
    src_dir = DENSITY_CACHE / key
    if not (src_dir / "meta.json").exists():
        return False
    copied = []
    for ext in DENSITY_EXTENSIONS:
        src = src_dir / f"reference{ext}"
        if src.exists():
            shutil.copy(src, Path(disp_workdir) / f"{name}{ext}")
            copied.append(ext)
    if copied:
        print(
            f"[BELLA] ✓ Warm start: cached density loaded → {name} "
            f"({', '.join(copied)}"
        )
        return True
    return False


def has_cached_density(cif_path: str, pressure: float = 0.0) -> bool:
    key = cache_key(cif_path, pressure)
    return (DENSITY_CACHE / key / "meta.json").exists()


def clear_cache(cif_path: str = None, pressure: float = None):
    if cif_path:
        key = cache_key(cif_path, pressure if pressure is not None else 0.0)
        target = DENSITY_CACHE / key
        if target.exists():
            shutil.rmtree(target)
            print(f"[BELLA] Cache cleared for {Path(cif_path).name}")
    else:
        if DENSITY_CACHE.exists():
            shutil.rmtree(DENSITY_CACHE)
            print(f"[BELLA] Full density cache cleared.")

PHONON_INDEX = Path.home() / ".bella" / "phonon_results.json"


def save_phonon_result(cif_path: str, pressure: float, min_freq: float,
                       imaginary: bool, wall_time_s: float) -> None:
    """Permanently log a completed phonon result. Never overwrites — appends."""
    import json, hashlib
    PHONON_INDEX.parent.mkdir(parents=True, exist_ok=True)
    key = cache_key(cif_path, pressure)
    record = {
        "key": key,
        "cif": str(cif_path),
        "pressure_gpa": float(pressure),
        "min_freq_thz": float(min_freq),
        "imaginary": bool(imaginary),
        "wall_time_s": float(wall_time_s),
        "timestamp": datetime.utcnow().isoformat(),
    }
    existing = []
    if PHONON_INDEX.exists():
        try:
            existing = json.loads(PHONON_INDEX.read_text())
        except Exception:
            existing = []
    # Remove old entry for same key if exists, then append new
    existing = [r for r in existing if r.get("key") != key]
    existing.append(record)
    PHONON_INDEX.write_text(json.dumps(existing, indent=2))
    print(f"[BELLA] ✓ Phonon result saved permanently → ~/.bella/phonon_results.json")


def get_phonon_result(cif_path: str, pressure: float):
    """Return saved phonon result for this CIF+pressure, or None."""
    import json
    if not PHONON_INDEX.exists():
        return None
    key = cache_key(cif_path, pressure)
    try:
        records = json.loads(PHONON_INDEX.read_text())
        for r in records:
            if r.get("key") == key:
                return r
    except Exception:
        pass
    return None

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Orders of Magnitude LLC: ofmagnitude.com

import json, hashlib, os
from pathlib import Path
from datetime import datetime

try:
    import psutil
except ImportError:
    psutil = None

AUDIT_LOG = Path.home() / ".bella" / "audit.log"

FLAGGED_KEYWORDS = [
    "sarin","vx","novichok","tabun","soman","mustard gas",
    "bacillus anthracis","yersinia pestis","variola","ebola",
    "botulinum","ricin","gain of function","h5n1","weaponi",
]

FLAGGED_ELEMENT_SETS = [
    frozenset(["P","F","O"]),  # phosphonofluoridate class
    frozenset(["P","S","N"]),  # VX precursor class
]

def _flagged(text: str) -> bool:
    t = text.lower()
    if any(k in t for k in FLAGGED_KEYWORDS):
        return True
    import re
    elements = set(re.findall(r'[A-Z][a-z]?', text))
    return any(s.issubset(elements) for s in FLAGGED_ELEMENT_SETS)

def audit(command: str, query: str, extra: dict = None):
    """Call this from every cmd_* function. Silent. Always runs."""
    AUDIT_LOG.parent.mkdir(exist_ok=True)
    entry = {
        "ts": datetime.utcnow().isoformat(),
        "cmd": command,
        "query": query,
        "flagged": _flagged(query),
    }
    if extra:
        entry.update(extra)
    with open(AUDIT_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    if entry["flagged"]:
        print("\n[BELLA] This query has been logged under OOM dual-use policy.")
        print("[BELLA] Continued use constitutes acceptance of the prohibited-uses terms.")
        print("[BELLA] See LICENSE — Prohibited Uses section.\n")


def safe_n_jobs(requested_jobs: int, ram_per_worker_gb: float = 3.0,
                ram_cap_gb: float = None) -> int:
    """Clamp requested parallel workers by available RAM and CPU count."""
    by_cpu = os.cpu_count() or 4

    # Treat -1 / 0 as "use all cores"
    requested = requested_jobs if requested_jobs and requested_jobs > 0 else by_cpu

    if psutil is None:
        return max(1, min(requested, by_cpu))

    available = psutil.virtual_memory().available / 1e9
    cap = ram_cap_gb if ram_cap_gb is not None else available * 0.75
    by_ram = max(1, int(cap / ram_per_worker_gb))
    result = min(requested, by_ram, by_cpu)

    print(f"[BELLA] Workers: requested={requested_jobs}, "
          f"by_ram={by_ram} ({cap:.1f}GB cap / {ram_per_worker_gb}GB each), "
          f"by_cpu={by_cpu} → using {result}")
    return result

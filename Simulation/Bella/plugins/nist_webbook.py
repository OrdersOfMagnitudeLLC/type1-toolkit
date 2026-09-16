# OOM Commercial License v1.0
# Copyright 2026 Orders of Magnitude LLC
# See /NS/LICENSE.md for terms.
# Copyright (C) 2026 Orders of Magnitude LLC <orders@ofmagnitude.com>
"""NIST Chemistry WebBook plugin for Bella.

No API key required.  Provides a formula -> enthalpy-of-formation lookup
and exposes a Bob-style search() so the loader does not skip it.
"""
import os
import re
import sys
import urllib.request
import urllib.parse

NIST_URL = os.environ.get(
    "NIST_WEBBOOK_URL", "https://webbook.nist.gov/cgi/cbook.cgi"
)


def name() -> str:
    return "NIST WebBook"


def lookup_by_formula(formula: str, timeout: int = 8) -> dict:
    """Return enthalpy of formation (kJ/mol) for a chemical formula if known.

    Uses the NIST Chemistry WebBook formula search and thermodynamic data pages.
    """
    # Normalise formula for a URL query
    clean = urllib.parse.quote(re.sub(r"[^A-Za-z0-9]", "", formula))
    if not clean:
        return {"enthalpy_of_formation_kj_mol": None, "source": "nist"}

    search_url = (
        f"{NIST_URL}?Formula={clean}&NoFrames=on&Units=SI&cTG=on"
    )
    try:
        with urllib.request.urlopen(search_url, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return {"enthalpy_of_formation_kj_mol": None, "source": "nist"}

    # Extract species IDs from the search results
    ids = re.findall(r'ID=([A-Za-z0-9\-]+)', text)
    ids = list(dict.fromkeys(ids))  # preserve order, dedupe

    for sid in ids[:3]:
        data_url = (
            f"{NIST_URL}?ID={sid}&Units=SI&Mask=2&NoFrames=on"
        )
        try:
            with urllib.request.urlopen(data_url, timeout=timeout) as resp:
                data = resp.read().decode("utf-8", errors="ignore")
        except Exception:
            continue

        # Look for standard enthalpy of formation in kJ/mol
        # Matches forms like "ΔfH&deg;gas = -241.8 kJ/mol" and variants
        patterns = [
            r"ΔfH°(?:[^=]{0,12})\s*=\s*([\-\d\.]+)\s*kJ/mol",
            r"ΔfH&deg;(?:[^=]{0,12})\s*=\s*([\-\d\.]+)\s*kJ/mol",
            r"Standard enthalpy of formation.*?([\-\d\.]+)\s*kJ/mol",
            r"Δ<sub>f</sub>H°(?:[^=]{0,12})\s*=\s*([\-\d\.]+)\s*kJ/mol",
        ]
        for pat in patterns:
            m = re.search(pat, data, re.IGNORECASE | re.DOTALL)
            if m:
                try:
                    return {
                        "enthalpy_of_formation_kj_mol": float(m.group(1)),
                        "source": "nist",
                    }
                except ValueError:
                    continue

    return {"enthalpy_of_formation_kj_mol": None, "source": "nist"}


def search(query: str, limit: int = 20) -> list[dict]:
    """Bob-plugin compatible search. The main discover pipeline uses the
    lookup_by_formula helper on each candidate, so this returns empty for
    natural-language queries.
    """
    # Only look up when query looks like a bare formula
    if re.match(r"^[A-Z][a-z]?[0-9]*", query) and not query.islower():
        res = lookup_by_formula(query)
        if res["enthalpy_of_formation_kj_mol"] is not None:
            return [
                {
                    "formula": query,
                    "source": "nist",
                    "properties": res,
                }
            ]
    return []

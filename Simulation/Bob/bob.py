# Copyright (c) 2026 Orders of Magnitude LLC
# Licensed under the OOM Commercial License v1.0
# See LICENSE.md in the repository root or ofmagnitude.com

"""
Bob data source: catalysis-hub.org integration + Materials Project / GNoME router.
"""
import json
import urllib.request
import urllib.error
import re
import os
import sys
import itertools

import requests

# Foam screener: first-principles phonon stability (T45+T2 derived)
_BELLA_DIR = os.path.join(os.path.dirname(__file__), "..", "Bella")
if _BELLA_DIR not in sys.path:
    sys.path.insert(0, _BELLA_DIR)

CATALYSIS_HUB_URL = "https://api.catalysis-hub.org/graphql"

DOMAIN_KEYWORDS = {
    "magnetocaloric": ["magnetocaloric", "magnetic cooling", "MCE", "magnetocaloric effect"],
    "multiferroic": ["multiferroic", "ferroelectric", "BiFeO3", "magnetoelectric"],
    "mxene": ["MXene", "Ti3C2", "2D carbide", "2D material", "MXenes"],
    "battery": ["silicon anode", "battery", "electrolyte", "lithium", "solid state"],
    "topological": ["topological", "Weyl", "semimetal", "MnBi"],
    "superconductor": ["superconductor", "superconducting", "Tc"],
}

# ─── Crustal abundance (ppm) for earth-abundant filtering ─────────────────
CRUSTAL_ABUNDANCE_PPM = {
    'H': 1400, 'Li': 20, 'B': 10, 'C': 200, 'N': 19, 'O': 461000,
    'F': 585, 'Na': 28300, 'Mg': 20900, 'Al': 81300, 'Si': 277200,
    'P': 1050, 'S': 260, 'Cl': 145, 'K': 25900, 'Ca': 36300, 'Sc': 22,
    'Ti': 6200, 'V': 160, 'Cr': 102, 'Mn': 950, 'Fe': 50000, 'Co': 25,
    'Ni': 84, 'Cu': 60, 'Zn': 70, 'Ga': 19, 'Ge': 1.5, 'Se': 0.05,
    'Rb': 90, 'Sr': 370, 'Y': 33, 'Zr': 165, 'Nb': 20, 'Mo': 1.2,
    'Sn': 2.2, 'Sb': 0.2, 'Ba': 425, 'Hf': 3, 'Ta': 2, 'W': 1.25,
}

def _parse_formula(formula):
    """Parse a chemical formula into {element: count} dict."""
    pattern = re.compile(r'([A-Z][a-z]?)(\d*)')
    out = {}
    for el, count_str in pattern.findall(formula):
        if not el:
            continue
        out[el] = out.get(el, 0) + (int(count_str) if count_str else 1)
    return out

def _formula_natoms(formula):
    """Total atoms in one formula unit."""
    comp = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
    return sum(int(c) if c else 1 for _, c in comp)

def _min_abundance_ppm(formula):
    """Return crustal abundance (ppm) of the rarest element in formula."""
    comp = _parse_formula(formula)
    if not comp:
        return 0.0
    return min(CRUSTAL_ABUNDANCE_PPM.get(el, 0.0) for el in comp)

def _stoich(a, x, b, y):
    """Build a formula string like Fe2Mn from elements and counts."""
    return a + (str(x) if x > 1 else "") + b + (str(y) if y > 1 else "")

# ─── Generative domain profiles ───────────────────────────────────────────
_GEN_PROFILES = {
    "semiconductor": {
        "description": "2D bimetallic transition metal dichalcogenide semiconductors",
        "crystal_system": "hexagonal",
        "r_atomic_A": 2.0,
        "metals": ["Ti", "V", "Cr", "Mn", "Fe", "Ni", "Zr", "Nb", "Mo", "Hf", "Ta", "W"],
        "chalcogens": ["S", "Se", "Te"],
        "pattern": "bimetallic_tmd",
        "min_abundance_ppm": 1.0,
        "min_metal_ppm": 100,
    },
    "mram": {
        "description": "Heusler alloy magnetic tunnel junctions for MRAM",
        "crystal_system": "cubic",
        "r_atomic_A": 2.0,
        "x_elements": ["Co", "Fe", "Mn", "Ni"],
        "y_elements": ["Cr", "V", "Ti", "Mn", "Fe"],
        "z_elements": ["Al", "Si", "Ga", "Ge", "Sn"],
        "pattern": "heusler",
        "min_abundance_ppm": 1.0,
    },
}

def _detect_gen_domain(query):
    """Detect generative domain from query text."""
    q = query.lower()
    if any(k in q for k in ["heusler", "mram", "magnetic tunnel", "spintronics"]):
        return "mram"
    if any(k in q for k in ["semiconductor", "dichalcogenide", "2d", "tmd", "cvd", "transition metal"]):
        return "semiconductor"
    if any(k in q for k in ["magnet", "tunnel junction"]):
        return "mram"
    return "semiconductor"

def _generate_tmd_formulas(profile, limit=50):
    """Generate bimetallic M1M2X2 transition metal dichalcogenide formulas only."""
    metals = profile["metals"]
    chalcogens = profile["chalcogens"]
    min_metal_ppm = profile.get("min_metal_ppm", 0)
    formulas = []
    for a, b in itertools.combinations(metals, 2):
        # Both metals must exceed min_metal_ppm crustal abundance
        if CRUSTAL_ABUNDANCE_PPM.get(a, 0) < min_metal_ppm:
            continue
        if CRUSTAL_ABUNDANCE_PPM.get(b, 0) < min_metal_ppm:
            continue
        for x in chalcogens:
            formulas.append(f"{a}{b}{x}2")
    return formulas

def _generate_heusler_formulas(profile, limit=80):
    """Generate X2YZ Heusler alloy formulas."""
    x_elems = profile["x_elements"]
    y_elems = profile["y_elements"]
    z_elems = profile["z_elements"]
    formulas = []
    for x in x_elems:
        for y in y_elems:
            for z in z_elems:
                if x == y or y == z or x == z:
                    continue
                formulas.append(f"{x}2{y}{z}")
    return formulas

def _normalize_formula(formula):
    """Normalize formula to canonical form: elements sorted alphabetically with counts."""
    from collections import Counter
    comp = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
    counts = Counter()
    for el, c in comp:
        counts[el] += int(c) if c else 1
    return ''.join(f'{el}{counts[el] if counts[el] > 1 else ""}' for el in sorted(counts))


def _mp_lookup_formula(formula):
    """Check if a formula exists in Materials Project. Returns True/False/None."""
    try:
        comp = _parse_formula(formula)
        elements = sorted(comp.keys())
        chemsys = "-".join(elements)
        norm = _normalize_formula(formula)
        results = _matproj_search(chemsys, limit=50)
        for r in results:
            if "error" in r:
                return None
            mp_f = r.get("formula", "")
            if _normalize_formula(mp_f) == norm:
                return True
        return False
    except Exception:
        return None


def _generative_screen(query, domain=None, limit=20):
    """Generate novel compositions and screen with foam phonon stability."""
    from foam_screener_v2 import phonon_stability, vegard_elastic_mix

    if domain is None:
        domain = _detect_gen_domain(query)
    profile = _GEN_PROFILES.get(domain)
    if not profile:
        return [{"error": f"Unknown generative domain: {domain}"}]

    if profile["pattern"] == "bimetallic_tmd":
        candidates = _generate_tmd_formulas(profile, limit=limit * 3)
    elif profile["pattern"] == "heusler":
        candidates = _generate_heusler_formulas(profile, limit=limit * 3)
    else:
        return [{"error": f"Unknown pattern: {profile['pattern']}"}]

    # Elements exempt from abundance filtering (chalcogens are always allowed)
    abundance_exempt = {"S", "Se", "N", "O", "P", "B", "C", "H", "F", "Cl"}

    # Minimum stability score: below this the foam screener says likely imaginary phonons
    min_stability_score = 0.30

    results = []
    for formula in candidates:
        comp = _parse_formula(formula)
        if not comp:
            continue

        # Abundance filter: only check non-exempt elements (metals)
        metal_ppms = [CRUSTAL_ABUNDANCE_PPM.get(el, 0) for el in comp if el not in abundance_exempt]
        if metal_ppms and min(metal_ppms) < profile["min_abundance_ppm"]:
            continue

        n_atoms = _formula_natoms(formula)

        # Try Anderson-Debye via Vegard's law elastic mixing
        vegard = vegard_elastic_mix(formula)
        debye_method_label = "MEDIUM"
        if vegard is not None:
            B, G, rho, V, n, _ = vegard
            try:
                r = phonon_stability(
                    formula,
                    crystal_system=profile["crystal_system"],
                    r_atomic_A=profile["r_atomic_A"],
                    n_atoms=n_atoms,
                    B_GPa=B, G_GPa=G, rho_gcc=rho, V_cell_A3=V,
                )
                debye_method_label = "HIGH"
            except Exception:
                continue
        else:
            try:
                r = phonon_stability(
                    formula,
                    crystal_system=profile["crystal_system"],
                    r_atomic_A=profile["r_atomic_A"],
                    n_atoms=n_atoms,
                )
            except Exception:
                continue

        if not r.get("stable"):
            continue

        score = r["stability_score"]
        if score < min_stability_score:
            continue

        # Confidence: HIGH (Anderson-Debye), MEDIUM (C_DEBYE), LOW (borderline)
        if debye_method_label == "HIGH":
            confidence = "HIGH" if score >= 0.5 else "LOW"
        else:
            confidence = "MEDIUM" if score >= 0.5 else "LOW"

        # MP cross-reference for NOVEL/KNOWN labeling (not for phonon validity)
        mp_known = _mp_lookup_formula(formula)
        if mp_known is True:
            novel_flag = "KNOWN"
        elif mp_known is False:
            novel_flag = "NOVEL"
        else:
            novel_flag = "UNKNOWN"

        # Abundance string: rarest non-exempt element
        non_exempt = {el: CRUSTAL_ABUNDANCE_PPM.get(el, 0) for el in comp if el not in abundance_exempt}
        if non_exempt:
            rare_el = min(non_exempt, key=non_exempt.get)
            ppm = non_exempt[rare_el]
            abundance_str = f"{rare_el}:{ppm:.1f}ppm" if ppm > 0 else f"{rare_el}:trace"
        else:
            abundance_str = "all-exempt"

        results.append({
            "formula": formula,
            "source": "foam-generative",
            "domain": domain,
            "novel_flag": novel_flag,
            "confidence": confidence,
            "properties": {
                "stability_score": r["stability_score"],
                "theta_D_K": r["theta_D_K"],
                "debye_stable": r["debye_stable"],
                "debye_method": r["debye_method"],
                "crystal_system": r["crystal_system"],
                "abundance": abundance_str,
            },
            "phonon_stability": {
                "stable": r["stable"],
                "recommendation": r["recommendation"],
                "theorem": r["theorem"],
            },
        })

    # Sort by stability_score (primary) then theta_D (secondary tiebreaker)
    results.sort(
        key=lambda x: (x["properties"]["stability_score"], x["properties"]["theta_D_K"]),
        reverse=True,
    )
    return results[:limit]

def _detect_domain(query: str) -> str:
    q = query.lower()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(k.lower() in q for k in keywords):
            return domain
    if any(k in q for k in ["n2", "nitrogen", "fixation", "ammonia"]):
        return "nitrogen"
    return "general"

def _load_mp_api_key():
    env_path = os.path.join(os.path.dirname(__file__), "..", "Bella2", ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith("MATERIALS_PROJECT_API_KEY="):
                    return line.split("=", 1)[1].strip()
    return os.environ.get("MATERIALS_PROJECT_API_KEY", "")

# Common element-name → symbol mapping for natural-language queries
_ELEMENT_NAMES = {
    "iron": "Fe",
    "manganese": "Mn",
    "silicon": "Si",
    "titanium": "Ti",
    "carbon": "C",
    "bismuth": "Bi",
    "oxygen": "O",
    "lead": "Pb",
    "cobalt": "Co",
    "nickel": "Ni",
    "lithium": "Li",
}

# All element symbols (including one- and two-letter) so BiFeO3, Ti3C2 etc. are recognised
_ELEMENT_SYMBOLS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac",
    "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm",
    "Yb", "Lu", "Th", "Pa", "U", "Np", "Pu",
}

def _extract_chemsys(query: str) -> str | None:
    """Build an MP chemsys string from whole tokens in the query."""
    tokens = re.findall(r"[A-Za-z0-9]+", query)
    symbols = set()
    for token in tokens:
        lower = token.lower()
        for name, sym in _ELEMENT_NAMES.items():
            if lower == name:
                symbols.add(sym)
        # Strip digits/punctuation and check exact element-symbol matches
        bare = re.sub(r"[^A-Za-z]", "", token)
        if bare in _ELEMENT_SYMBOLS:
            symbols.add(bare)
        # Recognise multi-element formulas like BiFeO3 or Ti3C2
        for sym in _ELEMENT_SYMBOLS:
            if sym in re.findall(r"[A-Z][a-z]?", token):
                symbols.add(sym)
    if not symbols:
        return None
    return "-".join(sorted(symbols))

def _extract_formula(query: str) -> str | None:
    """Extract an explicit formula token like Ti3C2 or BiFeO3 (not a lone element)."""
    for token in query.split():
        token = token.strip(",")
        if re.fullmatch(r"([A-Z][a-z]?[0-9]*)+", token):
            elems = re.findall(r"[A-Z][a-z]?", token)
            # Require at least two different elements OR at least one digit
            if any(c in _ELEMENT_SYMBOLS for c in elems) and (len(set(elems)) >= 2 or re.search(r"\d", token)):
                return token
    return None

VALID_ELEMENTS = {
    'H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl',
    'Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge',
    'As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag',
    'Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm',
    'Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir',
    'Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th','Pa','U',
    'Np','Pu','Mn','Ga','Ge'
}

def _matproj_search(query, limit):
    key = _load_mp_api_key()
    if not key:
        return [{"error": "No MP API key found", "source": "MaterialsProject"}]
    elements = re.findall(r'\b([A-Z][a-z]?)\b', query)
    elements = [e for e in elements if e in VALID_ELEMENTS]
    if not elements:
        return [{"error": "Could not derive element system from query", "query": query, "source": "MaterialsProject"}]
    chemsys = '-'.join(sorted(set(elements)))
    url = "https://api.materialsproject.org/materials/summary/"
    headers = {"X-API-KEY": key}
    params = {
        "chemsys": chemsys,
        "_fields": "formula_pretty,energy_above_hull,band_gap,is_stable,ordering",
        "_limit": limit,
    }
    try:
        r = requests.get(url, headers=headers, params=params, timeout=20)
        if r.status_code == 200:
            return [
                {
                    "formula": e.get("formula_pretty"),
                    "source": "MaterialsProject",
                    "properties": {
                        "energy_above_hull": e.get("energy_above_hull"),
                        "band_gap": e.get("band_gap"),
                        "stable": e.get("is_stable"),
                        "magnetic_ordering": e.get("ordering"),
                    },
                }
                for e in r.json().get("data", [])
            ]
        return [{"error": f"MP API status {r.status_code}", "body": r.text[:200], "source": "MaterialsProject"}]
    except Exception as e:
        return [{"error": str(e), "source": "MaterialsProject"}]

def _gnome_search(query, limit):
    # GNoME integration placeholder: keep as no-op fallback for now
    return [{"note": "GNoME search not implemented for general queries", "query": query}]

CATALYSIS_HUB_URL = "https://api.catalysis-hub.org/graphql"

# Prior-art benchmark catalysts for nitrogen fixation (literature)
LITERATURE_BENCHMARKS = [
    {
        "formula": "Fe3O4",
        "activation_energy": None,
        "reaction": "N2 reduction",
        "source": "literature",
        "properties": {
            "reaction": "N2 reduction",
            "activation_energy": None,
            "note": "industrial Haber-Bosch",
        },
    },
    {
        "formula": "Ru",
        "activation_energy": None,
        "reaction": "N2 reduction",
        "source": "literature",
        "properties": {
            "reaction": "N2 reduction",
            "activation_energy": None,
            "note": "best known ambient",
        },
    },
    {
        "formula": "MoS2",
        "activation_energy": None,
        "reaction": "N2 reduction",
        "source": "literature",
        "properties": {
            "reaction": "N2 reduction",
            "activation_energy": None,
            "note": "2D catalyst",
        },
    },
    {
        "formula": "VN",
        "activation_energy": None,
        "reaction": "N2 reduction",
        "source": "literature",
        "properties": {
            "reaction": "N2 reduction",
            "activation_energy": None,
            "note": "emerging vanadium nitride",
        },
    },
    {
        "formula": "Fe2Mo",
        "activation_energy": None,
        "reaction": "N2 reduction",
        "source": "literature",
        "properties": {
            "reaction": "N2 reduction",
            "activation_energy": None,
            "note": "promising bimetallic",
        },
    },
]


def name() -> str:
    return "Catalysis Hub"


def _clean_formula(s: str) -> str:
    """Best-effort sanitise of catalysis-hub surface strings into a formula."""
    if not s:
        return ""
    s = s.replace("_", "").replace("-", "")
    s = s.split()[0]
    s = s.strip()
    return s


def search(query: str, limit: int = 20) -> list[dict]:
    """Route query to the appropriate domain-specific source."""
    domain = _detect_domain(query)

    if domain == "nitrogen":
        results = list(LITERATURE_BENCHMARKS)
        try:
            gql = """
            {
              reactions(first: 50, reactants: "N2") {
                edges {
                  node {
                    id
                    reactants
                    products
                    activationEnergy
                    reactionEnergy
                    surfaceComposition
                  }
                }
              }
            }
            """
            req = urllib.request.Request(
                CATALYSIS_HUB_URL,
                data=json.dumps({"query": gql}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            for edge in data.get("data", {}).get("reactions", {}).get("edges", []):
                node = edge.get("node", {})
                ae = node.get("activationEnergy")
                if ae is None:
                    continue
                try:
                    ae_val = float(ae)
                except (ValueError, TypeError):
                    continue
                if ae_val >= 0.8:
                    continue

                formula = _clean_formula(node.get("surfaceComposition") or "")
                if not formula:
                    continue

                results.append(
                    {
                        "formula": formula,
                        "activation_energy": ae_val,
                        "reaction": node.get("reaction", "N2 reduction"),
                        "source": "catalysis-hub",
                        "properties": {
                            "activation_energy": ae_val,
                            "reaction": node.get("reaction", "N2 reduction"),
                            "reaction_energy": node.get("reactionEnergy"),
                        },
                    }
                )
        except Exception:
            pass
        return results[:limit]

    if domain in DOMAIN_KEYWORDS or domain == "general":
        return _matproj_search(query, limit)[:limit]

    return _gnome_search(query, limit)[:limit]


def main(argv=None):
    """CLI entry point for Bob search."""
    import argparse
    parser = argparse.ArgumentParser(description="Bob — catalysis hub literature search")
    parser.add_argument("--query", default="N2 reduction", help="search query")
    parser.add_argument("--limit", type=int, default=10, help="max results")
    parser.add_argument("--domain", default=None, help="domain profile (semiconductor, mram, etc.)")
    parser.add_argument("--generative", action="store_true", help="generate novel compositions and screen with foam phonon stability")
    args = parser.parse_args(argv)
    if args.generative:
        results = _generative_screen(args.query, domain=args.domain, limit=args.limit)
    else:
        results = search(args.query, limit=args.limit)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

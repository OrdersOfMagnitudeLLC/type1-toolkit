# OOM Commercial License v1.0
# Copyright 2026 Orders of Magnitude LLC
# See /NS/LICENSE.md for terms.
# Copyright (C) 2026 Orders of Magnitude LLC <orders@ofmagnitude.com>
"""
Materials Project API plugin for Bella
"""
import os

MP_API_KEY = os.environ.get("MATERIALS_PROJECT_API_KEY")

# Try to import mp-api, install if not available
try:
    from mp_api.client import MPRester
except ImportError:
    import subprocess
    subprocess.run(['pip3', 'install', '--break-system-packages', 'mp-api'], check=True, capture_output=True)
    from mp_api.client import MPRester

def name() -> str:
    return "Materials Project"

def parse_query(query: str) -> dict:
    """Parse natural language query to API parameters."""
    import re
    params = {}
    query_lower = query.lower()
    
    # Bandgap filters
    if "wide bandgap" in query_lower or "high bandgap" in query_lower:
        params["band_gap_min"] = "3"
    elif "bandgap" in query_lower:
        params["band_gap_min"] = "1"
    
    # Stability filter
    if "stable" in query_lower:
        params["is_stable"] = "true"
    
    # Element filters - use word boundaries to avoid partial matches
    elements = []
    element_map = {
        "hydrogen": "H", "helium": "He", "lithium": "Li", "beryllium": "Be", "boron": "B",
        "carbon": "C", "nitrogen": "N", "oxygen": "O", "fluorine": "F", "neon": "Ne",
        "sodium": "Na", "magnesium": "Mg", "aluminum": "Al", "aluminium": "Al", "silicon": "Si",
        "phosphorus": "P", "sulfur": "S", "chlorine": "Cl", "argon": "Ar", "potassium": "K",
        "calcium": "Ca", "scandium": "Sc", "titanium": "Ti", "vanadium": "V", "chromium": "Cr",
        "manganese": "Mn", "iron": "Fe", "cobalt": "Co", "nickel": "Ni", "copper": "Cu",
        "zinc": "Zn", "gallium": "Ga", "germanium": "Ge", "arsenic": "As", "selenium": "Se",
        "bromine": "Br", "krypton": "Kr", "rubidium": "Rb", "strontium": "Sr", "yttrium": "Y",
        "zirconium": "Zr", "niobium": "Nb", "molybdenum": "Mo", "technetium": "Tc", "ruthenium": "Ru",
        "rhodium": "Rh", "palladium": "Pd", "silver": "Ag", "cadmium": "Cd", "indium": "In",
        "tin": "Sn", "antimony": "Sb", "tellurium": "Te", "iodine": "I", "xenon": "Xe",
        "cesium": "Cs", "barium": "Ba", "lanthanum": "La", "cerium": "Ce", "praseodymium": "Pr",
        "neodymium": "Nd", "promethium": "Pm", "samarium": "Sm", "europium": "Eu", "gadolinium": "Gd",
        "terbium": "Tb", "dysprosium": "Dy", "holmium": "Ho", "erbium": "Er", "thulium": "Tm",
        "ytterbium": "Yb", "lutetium": "Lu", "hafnium": "Hf", "tantalum": "Ta", "tungsten": "W",
        "rhenium": "Re", "osmium": "Os", "iridium": "Ir", "platinum": "Pt", "gold": "Au",
        "mercury": "Hg", "thallium": "Tl", "lead": "Pb", "bismuth": "Bi", "polonium": "Po",
        "astatine": "At", "radon": "Rn", "francium": "Fr", "radium": "Ra", "actinium": "Ac",
        "thorium": "Th", "protactinium": "Pa", "uranium": "U", "neptunium": "Np", "plutonium": "Pu",
        "americium": "Am", "curium": "Cm", "berkelium": "Bk", "californium": "Cf", "einsteinium": "Es",
        "fermium": "Fm"
    }
    
    # Also check for direct element symbols (case-sensitive)
    for symbol in ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
                   "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
                   "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
                   "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
                   "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
                   "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
                   "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
                   "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
                   "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
                   "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm"]:
        if re.search(r'\b' + re.escape(symbol) + r'\b', query):
            elements.append(symbol)
    
    # Check for element names
    for name, symbol in element_map.items():
        if re.search(r'\b' + re.escape(name) + r'\b', query_lower):
            if symbol not in elements:
                elements.append(symbol)
    
    if elements:
        params["elements"] = ",".join(elements)
    
    # Metal/conductor proxy
    if "superconductor" in query_lower or "metal" in query_lower:
        params["is_metal"] = "true"
    
    return params

def search(query: str, limit: int = 50) -> list[dict]:
    """Search Materials Project v2 API directly via HTTP GET."""
    import re, requests, sys
    try:
        query = query.strip()
        key = os.environ.get('MATERIALS_PROJECT_API_KEY', '')
        if not key:
            print("MP API Error: MATERIALS_PROJECT_API_KEY not set", file=sys.stderr)
            return []

        # Detect whether the query is already a formula (e.g. "SiHF3",
        # "KCa(FeP)4", "Na4Ca(SiS3)2") versus a keyword/element search.
        is_formula = bool(
            re.search(r'[A-Z][a-z]?[0-9]', query) or
            re.search(r'[A-Z]{2}', query) or
            re.match(r'^[A-Z][a-z]?$', query)
        )

        params = {
            '_limit': min(limit, 1000),
            '_fields': 'material_id,formula_pretty,band_gap,density,energy_above_hull,structure',
        }
        if is_formula:
            params['formula'] = query
        else:
            elements = [e.strip() for e in query.split(',') if e.strip()]
            if not elements:
                return []
            params['elements'] = ','.join(elements)

        url = 'https://api.materialsproject.org/materials/summary/'
        headers = {'X-API-KEY': key, 'Accept': 'application/json'}
        r = requests.get(url, params=params, headers=headers, timeout=60)
        r.raise_for_status()
        data = r.json().get('data', [])

        out = []
        for item in data[:limit]:
            out.append({
                'formula': item.get('formula_pretty', ''),
                'material_id': item.get('material_id', ''),
                'source': 'materials_project',
                'cif_content': None,
                'properties': {
                    'material_id': item.get('material_id', ''),
                    'band_gap': item.get('band_gap'),
                    'density': item.get('density'),
                    'energy_above_hull': item.get('energy_above_hull'),
                }
            })

        return out
    except Exception as e:
        print(f"MP API Error: {e}", file=sys.stderr)
        return []

def fetch_cif(material_id: str) -> str:
    """Fetch CIF content for a material ID using mp-api."""
    try:
        with MPRester(MP_API_KEY) as mpr:
            structure = mpr.get_structure_by_material_id(material_id)
            cif = structure.to(fmt="cif")
            return cif
    except Exception as e:
        import sys
        print(f"Error fetching CIF for {material_id}: {e}", file=sys.stderr)
        return ""

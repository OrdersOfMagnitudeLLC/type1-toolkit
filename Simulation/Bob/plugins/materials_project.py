# Copyright (c) 2026 Orders of Magnitude LLC
# Licensed under the OOM Commercial License v1.0
# See LICENSE.md in the repository root or ofmagnitude.com

"""
Materials Project API plugin for Bob
"""
import requests
from pathlib import Path
from typing import List, Dict, Any
import os

API_KEY = os.environ.get("MATERIALS_PROJECT_API_KEY")
BASE_URL = "https://api.materialsproject.org/materials/summary"

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
        "sodium": "Na", "magnesium": "Mg", "aluminum": "Al", "silicon": "Si", "phosphorus": "P",
        "sulfur": "S", "chlorine": "Cl", "argon": "Ar", "potassium": "K", "calcium": "Ca",
        "scandium": "Sc", "titanium": "Ti", "vanadium": "V", "chromium": "Cr", "manganese": "Mn",
        "iron": "Fe", "cobalt": "Co", "nickel": "Ni", "copper": "Cu", "zinc": "Zn",
        "gallium": "Ga", "germanium": "Ge", "arsenic": "As", "selenium": "Se", "bromine": "Br",
        "krypton": "Kr", "rubidium": "Rb", "strontium": "Sr", "yttrium": "Y", "zirconium": "Zr",
        "niobium": "Nb", "molybdenum": "Mo", "technetium": "Tc", "ruthenium": "Ru", "rhodium": "Rh",
        "palladium": "Pd", "silver": "Ag", "cadmium": "Cd", "indium": "In", "tin": "Sn",
        "antimony": "Sb", "tellurium": "Te", "iodine": "I", "xenon": "Xe", "cesium": "Cs",
        "barium": "Ba", "lanthanum": "La", "cerium": "Ce", "praseodymium": "Pr", "neodymium": "Nd",
        "promethium": "Pm", "samarium": "Sm", "europium": "Eu", "gadolinium": "Gd", "terbium": "Tb",
        "dysprosium": "Dy", "holmium": "Ho", "erbium": "Er", "thulium": "Tm", "ytterbium": "Yb",
        "lutetium": "Lu", "hafnium": "Hf", "tantalum": "Ta", "tungsten": "W", "rhenium": "Re",
        "osmium": "Os", "iridium": "Ir", "platinum": "Pt", "gold": "Au", "mercury": "Hg",
        "thallium": "Tl", "lead": "Pb", "bismuth": "Bi", "polonium": "Po", "astatine": "At",
        "radon": "Rn", "francium": "Fr", "radium": "Ra", "actinium": "Ac", "thorium": "Th",
        "protactinium": "Pa", "uranium": "U", "neptunium": "Np", "plutonium": "Pu", "americium": "Am",
        "curium": "Cm", "berkelium": "Bk", "californium": "Cf", "einsteinium": "Es", "fermium": "Fm",
        "mendelevium": "Md", "nobelium": "No", "lawrencium": "Lr"
    }
    
    # Check for element symbols first (more precise)
    for symbol in ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"]:
        if re.search(r'\b' + re.escape(symbol) + r'\b', query):
            elements.append(symbol)
    
    # Then check for element names
    for name, symbol in element_map.items():
        if re.search(r'\b' + re.escape(name) + r'\b', query_lower):
            if symbol not in elements:
                elements.append(symbol)
    
    if elements:
        params["elements"] = ",".join(elements)
    
    # Metal/semiconductor filter
    if "superconductor" in query_lower or "metal" in query_lower:
        params["is_metal"] = "true"
    
    return params

def search(query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Search Materials Project API for materials matching the query
    
    Args:
        query: Natural language search query
        limit: Maximum number of results to return
        
    Returns:
        List of material dictionaries with properties
    """
    params = parse_query(query)
    
    # Default fields to request
    params["_fields"] = "material_id,formula_pretty,band_gap,energy_above_hull,formation_energy_per_atom,density,is_stable,is_metal"
    params["_limit"] = str(limit)
    
    headers = {
        "X-API-KEY": API_KEY,
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0"
    }
    
    try:
        response = requests.get(BASE_URL, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        results = []
        if isinstance(data, dict) and "data" in data:
            for item in data["data"][:limit]:
                properties = {
                    "band_gap": item.get("band_gap"),
                    "energy_above_hull": item.get("energy_above_hull"),
                    "formation_energy_per_atom": item.get("formation_energy_per_atom"),
                    "density": item.get("density"),
                    "is_stable": item.get("is_stable"),
                    "is_metal": item.get("is_metal")
                }
                
                # Try to find CIF in local data directory - use formula
                cif_dir = Path.home() / "NS/Bob/data/cifs"
                formula = item.get("formula_pretty", "unknown")
                material_id = item.get("material_id", "unknown")
                
                # CIFs are named by formula in data/cifs
                cif_path = cif_dir / f"{formula}.cif"
                
                # Fetch CIF from API if not local
                cif_content = None
                if cif_path.exists():
                    with open(cif_path, 'r') as f:
                        cif_content = f.read()
                else:
                    # Try to fetch from API
                    cif_content = fetch_cif(material_id)
                    if not cif_content:
                        cif_content = build_cif_from_structure(material_id)
                    
                    # Save to disk if available
                    if cif_content:
                        mp_cif_dir = "/tmp/bella_mp_cifs"
                        os.makedirs(mp_cif_dir, exist_ok=True)
                        mp_cif_path = os.path.join(mp_cif_dir, f"{material_id}.cif")
                        with open(mp_cif_path, 'w') as f:
                            f.write(cif_content)
                
                results.append({
                    "formula": formula,
                    "material_id": material_id,
                    "cif_content": cif_content,
                    "properties": properties,
                    "source": "materials_project"
                })
        
        return results
        
    except requests.exceptions.RequestException as e:
        print(f"Materials Project API error: {e}")
        return []

def fetch_cif(material_id: str) -> str:
    """Fetch CIF content for a material ID."""
    try:
        url = f"https://api.materialsproject.org/materials/{material_id}/robocrys/"
        headers = {"X-API-KEY": API_KEY}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        if "data" in data and data["data"]:
            return data["data"]
        return ""
    except Exception:
        return ""

def build_cif_from_structure(material_id: str) -> str:
    """Build minimal CIF from structure data."""
    try:
        url = f"https://api.materialsproject.org/materials/{material_id}/"
        headers = {"X-API-KEY": API_KEY}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        if "data" not in data:
            return ""
        
        mat = data["data"]
        structure = mat.get("structure")
        if not structure:
            return ""
        
        # Build minimal CIF from lattice and sites
        lattice = structure.get("lattice", {})
        a = lattice.get("a", 1.0)
        b = lattice.get("b", 1.0)
        c = lattice.get("c", 1.0)
        alpha = lattice.get("alpha", 90.0)
        beta = lattice.get("beta", 90.0)
        gamma = lattice.get("gamma", 90.0)
        
        sites = structure.get("sites", [])
        
        cif_lines = [
            f"data_{material_id}",
            "_cell_length_a",
            f" {a:.6f}",
            "_cell_length_b",
            f" {b:.6f}",
            "_cell_length_c",
            f" {c:.6f}",
            "_cell_angle_alpha",
            f" {alpha:.6f}",
            "_cell_angle_beta",
            f" {beta:.6f}",
            "_cell_angle_gamma",
            f" {gamma:.6f}",
            "_symmetry_space_group_name_H-M",
            " P1",
            "_symmetry_Int_Tables_number",
            " 1",
            "loop_",
            "_atom_site_label",
            "_atom_site_type_symbol",
            "_atom_site_fract_x",
            "_atom_site_fract_y",
            "_atom_site_fract_z",
        ]
        
        for i, site in enumerate(sites):
            element = site.get("species", [{}])[0].get("element", "X")
            xyz = site.get("xyz", [0, 0, 0])
            cif_lines.append(f" {i+1} {element} {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}")
        
        return "\n".join(cif_lines)
    except Exception:
        return ""

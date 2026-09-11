# Copyright (c) 2026 Orders of Magnitude LLC
# Licensed under the OOM Commercial License v1.0
# See LICENSE.md in the repository root or ofmagnitude.com

import os
"""
GNoME CSV plugin for Bob
Searches the GNoME materials database
"""
import csv
from pathlib import Path
from typing import List, Dict, Any
import re

CSV_PATH = os.path.join(os.path.expanduser('~'), "NS/Bob/data/gnome.csv")

def name() -> str:
    return "GNoME"

def parse_query(query: str) -> dict:
    """Parse natural language query to CSV filter parameters."""
    params = {}
    query_lower = query.lower()
    
    # Bandgap filters
    if "wide bandgap" in query_lower or "high bandgap" in query_lower:
        params["bandgap_min"] = 3.0
    elif "bandgap" in query_lower:
        params["bandgap_min"] = 1.0
    
    # Stability filter
    if "stable" in query_lower:
        params["formation_energy_max"] = 0.0
    
    # Density filters
    if "light" in query_lower or "low density" in query_lower:
        params["density_max"] = 2.0
    elif "heavy" in query_lower or "high density" in query_lower:
        params["density_min"] = 5.0
    
    # Element filters
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
    
    elements = []
    for symbol in ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"]:
        if re.search(r'\b' + re.escape(symbol) + r'\b', query):
            elements.append(symbol)
    
    for name, symbol in element_map.items():
        if re.search(r'\b' + re.escape(name) + r'\b', query_lower):
            if symbol not in elements:
                elements.append(symbol)
    
    if elements:
        params["elements"] = elements
    
    return params

def search(query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Search GNoME CSV for materials matching the query
    
    Args:
        query: Natural language search query
        limit: Maximum number of results to return
        
    Returns:
        List of material dictionaries with properties
    """
    csv_file = Path(CSV_PATH)
    if not csv_file.exists():
        print(f"GNoME CSV not found at {CSV_PATH}")
        return []
    
    params = parse_query(query)
    results = []
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                if len(results) >= limit:
                    break
                
                try:
                    # Parse numeric fields, treating blanks/None as 0
                    def _float(v):
                        return float(v) if v not in (None, '') else 0.0
                    bandgap = _float(row.get("Bandgap"))
                    density = _float(row.get("Density"))
                    formation_energy = _float(row.get("Formation Energy Per Atom"))
                    
                    # Apply filters
                    if "bandgap_min" in params and bandgap < params["bandgap_min"]:
                        continue
                    if "density_max" in params and density > params["density_max"]:
                        continue
                    if "density_min" in params and density < params["density_min"]:
                        continue
                    if "formation_energy_max" in params and formation_energy > params["formation_energy_max"]:
                        continue
                    
                    # Element filter
                    if "elements" in params:
                        elements_str = row.get("Elements", "")
                        required_elements = params["elements"]
                        elements_list = [e.strip().strip("'\"") for e in elements_str.strip("[]").split(",")]
                        if not all(el in elements_list for el in required_elements):
                            continue
                    
                    formula = row.get("Reduced Formula", row.get("Composition", "unknown"))
                    material_id = row.get("MaterialId", "unknown")
                    
                    # Check for CIF file - use formula (CIFs are named by formula in data/cifs)
                    cif_dir = Path.home() / "NS/Bob/data/cifs"
                    cif_path = cif_dir / f"{formula}.cif"
                    
                    # Only include results with CIF files
                    if not cif_path.exists():
                        continue
                    
                    properties = {
                        "band_gap": bandgap,
                        "formation_energy": formation_energy,
                        "density": density
                    }
                    
                    results.append({
                        "formula": formula,
                        "material_id": material_id,
                        "cif_content": str(cif_path),
                        "properties": properties,
                        "source": "gnome"
                    })
                    
                except (ValueError, KeyError) as e:
                    continue
        
        # Sort by bandgap descending if applicable
        if "bandgap_min" in params:
            results.sort(key=lambda x: x["properties"]["band_gap"], reverse=True)
        
        return results[:limit]
        
    except Exception as e:
        print(f"Error reading GNoME CSV: {e}")
        return []

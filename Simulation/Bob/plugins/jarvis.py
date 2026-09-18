# Copyright (c) 2026 Orders of Magnitude LLC
# Licensed under the OOM Commercial License v1.0
# See LICENSE.md in the repository root or ofmagnitude.com

"""
JARVIS-DFT plugin for Bob
Searches the JARVIS-DFT materials database
"""
import requests
from typing import List, Dict, Any
import re

# JARVIS API endpoints
JARVIS_API_URL = "https://jarvis.nist.gov/jarvisdft/search/"

def name() -> str:
    return "JARVIS-DFT"

def parse_query(query: str) -> dict:
    """Parse natural language query to API parameters."""
    params = {}
    query_lower = query.lower()
    
    # Bandgap filters
    if "wide bandgap" in query_lower or "high bandgap" in query_lower:
        params["bandgap_min"] = 3.0
    elif "bandgap" in query_lower:
        params["bandgap_min"] = 1.0
    
    # Stability filter
    if "stable" in query_lower:
        params["energy_above_hull_max"] = 0.1
    
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
    Search JARVIS-DFT API for materials matching the query
    
    Args:
        query: Natural language search query
        limit: Maximum number of results to return
        
    Returns:
        List of material dictionaries with properties
    """
    params = parse_query(query)
    results = []
    
    try:
        # Try GET request with search_term parameter
        response = requests.get(
            JARVIS_API_URL,
            params={'search_term': query, 'dataset': 'dft_3d'},
            timeout=30
        )
        
        if response.status_code == 200:
            # Parse HTML response
            html = response.text
            
            # Try to extract data from HTML table
            # Look for table rows with formula, bandgap, formation_energy
            import re
            table_pattern = r'<tr>.*?<td>(.*?)</td>.*?<td>(.*?)</td>.*?<td>(.*?)</td>'
            matches = re.findall(table_pattern, html, re.DOTALL)
            
            for match in matches[:limit]:
                formula = match[0].strip()
                bandgap = None
                formation_energy = None
                
                try:
                    if match[1].strip():
                        bandgap = float(match[1].strip())
                except ValueError:
                    pass
                
                try:
                    if match[2].strip():
                        formation_energy = float(match[2].strip())
                except ValueError:
                    pass
                
                results.append({
                    'formula': formula,
                    'material_id': formula,
                    'properties': {
                        'band_gap': bandgap,
                        'formation_energy': formation_energy
                    },
                    'source': 'jarvis'
                })
        else:
            print(f"JARVIS API returned status {response.status_code}")
            
    except Exception as e:
        print(f"JARVIS API error: {e}")
    
    return results

# OOM Commercial License v1.0
# Copyright 2026 Orders of Magnitude LLC
# See ./LICENSE.md for terms.
# Copyright (C) 2026 Orders of Magnitude LLC <orders@ofmagnitude.com>
"""
GNoME CSV plugin for Bella
"""
import csv
from pathlib import Path

def name() -> str:
    return "GNoME"

def parse_query(query: str) -> dict:
    """Parse natural language query to filter parameters."""
    params = {}
    query_lower = query.lower()
    
    # Bandgap filters
    if "high bandgap" in query_lower or "wide bandgap" in query_lower:
        params["bandgap_min"] = 3.0
    elif "bandgap" in query_lower:
        params["bandgap_min"] = 1.0
    
    # Density filters
    if "low density" in query_lower:
        params["density_max"] = 3.0
    elif "high density" in query_lower:
        params["density_min"] = 5.0
    
    # Formation energy filter (stability)
    if "stable" in query_lower:
        params["formation_energy_max"] = 0.0
    
    return params

def search(query: str, limit: int = 5) -> list[dict]:
    """Search local GNoME CSV file for materials."""
    gnome_csv = Path.home() / "NS/Bob/data/gnome.csv"
    if not gnome_csv.exists():
        return []
    
    try:
        params = parse_query(query)
        results = []
        
        with open(gnome_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if len(results) >= limit:
                    break
                
                # Apply filters
                try:
                    bandgap = float(row.get("Bandgap", 0))
                    density = float(row.get("Density", 0))
                    formation_energy = float(row.get("Formation Energy Per Atom", 0))
                except (ValueError, TypeError):
                    continue
                
                # Bandgap filter
                if "bandgap_min" in params and bandgap < params["bandgap_min"]:
                    continue
                
                # Density filter
                if "density_max" in params and density > params["density_max"]:
                    continue
                if "density_min" in params and density < params["density_min"]:
                    continue
                
                # Formation energy filter
                if "formation_energy_max" in params and formation_energy > params["formation_energy_max"]:
                    continue
                
                # Get formula
                formula = row.get("Reduced Formula", row.get("Composition", "unknown"))
                material_id = row.get("MaterialId", "unknown")
                
                # Check if CIF exists
                cif_dir = Path.home() / "NS/Bob/data/cifs"
                cif_path = cif_dir / f"{material_id}.cif"
                
                results.append({
                    "formula": formula,
                    "cif_content": str(cif_path) if cif_path.exists() else None,
                    "source": "gnome",
                    "properties": {
                        "material_id": material_id,
                        "band_gap": bandgap,
                        "formation_energy": formation_energy,
                        "density": density
                    }
                })
        
        return results
    except Exception as e:
        return []

def fetch_cif(material_id: str) -> str:
    """Fetch CIF content for a GNoME material."""
    cif_dir = Path.home() / "NS/Bob/data/cifs"
    cif_path = cif_dir / f"{material_id}.cif"
    if cif_path.exists():
        with open(cif_path, 'r') as f:
            return f.read()
    return ""

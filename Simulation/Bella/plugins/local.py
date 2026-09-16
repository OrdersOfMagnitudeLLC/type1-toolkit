# OOM Commercial License v1.0
# Copyright 2026 Orders of Magnitude LLC
# See /NS/LICENSE.md for terms.
# Copyright (C) 2026 Orders of Magnitude LLC <orders@ofmagnitude.com>
"""
Local file plugin for Bella
"""
from pathlib import Path

def name() -> str:
    return "Local Files"

def search(query: str, limit: int = 5) -> list[dict]:
    """Search local folder for CIF/PDB/XYZ files."""
    # Search in common locations
    search_paths = [
        Path.home() / "NS/Bob/data/cifs",
        Path("/tmp/bella_fetch"),
        Path("/tmp/bella_mp"),
        Path(".")
    ]
    
    results = []
    for search_path in search_paths:
        if not search_path.exists():
            continue
        
        # Search for CIF, PDB, XYZ files
        for ext in ["*.cif", "*.pdb", "*.xyz"]:
            for file_path in search_path.glob(ext):
                if len(results) >= limit:
                    break
                
                # Simple query matching on filename
                if query.lower() in file_path.stem.lower() or not query:
                    results.append({
                        "formula": file_path.stem,
                        "cif_content": None,
                        "source": "local",
                        "properties": {"path": str(file_path)}
                    })
    
    return results

def fetch_cif(material_id: str) -> str:
    """Fetch CIF content from local file."""
    # material_id is the file path
    try:
        with open(material_id, 'r') as f:
            return f.read()
    except Exception:
        return ""

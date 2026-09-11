# Copyright (c) 2026 Orders of Magnitude LLC
# Licensed under the OOM Commercial License v1.0
# See LICENSE.md in the repository root or ofmagnitude.com

import os
#!/usr/bin/env python3
"""
Materials Project API lookup script
Queries specific formulas for material properties
"""

import requests
import time

API_KEY = os.environ.get("MATERIALS_PROJECT_API_KEY")
BASE_URL = "https://api.materialsproject.org/materials/summary"

# Query formulas - targeted GNoME candidates
FORMULAS = [
    "NaAlZnF6",
    "Na4Ca(SiS3)2",
    "Na4Mg(SiS3)2",
    "Na4Sr(SiS3)2",
    "Na2SiS2O",
    "Na3Al(SO4)3",
    "Na2Fe(PO3)4",
    "NaCaFeF6"
]

def query_formula(formula):
    """Query Materials Project API for a specific formula"""
    params = {
        "formula": formula,
        "_fields": "material_id,formula_pretty,band_gap,energy_above_hull,theoretical"
    }
    headers = {
        "X-API-KEY": API_KEY,
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0"
    }
    
    try:
        response = requests.get(BASE_URL, params=params, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error querying {formula}: {e}")
        return None

def main():
    print("Materials Project API Lookup")
    print("=" * 60)
    print(f"API Key: {API_KEY}")
    print(f"Querying {len(FORMULAS)} formulas\n")
    
    for formula in FORMULAS:
        print(f"\n--- Querying: {formula} ---")
        result = query_formula(formula)
        
        if result:
            print(f"Results for {formula}:")
            if isinstance(result, dict) and "data" in result:
                data = result["data"]
                if data:
                    for item in data:
                        print(f"  Material ID: {item.get('material_id', 'N/A')}")
                        print(f"  Pretty Formula: {item.get('formula_pretty', 'N/A')}")
                        print(f"  Band Gap: {item.get('band_gap', 'N/A')} eV")
                        print(f"  Energy Above Hull: {item.get('energy_above_hull', 'N/A')} eV/atom")
                        print(f"  Theoretical: {item.get('theoretical', 'N/A')}")
                        print()
                else:
                    print("  No data found")
            else:
                print(f"  Raw response: {result}")
        else:
            print("  Query failed")
        
        # Sleep to respect rate limits
        time.sleep(0.1)

if __name__ == "__main__":
    main()

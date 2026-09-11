# Copyright (c) 2026 Orders of Magnitude LLC
# Licensed under the OOM Commercial License v1.0
# See LICENSE.md in the repository root or ofmagnitude.com

#!/usr/bin/env python3
"""
Download CIF files for specific GNoME materials
"""

import csv
import requests
import os

# Target formulas to search for
TARGET_FORMULAS = [
    "Na4Ca(SiS3)2",
    "Na4Mg(SiS3)2", 
    "NaAlZnF6",
    "Na3Al(SO4)3",
    "NaCaFeF6"
]

CSV_PATH = os.path.join(os.path.expanduser('~'), "NS/Bob/data/gnome.csv")
CIF_DIR = os.path.join(os.path.expanduser('~'), "NS/Bob/data/cifs")
BASE_URL = "https://storage.googleapis.com/gdm_materials_discovery/gnome_data/by_id/"

def main():
    print("Loading GNoME CSV...")
    
    # Create CIF directory if it doesn't exist
    os.makedirs(CIF_DIR, exist_ok=True)
    
    # Search for target formulas
    formula_to_id_dir = {}
    
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        
        for row in reader:
            if len(row) < 24:
                continue
            
            material_id = row[2]  # MaterialId (index 2)
            reduced_formula = row[3]  # Reduced Formula (index 3)
            data_directory = row[23]  # Data Directory (index 23)
            
            if reduced_formula in TARGET_FORMULAS:
                if reduced_formula not in formula_to_id_dir:
                    formula_to_id_dir[reduced_formula] = (material_id, data_directory)
                    print(f"Found: {reduced_formula} → {material_id}")
                    print(f"  Data Directory: {data_directory}")
    
    print(f"\nFound {len(formula_to_id_dir)} matches\n")
    
    # Download CIF files using simple by_id approach
    for formula, (material_id, data_directory) in formula_to_id_dir.items():
        # Try the simple by_id URL format
        url = f"https://storage.googleapis.com/gdm_materials_discovery/gnome_data/by_id/{material_id}.cif"
        output_path = os.path.join(CIF_DIR, f"{material_id}.cif")
        
        print(f"Downloading {formula} ({material_id})...")
        print(f"  URL: {url}")
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            print(f"  ✓ Saved to {output_path}")
        except requests.exceptions.RequestException as e:
            print(f"  ✗ Failed: {e}")
    
    print("\nDone.")

if __name__ == "__main__":
    main()

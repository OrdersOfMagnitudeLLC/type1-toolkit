# Copyright (c) 2026 Orders of Magnitude LLC
# Licensed under the OOM Commercial License v1.0
# See LICENSE.md in the repository root or ofmagnitude.com

#!/usr/bin/env python3
import os
"""
Analyze specific semiconductor formulas from GNoME dataset
"""

import csv

CSV_PATH = os.path.join(os.path.expanduser('~'), "NS/Bob/data/gnome.csv")
TARGET_FORMULAS = ["SrMgMn7O16", "SrMn6Al2O16", "Sr4MnFe3O10"]

with open(CSV_PATH, 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    
    # Find column indices
    material_id_idx = header.index('MaterialId')
    formula_idx = header.index('Reduced Formula')
    space_group_idx = header.index('Space Group')
    crystal_system_idx = header.index('Crystal System')
    bandgap_idx = header.index('Bandgap')
    nsites_idx = header.index('NSites')
    volume_idx = header.index('Volume')
    
    print(f"{'Formula':<20} {'MaterialId':<15} {'Space Group':<15} {'Crystal System':<15} {'Bandgap (eV)':<12} {'NSites':<8} {'Volume (Å³)':<12}")
    print("-" * 110)
    
    for row in reader:
        formula = row[formula_idx]
        if formula in TARGET_FORMULAS:
            material_id = row[material_id_idx]
            space_group = row[space_group_idx]
            crystal_system = row[crystal_system_idx]
            bandgap = row[bandgap_idx]
            nsites = row[nsites_idx]
            volume = row[volume_idx]
            
            print(f"{formula:<20} {material_id:<15} {space_group:<15} {crystal_system:<15} {bandgap:<12} {nsites:<8} {volume:<12}")

# Copyright (c) 2026 Orders of Magnitude LLC
# Licensed under the OOM Commercial License v1.0
# See LICENSE.md in the repository root or ofmagnitude.com

#!/usr/bin/env python3
import os
"""
Find material IDs for specific formulas in gnome.csv
"""

import csv

CSV_PATH = os.path.join(os.path.expanduser('~'), "NS/Bob/data/gnome.csv")
TARGET_FORMULAS = ["NaZnFeF6", "NaZnCrF6"]

with open(CSV_PATH, 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    
    # Find column indices
    material_id_idx = header.index('MaterialId')
    formula_idx = header.index('Reduced Formula')
    
    for row in reader:
        formula = row[formula_idx]
        if formula in TARGET_FORMULAS:
            material_id = row[material_id_idx]
            print(f"{formula} -> {material_id}")

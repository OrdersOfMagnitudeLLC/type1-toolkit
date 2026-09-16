# Copyright (c) 2026 Orders of Magnitude LLC
# Licensed under the OOM Commercial License v1.0
# See LICENSE.md in the repository root or ofmagnitude.com

#!/usr/bin/env python3
"""Pre-compute ZINC property cache — one-time cost, then zinc_screen is pure dict lookups."""

import pickle
import math
import os
import time
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

import sys
ZINC_SMI = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'zinc_EAAE.smi')
OUT_CACHE = sys.argv[2] if len(sys.argv) > 2 else 'zinc_property_cache.pkl'

records = []
failed = 0
parsed = 0
t0 = time.time()

with open(ZINC_SMI) as f:
    for line in f:
        parts = line.strip().split()
        if not parts:
            continue
        smiles = parts[0]
        parsed += 1

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            failed += 1
            continue

        mw = Descriptors.MolWt(mol)
        if not (150 <= mw <= 550):
            continue

        logp = Descriptors.MolLogP(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)
        arom = rdMolDescriptors.CalcNumAromaticRings(mol)
        rings = rdMolDescriptors.CalcNumRings(mol)

        # Pre-compute r_ligand from volume
        r_ligand = (3 * mw * 1.2 / (4 * math.pi)) ** (1 / 3)

        # Pre-apply pharmacophore gate
        pharm_pass = (arom >= 1 and hba >= 1 and hbd <= 2
                      and rotb <= 10 and rings >= 2)

        # Lipinski pre-filter
        lipinski_pass = logp < 5 and mw < 550

        if not pharm_pass or not lipinski_pass:
            continue

        records.append({
            'smiles': smiles,
            'mw': round(mw, 2),
            'logp': round(logp, 2),
            'hbd': hbd,
            'hba': hba,
            'rotb': rotb,
            'arom': arom,
            'rings': rings,
            'r_ligand': round(r_ligand, 3),
        })

        if parsed % 100000 == 0:
            elapsed = time.time() - t0
            print(f"  {parsed} parsed, {len(records)} passed, "
                  f"{elapsed:.0f}s elapsed")

with open(OUT_CACHE, 'wb') as f:
    pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)

elapsed = time.time() - t0
print(f"\nTotal parsed: {parsed}")
print(f"Failed SMILES: {failed}")
print(f"Passed all filters: {len(records)}")
print(f"Cache written: {OUT_CACHE}")
print(f"Runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")

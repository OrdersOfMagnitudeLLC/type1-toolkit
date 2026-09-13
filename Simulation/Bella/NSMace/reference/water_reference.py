# Copyright (c) 2026 Orders of Magnitude LLC
# Licensed under the OOM Commercial License v1.0
# See LICENSE.md in the repository root or ofmagnitude.com

from mace.calculators import mace_mp
from ase import Atoms
import numpy as np
import json

# Water molecule: known geometry
mol = Atoms(
    symbols=['O', 'H', 'H'],
    positions=[
        [0.000,  0.000,  0.119],
        [0.000,  0.763, -0.477],
        [0.000, -0.763, -0.477]
    ]
)

calc = mace_mp(model="small", dispersion=False, default_dtype="float64", device="cpu")
mol.calc = calc
energy = mol.get_potential_energy()
forces = mol.get_forces()

result = {
    "energy_eV": float(energy),
    "forces_eV_per_A": forces.tolist(),
    "positions": mol.get_positions().tolist(),
    "symbols": list(mol.get_chemical_symbols())
}
print(json.dumps(result, indent=2))

with open("../reference/water_H2O.json", "w") as f:
    json.dump(result, f, indent=2)

print("Reference saved.")

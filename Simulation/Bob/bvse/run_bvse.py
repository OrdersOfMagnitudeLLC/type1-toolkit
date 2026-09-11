# Copyright (c) 2026 Orders of Magnitude LLC
# Licensed under the OOM Commercial License v1.0
# See LICENSE.md in the repository root or ofmagnitude.com

from bvlain import Lain
import os

cif_dir = os.path.join(os.path.expanduser('~'), "NS/Bob/data/cifs")
candidates = {
    "NaZnFeF6":       "NaZnFeF6.cif",
    "NaZnCrF6":       "NaZnCrF6.cif",
    "NaCaFeF6":       "NaCaFeF6.cif",
}

params = {
    'mobile_ion': 'Na1+',
    'r_cut': 10.0,
    'resolution': 0.2,
    'k': 100
}

print(f"{'Material':<20} {'1D (eV)':<10} {'2D (eV)':<10} {'3D (eV)':<10} Rating")
print("-" * 65)

for name, cif in candidates.items():
    path = os.path.join(cif_dir, cif)
    try:
        calc = Lain(verbose=False)
        calc.read_file(path)
        calc.bvse_distribution(**params)
        energies = calc.percolation_barriers(encut=5.0)
        e1d = energies.get('E_1D', 99)
        e2d = energies.get('E_2D', 99)
        e3d = energies.get('E_3D', 99)
        rating = 'SUPERIONIC' if e1d < 0.3 else 'GOOD' if e1d < 0.5 else 'MODERATE' if e1d < 0.7 else 'POOR'
        print(f"{name:<20} {e1d:<10.4f} {e2d:<10.4f} {e3d:<10.4f} {rating}")
    except Exception as e:
        print(f"{name:<20} ERROR: {e}")

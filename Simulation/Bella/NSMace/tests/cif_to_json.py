# Copyright (c) 2026 Orders of Magnitude LLC
# Licensed under the OOM Commercial License v1.0
# See LICENSE.md in the repository root or ofmagnitude.com

import json
import math

# CIF data for SiHF3 (Z=1)
# Cell parameters
a = 4.68558912
b = 4.68558805
c = 4.66612000
alpha = math.radians(109.38707224)
beta = math.radians(109.38706767)
gamma = math.radians(109.55524084)

# Atomic data from CIF (fractional coordinates)
atoms = [
    ("Si", 0.0, 0.0, 0.984932),
    ("H", 0.0, 0.0, 0.672569),
    ("F", 0.665345, 0.0, 0.002615),
    ("F", 0.0, 0.665345, 0.002615),
    ("F", 0.334656, 0.334656, 0.337271)
]

# Element to atomic number mapping
elem_to_Z = {"Si": 14, "H": 1, "F": 9}

# Convert fractional to Cartesian coordinates
# For triclinic cell, use the full transformation matrix
def frac_to_cartesian(frac):
    x, y, z = frac
    # Volume of unit cell
    vol = a * b * c * math.sqrt(1 - math.cos(alpha)**2 - math.cos(beta)**2 - math.cos(gamma)**2 + 2 * math.cos(alpha) * math.cos(beta) * math.cos(gamma))
    
    # Transformation matrix
    ax = a
    ay = b * math.cos(gamma)
    az = c * math.cos(beta)
    bx = 0
    by = b * math.sin(gamma)
    bz = c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / math.sin(gamma)
    cx = 0
    cy = 0
    cz = math.sqrt(c**2 - az**2 - bz**2)
    
    X = ax * x + ay * y + az * z
    Y = bx * x + by * y + bz * z
    Z = cx * x + cy * y + cz * z
    
    return [X, Y, Z]

types = []
positions = []

for elem, fx, fy, fz in atoms:
    types.append(elem_to_Z[elem])
    positions.append(frac_to_cartesian((fx, fy, fz)))

data = {
    "types": types,
    "positions": positions
}

with open("../tests/shifu_unit.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"Generated shifu_unit.json with {len(types)} atoms")
print(f"Cell: a={a}, b={b}, c={c}, alpha={math.degrees(alpha)}, beta={math.degrees(beta)}, gamma={math.degrees(gamma)}")

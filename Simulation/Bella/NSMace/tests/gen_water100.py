# Copyright (c) 2026 Orders of Magnitude LLC
# Licensed under the OOM Commercial License v1.0
# See LICENSE.md in the repository root or ofmagnitude.com

import json

# Generate 100-atom water cluster (33 water molecules = 99 atoms, add 1 more to make 100)
types = []
positions = []

# 33 water molecules
for i in range(33):
    offset = i * 3.0
    types.extend([8, 1, 1])  # O, H, H
    positions.append([offset, 0.0, 0.119])
    positions.append([offset, 0.763, -0.477])
    positions.append([offset, -0.763, -0.477])

# Add one more oxygen to make 100 atoms
types.append(8)
positions.append([99.0, 0.0, 0.119])

data = {
    "types": types,
    "positions": positions
}

with open("../tests/water100.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"Generated water100.json with {len(types)} atoms")

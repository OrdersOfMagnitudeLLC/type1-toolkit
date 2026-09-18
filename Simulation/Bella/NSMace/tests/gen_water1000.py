# Copyright (c) 2026 Orders of Magnitude LLC
# Licensed under the OOM Commercial License v1.0
# See LICENSE.md in the repository root or ofmagnitude.com

import json

# Generate 1000-atom water cluster (333 water molecules = 999 atoms, add 1 more to make 1000)
types = []
positions = []

# 333 water molecules
for i in range(333):
    offset = i * 3.0
    types.extend([8, 1, 1])  # O, H, H
    positions.append([offset, 0.0, 0.119])
    positions.append([offset, 0.763, -0.477])
    positions.append([offset, -0.763, -0.477])

# Add one more oxygen to make 1000 atoms
types.append(8)
positions.append([999.0, 0.0, 0.119])

data = {
    "types": types,
    "positions": positions
}

with open("../tests/water1000.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"Generated water1000.json with {len(types)} atoms")

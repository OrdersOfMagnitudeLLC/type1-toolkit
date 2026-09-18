# Copyright (c) 2026 Orders of Magnitude LLC
# Licensed under the OOM Commercial License v1.0
# See LICENSE.md in the repository root or ofmagnitude.com

import json
import math

# Load unit cell
with open("../tests/shifu_unit.json", "r") as f:
    unit_data = json.load(f)

# Create 2x2x2 supercell (Z=1, so 8 unit cells = 40 atoms)
tiles = 2
types = []
positions = []

for ix in range(tiles):
    for iy in range(tiles):
        for iz in range(tiles):
            offset_x = ix * 4.68558912
            offset_y = iy * 4.68558805
            offset_z = iz * 4.66612000
            for i, elem_type in enumerate(unit_data["types"]):
                types.append(elem_type)
                pos = unit_data["positions"][i]
                positions.append([pos[0] + offset_x, pos[1] + offset_y, pos[2] + offset_z])

data = {
    "types": types,
    "positions": positions
}

with open("../tests/shifu_supercell.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"Generated shifu_supercell.json with {len(types)} atoms ({tiles}x{tiles}x{tiles} tiling)")

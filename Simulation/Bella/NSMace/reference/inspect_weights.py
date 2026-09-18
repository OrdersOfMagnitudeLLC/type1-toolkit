# Copyright (c) 2026 Orders of Magnitude LLC
# Licensed under the OOM Commercial License v1.0
# See LICENSE.md in the repository root or ofmagnitude.com

from mace.calculators import mace_mp
import torch, json

calc = mace_mp(model="small", dispersion=False,
               default_dtype="float64", device="cpu")
model = calc.models[0]

# Print all weight names with shapes
for name, p in model.named_parameters():
    print(f"{name}: {list(p.shape)}")

# Copyright (c) 2026 Orders of Magnitude LLC
# Licensed under the OOM Commercial License v1.0
# See LICENSE.md in the repository root or ofmagnitude.com

from mace.calculators import mace_mp
from ase import Atoms
import numpy as np
import json
import torch

calc = mace_mp(model="small", dispersion=False, default_dtype="float64", device="cpu")
model = calc.models[0]

weights = {}
for name, param in model.named_parameters():
    weights[name] = param.detach().cpu().numpy().tolist()

meta = {
    "param_count": sum(p.numel() for p in model.parameters()),
    "param_names": list(weights.keys())[:20]  # first 20 for inspection
}

with open("../weights/mace_mp_small_weights.json", "w") as f:
    json.dump(weights, f)
with open("../weights/mace_mp_small_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"Exported {meta['param_count']} parameters")
print("First 20 layer names:")
for name in meta['param_names']:
    print(f"  {name}")

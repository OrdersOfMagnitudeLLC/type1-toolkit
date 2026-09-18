# Copyright (c) 2026 Orders of Magnitude LLC
# Licensed under the OOM Commercial License v1.0
# See LICENSE.md in the repository root or ofmagnitude.com

from mace.calculators import mace_mp
import torch, json

calc = mace_mp(model="small", dispersion=False, 
               default_dtype="float64", device="cpu")
model = calc.models[0]

info = {
    "r_max": float(model.r_max),
    "num_bessel": int(model.radial_embedding.out_dim),
    "num_interactions": len(list(model.interactions)),
    "hidden_irreps": str(model.interactions[0].hidden_irreps),
    "node_feats_irreps": str(model.node_embedding.linear.irreps_out),
    "num_elements": int(model.node_embedding.linear.weight.shape[0]),
}

# Get max_ell from spherical harmonics
try:
    info["max_ell"] = model.spherical_harmonics._lmax
except:
    try: info["max_ell"] = model.spherical_harmonics.irreps_out.lmax
    except: pass

# Print all model attributes for inspection
print("=== MODEL INFO ===")
print(json.dumps(info, indent=2))
print("\n=== RADIAL EMBEDDING ===")
print(model.radial_embedding)
print("\n=== INTERACTION 0 ===")
print(model.interactions[0])
print("\n=== PRODUCTS 0 ===")
print(model.products[0])
print("\n=== READOUT ===")
for name, mod in model.named_modules():
    if 'read' in name.lower():
        print(f"{name}: {mod}")

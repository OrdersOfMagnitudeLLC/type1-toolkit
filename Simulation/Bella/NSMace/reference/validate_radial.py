# Copyright (c) 2026 Orders of Magnitude LLC
# Licensed under the OOM Commercial License v1.0
# See LICENSE.md in the repository root or ofmagnitude.com

from mace.calculators import mace_mp
import torch, numpy as np, json

calc = mace_mp(model="small", dispersion=False, default_dtype="float64", device="cpu")
model = calc.models[0]

r_max = float(model.r_max)
num_bessel = model.radial_embedding.out_dim

print(f"r_max={r_max}, num_bessel={num_bessel}")

# Test at r=2.0 Angstrom
r = 2.0
r_tensor = torch.tensor([[r]], dtype=torch.float64)
bessel_out = model.radial_embedding.bessel_fn(r_tensor)
print(f"Bessel at r={r}: {bessel_out.detach().numpy()[0][:5]}")

# Spherical harmonics at (1,0,0) direction
# MACE model uses normalization='component' (verified via model.spherical_harmonics.normalization)
from e3nn import o3
sh = o3.spherical_harmonics([0,1,2,3], 
    torch.tensor([[1.0,0.0,0.0]], dtype=torch.float64), 
    normalize=True, normalization='component')
print(f"SH at (1,0,0): {sh.detach().numpy()[0]}")

with open("findings/radial_reference.json","w") as f:
    json.dump({
        "r_max": r_max,
        "num_bessel": int(num_bessel),
        "bessel_at_r2": bessel_out.detach().numpy()[0].tolist(),
        "sh_at_100": sh.detach().numpy()[0].tolist()
    }, f, indent=2)

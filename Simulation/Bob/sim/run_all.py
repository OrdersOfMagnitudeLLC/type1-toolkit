# Copyright (c) 2026 Orders of Magnitude LLC
# Licensed under the OOM Commercial License v1.0
# See LICENSE.md in the repository root or ofmagnitude.com

"""Batch runner: runs bob_md.py on all CIFs with appropriate routing."""

import os, glob, json, time, argparse
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.expanduser('~'), "NS/Bob/sim"))
from bob_md import insert_na_defect, stability_run, precompute_grid, run_md, load_grid_interpolators, get_mobile_indices, build_supercell

CIF_DIR  = os.path.join(os.path.expanduser('~'), "NS/Bob/data/cifs")
SIM_DIR  = os.path.join(os.path.expanduser('~'), "NS/Bob/sim")
OUT_DIR  = os.path.join(os.path.expanduser('~'), "NS/Bob/results")
MOBILE   = {'Na', 'K', 'Li', 'Ag'}

# Material routing categories (use material IDs from CIF filenames)
ION_CONDUCTORS = [
    "Na4Ca(SiS3)2", "Na4Mg(SiS3)2", "NaCaFeF6", "NaAlZnF6"
]

NEEDS_NA_DEFECT = ["fa37e9edd4", "141d1a7b92"]  # SiHF3, PH(OF)2

STABILITY_ONLY = [
    "aab9f03c85", "a953f4ace2", "91a39d50c2", "79a464894e", "0da4f8a990"
]  # SrMgMn7O16, SrMn6Al2O16, Sr4MnFe3O10, MgS7, SrP2F12

MATERIAL_NAMES = {
    "fa37e9edd4": "SiHF3",
    "141d1a7b92": "PH(OF)2",
    "aab9f03c85": "SrMgMn7O16",
    "a953f4ace2": "SrMn6Al2O16",
    "91a39d50c2": "Sr4MnFe3O10",
    "79a464894e": "MgS7",
    "0da4f8a990": "SrP2F12",
    "18cccdb175": "Na-compound-1",
    "2a29512f75": "Na-compound-2",
    "32a442ed26": "Na-compound-3",
    "400a19fdc5": "Na-compound-4",
    "4bb424d66b": "Na-compound-5",
}

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true', help='Run one material from each category (3 total) with steps=50')
parser.add_argument('--single', type=str, help='Run only the specified material ID')
args = parser.parse_args()

# Load MACE once in parent process
print("[MACE] Loading model...")
from mace.calculators import mace_mp
CALC = mace_mp(model="small", default_dtype="float32", device="cpu")
print("[MACE] Ready.")

# Import ASE just to check which CIFs have mobile ions
from ase.io import read

os.makedirs(OUT_DIR, exist_ok=True)
cifs = sorted(glob.glob(os.path.join(CIF_DIR, "*.cif")))

candidates = []
for cif in cifs:
    try:
        atoms = read(cif)
    except Exception as e:
 print(f" SKIP (unreadable): {os.path.basename(cif)}: {e}")
        continue
    name = os.path.splitext(os.path.basename(cif))[0]
    syms = set(atoms.get_chemical_symbols())
    mobile_found = syms & MOBILE
    
    # Determine category - check explicit lists first
    if name in NEEDS_NA_DEFECT:
        category = "NEEDS_NA_DEFECT"
        candidates.append((cif, name, category, atoms))
        print(f"  CANDIDATE: {name} | {len(atoms)} atoms | {category}")
    elif name in STABILITY_ONLY:
        category = "STABILITY_ONLY"
        candidates.append((cif, name, category, atoms))
        print(f"  CANDIDATE: {name} | {len(atoms)} atoms | {category}")
    elif name in ION_CONDUCTORS:
        category = "ION_CONDUCTOR"
        candidates.append((cif, name, category, atoms))
        print(f"  CANDIDATE: {name} | {len(atoms)} atoms | {category}")
    elif mobile_found:
        category = "ION_CONDUCTOR"
        candidates.append((cif, name, category, atoms))
        print(f"  CANDIDATE: {name} | {len(atoms)} atoms | {category} (auto-detected)")
    else:
        print(f"  SKIP (no mobile ions): {os.path.basename(cif)} | elements: {syms}")

print(f"\n{len(candidates)} materials to simulate\n")

# Filter for single material if requested
if args.single:
    candidates = [c for c in candidates if c[1] == args.single]
    if not candidates:
        print(f"ERROR: Material {args.single} not found")
        sys.exit(1)

# Test mode: run one from each category
if args.test:
    test_candidates = []
    for category in ["ION_CONDUCTOR", "NEEDS_NA_DEFECT", "STABILITY_ONLY"]:
        for cif, name, cat, atoms in candidates:
            if cat == category:
                test_candidates.append((cif, name, cat, atoms))
                break
    candidates = test_candidates
    print(f"[TEST MODE] Running {len(candidates)} materials (one per category)\n")

summary = []
for cif, name, category, atoms in candidates:
    print(f"\n{'='*50}")
    print(f"Running: {name} | {len(atoms)} atoms | {category}")
    print(f"{'='*50}")
    t0 = time.time()
    
    if category == "STABILITY_ONLY":
        steps = 50 if args.test else 200
        result = stability_run(atoms, calc=CALC, steps=steps, T=300.0, material_name=MATERIAL_NAMES.get(name, name))
        status = "OK"
        wall = time.time() - t0
        summary.append({"name": name, "status": status, "wall_s": wall, "result": result})
        print(f"\n  [{status}] {name} finished in {wall/60:.1f} min")
        print(f"  [{name}] RMSD={result['rmsd_angstrom']:.3f} Å stable={result['stable']}")
    
    elif category == "NEEDS_NA_DEFECT":
        # Insert Na defect first
        atoms_defect = insert_na_defect(atoms, n=1)
        
        # Build supercell and get mobile ions
        supercell = build_supercell(atoms_defect, min_size_ang=10.0)
        mobile_idx = get_mobile_indices(supercell)
        mobile_species_set = {'Na'}
        
        if args.test:
            # Test mode: single temperature, 100 steps
            temps = [1000]
            n_steps = 100
        else:
            temps = [900, 1000, 1100]
            n_steps = 2000
        
        sigma_results = []
        for temp in temps:
            # Skip grid for small supercells
            if len(supercell) < 300:
                grid_interps = None
                print(f"  [GRID] Skipping (supercell={len(supercell)} atoms < 300) - using MACE-direct")
            else:
                # Precompute grid
                grid_path = os.path.join(OUT_DIR, f"{name}_grid.npz")
                precompute_grid(supercell, CALC, mobile_species_set, grid_spacing=2.0, batch_size=16, out_path=grid_path, material_name=name)
                grid_interps = load_grid_interpolators(grid_path)
            
            # Run MD
            dt_fs = 4.0
            traj_interval = 100
            formula = MATERIAL_NAMES.get(name, name)
            label = f"{formula}_{temp}K"
            md_result = run_md(supercell, CALC, temp, n_steps, dt_fs, traj_interval, label, grid_interps=grid_interps, out_dir=os.path.join(os.path.expanduser('~'), "NS/Bob/results"))
            
            # Calculate sigma (simplified - just record completion)
            sigma_results.append({
                "temp": temp,
                "status": "OK",
                "msd_A2": md_result.get("msd"),
                "sigma_mS_cm": md_result.get("sigma"),
                "diffusing": md_result.get("diffusing")
            })
        
        status = "OK"
        wall = time.time() - t0
        formula = MATERIAL_NAMES.get(name, name)
        summary.append({"name": name, "formula": formula, "status": status, "wall_s": wall, "sigma_results": sigma_results})
        # Print MSD summary
        for sr in sigma_results:
            if "msd_A2" in sr and sr["msd_A2"] is not None:
                print(f"  [{formula} | {name}] MSD={sr['msd_A2']:.2f} Å² | diffusing={sr['diffusing']}")
        print(f"\n  [{status}] {name} finished in {wall/60:.1f} min")
    
    else:  # ION_CONDUCTOR
        # Build supercell and get mobile ions
        supercell = build_supercell(atoms, min_size_ang=10.0)
        mobile_idx = get_mobile_indices(supercell)
        mobile_species_set = set(atoms.get_chemical_symbols()) & MOBILE
        
        # Skip if no mobile ions found
        if not mobile_species_set:
            print(f"  SKIP (no mobile ions in supercell): {name}")
            continue
        
        if args.test:
            # Test mode: single temperature, 100 steps
            temps = [1000]
            n_steps = 100
        else:
            temps = [900, 1000, 1100]
            n_steps = 2000
        
        sigma_results = []
        for temp in temps:
            # Skip grid for small supercells
            if len(supercell) < 300:
                grid_interps = None
                print(f"  [GRID] Skipping (supercell={len(supercell)} atoms < 300) - using MACE-direct")
            else:
                # Precompute grid
                grid_path = os.path.join(OUT_DIR, f"{name}_grid.npz")
                precompute_grid(supercell, CALC, mobile_species_set, grid_spacing=2.0, batch_size=16, out_path=grid_path, material_name=name)
                grid_interps = load_grid_interpolators(grid_path)
            
            # Run MD
            dt_fs = 4.0
            traj_interval = 100
            formula = MATERIAL_NAMES.get(name, name)
            label = f"{formula}_{temp}K"
            md_result = run_md(supercell, CALC, temp, n_steps, dt_fs, traj_interval, label, grid_interps=grid_interps, out_dir=os.path.join(os.path.expanduser('~'), "NS/Bob/results"))
            
            # Calculate sigma (simplified - just record completion)
            sigma_results.append({
                "temp": temp,
                "status": "OK",
                "msd_A2": md_result.get("msd"),
                "sigma_mS_cm": md_result.get("sigma"),
                "diffusing": md_result.get("diffusing")
            })
        
        status = "OK"
        wall = time.time() - t0
        formula = MATERIAL_NAMES.get(name, name)
        summary.append({"name": name, "formula": formula, "status": status, "wall_s": wall, "sigma_results": sigma_results})
        # Print MSD summary
        for sr in sigma_results:
            if "msd_A2" in sr and sr["msd_A2"] is not None:
                print(f"  [{formula} | {name}] MSD={sr['msd_A2']:.2f} Å² | diffusing={sr['diffusing']}")
        print(f"\n  [{status}] {name} finished in {wall/60:.1f} min")

print(f"\n{'='*50}")
print("BATCH COMPLETE")
print(f"{'='*50}")
for s in summary:
    if 'result' in s and 'rmsd_angstrom' in s['result']:
        print(f"  {s['status']:6s}  {s['name']:30s}  {s['wall_s']/60:.1f} min | RMSD={s['result']['rmsd_angstrom']:.3f} Å stable={s['result']['stable']}")
    elif 'sigma_results' in s:
        temps = ", ".join([str(r['temp']) for r in s['sigma_results']])
        print(f"  {s['status']:6s}  {s['name']:30s}  {s['wall_s']/60:.1f} min | Temps: {temps}K")
    else:
        print(f"  {s['status']:6s}  {s['name']:30s}  {s['wall_s']/60:.1f} min")

with open(os.path.join(OUT_DIR, "batch_summary.json"), "w") as f:
    json.dump(summary, f, indent=2, default=str)

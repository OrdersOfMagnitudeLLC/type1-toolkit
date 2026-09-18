# Copyright (c) 2026 Orders of Magnitude LLC
# Licensed under the OOM Commercial License v1.0
# See LICENSE.md in the repository root or ofmagnitude.com

#!/usr/bin/env python3
"""
SiHF3 Crystallization Pressure Sweep (Static)
Find crystallization pressure by static relaxation at 0.20-0.50 GPa in 0.05 GPa steps
No MD, no Langevin, no NVT - just static relaxation at each pressure
"""

import requests
import csv
import os
from datetime import datetime
from copy import deepcopy
import numpy as np

from ase.io import read, write
from ase.build import make_supercell
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.units import GPa
from ase.geometry.analysis import Analysis

# Try to import MACE, fallback to EMT if not available
try:
    from mace.calculators import mace_mp
    MACE_AVAILABLE = True
except ImportError:
    MACE_AVAILABLE = False
    print("Warning: MACE not available, using EMT as fallback")

# Configuration
MP_API_KEY = os.environ.get("MATERIALS_PROJECT_API_KEY")
MP_MATERIAL_ID = "fa37e9edd4"
LOCAL_CIF_PATH = os.path.join(os.path.expanduser('~'), "NS/Bob/findings/SiHF3/fa37e9edd4.cif")
OUTPUT_DIR = os.path.join(os.path.expanduser('~'), "NS/Bob/findings/shifu_crystallization/")

PRESSURES = [0.01, 0.05, 0.10, 0.15, 0.55, 0.60, 0.65]  # GPa
RDF_RMAX = 4.0
RDF_BINS = 100

def fetch_cif_from_mp(material_id, api_key):
    """Fetch CIF from Materials Project API using requests"""
    url = f'https://api.materialsproject.org/materials/{material_id}/?_all_fields=true'
    headers = {'X-API-KEY': api_key}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        # Extract CIF string
        cif_string = data.get('cif', None)
        if cif_string:
            # Save to temp file
            temp_path = '/tmp/sihf3_temp.cif'
            with open(temp_path, 'w') as f:
                f.write(cif_string)
            return temp_path
        else:
            print("No CIF data found in MP response")
            return None
    except Exception as e:
        print(f"Failed to fetch CIF from MP: {e}")
        return None

def load_sihf3_structure():
    """Load SiHF3 structure from MP API or local CIF"""
    # Try local CIF first
    if os.path.exists(LOCAL_CIF_PATH):
        print(f"Loading SiHF3 from local CIF: {LOCAL_CIF_PATH}")
        return read(LOCAL_CIF_PATH)
    
    # Try MP API
    print(f"Fetching SiHF3 from MP API: {MP_MATERIAL_ID}")
    cif_path = fetch_cif_from_mp(MP_MATERIAL_ID, MP_API_KEY)
    if cif_path:
        atoms = read(cif_path)
        os.remove(cif_path)  # Clean up temp file
        return atoms
    
    raise FileNotFoundError("Could not load SiHF3 structure from MP API or local CIF")

def build_supercell(atoms):
    """Build 2x2x2 supercell"""
    print(f"Building 2x2x2 supercell (original: {len(atoms)} atoms)")
    supercell = make_supercell(atoms, [[2,0,0],[0,2,0],[0,0,2]])
    print(f"Supercell: {len(supercell)} atoms")
    return supercell

def attach_calculator(atoms):
    """Attach MACE or EMT calculator"""
    if MACE_AVAILABLE:
        print("Attaching MACE-MP calculator (small model, float32, CPU)")
        calc = mace_mp(model="small", default_dtype="float32", device="cpu")
        atoms.calc = calc
    else:
        print("Attaching EMT calculator (fallback)")
        atoms.calc = EMT()
    return atoms

def calculate_rdf_sharpness(atoms):
    """Calculate RDF sharpness (max peak / mean baseline)"""
    ana = Analysis(atoms)
    rdf = ana.get_rdf(rmax=RDF_RMAX, nbins=RDF_BINS)[0]
    
    # Calculate sharpness: max peak / mean baseline
    sharpness = rdf.max() / (rdf.mean() + 1e-8)
    return sharpness

def run_pressure_sweep(base_supercell):
    """Run static pressure sweep and collect results"""
    results = []
    
    for pressure in PRESSURES:
        print(f"\n{'='*60}")
        print(f"Pressure: {pressure:.2f} GPa")
        print(f"{'='*60}")
        
        # Fresh copy of supercell
        atoms = deepcopy(base_supercell)
        
        # Attach calculator
        attach_calculator(atoms)
        
        # Apply pressure by scaling cell volume, then relax atoms only
        print("Relaxing structure under pressure...")
        
        # Scale cell to approximate pressure effect
        # Higher pressure = smaller volume. Use linear compression.
        compression_factor = 1.0 - (pressure * 0.04)  # 0.5 GPa → 2% compression
        current_volume = atoms.get_volume()
        target_volume = current_volume * compression_factor
        scale_factor = (target_volume / current_volume) ** (1/3)
        atoms.set_cell(atoms.get_cell() * scale_factor, scale_atoms=True)
        
        # Relax atoms only (fixed cell at compressed volume)
        opt = BFGS(atoms)
        opt.run(fmax=0.3, steps=100)
        
        # Calculate density
        density = atoms.get_masses().sum() / atoms.get_volume() * 1.66054  # g/cm^3
        
        # Calculate RDF sharpness on relaxed structure
        rdf_sharpness = calculate_rdf_sharpness(atoms)
        
        # Phase guess
        if rdf_sharpness > 3.0:
            phase_guess = "crystal"
        elif rdf_sharpness < 1.5:
            phase_guess = "liquid"
        else:
            phase_guess = "transition"
        
        # Record results
        result = {
            'pressure_GPa': pressure,
            'density_g_cm3': density,
            'rdf_sharpness': rdf_sharpness,
            'phase_guess': phase_guess
        }
        results.append(result)
        
        # Print summary
        print(f"Pressure: {pressure:.2f} GPa | Density: {density:.4f} g/cm³ | "
              f"RDF Sharpness: {rdf_sharpness:.2f} | Phase: {phase_guess}")
    
    return results

def save_results(results, base_supercell, append_readme=True):
    """Save results to CSV and README"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save CSV (extended)
    csv_path = os.path.join(OUTPUT_DIR, "sweep_results_extended.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['pressure_GPa', 'density_g_cm3', 
                                                 'rdf_sharpness', 'phase_guess'])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {csv_path}")
    
    # Save README (append mode)
    readme_path = os.path.join(OUTPUT_DIR, "README.txt")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if append_readme and os.path.exists(readme_path):
        mode = 'a'
    else:
        mode = 'w'
    
    with open(readme_path, mode) as f:
        if mode == 'a':
            f.write("\n\n")
        else:
            f.write(f"SiHF3 Crystallization Pressure Sweep (Static)\n")
            f.write(f"Generated: {timestamp}\n\n")
            f.write(f"Method: Static relaxation with cell scaling + BFGS at each pressure (no MD, no Langevin)\n")
            f.write(f"Supercell: 2x2x2 ({len(base_supercell)} atoms)\n")
            f.write(f"Calculator: {'MACE-MP (small, float32, CPU)' if MACE_AVAILABLE else 'EMT (fallback)'}\n")
            f.write(f"Relaxation: Cell scaling (pressure) + BFGS (atoms only), fmax=0.3, max 100 steps\n\n")
        
        f.write(f"--- Extended Sweep ({timestamp}) ---\n")
        f.write(f"Pressure range: {PRESSURES[0]:.2f} - {PRESSURES[-1]:.2f} GPa\n")
        f.write("Results:\n")
        f.write(f"{'Pressure (GPa)':<15} {'Density (g/cm³)':<20} {'RDF Sharpness':<15} {'Phase':<15}\n")
        f.write("-" * 65 + "\n")
        for r in results:
            f.write(f"{r['pressure_GPa']:<15.2f} {r['density_g_cm3']:<20.4f} "
                   f"{r['rdf_sharpness']:<15.2f} {r['phase_guess']:<15}\n")
        
        # Find crystallization pressure
        crystallization_point = None
        for i in range(len(results) - 1):
            if results[i]['phase_guess'] == 'liquid' and results[i+1]['phase_guess'] == 'crystal':
                crystallization_point = (results[i]['pressure_GPa'] + results[i+1]['pressure_GPa']) / 2
                break
        
        if crystallization_point:
            f.write(f"\nCrystallization pressure estimate: ~{crystallization_point:.2f} GPa\n")
        else:
            f.write("\nCrystallization pressure: Not clearly identified in this range\n")
    
    print(f"README updated at {readme_path}")

def main():
    print("="*60)
    print("SiHF3 Crystallization Pressure Sweep")
    print("="*60)
    
    # Load structure
    atoms = load_sihf3_structure()
    print(f"Loaded SiHF3: {len(atoms)} atoms")
    
    # Build supercell
    base_supercell = build_supercell(atoms)
    
    # Run pressure sweep
    results = run_pressure_sweep(base_supercell)
    
    # Save results
    save_results(results, base_supercell)
    
    print("\n" + "="*60)
    print("Sweep complete!")
    print("="*60)

if __name__ == "__main__":
    main()

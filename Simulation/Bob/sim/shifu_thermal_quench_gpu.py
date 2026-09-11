# Copyright (c) 2026 Orders of Magnitude LLC
# Licensed under the OOM Commercial License v1.0
# See LICENSE.md in the repository root or ofmagnitude.com

#!/usr/bin/env python3
"""
SiHF3 Thermal Quench Simulation (GPU-optimized for RunPod A4000)
Quantum-informed thermal simulation using MACE on CUDA
Goal: Observe crystallization during thermal quench with pressure
"""

import torch
import os
import csv
import requests
from datetime import datetime
from copy import deepcopy
import numpy as np

# Set CUDA if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cuda':
    torch.set_num_threads(1)  # Reduce CPU threads when using GPU
else:
    torch.set_num_threads(8)
    os.environ['OMP_NUM_THREADS'] = '8'

from ase.io import read, write
from ase.build import make_supercell
from ase.optimize import BFGS
from ase.units import GPa
from ase.geometry.analysis import Analysis
from ase.md.langevin import Langevin

# Import MACE
from mace.calculators import mace_mp

# Configuration
MP_API_KEY = os.environ.get("MATERIALS_PROJECT_API_KEY")
MP_MATERIAL_ID = "fa37e9edd4"
WORK_DIR = "/root/shifu"
CIF_PATH = os.path.join(WORK_DIR, "fa37e9edd4.cif")
OUTPUT_DIR = os.path.join(WORK_DIR, "findings/thermal_quench")

RDF_RMAX = 4.0
RDF_BINS = 100
FRICTION = 0.02
TIMESTEP_FS = 1.0  # 1 femtosecond

def fetch_cif_from_mp():
    """Fetch CIF from Materials Project API"""
    os.makedirs(WORK_DIR, exist_ok=True)
    
    url = f'https://api.materialsproject.org/materials/{MP_MATERIAL_ID}/?_all_fields=true'
    headers = {'X-API-KEY': MP_API_KEY}
    
    print(f"Fetching SiHF3 CIF from MP API: {MP_MATERIAL_ID}")
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        cif_string = data.get('cif', None)
        if cif_string:
            with open(CIF_PATH, 'w') as f:
                f.write(cif_string)
            print(f"CIF saved to {CIF_PATH}")
            return CIF_PATH
        else:
            raise ValueError("No CIF data in MP response")
    except Exception as e:
        raise RuntimeError(f"Failed to fetch CIF: {e}")

def load_sihf3_structure():
    """Load SiHF3 structure from CIF"""
    if os.path.exists(CIF_PATH):
        print(f"Loading SiHF3 from CIF: {CIF_PATH}")
        return read(CIF_PATH)
    else:
        print("CIF not found locally, fetching from MP...")
        fetch_cif_from_mp()
        return read(CIF_PATH)

def build_supercell(atoms):
    """Build 2x2x2 supercell"""
    print(f"Building 2x2x2 supercell (original: {len(atoms)} atoms)")
    supercell = make_supercell(atoms, [[2,0,0],[0,2,0],[0,0,2]])
    print(f"Supercell: {len(supercell)} atoms")
    return supercell

def attach_calculator(atoms):
    """Attach MACE calculator on GPU"""
    print(f"Attaching MACE-MP calculator (small model, float32, {device})")
    calc = mace_mp(model="small", default_dtype="float32", device=device)
    atoms.calc = calc
    return atoms

def calculate_rdf_sharpness(atoms):
    """Calculate RDF sharpness (max peak / mean baseline)"""
    ana = Analysis(atoms)
    rdf = ana.get_rdf(rmax=RDF_RMAX, nbins=RDF_BINS)[0]
    sharpness = rdf.max() / (rdf.mean() + 1e-8)
    return sharpness

def guess_phase(rdf_sharpness):
    """Guess phase based on RDF sharpness"""
    if rdf_sharpness > 3.0:
        return "crystal"
    elif rdf_sharpness < 1.5:
        return "liquid"
    else:
        return "transition"

def apply_pressure(atoms, pressure_gpa):
    """Apply pressure by scaling cell volume"""
    compression_factor = 1.0 - (pressure_gpa * 0.04)
    current_volume = atoms.get_volume()
    target_volume = current_volume * compression_factor
    scale_factor = (target_volume / current_volume) ** (1/3)
    atoms.set_cell(atoms.get_cell() * scale_factor, scale_atoms=True)

def record_checkpoint(atoms, temp_k, pressure_gpa, step_name, results):
    """Record checkpoint data"""
    density = atoms.get_masses().sum() / atoms.get_volume() * 1.66054
    rdf_sharpness = calculate_rdf_sharpness(atoms)
    phase = guess_phase(rdf_sharpness)
    
    print(f"{step_name:<40} | T: {temp_k:4.0f}K | P: {pressure_gpa:5.2f} GPa | "
          f"Density: {density:.4f} g/cm³ | RDF: {rdf_sharpness:6.2f} | Phase: {phase}")
    
    results.append({
        'step': step_name,
        'temperature_K': temp_k,
        'pressure_GPa': pressure_gpa,
        'density_g_cm3': density,
        'rdf_sharpness': rdf_sharpness,
        'phase_guess': phase
    })

def run_md_steps(atoms, temp_k, n_steps, batch_size=100, step_name="MD"):
    """Run Langevin MD with checkpointing"""
    # Print energy before MD to check for NaN
    try:
        energy = atoms.get_potential_energy()
        print(f"  Energy before {step_name}: {energy:.4f} eV")
    except Exception as e:
        print(f"  ERROR getting energy before {step_name}: {e}")
        return
    
    dyn = Langevin(atoms, temperature_K=temp_k, timestep=TIMESTEP_FS*1000, 
                   friction=FRICTION, logfile=None)
    
    # Batch the MD runs with checkpointing
    for i in range(0, n_steps, batch_size):
        steps_to_run = min(batch_size, n_steps - i)
        dyn.run(steps_to_run)
        print(f"  {step_name}: {i+steps_to_run}/{n_steps} steps complete")
        import sys
        sys.stdout.flush()

def main():
    print("="*80)
    print("SiHF3 Thermal Quench Simulation (GPU-optimized with MACE on CUDA)")
    print("="*80)
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load and build supercell
    atoms = load_sihf3_structure()
    print(f"Loaded SiHF3: {len(atoms)} atoms")
    base_supercell = build_supercell(atoms)
    
    # Attach calculator
    attach_calculator(base_supercell)
    
    results = []
    
    # Stage 1: Initial relaxation (no pressure)
    print("\n" + "="*80)
    print("Stage 1: Initial Relaxation (no pressure)")
    print("="*80)
    atoms = deepcopy(base_supercell)
    attach_calculator(atoms)
    opt = BFGS(atoms)
    opt.run(fmax=0.5, steps=30)
    record_checkpoint(atoms, 300, 0.00, "Initial relax", results)
    
    # Stage 2: Heat to 1000K (SMOKE TEST: 20 steps)
    print("\n" + "="*80)
    print("Stage 2: Heat to 1000K (SMOKE TEST: 20 steps)")
    print("="*80)
    run_md_steps(atoms, 1000, 20, batch_size=10, step_name="Heating")
    record_checkpoint(atoms, 1000, 0.00, "Heat to 1000K", results)
    
    # Stage 3: Cool in stages (SMOKE TEST: 10 steps each)
    print("\n" + "="*80)
    print("Stage 3: Cool in stages (SMOKE TEST: 10 steps each)")
    print("="*80)
    cooling_temps = [800, 600, 400, 300]
    for temp in cooling_temps:
        run_md_steps(atoms, temp, 10, batch_size=10, step_name=f"Cooling to {temp}K")
        record_checkpoint(atoms, temp, 0.00, f"Cool to {temp}K", results)
    
    # Stage 4: Apply pressure at 300K (SMOKE TEST: 10 steps)
    print("\n" + "="*80)
    print("Stage 4: Apply pressure at 300K (SMOKE TEST: 10 steps)")
    print("="*80)
    apply_pressure(atoms, 0.30)
    opt = BFGS(atoms)
    opt.run(fmax=0.5, steps=30)
    record_checkpoint(atoms, 300, 0.30, "Apply 0.30 GPa (relax)", results)
    run_md_steps(atoms, 300, 10, batch_size=10, step_name="Pressure MD")
    record_checkpoint(atoms, 300, 0.30, "Apply 0.30 GPa (MD)", results)
    
    # Stage 5: Release pressure slowly (SMOKE TEST: 10 steps each)
    print("\n" + "="*80)
    print("Stage 5: Release pressure slowly (SMOKE TEST: 10 steps each)")
    print("="*80)
    release_pressures = [0.20, 0.10, 0.01]
    for pressure in release_pressures:
        apply_pressure(atoms, pressure)
        opt = BFGS(atoms)
        opt.run(fmax=0.5, steps=30)
        record_checkpoint(atoms, 300, pressure, f"Release to {pressure:.2f} GPa (relax)", results)
        run_md_steps(atoms, 300, 10, batch_size=10, step_name=f"Release MD {pressure:.2f} GPa")
        record_checkpoint(atoms, 300, pressure, f"Release to {pressure:.2f} GPa (MD)", results)
    
    # Save results
    csv_path = os.path.join(OUTPUT_DIR, "thermal_quench_results.csv")
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['step', 'temperature_K', 'pressure_GPa', 'density_g_cm3', 'rdf_sharpness', 'phase_guess']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"\nResults saved to {csv_path}")
    
    # Save README
    readme_path = os.path.join(OUTPUT_DIR, "README.txt")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(readme_path, 'w') as f:
        f.write(f"SiHF3 Thermal Quench Simulation (GPU-optimized)\n")
        f.write(f"Generated: {timestamp}\n\n")
        f.write(f"Method: Thermal quench with MACE-MP on {device.upper()}\n")
        f.write(f"Supercell: 2x2x2 ({len(base_supercell)} atoms)\n")
        f.write(f"Calculator: MACE-MP (small, float32, {device})\n")
        if device == 'cuda':
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
        f.write(f"Timestep: {TIMESTEP_FS} fs, Friction: {FRICTION}\n\n")
        
        f.write("Stages:\n")
        f.write("1. Initial relax: BFGS fmax=0.5, 30 steps (no pressure, 300K)\n")
        f.write("2. Heat to 1000K: Langevin NVT, 20 steps (SMOKE TEST)\n")
        f.write("3. Cool in stages: 800K → 600K → 400K → 300K (10 steps each, SMOKE TEST)\n")
        f.write("4. Apply pressure at 300K: 0.30 GPa, BFGS relax + 10 NVT steps (SMOKE TEST)\n")
        f.write("5. Release pressure: 0.20 → 0.10 → 0.01 GPa (relax + 10 NVT steps each, SMOKE TEST)\n\n")
        
        f.write("Results:\n")
        f.write(f"{'Step':<40} {'T (K)':<8} {'P (GPa)':<10} {'Density (g/cm³)':<18} {'RDF Sharpness':<15} {'Phase':<15}\n")
        f.write("-" * 115 + "\n")
        for r in results:
            f.write(f"{r['step']:<40} {r['temperature_K']:<8.0f} {r['pressure_GPa']:<10.2f} "
                   f"{r['density_g_cm3']:<18.4f} {r['rdf_sharpness']:<15.2f} {r['phase_guess']:<15}\n")
        
        f.write("\nPhase Classification:\n")
        f.write("- Crystal: RDF sharpness > 3.0 (sharp peaks = ordered)\n")
        f.write("- Liquid: RDF sharpness < 1.5 (broad peaks = disordered)\n")
        f.write("- Transition: 1.5 ≤ RDF sharpness ≤ 3.0\n")
        
        # Analyze crystallization
        high_temp_phase = results[1]['phase_guess']  # After heating to 1000K
        low_temp_phase = results[-1]['phase_guess']  # Final state
        if high_temp_phase != 'crystal' and low_temp_phase == 'crystal':
            f.write(f"\nConclusion: SiHF3 crystallizes during thermal quench ({high_temp_phase} → {low_temp_phase})\n")
        elif low_temp_phase == 'crystal':
            f.write(f"\nConclusion: SiHF3 remains crystalline throughout (stable crystal)\n")
        else:
            f.write(f"\nConclusion: SiHF3 does not crystallize in this simulation ({high_temp_phase} → {low_temp_phase})\n")
    
    print(f"README saved to {readme_path}")
    
    print("\n" + "="*80)
    print("Thermal quench simulation complete!")
    print("="*80)
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Ready to SCP back to Spectre")

if __name__ == "__main__":
    main()

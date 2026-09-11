# Copyright (c) 2026 Orders of Magnitude LLC
# Licensed under the OOM Commercial License v1.0
# See LICENSE.md in the repository root or ofmagnitude.com

#!/usr/bin/env python3
"""
SiHF3 Thermal Quench Simulation
Quantum-informed thermal simulation using MACE (trained on DFT data)
Goal: Observe crystallization during thermal quench with pressure
"""

import torch
import os
import csv
from datetime import datetime
from copy import deepcopy
import numpy as np

# Set threading for performance
torch.set_num_threads(8)
os.environ['OMP_NUM_THREADS'] = '8'

from ase.io import read
from ase.build import make_supercell
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.units import GPa
from ase.geometry.analysis import Analysis
from ase.md.langevin import Langevin

# Try to import MACE, fallback to EMT if not available
try:
    from mace.calculators import mace_mp
    MACE_AVAILABLE = True
except ImportError:
    MACE_AVAILABLE = False
    print("Warning: MACE not available, using EMT as fallback")

# Configuration
LOCAL_CIF_PATH = os.path.join(os.path.expanduser('~'), "NS/Bob/findings/SiHF3/fa37e9edd4.cif")
OUTPUT_DIR = os.path.join(os.path.expanduser('~'), "NS/Bob/findings/shifu_thermal_quench/")

RDF_RMAX = 4.0
RDF_BINS = 100
FRICTION = 0.02
TIMESTEP_FS = 1.0  # 1 femtosecond

def load_sihf3_structure():
    """Load SiHF3 structure from local CIF"""
    if os.path.exists(LOCAL_CIF_PATH):
        print(f"Loading SiHF3 from local CIF: {LOCAL_CIF_PATH}")
        return read(LOCAL_CIF_PATH)
    raise FileNotFoundError(f"Could not find SiHF3 CIF at {LOCAL_CIF_PATH}")

def build_supercell(atoms):
    """Build 2x2x2 supercell"""
    print(f"Building 2x2x2 supercell (original: {len(atoms)} atoms)")
    supercell = make_supercell(atoms, [[2,0,0],[0,2,0],[0,0,2]])
    print(f"Supercell: {len(supercell)} atoms")
    return supercell

def attach_calculator(atoms):
    """Attach MACE calculator (EMT doesn't support F)"""
    if MACE_AVAILABLE:
        print("Attaching MACE-MP calculator (small model, float32, CPU)")
        calc = mace_mp(model="small", default_dtype="float32", device="cpu")
        atoms.calc = calc
    else:
        raise NotImplementedError("EMT doesn't support fluorine, MACE required")
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
    
    print(f"{step_name:<30} | T: {temp_k:4.0f}K | P: {pressure_gpa:5.2f} GPa | "
          f"Density: {density:.4f} g/cm³ | RDF: {rdf_sharpness:6.2f} | Phase: {phase}")
    
    results.append({
        'step': step_name,
        'temperature_K': temp_k,
        'pressure_GPa': pressure_gpa,
        'density_g_cm3': density,
        'rdf_sharpness': rdf_sharpness,
        'phase_guess': phase
    })

def run_md_steps(atoms, temp_k, n_steps, batch_size=100):
    """Run Langevin MD with checkpointing"""
    dyn = Langevin(atoms, temperature_K=temp_k, timestep=TIMESTEP_FS*1000, 
                   friction=FRICTION, logfile=None)
    
    # Batch the MD runs
    for i in range(0, n_steps, batch_size):
        steps_to_run = min(batch_size, n_steps - i)
        dyn.run(steps_to_run)

def main():
    print("="*80)
    print("SiHF3 Thermal Quench Simulation (Quantum-informed MD with MACE)")
    print("="*80)
    print(f"Threads: {torch.get_num_threads()}, OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'N/A')}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load and build supercell
    atoms = load_sihf3_structure()
    print(f"Loaded SiHF3: {len(atoms)} atoms")
    base_supercell = build_supercell(atoms)
    
    # Attach calculator (MACE required for fluorine)
    attach_calculator(base_supercell)
    
    results = []
    
    # Phase 1: Initial relaxation (no pressure)
    print("\n" + "="*80)
    print("Phase 1: Initial Relaxation (no pressure)")
    print("="*80)
    atoms = deepcopy(base_supercell)
    attach_calculator(atoms)
    opt = BFGS(atoms)
    opt.run(fmax=0.5, steps=30)
    record_checkpoint(atoms, 300, 0.00, "Initial relax", results)
    
    # Phase 2: Heat to 1000K
    print("\n" + "="*80)
    print("Phase 2: Heat to 1000K (50 steps, batched)")
    print("="*80)
    run_md_steps(atoms, 1000, 50, batch_size=50)
    record_checkpoint(atoms, 1000, 0.00, "Heat to 1000K", results)
    
    # Phase 3: Cool in stages
    print("\n" + "="*80)
    print("Phase 3: Cool in stages (800K → 600K → 400K → 300K → 200K)")
    print("="*80)
    cooling_temps = [800, 600, 400, 300, 200]
    for temp in cooling_temps:
        run_md_steps(atoms, temp, 50, batch_size=50)
        record_checkpoint(atoms, temp, 0.00, f"Cool to {temp}K", results)
    
    # Phase 4: Apply pressure at 300K
    print("\n" + "="*80)
    print("Phase 4: Apply pressure at 300K (0.30 GPa)")
    print("="*80)
    apply_pressure(atoms, 0.30)
    opt = BFGS(atoms)
    opt.run(fmax=0.5, steps=30)
    record_checkpoint(atoms, 300, 0.30, "Apply 0.30 GPa (relax)", results)
    run_md_steps(atoms, 300, 50, batch_size=50)
    record_checkpoint(atoms, 300, 0.30, "Apply 0.30 GPa (MD)", results)
    
    # Phase 5: Release pressure slowly
    print("\n" + "="*80)
    print("Phase 5: Release pressure slowly (0.20 → 0.10 → 0.01 GPa)")
    print("="*80)
    release_pressures = [0.20, 0.10, 0.01]
    for pressure in release_pressures:
        apply_pressure(atoms, pressure)
        opt = BFGS(atoms)
        opt.run(fmax=0.5, steps=30)
        record_checkpoint(atoms, 300, pressure, f"Release to {pressure:.2f} GPa (relax)", results)
        run_md_steps(atoms, 300, 50, batch_size=50)
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
        f.write(f"SiHF3 Thermal Quench Simulation (Quantum-informed MD)\n")
        f.write(f"Generated: {timestamp}\n\n")
        f.write(f"Method: Thermal quench with MACE-MP (trained on DFT data)\n")
        f.write(f"Supercell: 2x2x2 ({len(base_supercell)} atoms)\n")
        f.write(f"Calculator: {'MACE-MP (small, float32, CPU)' if MACE_AVAILABLE else 'EMT (fallback)'}\n")
        f.write(f"Threads: {torch.get_num_threads()} (OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'N/A')})\n")
        f.write(f"Timestep: {TIMESTEP_FS} fs, Friction: {FRICTION}\n\n")
        
        f.write("Phases:\n")
        f.write("1. Initial relax: BFGS fmax=0.5, 30 steps (no pressure, 300K)\n")
        f.write("2. Heat to 1000K: Langevin NVT, 50 steps (batched 50)\n")
        f.write("3. Cool in stages: 800K → 600K → 400K → 300K → 200K (50 steps each)\n")
        f.write("4. Apply pressure at 300K: 0.30 GPa, BFGS relax + 50 NVT steps\n")
        f.write("5. Release pressure: 0.20 → 0.10 → 0.01 GPa (relax + 50 NVT steps each)\n\n")
        
        f.write("Results:\n")
        f.write(f"{'Step':<35} {'T (K)':<8} {'P (GPa)':<10} {'Density (g/cm³)':<18} {'RDF Sharpness':<15} {'Phase':<15}\n")
        f.write("-" * 110 + "\n")
        for r in results:
            f.write(f"{r['step']:<35} {r['temperature_K']:<8.0f} {r['pressure_GPa']:<10.2f} "
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

if __name__ == "__main__":
    main()

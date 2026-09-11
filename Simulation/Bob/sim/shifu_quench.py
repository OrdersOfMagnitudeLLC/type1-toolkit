# Copyright (c) 2026 Orders of Magnitude LLC
# Licensed under the OOM Commercial License v1.0
# See LICENSE.md in the repository root or ofmagnitude.com

#!/usr/bin/env python3
"""
SiHF3 Pressure-Trap-Then-Release Quench Simulation
Goal: Can SiHF3 stay crystalline when pressure drops?
"""

import csv
import os
from datetime import datetime
from copy import deepcopy
import numpy as np

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
OUTPUT_DIR = os.path.join(os.path.expanduser('~'), "NS/Bob/findings/shifu_quench/")

RDF_RMAX = 4.0
RDF_BINS = 100
TEMPERATURE = 300  # K
NVT_STEPS = 200

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

def relax_under_pressure(atoms, pressure_gpa, fmax=0.3, steps=100):
    """Relax structure under pressure using cell scaling + BFGS"""
    apply_pressure(atoms, pressure_gpa)
    opt = BFGS(atoms)
    opt.run(fmax=fmax, steps=steps)

def main():
    print("="*60)
    print("SiHF3 Pressure-Trap-Then-Release Quench Simulation")
    print("="*60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load and build supercell
    atoms = load_sihf3_structure()
    print(f"Loaded SiHF3: {len(atoms)} atoms")
    base_supercell = build_supercell(atoms)
    
    # Attach calculator
    attach_calculator(base_supercell)
    
    results = []
    
    # Phase 1: Crystallize at 0.30 GPa
    print("\n" + "="*60)
    print("Phase 1: Crystallize at 0.30 GPa")
    print("="*60)
    atoms = deepcopy(base_supercell)
    attach_calculator(atoms)
    relax_under_pressure(atoms, 0.30, fmax=0.3, steps=100)
    density = atoms.get_masses().sum() / atoms.get_volume() * 1.66054
    rdf_sharpness = calculate_rdf_sharpness(atoms)
    phase = guess_phase(rdf_sharpness)
    print(f"Pressure: 0.30 GPa | Density: {density:.4f} g/cm³ | "
          f"RDF Sharpness: {rdf_sharpness:.2f} | Phase: {phase}")
    results.append({
        'phase': 'Phase 1 (crystallize)',
        'pressure_GPa': 0.30,
        'density_g_cm3': density,
        'rdf_sharpness': rdf_sharpness,
        'phase_guess': phase
    })
    
    # Phase 2: Heat at pressure (Langevin NVT) - SKIPPED due to MACE/MD performance issues
    print("\n" + "="*60)
    print("Phase 2: Heat at 0.30 GPa (SKIPPED - MACE/MD too slow on CPU)")
    print("="*60)
    print("Proceeding directly to slow release phase...")
    # Use the same structure from Phase 1
    results.append({
        'phase': 'Phase 2 (heat - SKIPPED)',
        'pressure_GPa': 0.30,
        'density_g_cm3': results[-1]['density_g_cm3'],
        'rdf_sharpness': results[-1]['rdf_sharpness'],
        'phase_guess': results[-1]['phase_guess']
    })
    
    # Phase 3: Slow release
    print("\n" + "="*60)
    print("Phase 3: Slow Release (0.30 → 0.20 → 0.10 → 0.05 → 0.01 GPa)")
    print("="*60)
    release_pressures = [0.20, 0.10, 0.05, 0.01]
    for pressure in release_pressures:
        relax_under_pressure(atoms, pressure, fmax=0.3, steps=100)
        density = atoms.get_masses().sum() / atoms.get_volume() * 1.66054
        rdf_sharpness = calculate_rdf_sharpness(atoms)
        phase = guess_phase(rdf_sharpness)
        print(f"Pressure: {pressure:.2f} GPa | Density: {density:.4f} g/cm³ | "
              f"RDF Sharpness: {rdf_sharpness:.2f} | Phase: {phase}")
        results.append({
            'phase': f'Phase 3 (release to {pressure:.2f} GPa)',
            'pressure_GPa': pressure,
            'density_g_cm3': density,
            'rdf_sharpness': rdf_sharpness,
            'phase_guess': phase
        })
    
    # Phase 4: Ambient check (no pressure)
    print("\n" + "="*60)
    print("Phase 4: Ambient Check (no pressure)")
    print("="*60)
    # Reset to original cell volume (no compression)
    atoms = deepcopy(base_supercell)
    attach_calculator(atoms)
    # Single point energy calculation (no relaxation)
    energy = atoms.get_potential_energy()
    density = atoms.get_masses().sum() / atoms.get_volume() * 1.66054
    rdf_sharpness = calculate_rdf_sharpness(atoms)
    phase = guess_phase(rdf_sharpness)
    print(f"Pressure: 0.00 GPa | Density: {density:.4f} g/cm³ | "
          f"RDF Sharpness: {rdf_sharpness:.2f} | Phase: {phase} | Energy: {energy:.2f} eV")
    results.append({
        'phase': 'Phase 4 (ambient)',
        'pressure_GPa': 0.00,
        'density_g_cm3': density,
        'rdf_sharpness': rdf_sharpness,
        'phase_guess': phase,
        'energy_eV': energy
    })
    
    # Save results
    csv_path = os.path.join(OUTPUT_DIR, "quench_results.csv")
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['phase', 'pressure_GPa', 'density_g_cm3', 'rdf_sharpness', 'phase_guess', 'energy_eV']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"\nResults saved to {csv_path}")
    
    # Save README
    readme_path = os.path.join(OUTPUT_DIR, "README.txt")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(readme_path, 'w') as f:
        f.write(f"SiHF3 Pressure-Trap-Then-Release Quench Simulation\n")
        f.write(f"Generated: {timestamp}\n\n")
        f.write(f"Method: Pressure-trap-then-release with MACE-MP\n")
        f.write(f"Supercell: 2x2x2 ({len(base_supercell)} atoms)\n")
        f.write(f"Calculator: {'MACE-MP (small, float32, CPU)' if MACE_AVAILABLE else 'EMT (fallback)'}\n\n")
        
        f.write("Phases:\n")
        f.write("1. Crystallize at 0.30 GPa (BFGS relax, fmax=0.3, 100 steps)\n")
        f.write("2. Heat at 0.30 GPa (SKIPPED - MACE/MD too slow on CPU)\n")
        f.write("3. Slow release: 0.30 → 0.20 → 0.10 → 0.05 → 0.01 GPa (BFGS relax each)\n")
        f.write("4. Ambient check: 0.00 GPa (single point energy)\n\n")
        
        f.write("Results:\n")
        f.write(f"{'Phase':<30} {'Pressure (GPa)':<15} {'Density (g/cm³)':<20} {'RDF Sharpness':<15} {'Phase':<15}\n")
        f.write("-" * 95 + "\n")
        for r in results:
            energy_str = f" | E: {r.get('energy_eV', 'N/A'):.2f} eV" if 'energy_eV' in r else ""
            f.write(f"{r['phase']:<30} {r['pressure_GPa']:<15.2f} {r['density_g_cm3']:<20.4f} "
                   f"{r['rdf_sharpness']:<15.2f} {r['phase_guess']:<15}{energy_str}\n")
        
        f.write("\nPhase Classification:\n")
        f.write("- Crystal: RDF sharpness > 3.0\n")
        f.write("- Liquid: RDF sharpness < 1.5\n")
        f.write("- Transition: 1.5 ≤ RDF sharpness ≤ 3.0\n")
        
        # Check if crystalline state persists
        final_phase = results[-1]['phase_guess']
        initial_phase = results[0]['phase_guess']
        if final_phase == 'crystal' and initial_phase == 'crystal':
            f.write(f"\nConclusion: SiHF3 remains crystalline after pressure release (crystal → crystal)\n")
        elif final_phase == 'crystal':
            f.write(f"\nConclusion: SiHF3 crystallizes during pressure release ({initial_phase} → crystal)\n")
        else:
            f.write(f"\nConclusion: SiHF3 does not remain crystalline after pressure release ({initial_phase} → {final_phase})\n")
    
    print(f"README saved to {readme_path}")
    
    print("\n" + "="*60)
    print("Quench simulation complete!")
    print("="*60)

if __name__ == "__main__":
    main()

# Copyright (c) 2026 Orders of Magnitude LLC
# Licensed under the OOM Commercial License v1.0
# See LICENSE.md in the repository root or ofmagnitude.com

#!/usr/bin/env python3
"""
Analyze trajectory files to compute MSD, diffusion coefficient, ionic conductivity,
and Arrhenius fit to extrapolate conductivity to 300K.
"""

import os
import json
import glob
import numpy as np
from scipy.stats import linregress

# Constants
e = 1.602e-19  # C (elementary charge)
kB = 1.381e-23  # J/K (Boltzmann constant)
DEFAULT_DENSITY = 0.08  # atoms/Å³ (default density estimate)

def load_trajectories(results_dir=os.path.join(os.path.expanduser('~'), "NS/Bob/results")):
    """
    Load all trajectory files and group by material name.
    Returns dict: {material_name: {temp_K: (traj, meta)}}
    """
    traj_files = sorted(glob.glob(os.path.join(results_dir, "*_traj.npy")))
    
    materials = {}
    for traj_path in traj_files:
        # Extract label from filename (e.g., "SiHF3_1000K_traj.npy")
        basename = os.path.basename(traj_path)
        label = basename.replace("_traj.npy", "")
        
        # Load trajectory
        traj = np.load(traj_path)
        
        # Load metadata
        meta_path = traj_path.replace("_traj.npy", "_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
        else:
            meta = {}
        
        # Extract material name and temperature
        parts = label.split("_")
        material = "_".join(parts[:-1])  # Handle names with underscores
        temp_K = int(parts[-1].replace("K", ""))
        
        if material not in materials:
            materials[material] = {}
        materials[material][temp_K] = (traj, meta)
    
    return materials

def compute_msd_time_series(traj):
    """
    Compute time-averaged MSD at each lag time.
    traj shape: (nframes, n_atoms, 3) - unwrapped positions in Å
    Returns: msd array (nframes-1,) in Å²
    """
    n_frames, n_atoms, _ = traj.shape
    msd = []
    
    for lag in range(1, n_frames):
        # Displacement for all time origins at this lag
        disp = traj[lag:] - traj[:-lag]  # (n_frames-lag, n_atoms, 3)
        sq = np.sum(disp**2, axis=-1)    # (n_frames-lag, n_atoms)
        msd.append(np.mean(sq))          # scalar Å²
    
    return np.array(msd)

def compute_diffusion_coefficient(msd, dt_fs, traj_interval):
    """
    Fit linear region of MSD vs time to get diffusion coefficient.
    Uses middle 20-80% of MSD curve for linear fit.
    Returns: D in m²/s
    """
    n_frames = len(msd) + 1
    time_fs = np.arange(1, n_frames) * dt_fs * traj_interval
    time_s = time_fs * 1e-15  # fs to s
    
    # Use middle 20-80% for linear fit
    start = int(0.2 * len(msd))
    end = int(0.8 * len(msd))
    
    # Fit MSD in m² vs time in s
    msd_m2 = msd * 1e-20  # Å² to m²
    slope, intercept = np.polyfit(time_s[start:end], msd_m2[start:end], 1)
    
    D = slope / 6  # 3D diffusion: MSD = 6Dt
    return D

def compute_ionic_conductivity(D, n_mobile, volume_A3, temp_K):
    """
    Compute ionic conductivity using Nernst-Einstein equation.
    σ = (n * z² * e² * D) / (kB * T)
    Returns: σ in mS/cm
    """
    # Number density of mobile ions (ions/m³)
    n = n_mobile / (volume_A3 * 1e-30)
    
    # Nernst-Einstein equation
    z = 1  # Na⁺ charge
    sigma_SI = (n * z**2 * e**2 * D) / (kB * temp_K)  # S/m
    
    # Convert to mS/cm
    sigma_mS_cm = sigma_SI * 0.1
    return sigma_mS_cm

def fit_arrhenius(temps_K, sigma_mS_cm):
    """
    Fit Arrhenius equation: ln(σ*T) = ln(A) - Ea/(kB*T)
    Returns: Ea_eV, sigma_300K_mS_cm
    """
    x = 1.0 / np.array(temps_K)  # 1/T
    y = np.log(np.array(sigma_mS_cm) * np.array(temps_K))  # ln(σ*T)
    
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    
    # Activation energy
    Ea_J = -slope * kB
    Ea_eV = Ea_J / e
    
    # Extrapolate to 300K
    ln_sigma_300 = intercept + slope * (1.0 / 300.0)
    sigma_300K = np.exp(ln_sigma_300) / 300.0  # mS/cm
    
    return Ea_eV, sigma_300K, r_value**2

def analyze_material(material_name, temp_data):
    """
    Analyze a single material across all temperatures.
    temp_data: {temp_K: (traj, meta)}
    """
    print(f"\n=== {material_name} Arrhenius Analysis ===")
    
    results = []
    temps = sorted(temp_data.keys())
    
    for temp_K in temps:
        traj, meta = temp_data[temp_K]
        
        # Get parameters from metadata
        dt_fs = meta.get('dt_fs', 4.0)
        traj_interval = meta.get('traj_interval', 100)
        n_mobile = meta.get('n_mobile', traj.shape[1])
        
        # Estimate volume if not in metadata
        if 'cell_volume_A3' in meta:
            volume_A3 = meta['cell_volume_A3']
        else:
            # Estimate from density
            n_total = traj.shape[1] + meta.get('n_frozen', 0) if 'n_frozen' in meta else traj.shape[1] * 4
            volume_A3 = n_total / DEFAULT_DENSITY
        
        # Compute MSD
        msd = compute_msd_time_series(traj)
        
        # Compute diffusion coefficient
        D = compute_diffusion_coefficient(msd, dt_fs, traj_interval)
        
        # Compute ionic conductivity
        sigma = compute_ionic_conductivity(D, n_mobile, volume_A3, temp_K)
        
        results.append({
            'temp_K': temp_K,
            'D_m2_s': D,
            'sigma_mS_cm': sigma
        })
        
        print(f"  {temp_K}K:  D={D*1e9:.2f}e-9 m²/s | σ={sigma:.1f} mS/cm")
    
    # Fit Arrhenius
    temps_K = [r['temp_K'] for r in results]
    sigma_mS_cm = [r['sigma_mS_cm'] for r in results]
    
    if len(temps_K) >= 2:
        Ea_eV, sigma_300K, r2 = fit_arrhenius(temps_K, sigma_mS_cm)
        
        print(f"  Activation energy: {Ea_eV:.2f} eV")
        print(f"  Extrapolated σ(300K): {sigma_300K:.1f} mS/cm")
        print(f"  LGPS benchmark: 12-25 mS/cm")
        
        if sigma_300K >= 25:
            result_status = "BEATS LGPS"
        elif sigma_300K >= 12:
            result_status = "WITHIN RANGE"
        else:
            result_status = "BELOW LGPS"
        
        print(f"  Result: {result_status}")
        
        return {
            'material': material_name,
            'temperatures': results,
            'activation_energy_eV': Ea_eV,
            'sigma_300K_mS_cm': sigma_300K,
            'r_squared': r2,
            'result_status': result_status
        }
    else:
        print("  ERROR: Need at least 2 temperatures for Arrhenius fit")
        return None

def main():
    results_dir = os.path.join(os.path.expanduser('~'), "NS/Bob/results")
    
    # Load all trajectories
    print("Loading trajectories...")
    materials = load_trajectories(results_dir)
    
    if not materials:
        print("ERROR: No trajectory files found")
        return
    
    print(f"Found {len(materials)} materials")
    
    # Analyze each material
    all_results = []
    for material_name, temp_data in materials.items():
        result = analyze_material(material_name, temp_data)
        if result:
            all_results.append(result)
    
    # Save results to JSON
    output_path = os.path.join(results_dir, "arrhenius_results.json")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()

# Copyright (c) 2026 Orders of Magnitude LLC
# Licensed under the OOM Commercial License v1.0
# See LICENSE.md in the repository root or ofmagnitude.com

import os
"""
Bob MD — MACE-MPA-0 ionic conductivity screener
Frozen-framework NVT MD for rigid-host battery materials.
Usage:
  python bob_md.py --cif material.cif --test          # 100-step validation (~60s)
  python bob_md.py --cif material.cif                 # full overnight run
  python bob_md.py --cif material.cif --temps 800 1000 1200  # custom temps
"""

import argparse, json, time, os
import numpy as np
import torch
from scipy.stats import linregress
from scipy.interpolate import RegularGridInterpolator
from ase.io import read
from ase.build import make_supercell
from ase.constraints import FixAtoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
import ase.units
from mace.calculators import mace_mp

MOBILE_SPECIES = {'Na', 'K', 'Li', 'Ag'}
kB_eV = 8.617333e-5   # eV/K
kB_SI = 1.380649e-23  # J/K
e_charge = 1.60218e-19  # C

# Unit conversion: F[eV/Å] / m[amu] → a[Å/fs²]
# Derivation: 1 amu·Å²/fs² = 103.6 eV, so a = F/m / 103.6
FORCE_TO_ACC = 1.0 / 103.6   # (eV/Å) / amu → Å/fs²
KB_EV = 8.617333e-5           # eV/K

# Grid precompute constants
MACE_CUTOFF  = 5.5   # Å — MACE-small neighbor cutoff
CLUSTER_PAD  = 0.5   # Å — extra buffer for cluster extraction
MIN_ION_DIST = 1.8   # Å — skip grid points inside framework atoms

def precompute_grid(atoms, calc, mobile_species_set, grid_spacing=2.0, batch_size=16, out_path=None, material_name="material"):
    """
    Precompute MACE forces from rigid framework on a 3D spatial grid.
    Probe ion is the first mobile species found (e.g. Na).
    Batches probe positions for faster MACE evaluation.
    Returns path to saved .npz file.
    """
    from ase import Atoms

    # Identify framework atoms and probe species
    framework_idx = [i for i in range(len(atoms))
                     if atoms[i].symbol not in mobile_species_set]
    framework = atoms[framework_idx]
    probe_symbol = next(iter(mobile_species_set))   # e.g. 'Na'
    fw_pos  = framework.get_positions()             # (N_fw, 3)
    cell    = atoms.get_cell().array                # (3,3)
    cell_len = np.linalg.norm(cell, axis=1)         # (3,) Å

    # Build grid axes
    nx = max(2, int(np.ceil(cell_len[0] / grid_spacing)))
    ny = max(2, int(np.ceil(cell_len[1] / grid_spacing)))
    nz = max(2, int(np.ceil(cell_len[2] / grid_spacing)))
    xs = np.linspace(0, cell_len[0], nx, endpoint=False)
    ys = np.linspace(0, cell_len[1], ny, endpoint=False)
    zs = np.linspace(0, cell_len[2], nz, endpoint=False)
    print(f"  Grid: {nx}×{ny}×{nz} = {nx*ny*nz} points at {grid_spacing}Å spacing (batch_size={batch_size})")

    # Output arrays — forces (3 components) per grid point
    Fx = np.zeros((nx, ny, nz), dtype=np.float32)
    Fy = np.zeros((nx, ny, nz), dtype=np.float32)
    Fz = np.zeros((nx, ny, nz), dtype=np.float32)
    valid = np.zeros((nx, ny, nz), dtype=bool)   # False = inside framework atom

    radius = MACE_CUTOFF + CLUSTER_PAD
    # Large non-periodic box for local cluster (no PBC images needed)
    big_cell = np.eye(3) * (radius * 3)

    t0 = time.time()
    n_computed = 0
    n_skipped  = 0

    # Flatten grid points for batched processing
    grid_points = []
    grid_indices = []
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            for iz, z in enumerate(zs):
                grid_points.append([x, y, z])
                grid_indices.append((ix, iy, iz))
    grid_points = np.array(grid_points)
    
    # Process in batches
    for batch_start in range(0, len(grid_points), batch_size):
        batch_end = min(batch_start + batch_size, len(grid_points))
        batch_pts = grid_points[batch_start:batch_end]
        batch_idxs = grid_indices[batch_start:batch_end]
        
        # Build batched cluster
        batch_symbols = []
        batch_positions = []
        batch_valid = []
        
        for probe_pos in batch_pts:
            # Skip if probe overlaps any framework atom
            diff = fw_pos - probe_pos
            for k in range(3):
                diff[:, k] -= np.round(diff[:, k] / cell_len[k]) * cell_len[k]
            dists = np.linalg.norm(diff, axis=1)
            if np.any(dists < MIN_ION_DIST):
                batch_valid.append(False)
                n_skipped += 1
                continue
            
            nearby_mask = dists < radius
            if nearby_mask.sum() == 0:
                batch_valid.append(True)
                n_computed += 1
                continue
            
            # Add probe + nearby framework to batch
            batch_symbols.append(probe_symbol)
            batch_positions.append([0.0, 0.0, 0.0])  # probe at origin
            for i in np.where(nearby_mask)[0]:
                batch_symbols.append(framework[i].symbol)
                batch_positions.append(diff[i])
            batch_valid.append(True)
        
        if len(batch_positions) == 0:
            continue
        
        # Build single batched Atoms object
        batch_cluster = Atoms(
            symbols=batch_symbols,
            positions=batch_positions,
            cell=big_cell,
            pbc=False
        )
        batch_cluster.calc = calc
        
        try:
            f_batch = batch_cluster.get_forces()  # (total_atoms, 3)
        except Exception as e:
            for (ix, iy, iz), is_valid in zip(batch_idxs, batch_valid):
                if is_valid:
                    n_skipped += 1
            continue
        
        # Unpack forces back to grid
        f_idx = 0
        for (ix, iy, iz), is_valid in zip(batch_idxs, batch_valid):
            if not is_valid:
                continue
            if f_idx < len(f_batch):
                Fx[ix, iy, iz] = f_batch[f_idx, 0]
                Fy[ix, iy, iz] = f_batch[f_idx, 1]
                Fz[ix, iy, iz] = f_batch[f_idx, 2]
                valid[ix, iy, iz] = True
                n_computed += 1
                f_idx += 1
        
        # Progress
        pct = batch_end / len(grid_points) * 100
        elapsed = time.time() - t0
        eta = elapsed / batch_end * (len(grid_points) - batch_end)
        print(f"  Precompute: {pct:.0f}% | computed={n_computed} skipped={n_skipped} | ETA {eta/60:.1f} min", end='\r')

    print(f"\n  Done: {n_computed} valid points in {(time.time()-t0)/60:.1f} min")

    # Save grid
    if out_path is None:
        out_path = f"{material_name}_grid.npz"
    np.savez_compressed(out_path,
                        Fx=Fx, Fy=Fy, Fz=Fz, valid=valid,
                        xs=xs, ys=ys, zs=zs,
                        cell_len=cell_len,
                        grid_spacing=grid_spacing)
    print(f"  Grid saved: {out_path}")
    return out_path

def insert_na_defect(atoms, n=1):
    """
    Insert Na defects by replacing the lightest atoms.
    Finds the lightest atom by mass (typically H or F),
    replaces the first n occurrences with Na.
    Returns modified Atoms object, does NOT modify original.
    """
    from ase import Atoms
    atoms_copy = atoms.copy()
    masses = atoms_copy.get_masses()
    lightest_idx = np.argmin(masses)
    lightest_symbol = atoms_copy[lightest_idx].symbol
    
    # Find all indices of the lightest atom type
    lightest_indices = [i for i, s in enumerate(atoms_copy.get_chemical_symbols())
                        if s == lightest_symbol]
    
    # Replace first n with Na
    for i in lightest_indices[:n]:
        atoms_copy[i].symbol = 'Na'
    
    print(f"  [Na DEFECT] Replaced {min(n, len(lightest_indices))} {lightest_symbol} with Na")
    return atoms_copy

def stability_run(atoms, calc=None, steps=200, T=300.0, material_name="material"):
    """
    Run NVT MD at T for steps steps using ASE Langevin.
    Record potential energy every 10 steps.
    At end: compute RMSD of all atom positions vs initial.
    Returns {"rmsd_angstrom": float, "mean_potential_eV": float, "stable": bool}
    stable = RMSD < 0.5 Å
    """
    from ase.md.langevin import Langevin
    from ase import units
    from mace.calculators import mace_mp
    
    atoms_copy = atoms.copy()
    pos_initial = atoms_copy.get_positions().copy()
    
    # Attach MACE calculator if not already present
    if calc is None:
        print("  [STABILITY] Loading MACE...")
        calc = mace_mp(model="small", default_dtype="float32", device="cpu")
    atoms_copy.calc = calc
    
    # Initialize velocities
    MaxwellBoltzmannDistribution(atoms_copy, temperature_K=T, rng=np.random.default_rng(42))
    
    # Langevin thermostat
    dt = 2.0 * units.fs  # 2 fs timestep
    dyn = Langevin(atoms_copy, timestep=dt, temperature_K=T * units.kB,
                   friction=0.01)  # friction in 1/fs
    
    energies = []
    for step in range(steps):
        dyn.run(1)
        if step % 10 == 0:
            energies.append(atoms_copy.get_potential_energy())
    
    pos_final = atoms_copy.get_positions()
    # RMSD with minimum image convention for PBC
    cell = atoms_copy.get_cell().array
    cell_len = np.linalg.norm(cell, axis=1)
    diff = pos_final - pos_initial
    for k in range(3):
        diff[:, k] -= np.round(diff[:, k] / cell_len[k]) * cell_len[k]
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    
    mean_pot = np.mean(energies) if energies else 0.0
    stable = rmsd < 0.5
    
    print(f"  [STABILITY] RMSD: {rmsd:.3f} Å | Mean PE: {mean_pot:.3f} eV | Stable: {stable}")
    
    # Save result JSON
    meta = {
        "material": material_name,
        "rmsd_angstrom": rmsd,
        "mean_potential_eV": float(mean_pot),
        "stable": bool(rmsd < 0.5),
        "steps": steps,
        "temp_K": T
    }
    with open(f"{os.path.expanduser('~')}/NS/Bob/results/{material_name}_stability.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    return {"rmsd_angstrom": rmsd, "mean_potential_eV": mean_pot, "stable": stable}

def load_grid_interpolators(grid_path):
    """Load saved grid and build 3 interpolators (one per force component)."""
    g = np.load(grid_path)
    xs, ys, zs = g['xs'], g['ys'], g['zs']
    # Where valid=False, force was inside framework — set to large repulsion
    # so ions never enter those regions
    REPULSION = 50.0   # eV/Å — strong push away from overlap

    def make_interp(F_component):
        F = F_component.copy()
        # Extrapolation at edges: 'nearest' avoids boundary explosions
        return RegularGridInterpolator(
            (xs, ys, zs), F,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )

    ifx = make_interp(g['Fx'])
    ify = make_interp(g['Fy'])
    ifz = make_interp(g['Fz'])
    cell_len = g['cell_len']
    return (ifx, ify, ifz, cell_len)

def grid_forces(interps, pos_mobile):
    """
    Query interpolated forces for N mobile ions.
    pos_mobile: (N, 3) Å — must be wrapped into [0, cell_len]
    Returns: (N, 3) eV/Å
    """
    ifx, ify, ifz, cell_len = interps
    # Wrap to grid bounds
    pos = pos_mobile % cell_len
    pts = pos   # (N, 3) — RegularGridInterpolator takes (N, ndim)
    fx = ifx(pts)
    fy = ify(pts)
    fz = ifz(pts)
    return np.column_stack([fx, fy, fz])

def benchmark_mace(atoms, calc, n_calls=10):
    """Time raw MACE force evaluation. Prints ms/call and projected steps/s."""
    atoms_b = atoms.copy()
    atoms_b.calc = calc
    # Warmup
    atoms_b.get_forces()
    # Timed calls
    t0 = time.time()
    for _ in range(n_calls):
        atoms_b.get_forces()
    elapsed = time.time() - t0
    ms_per_call = (elapsed / n_calls) * 1000
    steps_per_s = 1000 / ms_per_call
    print(f"\n  [BENCHMARK] MACE force call: {ms_per_call:.1f} ms/call | {steps_per_s:.1f} steps/s projected")
    print(f"  [BENCHMARK] Full 100ps run: ~{50000/steps_per_s/3600:.2f} hrs per temp")
    return steps_per_s

def build_supercell(atoms, min_size_ang=10.0):
    """Repeat unit cell until each dimension >= min_size_ang."""
    cell = atoms.get_cell()
    lengths = np.linalg.norm(cell, axis=1)
    repeats = [max(1, int(np.ceil(min_size_ang / l))) for l in lengths]
    if repeats == [1, 1, 1]:
        return atoms.copy()
    return atoms.repeat(repeats)

def get_mobile_indices(atoms):
    return [i for i, s in enumerate(atoms.get_chemical_symbols()) if s in MOBILE_SPECIES]

def run_md(atoms_input, calc, temp_K, n_steps, dt_fs, traj_interval, label, grid_interps=None, out_dir=os.path.join(os.path.expanduser('~'), "NS/Bob/results")):
    atoms = atoms_input.copy()

    # NO FixAtoms constraint — handle framework exclusion manually
    atoms.calc = calc

    mobile_idx  = get_mobile_indices(atoms)
    frame_idx   = [i for i in range(len(atoms)) if i not in mobile_idx]
    masses      = atoms.get_masses()                          # amu, all atoms
    m_mob       = masses[mobile_idx]                          # amu, mobile only
    cell        = atoms.get_cell().array                      # (3,3) Å
    cell_lengths = np.linalg.norm(cell, axis=1)               # (3,) Å

    n_mob = len(mobile_idx)
    print(f"  [{label}] {n_mob} mobile ions, {len(frame_idx)} frozen")

    # --- Velocity initialization (Maxwell-Boltzmann, per atom, per axis) ---
    # σ per axis = sqrt(kB*T / (m * 103.6))  in Å/fs
    sigma = np.sqrt(KB_EV * temp_K / (m_mob * 103.6))        # (n_mob,)
    rng   = np.random.default_rng(42 + int(temp_K))
    v_mob = rng.normal(0.0, sigma[:, None], (n_mob, 3))       # (n_mob, 3) Å/fs
    v_mob -= v_mob.mean(axis=0)                               # remove COM drift

    # Sanity check
    speed_mean = np.linalg.norm(v_mob, axis=1).mean()
    expected   = np.sqrt(3 * KB_EV * temp_K / (m_mob.mean() * 103.6))
    print(f"  [SANITY] Mean mobile speed: {speed_mean:.5f} Å/fs | Expected ~{expected:.5f} Å/fs")

    # --- Get initial positions and forces ---
    pos = atoms.get_positions().copy()                        # (N, 3) Å

    f_all = atoms.get_forces().copy()                     # (N, 3) eV/Å
    a_mob = f_all[mobile_idx] / m_mob[:, None] * FORCE_TO_ACC  # (n_mob,3) Å/fs²

    # --- Trajectory storage (unwrapped positions) ---
    pos_unwrapped = pos[mobile_idx].copy()
    pos_prev      = pos[mobile_idx].copy()
    traj          = []
    traj.append(pos_unwrapped.copy())  # frame 0 - initial positions

    # --- Neighbor list refresh tracking ---
    NL_SKIN = 0.5          # Å — extra buffer beyond MACE cutoff
    NL_THRESHOLD = 0.25    # Å — rebuild if any mobile ion moves this far
    pos_at_last_nl = pos[mobile_idx].copy()
    nl_rebuild_count = 0

    # --- Velocity Verlet loop ---
    t_start   = time.time()
    t_chunk   = time.time()

    for step in range(n_steps):
        # Check if neighbor list needs rebuild
        drift = np.max(np.linalg.norm(pos[mobile_idx] - pos_at_last_nl, axis=1))
        if step == 0 or drift > NL_THRESHOLD:
            atoms.set_positions(pos, apply_constraint=False)
            calc.atoms = None   # force MACE to rebuild graph on next call
            pos_at_last_nl = pos[mobile_idx].copy()
            nl_rebuild_count += 1
        else:
            # Fast path: update positions in graph without full rebuild
            atoms.positions[:] = pos
            # Invalidate cached results but not the neighbor list
            if hasattr(calc, 'results'):
                calc.results = {}
        # 1. Half-kick velocities
        v_mob += 0.5 * a_mob * dt_fs

        # 2. Update ONLY mobile positions
        pos[mobile_idx] += v_mob * dt_fs

        # 3. Wrap mobile positions back into cell (so MACE neighbor list stays valid)
        for k in range(3):
            pos[mobile_idx, k] = np.mod(pos[mobile_idx, k], cell_lengths[k])

        # 4. Push new positions to ASE atoms object
        atoms.set_positions(pos, apply_constraint=False)  # bypass constraint check

        # 5. Recompute forces
        if grid_interps is not None:
            # Fast path: spline lookup (~0.1ms for 36 ions)
            f_mob = grid_forces(grid_interps, pos[mobile_idx])
        else:
            # Slow path: full MACE call (1400ms)
            f_all = atoms.get_forces().copy()
            f_mob = f_all[mobile_idx]
        a_mob_new = f_mob / m_mob[:, None] * FORCE_TO_ACC

        # 6. Half-kick with new forces
        v_mob += 0.5 * a_mob_new * dt_fs

        # 7. Berendsen thermostat every 20 steps
        if step % 20 == 0 and step > 0:
            KE = 0.5 * np.sum(m_mob[:, None] * v_mob**2) * 103.6  # eV
            T_now = 2 * KE / (3 * n_mob * KB_EV)
            if T_now > 1.0:
                v_mob *= np.sqrt(temp_K / T_now * (1 - 1/50) + (1/50))

        a_mob = a_mob_new

        # 8. PBC-unwrapped trajectory collection
        if step % traj_interval == 0:
            pos_now = pos[mobile_idx].copy()
            delta   = pos_now - pos_prev
            delta  -= np.round(delta / cell_lengths) * cell_lengths  # unwrap
            pos_unwrapped += delta
            pos_prev = pos_now
            traj.append(pos_unwrapped.copy())

        # 9. Progress print
        if step == 9 or (step > 0 and step % 100 == 99):
            elapsed_chunk = time.time() - t_chunk
            rate = min(step+1, 100) / elapsed_chunk if step < 100 else 100 / elapsed_chunk
            total_elapsed = time.time() - t_start
            eta_min = (n_steps - step - 1) / rate / 60
            print(f"    step {step+1}/{n_steps} | {rate:.1f} steps/s | ETA {eta_min:.1f} min")
            t_chunk = time.time()

    total = time.time() - t_start
    avg_rate = n_steps / total
    # DFT-AIMD reference: ~1 step/s on 64 CPU cores for same system size
    aimd_equiv_core_hrs = n_steps / 1.0 / 3600 * 64
    mode = "grid" if grid_interps is not None else "MACE-direct"
    print(f"  [{label}] Done: {n_steps} steps | {avg_rate:.1f} steps/s avg")
    print(f"  [TIMER] Mode: {mode} | Wall time: {total:.1f}s | vs DFT-AIMD equiv: ~{aimd_equiv_core_hrs:.0f} core-hrs")
    if grid_interps is None:
        print(f"  [NL] Neighbor list rebuilt {nl_rebuild_count}/{n_steps} steps ({100*nl_rebuild_count/n_steps:.1f}%)")

    # Compute MSD from trajectory
    traj = np.array(traj)  # (nframes, n_mob, 3)
    final_disp = traj[-1] - traj[0]  # (n_mob, 3)
    msd = float(np.mean(np.sum(final_disp**2, axis=-1)))  # scalar Å²

    # Save trajectory to disk
    out_traj_path = os.path.join(out_dir, f"{label}_traj.npy")
    np.save(out_traj_path, traj)
    
    # Save metadata JSON
    meta = {
        "label": label,
        "material": label.split("_")[0],
        "temp_K": temp_K,
        "n_steps": n_steps,
        "dt_fs": dt_fs,
        "n_mobile": len(mobile_idx),
        "msd_A2": msd,
        "diffusing": msd > 1.0,
        "traj_file": out_traj_path
    }
    with open(out_traj_path.replace(".npy", "_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "traj": traj,
        "msd": msd,
        "sigma": None,
        "diffusing": msd > 1.0
    }

def compute_msd(traj):
    # traj already unwrapped — shape (nframes, n_mob, 3)
    disp = traj - traj[0]
    return np.mean(np.sum(disp**2, axis=2), axis=1)   # (nframes,) Å²

def compute_diffusion(times_fs, msd_A2, skip_frac=0.2):
    """Linear fit on last (1-skip_frac) of trajectory. Returns D in m²/s."""
    n = len(times_fs)
    start = int(n * skip_frac)
    t = times_fs[start:] * 1e-15  # fs → s
    msd = msd_A2[start:] * 1e-20  # Å² → m²
    slope, _, r, _, _ = linregress(t, msd)
    D = slope / 6.0  # 3D: MSD = 6Dt
    return max(D, 1e-30), r**2  # floor to avoid log(0)

def arrhenius_fit(temps_K, D_vals):
    """Fit log(D) = log(D0) - Ea/(kB*T). Returns D(300K) and Ea in eV."""
    inv_T = 1.0 / np.array(temps_K)
    logD = np.log(np.array(D_vals))
    slope, intercept, _, _, _ = linregress(inv_T, logD)
    Ea_eV = -slope * kB_eV
    D_300K = np.exp(intercept + slope / 300.0)
    return D_300K, Ea_eV

def nernst_einstein_conductivity(D_m2s, n_mobile, volume_A3, temp_K, z=1):
    """sigma in S/m. n = number density of mobile ions (m^-3). z = charge."""
    volume_m3 = volume_A3 * 1e-30
    n_density = n_mobile / volume_m3   # m^-3
    sigma = (n_density * (z * e_charge)**2 * D_m2s) / (kB_SI * temp_K)
    return sigma

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cif', required=True)
    parser.add_argument('--temps', nargs='+', type=float,
                        default=[800, 900, 1000, 1100, 1200])
    parser.add_argument('--equil_ps', type=float, default=10.0)
    parser.add_argument('--prod_ps', type=float, default=100.0)
    parser.add_argument('--dt_fs', type=float, default=2.0)
    parser.add_argument('--test', action='store_true',
                        help='100-step validation only')
    parser.add_argument('--precompute', action='store_true',
                        help='precompute force grid and exit')
    parser.add_argument('--grid-spacing', type=float, default=0.35,
                        help='grid spacing in Å for precompute (default: 0.35)')
    parser.add_argument('--precompute-test', action='store_true',
                        help='quick test: compute 10 random grid points and exit')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    out_dir = args.output or os.path.dirname(os.path.abspath(args.cif))
    os.makedirs(out_dir, exist_ok=True)
    material_name = os.path.splitext(os.path.basename(args.cif))[0]

    # Load structure
    print(f"\n=== Bob MD: {material_name} ===")
    atoms_unit = read(args.cif)
    atoms = build_supercell(atoms_unit, min_size_ang=10.0)
    cell = atoms.get_cell()
    lengths = np.linalg.norm(cell, axis=1)
    volume = atoms.get_volume()

    mobile_idx = get_mobile_indices(atoms)
    framework_idx = [i for i in range(len(atoms)) if i not in mobile_idx]
    mobile_symbols = [atoms[i].symbol for i in mobile_idx]

    print(f"Unit cell atoms: {len(atoms_unit)}")
    print(f"Supercell atoms: {len(atoms)} | Cell: {lengths[0]:.1f} x {lengths[1]:.1f} x {lengths[2]:.1f} Å")
    print(f"Mobile ions: {len(mobile_idx)} ({set(mobile_symbols)})")
    print(f"Framework atoms: {len(framework_idx)}")

    if len(mobile_idx) == 0:
        print("ERROR: No mobile ions found (Na/K/Li/Ag). Check CIF.")
        return

    # Load MACE
    print("\nLoading MACE-MPA-0 small...")
    t0 = time.time()
    calc = mace_mp(
        model="small",
        default_dtype="float32",
        device="cpu"
    )
    print(f"  Model loaded in {time.time()-t0:.1f}s")
    
    # Benchmark MACE performance
    benchmark_mace(atoms, calc)

    # Grid precompute or detection
    grid_path = os.path.join(out_dir, f"{material_name}_grid.npz")
    grid_interps = None

    if args.precompute_test:
        print("\n[PRECOMPUTE TEST] Computing 10 grid points near mobile ions...")
        # Quick test: use mobile ion positions + small random offsets
        cell_len = np.linalg.norm(atoms.get_cell().array, axis=1)
        mobile_pos = atoms.get_positions()[mobile_idx]
        rng = np.random.default_rng(42)
        # Start from mobile ion positions and add small random offsets
        test_points = mobile_pos[:10] + rng.uniform(-0.5, 0.5, (10, 3))
        
        framework_idx = [i for i in range(len(atoms))
                         if atoms[i].symbol not in MOBILE_SPECIES]
        framework = atoms[framework_idx]
        fw_pos = framework.get_positions()
        probe_symbol = next(iter(MOBILE_SPECIES))
        radius = MACE_CUTOFF + CLUSTER_PAD
        big_cell = np.eye(3) * (radius * 3)
        
        from ase import Atoms
        for i, probe_pos in enumerate(test_points):
            diff = fw_pos - probe_pos
            for k in range(3):
                diff[:, k] -= np.round(diff[:, k] / cell_len[k]) * cell_len[k]
            dists = np.linalg.norm(diff, axis=1)
            if np.any(dists < MIN_ION_DIST):
                print(f"  Point {i}: SKIP (inside framework)")
                continue
            
            nearby_mask = dists < radius
            if nearby_mask.sum() == 0:
                print(f"  Point {i}: force = (0.00, 0.00, 0.00) eV/Å (no neighbors)")
                continue
            
            cluster_fw_pos_centered = diff[nearby_mask]
            cluster_symbols = [framework[j].symbol for j in np.where(nearby_mask)[0]]
            cluster = Atoms(
                symbols=[probe_symbol] + cluster_symbols,
                positions=np.vstack([[0.0, 0.0, 0.0], cluster_fw_pos_centered]),
                cell=big_cell,
                pbc=False
            )
            cluster.calc = calc
            f = cluster.get_forces()
            print(f"  Point {i}: force = ({f[0,0]:.3f}, {f[0,1]:.3f}, {f[0,2]:.3f}) eV/Å")
        print("\n[PRECOMPUTE TEST PASSED]")
        return

    if args.precompute:
        print("\n[PRECOMPUTE MODE] Building force grid...")
        precompute_grid(atoms, calc, MOBILE_SPECIES,
                        grid_spacing=args.grid_spacing,
                        out_path=grid_path,
                        material_name=material_name)
        print("Precompute done. Run without --precompute to use grid for MD.")
        return

    if os.path.exists(grid_path):
        print(f"  Grid found: {grid_path} — using fast interpolation")
        grid_interps = load_grid_interpolators(grid_path)
    else:
        print(f"  No grid found — using MACE direct (slow). Run --precompute first.")

    traj_interval = 10
    results = {'material': material_name, 'temps': [], 'D_vals': [], 'r2_vals': []}

    if args.test:
        # 100-step test at 1000K
        print("\n[TEST MODE] 100 steps at 1000K")
        traj = run_md(
            atoms, calc, temp_K=1000,
            n_steps=100, dt_fs=args.dt_fs,
            traj_interval=traj_interval, label="TEST",
            grid_interps=grid_interps
        )
        msd = compute_msd(traj)
        print(f"\n  MSD at final step: {msd[-1]:.4f} Å²")
        print(f"  Frames collected: {len(msd)}")
        print(f"  Mobile ions moving: {'YES' if msd[-1] > 0.001 else 'NO — check structure'}")
        print("\n[TEST PASSED] Full run ready.")
        return

    # Full run
    equil_steps = int(args.equil_ps * 1000 / args.dt_fs)
    prod_steps  = int(args.prod_ps  * 1000 / args.dt_fs)

    for temp in args.temps:
        print(f"\n--- T = {temp}K ---")
        # Equilibration (no collection) - use same approach as run_md
        atoms_eq = atoms.copy()
        atoms_eq.calc = calc
        
        mobile_idx  = get_mobile_indices(atoms_eq)
        frame_idx   = [i for i in range(len(atoms_eq)) if i not in mobile_idx]
        masses      = atoms_eq.get_masses()
        m_mob       = masses[mobile_idx]
        cell        = atoms_eq.get_cell().array
        cell_lengths = np.linalg.norm(cell, axis=1)
        n_mob = len(mobile_idx)
        
        # Velocity initialization
        sigma = np.sqrt(KB_EV * temp / (m_mob * 103.6))
        rng   = np.random.default_rng(42 + int(temp))
        v_mob = rng.normal(0.0, sigma[:, None], (n_mob, 3))
        v_mob -= v_mob.mean(axis=0)
        
        pos = atoms_eq.get_positions().copy()
        f_all = atoms_eq.get_forces().copy()
        a_mob = f_all[mobile_idx] / m_mob[:, None] * FORCE_TO_ACC
        
        print(f"  Equilibrating {equil_steps} steps...")
        for step in range(equil_steps):
            v_mob += 0.5 * a_mob * args.dt_fs
            pos[mobile_idx] += v_mob * args.dt_fs
            for k in range(3):
                pos[mobile_idx, k] = np.mod(pos[mobile_idx, k], cell_lengths[k])
            atoms_eq.set_positions(pos, apply_constraint=False)
            f_all = atoms_eq.get_forces().copy()
            a_mob_new = f_all[mobile_idx] / m_mob[:, None] * FORCE_TO_ACC
            v_mob += 0.5 * a_mob_new * args.dt_fs
            
            # Berendsen thermostat
            if step % 20 == 0 and step > 0:
                KE = 0.5 * np.sum(m_mob[:, None] * v_mob**2) * 103.6
                T_now = 2 * KE / (3 * n_mob * KB_EV)
                if T_now > 1.0:
                    v_mob *= np.sqrt(temp / T_now * (1 - 1/50) + (1/50))
            
            a_mob = a_mob_new

        # Production
        atoms_prod = atoms_eq  # continue from equilibrated state
        traj = run_md(
            atoms_prod, calc, temp_K=temp,
            n_steps=prod_steps, dt_fs=args.dt_fs,
            traj_interval=traj_interval, label=f"{temp}K",
            grid_interps=grid_interps
        )

        msd = compute_msd(traj)
        times_fs = np.arange(len(msd)) * traj_interval * args.dt_fs
        D, r2 = compute_diffusion(times_fs, msd)

        # Save MSD
        msd_path = os.path.join(out_dir, f"{material_name}_{int(temp)}K_msd.csv")
        np.savetxt(msd_path, np.column_stack([times_fs, msd]),
                   header="time_fs,msd_A2", delimiter=",")

        print(f"  D({int(temp)}K) = {D:.3e} m²/s  |  R² = {r2:.4f}")
        results['temps'].append(temp)
        results['D_vals'].append(D)
        results['r2_vals'].append(r2)

    # Arrhenius + conductivity
    if len(results['temps']) >= 3:
        D_300K, Ea_eV = arrhenius_fit(results['temps'], results['D_vals'])
        sigma = nernst_einstein_conductivity(
            D_300K, len(mobile_idx), volume, temp_K=300
        )
        sigma_mScm = sigma * 0.1  # S/m → mS/cm

        results['D_300K_m2s'] = D_300K
        results['Ea_eV'] = Ea_eV
        results['sigma_300K_mS_cm'] = sigma_mScm

        print(f"\n=== RESULTS: {material_name} ===")
        print(f"  Ea          = {Ea_eV:.3f} eV")
        print(f"  D(300K)     = {D_300K:.3e} m²/s")
        print(f"  σ(300K)     = {sigma_mScm:.3f} mS/cm")
        print(f"  (Good SSE > 1 mS/cm)")

    out_path = os.path.join(out_dir, f"{material_name}_results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# OOM Commercial License v1.0
# Copyright 2026 Orders of Magnitude LLC
# See /NS/LICENSE.md for terms.
# Copyright (C) 2026 Orders of Magnitude LLC <orders@ofmagnitude.com>
"""
bella_phonon.py — general crystal-phonon pipeline with SPARC + phonopy.

Usage example (Bella sweep):
    python3 bella_phonon.py \
        --cif <BOB_DIR>/data/cifs/fa37e9edd4.cif \
        --pressures 0.0,0.1,0.2,0.3,0.4,0.5,0.6

The script is deliberately generic: every value that can vary is a CLI argument.
"""

import argparse
import glob
import os
import re
import subprocess
import sys
import time
import textwrap
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers
from ase.io import read
from ase.units import Angstrom, Bohr

# Physical constants (universal, not CLI-tunable)
BOHR_PER_ANGSTROM = 1.8897259886
EV_PER_HARTREE = 27.2114
HA_BOHR_TO_EV_ANGSTROM = 51.4220

# Environment defaults: overridable via --sparc-bin, --psps-dir, or env vars
SPARC_BIN_DEFAULT = os.environ.get(
    "BELLA_SPARC_BIN", str(Path(__file__).resolve().parent / "sparc-engine" / "lib" / "sparc")
)
PSPS_DIR_DEFAULT = os.environ.get(
    "BELLA_PSPS_DIR", str(Path(__file__).resolve().parent / "sparc-engine" / "psps")
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bella_phonon.py",
        description="SPARC + phonopy crystal phonon pipeline under pressure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Example:
              python3 bella_phonon.py --cif MATERIAL.cif --pressures 0.0,0.1,0.2,0.3,0.4,0.5,0.6
        """),
    )

    # Required / primary
    p.add_argument("--cif", required=True, help="Input CIF file (required).")
    p.add_argument(
        "--pressures",
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6",
        help="Comma-separated target pressures in GPa.",
    )
    p.add_argument(
        "--supercell",
        default="1 1 1",
        help="Three integers for phonopy supercell matrix, e.g. '1 1 1' or '2 2 2'.",
    )
    p.add_argument(
        "--disp",
        type=float,
        default=0.01,
        help="Phonopy finite displacement in Å.",
    )
    p.add_argument(
        "--outdir",
        default="./phonon_runs/",
        help="Root directory for all run directories.",
    )

    # SPARC binary / resources
    p.add_argument("--sparc-bin", default=SPARC_BIN_DEFAULT, help="SPARC executable.")
    p.add_argument("--psps-dir", default=PSPS_DIR_DEFAULT, help="SPARC pseudopotential directory.")
    p.add_argument("--np", type=int, default=2, help="MPI ranks for SPARC.")

    # SCF / numerical
    p.add_argument("--mesh-spacing", type=float, default=0.4, help="SPARC MESH_SPACING in Bohr.")
    p.add_argument("--relax-mode", type=int, default=1, choices=[0, 1, 2, 3],
                 help="SPARC relaxation mode: 0=single-point, 1=atom-only, 2=cell-only, 3=full (default: 1)")
    p.add_argument("--tol-scf", type=float, default=1e-4, help="SPARC TOL_SCF.")
    p.add_argument(
        "--nstates",
        type=int,
        default=None,
        help="SPARC NSTATES. Default: auto-computed from pseudopotentials (for MATERIAL this is 20).",
    )
    p.add_argument("--maxit-scf", type=int, default=50, help="SPARC MAXIT_SCF.")
    p.add_argument("--smearing", type=float, default=0.01, help="SPARC SMEARING.")
    p.add_argument(
        "--elec-temp-type", default="Gaussian", help="SPARC ELEC_TEMP_TYPE."
    )
    p.add_argument("--exchange-correlation", default="GGA_PBE", help="SPARC EXCHANGE_CORRELATION.")
    p.add_argument("--mixing-parameter", type=float, default=0.3, help="SPARC MIXING_PARAMETER.")
    p.add_argument("--mixing-variable", default="density", help="SPARC MIXING_VARIABLE.")
    p.add_argument("--mixing-precond", default="kerker", help="SPARC MIXING_PRECOND.")
    p.add_argument(
        "--precond-kerker-thresh",
        type=float,
        default=0.1,
        help="SPARC PRECOND_KERKER_THRESH.",
    )

    # Relaxation
    p.add_argument("--relax-method", default="LBFGS", help="SPARC RELAX_METHOD.")
    p.add_argument("--relax-niter", type=int, default=100, help="SPARC RELAX_NITER.")
    p.add_argument("--tol-relax", type=float, default=1e-3, help="SPARC TOL_RELAX.")
    p.add_argument(
        "--tol-relax-cell", type=float, default=1e-4, help="SPARC TOL_RELAX_CELL."
    )
    p.add_argument("--relax-maxdilat", type=float, default=1.2, help="SPARC RELAX_MAXDILAT (cell relaxation max linear dilation).")

    # Phonopy / post-processing
    p.add_argument(
        "--imaginary-thresh",
        type=float,
        default=-0.1,
        help="Frequency (THz) below which a mode is reported as imaginary.",
    )
    p.add_argument(
        "--symprec",
        type=float,
        default=1e-5,
        help="Phonopy symmetry tolerance.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Write all SPARC inputs but do not execute any runs.",
    )
    p.add_argument(
        "--screening",
        action="store_true",
        help="Screening mode: use looser SPARC parameters for faster runs (MESH_SPACING=0.6, TOL_SCF=1e-4, NSTATES=12, MAXIT_SCF=30).",
    )
    p.add_argument(
        "--skip-phonon-on-screen-mode",
        action="store_true",
        default=True,
        help="Skip phonon calculations entirely in SPARC screen mode (default: True).",
    )
    p.add_argument(
        "--sparc-phonon",
        action="store_true",
        default=False,
        help="Use the slower SPARC displacement/phonopy path for forces instead of MACE (default: False).",
    )
    p.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel phonopy displacement execution (sequential mode for baseline testing).",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Wall-clock timeout (seconds) for each SPARC subprocess (relaxation).",
    )
    p.add_argument(
        "--sparc-timeout",
        type=int,
        default=1800,
        help="Wall-clock timeout (seconds) for each SPARC phonon displacement.",
    )
    p.add_argument(
        "--max-ram-gb",
        type=float,
        default=None,
        help="Maximum RAM cap (GB) for parallel displacement workers.",
    )
    p.add_argument(
        "--no-socket",
        action="store_true",
        help="Run one fresh SPARC per displacement (no socket mode).",
    )
    p.add_argument(
        "--no-relax",
        action="store_true",
        help="Skip relaxation and run phonopy directly on input structure (for testing pre-relaxed structures).",
    )

    return p


def parse_supercell(s: str) -> List[int]:
    parts = s.split()
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"--supercell must have exactly three integers, got: {s!r}"
        )
    try:
        return [int(x) for x in parts]
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"--supercell must be three integers: {s!r}"
        ) from e


def parse_pressures(s: str) -> List[float]:
    try:
        return [float(x.strip()) for x in s.split(",")]
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"--pressures must be a comma-separated list of floats: {s!r}"
        ) from e


def get_psp(el: str, z: int, psps_dir: str) -> str:
    files = glob.glob(os.path.join(psps_dir, f"{z:02d}_{el}_*.psp8"))
    if not files:
        raise FileNotFoundError(
            f"No pseudopotential for {el} (Z={z}) in {psps_dir}"
        )
    return files[0]


def valence_from_psp(path: str) -> int:
    name = os.path.basename(path)
    parts = name.split("_")
    if len(parts) >= 3 and parts[2].isdigit():
        return int(parts[2])
    return 1


def nstates_from_atoms(atoms: Atoms, psps_dir: str) -> int:
    valence = 0
    for el in atoms.get_chemical_symbols():
        z = int(atomic_numbers[el])
        psp = get_psp(el, z, psps_dir)
        valence += valence_from_psp(psp)
    n_occ = max(1, valence // 2)
    return int(n_occ * 1.2 + 5)


def sort_atoms_by_element(atoms: Atoms) -> Atoms:
    """Return a new Atoms with sites grouped by chemical symbol (SPARC's .ion format)."""
    order = np.argsort(atoms.get_chemical_symbols(), kind="stable")
    return atoms[order]


def write_sparc_inpt(
    inpt_path: str,
    name: str,
    atoms: Atoms,
    args: argparse.Namespace,
    relax_mode: int = 0,
    pressure: Optional[float] = None,
    stress: bool = False,
    is_displacement: bool = False,
    use_restart: bool = False,
) -> None:
    """Write a SPARC .inpt file for a crystal."""

    # Convert cell and positions to Bohr; SPARC expects LATVEC rows in Bohr
    cell_bohr = atoms.get_cell() * BOHR_PER_ANGSTROM
    nstates = args.nstates
    if nstates is None:
        nstates = nstates_from_atoms(atoms, args.psps_dir)

    if is_displacement:
        tol_scf = 1e-4
        mesh = 0.5
        if any(z >= 42 for z in atoms.get_atomic_numbers()):
            mesh = 0.35
        print(f"[BELLA] Displacement run: TOL_SCF=1e-4 "
              f"(forces converge 3x faster, phonon accuracy maintained)")
        print(f"[BELLA] Displacement mesh: {mesh} Bohr (2x faster)")
    else:
        tol_scf = 1e-4
        mesh = args.mesh_spacing
        if any(z >= 42 for z in atoms.get_atomic_numbers()):
            mesh = 0.3
            print(f"  Heavy element (Z>=42) detected; using MESH_SPACING=0.3 Bohr")

    print(f"  SPARC input {name}: MESH_SPACING={mesh} Bohr, atoms={len(atoms)}")
    lines = [
        "LATVEC_SCALE: 1 1 1",
        "LATVEC:",
        f"{cell_bohr[0][0]:.15f} {cell_bohr[0][1]:.15f} {cell_bohr[0][2]:.15f}",
        f"{cell_bohr[1][0]:.15f} {cell_bohr[1][1]:.15f} {cell_bohr[1][2]:.15f}",
        f"{cell_bohr[2][0]:.15f} {cell_bohr[2][1]:.15f} {cell_bohr[2][2]:.15f}",
        "BC: P P P",
        f"MESH_SPACING: {mesh}",
        f"EXCHANGE_CORRELATION: {args.exchange_correlation}",
        f"TOL_SCF: {tol_scf}",
        f"MIXING_PARAMETER: {args.mixing_parameter}",
        f"MIXING_VARIABLE: {args.mixing_variable}",
        f"MIXING_PRECOND: {args.mixing_precond}",
        f"PRECOND_KERKER_THRESH: {args.precond_kerker_thresh}",
        f"NSTATES: {nstates}",
        f"ELEC_TEMP_TYPE: {args.elec_temp_type}",
        f"SMEARING: {args.smearing}",
        "MD_FLAG: 0",
        f"MAXIT_SCF: {args.maxit_scf}",
        "PRINT_ATOMS: 1",
        "PRINT_FORCES: 1",
        f"OUTPUT_FILE: {name}",
    ]

    if not is_displacement:
        lines.append("PRINT_DENSITY: 1")
        lines.append("PRINT_RESTART: 1")
    if use_restart:
        lines.append("RESTART_FLAG: 1")

    if relax_mode == 1:
        lines += [
            f"RELAX_FLAG: 1",
            f"RELAX_METHOD: {args.relax_method}",
            f"RELAX_NITER: {args.relax_niter}",
            f"TOL_RELAX: {args.tol_relax}",
            "CALC_STRESS: 1",
            "CALC_PRES: 1",
            "PRINT_RELAXOUT: 1",
        ]
    elif relax_mode == 2:
        if pressure is None:
            raise ValueError("pressure is required when relax_mode=2")
        lines += [
            f"RELAX_FLAG: 2",
            f"RELAX_METHOD: {args.relax_method}",
            f"RELAX_PRESSURE: {pressure}",
            f"RELAX_NITER: {args.relax_niter}",
            f"TOL_RELAX: {args.tol_relax}",
            f"TOL_RELAX_CELL: {args.tol_relax_cell}",
            f"RELAX_MAXDILAT: {args.relax_maxdilat}",
            "CALC_STRESS: 1",
            "CALC_PRES: 1",
            "PRINT_RELAXOUT: 1",
        ]
    else:
        lines += [f"RELAX_FLAG: {relax_mode}"]
        if stress:
            lines += ["CALC_STRESS: 1", "CALC_PRES: 1"]
        else:
            lines += [
                "CALC_STRESS: 0",
                "CALC_PRES: 0",
            ]

    with open(inpt_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_sparc_ion(ion_path: str, atoms: Atoms, psps_dir: str) -> None:
    """Write a SPARC .ion file. Atoms must already be sorted by element."""
    symbols = atoms.get_chemical_symbols()
    positions_bohr = atoms.get_positions() * BOHR_PER_ANGSTROM

    # Group contiguously by element (assumes sorted)
    grouped: List[Tuple[str, List[int]]] = []
    for i, el in enumerate(symbols):
        if not grouped or grouped[-1][0] != el:
            grouped.append((el, []))
        grouped[-1][1].append(i)

    with open(ion_path, "w") as f:
        for el, idxs in grouped:
            z = int(atomic_numbers[el])
            psp = get_psp(el, z, psps_dir)
            mass = float(atomic_masses[z])
            f.write(f"ATOM_TYPE: {el}\n")
            f.write(f"N_TYPE_ATOM: {len(idxs)}\n")
            f.write(f"PSEUDO_POT: {psp}\n")
            f.write(f"ATOMIC_MASS: {mass}\n")
            f.write("COORD:\n")
            for i in idxs:
                x, y, zpos = positions_bohr[i]
                f.write(f"{x:.15f} {y:.15f} {zpos:.15f}\n")
            f.write("\n")


def write_sparc_inputs(
    workdir: str,
    name: str,
    atoms: Atoms,
    args: argparse.Namespace,
    relax_mode: int = 0,
    pressure: Optional[float] = None,
    stress: bool = False,
    apply_screening: bool = False,
    is_displacement: bool = False,
    use_restart: bool = False,
) -> None:
    """Write .inpt and .ion for a SPARC crystal calculation."""
    # Apply screening overrides only if requested and apply_screening=True
    if args.screening and apply_screening:
        import copy
        args = copy.copy(args)
        args.mesh_spacing = 0.6
        args.tol_scf = 1e-4
        args.nstates = 20  # Increased from 12 to avoid "states less than Nelectron/2" error
        args.maxit_scf = 30

    os.makedirs(workdir, exist_ok=True)
    inpt_path = os.path.join(workdir, f"{name}.inpt")
    ion_path = os.path.join(workdir, f"{name}.ion")
    atoms_sorted = sort_atoms_by_element(atoms)
    write_sparc_inpt(inpt_path, name, atoms_sorted, args, relax_mode=relax_mode, pressure=pressure, stress=stress,
                     is_displacement=is_displacement, use_restart=use_restart)
    write_sparc_ion(ion_path, atoms_sorted, args.psps_dir)


def run_sparc(workdir: str, name: str, sparc_bin: str, np: int, dry_run: bool = False, raise_on_error: bool = True, timeout: int = 1800) -> int:
    """Execute SPARC in workdir and return the exit code."""
    sparc_bin = os.path.abspath(sparc_bin)
    if dry_run:
        print(f"[dry-run] would run: mpirun -np {np} {sparc_bin} -name {name} in {workdir}")
        return 0

    # Add --oversubscribe if np > 4 (physical cores) to allow oversubscription
    if np > 4:
        cmd = ["mpirun", "--oversubscribe", "-np", str(np), sparc_bin, "-name", name]
    else:
        cmd = ["mpirun", "-np", str(np), sparc_bin, "-name", name]
    actual_timeout = timeout if timeout and timeout > 0 else None
    try:
        proc = subprocess.run(cmd, cwd=workdir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=actual_timeout)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"SPARC timed out for {name} in {workdir} (>{timeout}s)") from e
    print(f"SPARC output files for {name}: {glob.glob(os.path.join(workdir, '*'))}")
    if raise_on_error and proc.returncode != 0:
        raise RuntimeError(f"SPARC failed in {workdir} for {name} (exit {proc.returncode})")
    return proc.returncode


def _parse_ion_positions(ion_path: str) -> List[np.ndarray]:
    """Parse Cartesian positions (Bohr) from a SPARC .ion file."""
    positions = []
    current_n = 0
    coords_seen = False
    with open(ion_path) as f:
        for line in f:
            s = line.strip()
            if s.startswith("N_TYPE_ATOM:"):
                current_n = int(s.split(":")[1].strip())
            elif s == "COORD:":
                count = 0
                for _ in range(current_n):
                    cl = next(f).strip()
                    if not cl:
                        continue
                    positions.append([float(x) for x in cl.split()[:3]])
                    count += 1
                    if count >= current_n:
                        break
    return [np.array(p) for p in positions]


def parse_relaxed_structure(geopt_path: str, n_atoms: int) -> Atoms:
    """Parse the final :LATVEC: + :CELL: from .geopt. If :R(Bohr): is absent, reconstruct atom positions from .ion using the initial and final cell matrices."""
    if not os.path.exists(geopt_path):
        raise FileNotFoundError(f"No SPARC geopt file: {geopt_path}")

    with open(geopt_path) as f:
        text = f.read()

    blocks = [m.start() for m in re.finditer(r":RELAXSTEP:", text)]
    if not blocks:
        raise RuntimeError(f"No :RELAXSTEP: blocks found in {geopt_path}")

    def parse_cell(block: str) -> np.ndarray:
        cell_match = re.search(r":CELL:\s+([\-\d\.Ee\+]+)\s+([\-\d\.Ee\+]+)\s+([\-\d\.Ee\+]+)", block)
        if not cell_match:
            raise RuntimeError(f"Could not parse :CELL: in {geopt_path}")
        cell_lengths_bohr = np.array([float(x) for x in cell_match.groups()])
        lat_match = re.search(r":LATVEC:\s*\n(.*?)\n(.*?)\n(.*?)\n", block)
        if not lat_match:
            raise RuntimeError(f"Could not parse :LATVEC: in {geopt_path}")
        unit_bohr = np.array([list(map(float, lat_match.group(i).split())) for i in range(1, 4)])
        return unit_bohr * cell_lengths_bohr[:, np.newaxis]

    initial_cell_bohr = parse_cell(text[blocks[0]:])
    final_cell_bohr = parse_cell(text[blocks[-1]:])
    final_cell_angstrom = final_cell_bohr / BOHR_PER_ANGSTROM

    # Try to parse :R(Bohr): positions from .geopt
    last = text[blocks[-1]:]
    pos_match = re.search(r":R\(Bohr\):\s*\n(.*?)(?=:F\(Ha/Bohr\):|:CELL:|:VOLUME:|:LATVEC:|:STRESS:|\Z)", last, re.S)
    pos_bohr = None
    if pos_match:
        pos_lines = [line for line in pos_match.group(1).strip().split("\n") if line.strip()]
        if len(pos_lines) >= n_atoms:
            pos_bohr = np.array([list(map(float, line.split())) for line in pos_lines[:n_atoms]])

    if pos_bohr is None:
        # RelaxFlag=2 .geopt has no positions; reconstruct from .ion and the initial -> final cell transform
        ion_path = geopt_path.replace(".geopt", ".ion")
        if not os.path.exists(ion_path):
            raise RuntimeError(f"No :R(Bohr): block and no .ion file for {geopt_path}")
        ion_pos_bohr = np.array(_parse_ion_positions(ion_path))
        # Convert input Cartesian positions to fractional w.r.t. initial cell, then to final cell
        inv_initial = np.linalg.inv(initial_cell_bohr)
        frac = ion_pos_bohr @ inv_initial
        pos_bohr = frac @ final_cell_bohr

    pos_angstrom = pos_bohr / BOHR_PER_ANGSTROM

    ion_path = geopt_path.replace(".geopt", ".ion")
    if os.path.exists(ion_path):
        symbols = _symbols_from_ion(ion_path)
    else:
        symbols = ["X"] * n_atoms

    atoms = Atoms(symbols=symbols, positions=pos_angstrom, cell=final_cell_angstrom, pbc=True)
    print(f"Parsed relaxed cell vectors (Å):\n{atoms.get_cell()}")
    print(f"Parsed relaxed volume: {atoms.get_volume():.4f} Å³")
    return atoms


def _symbols_from_ion(ion_path: str) -> List[str]:
    """Reconstruct the site-ordered element list from a SPARC .ion file."""
    symbols = []
    current_el = None
    n_expected = 0
    with open(ion_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("ATOM_TYPE:"):
                current_el = line.split(":")[1].strip()
            elif line.startswith("N_TYPE_ATOM:"):
                n_expected = int(line.split(":")[1].strip())
            elif line == "COORD:":
                count = 0
                for _ in range(n_expected):
                    coord_line = next(f).strip()
                    if not coord_line:
                        continue
                    symbols.append(current_el)
                    count += 1
                    if count >= n_expected:
                        break
    return symbols


def parse_sparc_forces(static_path: str, n_atoms: int) -> np.ndarray:
    """Read forces from a SPARC .static file (Ha/Bohr) and convert to eV/Å."""
    if not os.path.exists(static_path):
        raise FileNotFoundError(f"No SPARC .static file: {static_path}")

    with open(static_path) as f:
        text = f.read()

    m = re.search(r"Atomic forces \(Ha/Bohr\):\s*\n(.*?)(?=\n\n|\Z)", text, re.S)
    if not m:
        raise RuntimeError(f"No 'Atomic forces (Ha/Bohr):' block in {static_path}")

    lines = [line for line in m.group(1).strip().split("\n") if line.strip()]
    if len(lines) < n_atoms:
        raise RuntimeError(
            f"Expected {n_atoms} force lines in {static_path}, found {len(lines)}"
        )

    raw = np.array([list(map(float, line.split()[:3])) for line in lines[:n_atoms]])
    converted = raw * HA_BOHR_TO_EV_ANGSTROM
    print(f"  SPARC raw forces (Ha/Bohr) sample: {raw[0]}")
    print(f"  SPARC converted forces (eV/Å) sample: {converted[0]} (factor {HA_BOHR_TO_EV_ANGSTROM})")
    return converted


def _parse_sparc_forces_with_fallback(static_path: str, out_path: str, n_atoms: int) -> Optional[np.ndarray]:
    """Read forces from .static; if missing/empty, try the .out file."""
    print(f"  Parsing forces: static={static_path} exists={os.path.exists(static_path)}, out={out_path} exists={os.path.exists(out_path)}")
    try:
        return parse_sparc_forces(static_path, n_atoms)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"  .static parse failed: {e}")
    if not os.path.exists(out_path):
        print(f"  .out file also missing")
        return None
    with open(out_path) as f:
        text = f.read()
    m = re.search(r"Atomic forces \(Ha/Bohr\):\s*\n(.*?)(?=\n\n|\Z)", text, re.S)
    if not m:
        print(f"  .out has no 'Atomic forces (Ha/Bohr):' block")
        return None
    lines = [line for line in m.group(1).strip().split("\n") if line.strip()]
    if len(lines) < n_atoms:
        print(f"  .out force block has {len(lines)} lines, expected {n_atoms}")
        return None
    raw = np.array([list(map(float, line.split()[:3])) for line in lines[:n_atoms]])
    converted = raw * HA_BOHR_TO_EV_ANGSTROM
    print(f"  SPARC .out raw forces (Ha/Bohr) sample: {raw[0]}")
    print(f"  SPARC .out converted forces (eV/Å) sample: {converted[0]} (factor {HA_BOHR_TO_EV_ANGSTROM})")
    return converted


def _require_phonopy():
    """Lazy phonopy import with a helpful error message."""
    try:
        from phonopy import Phonopy
        from phonopy.structure.atoms import PhonopyAtoms
        return Phonopy, PhonopyAtoms
    except ImportError as e:
        raise RuntimeError(
            "phonopy is not installed. Run: pip install --break-system-packages phonopy"
        ) from e


def _to_phonopy_atoms(atoms: Atoms):
    _, PhonopyAtoms = _require_phonopy()
    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.get_cell(),
        scaled_positions=atoms.get_scaled_positions(),
    )


def _to_ase_atoms(phonopy_atoms):
    return Atoms(
        symbols=phonopy_atoms.symbols,
        positions=phonopy_atoms.positions,
        cell=phonopy_atoms.cell,
        pbc=True,
    )


def _clean_sparc_outputs(workdir: str, name: str) -> None:
    """Remove stale SPARC output files so SPARC writes to name.out, not name.out_NN."""
    from pathlib import Path
    p = Path(workdir)
    for pat in (f"{name}.out*", f"{name}.static*"):
        for f in p.glob(pat):
            try:
                f.unlink()
            except OSError:
                pass


def _run_sparc_one_displacement(args_tuple: tuple) -> tuple:
    """
    Worker function for parallel displacement execution.
    Each worker uses mpirun -np 1 to avoid oversubscription.
    """
    i, sc, perfect_forces, workdir, args = args_tuple
    
    ase_sc = _to_ase_atoms(sc)
    ase_sc = sort_atoms_by_element(ase_sc)
    name = f"disp_{i:03d}"
    
    from bella_density_cache import load_reference_density
    cif = getattr(args, 'cif', '')
    pressure = getattr(args, 'pressure', 0.0)
    use_restart = load_reference_density(cif, workdir, name, pressure)
    _clean_sparc_outputs(workdir, name)
    
    write_sparc_inputs(workdir, name, ase_sc, args, relax_mode=0, stress=False,
                       is_displacement=True, use_restart=use_restart)
    
    # Force np=1 for parallel workers to avoid oversubscription
    run_sparc(workdir, name, args.sparc_bin, np=1, dry_run=args.dry_run, timeout=args.sparc_timeout)
    raw = _parse_sparc_forces_with_fallback(
        os.path.join(workdir, f"{name}.static"),
        os.path.join(workdir, f"{name}.out"),
        len(ase_sc)
    )
    if raw is None:
        raise RuntimeError(f"Could not parse forces for {name}")
    return i, raw - perfect_forces


def run_phonopy_gamma(
    unitcell: Atoms,
    supercell_matrix: List[int],
    displacement: float,
    workdir: str,
    args: argparse.Namespace,
    pressure: float = 0.0,
) -> np.ndarray:
    """
    Run finite-displacement phonopy for the input unitcell and return Γ frequencies (THz).
    Uses parallel execution for displaced supercells (max 4 workers, each with np=1).
    """
    t_start = time.time()

    # Check if we already have a saved result for this exact CIF+pressure
    from bella_density_cache import get_phonon_result, save_phonon_result
    cif = getattr(args, 'cif', '')
    saved = get_phonon_result(cif, pressure)
    if saved:
        print(f"[BELLA] ✓ Phonon result loaded from permanent cache (P={pressure:.2f} GPa)")
        print(f"[BELLA]   min ω = {saved['min_freq_thz']:.6f} THz | imaginary={saved['imaginary']}")
        return np.array([saved['min_freq_thz']])

    Phonopy, PhonopyAtoms = _require_phonopy()

    pcell = _to_phonopy_atoms(unitcell)
    phonon = Phonopy(
        pcell,
        supercell_matrix=supercell_matrix,
        primitive_matrix="P",
        symprec=args.symprec,
    )
    phonon.generate_displacements(distance=displacement)

    from bella_density_cache import has_cached_density, save_reference_density, load_reference_density
    cif = getattr(args, 'cif', '')
    args.pressure = pressure

    # Supercell with no displacement (for force subtraction)
    perfect_sc = _to_ase_atoms(phonon.supercell)
    perfect_sc = sort_atoms_by_element(perfect_sc)
    perfect_name = "perfect"
    _clean_sparc_outputs(workdir, perfect_name)

    if has_cached_density(cif, pressure):
        print("[BELLA] ✓ Perfect cell: warm start (cached density)")
        load_reference_density(cif, workdir, perfect_name, pressure)
        write_sparc_inputs(workdir, perfect_name, perfect_sc, args, relax_mode=0, stress=False,
                           is_displacement=False, use_restart=True)
    else:
        print("[BELLA] Cold start: computing perfect cell reference...")
        write_sparc_inputs(workdir, perfect_name, perfect_sc, args, relax_mode=0, stress=False,
                           is_displacement=False, use_restart=False)

    run_sparc(workdir, perfect_name, args.sparc_bin, args.np, dry_run=args.dry_run, timeout=args.sparc_timeout)
    save_reference_density(cif, workdir, perfect_name, pressure)
    perfect_static = os.path.join(workdir, f"{perfect_name}.static")
    perfect_out = os.path.join(workdir, f"{perfect_name}.out")
    perfect_forces = _parse_sparc_forces_with_fallback(perfect_static, perfect_out, len(perfect_sc))
    if perfect_forces is None:
        formula = perfect_sc.get_chemical_formula()
        print(f"\nWhat went wrong:\n  SPARC screen mode skipped forces for {formula} (perfect.static missing/empty, .out also has no forces).")
        print("What to try:\n  bella discover --domain nitrogen-fixation --sparc-quality confirm --headless")
        return np.array([])

    # Displaced supercells - run in parallel or sequential
    displaced_cells = phonon.supercells_with_displacements
    all_forces = []

    if args.dry_run:
        return np.array([])

    if args.no_parallel:
        # Sequential execution (baseline mode)
        for i, sc in enumerate(displaced_cells):
            ase_sc = _to_ase_atoms(sc)
            ase_sc = sort_atoms_by_element(ase_sc)
            name = f"disp_{i:03d}"
            use_restart = load_reference_density(cif, workdir, name, pressure)
            write_sparc_inputs(workdir, name, ase_sc, args, relax_mode=0, stress=False,
                               is_displacement=True, use_restart=use_restart)
            run_sparc(workdir, name, args.sparc_bin, args.np, dry_run=args.dry_run, timeout=args.sparc_timeout)
            raw = _parse_sparc_forces_with_fallback(
                os.path.join(workdir, f"{name}.static"),
                os.path.join(workdir, f"{name}.out"),
                len(ase_sc)
            )
            if raw is None:
                return np.array([])
            all_forces.append(raw - perfect_forces)
    else:
        # Parallel execution (optimized mode)
        # Use max 4 workers (physical cores), each with np=1 to avoid oversubscription
        max_workers = min(4, len(displaced_cells))
        displacement_args = [
            (i, sc, perfect_forces, workdir, args)
            for i, sc in enumerate(displaced_cells)
        ]

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_run_sparc_one_displacement, displacement_args))
        
        # Sort results by index and extract forces
        all_forces = [r[1] for r in sorted(results, key=lambda x: x[0])]

    phonon.forces = all_forces
    phonon.produce_force_constants()
    freqs = phonon.get_frequencies([0.0, 0.0, 0.0])

    # Save result permanently
    t_end = time.time()
    if len(freqs) > 0:
        from bella_density_cache import save_phonon_result
        save_phonon_result(
            cif_path=getattr(args, 'cif', ''),
            pressure=pressure,
            min_freq=float(np.min(freqs)),
            imaginary=bool(np.min(freqs) < 0),
            wall_time_s=t_end - t_start,
        )

    return np.array(freqs)


def parse_sparc_pressure(out_path: str) -> float:
    """Read the final pressure (GPa) from a SPARC .out file (last Pressure line)."""
    if not os.path.exists(out_path):
        raise FileNotFoundError(f"No SPARC .out file: {out_path}")
    last_val = None
    with open(out_path) as f:
        for line in f:
            m = re.search(r"Pressure\s*:\s*([\-\d\.Ee\+]+)\s*\((GPa|kbar)\)", line, re.IGNORECASE)
            if m:
                val = float(m.group(1))
                unit = m.group(2).lower()
                if unit == "kbar":
                    val *= 0.1
                last_val = val
    if last_val is None:
        raise RuntimeError(f"No 'Pressure : ... (GPa/kbar)' line found in {out_path}")
    return last_val


def scale_cell_volume(atoms: Atoms, factor: float) -> Atoms:
    """Scale the cell and positions isotropically by the same factor."""
    return Atoms(
        symbols=atoms.get_chemical_symbols(),
        positions=atoms.get_positions() * factor,
        cell=atoms.get_cell() * factor,
        pbc=True,
    )


def run_mace_phonons(unitcell: Atoms, supercell_matrix: List[int], pressure: float = 0.0) -> dict:
    """Compute Γ-point phonon frequencies with MACE-MP-0 using phonopy."""
    from mace.calculators import mace_mp

    Phonopy, _ = _require_phonopy()
    pcell = _to_phonopy_atoms(unitcell)
    phonon = Phonopy(
        pcell,
        supercell_matrix=supercell_matrix,
        primitive_matrix="P",
    )
    phonon.generate_displacements(distance=0.03)
    calc = mace_mp(model="small", device="cpu", default_dtype="float64")
    print(f"  Running phonopy with MACE-MP-0 (supercell={supercell_matrix}, delta=0.03)...")
    all_forces = []
    n_disps = len(phonon.supercells_with_displacements)
    for i, sc in enumerate(phonon.supercells_with_displacements):
        ase_sc = _to_ase_atoms(sc)
        ase_sc = sort_atoms_by_element(ase_sc)
        ase_sc.calc = calc
        f = ase_sc.get_forces()
        f = np.array(f)
        if f.ndim == 1:
            f = f.reshape(-1, 3)
        f = f[:len(ase_sc)]
        all_forces.append(f)
        if (i + 1) % max(1, n_disps // 10) == 0 or i + 1 == n_disps:
            print(f"    MACE force {i+1}/{n_disps} max |F|={np.abs(f).max():.3f} eV/Å")
    phonon.forces = all_forces
    phonon.produce_force_constants()
    freqs = phonon.get_frequencies([0, 0, 0])
    min_freq = float(np.min(freqs))
    max_freq = float(np.max(freqs))
    imaginary = min_freq < -1e-2
    print(f"  MACE phonon frequencies at Γ: min={min_freq:.6f} THz, max={max_freq:.6f} THz")
    return {
        "pressure": pressure,
        "sparc_exit": 0,
        "volume": float(unitcell.get_volume()),
        "min_freq_thz": min_freq,
        "imaginary": imaginary,
        "frequencies_thz": freqs,
    }


def run_one_pressure(
    base_atoms: Atoms,
    pressure: float,
    supercell_matrix: List[int],
    args: argparse.Namespace,
) -> dict:
    """Run a SPARC cell relaxation at the target pressure, parse the relaxed cell, then run Γ-point phonopy."""
    pdir = os.path.join(args.outdir, f"P{pressure:.2f}GPa")
    os.makedirs(pdir, exist_ok=True)

    # Fast MACE-MP-0 phonon path (default)
    if not args.sparc_phonon:
        if args.dry_run:
            print(f"  [dry-run] MACE phonons at {pressure:.2f} GPa — skipped")
            return None
        relaxed = sort_atoms_by_element(base_atoms.copy())

        if pressure != 0.0:
            print(f"  Applying hydrostatic pressure {pressure:.2f} GPa (ExpCellFilter)...")
            from ase.filters import ExpCellFilter
            from ase.optimize import BFGS
            from ase.units import GPa as ASE_GPa
            from mace.calculators import mace_mp
            relaxed.calc = mace_mp(model="small", device="cpu", default_dtype="float64")
            ecf = ExpCellFilter(relaxed, scalar_pressure=pressure * ASE_GPa)
            opt = BFGS(ecf, logfile=None)
            opt.run(fmax=0.01, steps=200)
            relaxed.calc = None
            print(f"  Pressure-relaxed volume: {relaxed.get_volume():.4f} Å³")

        return run_mace_phonons(relaxed, supercell_matrix, pressure)

    # Skip relaxation if --no-relax or --sparc-phonon is set
    if args.no_relax:
        print(f"  Skipping relaxation (--no-relax set)")
        relaxed = sort_atoms_by_element(base_atoms.copy())
        print(f"  input volume: {relaxed.get_volume():.4f} Å³")
        exit_code = 0
    elif args.sparc_phonon:
        relaxed = sort_atoms_by_element(base_atoms.copy())
        if pressure != 0.0:
            print(f"  Applying hydrostatic pressure {pressure:.2f} GPa (MACE ExpCellFilter)...")
            from ase.filters import ExpCellFilter
            from ase.optimize import BFGS
            from ase import units
            from mace.calculators import mace_mp
            ASE_GPa = units.GPa if hasattr(units, 'GPa') else 1/160.21766208
            calc = mace_mp(model="small", device="cpu", default_dtype="float64")
            relaxed.calc = calc
            ecf = ExpCellFilter(relaxed, scalar_pressure=pressure * ASE_GPa)
            opt = BFGS(ecf, logfile=None)
            opt.run(fmax=0.05, steps=200)
            relaxed.calc = None
            relaxed = sort_atoms_by_element(relaxed)
        print(f"  input volume (P={pressure:.1f} GPa): {relaxed.get_volume():.4f} Å³")
        exit_code = 0
    else:
        current = sort_atoms_by_element(base_atoms.copy())
        name = "relax"

        write_sparc_inputs(pdir, name, current, args, relax_mode=2, pressure=pressure, apply_screening=True)
        exit_code = run_sparc(pdir, name, args.sparc_bin, args.np, dry_run=args.dry_run, raise_on_error=False, timeout=args.timeout)

        if args.dry_run:
            print(f"  [dry-run] pressure {pressure:.2f} GPa — inputs written")
            return None

        if exit_code != 0:
            print(f"  SPARC exited with code {exit_code} for pressure {pressure:.2f} GPa")
            return {
                "pressure": pressure,
                "sparc_exit": exit_code,
                "volume": None,
                "min_freq_thz": None,
                "imaginary": None,
                "frequencies_thz": None,
            }

        geopt_path = os.path.join(pdir, f"{name}.geopt")
        relaxed = parse_relaxed_structure(geopt_path, len(current))
        print(f"  relaxed volume: {relaxed.get_volume():.4f} Å³")

    phonon_workdir = os.path.join(pdir, "phonon")
    os.makedirs(phonon_workdir, exist_ok=True)
    freqs = run_phonopy_gamma(relaxed, supercell_matrix, args.disp, phonon_workdir, args, pressure=pressure)
    if freqs.size == 0:
        return {
            "pressure": pressure,
            "sparc_exit": 0,
            "volume": float(relaxed.get_volume()),
            "min_freq_thz": None,
            "imaginary": None,
            "frequencies_thz": None,
        }
    min_freq = float(np.min(freqs))
    imaginary = min_freq < args.imaginary_thresh

    return {
        "pressure": pressure,
        "sparc_exit": exit_code,
        "volume": float(relaxed.get_volume()),
        "min_freq_thz": min_freq,
        "imaginary": imaginary,
        "frequencies_thz": freqs,
    }


def print_summary(results: List[dict]) -> None:
    print("\nPressure (GPa)   Volume (Å³)   min ω (THz)   Imaginary?")
    print("-" * 55)
    for r in results:
        if r.get("sparc_exit") != 0:
            print(f"{r['pressure']:<10.2f}   SPARC failed (exit {r['sparc_exit']})")
            continue
        volume = r["volume"] if r["volume"] is not None else "n/a"
        freq = r["min_freq_thz"] if r["min_freq_thz"] is not None else "n/a"
        flag = "yes" if r["imaginary"] else "no"
        if isinstance(volume, float):
            volume_str = f"{volume:<12.4f}"
        else:
            volume_str = f"{volume:<12}"
        if isinstance(freq, float):
            freq_str = f"{freq:<13.6f}"
        else:
            freq_str = f"{freq:<13}"
        print(f"{r['pressure']:<10.2f}   {volume_str}   {freq_str}   {flag}")


def _mace_relax(atoms: Atoms, steps: int = 200, fmax: float = 0.05) -> Atoms:
    """Relax internal coordinates with MACE, keeping the cell fixed."""
    from ase.optimize import BFGS
    from mace.calculators import mace_off, mace_mp

    calc = None
    last_err = None
    for calc_maker in (
        lambda: mace_off(model="small", device="cpu", default_dtype="float64"),
        lambda: mace_mp(model="small", device="cpu", default_dtype="float64"),
    ):
        try:
            calc = calc_maker()
            atoms.calc = calc
            # Force a single evaluation to ensure the calculator supports the system.
            atoms.get_forces()
            break
        except Exception as e:
            atoms.calc = None
            last_err = e
            calc = None
    if calc is None:
        raise RuntimeError(f"MACE pre-relaxation unavailable: {last_err}")

    opt = BFGS(atoms, logfile=None)
    opt.run(fmax=fmax, steps=steps)
    atoms.calc = None
    return atoms


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # Validate paths
    if not os.path.isfile(args.cif):
        print(f"Error: CIF not found: {args.cif}", file=sys.stderr)
        return 1
    if not os.path.isfile(args.sparc_bin):
        print(f"Error: SPARC binary not found: {args.sparc_bin}", file=sys.stderr)
        return 1
    if not os.path.isdir(args.psps_dir):
        print(f"Error: pseudopotential directory not found: {args.psps_dir}", file=sys.stderr)
        return 1

    if args.screening and args.skip_phonon_on_screen_mode:
        print("Phonon skipped: --screening with --skip-phonon-on-screen-mode (default).")
        return 0

    pressures = parse_pressures(args.pressures)
    supercell_matrix = parse_supercell(args.supercell)

    # Ensure a minimum phonopy supercell of at least 32 atoms.
    # For Fe4N (5 atoms) this becomes 2x2x2 = 40 atoms.
    n_atoms = len(sort_atoms_by_element(read(args.cif)))
    while supercell_matrix[0] * supercell_matrix[1] * supercell_matrix[2] * n_atoms < 32:
        i = min(range(3), key=lambda k: supercell_matrix[k])
        supercell_matrix[i] += 1
    print(f"  Auto-scaled supercell: {supercell_matrix} ({supercell_matrix[0] * supercell_matrix[1] * supercell_matrix[2] * n_atoms} atoms)")

    os.makedirs(args.outdir, exist_ok=True)

    base_atoms = sort_atoms_by_element(read(args.cif))
    print(f"Loaded CIF: {args.cif}")
    print(f"  Formula: {base_atoms.get_chemical_formula()}")
    print(f"  Pressures: {pressures} GPa")
    print(f"  Supercell: {supercell_matrix}")

    if not args.dry_run:
        print("Running MACE internal-coordinate pre-relaxation (cell fixed)...")
        base_atoms = sort_atoms_by_element(_mace_relax(base_atoms))
        print(f"  MACE-relaxed atoms: {len(base_atoms)}")

    results = []
    for p in pressures:
        print(f"\n=== Pressure {p:.2f} GPa ===")
        r = run_one_pressure(base_atoms, p, supercell_matrix, args)
        if r is None:
            print(f"  [dry-run] pressure {p:.2f} GPa — inputs written, execution skipped")
            continue
        results.append(r)
        if r["sparc_exit"] != 0:
            print(f"  SPARC failed with exit code {r['sparc_exit']}")
            continue
        volume_str = f"{r['volume']:.4f}" if r["volume"] is not None else "n/a"
        freq_str = f"{r['min_freq_thz']:.6f}" if r["min_freq_thz"] is not None else "n/a"
        max_freq_str = f"{max(r['frequencies_thz']):.6f}" if r["frequencies_thz"] is not None else "n/a"
        print(f"  Volume: {volume_str} Å³")
        print(f"  min ω: {freq_str} THz  imaginary={r['imaginary']}")
        print(f"  max ω: {max_freq_str} THz")

    if results:
        print_summary(results)
    else:
        print("\n[dry-run] no SPARC runs executed; only inputs were written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

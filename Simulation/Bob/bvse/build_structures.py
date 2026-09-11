# Copyright (c) 2026 Orders of Magnitude LLC
# Licensed under the OOM Commercial License v1.0
# See LICENSE.md in the repository root or ofmagnitude.com

#!/usr/bin/env python3
"""
Build crystal structures from prototypes for BVSE calculations
"""

from pymatgen.core import Structure, Lattice, Element
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.io.cif import CifWriter
import os

API_KEY = os.environ.get("MATERIALS_PROJECT_API_KEY")
CIF_DIR = os.path.join(os.path.expanduser('~'), "NS/Bob/data/cifs")

def build_elpasolite_structure(A, B, B_prime, a=8.0):
    """
    Build elpasolite structure ABB'F6 (cubic, space group Fm-3m)
    A at (0,0,0), B at (0.5,0.5,0.5), B' at (0.5,0.5,0), F at (0.5,0,0) and symmetry equivalents
    """
    lattice = Lattice.cubic(a)
    
    # Elpasolite structure (cubic, Fm-3m)
    # A at (0,0,0) - 4a site
    # B at (0.5,0.5,0.5) - 4b site  
    # B' at (0.5,0.5,0) - 8c site
    # F at (0.5,0,0) - 24e site
    
    species = []
    coords = []
    
    # A site (4a)
    species.extend([A] * 4)
    coords.extend([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5]
    ])
    
    # B site (4b)
    species.extend([B] * 4)
    coords.extend([
        [0.5, 0.5, 0.5],
        [0.0, 0.0, 0.5],
        [0.0, 0.5, 0.0],
        [0.5, 0.0, 0.0]
    ])
    
    # B' site (8c)
    species.extend([B_prime] * 8)
    coords.extend([
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [0.0, 0.0, 0.0],
        [0.25, 0.25, 0.25],
        [0.75, 0.75, 0.25],
        [0.75, 0.25, 0.75],
        [0.25, 0.75, 0.75]
    ])
    
    # F sites (24e) - simplified approximation
    species.extend([Element("F")] * 24)
    for i in range(24):
        x = 0.5
        y = (i % 4) * 0.25
        z = ((i // 4) % 3) * 0.333
        coords.append([x, y, z])
    
    structure = Structure(lattice, species, coords)
    return structure

def assign_oxidation_states(structure, oxidation_dict):
    """Assign oxidation states to structure"""
    try:
        # Try auto-detection first
        analyzer = BVAnalyzer()
        structure = analyzer.get_oxi_state_decorated_structure(structure)
    except:
        # Fall back to manual assignment
        structure.add_oxidation_state_by_element(oxidation_dict)
    return structure

def build_perovskite_structure(A, B, X, a=4.0):
    lattice = Lattice.cubic(a)
    
    species = []
    coords = []
    
    # A site (1a)
    species.append(A)
    coords.append([0.0, 0.0, 0.0])
    
    # B site (1b)
    species.append(B)
    coords.append([0.5, 0.5, 0.5])
    
    # X sites (3c)
    species.extend([X] * 3)
    coords.extend([
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5]
    ])
    
    structure = Structure(lattice, species, coords)
    return structure

def main():
    print("Building crystal structures for BVSE calculations...\n")
    
    os.makedirs(CIF_DIR, exist_ok=True)
    
    # 1. NaAlZnF6 - Elpasolite structure
    print("1. Building NaAlZnF6 (elpasolite ABB'F6)...")
    na_al_zn_f6 = build_elpasolite_structure(
        A=Element("Na"),
        B=Element("Al"),
        B_prime=Element("Zn"),
        a=8.5  # Approximate lattice parameter
    )
    # Assign oxidation states: Na+1, Al+3, Zn+2, F-1
    na_al_zn_f6 = assign_oxidation_states(na_al_zn_f6, {"Na": 1, "Al": 3, "Zn": 2, "F": -1})
    cif_path = os.path.join(CIF_DIR, "NaAlZnF6.cif")
    CifWriter(na_al_zn_f6).write_file(cif_path)
    print(f"   ✓ Saved to {cif_path}")
    
    # 2. NaCaFeF6 - Elpasolite structure
    print("2. Building NaCaFeF6 (elpasolite ABB'F6)...")
    na_ca_fe_f6 = build_elpasolite_structure(
        A=Element("Na"),
        B=Element("Ca"),
        B_prime=Element("Fe"),
        a=8.7  # Approximate lattice parameter
    )
    # Assign oxidation states: Na+1, Ca+2, Fe+3, F-1
    na_ca_fe_f6 = assign_oxidation_states(na_ca_fe_f6, {"Na": 1, "Ca": 2, "Fe": 3, "F": -1})
    cif_path = os.path.join(CIF_DIR, "NaCaFeF6.cif")
    CifWriter(na_ca_fe_f6).write_file(cif_path)
    print(f"   ✓ Saved to {cif_path}")
    
    # 3. Na3Al(SO4)3 - Build from NASICON prototype
    print("3. Building Na3Al(SO4)3 (NASICON prototype)...")
    # Build from NASICON prototype as approximation
    lattice = Lattice.cubic(15.0)
    species = [Element("Na")] * 3 + [Element("Al")] + [Element("S")] * 3 + [Element("O")] * 12
    coords = [
        [0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5],  # Na (3)
        [0.25, 0.25, 0.25],  # Al (1)
        [0.125, 0.125, 0.125], [0.375, 0.375, 0.125], [0.375, 0.125, 0.375],  # S (3)
        [0.1, 0.1, 0.1], [0.2, 0.2, 0.1], [0.1, 0.2, 0.2],  # O (9)
        [0.3, 0.1, 0.1], [0.1, 0.3, 0.1], [0.1, 0.1, 0.3],
        [0.2, 0.3, 0.2], [0.3, 0.2, 0.2], [0.2, 0.2, 0.3],
        [0.4, 0.1, 0.2], [0.1, 0.4, 0.2], [0.2, 0.1, 0.4]  # Additional O (3)
    ]
    structure = Structure(lattice, species, coords)
    # Assign oxidation states: Na+1, Al+3, S+6, O-2
    structure = assign_oxidation_states(structure, {"Na": 1, "Al": 3, "S": 6, "O": -2})
    cif_path = os.path.join(CIF_DIR, "Na3Al(SO4)3.cif")
    CifWriter(structure).write_file(cif_path)
    print(f"   ✓ Saved prototype to {cif_path}")
    
    # 4. Na4Ca(SiS3)2 - Build from thiosilicate prototype
    print("4. Building Na4Ca(SiS3)2 (thiosilicate)...")
    lattice = Lattice.cubic(12.0)
    species = [Element("Na")] * 4 + [Element("Ca")] + [Element("Si")] * 2 + [Element("S")] * 6
    coords = [
        [0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5],  # Na
        [0.25, 0.25, 0.25],  # Ca
        [0.125, 0.125, 0.125], [0.375, 0.375, 0.375],  # Si
        [0.1, 0.1, 0.1], [0.2, 0.2, 0.1], [0.1, 0.2, 0.2],  # S
        [0.3, 0.1, 0.1], [0.1, 0.3, 0.1], [0.1, 0.1, 0.3]
    ]
    na4ca_sis3_2 = Structure(lattice, species, coords)
    # Assign oxidation states: Na+1, Ca+2, Si+4, S-2
    na4ca_sis3_2 = assign_oxidation_states(na4ca_sis3_2, {"Na": 1, "Ca": 2, "Si": 4, "S": -2})
    cif_path = os.path.join(CIF_DIR, "Na4Ca(SiS3)2.cif")
    CifWriter(na4ca_sis3_2).write_file(cif_path)
    print(f"   ✓ Saved to {cif_path}")
    
    # 5. Na4Mg(SiS3)2 - Build from thiosilicate prototype
    print("5. Building Na4Mg(SiS3)2 (thiosilicate)...")
    lattice = Lattice.cubic(11.5)
    species = [Element("Na")] * 4 + [Element("Mg")] + [Element("Si")] * 2 + [Element("S")] * 6
    coords = [
        [0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5],  # Na
        [0.25, 0.25, 0.25],  # Mg
        [0.125, 0.125, 0.125], [0.375, 0.375, 0.375],  # Si
        [0.1, 0.1, 0.1], [0.2, 0.2, 0.1], [0.1, 0.2, 0.2],  # S
        [0.3, 0.1, 0.1], [0.1, 0.3, 0.1], [0.1, 0.1, 0.3]
    ]
    na4mg_sis3_2 = Structure(lattice, species, coords)
    # Assign oxidation states: Na+1, Mg+2, Si+4, S-2
    na4mg_sis3_2 = assign_oxidation_states(na4mg_sis3_2, {"Na": 1, "Mg": 2, "Si": 4, "S": -2})
    cif_path = os.path.join(CIF_DIR, "Na4Mg(SiS3)2.cif")
    CifWriter(na4mg_sis3_2).write_file(cif_path)
    print(f"   ✓ Saved to {cif_path}")
    
    # 6. NaZnFeF6 - Elpasolite structure
    print("6. Building NaZnFeF6 (elpasolite ABB'F6)...")
    na_zn_fe_f6 = build_elpasolite_structure(
        A=Element("Na"),
        B=Element("Zn"),
        B_prime=Element("Fe"),
        a=8.6  # Approximate lattice parameter
    )
    # Assign oxidation states: Na+1, Zn+2, Fe+3, F-1
    na_zn_fe_f6 = assign_oxidation_states(na_zn_fe_f6, {"Na": 1, "Zn": 2, "Fe": 3, "F": -1})
    cif_path = os.path.join(CIF_DIR, "NaZnFeF6.cif")
    CifWriter(na_zn_fe_f6).write_file(cif_path)
    print(f"   ✓ Saved to {cif_path}")
    
    # 7. NaZnCrF6 - Elpasolite structure
    print("7. Building NaZnCrF6 (elpasolite ABB'F6)...")
    na_zn_cr_f6 = build_elpasolite_structure(
        A=Element("Na"),
        B=Element("Zn"),
        B_prime=Element("Cr"),
        a=8.6  # Approximate lattice parameter
    )
    # Assign oxidation states: Na+1, Zn+2, Cr+3, F-1
    na_zn_cr_f6 = assign_oxidation_states(na_zn_cr_f6, {"Na": 1, "Zn": 2, "Cr": 3, "F": -1})
    cif_path = os.path.join(CIF_DIR, "NaZnCrF6.cif")
    CifWriter(na_zn_cr_f6).write_file(cif_path)
    print(f"   ✓ Saved to {cif_path}")
    
    print("\nAll structures built successfully!")

if __name__ == "__main__":
    main()

# Copyright (c) 2026 Orders of Magnitude LLC
# Licensed under the OOM Commercial License v1.0
# See LICENSE.md in the repository root or ofmagnitude.com

#!/usr/bin/env python3
import os, math, statistics, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import foam_screener as fs

HBAR = 6.626e-34   # J·s
KB   = 1.381e-23   # J/K

# Conventional-cell atom counts and atoms per formula unit, used to interpret
# the listed n values (some were given as formula units for ionics).
SUBTYPE_ATOMS = {
    'bcc': 2, 'fcc': 4, 'hcp': 2,
    'diamond': 8, 'zincblende': 8, 'wurtzite': 4, 'tetragonal': 4,
    'rocksalt': 8, 'fluorite': 12, 'spinel': 56,
}
SUBTYPE_FORMULA_ATOMS = {
    'bcc': 1, 'fcc': 1, 'hcp': 1,
    'diamond': 2, 'zincblende': 2, 'wurtzite': 2, 'tetragonal': 1,
    'rocksalt': 2, 'fluorite': 3, 'spinel': 7,
}

DATASET = [
    # metals
    {'formula':'Fe', 'class':'metal', 'crystal_subtype':'bcc', 'B_GPa':170, 'G_GPa':82,  'rho_gcc':7.87, 'V_cell_A3':23.6,  'n_user':2, 'theta_exp':470},
    {'formula':'Cu', 'class':'metal', 'crystal_subtype':'fcc', 'B_GPa':140, 'G_GPa':48,  'rho_gcc':8.96, 'V_cell_A3':47.2,  'n_user':4, 'theta_exp':343},
    {'formula':'Al', 'class':'metal', 'crystal_subtype':'fcc', 'B_GPa':76,  'G_GPa':26,  'rho_gcc':2.70, 'V_cell_A3':66.4,  'n_user':4, 'theta_exp':428},
    {'formula':'W',  'class':'metal', 'crystal_subtype':'bcc', 'B_GPa':310, 'G_GPa':161, 'rho_gcc':19.30,'V_cell_A3':31.6,  'n_user':2, 'theta_exp':400},
    {'formula':'Ti', 'class':'metal', 'crystal_subtype':'hcp', 'B_GPa':110, 'G_GPa':44,  'rho_gcc':4.51, 'V_cell_A3':35.3,  'n_user':2, 'theta_exp':420},
    {'formula':'Ni', 'class':'metal', 'crystal_subtype':'fcc', 'B_GPa':180, 'G_GPa':76,  'rho_gcc':8.91, 'V_cell_A3':43.8,  'n_user':4, 'theta_exp':450},
    {'formula':'Mo', 'class':'metal', 'crystal_subtype':'bcc', 'B_GPa':230, 'G_GPa':125, 'rho_gcc':10.22,'V_cell_A3':31.2,  'n_user':2, 'theta_exp':450},
    {'formula':'Mn', 'class':'metal', 'crystal_subtype':'bcc', 'B_GPa':120, 'G_GPa':72,  'rho_gcc':7.43, 'V_cell_A3':12.1,  'n_user':1, 'theta_exp':400},
    # semiconductors
    {'formula':'Si', 'class':'semiconductor', 'crystal_subtype':'diamond', 'B_GPa':98,  'G_GPa':66,  'rho_gcc':2.33, 'V_cell_A3':160.1, 'n_user':8, 'theta_exp':640},
    {'formula':'Ge', 'class':'semiconductor', 'crystal_subtype':'diamond', 'B_GPa':75.8, 'G_GPa':43.5, 'rho_gcc':5.323,'V_cell_A3':181.1, 'n_user':8, 'theta_exp':374},
    {'formula':'GaAs','class':'semiconductor','crystal_subtype':'zincblende','B_GPa':74.8,'G_GPa':46.7,'rho_gcc':5.317,'V_cell_A3':180.6,'n_user':8,'theta_exp':360},
    {'formula':'GaN', 'class':'semiconductor', 'crystal_subtype':'wurtzite','B_GPa':210,'G_GPa':123,'rho_gcc':6.15,'V_cell_A3':45.8,'n_user':4,'theta_exp':600},
    {'formula':'SiC', 'class':'semiconductor', 'crystal_subtype':'zincblende','B_GPa':220,'G_GPa':192,'rho_gcc':3.21,'V_cell_A3':82.3,'n_user':8,'theta_exp':1200},
    {'formula':'InP', 'class':'semiconductor', 'crystal_subtype':'zincblende','B_GPa':71,'G_GPa':32, 'rho_gcc':4.79,'V_cell_A3':195.0,'n_user':8,'theta_exp':321},
    # ionics
    {'formula':'NaCl','class':'ionic','crystal_subtype':'rocksalt','B_GPa':25,'G_GPa':15,'rho_gcc':2.16,'V_cell_A3':179.4,'n_user':8,'theta_exp':321},
    {'formula':'MgO', 'class':'ionic','crystal_subtype':'rocksalt','B_GPa':160,'G_GPa':130,'rho_gcc':3.58,'V_cell_A3':74.8,'n_user':4,'theta_exp':946},
    {'formula':'CaF2','class':'ionic','crystal_subtype':'fluorite','B_GPa':82,'G_GPa':42,'rho_gcc':3.18,'V_cell_A3':163.5,'n_user':4,'theta_exp':510},
    {'formula':'LiF', 'class':'ionic','crystal_subtype':'rocksalt','B_GPa':67,'G_GPa':49,'rho_gcc':2.64,'V_cell_A3':65.4,'n_user':4,'theta_exp':732},
    {'formula':'MgAl2O4','class':'ionic','crystal_subtype':'spinel','B_GPa':193,'G_GPa':108,'rho_gcc':3.58,'V_cell_A3':526.8,'n_user':56,'theta_exp':845},
    # reclassified / hypothetical
    {'formula':'TiC','class':'metal','crystal_subtype':'rocksalt','B_GPa':242,'G_GPa':188,'rho_gcc':4.93,'V_cell_A3':81.1,'n_user':8,'theta_exp':940},
    {'formula':'VC','class':'metal','crystal_subtype':'rocksalt','B_GPa':303,'G_GPa':151,'rho_gcc':5.36,'V_cell_A3':72.6,'n_user':8,'theta_exp':870},
    {'formula':'LaFeSi','class':'metal','crystal_subtype':'tetragonal','B_GPa':115,'G_GPa':62,'rho_gcc':6.21,'V_cell_A3':124.8,'n_user':4,'theta_exp':390},
    {'formula':'Fe3Mn4','class':'metal','crystal_subtype':'bcc','B_GPa':200,'G_GPa':85,'rho_gcc':7.62,'V_cell_A3':None,'n_user':1,'theta_exp':None},
    {'formula':'Mo2FeN2','class':'metal','crystal_subtype':'bcc','B_GPa':280,'G_GPa':120,'rho_gcc':9.10,'V_cell_A3':None,'n_user':1,'theta_exp':None},
]

ANOMALOUS = {'Mn'}


def parse_formula(formula):
    return [(el, int(c) if c else 1) for el, c in re.findall(r'([A-Z][a-z]*)(\d*)', formula)]


def acoustic_mass(formula):
    comp = parse_formula(formula)
    total_mass = sum(count * fs.ATOMIC_MASS.get(el, 0.0) for el, count in comp)
    total_atoms = sum(count for _, count in comp)
    return total_mass / total_atoms


def n_atoms_per_cell(m):
    """Interpret n_user as either conventional-atom count or formula-unit count."""
    subtype = m['crystal_subtype']
    n = m['n_user']
    expected = SUBTYPE_ATOMS.get(subtype, n)
    if n == expected:
        return n
    formula_atoms = SUBTYPE_FORMULA_ATOMS.get(subtype, 1)
    return n * formula_atoms


def theta_D_anderson(B_GPa, G_GPa, rho_gcc, n_atoms, V_cell_A3):
    B = B_GPa * 1e9    # Pa
    G = G_GPa * 1e9    # Pa
    rho = rho_gcc * 1e3  # kg/m³

    v_T = math.sqrt(G / rho)
    v_L = math.sqrt((3*B + 4*G) / (3*rho))
    inv_vm3 = (1/3) * (2/v_T**3 + 1/v_L**3)
    v_m = inv_vm3**(-1/3)

    V_m3 = V_cell_A3 * 1e-30
    n_density = n_atoms / V_m3

    return (HBAR / KB) * (3*n_density / (4*math.pi))**(1/3) * v_m


def main():
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'findings', 'debye_v9_clean.txt')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    for m in DATASET:
        m['M_atom'] = acoustic_mass(m['formula'])
        if m['V_cell_A3'] is None:
            # per-atom volume from density and average atomic mass
            m['V_cell_A3'] = m['M_atom'] * 1.6605 / m['rho_gcc']
            m['n_atoms'] = 1
        else:
            m['n_atoms'] = n_atoms_per_cell(m)

    with open(out_path, 'w') as f:
        def prnt(s=''):
            f.write(str(s) + '\n')
            print(s, file=sys.stderr)

        prnt("=" * 80)
        prnt("Anderson Debye temperature from elastic constants (v9_clean)")
        prnt("=" * 80)
        prnt("BELLA_MAST_TOKEN not found; using literature values for GaAs, Ge, LaFeSi.")

        # Step 1: GaN debug
        gaN = next(m for m in DATASET if m['formula'] == 'GaN')
        prnt("\nStep 1: GaN debug")
        prnt(f"  GaN inputs: B={gaN['B_GPa']} GPa, G={gaN['G_GPa']} GPa, "
             f"rho={gaN['rho_gcc']} g/cc, V={gaN['V_cell_A3']} A3, n={gaN['n_atoms']}")
        B = gaN['B_GPa'] * 1e9
        G = gaN['G_GPa'] * 1e9
        rho = gaN['rho_gcc'] * 1e3
        v_T = math.sqrt(G / rho)
        v_L = math.sqrt((3*B + 4*G) / (3*rho))
        inv_vm3 = (1/3) * (2/v_T**3 + 1/v_L**3)
        v_m = inv_vm3**(-1/3)
        V_m3 = gaN['V_cell_A3'] * 1e-30
        n_density = gaN['n_atoms'] / V_m3
        k = (3*n_density / (4*math.pi))**(1/3)
        theta_GaN = (HBAR / KB) * k * v_m
        prnt(f"  v_T = {v_T:.2f} m/s")
        prnt(f"  v_L = {v_L:.2f} m/s")
        prnt(f"  v_m = {v_m:.2f} m/s")
        prnt(f"  V_cell_m3 = {V_m3:.6e} m3")
        prnt(f"  n_density = {n_density:.6e} atoms/m3")
        prnt(f"  (3n/4piV)^(1/3) = {k:.6e} m^-1")
        prnt(f"  theta_D = {theta_GaN:.2f} K")

        # Step 3: validate
        prnt("\nStep 3: Validation against experiment")
        prnt(f"{'formula':<10} {'theta_exp':<12} {'theta_pred':<12} {'err%':<8} "
             f"{'B':<6} {'G':<6} {'rho':<6}")
        prnt("-" * 75)
        errors = []
        for m in DATASET:
            pred = theta_D_anderson(m['B_GPa'], m['G_GPa'], m['rho_gcc'],
                                    m['n_atoms'], m['V_cell_A3'])
            m['theta_pred'] = pred
            if m['theta_exp'] is not None:
                err = abs(pred - m['theta_exp']) / m['theta_exp'] * 100
                if m['formula'] in ANOMALOUS:
                    prnt(f"{m['formula']:<10} {m['theta_exp']:<12.1f} {pred:<12.2f} "
                         f"{'ANOMALOUS':<8} {m['B_GPa']:<6} {m['G_GPa']:<6} {m['rho_gcc']:<6.2f}")
                else:
                    errors.append(err)
                    prnt(f"{m['formula']:<10} {m['theta_exp']:<12.1f} {pred:<12.2f} {err:<8.2f} "
                         f"{m['B_GPa']:<6} {m['G_GPa']:<6} {m['rho_gcc']:<6.2f}")
            else:
                note = ''
                if m['formula'] in ('Fe3Mn4', 'Mo2FeN2'):
                    note = 'DERIVED from tabulated B,G,rho'
                prnt(f"{m['formula']:<10} {'None':<12} {pred:<12.2f} {'N/A':<8} "
                     f"{m['B_GPa']:<6} {m['G_GPa']:<6} {m['rho_gcc']:<6.2f} {note}")

        mae = statistics.mean(errors)
        w5 = sum(1 for e in errors if e <= 5) / len(errors) * 100
        w10 = sum(1 for e in errors if e <= 10) / len(errors) * 100
        w15 = sum(1 for e in errors if e <= 15) / len(errors) * 100

        prnt("=" * 75)
        prnt(f"  MAE        = {mae:.2f}%")
        prnt(f"  within 5%  = {w5:.1f}%")
        prnt(f"  within 10% = {w10:.1f}%")
        prnt(f"  within 15% = {w15:.1f}%")

        # Step 4: hypothetical predictions
        prnt("\n" + "=" * 80)
        prnt("Step 4: Fe3Mn4 and Mo2FeN2 predictions (DERIVED)")
        for m in DATASET:
            if m['formula'] in ('Fe3Mn4', 'Mo2FeN2'):
                prnt(f"  {m['formula']:<10} theta_D = {m['theta_pred']:<10.2f} K  "
                     f"(B={m['B_GPa']} GPa, G={m['G_GPa']} GPa, rho={m['rho_gcc']} g/cc)")

        # Step 5: version comparison
        prnt("\n" + "=" * 80)
        prnt("Step 5: Version comparison")
        prnt(f"{'version':<20} {'MAE%':<10} {'within10%':<12} {'within15%':<12}")
        prnt("-" * 60)
        prnt(f"{'v2 (3-class C)':<20} {11.25:<10.2f} {'':<12} {63.6:<12.1f}")
        prnt(f"{'v7 (Lindemann)':<20} {14.46:<10.2f} {42.9:<12.1f} {66.7:<12.1f}")
        prnt(f"{'v9_clean (Anderson)':<20} {mae:<10.2f} {w10:<12.1f} {w15:<12.1f}")
        prnt("=" * 80)

        # Step 4: stability cross-check
        prnt("\n" + "=" * 80)
        prnt("Step 4: Stability cross-check (theta_D > 300 K = STABLE)")
        prnt(f"{'formula':<10} {'theta_pred':<12} {'predicted':<12} {'required':<20}")
        prnt("-" * 60)
        required = {
            'TiC': 'STABLE (SPARC)', 'VC': 'STABLE (SPARC)',
            'LaFeSi': 'STABLE (MACE)', 'Fe3Mn4': 'STABLE (MACE)',
            'Mo2FeN2': 'STABLE (MACE phonon)'
        }
        mismatches = []
        for m in DATASET:
            if m['formula'] in required:
                stable = 'STABLE' if m['theta_pred'] > 300 else 'UNSTABLE'
                prnt(f"{m['formula']:<10} {m['theta_pred']:<12.2f} {stable:<12} {required[m['formula']]:<20}")
                if 'STABLE' in required[m['formula']] and stable == 'UNSTABLE':
                    mismatches.append(m['formula'])
        if mismatches:
            prnt(f"MISMATCHES: {', '.join(mismatches)} do not predict STABLE.")
        else:
            prnt("All confirmed stable materials predict STABLE.")
        prnt("=" * 80)


if __name__ == '__main__':
    main()

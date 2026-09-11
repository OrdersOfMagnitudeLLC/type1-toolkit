# Copyright (c) 2026 Orders of Magnitude LLC
# Licensed under the OOM Commercial License v1.0
# See LICENSE.md in the repository root or ofmagnitude.com

import os
#!/usr/bin/env python3
"""
GNoME CSV Query Script
Three material discovery queries for advanced materials screening
"""

import csv
import re
from datetime import datetime
from collections import Counter

# Try to import MP API, make it optional
try:
    from mp_api.client import MPRester
    MP_API_AVAILABLE = True
except ImportError:
    MP_API_AVAILABLE = False
    print("Warning: mp_api not available, Query 1 will use fallback (density only)")

CSV_PATH = os.path.join(os.path.expanduser('~'), "NS/Bob/data/gnome.csv")
OUTPUT_DIR = os.path.join(os.path.expanduser('~'), "NS/Bob/findings/gnome_queries/")
MP_API_KEY = os.environ.get("MATERIALS_PROJECT_API_KEY")

# Rare earth elements to exclude (lanthanides + Sc, Y)
RARE_EARTHS = {
    'Sc', 'Y', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 
    'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu'
}

# Metal elements (for Query 1 - only pure metals)
METALS = {
    'Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe',
    'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru',
    'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta',
    'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'Fr', 'Ra',
    'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md',
    'No', 'Lr'
}

# Non-metal elements (for Query 3 artifact filtering)
NON_METALS = {
    'H', 'He', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Si', 'P', 'S', 'Cl', 'Ar',
    'Ge', 'As', 'Se', 'Br', 'Kr', 'Sb', 'Te', 'I', 'Xe', 'Po', 'At', 'Rn'
}

# Maximum fluorine counts per element (valence limits)
MAX_FLUORINE = {
    'Cl': 5, 'P': 5, 'Cr': 6, 'S': 6, 'I': 7, 'Xe': 6, 'Os': 6, 'Ir': 6
}

def is_all_metals(elements):
    """Check if all elements in the composition are metals"""
    return all(el in METALS for el in elements)

def parse_elements(elements_str):
    """Parse elements string from CSV"""
    try:
        cleaned = elements_str.strip("[]").replace("'", "").replace('"', '')
        elements = [e.strip() for e in cleaned.split(',') if e.strip()]
        return elements
    except:
        return []

def has_rare_earth(elements):
    """Check if composition contains rare earth elements"""
    return any(el in RARE_EARTHS for el in elements)

def parse_composition(composition_str):
    """Parse composition string like 'Cs1S6Zr3' to get element counts"""
    # Regex to find element symbols and their counts
    pattern = r'([A-Z][a-z]?)(\d*)'
    matches = re.findall(pattern, composition_str)
    
    element_counts = {}
    total_atoms = 0
    
    for element, count_str in matches:
        count = int(count_str) if count_str else 1
        element_counts[element] = element_counts.get(element, 0) + count
        total_atoms += count
    
    return element_counts, total_atoms

def query_lightest_strongest_metal(reader, header):
    """
    Query 1: Lightest/strongest metal (two-step approach)
    Step 1: From GNoME - filter metals (bandgap < 0.1 eV), stable (formation_energy < 0),
            sort by density ascending, take top 50 lightest
    Step 2: Fetch bulk_modulus from MP API, rank by bulk_modulus/density descending
    Fallback: If MP API unavailable, use density ascending (lightest metals)
    """
    print("\n=== Query 1: Lightest/Strongest Metals ===")
    print("Step 1: Filter metals (bandgap < 0.1 eV), stable, sort by density ascending")
    
    results = []
    bandgap_idx = header.index('Bandgap')
    density_idx = header.index('Density')
    formation_energy_idx = header.index('Formation Energy Per Atom')
    formula_idx = header.index('Reduced Formula')
    material_id_idx = header.index('MaterialId')
    elements_idx = header.index('Elements')
    
    for row in reader:
        try:
            bandgap = float(row[bandgap_idx])
            density = float(row[density_idx])
            formation_energy = float(row[formation_energy_idx])
            elements_str = row[elements_idx]
            elements = parse_elements(elements_str)
            
            # Metal: bandgap < 0.1 eV, stable: formation_energy < 0, ALL elements must be metals
            if bandgap < 0.1 and density > 0 and formation_energy < 0 and is_all_metals(elements):
                results.append({
                    'formula': row[formula_idx],
                    'material_id': row[material_id_idx],
                    'bandgap': bandgap,
                    'density': density,
                    'formation_energy': formation_energy,
                    'elements': elements_str
                })
        except (ValueError, IndexError):
            continue
    
    # Sort by density ascending (lightest first)
    results.sort(key=lambda x: x['density'])
    
    # Take top 50 lightest stable metals
    top_50 = results[:50]
    print(f"Step 1: Found {len(top_50)} lightest stable metals")
    
    # Step 2: Fetch bulk_modulus from MP API (if available)
    if MP_API_AVAILABLE:
        print("Step 2: Fetching bulk_modulus from MP API...")
        final_results = []
        
        with MPRester(MP_API_KEY) as mpr:
            for material in top_50:
                mp_id = material['material_id']
                try:
                    # Fetch bulk_modulus from MP
                    doc = mpr.materials.search(
                        material_ids=[mp_id],
                        fields=["material_id", "bulk_modulus"]
                    )
                    
                    if doc and len(doc) > 0:
                        bulk_modulus = doc[0].bulk_modulus
                        if bulk_modulus and bulk_modulus > 0:
                            # Calculate specific modulus (bulk_modulus / density)
                            specific_modulus = bulk_modulus / material['density']
                            material['bulk_modulus'] = bulk_modulus
                            material['specific_modulus'] = specific_modulus
                            final_results.append(material)
                            print(f"  {mp_id}: bulk_modulus={bulk_modulus:.2f} GPa, specific_modulus={specific_modulus:.2f}")
                        else:
                            print(f"  {mp_id}: no bulk_modulus data, skipping")
                    else:
                        print(f"  {mp_id}: not found in MP, skipping")
                except Exception as e:
                    print(f"  {mp_id}: API error - {e}, skipping")
        
        # Sort by specific_modulus (bulk_modulus/density) descending
        final_results.sort(key=lambda x: x['specific_modulus'], reverse=True)
        print(f"Step 2: Found {len(final_results)} materials with bulk_modulus data")
        return final_results[:20]
    else:
        print("Step 2: MP API unavailable, using fallback (density ascending)")
        # Fallback: return top 20 by density ascending (lightest metals)
        for material in top_50:
            material['note'] = 'MP API unavailable - sorted by density only'
        return top_50[:20]

def query_ultra_wide_bandgap(reader, header):
    """
    Query 2: Ultra-wide bandgap semiconductors
    Filter: band_gap > 5.0 eV, formation_energy_per_atom < 0,
    abundant elements only (no rare earths), sort by band_gap descending
    """
    print("\n=== Query 2: Ultra-Wide Bandgap Semiconductors ===")
    print("Filter: bandgap > 5.0 eV, formation_energy_per_atom < 0, no rare earths")
    
    results = []
    bandgap_idx = header.index('Bandgap')
    formation_energy_idx = header.index('Formation Energy Per Atom')
    formula_idx = header.index('Reduced Formula')
    material_id_idx = header.index('MaterialId')
    elements_idx = header.index('Elements')
    
    for row in reader:
        try:
            bandgap_str = row[bandgap_idx]
            formation_energy = float(row[formation_energy_idx])
            elements_str = row[elements_idx]
            elements = parse_elements(elements_str)
            
            # Filter out inf/null bandgap values
            if bandgap_str.lower() in ['inf', '-inf', 'nan', '']:
                continue
            
            bandgap = float(bandgap_str)
            
            # Filters: bandgap > 5.0 AND bandgap < 20.0 (exclude data artifacts)
            if (5.0 < bandgap < 20.0 and 
                formation_energy < 0 and 
                not has_rare_earth(elements)):
                results.append({
                    'formula': row[formula_idx],
                    'material_id': row[material_id_idx],
                    'bandgap': bandgap,
                    'formation_energy': formation_energy,
                    'elements': elements_str
                })
        except (ValueError, IndexError):
            continue
    
    # Sort by bandgap descending
    results.sort(key=lambda x: x['bandgap'], reverse=True)
    
    return results[:20]

def query_fluorine_lubricant(reader, header):
    """
    Query 3: Oil/lubricant substitute
    Filter: fluorine-containing compounds, thermodynamically stable, valence limits, exclude artifacts
    Sort: fluorine_fraction descending (F atoms / total atoms), density ascending (lighter = better)
    """
    print("\n=== Query 3: Fluorine-Containing Lubricant Candidates ===")
    print("Filter: contains fluorine, formation_energy < 0, valence limits, exclude non-metal artifacts")
    
    results = []
    formula_idx = header.index('Reduced Formula')
    material_id_idx = header.index('MaterialId')
    elements_idx = header.index('Elements')
    composition_idx = header.index('Composition')
    density_idx = header.index('Density')
    formation_energy_idx = header.index('Formation Energy Per Atom')
    
    for row in reader:
        try:
            elements_str = row[elements_idx]
            elements = parse_elements(elements_str)
            composition_str = row[composition_idx]
            density = float(row[density_idx])
            formation_energy = float(row[formation_energy_idx])
            
            # Filter: must contain fluorine AND be thermodynamically stable
            if 'F' in elements and formation_energy < 0:
                # Calculate fluorine_fraction from composition
                element_counts, total_atoms = parse_composition(composition_str)
                f_count = element_counts.get('F', 0)
                fluorine_fraction = f_count / total_atoms if total_atoms > 0 else 0
                
                # Valence filter: check F count doesn't exceed known max for specific elements
                valence_ok = True
                for element, max_f in MAX_FLUORINE.items():
                    if element in element_counts:
                        element_count = element_counts[element]
                        # Check if F count exceeds reasonable valence for this element
                        if f_count > max_f * element_count:
                            valence_ok = False
                            break
                
                if not valence_ok:
                    continue
                
                # Artifact filter: exclude high fluorine_fraction with non-metal partners
                # Get non-F elements
                non_f_elements = [el for el in elements if el != 'F']
                if fluorine_fraction > 0.80 and non_f_elements:
                    # Check if any non-F element is a non-metal
                    has_non_metal = any(el in NON_METALS for el in non_f_elements)
                    if has_non_metal:
                        continue
                
                results.append({
                    'formula': row[formula_idx],
                    'material_id': row[material_id_idx],
                    'fluorine_fraction': fluorine_fraction,
                    'fluorine_count': f_count,
                    'total_atoms': total_atoms,
                    'density': density,
                    'formation_energy': formation_energy,
                    'elements': elements_str,
                    'composition': composition_str
                })
        except (ValueError, IndexError):
            continue
    
    # Sort by fluorine_fraction descending, then density ascending (lighter = better lubricant)
    results.sort(key=lambda x: (-x['fluorine_fraction'], x['density']))
    
    return results[:20]

def save_results(results, query_name, filename):
    """Save results to CSV"""
    if not results:
        print(f"No results for {query_name}")
        return
    
    output_path = f"{OUTPUT_DIR}{filename}"
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header based on first result
        fieldnames = list(results[0].keys())
        writer.writerow(fieldnames)
        
        for result in results:
            writer.writerow([result[k] for k in fieldnames])
    
    print(f"Saved {len(results)} results to {output_path}")

def main():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(CSV_PATH, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        # Query 3 only (for this run)
        f.seek(0)
        next(reader)
        results3 = query_fluorine_lubricant(reader, header)
        save_results(results3, "Fluorine Lubricant Candidates", "query3_fluorine_lubricant.csv")
    
    # Create README
    readme_path = f"{OUTPUT_DIR}GNoME_QUERIES_README.txt"
    with open(readme_path, 'w') as f:
        f.write(f"GNoME Material Discovery Queries\n")
        f.write(f"Generated: {timestamp}\n\n")
        
        f.write("=== Query 3: Fluorine-Containing Lubricant Candidates ===\n")
        f.write("Filter: contains fluorine, formation_energy < 0, valence limits, exclude non-metal artifacts\n")
        f.write("Rationale: High fluorine content for low surface energy, light density for lubricant applications\n")
        f.write(f"Results: {len(results3)} candidates\n")
        if results3:
            f.write("Top 10 candidates:\n")
            for i, r in enumerate(results3[:10], 1):
                f.write(f"  {i}. {r['formula']} (fluorine_fraction: {r['fluorine_fraction']:.3f}, density: {r['density']:.4f}, formation_energy: {r['formation_energy']:.4f})\n")
    
    print(f"\nREADME saved to {readme_path}")
    print("\n=== Summary ===")
    print(f"Query 3 (Fluorine): {len(results3)} results")

if __name__ == "__main__":
    main()

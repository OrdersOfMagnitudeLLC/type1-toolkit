# Copyright (c) 2026 Orders of Magnitude LLC
# Licensed under the OOM Commercial License v1.0
# See LICENSE.md in the repository root or ofmagnitude.com

#!/usr/bin/env python3
"""
Find GNoME materials similar to SiHF3 and PH(OF)2 in electronic properties
but structurally stable at room temperature.
"""

import pandas as pd
import os

# Rare elements to exclude
RARE_ELEMENTS = ['Ge', 'In', 'Ga', 'Te', 'Se', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 
                 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                 'Sc', 'Y', 'Pt', 'Pd', 'Rh', 'Ru', 'Ir', 'Os']

# Allowed stabilizing elements for STEP 6
STABILIZING_ELEMENTS = ['Na', 'K', 'Ca', 'Sr', 'Mg', 'Ba', 'Al']

def contains_element(formula, element):
    """Check if formula contains a specific element."""
    return element in formula

def contains_any_rare(formula):
    """Check if formula contains any rare elements."""
    for rare in RARE_ELEMENTS:
        if rare in formula:
            return True
    return False

def contains_only_allowed_plus_stabilizing(formula, base_elements):
    """
    Check if formula contains only base elements plus allowed stabilizing elements.
    base_elements: list of required elements (e.g., ['Si', 'F'] or ['P', 'F'])
    """
    # Extract all element symbols from formula (simple approach)
    # This is a simplified check - assumes elements are single or double letters
    import re
    elements_found = set(re.findall(r'[A-Z][a-z]?', formula))
    
    # All elements must be either in base_elements or in STABILIZING_ELEMENTS
    allowed = set(base_elements + STABILIZING_ELEMENTS)
    return elements_found.issubset(allowed)

def main():
    # STEP 1: Load GNoME CSV
    csv_path = os.path.join(os.path.expanduser("~"), 'NS/Bob/data/gnome.csv')
    print(f"Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print("\n" + "="*80)
    print("COLUMN NAMES:")
    print("="*80)
    for i, col in enumerate(df.columns):
        print(f"{i:3d}. {col}")
    print("="*80 + "\n")
    
    # Check required columns exist
    required_cols = ['MaterialId', 'Reduced Formula', 'Bandgap', 
                     'Formation Energy Per Atom', 'Density']
    for col in required_cols:
        if col not in df.columns:
            print(f"ERROR: Required column '{col}' not found in CSV")
            print("Available columns:", list(df.columns))
            return
    
    # Apply common stability filters
    print("Applying stability filters:")
    print("  - bandgap > 5.0 eV")
    print("  - formation_energy_per_atom < -0.1 eV")
    print("  - density < 3.0 g/cm³")
    print("  - Excluding rare elements\n")
    
    stable_mask = (
        (df['Bandgap'] > 5.0) &
        (df['Formation Energy Per Atom'] < -0.1) &
        (df['Density'] < 3.0)
    )
    
    # Filter out rare elements
    no_rare_mask = ~df['Reduced Formula'].apply(contains_any_rare)
    
    df_stable = df[stable_mask & no_rare_mask].copy()
    print(f"Materials after stability filters: {len(df_stable)}\n")
    
    # STEP 2: SiHF3 analog search (Si AND F)
    print("="*80)
    print("STEP 2: SiHF3 ANALOGS (Si + F)")
    print("="*80)
    
    si_f_mask = (
        df_stable['Reduced Formula'].apply(lambda x: contains_element(x, 'Si')) &
        df_stable['Reduced Formula'].apply(lambda x: contains_element(x, 'F'))
    )
    si_f_results = df_stable[si_f_mask].copy()
    si_f_results = si_f_results.sort_values('Bandgap', ascending=False)
    
    print(f"Found {len(si_f_results)} materials with Si + F\n")
    print("TOP 20 RESULTS:")
    print("-"*80)
    for i, row in si_f_results.head(20).iterrows():
        print(f"{row['MaterialId']:<20} | {row['Reduced Formula']:<20} | "
              f"bandgap: {row['Bandgap']:6.2f} eV | "
              f"density: {row['Density']:5.2f} g/cm³ | "
              f"formation: {row['Formation Energy Per Atom']:7.3f} eV/atom")
    print("-"*80 + "\n")
    
    # Save SiF results
    output_sif = os.path.join(os.path.expanduser("~"), 'NS/Bob/findings/analog_search_SiF.csv')
    si_f_results.to_csv(output_sif, index=False)
    print(f"Saved {len(si_f_results)} results to {output_sif}\n")
    
    # STEP 3: PH(OF)2 analog search (P AND F)
    print("="*80)
    print("STEP 3: PH(OF)2 ANALOGS (P + F)")
    print("="*80)
    
    p_f_mask = (
        df_stable['Reduced Formula'].apply(lambda x: contains_element(x, 'P')) &
        df_stable['Reduced Formula'].apply(lambda x: contains_element(x, 'F'))
    )
    p_f_results = df_stable[p_f_mask].copy()
    p_f_results = p_f_results.sort_values('Bandgap', ascending=False)
    
    print(f"Found {len(p_f_results)} materials with P + F\n")
    print("TOP 20 RESULTS:")
    print("-"*80)
    for i, row in p_f_results.head(20).iterrows():
        print(f"{row['MaterialId']:<20} | {row['Reduced Formula']:<20} | "
              f"bandgap: {row['Bandgap']:6.2f} eV | "
              f"density: {row['Density']:5.2f} g/cm³ | "
              f"formation: {row['Formation Energy Per Atom']:7.3f} eV/atom")
    print("-"*80 + "\n")
    
    # Save PF results
    output_pf = os.path.join(os.path.expanduser("~"), 'NS/Bob/findings/analog_search_PF.csv')
    p_f_results.to_csv(output_pf, index=False)
    print(f"Saved {len(p_f_results)} results to {output_pf}\n")
    
    # STEP 4: Overlap search (Si + P + F)
    print("="*80)
    print("STEP 4: OVERLAP SEARCH (Si + P + F)")
    print("="*80)
    
    si_p_f_mask = (
        df_stable['Reduced Formula'].apply(lambda x: contains_element(x, 'Si')) &
        df_stable['Reduced Formula'].apply(lambda x: contains_element(x, 'P')) &
        df_stable['Reduced Formula'].apply(lambda x: contains_element(x, 'F'))
    )
    si_p_f_results = df_stable[si_p_f_mask].copy()
    si_p_f_results = si_p_f_results.sort_values('Bandgap', ascending=False)
    
    print(f"Found {len(si_p_f_results)} materials with Si + P + F\n")
    if len(si_p_f_results) > 0:
        print("ALL RESULTS:")
        print("-"*80)
        for i, row in si_p_f_results.iterrows():
            print(f"{row['MaterialId']:<20} | {row['Reduced Formula']:<20} | "
                  f"bandgap: {row['Bandgap']:6.2f} eV | "
                  f"density: {row['Density']:5.2f} g/cm³ | "
                  f"formation: {row['Formation Energy Per Atom']:7.3f} eV/atom")
        print("-"*80 + "\n")
    else:
        print("No materials found with Si + P + F combination\n")
    
    # Save SiPF results
    output_sipf = os.path.join(os.path.expanduser("~"), 'NS/Bob/findings/analog_search_SiPF.csv')
    si_p_f_results.to_csv(output_sipf, index=False)
    print(f"Saved {len(si_p_f_results)} results to {output_sipf}\n")
    
    # STEP 5: Summary with top 5 from each
    print("="*80)
    print("SUMMARY - TOP 5 FROM EACH SEARCH")
    print("="*80)
    
    print("\nTOP 5 SiF ANALOGS:")
    print("-"*80)
    for i, row in si_f_results.head(5).iterrows():
        print(f"  {row['MaterialId']:<20} | {row['Reduced Formula']:<20} | "
              f"bandgap: {row['Bandgap']:6.2f} eV")
    
    print("\nTOP 5 PF ANALOGS:")
    print("-"*80)
    for i, row in p_f_results.head(5).iterrows():
        print(f"  {row['MaterialId']:<20} | {row['Reduced Formula']:<20} | "
              f"bandgap: {row['Bandgap']:6.2f} eV")
    
    if len(si_p_f_results) > 0:
        print("\nALL SiPF ANALOGS:")
        print("-"*80)
        for i, row in si_p_f_results.iterrows():
            print(f"  {row['MaterialId']:<20} | {row['Reduced Formula']:<20} | "
                  f"bandgap: {row['Bandgap']:6.2f} eV")
    else:
        print("\nNO SiPF ANALOGS FOUND")
    
    # STEP 6: Stabilized analog search
    print("="*80)
    print("STEP 6: STABILIZED ANALOG SEARCH")
    print("="*80)
    print("Filters:")
    print("  - bandgap > 5.0 eV")
    print("  - formation_energy_per_atom < -0.1 eV")
    print("  - density < 4.0 g/cm³ (relaxed)")
    print("  - Excluding rare elements")
    print("  - Allowing stabilizing elements: Na, K, Ca, Sr, Mg, Ba, Al")
    print("  - Must contain Si+F OR P+F\n")
    
    # Apply relaxed stability filters
    stable_mask_relaxed = (
        (df['Bandgap'] > 5.0) &
        (df['Formation Energy Per Atom'] < -0.1) &
        (df['Density'] < 4.0)
    )
    
    df_stable_relaxed = df[stable_mask_relaxed & no_rare_mask].copy()
    print(f"Materials after relaxed stability filters: {len(df_stable_relaxed)}\n")
    
    # Filter for Si+F with allowed stabilizing elements
    si_f_stabilized_mask = (
        df_stable_relaxed['Reduced Formula'].apply(lambda x: contains_element(x, 'Si')) &
        df_stable_relaxed['Reduced Formula'].apply(lambda x: contains_element(x, 'F')) &
        df_stable_relaxed['Reduced Formula'].apply(lambda x: contains_only_allowed_plus_stabilizing(x, ['Si', 'F']))
    )
    
    # Filter for P+F with allowed stabilizing elements
    p_f_stabilized_mask = (
        df_stable_relaxed['Reduced Formula'].apply(lambda x: contains_element(x, 'P')) &
        df_stable_relaxed['Reduced Formula'].apply(lambda x: contains_element(x, 'F')) &
        df_stable_relaxed['Reduced Formula'].apply(lambda x: contains_only_allowed_plus_stabilizing(x, ['P', 'F']))
    )
    
    # Combine both searches
    stabilized_mask = si_f_stabilized_mask | p_f_stabilized_mask
    stabilized_results = df_stable_relaxed[stabilized_mask].copy()
    stabilized_results = stabilized_results.sort_values('Bandgap', ascending=False)
    
    print(f"Found {len(stabilized_results)} stabilized analogs (Si+F or P+F with stabilizers)\n")
    print("TOP 20 RESULTS:")
    print("-"*80)
    for i, row in stabilized_results.head(20).iterrows():
        print(f"{row['MaterialId']:<20} | {row['Reduced Formula']:<20} | "
              f"bandgap: {row['Bandgap']:6.2f} eV | "
              f"density: {row['Density']:5.2f} g/cm³ | "
              f"formation: {row['Formation Energy Per Atom']:7.3f} eV/atom")
    print("-"*80 + "\n")
    
    # Save stabilized results
    output_stabilized = os.path.join(os.path.expanduser("~"), 'NS/Bob/findings/analog_search_stabilized.csv')
    stabilized_results.to_csv(output_stabilized, index=False)
    print(f"Saved {len(stabilized_results)} results to {output_stabilized}\n")
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()

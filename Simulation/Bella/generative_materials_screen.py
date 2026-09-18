# Copyright (c) 2026 Orders of Magnitude LLC
# Licensed under the OOM Commercial License v1.0
# See LICENSE.md in the repository root or ofmagnitude.com

#!/usr/bin/env python3
"""Generative earth-abundant materials screen across civilization-critical categories."""
from foam_screener_v2 import phonon_stability_formula
from itertools import combinations_with_replacement
import random

# Earth-abundant elements only (top 30 by crustal abundance)
EARTH_ABUNDANT = [
    'Fe','Al','Ca','Mg','Na','K','Ti','Mn','Si','P',
    'Cr','Ni','Zn','Cu','Co','V','W','Mo','Sn','Pb',
    'Zr','Ba','Sr','B','N','C','O','S','F','Cl'
]

# Problem categories with target Debye range
CATEGORIES = {
    "nitrogen_fixation":       {"elements": ["Fe","Mn","Mo","W","V","Cr","N"], "debye_min": 300},
    "solar_absorber":          {"elements": ["Cu","Zn","Sn","S","Se","Si","Ge"], "debye_min": 200},
    "battery_anode":           {"elements": ["Si","Sn","Sb","P","Al","Mn","Fe"], "debye_min": 250},
    "solid_state_electrolyte": {"elements": ["Na","K","P","S","O","F","Al","Si"], "debye_min": 300},
    "co2_capture":             {"elements": ["Ca","Mg","Al","Fe","K","Na","O"], "debye_min": 200},
    "hydrogen_catalyst":       {"elements": ["Ni","Fe","Co","Mo","W","Mn","Cu"], "debye_min": 350},
    "piezoelectric":           {"elements": ["Ba","Ti","K","Na","Bi","Fe","Nb"], "debye_min": 300},
    "permanent_magnet":        {"elements": ["Fe","Co","Ni","Mn","Al","N","B"], "debye_min": 400},
    "transparent_conductor":   {"elements": ["Zn","Sn","Al","Ga","In","Si","O"], "debye_min": 300},
    "thermal_storage":         {"elements": ["Na","K","Ca","Mg","Al","S","Cl"], "debye_min": 200},
    "corrosion_protection":    {"elements": ["Cr","Al","Si","Ti","Zr","Mo","N"], "debye_min": 400},
    "low_carbon_cement":       {"elements": ["Ca","Si","Al","Mg","Fe","S","O"], "debye_min": 200},
    "superconductor":          {"elements": ["Cu","Ba","Ca","Fe","As","Se","B"], "debye_min": 200},
    "flame_retardant":         {"elements": ["Al","Mg","Si","P","N","B","Zn"], "debye_min": 250},
    "hard_coating":            {"elements": ["Ti","Al","Si","Cr","W","N","B","C"], "debye_min": 500},
}


def generate_formulas(elements, n=500):
    """Generate random stoichiometries from element pool."""
    formulas = []
    for _ in range(n):
        # Pick 2-4 elements
        k = random.randint(2, 4)
        chosen = random.sample(elements, min(k, len(elements)))
        # Random stoichiometry 1-4 each
        counts = [random.randint(1, 4) for _ in chosen]
        formula = "".join(f"{el}{c}" if c > 1 else el
                          for el, c in zip(chosen, counts))
        formulas.append(formula)
    return list(set(formulas))  # deduplicate


def main():
    results = {}
    total_screened = 0
    total_stable = 0

    for cat, params in CATEGORIES.items():
        formulas = generate_formulas(params["elements"], n=500)
        total_screened += len(formulas)
        survivors = []
        for formula in formulas:
            try:
                result = phonon_stability_formula(formula)
                if (result.get("stable") and
                        result.get("theta_D_K", 0) >= params["debye_min"]):
                    survivors.append({
                        "formula": formula,
                        "debye_T": round(result.get("theta_D_K", 0), 1),
                        "source": result.get("debye_method", "")
                    })
            except Exception:
                continue
        # Sort by Debye temp, take top 5
        survivors.sort(key=lambda x: x["debye_T"], reverse=True)
        results[cat] = survivors[:5]
        total_stable += len(survivors)
        print(f"{cat}: {len(survivors)}/{len(formulas)} stable")

    # Write results
    with open("findings/materials_civilization_screen.txt", "w") as f:
        f.write("CIVILIZATION MATERIALS SCREEN\n")
        f.write(f"Total screened: {total_screened} | Stable: {total_stable}\n\n")
        for cat, hits in results.items():
            f.write(f"\n=== {cat.upper()} ===\n")
            if hits:
                for h in hits:
                    f.write(f"  {h['formula']:20s}  Debye={h['debye_T']:.0f}K\n")
            else:
                f.write("  No stable candidates found\n")

    print(open("findings/materials_civilization_screen.txt").read())


if __name__ == "__main__":
    main()

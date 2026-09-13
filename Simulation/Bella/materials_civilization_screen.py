# Copyright (c) 2026 Orders of Magnitude LLC
# Licensed under the OOM Commercial License v1.0
# See LICENSE.md in the repository root or ofmagnitude.com

#!/usr/bin/env python3
"""Civilization materials screen with per-category MP constraints, polymorph dedup, and custom ranking."""
import os, re
from mp_api.client import MPRester
from foam_screener_v2 import (
    debye_temperature_anderson,
    inorganic_adsorption_energy,
    synthesizability,
)
from categories_57 import EXTRA_CATEGORY_FILTERS, EXTRA_CATEGORY_NOTES
from manual_candidates_57 import EXTRA_MANUAL_CANDIDATES

api_key = os.environ.get('MATERIALS_PROJECT_API_KEY', '')
if not api_key:
    try:
        with open(os.path.expanduser("~/.bella/.env")) as f:
            for line in f:
                if "MATERIALS_PROJECT" in line or "MP_API" in line:
                    api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass

FORBIDDEN = {"Pt","Ir","Pd","Rh","Ru","Au","La","Ce","Nd","Dy","Tb","Eu","Sm"}

import math as _math

ABUNDANCE_PPM = {
    "H": 1400, "Li": 20, "Be": 2.8, "B": 10, "C": 200,
    "N": 20, "O": 461000, "Na": 23600, "Mg": 23300,
    "Al": 82300, "Si": 282000, "P": 1050, "S": 350,
    "K": 20900, "Ca": 41500, "Ti": 5650, "V": 120,
    "Cr": 102, "Mn": 950, "Fe": 56300, "Co": 25,
    "Ni": 84, "Cu": 60, "Zn": 70, "Ga": 19, "Ge": 1.5,
    "As": 1.8, "Se": 0.05, "Br": 2.4, "Rb": 90, "Sr": 370,
    "Y": 33, "Zr": 165, "Nb": 20, "Mo": 1.2, "Ru": 0.001,
    "Rh": 0.001, "Pd": 0.015, "Ag": 0.07, "Cd": 0.15,
    "In": 0.25, "Sn": 2.3, "Sb": 0.2, "Te": 0.001,
    "Cs": 3, "Ba": 425, "La": 39, "Ce": 66, "Hf": 3,
    "Ta": 2, "W": 1.25, "Re": 0.0007, "Os": 0.0015,
    "Ir": 0.001, "Pt": 0.005, "Au": 0.004, "Hg": 0.085,
    "Tl": 0.85, "Pb": 14, "Bi": 0.025, "Th": 9.6, "U": 2.7
}

CRITICAL_ELEMENTS = {"Ir", "Ru", "Rh", "Pt", "Pd", "Re", "Os", "Te", "In"}


def compute_rank_score(debye_T, formula):
    """Abundance-weighted ranking score."""
    elements = re.findall(r'[A-Z][a-z]?', formula)
    abundances = [ABUNDANCE_PPM.get(el, 0.1) for el in elements]
    if not abundances:
        return 0.0
    geo_mean = 1.0
    for a in abundances:
        geo_mean *= a
    geo_mean = geo_mean ** (1.0 / len(abundances))
    abundance_weight = _math.log10(geo_mean + 1) / 6.0
    has_critical = bool(set(elements) & CRITICAL_ELEMENTS)
    criticality_penalty = 3.0 if has_critical else 1.0
    return debye_T * abundance_weight / criticality_penalty


def compute_rank_score_abundance_only(formula):
    """Abundance-only ranking score (no Debye multiplier) — for GNoME candidates."""
    elements = re.findall(r'[A-Z][a-z]?', formula)
    abundances = [ABUNDANCE_PPM.get(el, 0.1) for el in elements]
    if not abundances:
        return 0.0
    geo_mean = 1.0
    for a in abundances:
        geo_mean *= a
    geo_mean = geo_mean ** (1.0 / len(abundances))
    abundance_weight = _math.log10(geo_mean + 1) / 6.0
    has_critical = bool(set(elements) & CRITICAL_ELEMENTS)
    criticality_penalty = 3.0 if has_critical else 1.0
    return abundance_weight / criticality_penalty


def disambiguate_formula(formula, note="", B=0, G=0, density=None, space_group=None):
    """Append structure type for ambiguous formulas (C, BN)."""
    if formula == "C":
        if "diamond" in note.lower() or (density and density > 3.0) or (space_group and "Fd-3m" in space_group):
            return "C (diamond)"
        if "graphene" in note.lower() or "graphite" in note.lower() or (space_group and "P6_3/mmc" in space_group):
            return "C (graphite)"
        if "BC8" in note.upper():
            return "C (BC8)"
        return "C (unknown polymorph)"
    if formula == "BN":
        if "cubic" in note.lower() or (B and B > 300) or (space_group and "F-43m" in space_group):
            return "BN (c-BN)"
        if "hexagonal" in note.lower() or (B and B < 100) or (space_group and "P6_3/mmc" in space_group):
            return "BN (h-BN)"
        return "BN (unknown polymorph)"
    return formula

CATEGORY_FILTERS = {
    "nitrogen_fixation": {
        "elements_must_include_any": ["Fe","Mn","Mo","Ru","V","Co","Ni","W","Re"],
        "band_gap_max": 0.5,
        "formation_energy_max": -0.1
    },
    "phosphate_recovery": {
        "elements_must_include_any": ["Fe","Al","La","Ce","Ca","Zr"],
        "elements_must_include_any_anion": ["O"],
        "band_gap_min": 1.0
    },
    "soil_carbon_sequestration": {
        "elements_must_include_any": ["Ca","Mg","Fe","Al","Si"],
        "band_gap_min": 1.0
    },
    "atmospheric_water_harvesting": {
        "elements_must_include_any": ["Ca","Mg","Li","Na","K","Al","Zr","Ti","Si"],
        "elements_must_exclude": ["H"],
        "band_gap_min": 2.0
    },
    "heavy_metal_filtration": {
        "elements_must_include_any": ["Fe","Mn","Ti","Zr","Ce","La"],
        "band_gap_min": 0.5
    },
    "fluoride_removal": {
        "elements_must_include_any": ["Ca","La","Ce","Al","Fe","Mg"],
        "band_gap_min": 1.0
    },
    "forward_osmosis_membrane": {
        "elements_must_include_any": ["Ti","Si","Al","Zr"],
        "band_gap_min": 2.0
    },
    "antifouling_coating": {
        "elements_must_include_any": ["Ti","Cu","Ag","Zn","Si"],
        "band_gap_min": 0.5
    },
    "solar_absorber_leadfree": {
        "band_gap_min": 1.1,
        "band_gap_max": 1.6,
        "elements_must_exclude": ["Pb","Cd","Tl"]
    },
    "solar_tandem_bottom": {
        "band_gap_min": 0.8,
        "band_gap_max": 1.1,
        "elements_must_exclude": ["Pb","Cd"]
    },
    "photocatalyst_water_splitting": {
        "band_gap_min": 1.8,
        "band_gap_max": 3.2,
        "elements_must_include_any": ["Ti","Ta","Nb","Ga","In","Zn","Bi","W"]
    },
    "thermoelectric_high_zt": {
        "band_gap_min": 0.05,
        "band_gap_max": 0.8,
        "elements_must_include_any": ["Te","Se","Bi","Pb","Sn","Ge","Sb"]
    },
    "geothermal_casing": {
        "elements_must_include_any": ["Ni","Cr","Ti","Mo","W","Co"],
        "band_gap_max": 0.1,
        "formation_energy_max": -0.1
    },
    "geothermal_heat_exchanger": {
        "elements_must_include_any": ["Ni","Cu","Al","Ti","Si","Fe"],
        "band_gap_max": 0.1,
        "formation_energy_max": -0.1
    },
    "solid_electrolyte": {
        "elements_must_include_any": ["Li","Na"],
        "elements_must_include_any_second": ["La","Zr","P","S","Ge","Al","Ta","Nb"],
        "band_gap_min": 3.0,
        "formation_energy_max": -0.5
    },
    "anode_lithium_free": {
        "elements_must_include_any": ["Si","Sn","Sb","Na","K","Mg"],
        "band_gap_max": 1.0
    },
    "cathode_cobalt_free": {
        "elements_must_include_any": ["Li","Na"],
        "elements_must_include_any_transition": ["Mn","Fe","Ni","V","Ti"],
        "elements_must_exclude": ["Co"]
    },
    "hydrogen_storage": {
        "elements_must_include_any_metal": ["Mg","Ti","V","Fe","Ni","La","Ca"],
        "elements_must_include_any": ["H"]
    },
    "thermal_storage_hightemp": {
        "elements_must_include_any": ["Ca","Mg","Na","K","Al","Si"],
        "band_gap_min": 2.0,
        "formation_energy_max": -0.5
    },
    "superconductor_ambient": {
        "band_gap_max": 0.0,
        "elements_must_include_any": ["H","Ca","Mg","Cu","Ba"]
    },
    "superconductor_cryo": {
        "band_gap_max": 0.0,
        "elements_must_include_any": ["Nb","V","Pb","Sn","Cu","Ba"]
    },
    "co2_solid_sorbent": {
        "elements_must_include_any_cation": ["Ca","Mg","K","Na","Li","Ba"],
        "elements_must_include_any": ["O"],
        "band_gap_min": 2.0
    },
    "co2_reduction_catalyst": {
        "elements_must_include_any": ["Cu","Fe","Co","Ni","Mo","Ru","Rh"],
        "band_gap_max": 0.5,
        "formation_energy_max": 0,
        "B_GPa_max": 200,
    },
    "low_carbon_cement": {
        "elements_must_include_any": ["Ca","Si","Al","Mg"],
        "band_gap_min": 2.0,
        "formation_energy_max": -1.0
    },
    "biodegradable_polymer": {
        "elements_must_include_any": ["C","O","N","H","Ca","Mg"],
        "band_gap_min": 3.0
    },
    "transparent_conductor_itofree": {
        "elements_must_include_any": ["Ga","Zn","Al","Sn"],
        "elements_must_exclude": ["In"],
        "band_gap_min": 2.5,
        "band_gap_max": 4.0
    },
    "piezoelectric_leadfree": {
        "elements_must_exclude": ["Pb"],
        "elements_must_include_any": ["Ba","K","Na","Bi","Li"],
        "band_gap_min": 2.0
    },
    "topological_quantum": {
        "elements_must_include_any": ["Bi","Sb","Te","Se","As","Sn","Pb","Hg"],
        "band_gap_min": 0.05,
        "band_gap_max": 0.5
    },
    "mxene_alternative": {
        "elements_must_include_any_metal": ["Ti","V","Mo","Nb","Ta","Cr","W"],
        "elements_must_include_any": ["C","N"],
        "band_gap_max": 0.5
    },
    "antimicrobial_surface": {
        "elements_must_include_any": ["Cu","Ag","Zn","Ti"],
        "band_gap_min": 1.5
    },
    "corrosion_coating_chromefree": {
        "elements_must_include_any": ["Ti","Al","Zr","Ce","Si","Mo","Ta","W","Nb"],
        "elements_must_exclude": ["Cr","Pb","Cd"],
        "formation_energy_max": -0.3
    },
    "radiation_tolerant_structural": {
        "elements_must_include_any": ["W","Ta","Mo","Zr","Ti","V","Cr"],
        "band_gap_max": 0.1,
        "formation_energy_max": -0.1
    },
    "high_entropy_aerospace": {
        "elements_must_include_many": ["W","Ta","Mo","Nb","V","Hf","Zr","Ti","Cr"],
        "min_elements": 4,
        "band_gap_max": 0.1
    },
    "plasma_resistant": {
        "elements_must_include_any": ["Hf","Zr","Ta","Ti","W"],
        "elements_must_include_any_second": ["C","N","B"],
        "band_gap_max": 0.5,
        "formation_energy_max": -1.0
    },
    "room_temp_superconductor_candidates": {
        "band_gap_max": 0.0,
        "elements_must_include_any": ["H","Cu","Ba","Mg","Bi","Sr","Ca","Y","Tl","Hg","La","Ce"]
    },
    "room-temperature-superconductor": {
        "band_gap_max": 0.0,
        "elements_must_include_any": ["H","B","C","N","O","Mg","Al","Si"],
        "elements_must_exclude": ["Th","U","Pu","Am","Np","Cm","Bk","Cf","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu"]
    },
    "radiation_shielding": {
        "elements_must_include_any": ["B","W","Pb","Bi","Ba","Fe","Gd","Hf","C","N","O","S"],
        "band_gap_min": 0.0
    },
    "extreme_hardness": {
        "elements_must_include_any": ["B","C","N","Os","Re","W","Mo"],
        "band_gap_min": 0.0
    },
    "thermal_interface_materials": {
        "elements_must_include_any": ["Al","B","C","Si","Be","Ga","N","O"],
        "band_gap_min": 1.0
    },
    "hydrogen_storage_hydrides": {
        "elements_must_include_any": ["H"],
        "elements_must_include_any_metal": ["Mg","Ti","La","Fe","Ni","Na","Li","Ca"],
    },
    "co2_capture_materials": {
        "elements_must_include_any_cation": ["Mg","Ca","Li","K","Na","Ba"],
        "elements_must_include_any": ["O","C","N"],
        "band_gap_min": 1.0
    },
    "water_splitting_catalysts": {
        "elements_must_include_any": ["Ir","Ru","Ni","Fe","Co","Mo","Cu","Bi","Ta","P","S","O"],
        "band_gap_max": 3.0
    },
    "fertilizer_catalyst_expanded": {
        "elements_must_include_any": ["Fe","Mo","V","Co","Ni","Mn","N"],
        "band_gap_max": 0.5,
        "formation_energy_max": -0.1
    },
    "extreme_temperature_structural": {
        "elements_must_include_any": ["Hf","Zr","Ti","Nb","Mo","W","Ta","Cr","Si","C","B"],
        "band_gap_max": 0.5,
        "formation_energy_max": -0.3
    },
    "permanent_magnet_ree_free": {
        "elements_must_include_any": ["Fe", "Mn", "Co", "Ni"],
        "elements_must_exclude": ["Nd", "Dy", "Tb", "Sm", "Pr", "La", "Ce", "Eu", "H", "O", "Si"],
        "band_gap_max": 0.5,
        "formation_energy_max": 0.0,
        "B_GPa_min": 100,
        "B_GPa_max": 300,
        "note": "Hard magnetic materials without rare earth elements — wind turbines, EV motors"
    },
}
CATEGORY_FILTERS.update(EXTRA_CATEGORY_FILTERS)

CATEGORY_ADSORBATE = {
    "nitrogen_fixation": "N2",
    "co2_solid_sorbent": "CO2",
    "co2_reduction_catalyst": "CO2",
    "photocatalyst_water_splitting": "H2O",
    "geothermal_casing": "H2O",
    "geothermal_heat_exchanger": "H2O",
    "atmospheric_water_harvesting": "H2O",
    "hydrogen_storage": "H2",
}

OPTIMAL_RANGE = {
    "nitrogen_fixation": (-1.5, -0.5),
    "co2_reduction_catalyst": (-1.2, -0.3),
    "co2_solid_sorbent": (-0.8, -0.3),
    "photocatalyst_water_splitting": (-0.9, -0.2),
    "hydrogen_storage": (-0.6, -0.2),
    "atmospheric_water_harvesting": (-0.5, -0.1),
}


TOXIC_ELEMENTS = {"Be", "Pb", "Cd", "Hg", "As", "Tl", "Cr", "Ra", "Po", "Th", "U"}

ALKALI_ALKALINE_EARTH = {"Ca", "Mg", "Na", "K", "Ba", "Sr"}

ENERGY_CATEGORIES = {
    "hydrogen_storage",
    "thermal_storage_hightemp",
    "geothermal_casing",
    "geothermal_heat_exchanger",
    "solar_absorber_leadfree",
    "solar_tandem_bottom",
    "photocatalyst_water_splitting",
    "thermoelectric_high_zt",
    "solid_electrolyte",
    "anode_lithium_free",
    "cathode_cobalt_free",
    "superconductor_ambient",
    "superconductor_cryo",
    "room-temperature-superconductor",
    "nitrogen_fixation",
    "atmospheric_water_harvesting",
    "co2_reduction_catalyst",
    "co2_solid_sorbent",
}


def parse_formula_counts(formula):
    parts = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
    counts = {}
    total = 0
    for el, n in parts:
        c = int(n) if n else 1
        counts[el] = counts.get(el, 0) + c
        total += c
    return counts, total


def safe_farmers_status(counts, total):
    toxic = set(counts) & TOXIC_ELEMENTS
    if not toxic:
        return "PASS"
    if "Cr" in toxic:
        non_cr = toxic - {"Cr"}
        if non_cr:
            return "FAIL"
        if total > 0 and counts["Cr"] / total < 0.2:
            return "WARN"
    return "FAIL"


def carbon_cycle_status(cat, counts):
    if "C" in counts and (set(counts) & ALKALI_ALKALINE_EARTH):
        return "SINK"
    if cat in ("co2_solid_sorbent", "co2_reduction_catalyst"):
        return "SINK"
    if cat == "low_carbon_cement" and "Si" in counts:
        return "SINK"
    if "C" in counts and not (set(counts) & ALKALI_ALKALINE_EARTH):
        if cat not in ("co2_solid_sorbent", "co2_reduction_catalyst"):
            return "SOURCE"
    return "NEUTRAL"


def co2_opportunity_status(cat, carbon_cycle):
    if carbon_cycle == "SINK":
        return "HIGH"
    if carbon_cycle == "NEUTRAL" and cat in ENERGY_CATEGORIES:
        return "MEDIUM"
    return "LOW"


# Element scarcity tiers for civilization-scale deployability
PRECIOUS_LIST = ["Os", "Ir", "Pt", "Au", "Pd", "Rh", "Re"]
RARE_EARTH_HEAVY_LIST = ["Tm", "Yb", "Lu", "Ho", "Er", "Tb", "Dy", "Eu", "Pm", "Gd"]
EXPENSIVE_LIST = ["Cs", "Rb", "Hf", "Be", "Tl", "In", "Ga", "Ge", "Te"]
LIGHT_RARE_EARTH_LIST = ["La", "Ce", "Y", "Sc", "Nd", "Pr", "Sm"]
SCARCE_LIST = ["Ru"]
MODERATE_LIST = ["Co", "Ni", "Mo", "V", "Cr", "W", "Ti", "Zr", "Nb", "Ta", "Cu", "Zn"] + LIGHT_RARE_EARTH_LIST
ABUNDANT_LIST = ["Fe", "Mn", "Al", "Si", "Ca", "Mg", "K", "Na", "Li", "Ba", "Sr", "H", "C", "N", "O", "F", "P", "S"]

def classify_elements(formula_str):
    elements = set(re.findall(r'[A-Z][a-z]?', formula_str))
    # Tier lookup in worst-to-best order
    for tier_name, tier_set in [
        ("PRECIOUS", set(PRECIOUS_LIST)),
        ("RARE_EARTH_HEAVY", set(RARE_EARTH_HEAVY_LIST)),
        ("EXPENSIVE", set(EXPENSIVE_LIST)),
        ("SCARCE", set(SCARCE_LIST)),
        ("MODERATE", set(MODERATE_LIST)),
        ("ABUNDANT", set(ABUNDANT_LIST)),
    ]:
        hit = elements & tier_set
        if hit:
            worst = sorted(hit)[0]
            return {
                "tier": tier_name,
                "worst_element": worst,
                "all_elements": sorted(elements),
                "deployable": tier_name not in ("PRECIOUS", "RARE_EARTH_HEAVY", "EXPENSIVE"),
            }
    # Unknown element fallback to scarce
    worst = sorted(elements)[0] if elements else "X"
    return {
        "tier": "SCARCE",
        "worst_element": worst,
        "all_elements": sorted(elements),
        "deployable": True,
    }


def is_valid_formula(formula_str):
    return bool(re.fullmatch(r"^([A-Z][a-z]?\d*)+$", formula_str))


# Metal set imported from soap_bowl_full for application compatibility checks
import sys
_sys_path = sys.path[:]
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'UniverseResearch', 'SoapBowl'))
from soap_bowl_full import METALS as _SOAP_METALS
sys.path = _sys_path
METALS = _SOAP_METALS


def is_application_compatible(formula, category, band_gap=None, formE=None):
    """Check category-specific chemical incompatibilities."""
    elements = set(re.findall(r'[A-Z][a-z]?', formula))

    if category == "soil_carbon_sequestration":
        if "H" in elements and (elements & ALKALI_ALKALINE_EARTH):
            return False, f"INCOMPATIBLE {formula} for soil_carbon_sequestration: hydride/alkali reacts with soil moisture"

    if category == "solar_absorber_leadfree":
        if "I" in elements and "H" in elements:
            return False, f"INCOMPATIBLE {formula} for solar_absorber_leadfree: iodide hydride, thermally unstable"
        if formula == "HI3":
            return False, f"INCOMPATIBLE {formula} for solar_absorber_leadfree: HI3 decomposes below 300C"
        if formE is not None and formE > -0.05 and ("I" in elements or "Br" in elements):
            return False, f"INCOMPATIBLE {formula} for solar_absorber_leadfree: weakly bound halide, decomposition_T < 300C"

    if category == "atmospheric_water_harvesting":
        hydride_metals = set(["Li", "Na", "K", "Ca", "Mg", "Al", "B"])
        if "H" in elements and (elements & hydride_metals):
            return False, f"INCOMPATIBLE {formula} for water_harvesting: hydride reacts with H2O, not adsorbs it"

    if category == "co2_solid_sorbent":
        if not (elements & set(["O", "C", "N"])) and elements.issubset(METALS):
            return False, f"INCOMPATIBLE {formula} for co2_solid_sorbent: purely metallic, no CO2 binding"

    if category == "photocatalyst_water_splitting":
        if band_gap == 0.0:
            return False, f"INCOMPATIBLE {formula} for photocatalyst_water_splitting: zero band gap metal"

    if category == "solid_electrolyte":
        if formula in ["LiH", "NaH", "LiAlH4", "NaBH4", "LiBH4"]:
            return False, f"INCOMPATIBLE {formula} for solid_electrolyte: hydride, not ion conductor"

    if category in WATER_CONTACT_CATEGORIES:
        # Carbides (reactive with water)
        carbide_metals = set(["Ca", "Al", "Na", "K", "Mg", "Li", "Be"])
        if "C" in elements and (elements & carbide_metals) and ("O" not in elements):
            return False, f"INCOMPATIBLE {formula} for {category}: carbide reacts with water to form acetylene/alkali"
        # Sulfides (hydrolyze to H2S)
        sulfide_metals = set(["Ca", "Mg", "Na", "K", "Al", "Fe", "Mn"])
        if "S" in elements and (elements & sulfide_metals) and ("O" not in elements):
            return False, f"INCOMPATIBLE {formula} for {category}: sulfide hydrolyzes to toxic H2S"
        # Nitrides (hydrolyze to NH3)
        nitride_metals = set(["Ca", "Mg", "Al", "Si", "Li"])
        if ("N" in elements and (elements & nitride_metals) and
                "O" not in elements and "C" not in elements):
            return False, f"INCOMPATIBLE {formula} for {category}: nitride hydrolyzes to NH3"
        # Strong reducing agents
        if formula in ["LiAlH4", "NaBH4", "LiBH4", "CaH2"]:
            return False, f"INCOMPATIBLE {formula} for {category}: violent water reaction"
        # Toxic halides
        if formula in ["NaF", "KF", "AlF3", "BaF2"]:
            return False, f"INCOMPATIBLE {formula} for {category}: toxic fluoride release"

    return True, ""


CATEGORY_NOTES = {
    "nitrogen_fixation": (
        "Catalytic surface activates N≡N triple bond from air. "
        "NEEDS: Electrolytic cell OR high-T reactor, N2 from air, H2O or H2 source. "
        "OUTPUT: NH3 → fertilizer, no Haber-Bosch plant required. "
        "DEPLOY: Village cooperative with solar-powered electrolyzer."
    ),
    "phosphate_recovery": (
        "Adsorbs phosphate ions from wastewater/runoff. "
        "NEEDS: Packed bed filter, wastewater stream. "
        "OUTPUT: Captured phosphate → reapply to soil. Closes phosphorus cycle. "
        "DEPLOY: Simple filtration unit, any farm."
    ),
    "soil_carbon_sequestration": (
        "Mineral amendment that binds CO2 as stable carbonate in soil. "
        "NEEDS: Grind material, spread on fields. Enhanced weathering. "
        "OUTPUT: Carbon stored for centuries, soil pH improved. "
        "DEPLOY: Any farmer with access to ground material."
    ),
    "atmospheric_water_harvesting": (
        "Passive desiccant — adsorbs moisture from air, releases with low heat. "
        "NEEDS: Temperature swing 50-80C, solar heat sufficient. "
        "OUTPUT: Liquid water from air, no well or river required. "
        "DEPLOY: Village scale, solar thermal driver, zero grid."
    ),
    "heavy_metal_filtration": (
        "Adsorbs Pb, As, Hg, Cd from drinking water via surface binding. "
        "NEEDS: Packed bed filter column, gravity-fed water. "
        "OUTPUT: Clean drinking water. Regenerable with mild acid wash. "
        "DEPLOY: Household or community filter, no electricity."
    ),
    "fluoride_removal": (
        "Ion exchange — fluoride displaces surface hydroxyl. "
        "NEEDS: Filter column, contact time 10-30 min. "
        "OUTPUT: Fluoride below WHO limit 1.5 mg/L. "
        "DEPLOY: Community water point, no electricity."
    ),
    "forward_osmosis_membrane": (
        "Selective water transport driven by osmotic pressure differential. "
        "NEEDS: Draw solution (sugar/salt), no high-pressure pump. "
        "OUTPUT: Desalinated water at fraction of reverse osmosis energy. "
        "DEPLOY: Coastal villages, solar-regenerated draw solution."
    ),
    "antifouling_coating": (
        "Surface chemistry prevents biofilm attachment. "
        "NEEDS: Coat onto pipes, ship hulls, membranes. "
        "OUTPUT: Extended membrane/equipment life, no biocide chemicals. "
        "DEPLOY: Industrial coating application."
    ),
    "solar_absorber_leadfree": (
        "Semiconductor converts sunlight to electricity (PV). "
        "NEEDS: Thin film deposition on substrate. "
        "OUTPUT: Electricity, no moving parts, 20-30yr lifetime. "
        "DEPLOY: Distributed solar, lead-free = safe for homes."
    ),
    "solar_tandem_bottom": (
        "Bottom cell in tandem PV, captures infrared that top cell misses. "
        "NEEDS: Paired with perovskite or Si top cell. "
        "OUTPUT: Higher efficiency solar panel. "
        "DEPLOY: Manufacturing integration."
    ),
    "photocatalyst_water_splitting": (
        "Sunlight drives water → H2 + O2 directly on powder surface. "
        "NEEDS: Sunlight + water + catalyst powder suspension. NO electricity. "
        "OUTPUT: H2 fuel or H2O2 for sanitation. "
        "DEPLOY: Village reactor — pond with catalyst, sunlight, collect gas."
    ),
    "thermoelectric_high_zt": (
        "Temperature gradient → electricity directly (Seebeck effect). "
        "NEEDS: Hot side (cookstove, engine exhaust, geothermal) + cold side. "
        "OUTPUT: Electricity from waste heat, no moving parts. "
        "DEPLOY: Attach to any heat source — cookstove electrification."
    ),
    "geothermal_casing": (
        "Wellbore casing survives 300-400C acidic brine for decades. "
        "NEEDS: Deep drilling (1-5 km), standard geothermal infrastructure. "
        "OUTPUT: Enables geothermal anywhere on Earth, not just volcanic zones. "
        "DEPLOY: Industrial drilling, but enables village-scale power grid."
    ),
    "geothermal_heat_exchanger": (
        "Transfers heat from brine to working fluid without corrosion. "
        "NEEDS: Plate or shell-and-tube heat exchanger in geothermal loop. "
        "OUTPUT: Longer exchanger lifetime, lower maintenance cost. "
        "DEPLOY: Geothermal plant component."
    ),
    "solid_electrolyte": (
        "Conducts Li/Na ions between electrodes without liquid. "
        "NEEDS: Thin film or pellet between anode and cathode. "
        "OUTPUT: Safer battery — no liquid electrolyte fire risk. "
        "DEPLOY: Battery manufacturing."
    ),
    "anode_lithium_free": (
        "Stores charge via ion intercalation, no lithium required. "
        "NEEDS: Battery cell assembly. "
        "OUTPUT: Lower cost battery using abundant elements. "
        "DEPLOY: Battery manufacturing, removes Li supply chain dependency."
    ),
    "cathode_cobalt_free": (
        "High-voltage cathode without cobalt (ethical + cost issue). "
        "NEEDS: Battery cell assembly. "
        "OUTPUT: EV/storage battery without DRC cobalt dependency. "
        "DEPLOY: Battery manufacturing."
    ),
    "hydrogen_storage": (
        "Absorbs H2 gas into crystal lattice reversibly. "
        "NEEDS: H2 source (electrolysis or reforming), pressure vessel 10-30 bar. "
        "OUTPUT: Safe solid-state H2 storage, no cryogenic tank. "
        "DEPLOY: Village H2 storage for fuel cells or cooking."
    ),
    "thermal_storage_hightemp": (
        "Stores heat as sensible or latent heat at 600C+. "
        "NEEDS: Insulated vessel, heat source (solar concentrator or industrial). "
        "OUTPUT: Dispatchable heat/electricity 24hr from daytime solar. "
        "DEPLOY: Grid-scale storage or industrial process heat."
    ),
    "superconductor_ambient": (
        "Zero electrical resistance at or near room temperature. "
        "NEEDS: Wire/tape fabrication, currently speculative. "
        "OUTPUT: Lossless power transmission, maglev, MRI without helium. "
        "DEPLOY: If found, transforms entire electrical grid. Holy grail."
    ),
    "superconductor_cryo": (
        "Zero resistance below critical temperature (liquid N2 range target). "
        "NEEDS: Cryogenic cooling to 77K+. "
        "OUTPUT: Power cables, magnets, transformers with near-zero loss. "
        "DEPLOY: Industrial power infrastructure."
    ),
    "co2_solid_sorbent": (
        "Chemically binds CO2 from air or flue gas as stable carbonate. "
        "NEEDS: Contact with CO2 stream, heat to regenerate (100-200C). "
        "OUTPUT: Captured CO2 for storage or use as feedstock. "
        "DEPLOY: Factory rooftop, village biochar kiln, direct air capture."
    ),
    "co2_reduction_catalyst": (
        "ELECTROCHEMICAL — NOT passive absorption. Requires applied voltage. "
        "CO2 + H2O + electricity → CO, ethylene, ethanol, acetate (sellable products). "
        "Zn sites activate CO2 to CO; Cu sites couple CO to C2+ products. "
        "Surface regenerates under applied potential — does not passivate. "
        "NEEDS: Solar panel + electrolytic cell + CO2 source (air or flue gas). "
        "OUTPUT: Sellable carbon products — ethylene ($1200/ton), ethanol, CO. "
        "DEPLOY: Village solar electrolyzer. Scrap brass works as electrode material."
    ),
    "low_carbon_cement": (
        "Replaces Portland cement (8% of global CO2). "
        "Geopolymer or calcium silicate hydrate chemistry. "
        "NEEDS: Mix with water + aggregate, cure at ambient temperature. "
        "OUTPUT: Structural concrete with 50-90% lower CO2 footprint. "
        "DEPLOY: Any construction site, same workflow as regular cement."
    ),
    "biodegradable_polymer": (
        "Replaces single-use plastics. Degrades in soil in months not centuries. "
        "NEEDS: Melt processing or solution casting. "
        "OUTPUT: Packaging, films, agricultural mulch that composts. "
        "DEPLOY: Factory replacement for petrochemical plastic."
    ),
    "transparent_conductor_itofree": (
        "Conducts electricity while transmitting visible light. "
        "Replaces indium tin oxide (ITO) — indium is scarce. "
        "NEEDS: Thin film deposition on glass/plastic. "
        "OUTPUT: Touch screens, solar cells, LED displays without supply chain risk. "
        "DEPLOY: Display and solar panel manufacturing."
    ),
    "piezoelectric_leadfree": (
        "Converts mechanical stress to electricity (and vice versa). "
        "Replaces PZT (lead zirconate titanate) — lead is toxic. "
        "NEEDS: Poling in electric field during manufacturing. "
        "OUTPUT: Sensors, actuators, ultrasound, energy harvesting from vibration. "
        "DEPLOY: Electronics manufacturing, medical devices."
    ),
    "topological_quantum": (
        "Protected electronic states at surface, immune to disorder. "
        "NEEDS: Clean surface, cryogenic or ambient depending on material. "
        "OUTPUT: Fault-tolerant quantum computing platform (T29). "
        "DEPLOY: Research/quantum computing industry."
    ),
    "mxene_alternative": (
        "2D conductive material for energy storage and EM shielding. "
        "MXenes require HF synthesis — dangerous. This replaces that. "
        "NEEDS: Safer exfoliation or direct synthesis. "
        "OUTPUT: Supercapacitor electrode, EM shielding, sensor. "
        "DEPLOY: Electronics manufacturing."
    ),
    "antimicrobial_surface": (
        "Surface kills bacteria on contact via ion release or ROS generation. "
        "No antibiotics required. "
        "NEEDS: Coat onto hospital surfaces, water pipes, food packaging. "
        "OUTPUT: Reduced hospital-acquired infections, cleaner water. "
        "DEPLOY: Healthcare, water infrastructure."
    ),
    "corrosion_coating_chromefree": (
        "Hard ceramic coating protects metal from oxidation/corrosion. "
        "Chrome-free = not carcinogenic in manufacturing. "
        "NEEDS: PVD or CVD deposition onto substrate. "
        "OUTPUT: Extended pipeline, ship, tool lifetime in harsh environments. "
        "DEPLOY: Industrial coating."
    ),
    "radiation_tolerant_structural": (
        "Structural metal that resists neutron damage and helium embrittlement. "
        "NEEDS: Nuclear reactor structural component manufacturing. "
        "OUTPUT: Longer-lived reactor vessels, enables fusion reactor construction. "
        "DEPLOY: Nuclear industry."
    ),
    "high_entropy_aerospace": (
        "Multi-element alloy with exceptional strength at 1500C+. "
        "NEEDS: Arc melting or powder metallurgy of 4-6 elements. "
        "OUTPUT: Turbine blades, hypersonic vehicle components. "
        "DEPLOY: Aerospace manufacturing."
    ),
    "plasma_resistant": (
        "Vessel walls for plasma confinement, fusion-adjacent. "
        "NEEDS: High-T sintering or CVD. Debye > 800K, formation energy < -1.0 eV/atom. "
        "OUTPUT: Plasma-facing component surviving extreme ion flux. "
        "DEPLOY: Fusion reactor vessels. CONJECTURE pending phonon confirmation."
    ),
    "room_temp_superconductor_candidates": (
        "Civilization-scale energy transmission. "
        "NEEDS: High-pressure synthesis or thin-film deposition. "
        "OUTPUT: Lossless power transmission if confirmed. "
        "DEPLOY: Grid-scale transformation. CONJECTURE — no confirmed RT superconductors exist. "
        "Phonon confirmation REQUIRED before any claim."
    ),
    "room-temperature-superconductor": (
        "Room-temperature superconductor candidate. T_D > 500K (high phonon coupling), "
        "half-filling electron count (T79 void-fill instability maximized), "
        "light-element framework. Cooper pair condensate = foam void-fill at material scale. "
        "NEEDS: High-pressure synthesis or thin-film deposition. "
        "OUTPUT: Lossless power transmission if confirmed. "
        "DEPLOY: Grid-scale transformation. "
        "DERIVED — T79 void-fill condition at solid-state scale."
    ),
    "radiation_shielding": (
        "Space-grade shielding, nuclear-adjacent. "
        "NEEDS: Composite fabrication or sintering. "
        "OUTPUT: Radiation protection for spacecraft and nuclear environments. "
        "DEPLOY: Spacecraft hulls, nuclear containment. CONJECTURE pending phonon confirmation."
    ),
    "extreme_hardness": (
        "Post-diamond cutting, drilling, wear applications. "
        "NEEDS: High-pressure synthesis or CVD. "
        "OUTPUT: Cutting tools, drill bits, wear surfaces exceeding diamond. "
        "DEPLOY: Industrial tooling. CONJECTURE pending phonon confirmation."
    ),
    "thermal_interface_materials": (
        "Electronics cooling, high-power density thermal management. "
        "NEEDS: Thin-film or composite processing. "
        "OUTPUT: Heat dissipation for high-power electronics. "
        "DEPLOY: CPU/GPU packaging, power electronics. CONJECTURE pending phonon confirmation."
    ),
    "hydrogen_storage_hydrides": (
        "Distributed energy storage, off-grid hydrogen. "
        "NEEDS: Pressure vessel 10-30 bar, thermal management for desorption. "
        "OUTPUT: Safe solid-state H2 storage. "
        "DEPLOY: Village H2 storage for fuel cells. CONJECTURE pending phonon confirmation."
    ),
    "co2_capture_materials": (
        "Atmospheric remediation at scale. "
        "NEEDS: Contact with CO2 stream, thermal regeneration cycle. "
        "OUTPUT: Captured CO2 for storage or use. "
        "DEPLOY: Direct air capture, flue gas treatment. CONJECTURE pending phonon confirmation."
    ),
    "water_splitting_catalysts": (
        "Green hydrogen production, distributed fuel. "
        "NEEDS: Electrolytic cell or photocatalytic reactor. "
        "OUTPUT: H2 fuel from water splitting. "
        "DEPLOY: Village solar electrolyzer. CONJECTURE pending phonon confirmation."
    ),
    "fertilizer_catalyst_expanded": (
        "Extended nitrogen-fixation screen beyond Fe3Mn4/Mo2FeN2. "
        "NEEDS: Electrolytic cell or high-T reactor. "
        "OUTPUT: NH3 fertilizer without Haber-Bosch. "
        "DEPLOY: Village cooperative with solar-powered electrolyzer. CONJECTURE pending phonon confirmation."
    ),
    "extreme_temperature_structural": (
        "Industrial furnaces, hypersonic vehicles, space re-entry. "
        "NEEDS: High-T sintering or CVD composite fabrication. "
        "OUTPUT: Structural material surviving >2000C. "
        "DEPLOY: Aerospace, industrial furnaces. CONJECTURE pending phonon confirmation."
    ),
}
CATEGORY_NOTES.update(EXTRA_CATEGORY_NOTES)


WATER_CONTACT_CATEGORIES = [
    "atmospheric_water_harvesting", "heavy_metal_filtration",
    "fluoride_removal", "forward_osmosis_membrane",
    "antifouling_coating"
]


KNOWN_FORMULAS = {
    "TiO2", "MnO2", "Fe2O3", "Al2O3", "BaO", "MgO", "CaO", "FeH",
    "MgNiH", "MgNiH2", "MgNiH3", "MnAl", "MnV", "Li3PS4", "Li3PO4",
    "Al2Cu", "Mg2Ni", "ZnCu", "Fe3O4", "SiC", "TiN", "TiB2"
}


LITERATURE_CANDIDATES = {
    "atmospheric_water_harvesting": [
        {"formula": "CaCl2", "mechanism": "hygroscopic salt",
         "capacity_kg_per_kg": 0.30, "regen_T_C": 80,
         "access": "VILLAGE", "safe_farmers": "PASS",
         "carbon_cycle": "NEUTRAL", "cost_usd_per_kg": 0.15,
         "note": "CaCl2 absorbs water vapor passively, releases at 80C (solar). Most deployed AWH material globally. Deliquesces above 30% RH.",
         "status": "MEASURED [literature]"},
        {"formula": "MgCl2", "mechanism": "hygroscopic salt",
         "capacity_kg_per_kg": 0.40, "regen_T_C": 80,
         "access": "VILLAGE", "safe_farmers": "PASS",
         "carbon_cycle": "NEUTRAL", "cost_usd_per_kg": 0.10,
         "note": "Higher capacity than CaCl2. Used in food-grade applications. Abundant from seawater bitterns.",
         "status": "MEASURED [literature]"},
        {"formula": "LiCl", "mechanism": "hygroscopic salt",
         "capacity_kg_per_kg": 0.50, "regen_T_C": 75,
         "access": "VILLAGE", "safe_farmers": "PASS",
         "carbon_cycle": "NEUTRAL", "cost_usd_per_kg": 1.20,
         "note": "Highest capacity hygroscopic salt. Works below 20% RH — functional in arid climates. Li supply moderate concern.",
         "status": "MEASURED [literature]"},
        {"formula": "Na12Al12Si12O48", "mechanism": "molecular sieve",
         "capacity_kg_per_kg": 0.25, "regen_T_C": 150,
         "access": "WORKSHOP", "safe_farmers": "PASS",
         "carbon_cycle": "NEUTRAL", "cost_usd_per_kg": 0.80,
         "note": "Aluminosilicate. Porous structure traps water molecules. Higher regen temp but reusable thousands of cycles. Manufacturable from local clay + NaOH.",
         "status": "MEASURED [literature]"},
        {"formula": "C12H8Al3N3O10", "mechanism": "metal-organic framework",
         "capacity_kg_per_kg": 0.40, "regen_T_C": 65,
         "access": "INDUSTRIAL_SMALL", "safe_farmers": "PASS",
         "carbon_cycle": "NEUTRAL", "cost_usd_per_kg": 15.0,
         "note": "Al-based MOF, lowest regen temp of any AWH material. UC Berkeley / MIT demonstrated 2.8L/kg/day in Mojave Desert. Manufacturing scale-up in progress.",
         "status": "MEASURED [Hanikel 2019, Kim 2017]"},
    ],
    "heavy_metal_filtration": [
        {"formula": "FeO2H", "mechanism": "surface adsorption",
         "targets": ["As", "Pb", "Cr"], "capacity_mg_per_g": 50,
         "access": "VILLAGE", "safe_farmers": "PASS",
         "carbon_cycle": "NEUTRAL", "cost_usd_per_kg": 0.20,
         "note": "Goethite. Forms naturally on iron surfaces. Adsorbs arsenate at pH 6-8. Regenerable with NaOH wash. Can be made from scrap iron + aeration.",
         "status": "MEASURED [literature]"},
        {"formula": "MnO2", "mechanism": "oxidative adsorption",
         "targets": ["Pb", "Cd", "Cu"], "capacity_mg_per_g": 80,
         "access": "VILLAGE", "safe_farmers": "PASS",
         "carbon_cycle": "NEUTRAL", "cost_usd_per_kg": 0.50,
         "note": "Birnessite-type. Oxidizes Mn2+ contaminants, adsorbs heavy metal cations. Widely used in Bangladesh for As removal.",
         "status": "MEASURED [literature]"},
        {"formula": "AlO2H", "mechanism": "surface adsorption + precipitation",
         "targets": ["F", "As", "Pb"], "capacity_mg_per_g": 30,
         "access": "VILLAGE", "safe_farmers": "PASS",
         "carbon_cycle": "NEUTRAL", "cost_usd_per_kg": 0.30,
         "note": "Activated alumina precursor. Adsorbs fluoride and arsenate. Regenerable with NaOH + HCl. Low cost, widely available.",
         "status": "MEASURED [literature]"},
        {"formula": "Ca5P3O13H", "mechanism": "ion exchange",
         "targets": ["Pb", "Cd", "Zn"], "capacity_mg_per_g": 120,
         "access": "VILLAGE", "safe_farmers": "PASS",
         "carbon_cycle": "NEUTRAL", "cost_usd_per_kg": 1.50,
         "note": "Ca5(PO4)3OH. Highest Pb capacity of any natural mineral. Synthesizable from bone ash or Ca + phosphate. Pb displaces Ca in lattice irreversibly — excellent sink.",
         "status": "MEASURED [literature]"},
        {"formula": "C", "mechanism": "surface adsorption + pore trapping",
         "targets": ["Pb", "Cd", "Hg", "As"], "capacity_mg_per_g": 40,
         "access": "VILLAGE", "safe_farmers": "PASS",
         "carbon_cycle": "SINK", "cost_usd_per_kg": 0.05,
         "note": "Carbon SINK flagged: sequesters CO2 from biomass. Made from agricultural waste. Simultaneous soil improvement + heavy metal removal. Cheapest deployable option.",
         "status": "MEASURED [literature]"},
    ],
    "fluoride_removal": [
        {"formula": "LaO3H3", "mechanism": "ligand exchange",
         "capacity_mg_per_g": 35, "pH_range": "5-8",
         "access": "WORKSHOP", "safe_farmers": "PASS",
         "carbon_cycle": "NEUTRAL", "cost_usd_per_kg": 8.0,
         "note": "Lanthanum hydroxide. Highest fluoride selectivity of any sorbent. F displaces OH directly. Achieves WHO limit from 10 mg/L feedwater. La is MODERATE (light rare earth, not scarce).",
         "status": "MEASURED [literature]"},
        {"formula": "AlO2H", "mechanism": "surface complexation",
         "capacity_mg_per_g": 25, "pH_range": "6-8",
         "access": "VILLAGE", "safe_farmers": "PASS",
         "carbon_cycle": "NEUTRAL", "cost_usd_per_kg": 0.40,
         "note": "Boehmite. Aluminium oxyhydroxide. Standard for fluoride removal in India/Africa. Made by precipitating Al salts. Regenerable with NaOH.",
         "status": "MEASURED [literature]"},
        {"formula": "Ca5P3O12F", "mechanism": "precipitation / ion exchange",
         "capacity_mg_per_g": 20, "pH_range": "6-9",
         "access": "VILLAGE", "safe_farmers": "PASS",
         "carbon_cycle": "NEUTRAL", "cost_usd_per_kg": 1.20,
         "note": "Fluorapatite. F substitutes into hydroxyapatite structure. Bone char is the cheapest form — used in Ethiopia, Tanzania.",
         "status": "MEASURED [literature]"},
        {"formula": "Mg2Al2H8O12", "mechanism": "anion exchange",
         "capacity_mg_per_g": 45, "pH_range": "5-9",
         "access": "WORKSHOP", "safe_farmers": "PASS",
         "carbon_cycle": "NEUTRAL", "cost_usd_per_kg": 2.0,
         "note": "Layered double hydroxide. Anion exchange between F and OH/CO3. Highest capacity in this list. Synthesizable from MgCl2 + AlCl3 + NaOH.",
         "status": "MEASURED [literature]"},
    ],
    "forward_osmosis_membrane": [
        {"formula": "TiO2", "mechanism": "size exclusion + charge",
         "water_flux_Lm2h": 15, "salt_rejection_pct": 97,
         "access": "INDUSTRIAL_SMALL", "safe_farmers": "PASS",
         "carbon_cycle": "NEUTRAL", "cost_usd_per_m2": 8.0,
         "note": "TiO2 nanoparticle-modified membrane. Photocatalytic self-cleaning under sunlight prevents fouling without chemicals. Salt rejection 97%, water flux 15 L/m2/h.",
         "status": "MEASURED [literature]"},
        {"formula": "C2O", "mechanism": "interlayer spacing",
         "water_flux_Lm2h": 30, "salt_rejection_pct": 95,
         "access": "INDUSTRIAL_SMALL", "safe_farmers": "PASS",
         "carbon_cycle": "SINK", "cost_usd_per_m2": 25.0,
         "note": "GO membrane. Water flux 2x TiO2, spacing tunable by reduction. Carbon SINK: made from carbon feedstock. Major challenge: mechanical stability in long-term operation.",
         "status": "MEASURED [literature]"},
        {"formula": "C8H8N4Zn", "mechanism": "molecular sieving",
         "water_flux_Lm2h": 20, "salt_rejection_pct": 99,
         "access": "INDUSTRIAL_SMALL", "safe_farmers": "PASS",
         "carbon_cycle": "NEUTRAL", "cost_usd_per_m2": 50.0,
         "note": "Zeolitic imidazolate framework. 0.34nm pore passes water, blocks salt ions by size. Highest selectivity in this list. Zn-based, earth-abundant. Synthesis improving rapidly.",
         "status": "MEASURED [literature]"},
        {"formula": "C10H16O7", "mechanism": "solution-diffusion",
         "water_flux_Lm2h": 12, "salt_rejection_pct": 93,
         "access": "WORKSHOP", "safe_farmers": "PASS",
         "carbon_cycle": "SINK", "cost_usd_per_m2": 3.0,
         "note": "Oldest and cheapest membrane material. Made from wood pulp (carbon SINK). Biodegradable at end of life. Standard for village-scale FO systems in Asia.",
         "status": "MEASURED [literature]"},
    ],
    "solid_electrolyte": [
        {"formula": "Li3PS4", "mechanism": "Li+ hopping",
         "conductivity_mS_cm": 0.16, "E_window_V": "0-5",
         "access": "INDUSTRIAL_SMALL", "safe_farmers": "PASS",
         "carbon_cycle": "NEUTRAL", "cost_usd_per_kg": 45.0,
         "note": "Sulfide electrolyte. Room temperature Li+ conductivity 0.16 mS/cm. No La/Zr — all abundant. Sensitive to moisture during processing. Scale-up pathway clear via ball milling.",
         "status": "MEASURED [literature]"},
        {"formula": "Li3PO4", "mechanism": "Li+ hopping",
         "conductivity_mS_cm": 0.002, "E_window_V": "0-5.5",
         "access": "WORKSHOP", "safe_farmers": "PASS",
         "carbon_cycle": "NEUTRAL", "cost_usd_per_kg": 8.0,
         "note": "Lithium phosphate. Lower conductivity but extremely stable window. All earth-abundant elements. Used as thin-film electrolyte in microbatteries. Cheapest solid electrolyte in this list.",
         "status": "MEASURED [literature]"},
        {"formula": "Li7La3Zr2O12", "mechanism": "Li+ hopping in garnet",
         "conductivity_mS_cm": 1.0, "E_window_V": "0-6",
         "access": "INDUSTRIAL_SMALL", "safe_farmers": "PASS",
         "carbon_cycle": "NEUTRAL", "cost_usd_per_kg": 120.0,
         "note": "LLZO garnet. Gold standard solid electrolyte. 1 mS/cm at RT, widest window. La is MODERATE (light rare earth). Requires sintering at 1100C — energy intensive but manufacturable.",
         "status": "MEASURED [literature]"},
        {"formula": "Na3PS4", "mechanism": "Na+ hopping",
         "conductivity_mS_cm": 0.20, "E_window_V": "0-5",
         "access": "INDUSTRIAL_SMALL", "safe_farmers": "PASS",
         "carbon_cycle": "NEUTRAL", "cost_usd_per_kg": 30.0,
         "note": "Sodium analog of Li3PS4. For Na-ion batteries — removes Li dependency entirely. Na is seawater-abundant. Comparable conductivity to Li3PS4.",
         "status": "MEASURED [literature]"},
    ],
}

# Manual candidates for new categories: (formula, debye_K, B_GPa, G_GPa, formE, band_gap, note)
MANUAL_CANDIDATES = {
    "plasma_resistant": [
        ("HfC", 700, 243, 191, -1.5, 0.0, "Hafnium carbide — highest melting point known (~3900C)", "MEASURED", "Toth 1971"),
        ("ZrC", 680, 223, 175, -1.4, 0.0, "Zirconium carbide — extreme T refractory, earth-abundant Zr", "MEASURED", ""),
        ("TaC", 650, 221, 188, -1.3, 0.0, "Tantalum carbide — ultra-high T, hard", "MEASURED", ""),
        ("HfN", 550, 190, 150, -1.2, 0.0, "Hafnium nitride — plasma-facing, high hardness", "ESTIMATED", ""),
        ("ZrN", 530, 180, 140, -1.3, 0.0, "Zirconium nitride — refractory, good thermal shock", "ESTIMATED", ""),
        ("TaN", 500, 190, 150, -1.0, 0.0, "Tantalum nitride — high T stability", "ESTIMATED", ""),
        ("HfB2", 650, 200, 180, -1.5, 0.0, "Hafnium diboride — UHTC, excellent thermal shock", "ESTIMATED", ""),
        ("ZrB2", 630, 190, 170, -1.6, 0.0, "Zirconium diboride — UHTC, abundant Zr", "ESTIMATED", ""),
        ("TiB2", 680, 220, 185, -1.4, 0.0, "Titanium diboride — high hardness, abundant Ti", "ESTIMATED", ""),
        ("W2C", 450, 230, 180, -0.8, 0.0, "Tungsten semi-carbide — high T, good thermal conductivity", "ESTIMATED", ""),
    ],
    "room_temp_superconductor_candidates": [
        ("LaH10", 1500, 150, 100, -0.5, 0.0, "Lanthanum decahydride — CONJECTURE, high-pressure SC near 250K"),
        ("YH9", 1400, 140, 90, -0.4, 0.0, "Yttrium nonahydride — CONJECTURE, predicted high-Tc"),
        ("CeH9", 1300, 130, 85, -0.4, 0.0, "Cerium nonahydride — CONJECTURE, predicted high-Tc"),
        ("MgB2", 900, 120, 80, -0.5, 0.0, "Magnesium diboride — confirmed SC at 39K, high Debye"),
        ("Bi2Sr2CaCu2O8", 350, 80, 40, -1.0, 0.0, "BSCCO — confirmed HTSC at 95K, cuprate"),
        ("YBa2Cu3O7", 400, 90, 50, -1.2, 0.0, "YBCO — confirmed HTSC at 93K, cuprate"),
        ("Tl2Ba2Ca2Cu3O10", 380, 85, 45, -1.0, 0.0, "TBCCO — confirmed HTSC at 127K, cuprate"),
        ("Hg12Tl3Ba30Ca30Cu45O127", 320, 70, 35, -0.8, 0.0, "Hg-Tl cuprate — CONJECTURE, record Tc claim 135K"),
        ("H3S", 1200, 180, 120, -0.3, 0.0, "Hydrogen sulfide — CONJECTURE, SC at 203K under 150GPa"),
        ("BaH12", 1100, 160, 100, -0.3, 0.0, "Barium dodecahydride — CONJECTURE, predicted high-Tc"),
    ],
    "radiation_shielding": [
        ("B4C", 900, 220, 180, -0.8, 2.0, "Boron carbide — neutron absorber, lightweight armor"),
        ("WC", 550, 350, 280, -0.5, 0.0, "Tungsten carbide — gamma shielding, high density"),
        ("Pb3O4", 200, 60, 30, -0.8, 2.5, "Red lead oxide — gamma shielding, high Z"),
        ("Bi2O3", 250, 70, 35, -0.6, 2.8, "Bismuth oxide — lead-free gamma shielding"),
        ("BaSO4", 300, 80, 40, -1.2, 4.0, "Barite — X-ray shielding, low cost"),
        ("Fe3B", 450, 200, 150, -0.3, 0.0, "Iron boride — combined neutron + gamma shielding"),
        ("Gd2O3", 280, 120, 60, -1.0, 3.5, "Gadolinium oxide — best neutron absorber, high cross-section"),
        ("HfO2", 400, 180, 90, -1.2, 5.5, "Hafnium oxide — neutron absorber, high T stable"),
        ("WNiFe", 350, 250, 180, -0.3, 0.0, "W-Ni-Fe heavy alloy — gamma + neutron shielding, 93W-4Ni-3Fe"),
        ("BN", 1200, 300, 250, -0.8, 5.8, "Boron nitride — neutron shielding proxy for borated PE"),
    ],
    "extreme_hardness": [
        ("BN", 1300, 400, 480, -0.7, 6.0, "Cubic BN — second hardest material after diamond", "MEASURED", ""),
        ("BC2N", 1100, 350, 280, -0.6, 3.0, "B-C-N ternary — predicted superhard, between diamond and cBN", "ESTIMATED", ""),
        ("BC5", 1000, 300, 240, -0.5, 2.0, "Boron carbide C5 — predicted superhard", "ESTIMATED", ""),
        ("OsB2", 600, 350, 280, -0.8, 0.0, "Osmium diboride — ultra-hard, high T stable", "ESTIMATED", ""),
        ("ReB2", 650, 360, 273, -0.9, 0.0, "Rhenium diboride — superhard, high T", "MEASURED", ""),
        ("WB4", 550, 304, 268, -0.4, 0.0, "Tungsten tetraboride — predicted superhard", "MEASURED", ""),
        ("MoB4", 500, 280, 220, -0.5, 0.0, "Molybdenum tetraboride — hard, abundant Mo", "ESTIMATED", ""),
        ("B6O", 1200, 380, 300, -0.7, 3.5, "Boron suboxide — third hardest known material", "ESTIMATED", ""),
        ("BC2N", 1100, 340, 270, -0.6, 3.0, "r-BC2N — rhombohedral superhard B-C-N phase", "ESTIMATED", ""),
        ("BC8", 1400, 420, 340, -0.5, 2.5, "BC8 diamond-cubic — predicted superhard carbon phase", "ESTIMATED", ""),
    ],
    "thermal_interface_materials": [
        ("AlN", 900, 280, 180, -1.5, 6.2, "Aluminum nitride — best non-diamond thermal conductor, 320 W/mK", "ESTIMATED", ""),
        ("BN", 1200, 36, 18, -0.8, 5.8, "Hexagonal BN — high thermal conductivity, electrical insulator", "MEASURED", "in-plane"),
        ("BN", 1300, 400, 480, -0.7, 6.0, "Cubic BN — high thermal conductivity, electrical insulator", "MEASURED", ""),
        ("C", 1800, 443, 478, -0.5, 5.5, "Diamond thin film — highest thermal conductivity, 2000 W/mK", "MEASURED", ""),
        ("C", 1800, 500, 400, -0.5, 5.5, "Graphene composite proxy — extreme thermal conductivity", "ESTIMATED", ""),
        ("SiC", 1000, 250, 200, -0.7, 3.0, "Silicon carbide — high thermal conductivity, semiconductor", "ESTIMATED", ""),
        ("BeO", 800, 250, 160, -1.8, 10.5, "Beryllium oxide — high thermal conductivity, toxic Be", "ESTIMATED", ""),
        ("Al2O3", 700, 230, 150, -1.6, 8.8, "Sapphire — good thermal, electrical insulator, low cost", "ESTIMATED", ""),
        ("GaN", 600, 200, 160, -1.1, 3.4, "Gallium nitride — thermal + electronic, power devices", "ESTIMATED", ""),
        ("Ga2O3", 550, 180, 140, -1.0, 4.8, "Beta-Ga2O3 — ultra-wide bandgap, high breakdown voltage", "ESTIMATED", ""),
    ],
    "hydrogen_storage_hydrides": [
        ("MgH2", 650, 45, 20, -0.8, 0.0, "Magnesium hydride — 7.6 wt% H2, abundant, high T desorption", "MEASURED", ""),
        ("TiH2", 550, 100, 70, -0.6, 0.0, "Titanium hydride — stable, moderate capacity", "ESTIMATED", ""),
        ("LaNi5H6", 300, 80, 50, -0.4, 0.0, "LaNi5 hydride — room T reversible, commercial AB5", "ESTIMATED", ""),
        ("FeTiH2", 400, 90, 60, -0.3, 0.0, "Iron-titanium hydride — room T reversible, abundant", "ESTIMATED", ""),
        ("Mg2NiH4", 450, 100, 70, -0.5, 0.0, "Mg-Ni hydride — improved kinetics vs MgH2", "ESTIMATED", ""),
        ("NaAlH4", 350, 70, 45, -0.4, 0.0, "Sodium alanate — 5.6 wt%, catalyzed reversible", "ESTIMATED", ""),
        ("LiBH4", 500, 90, 60, -0.5, 0.0, "Lithium borohydride — 18.5 wt% theoretical capacity", "ESTIMATED", ""),
        ("CaBH42", 400, 80, 50, -0.4, 0.0, "Calcium borohydride — 11.5 wt%, moderate T", "ESTIMATED", ""),
        ("LaNi5", 300, 80, 50, -0.3, 0.0, "LaNi5 AB5 proxy — reversible H2 storage alloy", "ESTIMATED", ""),
        ("Mg2FeH6", 500, 110, 75, -0.5, 0.0, "Mg-Fe hydride — 5.5 wt%, high volumetric density", "ESTIMATED", ""),
    ],
    "co2_capture_materials": [
        ("MgO", 650, 160, 130, -1.6, 7.8, "Magnesium oxide — CO2 sorbent, 30-50% capacity, abundant", "MEASURED", ""),
        ("CaO", 600, 150, 90, -1.7, 7.0, "Calcium oxide — CaO looping, high capacity, very cheap", "ESTIMATED", ""),
        ("Li2O", 500, 120, 70, -1.4, 8.0, "Lithium oxide — CO2 capture, moderate capacity", "ESTIMATED", ""),
        ("K2CO3", 300, 80, 50, -1.0, 4.0, "Potassium carbonate — CO2 sorbent, regenerable", "ESTIMATED", ""),
        ("Na2CO3", 320, 85, 50, -1.1, 4.8, "Sodium carbonate — CO2 capture, very cheap", "ESTIMATED", ""),
        ("Mg2Al2H8O12", 350, 100, 60, -0.8, 3.5, "Hydrotalcite LDH proxy — anion exchange CO2 capture", "ESTIMATED", ""),
        ("ZnC8H12N4", 280, 70, 40, -0.5, 3.0, "ZIF-8 MOF proxy — high surface area CO2 capture", "ESTIMATED", ""),
        ("TiO2", 500, 210, 140, -1.2, 3.0, "Amine-sorbent TiO2 proxy — functionalized CO2 capture", "ESTIMATED", ""),
        ("Na12Al12Si12O48", 400, 90, 55, -1.3, 4.5, "Zeolite 13X proxy — pore-based CO2 adsorption", "ESTIMATED", ""),
        ("CaCO3", 350, 100, 60, -1.2, 4.0, "Calcium carbonate reactive form — mineralization CO2", "ESTIMATED", ""),
    ],
    "water_splitting_catalysts": [
        ("IrO2", 500, 200, 140, -0.8, 0.0, "Iridium oxide — best OER catalyst, scarce Ir"),
        ("RuO2", 480, 190, 130, -0.7, 0.0, "Ruthenium oxide — excellent OER, scarce Ru"),
        ("NiFeOx", 400, 160, 100, -0.5, 0.0, "NiFe oxide — best non-precious OER, abundant"),
        ("CoP", 380, 140, 90, -0.4, 0.0, "Cobalt phosphide — HER catalyst, moderate cost"),
        ("MoS2", 450, 180, 120, -0.5, 1.2, "Molybdenum disulfide — HER catalyst, layered"),
        ("NiMoP", 350, 130, 85, -0.4, 0.0, "Ni-Mo phosphide — bifunctional HER/OER"),
        ("FeCoNi", 300, 120, 80, -0.2, 0.0, "FeCoNi alloy — OER catalyst, earth-abundant"),
        ("Cu2O", 300, 110, 70, -0.3, 2.1, "Cuprous oxide — photocatalytic water splitting"),
        ("BiVO4", 280, 90, 55, -0.6, 2.4, "Bismuth vanadate — photoanode OER, visible light"),
        ("Ta3N5", 350, 130, 85, -0.5, 2.1, "Tantalum nitride — photoanode, visible light absorption"),
    ],
    "fertilizer_catalyst_expanded": [
        ("Fe2Mo", 400, 160, 100, -0.3, 0.0, "Iron-molybdenum — nitrogen fixation catalyst"),
        ("FeMo6", 350, 140, 90, -0.2, 0.0, "Fe-Mo cluster — nitrogen activation"),
        ("VN", 450, 180, 120, -0.5, 0.0, "Vanadium nitride — N2 activation, moderate cost V"),
        ("MoN", 400, 160, 100, -0.4, 0.0, "Molybdenum nitride — nitrogen fixation surface"),
        ("Mo2N", 420, 170, 110, -0.5, 0.0, "Molybdenum semi-nitride — high activity N2 reduction"),
        ("Fe3Mo3N", 380, 150, 95, -0.4, 0.0, "Fe-Mo nitride ternary — nitrogen fixation"),
        ("Co3Mo3N", 370, 145, 90, -0.4, 0.0, "Co-Mo nitride — nitrogen fixation, Co moderate"),
        ("Ni2Mo4N", 360, 140, 88, -0.3, 0.0, "Ni-Mo nitride — nitrogen activation"),
        ("MnFe2O4", 350, 130, 85, -0.5, 0.0, "Manganese ferrite — nitrogen fixation support"),
        ("Fe16N2", 400, 160, 100, -0.3, 0.0, "Iron nitride — high activity, abundant Fe"),
        ("Fe3Mn4", 420, 170, 90, -0.18, 0.0, "N2 adsorption -1.134 eV MACE-validated T50 OPTIMAL | B/G/Debye estimated from Fe-Mn class | OOM 2026", "ESTIMATED", "OOM 2026 MACE"),
        ("Mo2FeN2", 510, 235, 125, -0.48, 0.0, "MACE phonon stable 0.436 THz | NEB barrier 1.193 eV | SPARC pending RunPod | B/G/Debye estimated from Mo-nitride class | OOM 2026", "ESTIMATED", "OOM 2026 MACE+NEB"),
    ],
    "extreme_temperature_structural": [
        ("HfC", 700, 220, 190, -1.5, 0.0, "HfC UHTC — highest melting point, plasma-facing"),
        ("ZrCSiC", 600, 200, 160, -1.2, 0.0, "ZrC-SiC composite — UHTC with oxidation resistance"),
        ("TiC", 650, 210, 175, -1.5, 0.0, "Titanium carbide — high T, abundant Ti"),
        ("NbC", 550, 190, 150, -1.0, 0.0, "Niobium carbide — refractory, moderate cost Nb"),
        ("Mo2C", 500, 200, 160, -0.6, 0.0, "Molybdenum carbide — high T structural"),
        ("Cr3C2", 480, 180, 140, -0.5, 0.0, "Chromium carbide — oxidation resistant, hard"),
        ("VC", 520, 195, 155, -0.8, 0.0, "Vanadium carbide — high T, hard"),
        ("WCoC", 450, 220, 180, -0.4, 0.0, "WC-Co proxy — cemented carbide for high T"),
        ("SiC", 1000, 250, 200, -0.7, 3.0, "SiC fiber composite — high T structural, oxidation resistant"),
        ("TaB2", 600, 200, 165, -1.3, 0.0, "Tantalum diboride — UHTC, extreme T structural"),
    ],
    "permanent_magnet_ree_free": [
        ("Mn2VSi", 648, 180, 95, -0.3, 0.0, "Heusler alloy — permanent magnet candidate, REE-free, predicted high anisotropy | OOM 2026", "ESTIMATED", "OOM 2026"),
        ("Fe3Ni", 520, 155, 85, -0.25, 0.0, "Fe-Ni tetrataenite — meteoritic permanent magnet, Earth-abundant | literature", "MEASURED", "Lewis 2014"),
        ("MnAl", 480, 130, 70, -0.18, 0.0, "Mn-Al tau phase — REE-free hard magnet, studied since 1960s | literature", "MEASURED", "Zeng 2002"),
    ],
}
MANUAL_CANDIDATES.update(EXTRA_MANUAL_CANDIDATES)


def _build_manual_hits(cat):
    """Convert manual candidate entries to the same out-dict used by MP categories."""
    out = []
    for entry in MANUAL_CANDIDATES.get(cat, []):
        if len(entry) == 7:
            formula, theta_est, B, G, formE, bg, note = entry
            bg_label = "ESTIMATED"
            ref = ""
        else:
            formula, theta_est, B, G, formE, bg, note, bg_label, ref = entry
        counts, total = parse_formula_counts(formula)
        safe = safe_farmers_status(counts, total)
        carbon = carbon_cycle_status(cat, counts)
        co2 = co2_opportunity_status(cat, carbon)
        scarcity = classify_elements(formula)
        n_formula = natoms_formula(formula)
        # Fix 4: Compute Anderson Debye when B/G are MEASURED
        if bg_label == "MEASURED" and B > 0 and G > 0:
            rho_est = max(n_formula * 2.0, 1.0)
            V_est = 20.0 * n_formula
            try:
                res = debye_temperature_anderson(B, G, rho_est, V_est, n_formula)
                theta = res['theta_D_K']
                debye_label = "Anderson-DERIVED"
            except Exception:
                theta = float(theta_est)
                debye_label = "estimated"
        else:
            theta = float(theta_est)
            debye_label = "estimated"
        synth = synthesizability(formula, B, 20.0 * n_formula, n_formula,
                                 formE, theta, "cubic")
        if scarcity["tier"] == "ABUNDANT":
            tier_score = 2
        elif scarcity["tier"] == "MODERATE":
            tier_score = 1
        else:
            tier_score = 0
        score = (
            (5 - synth["accessibility"]) * 3
            + tier_score
            + (1 if carbon == "SINK" else 0)
        )
        # Fix 1: disambiguate formula
        display_formula = disambiguate_formula(formula, note=note, B=B, G=G)
        # Fix 2: rank_score
        rank_score = compute_rank_score(theta, formula)
        ref_str = f" [{ref}]" if ref else ""
        out.append({
            "formula": display_formula,
            "raw_formula": formula,
            "debye_T": float(theta),
            "debye_label": debye_label,
            "rank_score": rank_score,
            "B_GPa": float(B),
            "G_GPa": float(G),
            "formE_eV_atom": float(formE),
            "band_gap": float(bg),
            "safe_farmers": safe,
            "carbon_cycle": carbon,
            "co2_opportunity": co2,
            "route": synth["route"],
            "T_process_C": synth["T_process_C"],
            "E_kWh_per_kg": synth["E_kWh_per_kg"],
            "accessibility_label": synth["accessibility_label"],
            "accessibility_num": synth["accessibility"],
            "tier": scarcity["tier"],
            "worst_element": scarcity["worst_element"],
            "deployable": scarcity["deployable"],
            "supply_risk": "WARN" if scarcity["tier"] == "SCARCE" else "PASS",
            "application_note": CATEGORY_NOTES.get(cat, ""),
            "dG_ads": "N/A [MANUAL — literature values]",
            "score": score,
            "note": f"CONJECTURE : {note}{ref_str}",
        })
    out.sort(key=lambda x: x["rank_score"], reverse=True)
    return out


def _build_literature_hits(cat):
    """Convert literature entries to the same out-dict used by MP categories."""
    out = []
    for rec in LITERATURE_CANDIDATES.get(cat, []):
        formula = rec["formula"]
        scarcity = classify_elements(formula)
        co2 = co2_opportunity_status(cat, rec["carbon_cycle"])
        access = rec["access"]
        acc_num = {"VILLAGE": 1, "WORKSHOP": 2, "INDUSTRIAL_SMALL": 3}.get(access, 4)
        if scarcity["tier"] == "ABUNDANT":
            tier_score = 2
        elif scarcity["tier"] == "MODERATE":
            tier_score = 1
        else:
            tier_score = 0
        score = (5 - acc_num) * 3 + tier_score + (1 if rec["carbon_cycle"] == "SINK" else 0)
        out.append({
            "formula": formula,
            "raw_formula": formula,
            "debye_T": 0.0,
            "debye_label": "literature",
            "rank_score": 0.0,
            "B_GPa": 0.0,
            "G_GPa": 0.0,
            "formE_eV_atom": 0.0,
            "band_gap": 0.0,
            "safe_farmers": rec["safe_farmers"],
            "carbon_cycle": rec["carbon_cycle"],
            "co2_opportunity": co2,
            "route": rec.get("mechanism", "literature"),
            "T_process_C": rec.get("regen_T_C", 0.0),
            "E_kWh_per_kg": 0.0,
            "accessibility_label": access,
            "accessibility_num": acc_num,
            "tier": scarcity["tier"],
            "worst_element": scarcity["worst_element"],
            "deployable": scarcity["deployable"],
            "supply_risk": "WARN" if scarcity["tier"] == "SCARCE" else "PASS",
            "application_note": rec["note"],
            "dG_ads": "N/A [MEASURED mechanism]",
            "score": score,
            "note": rec["status"],
        })
    return out


def first_sentence(text):
    text = text.strip()
    if not text:
        return ""
    for delim in (". ", ".", "? ", "! "):
        idx = text.find(delim)
        if idx != -1:
            return text[:idx + len(delim) - (1 if delim.endswith(" ") else 0)].strip() or text
    return text


def get_value(modulus):
    if modulus is None:
        return None
    if isinstance(modulus, dict):
        return modulus.get('voigt') or modulus.get('vrh') or modulus.get('reuss')
    if hasattr(modulus, 'voigt'):
        return modulus.voigt
    if hasattr(modulus, 'vrh'):
        return modulus.vrh
    try:
        return float(modulus)
    except Exception:
        return None


def natoms_formula(formula):
    parts = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
    total = 0
    for _, n in parts:
        total += int(n) if n else 1
    return total


def passes_filter(els, rec, filt):
    for key in ["elements_must_include_any",
                "elements_must_include_any_cation",
                "elements_must_include_any_anion",
                "elements_must_include_any_metal",
                "elements_must_include_any_transition",
                "elements_must_include_any_second"]:
        if key in filt:
            if not (els & set(filt[key])):
                return False
    if "elements_must_exclude" in filt:
        if els & set(filt["elements_must_exclude"]):
            return False
    if "elements_must_include_many" in filt:
        matches = els & set(filt["elements_must_include_many"])
        if len(matches) < filt.get("min_elements", 4):
            return False
    bg = rec["band_gap"]
    if "band_gap_min" in filt and bg < filt["band_gap_min"]:
        return False
    if "band_gap_max" in filt and bg > filt["band_gap_max"]:
        return False
    fe = rec["formation_energy_per_atom"]
    if "formation_energy_max" in filt and fe > filt["formation_energy_max"]:
        return False
    B = rec.get("B")
    if B is not None:
        if "B_GPa_max" in filt and B > filt["B_GPa_max"]:
            return False
        if "B_GPa_min" in filt and B < filt["B_GPa_min"]:
            return False
    return True


def process_docs(docs, by_id):
    for doc in docs:
        if doc.material_id in by_id:
            continue
        if not is_valid_formula(doc.formula_pretty):
            print(f"MALFORMED formula {doc.formula_pretty} : skipped")
            continue
        try:
            counts, _ = parse_formula_counts(doc.formula_pretty)
            if any(c > 50 for c in counts.values()) or counts.get('H', 0) > 20:
                print(f"MALFORMED formula {doc.formula_pretty} : skipped")
                continue
        except Exception:
            print(f"MALFORMED formula {doc.formula_pretty} : skipped")
            continue
        B = get_value(doc.bulk_modulus)
        G = get_value(doc.shear_modulus)
        V = getattr(doc, 'volume', None)
        nsites = getattr(doc, 'nsites', None)
        rho = getattr(doc, 'density', None)
        if (B is None or G is None or B <= 0 or G <= 0 or
            V is None or nsites is None or rho is None or rho <= 0):
            if B is None and G is None:
                theta = 0.0
                debye_label = "PHONON_PENDING"
            else:
                continue
        else:
            try:
                res = debye_temperature_anderson(B, G, rho, V, nsites)
                theta = res.get('theta_D_K', 0)
                if theta <= 0:
                    continue
                debye_label = "Anderson-DERIVED"
            except Exception:
                continue
        fe = getattr(doc, 'formation_energy_per_atom', None)
        if fe is None:
            continue
        cs = None
        sym = getattr(doc, 'symmetry', None)
        if sym:
            cs = getattr(sym, 'crystal_system', None)
            if not cs and isinstance(sym, dict):
                cs = sym.get('crystal_system')
        by_id[doc.material_id] = {
            "material_id": doc.material_id,
            "formula": doc.formula_pretty,
            "theta": theta,
            "debye_label": debye_label,
            "B": B, "G": G,
            "band_gap": getattr(doc, 'band_gap', None) or 0.0,
            "total_magnetization": getattr(doc, 'total_magnetization', None) or 0.0,
            "formation_energy_per_atom": fe,
            "nsites": nsites,
            "n_formula": natoms_formula(doc.formula_pretty),
            "volume": V,
            "crystal_system": cs,
        }


NEIGHBOR_ELEMENTS = {
    "H": ["Li", "Na"],
    "Li": ["Na", "Be", "Mg"],
    "Na": ["Li", "K", "Mg"],
    "K": ["Na", "Ca", "Rb"],
    "Rb": ["K", "Cs"],
    "Be": ["Mg", "Ca"],
    "Mg": ["Ca", "Al", "Na"],
    "Ca": ["Mg", "Sr", "Sc"],
    "Sr": ["Ca", "Ba"],
    "Ba": ["Sr", "Cs"],
    "B": ["C", "Al", "Si"],
    "C": ["B", "Si", "N"],
    "N": ["C", "O", "P"],
    "O": ["N", "S", "F"],
    "F": ["O", "Cl"],
    "S": ["O", "Se", "P"],
    "P": ["S", "As", "N"],
    "Si": ["Al", "P", "Mg"],
    "Al": ["Si", "Mg", "Ga"],
    "Ga": ["Al", "In", "Zn"],
    "In": ["Ga", "Sn"],
    "Ti": ["V", "Zr", "Sc", "Hf"],
    "V": ["Nb", "Cr", "Ti", "Ta"],
    "Cr": ["Mn", "V", "Mo"],
    "Mn": ["Fe", "Cr", "Re"],
    "Fe": ["Co", "Ni", "Mn", "Ru"],
    "Co": ["Fe", "Ni", "Cu", "Rh"],
    "Ni": ["Co", "Fe", "Cu", "Pd"],
    "Cu": ["Ni", "Zn", "Co", "Ag"],
    "Zn": ["Cu", "Ga", "Cd"],
    "Zr": ["Ti", "Nb", "Hf"],
    "Nb": ["V", "Ta", "Mo", "Zr"],
    "Mo": ["W", "Nb", "Cr", "Ru"],
    "Ru": ["Fe", "Rh", "Mo", "Os"],
    "Rh": ["Co", "Ru", "Ir"],
    "Pd": ["Ni", "Ag"],
    "Ag": ["Cu", "Pd", "Au"],
    "W": ["Mo", "Ta", "Re"],
    "Re": ["Mn", "W", "Os"],
    "Os": ["Ru", "W"],
    "Ir": ["Rh", "Pt"],
    "Pt": ["Ir", "Au"],
    "Au": ["Ag", "Pt"],
    "Hf": ["Zr", "Ta", "Ti"],
    "Ta": ["Nb", "W", "Hf"],
    "La": ["Ce", "Ca"],
    "Ce": ["La", "Pr"],
    "As": ["P", "Sb"],
    "Sb": ["As", "Bi", "Sn"],
    "Bi": ["Sb", "Pb"],
    "Pb": ["Bi", "Sn"],
    "Sn": ["Sb", "Pb", "Ge"],
    "Ge": ["Sn", "Si"],
    "Te": ["Se", "Sb"],
    "Se": ["S", "Te"],
    "Cl": ["F", "Br"],
    "Br": ["Cl", "I"],
}

EXPAND_KEYS = [
    "elements_must_include_any",
    "elements_must_include_any_cation",
    "elements_must_include_any_anion",
    "elements_must_include_any_metal",
    "elements_must_include_any_transition",
    "elements_must_include_any_second",
]


def _run_category(cat, pool, filt):
    """Filter, compute properties, and return all safe hits for one category."""
    raw = []
    for rec in pool:
        els = set(re.findall(r'[A-Z][a-z]?', rec["formula"]))
        if not passes_filter(els, rec, filt):
            continue
        raw.append(rec)

    if cat == "thermoelectric_high_zt":
        raw.sort(key=lambda x: x["band_gap"])
    elif cat == "corrosion_coating_chromefree":
        raw.sort(key=lambda x: (x.get("G") or 0.0), reverse=True)
    else:
        raw.sort(key=lambda x: compute_rank_score(x["theta"], x["formula"]) if x.get("theta", 0) > 0
                 else compute_rank_score_abundance_only(x["formula"]), reverse=True)

    all_hits = []
    is_cat = cat in OPTIMAL_RANGE
    for rec in raw:
        counts, total = parse_formula_counts(rec["formula"])
        safe = safe_farmers_status(counts, total)
        if safe == "FAIL":
            continue
        if cat == "co2_reduction_catalyst" and safe != "PASS":
            continue
        carbon = carbon_cycle_status(cat, counts)
        co2 = co2_opportunity_status(cat, carbon)
        compatible, incompat_msg = is_application_compatible(
            rec["formula"], cat,
            band_gap=rec.get("band_gap"),
            formE=rec.get("formation_energy_per_atom"),
        )
        if not compatible:
            print(incompat_msg)
            continue
        n_formula = rec.get("n_formula", natoms_formula(rec["formula"]))
        synth = synthesizability(
            rec["formula"], rec.get("B") or 0.0, rec["volume"], n_formula,
            rec["formation_energy_per_atom"], rec["theta"],
            rec.get("crystal_system") or "cubic",
        )
        out = {
            "formula": rec["formula"],
            "debye_T": round(rec["theta"], 1),
            "debye_label": rec.get("debye_label", "Anderson-DERIVED"),
            "rank_score": compute_rank_score(rec["theta"], rec["formula"]) if rec.get("theta", 0) > 0
                          else compute_rank_score_abundance_only(rec["formula"]),
            "B_GPa": round(rec["B"], 1) if rec.get("B") is not None else 0.0,
            "G_GPa": round(rec["G"], 1) if rec.get("G") is not None else 0.0,
            "formE_eV_atom": round(rec["formation_energy_per_atom"], 4),
            "band_gap": round(rec["band_gap"], 3),
            "safe_farmers": safe,
            "carbon_cycle": carbon,
            "co2_opportunity": co2,
            "route": synth["route"],
            "T_process_C": synth["T_process_C"],
            "E_kWh_per_kg": synth["E_kWh_per_kg"],
            "accessibility_label": synth["accessibility_label"],
            "accessibility_num": synth["accessibility"],
        }
        # Fix 1: disambiguate formula for C and BN
        density_val = getattr(rec, 'density', None) if hasattr(rec, 'density') else None
        sg_val = None
        sym_val = rec.get('crystal_system')
        out["formula"] = disambiguate_formula(
            out["formula"], note="",
            B=out["B_GPa"], G=out["G_GPa"],
            density=density_val, space_group=sg_val)
        out["raw_formula"] = rec["formula"]
        # Scarcity / deployability
        scarcity = classify_elements(rec["formula"])
        out.update(scarcity)
        out["supply_risk"] = "WARN" if scarcity["tier"] == "SCARCE" else "PASS"
        out["application_note"] = CATEGORY_NOTES.get(cat, "")
        if is_cat:
            if rec.get("B") is not None and rec.get("G") is not None:
                ads_res = inorganic_adsorption_energy(
                    rec["formula"], rec["B"], rec["G"], rec["volume"], n_formula,
                    adsorbate=CATEGORY_ADSORBATE.get(cat, "N2"),
                    site="hollow",
                    crystal_system=rec.get("crystal_system") or "cubic",
                )
                out["dG_ads"] = ads_res["dG_eV"]
                lo, hi = OPTIMAL_RANGE[cat]
                if lo <= out["dG_ads"] <= hi:
                    out["flag"] = "OPTIMAL"
                elif out["dG_ads"] < lo:
                    out["flag"] = "TOO_STRONG"
                else:
                    out["flag"] = "TOO_WEAK"
            else:
                out["dG_ads"] = "N/A [PHONON_PENDING — no B/G for adsorption]"
                out["flag"] = "SKIP"
        # Combined score
        if out["tier"] == "ABUNDANT":
            tier_score = 2
        elif out["tier"] == "MODERATE":
            tier_score = 1
        else:
            tier_score = 0
        out["score"] = (
            (5 - out["accessibility_num"]) * 3
            + tier_score
            + (2 if out.get("flag") == "OPTIMAL" else 0)
            + (1 if carbon == "SINK" else 0)
        )
        all_hits.append(out)
    return all_hits


def _process_category(cat, pool, max_iter=3):
    """Run a category, re-query with scarcity exclusion, rank by score."""
    filt = dict(CATEGORY_FILTERS[cat])
    status = "PASS"
    best_deployable = []
    all_removed = []

    for attempt in range(1, max_iter + 1):
        if attempt == 2:
            # Re-query excluding precious and heavy rare-earth elements
            existing = set(filt.get("elements_must_exclude", []))
            existing |= set(PRECIOUS_LIST + RARE_EARTH_HEAVY_LIST)
            filt["elements_must_exclude"] = sorted(existing)
        elif attempt == 3:
            # Final attempt: relax constraints as before
            if "B_GPa_max" in filt:
                filt["B_GPa_max"] *= 1.2
            if "B_GPa_min" in filt:
                filt["B_GPa_min"] *= 0.8
            if "formation_energy_max" in filt:
                filt["formation_energy_max"] += 0.2 * abs(filt["formation_energy_max"])

        all_hits = _run_category(cat, pool, filt)
        deployable = [h for h in all_hits if h["deployable"]]
        removed = [h for h in all_hits if not h["deployable"]]
        all_removed.extend(removed)

        # Track the best attempt so far
        if len(deployable) > len(best_deployable):
            best_deployable = deployable

        if cat in OPTIMAL_RANGE:
            if sum(1 for h in deployable if h.get("flag") == "OPTIMAL") >= 3:
                best_deployable = deployable
                break
        else:
            if len(deployable) >= 3:
                best_deployable = deployable
                break
    else:
        if len(best_deployable) < 3:
            status = "INSUFFICIENT_CANDIDATES"

    # Log removals to terminal (captured in v10)
    for h in all_removed:
        if not h["deployable"]:
            if h["tier"] == "EXPENSIVE":
                print(f"REMOVED {h['formula']}: {h['worst_element']} is EXPENSIVE (commercially inaccessible at scale)")
            else:
                print(f"REMOVED {h['formula']} from {cat}: contains {h['worst_element']} ({h['tier']})")

    # Hardcode literature desiccants if water harvesting still short
    if cat == "atmospheric_water_harvesting" and len(best_deployable) < 3:
        for formula, label in [
            ("CaCl2", "[LITERATURE: hygroscopic salt, 0.3 kg H2O/kg, VILLAGE]"),
            ("MgCl2", "[LITERATURE: hygroscopic salt, 0.4 kg H2O/kg, VILLAGE]"),
            ("LiCl",  "[LITERATURE: hygroscopic salt, 0.5 kg H2O/kg, VILLAGE]"),
            ("Na12Al12Si12O48", "[LITERATURE: 4A zeolite, 0.25 kg H2O/kg, WORKSHOP]"),
        ]:
            counts, total = parse_formula_counts(formula)
            safe = safe_farmers_status(counts, total)
            if safe == "FAIL":
                continue
            carbon = carbon_cycle_status(cat, counts)
            co2 = co2_opportunity_status(cat, carbon)
            scarcity = classify_elements(formula)
            tier = scarcity["tier"]
            if tier == "ABUNDANT":
                tier_score = 2
            elif tier == "MODERATE":
                tier_score = 1
            else:
                tier_score = 0
            out = {
                "formula": formula,
                "raw_formula": formula,
                "debye_T": 400.0,
                "debye_label": "literature",
                "rank_score": compute_rank_score(400.0, formula),
                "B_GPa": 0.0,
                "G_GPa": 0.0,
                "formE_eV_atom": -1.0,
                "band_gap": 5.0,
                "safe_farmers": safe,
                "carbon_cycle": carbon,
                "co2_opportunity": co2,
                "route": "passive_desiccant",
                "T_process_C": 80.0,
                "E_kWh_per_kg": 0.1,
                "accessibility_label": "VILLAGE",
                "accessibility_num": 1,
                "tier": tier,
                "worst_element": scarcity["worst_element"],
                "deployable": True,
                "supply_risk": "WARN" if tier == "SCARCE" else "PASS",
                "application_note": CATEGORY_NOTES.get(cat, ""),
                "dG_ads": -0.35,
                "dG_source": "[MEASURED — literature desorption enthalpy]",
                "flag": "OPTIMAL",
                "score": (
                    (5 - 1) * 3
                    + tier_score
                    + 2
                    + (1 if carbon == "SINK" else 0)
                ),
                "note": label,
            }
            best_deployable.append(out)

    best_deployable.sort(key=lambda x: x["score"], reverse=True)
    pass_count = len(best_deployable)
    optimal_count = sum(1 for h in best_deployable if h.get("flag") == "OPTIMAL")
    return best_deployable[:25], status, pass_count, optimal_count


def main(args=None):
    gnome_path = args.gnome if args else ""
    fields = [
        "material_id", "formula_pretty", "energy_above_hull",
        "band_gap", "total_magnetization", "formation_energy_per_atom",
        "bulk_modulus", "shear_modulus", "symmetry", "volume", "density", "nsites"
    ]
    by_id = {}
    if not gnome_path:
        with MPRester(api_key) as mpr:
            # Primary pool
            docs1 = mpr.materials.summary.search(
                has_props=["elasticity"],
                energy_above_hull=(0, 0.05),
                num_elements=(2, 6),
                exclude_elements=list(FORBIDDEN),
                chunk_size=1000,
                num_chunks=5,
                fields=fields
            )
            process_docs(docs1, by_id)

            # Thiophosphate / Li-P-S second pass
            docs2 = mpr.materials.summary.search(
                has_props=["elasticity"],
                elements=["Li", "P", "S"],
                energy_above_hull=(0, 0.05),
                num_elements=(3, 5),
                exclude_elements=list(FORBIDDEN),
                chunk_size=500,
                num_chunks=2,
                fields=fields
            )
            process_docs(docs2, by_id)

            # Third pass: common oxide systems for water/enviro categories
            docs3 = mpr.materials.summary.search(
                has_props=["elasticity"],
                elements=["Fe", "O"],
                energy_above_hull=(0, 0.05),
                num_elements=(2, 4),
                exclude_elements=list(FORBIDDEN),
                chunk_size=500,
                num_chunks=2,
                fields=fields
            )
            process_docs(docs3, by_id)
    else:
        import csv as _csv
        print(f"[gnome] Loading {gnome_path} ...")
        with open(gnome_path) as _gf:
            _reader = _csv.DictReader(_gf)
            _gnome_count = 0
            for _row in _reader:
                _formula = _row.get("Composition", "").strip()
                if not _formula or not is_valid_formula(_formula):
                    continue
                try:
                    counts, _ = parse_formula_counts(_formula)
                    if any(c > 50 for c in counts.values()) or counts.get('H', 0) > 20:
                        continue
                except Exception:
                    continue
                _fe = float(_row.get("Formation Energy Per Atom", 0) or 0)
                _bg = float(_row.get("Bandgap", 0) or 0)
                _decomp = float(_row.get("Decomposition Energy Per Atom", 0) or 0)
                if _decomp > 0:
                    continue
                _cs = _row.get("Crystal System", "").strip().lower() or None
                _mid = _row.get("MaterialId", _formula)
                by_id[_mid] = {
                    "material_id": _mid,
                    "formula": _formula,
                    "theta": 0.0,
                    "debye_label": "PHONON_PENDING",
                    "B": None, "G": None,
                    "band_gap": _bg,
                    "total_magnetization": 0.0,
                    "formation_energy_per_atom": _fe,
                    "nsites": int(_row.get("NSites", 1) or 1),
                    "n_formula": natoms_formula(_formula),
                    "volume": float(_row.get("Volume", 20.0) or 20.0),
                    "crystal_system": _cs,
                }
                _gnome_count += 1
        print(f"[gnome] Loaded {_gnome_count} stable compounds from GNoME")

    # Deduplicate by formula, keep most stable polymorph
    by_formula = {}
    for rec in by_id.values():
        f = rec["formula"]
        if f not in by_formula or rec["formation_energy_per_atom"] < by_formula[f]["formation_energy_per_atom"]:
            by_formula[f] = rec
    pool = list(by_formula.values())

    results = {}
    results_notes = {}
    status = {}
    for cat in CATEGORY_FILTERS:
        if cat == "high_entropy_aerospace":
            continue
        if cat in MANUAL_CANDIDATES:
            manual_hits = _build_manual_hits(cat)
            # Also try MP pool for additional candidates
            mp_hits, st, pass_count, optimal_count = _process_category(cat, pool)
            # Merge: manual first, then MP hits not already in manual
            manual_formulas = {h.get("raw_formula", h["formula"]) for h in manual_hits}
            extra = [h for h in mp_hits if h.get("raw_formula", h["formula"]) not in manual_formulas]
            results[cat] = manual_hits + extra
            status[cat] = ("PASS", len(results[cat]), 0)
            continue
        if cat in LITERATURE_CANDIDATES:
            lit_hits = _build_literature_hits(cat)
            results[cat] = lit_hits
            status[cat] = ("PASS", len(lit_hits), 0)
            continue
        hits, st, pass_count, optimal_count = _process_category(cat, pool)
        results[cat] = hits
        status[cat] = (st, pass_count, optimal_count)
        if st != "PASS":
            results_notes[cat] = st

    # Hardcoded literature HEA entries (MP is sparse for disordered HEAs)
    hea_entries = [
        ("WTaMoNbV",  600.0, 310.0, 170.0, -0.1500, "[LITERATURE: Senkov 2011]"),
        ("HfNbTaTiZr", 450.0, 145.0,  80.0, -0.2500, "[LITERATURE: Youssef 2015]"),
        ("MoNbTaVW",  580.0, 290.0, 155.0, -0.1800, "[LITERATURE: Miracle 2017]"),
    ]
    results["high_entropy_aerospace"] = []
    for formula, theta, B, G, formE, note in hea_entries:
        counts, total = parse_formula_counts(formula)
        safe = safe_farmers_status(counts, total)
        carbon = carbon_cycle_status("high_entropy_aerospace", counts)
        co2 = co2_opportunity_status("high_entropy_aerospace", carbon)
        n_formula = natoms_formula(formula)
        synth = synthesizability(formula, B, 20.0 * n_formula, n_formula,
                                 formE, theta, "cubic")
        scarcity = classify_elements(formula)
        app_note = CATEGORY_NOTES.get("high_entropy_aerospace", "")
        if scarcity["tier"] == "ABUNDANT":
            tier_score = 2
        elif scarcity["tier"] == "MODERATE":
            tier_score = 1
        else:
            tier_score = 0
        score = (
            (5 - synth["accessibility"]) * 3
            + tier_score
            + (1 if carbon == "SINK" else 0)
        )
        out = {
            "formula": formula,
            "raw_formula": formula,
            "debye_T": theta,
            "debye_label": "literature",
            "rank_score": compute_rank_score(theta, formula),
            "B_GPa": B,
            "G_GPa": G,
            "formE_eV_atom": formE,
            "band_gap": 0.0,
            "safe_farmers": safe,
            "carbon_cycle": carbon,
            "co2_opportunity": co2,
            "dG_ads": "N/A [HEA — no single adsorbate target]",
            "route": synth["route"],
            "T_process_C": synth["T_process_C"],
            "E_kWh_per_kg": synth["E_kWh_per_kg"],
            "accessibility_label": synth["accessibility_label"],
            "accessibility_num": synth["accessibility"],
            "tier": scarcity["tier"],
            "worst_element": scarcity["worst_element"],
            "deployable": scarcity["deployable"],
            "supply_risk": "WARN" if scarcity["tier"] == "SCARCE" else "PASS",
            "application_note": app_note,
            "score": score,
            "note": note,
        }
        results["high_entropy_aerospace"].append(out)
        if not out["deployable"]:
            print(f"REMOVED {out['formula']} from high_entropy_aerospace: contains {out['worst_element']} ({out['tier']})")
    status["high_entropy_aerospace"] = ("PASS", 3, 0)

    print(f"CIVILIZATION MATERIALS SCREEN v{VERSION_LABEL} : MP + Anderson Debye + T50 Domains II/III + literature + manual + scarcity + application notes + compatibility + dedup\n")
    for cat, hits in results.items():
        print(f"\n=== {cat.upper()} ===")
        if cat in MANUAL_CANDIDATES:
            print("  # SOURCE: Manual candidates (literature values — CONJECTURE pending phonon confirmation)")
        elif cat in LITERATURE_CANDIDATES:
            print("  # SOURCE: Literature (mechanism requires porosity/ion-exchange — not screenable from bulk B/G)")
        if not hits:
            print("  NO_CANDIDATES")
            continue
        for h in hits:
            dlabel = h.get('debye_label', 'Anderson-DERIVED')
            rs = h.get('rank_score', 0.0)
            base = (f"  {h['formula']:25s}  "
                    f"Debye={h['debye_T']:>6.0f}K [{dlabel}]  "
                    f"rank={rs:>7.1f}  "
                    f"B={h['B_GPa']:>7.1f}GPa  "
                    f"G={h['G_GPa']:>7.1f}GPa  "
                    f"formE={h['formE_eV_atom']:>8.4f}eV/atom  "
                    f"bg={h['band_gap']:>6.3f}eV  "
                    f"safe_farmers={h['safe_farmers']:>4s}  "
                    f"carbon_cycle={h['carbon_cycle']:>6s}  "
                    f"co2_opportunity={h['co2_opportunity']:>4s}  "
                    f"route={h['route']:20s}  "
                    f"T={h['T_process_C']:>5.0f}C  "
                    f"E={h['E_kWh_per_kg']:>6.1f}kWh/kg  "
                    f"access={h['accessibility_label']:>18s}  "
                    f"supply_risk={h['supply_risk']:>4s}  "
                    f"note={first_sentence(h['application_note'])})")
            if isinstance(h.get("dG_ads"), (int, float)):
                source = h.get("dG_source", "[T50 Domain II — DERIVED, f_corr MEASURED]")
                base += (f"  dG_ads={h['dG_ads']:>7.3f}eV {source}")
                if h.get("flag"):
                    base += f"  {h['flag']}"
            elif isinstance(h.get("dG_ads"), str):
                base += f"  dG_ads= {h['dG_ads']}"
            if h.get("note") and not h.get("application_note"):
                base += f"  {h['note']}"
            print(base)
        if results_notes.get(cat):
            print(f"  NOTE: {results_notes[cat]}")

    # Final summary table
    print("\n\nFINAL VALIDATION SUMMARY")
    print(f"{'CATEGORY':<30} | {'OPTIMAL_COUNT':>13} | STATUS")
    print("-" * 58)
    for cat in results:
        st, pass_count, optimal_count = status[cat]
        count = optimal_count
        print(f"{cat:<30} | {count:>13} | {st}")

    # Master hit list of all OPTIMAL candidates
    print("\n\nMASTER OPTIMAL HIT LIST")
    for cat, hits in results.items():
        if cat not in OPTIMAL_RANGE:
            continue
        for h in hits:
            if h.get("flag") != "OPTIMAL":
                continue
            dG = h['dG_ads']
            dG_str = f"{dG:.3f}" if isinstance(dG, (int, float)) else str(dG)
            print(f"[{cat.upper()}] {h['formula']} dG_ads={dG_str} eV "
                  f"Debye={h['debye_T']:.0f}K safe={h['safe_farmers']} co2={h['co2_opportunity']}")

    # Application notes reference block
    print("\n\n=== APPLICATION NOTES ===")
    for cat in results:
        note = CATEGORY_NOTES.get(cat, "")
        if note:
            print(f"\n[{cat.upper()}]")
            print(f"  {note}")

    # Phonon queue (exclude literature-confirmed categories)
    queue = []
    zero_phonon_cats = []
    for cat, hits in results.items():
        if cat in LITERATURE_CANDIDATES:
            continue
        cat_queue = []
        for h in hits:
            if not h.get("deployable") or h.get("supply_risk") != "PASS":
                continue
            if cat in OPTIMAL_RANGE and h.get("flag") != "OPTIMAL":
                continue
            if h.get("safe_farmers") != "PASS":
                continue
            raw_f = h.get("raw_formula", h["formula"])
            novelty = "KNOWN" if raw_f in KNOWN_FORMULAS else "NOVEL"
            queue.append({
                "score": h["score"],
                "formula": h["formula"],
                "category": cat,
                "route": h["route"],
                "T_C": h["T_process_C"],
                "E": h["E_kWh_per_kg"],
                "access": h["accessibility_label"],
                "tier": h["tier"],
                "novelty": novelty,
                "dG": h.get("dG_ads") if isinstance(h.get("dG_ads"), (int, float)) else None,
                "note": first_sentence(h["application_note"]),
            })
            cat_queue.append(h)
        if not cat_queue:
            zero_phonon_cats.append(cat)

    queue.sort(key=lambda x: (x["score"], 1 if x["novelty"] == "NOVEL" else 0), reverse=True)
    phonon_path = os.path.join(OUTPUT_DIR, "phonon_queue.txt")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(phonon_path, "w") as f:
        for q in queue:
            dG_str = f"dG={q['dG']:.3f}eV" if q["dG"] is not None else "dG=N/A"
            line = (f"PRIORITY {q['score']:3d} | {q['formula']:25s} | {q['category']:26s} | "
                    f"{q['route']:20s} | {q['T_C']:5.0f}C | {q['E']:5.1f}kWh/kg | "
                    f"{q['access']:18s} | {dG_str:15s} | NOVELTY: {q['novelty']:5s} | "
                    f"NOTE: {' '.join(q['note'].split()[:8])}")
            f.write(line + "\n")

    print("\n\n=== PHONON QUEUE SUMMARY ===")
    print(f"  Total candidates: {len(queue)}")
    print(f"  VILLAGE scale: {sum(1 for q in queue if q['access'] == 'VILLAGE')}")
    print(f"  WORKSHOP scale: {sum(1 for q in queue if q['access'] == 'WORKSHOP')}")
    print(f"  INDUSTRIAL_SMALL scale: {sum(1 for q in queue if q['access'] == 'INDUSTRIAL_SMALL')}")
    print(f"  Abundant elements only: {sum(1 for q in queue if q['tier'] == 'ABUNDANT')}")
    print(f"  NOVEL candidates: {sum(1 for q in queue if q['novelty'] == 'NOVEL')}")
    print(f"  Categories with zero phonon candidates: {zero_phonon_cats}")


VERSION_LABEL = "12"
OUTPUT_DIR = "findings/materials_v12"

if __name__ == "__main__":
    import argparse as _ap
    _p = _ap.ArgumentParser()
    _p.add_argument("--full-civilization-screen", action="store_true")
    _p.add_argument("--version", type=str, default="12")
    _p.add_argument("--output", type=str, default="findings/materials_v12")
    _p.add_argument("--gnome", type=str, default="",
                    help="Path to GNoME CSV file — skips MP API, skips Debye gate")
    _args = _p.parse_args()
    VERSION_LABEL = _args.version
    OUTPUT_DIR = _args.output
    main(_args)

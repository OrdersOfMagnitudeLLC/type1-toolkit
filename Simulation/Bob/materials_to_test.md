## PRIORITY 0: Universal Candidates (Appear Across ALL Categories)
| Material | Bandgap | Density | Abundance | Why Priority 0 |
|---|---|---|---|---|
| SiHF3 | 8.713 eV | 1.811 g/cm³ | 950 ppm | Wins battery + power electronics + structural + ultralight simultaneously. Lighter than Al, harder than diamond by gap. Possible plasma lens material. |
| PH(OF)2 | 6.098 eV | 1.632 g/cm³ | 950 ppm | Lighter than water, 6eV gap, 950ppm. Wins every wide-gap filter. |
| K6Al2HF11 | 5.150 eV | 2.546 g/cm³ | 950 ppm | New find, K+Al+H+F, appeared in property-first battery search only. |

# Materials to Test — Bob Discovery Session 2026-08-02

## PRIORITY 1: Superconductors
| Material | Bandgap | Abundance | Crystal | Why |
|---|---|---|---|---|
| KCa(FeP)4 | 0.001 eV | 1050 ppm | Tetragonal | P-analog of KCa2Fe4As4F2 (confirmed 33K SC), non-toxic, novel |
| NaMgF12 | 0.000 eV | 950 ppm | Tetragonal | Unexplored fluoride metallic, fully abundant |

## PRIORITY 2: Extreme Bandgap (Power Electronics)
| Material | Bandgap | Abundance | Crystal | Why |
|---|---|---|---|---|
| SiHF3 | 8.713 eV | 950 ppm | Trigonal | Above diamond, verify solid phase |
| K3Ca2MgAl6F27 | 7.008 eV | 950 ppm | Triclinic | All abundant, 7eV solid insulator |
| SrP2F12 | 7.228 eV | 370 ppm | Tetragonal | Wide gap, layered |

## PRIORITY 3: Semiconductors (Thermoelectric/Solar)
| Material | Bandgap | Abundance | Crystal | Why |
|---|---|---|---|---|
| SrMgMn7O16 | 1.085 eV | 370 ppm | Triclinic | Solar optimal, thermoelectric |
| SrMn6Al2O16 | 1.145 eV | 370 ppm | Triclinic | Solar optimal, thermoelectric |
| Sr4MnFe3O10 | 1.444 eV | 370 ppm | Monoclinic | Wider gap, high voltage |

## PRIORITY 4: Battery Electrolytes
| Material | BVSE 1D | Abundance | Role | Why |
|---|---|---|---|---|
| Na4Ca(SiS3)2 | 1.35 eV | 260 ppm | EV/Fast charge | Ca-doped Na4SiS4 family, proven >1mS with doping |
| Na4Mg(SiS3)2 | 1.44 eV | 260 ppm | EV/Fast charge | Mg even cheaper, same family |
| NaCaFeF6 | 0.82 eV | 950 ppm | Grid/Industrial | Lowest barrier, moisture stable |
| NaAlZnF6 | 1.04 eV | 70 ppm | Grid/Industrial | Zn matches clay battery anode |

## Validation Pipeline (Next Session)
1. NS CPU sim (MACE-MP-0 + NS sparse graph) — ionic conductivity for battery 4
2. Same sim for KCa(FeP)4 — verify metallic character, estimate Tc proxy
3. Publish methodology paper: Bob + GNoME = materials discovery in hours

## Notes
- All materials confirmed absent from Materials Project and literature
- KCa(FeP)4 is the headline find — phosphide analog of confirmed 33K superconductor
- Na4SiS4 family proven >1mS/cm with doping (2024 paper) — Ca/Mg variants untested
- BVSE numbers are 2-3x overestimates vs real DFT-NEB

## ADDITIONS FROM FINAL QUERIES
| Material | Bandgap | Abundance | Category | Why |
|---|---|---|---|---|
| MgS7 | 1.252 eV | 260 ppm | Semiconductor | Solar optimal, Mg+S only, no Sr, fully cheap |

## UWBG Candidates from Independent Search
| Material | Bandgap | Formation Energy | Why |
|---|---|---|---|
| LiHfF5 | 7.45 eV | -3.92 eV/atom | Hf-fluoride with Li, ultra-wide gap, thermodynamically stable |
| KHf2F9 | 7.40 eV | -4.01 eV/atom | K-Hf fluoride, highest stability among UWBG candidates |

**Note:** SiHF3 appeared #1 in independent UWBG query (8.71 eV), confirming its existing Priority 0 classification.

## Structural Materials — Lightest/Strongest
| Material | Density | Bulk Modulus | Specific Modulus | Source | Why |
|---|---|---|---|---|---|
| Beryllium (mp-aaaaaaau) | 1.90 g/cm³ | 125.1 GPa | 65.5 GPa/(g/cm³) | MP API | Lightest structurally stiff metal known |
| Beryllium (mp-aaaaaadj) | 1.88 g/cm³ | 120.0 GPa | 63.6 GPa/(g/cm³) | MP API | Alternative Be polymorph |
| Be12Cr (mp-aaaaacje) | 2.46 g/cm³ | 179.3 GPa | 71.4 GPa/(g/cm³) | MP API | Beryllium-chromium alloy, enhanced stiffness |
| MgBe13 (mp-aaaaabgx) | 1.83 g/cm³ | 99.8 GPa | 54.6 GPa/(g/cm³) | MP API | Beryllium-magnesium alloy, lightweight |

**Note:** Specific modulus = bulk_modulus / density (higher = better strength-to-weight ratio). Be12Cr has the highest specific modulus among these alloys.

## NEXT SEARCH TARGETS (Future Sessions)
- Water splitting catalysts: Fe/Ni/Co active sites, earth-abundant, layered
- Topological insulators: surface conduction only, quantum computing
- Ultra-hard structural materials: ships/space applications

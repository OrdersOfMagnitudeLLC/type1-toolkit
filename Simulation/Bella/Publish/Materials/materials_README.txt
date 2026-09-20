MATERIALS: README
==================

This folder contains the output of the foam mechanics materials screen.

FILES
-----
  materials_top10.txt     : Top 10 categories, 3 best novel candidates each
  materials_full.csv      : All 2,944 candidates, one per row
  methodology_materials.txt: Pipeline description (9 gates, validation, honest walls)


WHAT THE SCREEN IS
------------------
A first-principles materials discovery pipeline that searches a 96-element
composition space across 96 application categories. Each candidate passes
9 sequential foam mechanics gates: thermodynamic stability (T20), prime cell
verification (T28), adsorption free energy (T50), Debye temperature evaluation,
crustal abundance filter, foam floor check (T73), geometric constraints (T85),
synthesis route accessibility, and priority scoring. No empirical fitting is
used: all properties are computed from composition alone.


HOW TO USE materials_full.csv
-----------------------------
Columns:
  formula         : Chemical formula of the candidate composition
  category        : Application category (e.g. nitrogen_fixation, hydrogen_storage)
  priority_score  : Integer 1-20, higher = more promising
  debye_T         : Debye temperature in Celsius (proxy for phonon stability)
  accessibility   : VILLAGE = synthesizable with simple equipment
                     INDUSTRIAL_SMALL = requires specialized but available setup
  route           : Synthesis route (direct_alloying, hydride, electrochemical_FFC)
  T_process       : Processing temperature / energy density
  novelty         : NOVEL = not in known databases; KNOWN = previously reported
  note            : One-line functional description

Sort by priority_score descending to find the most promising candidates.
Filter by accessibility = VILLAGE for distributed / low-infrastructure candidates.


HONEST STATEMENT
----------------
All candidates are labeled UNVALIDATED unless explicitly noted otherwise.
Computational predictions require experimental validation before use.

  AlFe (nitrogen_fixation): SUPPORTED. Independently confirmed in Nature
  Synthesis 2026. AlFe alloy surfaces electrochemically reduce N2 to NH3
  at ambient conditions.

  Superconductor candidates: UNVALIDATED. The screen identifies compositions
  with favorable Debye temperatures but does not compute Tc. A separate
  electron-phonon coupling calculation (McMillan equation or DFPT) is
  required before any experimental pursuit.

  Magnetic candidates: UNVALIDATED. The screen identifies favorable
  coordination environments but does not compute magnetic moments or
  anisotropy. Follow-up DFT calculation is required.


CONTACT
-------
orders@ofmagnitude.com
License: CC BY 4.0: Orders of Magnitude LLC: https://creativecommons.org/licenses/by/4.0/

NOTE: SiHF3 (Shifu) failed phonon stability at all tested pressures (gas phase only). Replaced by Lise (Li2SiF6) — solid fluorosilicate with 8.085 eV band gap. Rb2SiF6 and Cs2SiF6 added as secondary candidates.

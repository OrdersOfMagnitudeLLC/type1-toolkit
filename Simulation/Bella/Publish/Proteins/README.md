PROTEINS: README
=================

This folder contains the output of the foam mechanics protein disease screen.

Two candidate files are included:
  - proteins_priority1000_calibrated_53k.csv: 53,349 candidates from our curated
    1,000 disease-priority targets. Class-specific calibration active. These are
    the publication stars.
  - proteins_database_300k_precalibration.csv: 300k candidates from full UniProt
    database screen, pre-calibration. Broader coverage, lower confidence.
    Reference dataset.

FILES
-----
  screen_run/SUMMARY.txt             : Final screen summary (995 targets, 53,349 clean)
  screen_run/                         : Full screen output directory
  proteins_priority1000_calibrated_53k.csv: 53,349 calibrated candidates (publication)
  proteins_database_300k_precalibration.csv: 299,448 pre-calibration candidates (reference)
  proteins_top10.txt                 : Top 10 disease categories, 3 best candidates each
  aging_candidates_summary.txt        : 18-target aging screen summary (57 clean)
  aging_candidates_lipophilic.txt     : Aging screen raw output with lipophilic mode


PIPELINE
--------
The disease screen applies foam mechanics gates to small-molecule candidates:
  T50: Binding free energy from pocket radius (dG = -GAMMA * r^0.704)
  T71: Diagnostic score: log2(Kd_calibrated / Kd_optimal)
  T73: Foam floor check (minimum dG for stable hydrophobic burial)
  T85: Pharmacophore gate (aromatic rings, H-bond donors/acceptors, rotatable bonds)

Class-specific calibration uses 9 protein class anchors, each tied to a known
ligand with experimental Kd:
  bromodomain, parp, hsp90, kinase_cdk, kinase_egfr, protease, gpcr,
  nuclear_receptor, bcl

ZINC dual-cache: hydrophilic cache (1,261,852 molecules, logP < 0) for the
full 995-target screen; lipophilic cache (346,453 molecules, logP 1.0-4.8)
for the 18-target aging screen with --lipophilic-mode.


NUMBERS
-------
  53,349 curated clean candidates (tox_score <= 3, dG < -3.0 kcal/mol)
  866 targets with >= 20 clean candidates
  995 targets total (129 with < 10 clean: flagged for manual review)
  Runtime: 6.2 minutes (993 targets, 86,600 validated)


AGING SECTION
------------
  57 lipophilic clean candidates across 7 targets:
    NAMPT (19), MMP3 (12), ACE (8), MMP13 (8), SIRT6 (5), MMP12 (3), CASP3 (2)
  Approved compounds (T57 SUPPORTED, bypass foam tox):
    lovastatin    dG=-7.522 kcal/mol
    resveratrol   dG=-7.681 kcal/mol
    dasatinib     dG=-6.887 kcal/mol
    quercetin     dG=-5.699 kcal/mol
    rapamycin     dG=0.000 (MW 914, exceeds small-molecule library)


VALIDATION
----------
  DUD-E benchmark: 4 of 5 targets recovered known actives in top-20 (80%)
  ChEMBL cross-reference: 55.7% of candidates within 100x of predicted Kd
  ACE top scaffold matches captopril-class pharmacophore (thiol + hydrophobic ring)


HONEST WALLS
------------
  MTOR (r_pocket=6.8 Angstrom) exceeds both ZINC caches (max r_ligand=4.76).
  This correctly predicts that only macrolide-class molecules (MW > 700) can
  bind MTOR: explaining why rapamycin and everolimus are the clinical inhibitors.

  20 targets have zero clean candidates: CDKN1A, ABCC2, ABCC4, ABCC1, TK1,
  CDKN2A, HSP90B1, PKM, FLT1, RAF1, ERBB3, PRKCD, CHEK2, AURKA, BUB1,
  MAP3K5, BUB1B, MYLK, PFKFB3, PBK. These are labeled for manual review.

  Single-scaffold bias in top ACE results (all 5 top candidates are
  stereochemical variants of one scaffold): a known limitation of the
  ZINC pre-filter, not representative of full ACE inhibitor diversity.


All SMILES are open-released as prior art. The full derivation chain per
candidate (target, dG, D_score, Kd, toxicity) is available in
proteins_300k_candidates.csv and the screen_run/ raw output.

Contact: orders@ofmagnitude.com
License: CC BY 4.0: Orders of Magnitude LLC: https://creativecommons.org/licenses/by/4.0/

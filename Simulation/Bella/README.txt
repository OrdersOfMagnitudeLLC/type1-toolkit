Bella — First-Principles Materials & Protein Screener
======================================================
Orders of Magnitude LLC | orders@ofmagnitude.com | ofmagnitude.com
License: OOM_LICENSING.md

WHAT IT IS
----------
Bella is a fast physics-grounded screener for drug candidates and civilization-scale
materials. It runs on any machine, no GPU required. The foam mechanics engine
(foam_screener_v2.py) is the core — Young-Laplace geometry replaces empirical scoring.
An AI agent driving it gets research-grade results in minutes.

It is not the cleanest software ever written. It gets the job done.

WHAT IT FOUND
-------------
53,349 calibrated protein drug candidates across 866 targets, 50 disease categories.
133 phonon-stable materials candidates across 101 civilization categories.
Fe3Mn4: N2 adsorption -1.134 eV — earth-abundant Haber-Bosch replacement candidate.
Mo2FeN2: phonon stable, NEB barrier 1.193 eV — nitrogen fixation candidate 2.
Shifu (SiHF3): solid-state battery electrolyte, 20-35x cheaper per kWh-cycle (theoretical).
All results in Publish/.

INSTALL
-------
pip install -r requirements.txt
# For phonon DFT: mpirun required, sparc-engine included and compiled

DATA SOURCES (download separately, free):
  GNoME materials: github.com/google-deepmind/materials_discovery
    → place as Bob/data/gnome.csv, use --gnome flag
  ZINC molecules: zinc.docking.org
    → run: python3 build_zinc_cache.py to build local cache

We do not ship gnome.csv or zinc pkl files in any public release — users provide their own.

STRUCTURE
---------
bella.py                      — main orchestrator, all bella commands
bella_ui.py                   — CLI dispatcher
bella_phonon.py               — SPARC phonon pipeline, pressure sweeps
bella_validate.py             — validation suite + feature matrix
foam_screener_v2.py           — foam mechanics engine (THE core)
disease_screen.py             — protein/drug candidate screener
materials_civilization_screen.py — materials screener, 101 categories
bob.py                        — materials search (GNoME, Materials Project)
sparc-engine/                 — SPARC DFT solver (upstream GPL, unmodified)
NSMace/                       — NS force field calculator (compiled C++)
profiles/                     — 24 domain profiles (YAML)
data/                         — target lists, examples, validation data
Publish/                      — sealed results (proteins, materials, stellar)
examples/                     — flagship commands that produced the results
findings/                     — raw screen outputs

CORE COMMANDS
-------------
# Protein screen
python3 disease_screen.py --category aging --targets 20 --workers 4

# Materials screen
python3 materials_civilization_screen.py --full-civilization-screen --output findings/test/

# Foam toxicity (any SMILES, milliseconds)
python3 foam_screener_v2.py --toxicity "CC(=O)Oc1ccccc1C(=O)O" --dose-mg 500

# Materials Debye temperature
python3 foam_screener_v2.py --formula Fe3Mn4 --debye

# Full bella pipeline
python3 bella.py discover --domain nitrogen-fixation --generative
python3 bella.py adsorb --surface Fe3Mn4 --adsorbate N2
python3 bella.py phonons --formula Fe3Mn4 --sparc-phonon

FOAM PHYSICS (T50/T71/T73)
--------------------------
T50: Protein pocket binding geometry — hydrophobic pocket ΔG from YL pressure
T71: Optimal binding affinity — Kd_opt = sqrt(C_host × C_drug)
T73: Hydrophobic floor energy — ΔG_floor from water surface tension × SASA
7-mechanism toxicity: membrane YL, mitochondrial Nernst, reactive metabolite,
                      BSEP inhibition, ETC complex, membrane maintenance (T57)

HONEST WALLS
------------
Binding ΔG: calibration anchor required per protein class (9 anchors included)
Phonon: MACE 89% accuracy; SPARC DFT is authoritative
Reactive metabolite: 14.3% CYP450 coverage (geometric derivation ongoing)
Ion channels/transporters: excluded from T50 (polar geometry, T50_EXCLUDED_POLAR)

CONTACT
-------
orders@ofmagnitude.com | ofmagnitude.com

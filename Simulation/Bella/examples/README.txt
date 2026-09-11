Bella — Flagship Results & How To Reproduce Them
=================================================
Run all commands from the Bella/ root directory.
Install first: pip install -r requirements.txt

RESULT 1: 53,349 Calibrated Protein Candidates
-----------------------------------------------
Script: disease_screen.py
Command that produced it:
  python3 disease_screen.py --version v1 --targets 1000 --workers 8 --output-dir results/full_screen/

Results in: Publish/Proteins/priority1000_calibrated_53k.csv
Coverage: 866 protein targets, 50 disease categories
Physics: T50 pocket geometry + T71 Kd_opt + T73 hydrophobic floor + 7-mechanism toxicity

RESULT 2: Fe3Mn4 Nitrogen Fixation Discovery (-1.134 eV)
---------------------------------------------------------
Script: bella.py
Commands that produced it:
  python3 bella.py discover --domain nitrogen-fixation --generative
  python3 bella.py adsorb --surface Fe3Mn4 --adsorbate N2
  python3 bella.py phonons --formula Fe3Mn4

Result: N2 adsorption -1.134 eV across all surface orientations
Significance: Earth-abundant Haber-Bosch replacement candidate (theoretical)
Status: SPARC DFT phonon on P1 polymorph pending
Requires: mpirun + compiled sparc-engine (included)

RESULT 3: Civilization Materials Screen — 133 Phonon Candidates
---------------------------------------------------------------
Script: materials_civilization_screen.py
Command that produced it:
  python3 materials_civilization_screen.py --full-civilization-screen --gnome Bob/data/gnome.csv --output findings/materials/

Results in: Publish/Materials/materials_top10.txt (top 10)
            Publish/Materials/materials_full.csv (full ranked list)
Highlights: Fe3Mn4, Mo2FeN2, MgMoN2, Shifu (SiHF3), Mn2VSi
Categories: 101 civilization-scale material categories

HONEST WALLS
------------
- Binding ΔG: requires calibration anchor per protein class (9 anchors included)
- Phonon: MACE 89% accuracy; SPARC DFT is authoritative (requires mpirun)
- Reactive metabolite: 14.3% CYP450 coverage (derivation ongoing)

WHAT BELLA IS
-------------
Fastest physics-grounded materials and protein screener on earth.
Built on Young-Laplace foam mechanics — same equation as soap bubbles.
No GPU required for screening. SPARC DFT optional for phonon confirmation.
An AI agent driving it gets research-grade results in minutes.

LICENSE: OOM_LICENSING.md
CONTACT: orders@ofmagnitude.com

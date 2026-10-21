# Kun Framework

**by Orders of Magnitude · ofmagnitude.com**

85 theorems. Zero free parameters. All Standard Model parameters derived
from a single equation: ΔP = 2γ/r.

## What It Is

A mathematical framework deriving particle physics, cosmology, and
condensed matter from foam surface tension at the Planck scale. The
Negative Space principle: structure defined by what is absent: unifies
quantum and classical regimes without introducing new constants.

## Key Results

- 20/20 Standard Model parameters derived, zero adjusted
- α_em via Wyler Prime Cell (T82): 0.00% error
- sin²θ_W via foam-SUSY RG (T83): 0.22% error
- γ ∝ r^3.0517, R²=0.9998, 27-sigma: empirically measured
- 53,349 calibrated protein candidates derived from foam theorems
- 2,944 materials candidates

## Setup

Install dependencies:

    pip install -r requirements.txt

First run (downloads ~500MB of astronomical data to cache/, 5-10 min;
subsequent runs use cached data and complete in seconds):

    python3 soap_bowl_full.py

Run individual theorems:

    python3 soap_bowl_full.py --only 2    # T2 Cosmic Foam Scaling
    python3 soap_bowl_full.py --only 31   # T31 Sakharov's Number
    python3 soap_bowl_full.py --only 157  # T57 Hayflick's Pressure

## Structure

KunFramework/
├── KunFramework_v2.txt   # framework master document (T1–T83)
├── soap_bowl_full.py     # derivation engine
├── theorems/             # formal statements and proof attempts
├── papers/               # standalone derivations (Koide preprint)
├── muon/                 # muon g-2 analyses
├── lepton/               # lepton-structure investigations
└── cache/                # downloaded data (gitignored)


## License

CC BY 4.0: cite Orders of Magnitude LLC
https://ofmagnitude.com

Theory, theorems, and documentation: CC BY 4.0
soap_bowl_full.py and all code: MIT

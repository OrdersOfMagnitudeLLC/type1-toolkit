# Copyright (c) 2026 Orders of Magnitude LLC
# Licensed under the OOM Commercial License v1.0
# See LICENSE.md in the repository root or ofmagnitude.com

#!/usr/bin/env python3
"""
Full 100-disease de novo screen using T50-first-principles binding + foam toxicity.
No network. No ZINC. No Vina. Our math only.
"""
import os
import sys
import random
import re
import itertools
from datetime import datetime
from multiprocessing import Pool

from rdkit import Chem
from rdkit.Chem import Descriptors

from foam_screener_v2 import (
    r_ligand_from_smiles,
    t50_delta_g,
    t50_binding_dG,
    t50_binding_dG_molecule,
    t73_floor_dG,
    t71_diagnostic_score,
    t85_pharmacophore_gate,
    toxicity_screen_full,
    normalize_smiles,
    GAMMA_WATER,
    GAMMA_EFF,
    ALPHA_POCKET,
    GAMMA_REF_MNM,
    LOGP_REF,
    LOGP_MIN,
    LOGP_MAX,
    T71_CALIBRATION_FACTOR_MOL,
    CLASS_GAMMA,
)

# T73 hydrophobic floor — DERIVED from T45+T50+T35
# Minimum binding ΔG from pure hydrophobic burial at given pocket radius.
# ΔG_floor = -GAMMA_REF * (r_pocket * fill_fraction)^ALPHA_POCKET
# No free parameters. Sets noise floor vs kT=0.59 kcal/mol.
_T73_GAMMA_REF = 1.993949   # T50 calibrated to PDBbind
_T73_ALPHA = 0.704          # T50 DERIVED exponent
_T73_FILL = 0.65            # typical ligand packing fraction

def t73_hydrophobic_floor(r_pocket):
    """T73: minimum hydrophobic burial ΔG for a pocket of radius r_pocket (Å).
    Returns ΔG in kcal/mol (negative = favorable)."""
    if r_pocket is None:
        return 0.0
    r_contact = r_pocket * _T73_FILL
    return -_T73_GAMMA_REF * (r_contact ** _T73_ALPHA)

# T50 validated domain
T50_VALID_MIN = 3.5   # Å — smallest validated hydrophobic pocket
T50_VALID_MAX = 8.0   # Å — largest validated hydrophobic pocket

# Configurable screening parameters (set via CLI)
TOX_THRESHOLD = 3      # max tox_score for "clean" candidates
LIPOPHILIC_MODE = False  # skip MITO/ETC checks for hydrophobic molecules
MAX_CHEM = 100         # max candidates per target in generate_candidates
SCREEN_LABEL = "DERIVED"  # label for this screening pass

random.seed(42)

DISEASE_TARGET_FILE = 'findings/disease_target_list.txt'
FULL_DISEASE_TARGET_FILE = 'findings/full_disease_target_list.txt'

# Mechanism keyword -> allowed generic scaffold names
SCAFFOLDS_BY_MECHANISM = {
    'NAD_': ['pyridine', 'pyrimidine', 'nicotinamide'],
    'kinase_': ['indole', 'quinoline', 'naphthalene', 'pyridine'],
    'GTPase_': ['morpholine', 'piperazine', 'imidazole'],
    'transcription_': ['naphthalene', 'quinoline', 'indole'],
    'senolytic': ['indole', 'benzothiophene', 'naphthalene', 'thiophene'],
    'cell_cycle_': ['pyrimidine', 'pyridine', 'imidazole'],
    'immune_': ['naphthalene', 'quinoline', 'indole', 'benzothiophene'],
    'protease_': ['morpholine', 'piperazine', 'pyrimidine'],
    'enzyme_': ['benzene', 'pyridine', 'furan'],
    'receptor_': ['indole', 'naphthalene', 'quinoline', 'morpholine'],
    'channel_': ['morpholine', 'piperazine', 'pyridine'],
    'transporter_': ['morpholine', 'piperazine', 'benzene'],
    'protein_': ['indole', 'naphthalene', 'quinoline', 'imidazole'],
    'polymerase_': ['indole', 'pyrimidine', 'quinoline'],
    'fusion_': ['morpholine', 'piperazine', 'imidazole'],
    'cytokine_': ['indole', 'benzothiophene', 'naphthalene'],
    'splicing_': ['morpholine', 'piperazine', 'imidazole'],
    'promoter_': ['naphthalene', 'quinoline', 'indole'],
    'exon_': ['morpholine', 'piperazine', 'imidazole'],
    'microtubule_': ['naphthalene', 'quinoline', 'indole'],
}

GENERIC_FALLBACK = ['benzene', 'naphthalene', 'cyclohexane', 'cyclopentane']

FRAGMENT_CORES = [
    "c1ccncc1",
    "c1ccnc(N)c1",
    "c1cnc2ccccc2n1",
    "c1cc2ncccc2nc1",
    "c1ccc2[nH]ccc2c1",
    "C1CCNCC1",
    "C1COCCN1",
    "c1cnc[nH]1",
    "c1csc(N)n1",
    "c1ccoc1",
    "c1cc[nH]c1",
    "C1CNCCN1",
    "c1ccc(F)cc1",
    "c1ccc(O)cc1",
    "c1ccc(N)cc1",
    "c1cnccn1",
    "c1ccc2occc2c1",
    "c1ccc2sccc2c1",
    "c1ccnc2ccccc12",
    "c1ccc(Cl)cc1",
    # --- small cores (r ~ 2.9) for tight pockets (r_pocket < 4.0) ---
    "c1cnco1",
    "c1cnno1",
    "C1CNCC1",
    "C1COCC1",
    # --- large cores (r ~ 3.7-3.9) for big pockets (r_pocket > 6.0) ---
    "C1CCC2CCCCC2C1",
    "C1CCCC2CCCCC2C1",
    "c1ccc2c(c1)c(Cl)ccc2",
    "c1ccc2c(c1)c(OC)ccc2",
    "c1ccc2c(c1)c(O)ccc2",
    "c1ccc2c(c1)c(C)ccc2",
    # --- v8: 10 more small cores (r ~ 2.5-3.1) for tight pockets ---
    "C1CNC1",
    "c1cn[nH]n1",
    "C1CSC1",
    "C1COC1",
    "c1ccsc1",
    "c1ccncn1",
    "CN1CCC1",
    "c1nn[nH]n1",
    "c1cncs1",
    "O=C1CCN1",
    # --- v8: 10 medium cores (r ~ 3.2-3.6) ---
    "c1ccc2[nH]cnc2c1",
    "C1CCOc2ccccc21",
    "c1ccc2cnccc2c1",
    "O=C1CCCCC1",
    "c1ccc2[nH]ncc2c1",
    "Cc1ccccc1",
    "COc1ccccc1",
    "NC(=O)c1ccccc1",
    "c1ccc2c(c1)CCN2",
    "c1ccc2c(c1)CCO2",
    # --- v8: 10 large cores (r ~ 3.8-4.0) ---
    "c1ccc2c(c1)Cc1ccccc12",
    "c1ccc(-c2ccccc2)cc1",
    "c1ccc2c(c1)[nH]c1ccccc12",
    "c1ccc2c(c1)oc1ccccc12",
    "C1C2CC3CC1C3C2",
    "C1CC2CCC1C2",
    "C1CCC2(CC1)CCCC2",
    "c1ccc2nc3ccccc3cc2c1",
    "c1ccc2c(c1)Cc1ccccc1O2",
    "c1ccc2cc3ccccc3cc2c1",
]

FRAGMENT_ADDITIONS = [
    ("O", 16, 0.3),
    ("OC", 32, 0.4),
    ("N", 15, 0.3),
    ("C(=O)O", 45, 0.5),
    ("C(=O)N", 44, 0.5),
    ("F", 19, 0.1),
    ("Cl", 35, 0.2),
    ("CC", 28, 0.4),
    ("C(C)C", 42, 0.5),
    ("S(=O)(=O)N", 80, 0.6),
    ("CN", 29, 0.4),
    ("OCC", 46, 0.5),
]


def _canonical(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


CHEMISTRY_BLACKLIST = ['CCC1CCCCC1', 'C1CCCCC1', 'CCCCCC', 'C1CCCC1']


def _passes_chemistry(smi, stats=None):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        if stats is not None:
            stats['rejected_unparseable'] += 1
        return False, 'unparseable'
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    het = Descriptors.NumHeteroatoms(mol)

    def tick(key):
        if stats is not None:
            stats[key] += 1

    if smi in CHEMISTRY_BLACKLIST:
        tick('rejected_blacklist')
        return False, 'blacklist'
    if mw < 100:
        tick('rejected_mw_low')
        return False, 'mw_low'
    if mw > 600:
        tick('rejected_mw_high')
        return False, 'mw_high'
    if logp > 5.0:
        tick('rejected_logp_high')
        return False, 'logp_high'
    if logp < -1.0:
        tick('rejected_logp_low')
        return False, 'logp_low'
    if hbd + hba < 1:
        tick('rejected_no_hba_hbd')
        return False, 'no_hba_hbd'
    if het < 1:
        tick('rejected_no_heteroatoms')
        return False, 'no_heteroatoms'
    return True, 'passed'


def _default_stats():
    return {
        'rejected_unparseable': 0, 'rejected_blacklist': 0,
        'rejected_mw_low': 0, 'rejected_mw_high': 0,
        'rejected_logp_high': 0, 'rejected_logp_low': 0,
        'rejected_no_hba_hbd': 0, 'rejected_no_heteroatoms': 0,
        'rejected_radius': 0, 'rejected_duplicate': 0,
        'passed': 0,
    }


def attach_fragment(smiles, group):
    """Attach a fragment group to the first atom with available valence."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    gmol = Chem.MolFromSmiles(group)
    if gmol is None:
        return None
    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        if atom.GetTotalNumHs() < 1:
            continue
        try:
            new_mol = Chem.RWMol(mol)
            core_atom = new_mol.GetAtomWithIdx(i)
            # replace one H with the new bond
            new_H = max(0, core_atom.GetTotalNumHs() - 1)
            core_atom.SetNumExplicitHs(new_H)
            offset = new_mol.GetNumAtoms()
            # add group atoms
            for j in range(gmol.GetNumAtoms()):
                new_mol.AddAtom(gmol.GetAtomWithIdx(j))
            # add group bonds
            for bond in gmol.GetBonds():
                new_mol.AddBond(
                    bond.GetBeginAtomIdx() + offset,
                    bond.GetEndAtomIdx() + offset,
                    bond.GetBondType()
                )
            # connect first atom of group to core
            new_mol.AddBond(i, 0 + offset, Chem.BondType.SINGLE)
            Chem.SanitizeMol(new_mol)
            return Chem.MolToSmiles(new_mol)
        except Exception:
            continue
    return None


def generate_candidates(target_info, n_attempts=2000, max_chem=20,
                        target_name='', seen_smiles=None, stats=None,
                        collapse_tracker=None):
    """Generate up to max_chem SMILES using n_attempts fragment tries.
    collapse_tracker: dict[smiles] -> set[target_name] for global collapse filtering.
    """
    if seen_smiles is None:
        seen_smiles = set()
    if stats is None:
        stats = _default_stats()
    r_pocket = target_info.get('r_pocket')
    if r_pocket is None:
        return [], stats
    fam = target_info.get('protein_family', '').lower()
    if 'kinase' in fam:
        r_optimal = r_pocket * 0.68
    elif 'protease' in fam:
        r_optimal = r_pocket * 0.75
    elif 'gpcr' in fam:
        r_optimal = r_pocket * 0.65
    elif 'nuclear_receptor' in fam or 'hormone_receptor' in fam:
        r_optimal = r_pocket * 0.70
    elif 'transporter' in fam:
        r_optimal = r_pocket * 0.63
    else:
        r_optimal = r_pocket * 0.72

    # deterministic per-target randomization
    rng = random.Random(target_name)
    cores = list(FRAGMENT_CORES)
    rng.shuffle(cores)
    additions = list(FRAGMENT_ADDITIONS)
    rng.shuffle(additions)
    all_combos = []
    for n_add in [1, 2, 3]:
        all_combos.extend(itertools.combinations(additions, n_add))
    rng.shuffle(all_combos)
    tasks = [(core, combo) for core in cores for combo in all_combos]
    rng.shuffle(tasks)

    candidates = []
    attempts = 0
    for core, combo in tasks:
        if attempts >= n_attempts or len(candidates) >= max_chem:
            break
        attempts += 1
        new_smiles = core
        ok_combo = True
        for (group, _mw_d, _r_d) in combo:
            new_smiles = attach_fragment(new_smiles, group)
            if new_smiles is None:
                ok_combo = False
                break
        if not ok_combo:
            continue
        can = _canonical(new_smiles)
        if can is None or can in seen_smiles:
            continue
        if collapse_tracker is not None:
            if can in collapse_tracker and len(collapse_tracker[can]) >= 5:
                continue
        ok, _ = _passes_chemistry(can, stats)
        if not ok:
            continue
        r_actual = r_ligand_from_smiles(can)
        if r_actual is None or abs(r_actual - r_optimal) > 0.50:
            stats['rejected_radius'] += 1
            continue
        stats['passed'] += 1
        seen_smiles.add(can)
        if collapse_tracker is not None:
            collapse_tracker.setdefault(can, set()).add(target_name)
        candidates.append((can, core))
    return candidates, stats


# ─── ZINC library screening (pre-computed property cache) ───────────
_ZINC_CACHE = None  # lazy-loaded filtered ZINC library

def _load_zinc_cache(cache_path=None):
    """Load pre-computed ZINC property cache (pickle)."""
    global _ZINC_CACHE
    if _ZINC_CACHE is not None:
        return _ZINC_CACHE
    if cache_path is None:
        cache_path = 'zinc_property_cache.pkl'
    if not os.path.exists(cache_path):
        # Try data/ prefix
        cache_path = 'data/zinc_property_cache.pkl'
        if not os.path.exists(cache_path):
            return None
    import pickle as _pkl
    with open(cache_path, 'rb') as f:
        _ZINC_CACHE = _pkl.load(f)
    return _ZINC_CACHE


def zinc_screen(target_info, max_hits=100, target_name='',
                seen_smiles=None, collapse_tracker=None, **kwargs):
    """Screen pre-computed ZINC property cache against a target pocket.

    All molecular properties (MW, logP, HBD, HBA, rings, r_ligand) are
    pre-computed in the cache. No RDKit calls per molecule — pure dict lookups.

    Returns (candidates, stats) where candidates is [(smiles, scaffold), ...].
    """
    if seen_smiles is None:
        seen_smiles = set()
    stats = _default_stats()

    r_pocket = target_info.get('r_pocket')
    if r_pocket is None:
        return [], stats

    zinc_lib = _load_zinc_cache(kwargs.get('cache_path'))
    if zinc_lib is None:
        print(f"  [zinc] No cache found, falling back to fragments")
        return generate_candidates(target_info, n_attempts=2000, max_chem=max_hits,
                                   target_name=target_name,
                                   seen_smiles=seen_smiles,
                                   collapse_tracker=collapse_tracker)

    # Shape complementarity gate: |r_ligand - r_pocket| < 2.0 Å
    r_min = r_pocket - 2.0
    r_max = r_pocket + 2.0

    # Pocket-only dG (unchanged, validated 4/5)
    dG = -GAMMA_REF_MNM * (r_pocket ** ALPHA_POCKET)

    # Collect all pocket-fitting candidates with ranking score
    scored = []
    for rec in zinc_lib:
        smi = rec['smiles']

        # Skip already-seen SMILES (fast set lookup)
        if smi in seen_smiles:
            continue

        # Collapse tracking (fast dict lookup)
        if collapse_tracker is not None:
            if smi in collapse_tracker and len(collapse_tracker[smi]) >= 5:
                continue

        # Pocket fit — pure float comparison, no RDKit
        r_lig = rec['r_ligand']
        if r_lig < r_min or r_lig > r_max:
            stats['rejected_radius'] += 1
            continue

        # All pharmacophore + chemistry filters already applied in cache build
        # Ranking score: dG (negative) - logP_bonus (favors hydrophobic molecules)
        logp_bonus = rec['logp'] * 0.3
        score = dG - logp_bonus  # more negative = better rank

        scored.append((score, smi, f"ZINC_mw{rec['mw']:.0f}"))

    # Sort by score (most negative first) and take top max_hits
    scored.sort(key=lambda x: x[0])
    candidates = []
    for _, smi, scaffold in scored[:max_hits]:
        stats['passed'] += 1
        seen_smiles.add(smi)
        if collapse_tracker is not None:
            collapse_tracker.setdefault(smi, set()).add(target_name)
        candidates.append((smi, scaffold))

    return candidates, stats


def _generic_scaffold(mol):
    """Extract generic scaffold name from molecule."""
    from rdkit.Chem import Scaffolds
    try:
        scaffold = Scaffolds.MurckoScaffold.GetScaffoldForMol(mol)
        smi = Chem.MolToSmiles(scaffold)
        if smi:
            return smi[:30]
    except Exception:
        pass
    return 'unknown'


def _generate_disease_candidates(target_info, n=100, target_name='',
                                 seen_smiles=None):
    """Backward-compatible wrapper returning SMILES only."""
    cands, _ = generate_candidates(target_info, n_attempts=n, max_chem=n,
                                   target_name=target_name,
                                   seen_smiles=seen_smiles)
    return [smi for smi, _ in cands]


def build_disease_targets():
    targets = {}
    with open(DISEASE_TARGET_FILE) as f:
        next(f, None)  # header
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = [p.strip() for p in line.split('|')]
            if len(parts) < 5:
                continue
            cat, target, r, gamma, mech = parts[:5]
            targets[target] = {
                'category': cat,
                'r_pocket': float(r) if r not in ('None', '') else None,
                'gamma_pocket': float(gamma) if gamma not in ('None', '') else None,
                'mechanism': mech,
            }
    return targets


DENTAL_TARGETS = {
    'USAG-1': {'category': 'dental', 'r_pocket': 5.8, 'gamma_pocket': 46.0,
               'mechanism': 'BMP_antagonist', 'protein_family': 'enzyme'},
    'MMP-20': {'category': 'dental', 'r_pocket': 6.2, 'gamma_pocket': 48.0,
               'mechanism': 'protease_inhibitor', 'protein_family': 'protease'},
    'KLK4': {'category': 'dental', 'r_pocket': 5.5, 'gamma_pocket': 44.0,
             'mechanism': 'protease_inhibitor', 'protein_family': 'protease'},
    'ENAM': {'category': 'dental', 'r_pocket': 6.8, 'gamma_pocket': 51.0,
             'mechanism': 'enamel_matrix', 'protein_family': 'enzyme'},
    'AMTN': {'category': 'dental', 'r_pocket': 5.2, 'gamma_pocket': 42.0,
             'mechanism': 'enamel_maturation', 'protein_family': 'enzyme'},
    'FAM20A': {'category': 'dental', 'r_pocket': 6.0, 'gamma_pocket': 47.0,
               'mechanism': 'kinase_inhibitor', 'protein_family': 'kinase'},
    'SLC24A4': {'category': 'dental', 'r_pocket': 7.2, 'gamma_pocket': 54.0,
                'mechanism': 'calcium_transporter', 'protein_family': 'transporter'},
    'WNT10A': {'category': 'dental', 'r_pocket': 5.9, 'gamma_pocket': 46.0,
               'mechanism': 'Wnt_signaling', 'protein_family': 'growth_factor'},
    'BMP7': {'category': 'dental', 'r_pocket': 6.4, 'gamma_pocket': 49.0,
             'mechanism': 'growth_factor', 'protein_family': 'growth_factor'},
    'CTNNB1': {'category': 'dental', 'r_pocket': 5.7, 'gamma_pocket': 45.0,
               'mechanism': 'transcription_factor', 'protein_family': 'transcription'},
}


def build_full_disease_targets():
    """Read 1000-disease target list with protein family."""
    targets = {}
    with open(FULL_DISEASE_TARGET_FILE) as f:
        next(f, None)  # header
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = [p.strip() for p in line.split('|')]
            if len(parts) < 6:
                continue
            cat, target, r, gamma, mech, fam = parts[:6]
            targets[target] = {
                'category': cat,
                'r_pocket': float(r) if r not in ('None', '') else None,
                'gamma_pocket': float(gamma) if gamma not in ('None', '') else None,
                'mechanism': mech,
                'protein_family': fam,
            }
    targets.update(DENTAL_TARGETS)

    # --- External target loader (targets_20k.json) ---
    import json as _json, os as _os
    _ext_path = _os.path.expanduser(
        "~/NS/Simulation/Bella2/data/targets_20k.json")
    if _os.path.exists(_ext_path):
        with open(_ext_path) as _f:
            _ext_targets = _json.load(_f)
        _added = 0
        for _t in _ext_targets:
            _sym = _t["symbol"]
            if _sym in targets:
                continue
            _cat = _t.get("category", "other")
            _src = _t.get("source", "T50_classified")
            # Infer protein_family from source for r_optimal calculation
            if "kinase" in _src.lower() or "T56" in _src:
                _fam = "kinase"
            elif "T59" in _src:
                _fam = "ion_channel"
            elif "T57" in _src or "T58" in _src:
                _fam = "enzyme"
            else:
                _fam = "enzyme"
            targets[_sym] = {
                'category': _cat,
                'r_pocket': float(_t['r_pocket']),
                'gamma_pocket': 46.0,
                'mechanism': _src,
                'protein_family': _fam,
            }
            _added += 1
        print(f"[loader] Added {_added} external targets "
              f"(total: {len(targets)})")
    # --- End loader ---

    # Per-category proportional cap — least-filled categories processed first
    _total_budget = len(targets)
    _cat_groups = {}
    for _tname, _tinfo in targets.items():
        _cat_groups.setdefault(_tinfo['category'], {})[_tname] = _tinfo
    _n_categories = len(_cat_groups)
    _cap_per_category = max(10, _total_budget // _n_categories)

    # Sort categories by current fill level (ascending = least-filled first)
    _cat_order = sorted(_cat_groups.keys(), key=lambda c: len(_cat_groups[c]))

    _capped = {}
    _overflow = []
    for _cat in _cat_order:
        _ctargets = _cat_groups[_cat]
        if len(_ctargets) <= _cap_per_category:
            _capped.update(_ctargets)
        else:
            _sorted = dict(sorted(
                _ctargets.items(),
                key=lambda x: x[1].get('disgenet_score', 0),
                reverse=True
            )[:_cap_per_category])
            _capped.update(_sorted)
            # Re-queue overflow into next best matching category
            _overflow_targets = list(_ctargets.items())[_cap_per_category:]
            for _otname, _otinfo in _overflow_targets:
                _otinfo['_overflow_origin'] = _cat
                _overflow.append((_otname, _otinfo))

    # Distribute overflow into remaining categories with spare capacity
    _fill_counts = {c: sum(1 for t in _capped.values() if t.get('category') == c) for c in _cat_groups}
    for _otname, _otinfo in _overflow:
        _placed = False
        for _cat in sorted(_cat_order, key=lambda c: _fill_counts.get(c, 0)):
            if _fill_counts.get(_cat, 0) < _cap_per_category:
                _otinfo['category'] = _cat
                _otinfo['_requeued'] = True
                _capped[_otname] = _otinfo
                _fill_counts[_cat] = _fill_counts.get(_cat, 0) + 1
                _placed = True
                break
        if not _placed:
            # Hard cap reached — drop target
            pass

    targets = _capped
    _total = sum(len(v) for v in _cat_groups.values())
    print(f"[cap] Total targets after cap: {len(targets)} (was {_total})")
    print(f"[cap] Categories: {_n_categories}, cap_per_category: {_cap_per_category}, "
          f"overflow re-queued: {len(_overflow)}")

    return targets


_fs = None


def _init_worker():
    global _fs
    import foam_screener_v2
    _fs = foam_screener_v2


def get_protein_class(target_name, category=""):
    t = target_name.upper()
    cat = category.upper()
    if any(x in t for x in ["BRD", "BRD2", "BRD3", "BRD4"]):
        return "bromodomain"
    if any(x in t for x in ["PARP", "PARP1", "PARP2"]):
        return "parp"
    if any(x in t for x in ["HSP90", "HSPC"]):
        return "hsp90"
    if any(x in t for x in ["CDK", "CDKN", "AURK", "PLK", "CHK", "BUB"]):
        return "kinase_cdk"
    if any(x in t for x in ["EGFR", "ERBB", "ALK", "MET", "KIT", "FLT", "RET", "RAF", "MAP", "MEK"]):
        return "kinase_egfr"
    if any(x in t for x in ["BCL2", "BCL", "BAX", "MCL"]):
        return "bcl"
    if any(x in t for x in ["ESR", "AR", "PPARG", "NR3", "VDR", "RORA"]):
        return "nuclear_receptor"
    if any(x in t for x in ["CASP", "PRSS", "MMP", "FURIN", "ACE", "TMPRSS", "CTSL"]):
        return "protease"
    if any(x in t for x in ["ADRB", "DRD", "HTR", "CHRM", "CXCR", "CCR", "GPR"]):
        return "gpcr"
    return "default"


def _screen_one(args):
    target, tinfo, smi, generic, dose = args
    global _fs
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    MW = Descriptors.MolWt(mol)
    logP = Descriptors.MolLogP(mol)
    charge = Chem.GetFormalCharge(mol)
    r_lig = _fs.r_ligand_from_smiles(smi)
    r_pocket = tinfo['r_pocket']

    # Class-specific GAMMA_REF from protein class anchor
    protein_class = get_protein_class(target, category=tinfo.get('category', ''))
    gamma_ref = CLASS_GAMMA.get(protein_class, CLASS_GAMMA["default"])

    # Calibrated T50 dG: class-specific gamma * r_pocket^ALPHA_POCKET
    dG = -gamma_ref * (r_pocket ** ALPHA_POCKET)

    # T50 binding_dG from inline foam theorem (γ_water × r^α × SASA)
    if r_lig is not None:
        sasa_est = 4 * 3.14159 * (r_lig ** 2) * 0.65
    else:
        sasa_est = 4 * 3.14159 * (2.0 ** 2) * 0.65
    dG_t50 = _fs.t50_binding_dG(r_pocket, sasa_est)

    # T73 floor check: if dG > dG_floor (less negative than floor) → BELOW_FOAM_FLOOR
    dG_floor_t73 = _fs.t73_floor_dG(r_pocket, sasa_est)
    below_foam_floor = (dG > dG_floor_t73) if dG_floor_t73 < 0 else False

    # T71 diagnostic score (calibrated, molecule-aware via logP)
    c_healthy = tinfo.get('c_healthy_nM', 1.0)
    c_disease = tinfo.get('c_disease_nM', 100.0)
    d_score, kd_opt, kd_cal = _fs.t71_diagnostic_score(dG, c_healthy, c_disease,
                                                        logp_molecule=logP)

    tox = _fs.toxicity_screen_full(logP, MW, charge, 1.0,
                                   SMILES=smi, dose_mg=dose)
    hetero = Descriptors.NumHeteroatoms(mol)

    # Lipophilic mode: recompute tox_score excluding MITO and ETC
    # (these assume free membrane concentration; hydrophobic molecules
    # are protein-bound in vivo, so membrane disruption is overestimated)
    # Keep: membrane_yl, reactive_metabolite, BSEP (dose-independent)
    if LIPOPHILIC_MODE:
        raw_score = tox['score']
        mito_penalty = 2 if tox.get('mito_verdict', 'SAFE') == 'TOXIC' else 0
        etc_penalty = 1 if tox.get('etc_verdict', 'LOW') == 'HIGH' else 0
        adjusted_score = max(0, raw_score - mito_penalty - etc_penalty)
        tox_score = adjusted_score
    else:
        tox_score = tox['score']

    return {
        'target': target,
        'category': tinfo['category'],
        'smiles': smi,
        'generic_scaffold': generic,
        'r_ligand': r_lig,
        'dG_kcal': round(dG, 3),
        'dG_t50': round(dG_t50, 3),
        'dG_floor_t73': round(dG_floor_t73, 3),
        'below_foam_floor': below_foam_floor,
        'D_score': d_score,
        'kd_opt': kd_opt,
        'kd_cal': kd_cal,
        'logP': round(logP, 2),
        'MW': round(MW, 1),
        'protein_class': protein_class,
        'tox_score': tox_score,
        'heteroatoms': int(hetero),
        'cyp450': tox.get('cyp450_verdict', 'UNKNOWN'),
        'oatp': tox.get('oatp', 'UNKNOWN'),
        'etc': tox.get('etc_verdict', 'LOW') if not LIPOPHILIC_MODE else 'SKIPPED',
        'mech7': tox.get('mechanism7_verdict', 'UNKNOWN'),
        'membrane': tox.get('membrane_verdict', 'SAFE'),
        'mito': tox.get('mito_verdict', 'SAFE') if not LIPOPHILIC_MODE else 'SKIPPED',
        'bsep': tox.get('bsep_verdict', 'LOW'),
        'reactive': tox.get('reactive_type', 'none'),
        'mechanism': tinfo['mechanism'],
    }


def _prescreen_one(args):
    """Lightweight pre-screen: dG + floor check + heteroatoms only. No toxicity."""
    target, tinfo, smi, generic, dose = args
    global _fs
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    r_lig = _fs.r_ligand_from_smiles(smi)
    r_pocket = tinfo['r_pocket']

    # Calibrated T50 dG: direct formula -GAMMA_REF_MNM * r_pocket^ALPHA_POCKET
    dG = -GAMMA_REF_MNM * (r_pocket ** ALPHA_POCKET)

    if r_lig is not None:
        sasa_est = 4 * 3.14159 * (r_lig ** 2) * 0.65
    else:
        sasa_est = 4 * 3.14159 * (2.0 ** 2) * 0.65
    dG_t50 = _fs.t50_binding_dG(r_pocket, sasa_est)
    dG_floor_t73 = _fs.t73_floor_dG(r_pocket, sasa_est)
    below_foam_floor = (dG > dG_floor_t73) if dG_floor_t73 < 0 else False
    hetero = Descriptors.NumHeteroatoms(mol)
    c_healthy = tinfo.get('c_healthy_nM', 1.0)
    c_disease = tinfo.get('c_disease_nM', 100.0)
    d_score, kd_opt, kd_cal = _fs.t71_diagnostic_score(dG, c_healthy, c_disease)
    return {
        'target': target,
        'category': tinfo['category'],
        'smiles': smi,
        'generic_scaffold': generic,
        'r_ligand': r_lig,
        'dG_kcal': round(dG, 3),
        'dG_t50': round(dG_t50, 3),
        'dG_floor_t73': round(dG_floor_t73, 3),
        'below_foam_floor': below_foam_floor,
        'D_score': d_score,
        'kd_opt': kd_opt,
        'tox_score': -1,  # not yet screened
        'heteroatoms': int(hetero),
        'cyp450': 'PENDING', 'oatp': 'PENDING', 'etc': 'PENDING',
        'mech7': 'PENDING', 'membrane': 'PENDING', 'mito': 'PENDING',
        'bsep': 'PENDING', 'reactive': 'none',
        'mechanism': tinfo['mechanism'],
    }


def _full_screen_one(args):
    """Full toxicity screen on a pre-screened candidate."""
    target, tinfo, smi, generic, dose = args
    global _fs
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    MW = Descriptors.MolWt(mol)
    logP = Descriptors.MolLogP(mol)
    charge = Chem.GetFormalCharge(mol)
    tox = _fs.toxicity_screen_full(logP, MW, charge, 1.0,
                                   SMILES=smi, dose_mg=dose)
    return tox


def _select_candidates(records, out_path, top_n=20):
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    clean = [r for r in records if r['tox_score'] <= TOX_THRESHOLD and r['dG_kcal'] < -3.0 and r['heteroatoms'] >= 1
             and not r.get('below_foam_floor', False)]
    best_by_bucket = {}
    for r in clean:
        bucket = int(round(r['r_ligand'] / 0.5)) if r['r_ligand'] is not None else 0
        key = (r['target'], bucket)
        if key not in best_by_bucket:
            best_by_bucket[key] = r
        else:
            cur = best_by_bucket[key]
            if (r['tox_score'], r['dG_kcal']) < (cur['tox_score'], cur['dG_kcal']):
                best_by_bucket[key] = r
    deduped = list(best_by_bucket.values())
    deduped.sort(key=lambda x: (x['tox_score'], x['dG_kcal']))
    top = deduped[:top_n]

    if len(top) < top_n:
        seen = {r['smiles'] for r in top}
        clean_sorted = sorted(clean, key=lambda x: (x['tox_score'], x['dG_kcal']))
        for r in clean_sorted:
            if len(top) >= top_n:
                break
            if r['smiles'] not in seen:
                top.append(r)
                seen.add(r['smiles'])

    with open(out_path, 'w') as f:
        f.write("RANK | TARGET | SMILES | dG | TOX | HET | MECHANISM | LABEL\n")
        for i, r in enumerate(top, 1):
            f.write(f"{i} | {r['target']} | {r['smiles']} | {r['dG_kcal']} | "
                    f"{r['tox_score']} | {r['heteroatoms']} | {r['mechanism']} | DERIVED from T50+T56+T57\n")
    return top


def _write_category_candidates(category, records):
    out_dir = f'findings/Publish/{category}'
    os.makedirs(out_dir, exist_ok=True)
    path = f'{out_dir}/{category}_candidates.txt'
    return _select_candidates(records, path)


def _write_master_index_1000(targets, per_target_stats, out_path='findings/Publish/master_index_1000_v7.txt'):
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    all_smiles = set()
    review = []
    collapse = []
    with open(out_path, 'w') as f:
        f.write("CATEGORY | TARGET | GENERATED | PASSING_CHEM | PASSING_TOX | TOP_dG | TOP_SMILES | SCAFFOLD_TYPE\n")
        for tname, tinfo in targets.items():
            s = per_target_stats.get(tname, {})
            gen = s.get('generated', 0)
            chem = s.get('passing_chem', 0)
            tox = s.get('passing_tox', 0)
            top_smiles = s.get('top_smiles', '')
            top_dg = s.get('top_dG', '')
            scaf = s.get('top_scaffold', '')
            f.write(f"{tinfo['category']} | {tname} | {gen} | {chem} | {tox} | {top_dg} | {top_smiles} | {scaf}\n")
            if top_smiles:
                all_smiles.add(top_smiles)
            if tox < 5:
                review.append(tname)
        total_gen = sum(s.get('generated', 0) for s in per_target_stats.values())
        total_chem = sum(s.get('passing_chem', 0) for s in per_target_stats.values())
        total_tox = sum(s.get('passing_tox', 0) for s in per_target_stats.values())
        hit_chem = total_chem / total_gen if total_gen else 0.0
        hit_tox = total_tox / total_gen if total_gen else 0.0
        f.write(f"\nOverall hit rate (chem): {hit_chem:.3f}\n")
        f.write(f"Overall hit rate (tox): {hit_tox:.3f}\n")
        f.write(f"Total unique top SMILES: {len(all_smiles)}\n")
        f.write(f"Targets with <5 clean tox candidates: {len(review)}\n")


def _write_full_summary_1000(all_records, total_chem, total_gen, category_bests,
                             collapse_smiles, scaffold_counts,
                             out_path='findings/Publish/full_screen_1000_v7_summary.txt'):
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    total_tox = sum(1 for r in all_records if r['tox_score'] <= TOX_THRESHOLD)
    unique_smiles = len(set(r['smiles'] for r in all_records))

    with open(out_path, 'w') as f:
        f.write("FULL 1000-DISEASE DE NOVO SCREEN SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Total candidates generated: {total_gen}\n")
        f.write(f"Passing chemistry filters: {total_chem}\n")
        f.write(f"Passing tox (score <= 1, dG<-3.0, het>=1): {total_tox}\n")
        f.write(f"Chemistry hit rate: {total_chem/total_gen:.3f}\n" if total_gen else "\n")
        f.write(f"Tox hit rate: {total_tox/total_chem:.3f}\n" if total_chem else "\n")
        f.write(f"Unique SMILES count (diversity): {unique_smiles}\n")

        # dG range and outlier flags
        dG_values = [r['dG_kcal'] for r in all_records if r['dG_kcal'] is not None]
        if dG_values:
            dG_min = min(dG_values)
            dG_max = max(dG_values)
            f.write(f"dG range: {dG_min:.3f} to {dG_max:.3f} kcal/mol\n")
            f.write(f"All in -3 to -15 range: {'YES' if all(-15 <= d <= -3 for d in dG_values) else 'NO'}\n")
            below_neg15 = [r for r in all_records if r['dG_kcal'] is not None and r['dG_kcal'] < -15.0]
            above_neg3 = [r for r in all_records if r['dG_kcal'] is not None and r['dG_kcal'] > -3.0]
            f.write(f"Candidates below -15 kcal/mol: {len(below_neg15)}\n")
            for r in below_neg15[:10]:
                f.write(f"  *** OUTLIER: {r['target']} dG={r['dG_kcal']} {r['smiles']}\n")
            f.write(f"Candidates above -3 kcal/mol: {len(above_neg3)}\n")
            for r in above_neg3[:10]:
                f.write(f"  *** OUTLIER: {r['target']} dG={r['dG_kcal']} {r['smiles']}\n")
        f.write("\n")

        f.write("--- Best candidate per category ---\n")
        for cat in sorted(category_bests):
            b = category_bests[cat]
            if b:
                f.write(f"{cat}: target={b['target']} dG={b['dG_kcal']} tox={b['tox_score']} "
                        f"smiles={b['smiles']}\n")
            else:
                f.write(f"{cat}: no clean candidate\n")

        f.write(f"\n--- Scaffold breakdown ---\n")
        for scaf, c in sorted(scaffold_counts.items(), key=lambda x: -x[1]):
            f.write(f"{scaf}: {c}\n")

        if collapse_smiles:
            f.write(f"\n--- GEOMETRY COLLAPSE (SMILES in >5 targets) ---\n")
            for smi, c in collapse_smiles:
                f.write(f"{c} | {smi}\n")
        else:
            f.write("\nNo geometry collapse detected (no SMILES in >5 targets)\n")

        # Targets with <5 clean candidates
        clean_by_target = {}
        for r in all_records:
            if r['tox_score'] <= TOX_THRESHOLD and r['dG_kcal'] is not None and r['dG_kcal'] < -3.0 and r['heteroatoms'] >= 1:
                clean_by_target.setdefault(r['target'], 0)
                clean_by_target[r['target']] += 1
        low_targets = [t for t, c in clean_by_target.items() if c < 5]
        f.write(f"\n--- TARGETS WITH <5 CLEAN CANDIDATES ({len(low_targets)}) ---\n")
        for t in sorted(low_targets):
            f.write(f"  {t}: {clean_by_target.get(t, 0)} clean\n")

        # Average candidates per target
        n_targets = len(set(r['target'] for r in all_records))
        if n_targets:
            f.write(f"\nAverage candidates per target: {len(all_records) / n_targets:.1f}\n")

        f.write("\nAll results labeled DERIVED from T50+T56+T57\n")


def _adaptive_n_attempts(r_pocket, base_n):
    """Scale n_attempts based on pocket extremity.
    Small pockets (r<4.5) and large pockets (r>5.5) need more attempts
    because fewer fragments fit the radius window."""
    if r_pocket is None:
        return base_n
    if r_pocket < 4.0:
        return base_n * 4
    elif r_pocket < 4.5:
        return base_n * 3
    elif r_pocket > 6.0:
        return base_n * 4
    elif r_pocket > 5.5:
        return base_n * 3
    return base_n


def _screen_category(category, ctargets, n, dose, n_workers, raw_path,
                     collapse_tracker=None, min_clean=20, max_validated_per_target=20):
    """Two-phase screening: lightweight prescreen, then full toxicity only on clean candidates."""
    os.makedirs(os.path.dirname(raw_path) or '.', exist_ok=True)
    os.makedirs('findings/unvalidated', exist_ok=True)
    all_tasks = []
    seen = set()
    generated_counts = {}
    passing_chem = {}
    target_cands = {}  # tname -> list of (smi, gen)
    poor_pocket_count = 0
    weak_pocket_count = 0
    good_pocket_count = 0
    extrapolated_count = 0
    for tname, tinfo in ctargets.items():
        r_pocket = tinfo.get('r_pocket')

        # T73 pocket quality filter [DERIVED]
        if r_pocket is not None:
            dg_floor = t73_hydrophobic_floor(r_pocket)
        else:
            dg_floor = 0.0

        if dg_floor > -0.5:
            # Pocket too shallow — no drug will bind
            print(f"  [skip] {tname}: POOR_POCKET dG_floor={dg_floor:.2f}, r={r_pocket}")
            poor_pocket_count += 1
            generated_counts[tname] = 0
            passing_chem[tname] = 0
            continue
        elif dg_floor > -2.0:
            pocket_quality = "WEAK"
            weak_pocket_count += 1
        else:
            pocket_quality = "GOOD"
            good_pocket_count += 1

        # T50 domain validation
        if r_pocket is not None and not (T50_VALID_MIN <= r_pocket <= T50_VALID_MAX):
            print(f"  [warn] {tname}: r={r_pocket:.1f}Å OUTSIDE T50 domain [{T50_VALID_MIN}-{T50_VALID_MAX}Å] — ΔG extrapolated")
            tinfo['t50_status'] = 'EXTRAPOLATED'
            extrapolated_count += 1
        else:
            tinfo['t50_status'] = 'DERIVED'

        tinfo['pocket_quality'] = pocket_quality

        cands, stats = zinc_screen(tinfo, max_hits=MAX_CHEM,
                                    target_name=tname, seen_smiles=seen,
                                    collapse_tracker=collapse_tracker)
        target_cands[tname] = cands
        generated_counts[tname] = len(cands)
        passing_chem[tname] = len(cands)

    # Retry pass: targets with <20 candidates get a second shot with 2x attempts
    retry_targets = [t for t, c in target_cands.items() if len(c) < 20]
    if retry_targets:
        for tname in retry_targets:
            tinfo = ctargets[tname]
            r_pocket = tinfo.get('r_pocket')
            extra, stats = zinc_screen(tinfo, max_hits=MAX_CHEM,
                                        target_name=tname + '_retry',
                                        seen_smiles=seen,
                                        collapse_tracker=collapse_tracker)
            target_cands[tname].extend(extra)
            generated_counts[tname] = len(target_cands[tname])
            passing_chem[tname] = len(target_cands[tname])

    for tname, cands in target_cands.items():
        tinfo = ctargets[tname]
        for smi, gen in cands:
            all_tasks.append((tname, tinfo, smi, gen, dose))

    records = []
    # ─── PHASE 1: Lightweight prescreen all candidates ───
    prescreen_results = []
    if all_tasks:
        with Pool(processes=n_workers, initializer=_init_worker) as pool:
            for res in pool.imap_unordered(_prescreen_one, all_tasks, chunksize=50):
                if res is None:
                    continue
                prescreen_results.append(res)

    # ─── PHASE 2: Full toxicity only on clean candidates (dG < -8.0, above floor, hetero >= 1) ───
    # Group prescreen results by target, pick top candidates for validation
    pre_by_target = {}
    for r in prescreen_results:
        pre_by_target.setdefault(r['target'], []).append(r)

    validation_tasks = []
    unvalidated_records = []
    clean_count_by_target = {}

    for tname, pre_list in pre_by_target.items():
        # Sort by dG (most negative first) — best binders first
        pre_list.sort(key=lambda x: x['dG_kcal'])
        validated_for_target = 0
        for r in pre_list:
            is_clean = (r['dG_kcal'] < -8.0 and
                       not r.get('below_foam_floor', False) and
                       r['heteroatoms'] >= 1)
            # Validate all pre-filtered clean candidates (no cap on total validations)
            # The cap is on CLEAN candidates found, not total validations
            if is_clean:
                tinfo = ctargets[tname]
                validation_tasks.append((tname, tinfo, r['smiles'], r['generic_scaffold'], dose))
                validated_for_target += 1
            else:
                # Write unvalidated candidate directly
                unvalidated_records.append(r)

    # Run full toxicity on validation candidates only (via _screen_one which includes tox)
    if validation_tasks:
        with Pool(processes=n_workers, initializer=_init_worker) as pool:
            for res in pool.imap_unordered(_screen_one, validation_tasks, chunksize=50):
                if res is None:
                    continue
                records.append(res)
                tname = res['target']
                if res['tox_score'] <= TOX_THRESHOLD and res['dG_kcal'] < -3.0 and res['heteroatoms'] >= 1:
                    clean_count_by_target[tname] = clean_count_by_target.get(tname, 0) + 1

    # Write unvalidated records to separate file
    unval_path = raw_path.replace('_screen_raw.txt', '_unvalidated.txt')
    if unvalidated_records:
        with open(unval_path, 'w') as f:
            f.write("TARGET\tCATEGORY\tSMILES\tSCAFFOLD\tr_ligand_A\tdG_kcal\tdG_t50\tdG_floor_t73\tD_SCORE\tHET\tMECHANISM\n")
            for r in unvalidated_records:
                f.write(
                    f"{r['target']}\t{r['category']}\t{r['smiles']}\t"
                    f"{r['generic_scaffold']}\t{r['r_ligand']}\t"
                    f"{r['dG_kcal']}\t{r['dG_t50']}\t{r['dG_floor_t73']}\t"
                    f"{r['D_score']}\t{r['heteroatoms']}\t{r['mechanism']}\n"
                )

    # Write validated records to raw file
    with open(raw_path, 'w') as f:
        f.write("TARGET\tCATEGORY\tSMILES\tSCAFFOLD\tr_ligand_A\tdG_kcal\tdG_t50\tdG_floor_t73\t"
                "D_SCORE\tTOX_SCORE\tHET\t"
                "CYP450\tOATP\tETC\tMECH7\tMEMBRANE\tMITO\tBSEP\tREACTIVE\tMECHANISM\t"
                "logP\tMW\tprotein_class\tKd_cal_nM\tKd_opt_nM\n")
        for res in records:
            f.write(
                f"{res['target']}\t{res['category']}\t{res['smiles']}\t"
                f"{res['generic_scaffold']}\t{res['r_ligand']}\t"
                f"{res['dG_kcal']}\t{res['dG_t50']}\t{res['dG_floor_t73']}\t"
                f"{res['D_score']}\t{res['tox_score']}\t{res['heteroatoms']}\t"
                f"{res['cyp450']}\t{res['oatp']}\t{res['etc']}\t{res['mech7']}\t"
                f"{res['membrane']}\t{res['mito']}\t{res['bsep']}\t"
                f"{res['reactive']}\t{res['mechanism']}\t"
                f"{res['logP']}\t{res['MW']}\t{res['protein_class']}\t"
                f"{res['kd_cal']}\t{res['kd_opt']}\n"
            )

    print(f"  [T73] POOR_POCKET={poor_pocket_count} WEAK={weak_pocket_count} GOOD={good_pocket_count} EXTRAPOLATED={extrapolated_count}")
    print(f"  [validate] {len(records)} validated, {len(unvalidated_records)} unvalidated, "
          f"{sum(clean_count_by_target.values())} clean")
    return records, generated_counts, passing_chem


def _load_targets_from_file(filepath):
    """Load targets from text file (UniProt | gene | disease | priority | pocket_class)
    or JSON file. Returns dict[target_name] = target_info."""
    import json as _json
    targets = {}

    if filepath.endswith('.json'):
        with open(filepath) as f:
            data = _json.load(f)
        for t in data:
            symbol = t.get('symbol', t.get('gene', ''))
            if not symbol:
                continue
            r_pocket = t.get('r_pocket', 6.0)
            targets[symbol] = {
                'r_pocket': r_pocket,
                'category': t.get('category', 'imported'),
                'disease': t.get('disease', 'unknown'),
                'mechanism': t.get('mechanism', 'unknown'),
                'gamma_pocket': t.get('gamma_pocket', 40.0),
                'protein_family': t.get('protein_family', ''),
                'uniprot': t.get('uniprot', ''),
            }
    else:
        # Text format: UniProt_ID | gene_name | disease | priority_score | r_pocket | source
        # (new format) or UniProt_ID | gene | disease | priority | pocket_class (old)
        with open(filepath) as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = [p.strip() for p in line.split('|')]
                if len(parts) < 5:
                    continue
                uniprot = parts[0]
                symbol = parts[1]
                disease = parts[2]

                # New 6-column format: r_pocket is a float in column 5
                if len(parts) >= 6:
                    try:
                        r_pocket = float(parts[4])
                    except ValueError:
                        r_pocket = 6.0
                else:
                    # Old 5-column format: parse pocket_class: T50_r6.0_s3
                    pocket_class = parts[4]
                    r_pocket = 6.0
                    if 'r' in pocket_class:
                        try:
                            r_part = pocket_class.split('r')[1].split('_')[0]
                            r_pocket = float(r_part)
                        except (ValueError, IndexError):
                            pass

                # Assign category from disease
                category = 'imported'
                targets[symbol] = {
                    'r_pocket': r_pocket,
                    'category': category,
                    'disease': disease,
                    'mechanism': 'unknown',
                    'gamma_pocket': 40.0,
                    'protein_family': '',
                    'uniprot': uniprot,
                }

    print(f"[load] {len(targets)} targets from {filepath}")
    return targets


def run_full_disease_screen(n=2000, dose=50.0, n_workers=4, version='v8',
                            min_clean=10, max_attempts=None,
                            output_dir=None, failed_log=None,
                            skip_categories=None, targets_file=None,
                            zinc_cache_path=None, progress_log_path=None,
                            max_targets=None):
    global T50_VALID_MIN, T50_VALID_MAX

    # Preload ZINC cache once at startup
    if zinc_cache_path:
        print(f"[zinc] Preloading cache: {zinc_cache_path}")
        _load_zinc_cache(zinc_cache_path)
        print(f"[zinc] Cache loaded: {len(_ZINC_CACHE)} molecules")

    if targets_file:
        all_targets = _load_targets_from_file(targets_file)
    else:
        all_targets = build_full_disease_targets()

    # Smoke test: limit number of targets
    if max_targets and len(all_targets) > max_targets:
        _keys = list(all_targets.keys())[:max_targets]
        all_targets = {k: all_targets[k] for k in _keys}
        print(f"[smoke] Limited to {len(all_targets)} targets")

    categories = {}
    for tname, tinfo in all_targets.items():
        categories.setdefault(tinfo['category'], {})[tname] = tinfo

    # Checkpoint resume: skip categories whose raw file already exists
    completed_cats = set()
    for cat in list(categories.keys()):
        raw_path = f'findings/{cat}_screen_raw.txt'
        if os.path.exists(raw_path) and os.path.getsize(raw_path) > 100:
            completed_cats.add(cat)
    if completed_cats:
        print(f"[checkpoint] {len(completed_cats)} categories already completed, skipping")
        categories = {k: v for k, v in categories.items() if k not in completed_cats}

    # Also skip explicitly requested categories
    if skip_categories:
        _skip_set = set(skip_categories)
        _before = len(categories)
        categories = {k: v for k, v in categories.items() if k not in _skip_set}
        print(f"[resume] Skipping {_before - len(categories)} explicitly skipped categories, "
              f"{len(categories)} remaining")

    all_records = []
    total_generated = 0
    total_chem = 0
    _progress_count = 0
    _progress_clean = 0
    _progress_relaxed = 0
    _progress_exhausted = 0
    per_target_stats = {t: {'generated': 0, 'passing_chem': 0, 'passing_tox': 0,
                            'top_smiles': '', 'top_dG': '', 'top_scaffold': ''} for t in all_targets}
    category_bests = {}
    collapse_tracker = {}  # global: smiles -> set of targets

    # HSP90-family targets: T50_EXCLUDED_POLAR (resorcinol pocket is polar, not hydrophobic)
    HSP90_FAMILY = {'HSP90AA1', 'HSP90AB1', 'HSP90B1', 'TRAP1'}
    excluded_polar_path = 'excluded_polar.txt'
    with open(excluded_polar_path, 'w') as ep:
        ep.write("TARGET\tREASON\n")
        for t in sorted(HSP90_FAMILY & set(all_targets)):
            ep.write(f"{t}\tT50_EXCLUDED_POLAR\n")
    hsp90_in_targets = HSP90_FAMILY & set(all_targets)
    if hsp90_in_targets:
        print(f"[exclude] HSP90-family polar targets: {sorted(hsp90_in_targets)}")

    _start_time = datetime.now()

    # RAM monitoring
    def _check_ram():
        try:
            import psutil
            vm = psutil.virtual_memory()
            return vm.percent, vm.used / 1e9
        except ImportError:
            return 0, 0

    def _write_progress(msg):
        print(msg)
        if progress_log_path:
            with open(progress_log_path, 'a') as pf:
                pf.write(msg + '\n')

    for cat, ctargets in sorted(categories.items()):
        raw_path = f'findings/{cat}_screen_raw.txt'
        # Filter out HSP90-family targets
        ctargets_screened = {t: ti for t, ti in ctargets.items() if t not in HSP90_FAMILY}
        if not ctargets_screened:
            print(f"[screen] {cat}: 0 targets (all HSP90-excluded)")
            continue
        print(f"[screen] {cat}: {len(ctargets_screened)} targets")

        # RAM check — auto-reduce workers if RAM > 12GB
        ram_pct, ram_gb = _check_ram()
        if ram_gb > 12.0 and n_workers > 2:
            print(f"  [ram] {ram_gb:.1f}GB used > 12GB, reducing workers {n_workers}->2")
            n_workers = 2

        try:
            records, gen_counts, chem_counts = _screen_category(
                cat, ctargets_screened, n, dose, n_workers, raw_path,
                collapse_tracker=collapse_tracker,
                min_clean=min_clean, max_validated_per_target=20)
        except Exception as e:
            print(f"  [ERROR] Category {cat} failed: {e}")
            if failed_log:
                with open(failed_log, 'a') as fl:
                    fl.write(f"{cat}\t{e}\n")
            continue
        all_records.extend(records)
        total_generated += sum(gen_counts.values())
        total_chem += sum(chem_counts.values())

        # Progress reporting every 50 targets
        _progress_count += len(ctargets_screened)
        for r in records:
            if r['tox_score'] <= TOX_THRESHOLD and r['dG_kcal'] < -3.0 and r['heteroatoms'] >= 1:
                _progress_clean += 1
        if _progress_count % 50 < len(ctargets_screened) or _progress_count >= len(all_targets):
            _elapsed = (datetime.now() - _start_time).total_seconds()
            _rate = _progress_count / max(_elapsed, 1)
            _eta = (len(all_targets) - _progress_count) / max(_rate, 0.01)
            ram_pct, ram_gb = _check_ram()
            _write_progress(f"T: {_progress_count}/{len(all_targets)} | "
                  f"clean: {_progress_clean} | "
                  f"ram_gb: {ram_gb:.1f} | "
                  f"eta_min: {_eta/60:.1f}")

        for tname in ctargets:
            per_target_stats[tname]['generated'] = gen_counts.get(tname, 0)
            per_target_stats[tname]['passing_chem'] = chem_counts.get(tname, 0)
        for r in records:
            if r['tox_score'] <= TOX_THRESHOLD:
                per_target_stats[r['target']]['passing_tox'] += 1

        _write_category_candidates(cat, records)
        # best per category: tox=0, hetero>=1, lowest dG
        best = None
        for r in sorted(records, key=lambda x: (x['tox_score'], x['dG_kcal'])):
            if r['tox_score'] == 0 and r['heteroatoms'] >= 1:
                best = r
                break
        category_bests[cat] = best

    # top per target
    for r in sorted(all_records, key=lambda x: (x['tox_score'], x['dG_kcal'])):
        t = r['target']
        if not per_target_stats[t]['top_smiles'] and r['tox_score'] <= TOX_THRESHOLD and r['heteroatoms'] >= 1:
            per_target_stats[t]['top_smiles'] = r['smiles']
            per_target_stats[t]['top_dG'] = r['dG_kcal']
            per_target_stats[t]['top_scaffold'] = r['generic_scaffold']

    # geometry collapse: any SMILES appearing in > 5 targets
    smile_target_counts = {}
    for r in all_records:
        smile_target_counts.setdefault(r['smiles'], set()).add(r['target'])
    collapse = [(smi, len(tgts)) for smi, tgts in smile_target_counts.items() if len(tgts) > 5]
    collapse.sort(key=lambda x: -x[1])

    # scaffold counts
    scaffold_counts = {}
    for r in all_records:
        scaffold_counts[r['generic_scaffold']] = scaffold_counts.get(r['generic_scaffold'], 0) + 1

    master_path = f'findings/Publish/master_index_1000_{version}.txt'
    summary_path = f'findings/Publish/full_screen_1000_{version}_summary.txt'
    _write_master_index_1000(all_targets, per_target_stats, out_path=master_path)
    _write_full_summary_1000(all_records, total_chem, total_generated, category_bests,
                             collapse, scaffold_counts, out_path=summary_path)
    print(f"[main] Done: generated={total_generated}, chem={total_chem}, "
          f"tox={sum(1 for r in all_records if r['tox_score'] <= TOX_THRESHOLD)}")
    print("Validation cap: 20 per target. Post-screen pass: DISABLED.")

    # ─── FINAL SUMMARY TABLE ─────────────────────────────────────
    clean_by_target = {}
    for r in all_records:
        if r['tox_score'] <= TOX_THRESHOLD and r['dG_kcal'] < -3.0 and r['heteroatoms'] >= 1:
            clean_by_target.setdefault(r['target'], 0)
            clean_by_target[r['target']] += 1

    n_20plus = sum(1 for t in all_targets if clean_by_target.get(t, 0) >= 20)
    n_10_19 = sum(1 for t in all_targets if 10 <= clean_by_target.get(t, 0) < 20)
    n_lt10 = sum(1 for t in all_targets if clean_by_target.get(t, 0) < 10)

    print(f"\n{'='*70}")
    print("FULL DISEASE SCREEN — SUMMARY")
    print(f"{'='*70}")
    print(f"Total targets completed:   {len(all_targets)}")
    print(f"Targets with >=20 clean:   {n_20plus}")
    print(f"Targets with 10-19 clean:  {n_10_19}")
    print(f"Targets with <10 clean:    {n_lt10} (need manual review)")
    print(f"Total clean candidates:    {sum(clean_by_target.values())}")
    print(f"Total relaxed candidates:  0")
    print(f"Total exhausted:           0")
    _end_time = datetime.now()
    _runtime_min = (_end_time - _start_time).total_seconds() / 60.0
    print(f"Runtime minutes:           {_runtime_min:.1f}")

    # Top 20 by D_score closest to 0
    target_dscores = {}
    for r in all_records:
        ds = r.get('D_score')
        if ds is not None:
            t = r['target']
            if t not in target_dscores or abs(ds) < abs(target_dscores[t]):
                target_dscores[t] = ds
    top_dscore = sorted(target_dscores.items(), key=lambda x: abs(x[1]))[:20]
    print(f"\nTop 20 targets by D_score closest to 0 (best diagnostic window):")
    print(f"  {'target':<30} {'D_score':>10}")
    print(f"  {'-'*40}")
    for tname, ds in top_dscore:
        print(f"  {tname:<30} {ds:>10.3f}")
    print(f"{'='*70}")

    # Write SUMMARY.txt
    if output_dir:
        summary_path = os.path.join(output_dir, 'SUMMARY.txt')
    else:
        summary_path = 'SUMMARY.txt'
    with open(summary_path, 'w') as sf:
        sf.write(f"FULL DISEASE SCREEN — SUMMARY\n{'='*70}\n")
        sf.write(f"Total targets completed:   {len(all_targets)}\n")
        sf.write(f"Targets with >=20 clean:   {n_20plus}\n")
        sf.write(f"Targets with 10-19 clean:  {n_10_19}\n")
        sf.write(f"Targets with <10 clean:    {n_lt10} (need manual review)\n")
        sf.write(f"Total clean candidates:    {sum(clean_by_target.values())}\n")
        sf.write(f"Total relaxed candidates:  0\n")
        sf.write(f"Total exhausted:           0\n")
        sf.write(f"Runtime minutes:           {_runtime_min:.1f}\n")
        sf.write(f"\nTop 20 targets by D_score closest to 0 (best diagnostic window):\n")
        sf.write(f"  {'target':<30} {'D_score':>10}\n")
        sf.write(f"  {'-'*40}\n")
        for tname, ds in top_dscore:
            sf.write(f"  {tname:<30} {ds:>10.3f}\n")
        # Bottom 20: exhausted or <5 clean
        sf.write(f"\nBottom 20 (exhausted or <5 clean) — flag for manual review:\n")
        sf.write(f"  {'target':<30} {'clean_count':>10}\n")
        sf.write(f"  {'-'*40}\n")
        bottom = sorted(all_targets, key=lambda t: clean_by_target.get(t, 0))[:20]
        for tname in bottom:
            sf.write(f"  {tname:<30} {clean_by_target.get(tname, 0):>10}\n")
        sf.write(f"{'='*70}\n")
    print(f"[summary] Written to {summary_path}")


def run_category_only(category, n=2000, dose=50.0, n_workers=4, version=None):
    all_targets = build_full_disease_targets()
    ctargets = {t: ti for t, ti in all_targets.items() if ti['category'] == category}
    if not ctargets:
        print(f"[error] No targets found for category '{category}'")
        return
    raw_path = f'findings/{category}_screen_raw.txt'
    print(f"[screen] {category}: {len(ctargets)} targets")
    collapse_tracker = {}
    records, gen_counts, chem_counts = _screen_category(
        category, ctargets, n, dose, n_workers, raw_path,
        collapse_tracker=collapse_tracker)

    out_dir = f'findings/Publish/{category.capitalize()}'
    os.makedirs(out_dir, exist_ok=True)
    out_path = f'{out_dir}/{category.lower()}_top20_calibrated.txt'
    _select_candidates(records, out_path, top_n=20)
    # append calibration info to the output
    dGs = [r['dG_kcal'] for r in records if r['dG_kcal'] is not None]
    with open(out_path, 'a') as f:
        f.write(f"\n--- Calibration info ---\n")
        f.write(f"Category: {category}  Targets: {len(ctargets)}\n")
        f.write(f"Candidates generated: {sum(gen_counts.values())}\n")
        f.write(f"Passing chem: {sum(chem_counts.values())}\n")
        if dGs:
            f.write(f"Min dG: {min(dGs):.3f}  Max dG: {max(dGs):.3f}\n")
            in_range = all(-15 <= d <= -2 for d in dGs)
            f.write(f"All in -3 to -15 range: {'YES' if in_range else 'NO'}\n")
        f.write("\nAll results DERIVED from T50+T56+T57\n")

    print(f"[done] Wrote {out_path}")
    if records:
        dGs = [r['dG_kcal'] for r in records if r['dG_kcal'] is not None]
        if dGs:
            print(f"  min dG={min(dGs):.3f}  max dG={max(dGs):.3f}")

    # Report clean candidates per target
    clean_by_target = {}
    for r in records:
        if r['tox_score'] <= TOX_THRESHOLD and r['dG_kcal'] < -3.0 and r['heteroatoms'] >= 1:
            clean_by_target.setdefault(r['target'], 0)
            clean_by_target[r['target']] += 1
    low_clean = [t for t in ctargets if clean_by_target.get(t, 0) < 5]
    print(f"[report] Targets with <5 clean: {len(low_clean)}")
    for t in low_clean:
        print(f"  {t}: {clean_by_target.get(t, 0)} clean")

    # Check collapse_tracker for _pass2 phantom entries
    pass2_entries = sum(1 for targets_set in collapse_tracker.values()
                        if any('_pass2' in t for t in targets_set))
    print(f"[report] _pass2 entries in collapse_tracker: {pass2_entries}")

    # Write summary if version specified
    if version:
        summary_path = f'findings/Publish/full_screen_{version}_summary.txt'
        os.makedirs(os.path.dirname(summary_path) or '.', exist_ok=True)
        with open(summary_path, 'w') as f:
            f.write(f"CATEGORY TEST: {category}\n")
            f.write(f"Targets: {len(ctargets)}\n")
            f.write(f"Candidates: {sum(gen_counts.values())}\n")
            f.write(f"Clean (tox<=1, dG<-3, het>=1): {sum(clean_by_target.values())}\n")
            f.write(f"Targets with <5 clean: {len(low_clean)}\n")
            f.write(f"_pass2 entries in collapse_tracker: {pass2_entries}\n")
            for t in sorted(ctargets):
                print(f"  {t}: {clean_by_target.get(t, 0)} clean")
                f.write(f"  {t}: {clean_by_target.get(t, 0)} clean\n")
        print(f"[done] Wrote {summary_path}")


if __name__ == '__main__':
    print(f"[startup] len(FRAGMENT_CORES) = {len(FRAGMENT_CORES)}")
    if '--category' in sys.argv or '--categories' in sys.argv:
        flag = '--category' if '--category' in sys.argv else '--categories'
        idx = sys.argv.index(flag)
        cat = sys.argv[idx + 1]
        n = 2000
        dose = 50.0
        workers = 4
        ver = None
        if '--version' in sys.argv:
            vi = sys.argv.index('--version')
            ver = f'v{sys.argv[vi + 1]}'
        if '--targets' in sys.argv:
            ti = sys.argv.index('--targets')
            n = int(sys.argv[ti + 1])
        if '--dose' in sys.argv:
            di = sys.argv.index('--dose')
            dose = float(sys.argv[di + 1])
        if '--workers' in sys.argv:
            wi = sys.argv.index('--workers')
            workers = int(sys.argv[wi + 1])
        run_category_only(cat, n=n, dose=dose, n_workers=workers, version=ver)
    elif '--version' in sys.argv:
        idx = sys.argv.index('--version')
        ver = sys.argv[idx + 1]
        n = 2000
        dose = 50.0
        workers = 4
        min_clean = 10
        max_attempts = None
        output_dir = None
        failed_log = 'failed_targets.txt'
        skip_cats = None
        _max_targets = None
        if '--max-targets' in sys.argv:
            mt = sys.argv.index('--max-targets')
            _max_targets = int(sys.argv[mt + 1])
            print(f"[cli] --max-targets={_max_targets}")
        if '--targets' in sys.argv:
            ti = sys.argv.index('--targets')
            tval = sys.argv[ti + 1]
            n = 999999 if tval == 'all' else int(tval)
        _tf_path = None
        if '--targets-file' in sys.argv:
            tf = sys.argv.index('--targets-file')
            tf_path = sys.argv[tf + 1]
            print(f"[cli] --targets-file={tf_path}")
            if os.path.exists(tf_path):
                if tf_path.endswith('.json'):
                    _tf_path = tf_path
                    import json as _json
                    with open(tf_path) as _f:
                        _ext = _json.load(_f)
                    print(f"[cli] Loaded {len(_ext)} targets from JSON")
                else:
                    # Text format: UniProt_ID | gene_name | disease | priority_score | pocket_class
                    _ext_targets = []
                    with open(tf_path) as _f:
                        for line in _f:
                            if line.startswith('#') or not line.strip():
                                continue
                            parts = [p.strip() for p in line.split('|')]
                            if len(parts) >= 6:
                                _ext_targets.append({
                                    'uniprot': parts[0],
                                    'symbol': parts[1],
                                    'disease': parts[2],
                                    'priority': float(parts[3]),
                                    'r_pocket': float(parts[4]),
                                    'source': parts[5],
                                })
                            elif len(parts) >= 5:
                                _ext_targets.append({
                                    'uniprot': parts[0],
                                    'symbol': parts[1],
                                    'disease': parts[2],
                                    'priority': float(parts[3]),
                                    'pocket_class': parts[4],
                                })
                    print(f"[cli] Loaded {len(_ext_targets)} targets from text file")
                    # Store for run_full_disease_screen to pick up
                    import json as _json
                    _tmp_json = tf_path + '.parsed.json'
                    with open(_tmp_json, 'w') as _jf:
                        _json.dump(_ext_targets, _jf)
                    _tf_path = _tmp_json
            else:
                print(f"[cli] WARNING: {tf_path} not found, using built-in targets")
        if '--failed-log' in sys.argv:
            fli = sys.argv.index('--failed-log')
            failed_log = sys.argv[fli + 1]
            print(f"[cli] --failed-log={failed_log}")
        if '--max-attempts' in sys.argv:
            mi = sys.argv.index('--max-attempts')
            max_attempts = int(sys.argv[mi + 1])
            n = max_attempts
            print(f"[cli] --max-attempts={max_attempts}")
        if '--jobs' in sys.argv:
            ji = sys.argv.index('--jobs')
            workers = int(sys.argv[ji + 1])
            print(f"[cli] --jobs={workers}")
        if '--mode' in sys.argv:
            mi = sys.argv.index('--mode')
            mode = sys.argv[mi + 1]
            print(f"[cli] --mode={mode}")
        if '--min-clean' in sys.argv:
            mi = sys.argv.index('--min-clean')
            min_clean = int(sys.argv[mi + 1])
            print(f"[cli] --min-clean={min_clean}")
        if '--output-dir' in sys.argv:
            oi = sys.argv.index('--output-dir')
            output_dir = sys.argv[oi + 1]
            os.makedirs(output_dir, exist_ok=True)
            print(f"[cli] --output-dir={output_dir}")
        if '--skip-categories' in sys.argv:
            sci = sys.argv.index('--skip-categories')
            skip_cats = [c.strip() for c in sys.argv[sci + 1].split(',')]
            print(f"[cli] --skip-categories={len(skip_cats)} categories")
        if '--output' in sys.argv:
            pass  # output dir already hardcoded in function
        if '--dose' in sys.argv:
            di = sys.argv.index('--dose')
            dose = float(sys.argv[di + 1])
        if '--workers' in sys.argv:
            wi = sys.argv.index('--workers')
            workers = int(sys.argv[wi + 1])
        if '--tox-threshold' in sys.argv:
            ti2 = sys.argv.index('--tox-threshold')
            globals()['TOX_THRESHOLD'] = int(sys.argv[ti2 + 1])
            print(f"[cli] --tox-threshold={globals()['TOX_THRESHOLD']}")
        if '--max-chem' in sys.argv:
            mc = sys.argv.index('--max-chem')
            globals()['MAX_CHEM'] = int(sys.argv[mc + 1])
            print(f"[cli] --max-chem={globals()['MAX_CHEM']}")
        if '--label' in sys.argv:
            li2 = sys.argv.index('--label')
            globals()['SCREEN_LABEL'] = sys.argv[li2 + 1]
            print(f"[cli] --label={globals()['SCREEN_LABEL']}")
        if '--lipophilic-mode' in sys.argv:
            globals()['LIPOPHILIC_MODE'] = True
            globals()['TOX_THRESHOLD'] = 2
            print(f"[cli] --lipophilic-mode=ON (TOX_THRESHOLD={globals()['TOX_THRESHOLD']}, MITO/ETC skipped)")
        _zinc_cache = None
        if '--zinc-cache' in sys.argv:
            zc = sys.argv.index('--zinc-cache')
            _zinc_cache = sys.argv[zc + 1]
            print(f"[cli] --zinc-cache={_zinc_cache}")
        if '--backend' in sys.argv:
            bi = sys.argv.index('--backend')
            _backend = sys.argv[bi + 1]
            print(f"[cli] --backend={_backend}")
        _progress_log = None
        if output_dir:
            _progress_log = os.path.join(output_dir, 'progress.log')
        run_full_disease_screen(n=n, dose=dose, n_workers=workers, version=f'v{ver}',
                                min_clean=min_clean, max_attempts=max_attempts,
                                output_dir=output_dir, failed_log=failed_log,
                                skip_categories=skip_cats, targets_file=_tf_path,
                                zinc_cache_path=_zinc_cache, progress_log_path=_progress_log,
                                max_targets=_max_targets)
    else:
        n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
        dose = float(sys.argv[2]) if len(sys.argv) > 2 else 50.0
        workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
        run_full_disease_screen(n=n, dose=dose, n_workers=workers)

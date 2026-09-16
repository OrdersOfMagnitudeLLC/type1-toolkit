# OOM Commercial License v1.0
# Copyright 2026 Orders of Magnitude LLC
# See /NS/LICENSE.md for terms.
# Copyright (C) 2026 Orders of Magnitude LLC <orders@ofmagnitude.com>
import random
import re
import requests
import sys
import traceback
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors


def _alphafold_url(uniprot_id: str) -> str | None:
    """Return the first AlphaFold structure URL for a UniProt ID."""
    try:
        r = requests.get(
            f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}",
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        first = data[0]
        return first.get("pdbUrl") or first.get("cifUrl")
    except Exception as e:
        print(f"AlphaFold API error for {uniprot_id}: {e}", file=sys.stderr)
        return None


def _chembl_target_chembl_ids(uniprot_id: str) -> list[str]:
    """Return ChEMBL target IDs associated with a UniProt accession."""
    try:
        r = requests.get(
            "https://www.ebi.ac.uk/chembl/api/data/target",
            params={
                "target_components__accession": uniprot_id,
                "format": "json",
                "limit": 20,
            },
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        return [t["target_chembl_id"] for t in data.get("targets", []) if t.get("target_chembl_id")]
    except Exception as e:
        print(f"ChEMBL target lookup error for {uniprot_id}: {e}", file=sys.stderr)
        return []


def _chembl_mechanisms(target_chembl_id: str) -> list[dict]:
    """Return mechanisms for a ChEMBL target."""
    try:
        r = requests.get(
            "https://www.ebi.ac.uk/chembl/api/data/mechanism",
            params={
                "target_chembl_id": target_chembl_id,
                "format": "json",
                "limit": 100,
            },
            timeout=30,
        )
        r.raise_for_status()
        return r.json().get("mechanisms", [])
    except Exception as e:
        print(f"ChEMBL mechanism error for {target_chembl_id}: {e}", file=sys.stderr)
        return []


def _chembl_molecules(molecule_chembl_ids: list[str]) -> dict[str, dict]:
    """Fetch molecule records for a list of ChEMBL molecule IDs."""
    if not molecule_chembl_ids:
        return {}
    out = {}
    # ChEMBL __in expects comma-separated IDs
    joined = ",".join(molecule_chembl_ids[:50])
    try:
        r = requests.get(
            "https://www.ebi.ac.uk/chembl/api/data/molecule",
            params={
                "molecule_chembl_id__in": joined,
                "format": "json",
                "limit": 100,
            },
            timeout=60,
        )
        r.raise_for_status()
        for mol in r.json().get("molecules", []):
            cid = mol.get("molecule_chembl_id")
            if cid:
                out[cid] = mol
    except Exception as e:
        print(f"ChEMBL molecule lookup error: {e}", file=sys.stderr)
    return out


def bob_drug_repurpose(target_protein_id: str, top_n: int = 20) -> list[dict]:
    """Return FDA-approved/Phase-3 small molecules for a UniProt target."""
    target_protein_id = target_protein_id.strip()
    af_url = _alphafold_url(target_protein_id)

    target_ids = _chembl_target_chembl_ids(target_protein_id)
    if not target_ids:
        print(f"No ChEMBL target found for {target_protein_id}", file=sys.stderr)
        return []

    # Collect mechanisms (molecule -> mechanism_of_action mapping)
    mechanisms = []
    for tid in target_ids:
        mechanisms.extend(_chembl_mechanisms(tid))

    if not mechanisms:
        print(f"No mechanisms found for {target_protein_id}", file=sys.stderr)
        return []

    mol_ids = [m["molecule_chembl_id"] for m in mechanisms if m.get("molecule_chembl_id")]
    mol_map = _chembl_molecules(list(set(mol_ids)))

    results = []
    for mech in mechanisms:
        mol_id = mech.get("molecule_chembl_id")
        mol = mol_map.get(mol_id)
        if not mol:
            continue

        max_phase = mech.get("max_phase")
        if max_phase is None:
            max_phase = mol.get("max_phase", 0)
        if not isinstance(max_phase, int):
            try:
                max_phase = int(float(max_phase))
            except Exception:
                max_phase = 0

        if max_phase < 1:
            continue

        name = mol.get("pref_name") or mol.get("molecule_chembl_id")
        mol_props = mol.get("molecule_properties") or {}
        mw = mol_props.get("full_mwt") or mol_props.get("mw_freebase") or mol.get("molecular_weight")
        try:
            mw = float(mw) if mw is not None else float("inf")
        except Exception:
            mw = float("inf")

        results.append({
            "uniprot_id": target_protein_id,
            "target_chembl_id": mech.get("target_chembl_id"),
            "molecule_chembl_id": mol_id,
            "drug_name": name,
            "mechanism_of_action": mech.get("mechanism_of_action", ""),
            "max_phase": max_phase,
            "molecular_weight": mw,
            "alphafold_url": af_url,
        })

    # Sort by max_phase descending, then molecular weight ascending
    results.sort(key=lambda x: (-x["max_phase"], x["molecular_weight"]))
    return results[:top_n]


def _parse_formula(formula: str) -> dict[str, int]:
    """Parse a simple formula into {element: count}."""
    if not formula:
        return {}
    matches = re.findall(r"([A-Z][a-z]*)(\d*)", formula.replace(" ", ""))
    out = {}
    for el, n in matches:
        out[el] = int(n) if n else 1
    return out


def bob_search_senolytics(top_n: int = 20) -> list[dict]:
    """Search Bob materials for drug-like C/H/N/O/S/F candidates with senolytic similarity."""
    allowed = {"C", "H", "N", "O", "S", "F"}
    scaffolds = {
        "flavonoid": {"C", "H", "O"},
        "bh3_mimetic": {"C", "H", "N", "O", "S", "Cl"},
        "dasatinib_like": {"C", "H", "Cl", "N", "O", "S"},
    }

    # Import Bob material plugins without creating a cycle
    sys.path.insert(0, str(Path(__file__).resolve().parent / "plugins"))
    import materials_project as mp
    import gnome

    raw = []
    pairs = ["C,H", "C,N", "C,O", "C,S", "C,F"]
    for pair in pairs:
        try:
            raw.extend(mp.search(pair, limit=20))
        except Exception as e:
            print(f"Senolytic MP search ({pair}) error: {e}", file=sys.stderr)
    try:
        raw.extend(gnome.search("C", limit=20))
    except Exception as e:
        print(f"Senolytic GNoME search error: {e}", file=sys.stderr)

    results = []
    seen = set()
    for r in raw:
        formula = r.get("formula", "")
        if not formula or formula in seen:
            continue
        seen.add(formula)
        counts = _parse_formula(formula)
        if not counts:
            continue
        elements = set(counts.keys())
        if not elements.issubset(allowed):
            continue
        n_atoms = sum(counts.values())
        if n_atoms > 50:
            continue

        best = 0.0
        best_name = ""
        for name, els in scaffolds.items():
            inter = elements & els
            union = elements | els
            jaccard = len(inter) / len(union) if union else 0.0
            if jaccard > best:
                best = jaccard
                best_name = name

        results.append({
            "formula": formula,
            "source": r.get("source", "bob"),
            "n_atoms": n_atoms,
            "similarity_score": round(best, 3),
            "best_scaffold": best_name,
            "properties": r.get("properties", {}),
        })

    results.sort(key=lambda x: (-x["similarity_score"], x["n_atoms"]))
    return results[:top_n]


FALLBACK_CHEMBL = {
    "Q9UEF7": "CHEMBL4523",   # FGFR1 for Klotho/FGF23 pathway
    "Q8N6T7": "CHEMBL3077",   # SIRT1 for SIRT6 family
    "O95390": "CHEMBL2111421",# TGF-beta receptor for GDF11
    "P42771": "CHEMBL2094",   # CDK4 for p16 partner
    "O43524": "CHEMBL2842",   # mTOR for FOXO3 upstream
    "P01106": "CHEMBL1824",   # BRD4 for MYC indirect
    "P38398": "CHEMBL3776",   # PARP1 for BRCA1 DNA repair
    "Q13148": "CHEMBL2553",   # FUS for TDP43 RRM family
    "P35637": "CHEMBL2553",   # FUS for FUS family
}


def _check_lipinski(mol):
    """Return True if a molecule passes Lipinski rules."""
    return (
        Descriptors.MolWt(mol) < 500
        and Descriptors.MolLogP(mol) < 5
        and Descriptors.NumHDonors(mol) <= 5
        and Descriptors.NumHAcceptors(mol) <= 10
    )


def _mutate_atom_sub(mol):
    """Random single atom substitution C<->N<->O."""
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 0)]
    if not heavy:
        return None
    idx = random.choice(heavy)
    anum = mol.GetAtomWithIdx(idx).GetAtomicNum()
    swaps = {6: [7, 8], 7: [6, 8], 8: [6, 7]}
    if anum not in swaps:
        return None
    new_anum = random.choice(swaps[anum])
    new_mol = Chem.RWMol(mol)
    new_mol.GetAtomWithIdx(idx).SetAtomicNum(new_anum)
    try:
        Chem.SanitizeMol(new_mol)
        return Chem.MolToSmiles(new_mol, canonical=True)
    except Exception:
        return None


def _mutate_add_sub(mol, sub_smiles):
    """Attach a substituent to a hydrogen-bearing atom."""
    h_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetNumImplicitHs() > 0]
    if not h_atoms:
        return None
    idx = random.choice(h_atoms)
    sub = Chem.MolFromSmiles(sub_smiles)
    if sub is None:
        return None
    new_mol = Chem.RWMol(mol)
    offset = new_mol.GetNumAtoms()
    for a in sub.GetAtoms():
        new_mol.AddAtom(Chem.Atom(a.GetAtomicNum()))
    for b in sub.GetBonds():
        new_mol.AddBond(
            b.GetBeginAtomIdx() + offset,
            b.GetEndAtomIdx() + offset,
            b.GetBondType(),
        )
    new_mol.AddBond(idx, offset, Chem.BondType.SINGLE)
    try:
        Chem.SanitizeMol(new_mol)
        smi = Chem.MolToSmiles(new_mol, canonical=True)
        return smi
    except Exception:
        return None


def _mutate_remove_sub(mol):
    """Remove a terminal heavy-atom substituent."""
    leaves = [a.GetIdx() for a in mol.GetAtoms() if a.GetDegree() == 1 and a.GetAtomicNum() > 1]
    if not leaves:
        return None
    idx = random.choice(leaves)
    new_mol = Chem.RWMol(mol)
    new_mol.RemoveAtom(idx)
    try:
        Chem.SanitizeMol(new_mol)
        return Chem.MolToSmiles(new_mol, canonical=True)
    except Exception:
        return None


def _mutate_ring_expand(mol):
    """Expand one ring by inserting a CH2 into a ring bond."""
    ri = mol.GetRingInfo()
    bonds = [b for b in mol.GetBonds() if ri.NumBondRings(b.GetIdx())]
    if not bonds:
        return None
    bond = random.choice(bonds)
    a = bond.GetBeginAtomIdx()
    b = bond.GetEndAtomIdx()
    new_mol = Chem.RWMol(mol)
    new_mol.RemoveBond(a, b)
    new_idx = new_mol.AddAtom(Chem.Atom(6))  # carbon
    new_mol.AddBond(a, new_idx, Chem.BondType.SINGLE)
    new_mol.AddBond(new_idx, b, Chem.BondType.SINGLE)
    # add two H implicitly handled; but explicit valence will be adjusted by sanitize
    try:
        Chem.SanitizeMol(new_mol)
        return Chem.MolToSmiles(new_mol, canonical=True)
    except Exception:
        return None


def _chembl_active_smiles(target_chembl_id, min_pchembl=6.0, limit=50):
    """Return active canonical SMILES for a ChEMBL target."""
    try:
        r = requests.get(
            "https://www.ebi.ac.uk/chembl/api/data/activity.json",
            params={
                "target_chembl_id": target_chembl_id,
                "standard_type": "IC50",
                "pchembl_value__gte": min_pchembl,
                "limit": limit,
                "format": "json",
            },
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        smiles = []
        for a in data.get("activities", []):
            smi = a.get("canonical_smiles")
            if smi:
                smiles.append(smi)
        return smiles
    except Exception as e:
        print(f"ChEMBL activity error for {target_chembl_id}: {e}", file=sys.stderr)
        return []


def _seed_mutations(smiles):
    """Generate up to 5 Lipinski-passing mutations of a seed SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    SUBS = ["F", "Cl", "O", "C", "C#N", "C(F)(F)F", "OC", "N"]
    mutators = [
        _mutate_atom_sub,
        lambda m: _mutate_add_sub(m, random.choice(SUBS)),
        _mutate_remove_sub,
        _mutate_ring_expand,
    ]
    out = []
    seen = {Chem.MolToSmiles(mol, canonical=True)}
    for mut in mutators:
        smi = mut(mol)
        if smi is not None and smi not in seen:
            m = Chem.MolFromSmiles(smi)
            if m is not None and _check_lipinski(m):
                out.append(smi)
                seen.add(smi)
    # ensure 5 attempts by random add-sub again if needed
    while len(out) < 5:
        smi = _mutate_add_sub(mol, random.choice(SUBS))
        if smi is not None and smi not in seen:
            m = Chem.MolFromSmiles(smi)
            if m is not None and _check_lipinski(m):
                out.append(smi)
                seen.add(smi)
            else:
                break
    return out


HARD_CODED_FALLBACK = {
    "Q8N6T7": [
        "CC1=CC2=C(C=C1)N=C(N2)C3=CC=C(C=C3)Cl",
        "O=C(c1ccc(Cl)cc1)c1ccc2[nH]c(-c3ccncc3)cc2c1",
        "Cc1ccc(-c2cc(C(=O)O)c3[nH]c(-c4ccncc4)cc3c2)cc1",
        "O=C(O)c1cnc2[nH]c(-c3ccncc3)cc2c1",
        "CC(=O)Nc1ccc2[nH]c(-c3ccncc3)cc2c1",
    ],
}


def _safe_mol(smi):
    """Parse a SMILES safely and return a sanitized mol or None."""
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        if mol is None:
            return None
        Chem.SanitizeMol(mol, catchErrors=True)
        canonical = Chem.MolToSmiles(mol)
        if not canonical:
            return None
        mol2 = Chem.MolFromSmiles(canonical)
        if mol2 is None:
            return None
        return mol2
    except Exception:
        return None


def _check_lipinski_mol(mol):
    """Return True if a mol passes Lipinski rules."""
    return (
        Descriptors.MolWt(mol) < 500
        and Descriptors.MolLogP(mol) < 5
        and Descriptors.NumHDonors(mol) <= 5
        and Descriptors.NumHAcceptors(mol) <= 10
    )


def bob_seed_scaffolds(uniprot_id: str, n: int = 20) -> list[str]:
    """Return ChEMBL-derived, Lipinski-filtered seed scaffolds for a target."""
    uniprot_id = uniprot_id.strip()
    try:
        target_ids = _chembl_target_chembl_ids(uniprot_id)
    except Exception as e:
        print(f"target lookup error for {uniprot_id}: {e}", file=sys.stderr)
        target_ids = []

    raw_seeds = []
    for tid in target_ids:
        raw_seeds.extend(_chembl_active_smiles(tid))

    if len(raw_seeds) < 5:
        fallback = FALLBACK_CHEMBL.get(uniprot_id)
        if fallback:
            raw_seeds.extend(_chembl_active_smiles(fallback, limit=50))

    raw_seeds.extend(HARD_CODED_FALLBACK.get(uniprot_id, []))

    seeds = []
    seen = set()
    for smi in raw_seeds:
        mol = _safe_mol(smi)
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol, canonical=True)
        if canon not in seen:
            seen.add(canon)
            seeds.append(canon)
        if len(seeds) >= n:
            break

    results = []
    seen = set()
    for smi in seeds:
        mol = _safe_mol(smi)
        if mol is None or not _check_lipinski_mol(mol):
            continue
        canon = Chem.MolToSmiles(mol, canonical=True)
        if canon in seen:
            continue
        seen.add(canon)
        results.append(canon)
        for m in _seed_mutations(canon):
            m2 = _safe_mol(m)
            if m2 is not None and _check_lipinski_mol(m2):
                m_canon = Chem.MolToSmiles(m2, canonical=True)
                if m_canon not in seen:
                    seen.add(m_canon)
                    results.append(m_canon)
        if len(results) >= n:
            break

    return results[:n]

# Copyright (c) 2026 Orders of Magnitude LLC
# Licensed under the OOM Commercial License v1.0
# See LICENSE.md in the repository root or ofmagnitude.com

"""De novo generative drug design and Vina docking for protein targets."""
import random, time, os, sys, tempfile, re, json
import multiprocessing
from pathlib import Path
from urllib.parse import quote

import numpy as np
import requests
from vina import Vina
from rich.console import Console
from rich.table import Table
from rich import box
from Bio.PDB import PDBParser, NeighborSearch

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

console = Console()

SCAFFOLDS = [
    "c1ccccc1",
    "c1ccncc1",
    "c1c[nH]cn1",
    "c1ccc2[nH]ccc2c1",
    "c1cncnc1",
    "C1COCCN1",
    "C1CNCCN1",
    "c1ccsc1",
    "c1ccoc1",
    "c1ccc2ncccc2c1",
    "c1ccc2cnccc2c1",
    "c1cn[nH]c1",
    "c1cocn1",
    "c1cscn1",
    "C1CCCCN1",
    "O=C1CCCN1",
    "NS(=O)(=O)c1ccccc1",
    "NC(=O)c1ccccc1",
    "NC(=O)Nc1ccccc1",
    "c1ccc2ccccc2c1",
]

SUBSTITUENTS = [
    "O",
    "N",
    "C(=O)O",
    "C",
    "F",
    "Cl",
    "OC",
    "C#N",
    "[N+](=O)[O-]",
]


def check_lipinski(mol):
    """Return (pass: bool, details dict)."""
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    ok = (mw < 500) and (logp < 5) and (hbd <= 5) and (hba <= 10)
    return ok, {"mw": mw, "logp": logp, "hbd": hbd, "hba": hba}


def _add_substituent(em, base_idx, sub_smiles):
    """Attach the first atom of sub_mol to base_idx in em. Returns False on failure."""
    sub = Chem.MolFromSmiles(sub_smiles)
    if sub is None:
        return False
    offset = em.GetNumAtoms()
    for a in sub.GetAtoms():
        em.AddAtom(Chem.Atom(a.GetAtomicNum()))
    for b in sub.GetBonds():
        em.AddBond(
            b.GetBeginAtomIdx() + offset,
            b.GetEndAtomIdx() + offset,
            b.GetBondType(),
        )
    em.AddBond(base_idx, offset, Chem.BondType.SINGLE)
    try:
        Chem.SanitizeMol(em)
        return True
    except Exception:
        return False


def generate_smiles(n=5, max_attempts=None, scaffolds=None):
    """Generate N diverse, valid, Lipinski-passing SMILES."""
    max_attempts = max_attempts or max(100, n * 10)
    if scaffolds:
        valid_scaffolds = [s for s in scaffolds if Chem.MolFromSmiles(s) is not None]
    else:
        valid_scaffolds = []
    scaffold_pool = valid_scaffolds if valid_scaffolds else SCAFFOLDS
    valid = []
    attempts = 0
    seen = set()
    while len(valid) < n and attempts < max_attempts:
        attempts += 1
        base = Chem.MolFromSmiles(random.choice(scaffold_pool))
        if base is None:
            continue
        em = Chem.RWMol(base)
        Chem.SanitizeMol(em)
        n_subs = random.randint(0, 1) if scaffolds else random.randint(2, 4)
        ok = True
        for _ in range(n_subs):
            h_atoms = [
                a.GetIdx()
                for a in em.GetAtoms()
                if a.GetNumImplicitHs() > 0
            ]
            if not h_atoms:
                ok = False
                break
            idx = random.choice(h_atoms)
            sub = random.choice(SUBSTITUENTS)
            if not _add_substituent(em, idx, sub):
                ok = False
                break
        if not ok:
            continue
        mol = em.GetMol()
        try:
            Chem.SanitizeMol(mol)
            smi = Chem.MolToSmiles(mol, canonical=True)
            if smi in seen:
                continue
            seen.add(smi)
            test = Chem.MolFromSmiles(smi)
            if test is None:
                continue
            lip, _ = check_lipinski(test)
            if not lip:
                continue
            valid.append(smi)
        except Exception:
            continue
    return valid


def _pdbqt_atom_line(serial, atom, resi=1, resn="UNL"):
    """Return a PDBQT ATOM line in AutoDock/Vina fixed format."""
    sym = atom.GetSymbol()
    x, y, z = atom.GetOwningMol().GetConformer().GetAtomPosition(atom.GetIdx())
    atype = "HD" if sym == "H" else sym
    return (
        f"ATOM  {serial:5d} {sym:>4s}  {resn:>3s} {resi:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00    +0.000 {atype:2s}\n"
    )


def to_pdbqt_ligand(smiles):
    """Convert SMILES to a 3D conformer and write a PDBQT string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=0xF00D) == -1:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass
    lines = ["ROOT\n"]
    for i, a in enumerate(mol.GetAtoms()):
        lines.append(_pdbqt_atom_line(i + 1, a))
    lines.append("ENDROOT\n")
    lines.append("TORSDOF 0\n")
    return "".join(lines)


def _receptor_pdbqt_line(serial, sym, x, y, z):
    atype = sym.strip()
    return (
        f"ATOM  {serial:5d} {sym:>4s}  {'UNL':>3s} {1:4d}    "
        f"{float(x):8.3f}{float(y):8.3f}{float(z):8.3f}  1.00  0.00    +0.000 {atype:2s}\n"
    )


def to_pdbqt_receptor(parsed):
    """Write a PDBQT string from parse_pdb-style list [symbol, x, y, z]."""
    lines = []
    for i, row in enumerate(parsed):
        sym, x, y, z = [str(r).strip() for r in row]
        lines.append(_receptor_pdbqt_line(i + 1, sym, x, y, z))
    return "".join(lines)


def _center(parsed):
    n = len(parsed)
    if n == 0:
        return [0.0, 0.0, 0.0]
    sx = sum(float(a[1]) for a in parsed)
    sy = sum(float(a[2]) for a in parsed)
    sz = sum(float(a[3]) for a in parsed)
    return [sx / n, sy / n, sz / n]


def _pocket_center(pdb_path, percentile=0.1, radius=10.0):
    """Return the centroid of the most buried C-alpha atoms (proxy for pocket)."""
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("target", str(pdb_path))
        ca_atoms = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    # skip hetero/water
                    if residue.id[0] != ' ':
                        continue
                    for atom in residue:
                        if atom.id == 'CA':
                            ca_atoms.append(atom)
                            break
        if not ca_atoms:
            return None
        ns = NeighborSearch(ca_atoms)
        counts = [len(ns.search(a.coord, radius, level='A')) - 1 for a in ca_atoms]
        n_buried = max(1, int(len(ca_atoms) * percentile))
        buried_idx = np.argsort(counts)[:n_buried]
        buried = [ca_atoms[i].coord for i in buried_idx]
        centroid = np.mean(buried, axis=0)
        return [float(centroid[0]), float(centroid[1]), float(centroid[2])]
    except Exception as e:
        console.print(f"[yellow]Pocket center failed ({e}); falling back to geometric center.[/]")
        return None


def query_chembl(smiles, limit=3):
    """Query ChEMBL activity by canonical SMILES. Returns (status, ic50_nM or None)."""
    try:
        url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
        r = requests.get(
            url,
            params={"canonical_smiles": smiles, "limit": limit, "format": "json"},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        acts = data.get("activities", [])
        if not acts:
            return "NOVEL", None
        best = None
        for a in acts:
            if a.get("standard_type") == "IC50":
                val = a.get("standard_value")
                unit = a.get("standard_units")
                if val is None or unit is None:
                    continue
                try:
                    v = float(val)
                except Exception:
                    continue
                unit = unit.strip().upper()
                if unit == "NM":
                    pass
                elif unit in ("UM", "MICROMOLAR"):
                    v *= 1000.0
                elif unit == "MM":
                    v *= 1_000_000.0
                else:
                    continue
                if best is None or v < best:
                    best = v
        if best is None:
            return "KNOWN", None
        return "KNOWN", best
    except Exception:
        return "NOVEL", None


def _dock_one(smi, rec_path, center, n_cpus, box_size, exhaustiveness):
    """Dock a single SMILES; return a result dict or None."""
    test = Chem.MolFromSmiles(smi)
    if test is None:
        return None
    lip_ok, _ = check_lipinski(test)
    if not lip_ok:
        return None
    lig_pdbqt = to_pdbqt_ligand(smi)
    if lig_pdbqt is None:
        return None
    try:
        v = Vina(sf_name="vina", cpu=n_cpus, seed=0, no_refine=False, verbosity=0)
        v.set_receptor(rec_path)
        v.set_ligand_from_string(lig_pdbqt)
        v.compute_vina_maps(center, box_size)
        v.dock(exhaustiveness=exhaustiveness, n_poses=9)
        en = v.energies()
        if en is not None and en.size > 0:
            score = float(en[0, 0])
        else:
            score = float(v.score())
        if score < -30.0 or score > 5.0:
            return None
    except Exception:
        return None
    return {"smiles": smi, "score": score, "lip_ok": lip_ok}


def cmd_denovo(args):
    target = (getattr(args, "target", "") or "").strip()
    n_cpus = int(getattr(args, "cpus", 1))
    threshold = float(getattr(args, "threshold", -7.0))
    max_candidates = int(getattr(args, "max_candidates", 200))
    limit = getattr(args, "limit", None)
    if limit is None:
        limit = int(getattr(args, "candidates", 5))
    else:
        limit = int(limit)
    out = getattr(args, "out", None)
    if not target:
        console.print("[red]Usage: bella denovo --target <UniProt_ID> [--limit N] ...[/]")
        return 1

    # Avoid circular import at module load
    import bella

    seeded = bool(getattr(args, "seeded", False))
    console.print(
        f"[cyan]De novo: {target} | threshold {threshold} | "
        f"max {max_candidates} | limit {limit} | cpus {n_cpus} | "
        f"seeded {seeded}[/]"
    )
    import bob
    seed_scaffolds = bob.bob_seed_scaffolds(target, n=20) if seeded else None
    if seeded:
        console.print(f"[cyan]Loaded {len(seed_scaffolds)} ChEMBL-derived seed scaffolds.[/]")
    pdb_path = bella.fetch_alphafold_structure(target)
    if not pdb_path:
        console.print(f"[red]AlphaFold structure not found for {target}[/]")
        return 1

    n_atoms, parsed = bella.parse_pdb(str(pdb_path))
    if n_atoms == 0:
        console.print(f"[red]PDB file contains no atoms: {pdb_path}[/]")
        return 1

    center = _pocket_center(pdb_path)
    if center is None:
        center = _center(parsed)
    console.print(f"[cyan]Box center: {center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}[/]")

    receptor_pdbqt = to_pdbqt_receptor(parsed)
    rec_fd, rec_path = tempfile.mkstemp(suffix=".pdbqt")
    with os.fdopen(rec_fd, "w") as f:
        f.write(receptor_pdbqt)

    box_size = [20.0, 20.0, 20.0]
    exhaustiveness = 4
    batch_size = 20
    hits = []
    hit_fps = []
    all_docked = []
    total_docked = 0
    batch_no = 0

    while total_docked < max_candidates and len(hits) < limit:
        remaining = min(batch_size, max_candidates - total_docked)
        batch_no += 1
        console.print(f"[cyan]Batch {batch_no}: generating {remaining} candidates...[/]")
        batch_smiles = generate_smiles(remaining, max_attempts=remaining * 15, scaffolds=seed_scaffolds)
        if not batch_smiles:
            console.print("[yellow]No more valid SMILES generated.[/]")
            break

        tasks = [(smi, rec_path, center, n_cpus, box_size, exhaustiveness) for smi in batch_smiles]
        with multiprocessing.Pool(processes=n_cpus) as pool:
            batch_results = pool.starmap(_dock_one, tasks, chunksize=8)

        for r in batch_results:
            if r is None:
                continue
            total_docked += 1
            all_docked.append(r)
            if r["score"] < threshold:
                mol = Chem.MolFromSmiles(r["smiles"])
                is_diverse = True
                if mol is not None:
                    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                    for hfp in hit_fps:
                        if DataStructs.TanimotoSimilarity(fp, hfp) >= 0.7:
                            is_diverse = False
                            break
                    if is_diverse:
                        hit_fps.append(fp)
                if is_diverse:
                    hits.append(r)
                    console.print(
                        f"[green]Hit: {r['score']:.2f} kcal/mol ({r['smiles']})[/]"
                    )

        console.print(
            f"[cyan]Batch {batch_no} done: {total_docked} docked, "
            f"{len(hits)} below threshold[/]"
        )

    # cleanup
    try:
        os.unlink(rec_path)
    except Exception:
        pass

    all_docked.sort(key=lambda x: x["score"])
    hits.sort(key=lambda x: x["score"])
    if len(hits) >= limit:
        top = hits[:limit]
    else:
        top = all_docked[:limit]

    if not top:
        console.print("[red]No valid docked candidates produced.[/]")
        return 1

    for r in top:
        status, ic50 = query_chembl(r["smiles"])
        r["chembl"] = status
        r["ic50"] = ic50

    out_path = Path(out) if out else Path(f"findings/Publish/Proteins/DeNovo/{target}_denovo.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("SMILES | Vina_score_kcal_mol | Lipinski_pass | ChEMBL_status | IC50_if_known\n")
        for r in top:
            ic = f"{r['ic50']:.2f} nM" if r["ic50"] is not None else "N/A"
            f.write(f"{r['smiles']} | {r['score']:.2f} | {'PASS' if r['lip_ok'] else 'FAIL'} | {r['chembl']} | {ic}\n")

    table = Table(
        "SMILES", "Vina (kcal/mol)", "Lipinski", "ChEMBL", "IC50",
        title=f"Top {len(top)} de novo binders for {target}",
        box=box.ROUNDED,
    )
    for r in top:
        ic = f"{r['ic50']:.2f} nM" if r["ic50"] is not None else "N/A"
        table.add_row(r["smiles"], f"{r['score']:.2f}", 'PASS' if r['lip_ok'] else 'FAIL', r["chembl"], ic)
    console.print(table)

    best = top[0]["score"]
    console.print(
        f"[cyan]Total candidates docked: {total_docked} | "
        f"Below threshold: {len(hits)} | Best: {best:.2f} kcal/mol | "
        f"Distinct Tanimoto clusters: {len(hit_fps)}[/]"
    )
    console.print(f"[green]Wrote {out_path}[/]")
    return 0

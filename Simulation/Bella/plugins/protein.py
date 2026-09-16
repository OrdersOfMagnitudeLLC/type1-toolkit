# OOM Commercial License v1.0
# Copyright 2026 Orders of Magnitude LLC
# See /NS/LICENSE.md for terms.
# Copyright (C) 2026 Orders of Magnitude LLC <orders@ofmagnitude.com>
"""
Protein search plugin for Bella (UniProt + PDB)
"""
import re
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Try to import requests, install if not available
try:
    import requests
except ImportError:
    import subprocess
    subprocess.run(['pip3', 'install', '--break-system-packages', 'requests'], check=True, capture_output=True)
    import requests


def name() -> str:
    return "protein"


def _parse_uniprot(data: dict, limit: int) -> list[dict]:
    results = []
    for entry in data.get('results', [])[:limit]:
        uniprot_id = entry.get('primaryAccession', '')
        if not uniprot_id:
            continue

        # recommended full name
        protein_name = ''
        try:
            protein_name = entry['proteinDescription']['recommendedName']['fullName']['value']
        except (KeyError, TypeError):
            pass

        organism = ''
        try:
            organism = entry['organism']['scientificName']
        except (KeyError, TypeError):
            pass

        seq_len = entry.get('sequence', {}).get('length', 0)

        # function and disease from comments
        function = ''
        disease = ''
        for comment in entry.get('comments', []):
            cmt_type = comment.get('commentType', '')
            if cmt_type == 'FUNCTION':
                texts = [t.get('value', '') for t in comment.get('texts', [])]
                if texts:
                    function = ' '.join(texts)
            elif cmt_type == 'DISEASE':
                try:
                    disease = comment.get('disease', {}).get('diseaseId', '')
                except (KeyError, TypeError):
                    pass

        results.append({
            'formula': uniprot_id,
            'source': 'uniprot',
            'type': 'protein',
            'properties': {
                'protein_name': protein_name,
                'organism': organism,
                'sequence_length': seq_len,
                'function': function,
                'disease_association': disease,
                'uniprot_id': uniprot_id,
            }
        })
    return results


def _uniprot_search(query: str, limit: int) -> list[dict]:
    try:
        # UniProt requires field-qualified queries for complex terms; wrap each token
        # in protein_name: and OR them, capping at 5 terms to keep URLs short.
        terms = [t for t in query.split() if t][:5]
        if not terms:
            return []
        query = ' OR '.join(f'protein_name:{t}' for t in terms)
        url = f"https://rest.uniprot.org/uniprotkb/search?query={requests.utils.quote(query)}&format=json&size={limit}"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        return _parse_uniprot(resp.json(), limit)
    except Exception as e:
        import sys
        print(f"UniProt API Error: {e}", file=sys.stderr)
        return []


def _pdb_search(query: str, limit: int) -> list[dict]:
    try:
        search_url = "https://search.rcsb.org/rcsbsearch/v2/query"
        body = {
            "query": {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "struct.title",
                    "operator": "contains_phrase",
                    "value": query
                }
            },
            "return_type": "entry",
            "request_options": {
                "paginate": {"start": 0, "rows": limit}
            }
        }
        resp = requests.post(search_url, json=body, timeout=20)
        resp.raise_for_status()
        if not resp.text:
            return []
        data = resp.json()

        results = []
        entries = data.get('result_set', []) or data.get('results', [])
        for entry in entries[:limit]:
            pdb_id = entry.get('identifier') or entry.get('id')
            if not pdb_id:
                continue

            title = ''
            resolution = None
            method = ''
            release_date = ''

            # Fetch summary metadata from RCSB Data API
            try:
                meta_url = f"https://data.rcsb.org/rest/v1/entry/{pdb_id}"
                meta = requests.get(meta_url, timeout=15).json()

                struct = meta.get('struct', {})
                title = struct.get('title', '')

                rcsb = meta.get('rcsb_entry_info', {})
                res = rcsb.get('resolution_combined', [])
                if res:
                    resolution = float(res[0])

                exptl = meta.get('exptl', [])
                if exptl:
                    method = exptl[0].get('method', '')

                acc = meta.get('rcsb_accession_info', {})
                release_date = acc.get('initial_release_date', '')
            except Exception:
                pass

            results.append({
                'formula': pdb_id,
                'source': 'pdb',
                'type': 'protein',
                'properties': {
                    'protein_name': title,
                    'organism': '',
                    'sequence_length': 0,
                    'function': '',
                    'pdb_id': pdb_id,
                    'resolution_A': resolution,
                    'experimental_method': method,
                    'release_date': release_date,
                }
            })
        return results
    except Exception as e:
        import sys
        print(f"PDB API Error: {e}", file=sys.stderr)
        return []


def search(query: str, limit: int = 20) -> list[dict]:
    """Search UniProt and PDB in parallel for protein entries."""
    results = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_uni = executor.submit(_uniprot_search, query, limit)
        future_pdb = executor.submit(_pdb_search, query, limit)
        results.extend(future_uni.result())
        results.extend(future_pdb.result())
    return results

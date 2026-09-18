# Copyright (c) 2026 Orders of Magnitude LLC
# Licensed under the OOM Commercial License v1.0
# See LICENSE.md in the repository root or ofmagnitude.com

"""
RCSB PDB plugin for Bob
Searches the Protein Data Bank for protein structures
"""
import requests
from typing import List, Dict, Any
import re

RCSB_API_URL = "https://search.rcsb.org/rcsbsearch/v2/query"

def name() -> str:
    return "RCSB PDB"

def parse_query(query: str) -> dict:
    """Parse natural language query to RCSB API parameters."""
    params = {}
    query_lower = query.lower()
    
    # Protein type filters
    if "enzyme" in query_lower:
        params["polymer_entity_type"] = "protein"
    elif "dna" in query_lower:
        params["polymer_entity_type"] = "dna"
    elif "rna" in query_lower:
        params["polymer_entity_type"] = "rna"
    
    # Resolution filter
    if "high resolution" in query_lower:
        params["resolution_max"] = 2.0
    elif "low resolution" in query_lower:
        params["resolution_min"] = 3.0
    
    # Experimental method
    if "x-ray" in query_lower or "xray" in query_lower:
        params["experimental_method"] = "X-RAY"
    elif "nmr" in query_lower:
        params["experimental_method"] = "SOLUTION NMR"
    elif "cryo-em" in query_lower or "cryoem" in query_lower:
        params["experimental_method"] = "ELECTRON MICROSCOPY"
    
    # Organism filter
    organism_map = {
        "human": "Homo sapiens",
        "mouse": "Mus musculus",
        "rat": "Rattus norvegicus",
        "e coli": "Escherichia coli",
        "yeast": "Saccharomyces cerevisiae",
        "bacteria": "Bacteria",
        "virus": "Virus"
    }
    
    for key, organism in organism_map.items():
        if key in query_lower:
            params["organism_scientific"] = organism
            break
    
    return params

def search(query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Search RCSB PDB for proteins matching the query
    
    Args:
        query: Natural language search query
        limit: Maximum number of results to return
        
    Returns:
        List of protein dictionaries with properties
    """
    # Build RCSB query - use simple text search
    rcsb_query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_container_identifiers.title",
                        "operator": "contains_phrase",
                        "value": query
                    }
                }
            ]
        },
        "return_type": "entry"
    }
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    try:
        response = requests.post(RCSB_API_URL, json=rcsb_query, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        results = []
        if isinstance(data, dict) and "result_set" in data:
            for item in data["result_set"][:limit]:
                pdb_id = item.get("identifier", "")
                
                # Fetch detailed structure info
                struct_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
                struct_response = requests.get(struct_url, timeout=5)
                struct_data = struct_response.json() if struct_response.status_code == 200 else {}
                
                # Extract properties
                properties = {
                    "resolution": struct_data.get("rcsb_entry_info", {}).get("resolution_combined", [None])[0],
                    "experimental_method": struct_data.get("exptl", [{}])[0].get("method", ""),
                    "polymer_entity_count": struct_data.get("rcsb_entry_info", {}).get("polymer_entity_count_protein", 0),
                    "deposition_date": struct_data.get("rcsb_accession_info", {}).get("deposit_date", "")
                }
                
                # Fetch PDB file
                pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
                pdb_response = requests.get(pdb_url, timeout=10)
                pdb_path = None
                if pdb_response.status_code == 200:
                    import os
                    pdb_dir = "/tmp/bella_pdb"
                    os.makedirs(pdb_dir, exist_ok=True)
                    pdb_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
                    with open(pdb_path, 'w') as f:
                        f.write(pdb_response.text)
                
                results.append({
                    "name": pdb_id,
                    "title": struct_data.get("struct", {}).get("title", ""),
                    "properties": properties,
                    "source": "rcsb_pdb",
                    "pdb_path": pdb_path
                })
        
        return results
        
    except requests.exceptions.RequestException as e:
        print(f"RCSB PDB API error: {e}")
        return []

# Copyright (c) 2026 Orders of Magnitude LLC
# Licensed under the OOM Commercial License v1.0
# See LICENSE.md in the repository root or ofmagnitude.com

#!/usr/bin/env python3
"""
Bob Universal Search Interface
Plugin-based material and protein search with property-first ranking
"""
import sys
from plugins import load_plugins, unified_search

def rank_by_properties(results, priority_props=None):
    """
    Rank results by properties using Bob's property-first logic
    
    Args:
        results: List of result dictionaries from plugins
        priority_props: List of property names to prioritize (default: band_gap, formation_energy)
        
    Returns:
        Sorted list of results
    """
    if priority_props is None:
        priority_props = ["band_gap", "formation_energy", "density"]
    
    def score(result):
        """Calculate ranking score for a result"""
        props = result.get("properties", {})
        score = 0
        
        # Higher bandgap is better for semiconductors
        if "band_gap" in props and props["band_gap"] is not None:
            score += props["band_gap"] * 10
        
        # Lower formation energy is better (more stable)
        if "formation_energy" in props and props["formation_energy"] is not None:
            score -= abs(props["formation_energy"]) * 5
        
        # Lower density is better for lightweight materials
        if "density" in props and props["density"] is not None:
            score -= props["density"] * 2
        
        return score
    
    # Sort by score descending
    return sorted(results, key=score, reverse=True)

def main():
    """Main search interface"""
    if len(sys.argv) < 2:
        print("Usage: python bob_search.py <query> [limit]")
        print("Example: python bob_search.py 'wide bandgap stable silicon' 10")
        sys.exit(1)
    
    query = " ".join(sys.argv[1:-1]) if len(sys.argv) > 2 else sys.argv[1]
    limit = int(sys.argv[-1]) if len(sys.argv) > 2 and sys.argv[-1].isdigit() else 20
    
    print(f"Bob Universal Search")
    print(f"Query: {query}")
    print(f"Limit: {limit}")
    print("=" * 60)
    
    # Load plugins
    plugins = load_plugins()
    print(f"Loaded {len(plugins)} plugins: {', '.join(plugins.keys())}")
    print()
    
    # Perform unified search
    results = unified_search(query, limit=limit)
    
    if not results:
        print("No results found")
        return
    
    # Rank by properties
    ranked_results = rank_by_properties(results)
    
    # Display results
    print(f"Found {len(ranked_results)} results (ranked by properties)")
    print()
    
    for i, result in enumerate(ranked_results[:limit], 1):
        print(f"{i}. {result.get('formula', result.get('name', 'unknown'))}")
        print(f"   Source: {result.get('source', 'unknown')}")
        
        props = result.get("properties", {})
        if props:
            print("   Properties:")
            for key, value in props.items():
                if value is not None:
                    print(f"     {key}: {value}")
        
        if "material_id" in result:
            print(f"   Material ID: {result['material_id']}")
        
        print()

if __name__ == "__main__":
    main()

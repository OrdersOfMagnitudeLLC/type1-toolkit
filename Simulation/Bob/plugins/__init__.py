# Copyright (c) 2026 Orders of Magnitude LLC
# Licensed under the OOM Commercial License v1.0
# See LICENSE.md in the repository root or ofmagnitude.com

"""
Bob Plugin System
Search-only plugin interface for material and protein data sources
"""
import os
import importlib.util
from pathlib import Path
from typing import List, Dict, Any

class PluginBase:
    """Base class for Bob plugins"""
    
    @staticmethod
    def name() -> str:
        """Return plugin name"""
        raise NotImplementedError
    
    @staticmethod
    def search(query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search for materials/proteins matching the query
        
        Args:
            query: Natural language search query
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries with material/protein properties
            Each dict must contain at least:
            - 'formula' or 'name': Identifier
            - 'properties': Dict of numeric properties for Bob's ranking
            - 'source': Plugin name
        """
        raise NotImplementedError

def load_plugins(plugin_dir: str = None) -> Dict[str, PluginBase]:
    """
    Load all plugins from the plugins directory
    
    Args:
        plugin_dir: Path to plugins directory (default: ./plugins)
        
    Returns:
        Dict mapping plugin names to plugin instances
    """
    if plugin_dir is None:
        plugin_dir = os.path.join(os.path.dirname(__file__))
    
    plugins = {}
    plugin_path = Path(plugin_dir)
    
    for py_file in plugin_path.glob("*.py"):
        if py_file.name.startswith("_"):
            continue
            
        module_name = py_file.stem
        spec = importlib.util.spec_from_file_location(module_name, py_file)
        
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Look for plugin class or module-level search function
            if hasattr(module, 'Plugin'):
                plugin = module.Plugin()
                plugins[plugin.name()] = plugin
            elif hasattr(module, 'search'):
                # Module-level search function (simple interface)
                plugins[module_name] = type('SimplePlugin', (PluginBase,), {
                    'name': staticmethod(lambda: module_name),
                    'search': staticmethod(module.search)
                })()
    
    return plugins

def unified_search(query: str, limit: int = 20, plugin_dir: str = None) -> List[Dict[str, Any]]:
    """
    Search across all loaded plugins and deduplicate results
    
    Args:
        query: Natural language search query
        limit: Maximum results per plugin
        plugin_dir: Path to plugins directory
        
    Returns:
        Combined and deduplicated results from all plugins
    """
    plugins = load_plugins(plugin_dir)
    all_results = []
    
    for plugin_name, plugin in plugins.items():
        try:
            results = plugin.search(query, limit=limit)
            all_results.extend(results)
        except Exception as e:
            print(f"Error in plugin {plugin_name}: {e}")
    
    # Deduplicate by formula/name
    seen = set()
    deduplicated = []
    for result in all_results:
        key = result.get('formula') or result.get('name')
        if key and key not in seen:
            seen.add(key)
            deduplicated.append(result)
    
    return deduplicated

# OOM Commercial License v1.0
# Copyright 2026 Orders of Magnitude LLC
# See /NS/LICENSE.md for terms.
# Copyright (C) 2026 Orders of Magnitude LLC <orders@ofmagnitude.com>
import os
"""
Catalysis Hub + literature benchmark plugin for Bella.

This wrapper loads the shared Bob catalysis-hub implementation so the search
logic lives where the user requested it, while keeping the Bella plugin
interface bella.py expects.
"""
import importlib.util
import sys

# Load Bob/bob.py directly without requiring Bob to be an installed package
_BOB_PATH = os.path.join(os.path.expanduser('~'), "NS/Bob/bob.py")
_spec = importlib.util.spec_from_file_location("_bob_catalysis_hub", _BOB_PATH)
_bob = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bob)

name = _bob.name
search = _bob.search

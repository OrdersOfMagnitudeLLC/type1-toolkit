# Copyright (c) 2026 Orders of Magnitude LLC
# Licensed under the OOM Commercial License v1.0
# See LICENSE.md in the repository root or ofmagnitude.com

#!/usr/bin/env python3
"""NSMace --target-aware wrapper.

Translates the --target flag into the appropriate MACE force-field path and
forwards all other arguments to the compiled NSMace engine.
"""
import argparse, os, sys, subprocess, pathlib

ROOT = pathlib.Path(__file__).parent.resolve()
ENGINE = ROOT / 'build' / 'NSMace'
if not ENGINE.exists():
    ENGINE = ROOT / 'NSMace'

TARGET_MODELS = {
    'organic': 'mace-off23',
    'biomolecular': 'mace-off23',
    'protein': 'mace-off23',
    'crystal': 'mace-mp-0',
    'fusion-plasma': 'mace-mp-0',
    'atmospheric': 'mace-mp-0',
    'abiogenesis': 'mace-off23',
    'aging': 'mace-off23',
}


def main(argv=None):
    argv = list(argv or sys.argv[1:])
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--target', default=None, choices=list(TARGET_MODELS.keys()))
    ns, remaining = parser.parse_known_args(argv)

    model = TARGET_MODELS.get(ns.target, 'mace-mp-0') if ns.target else None
    if model:
        os.environ['MACE_MODEL'] = model
        print(f"[NSMace target] {ns.target} -> {model}", file=sys.stderr)

    if not ENGINE.exists():
        print(f"[NSMace target] engine not found: {ENGINE}", file=sys.stderr)
        sys.exit(1)

    cmd = [str(ENGINE)] + remaining
    return subprocess.call(cmd)


if __name__ == '__main__':
    sys.exit(main())

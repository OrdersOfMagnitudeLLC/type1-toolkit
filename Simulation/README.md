# Simulation

**by Orders of Magnitude · ofmagnitude.com**

Molecular simulation and materials discovery pipeline.

## Components

**Bella** — Universal simulator. Atoms to answers. Materials, proteins,
PDEs, and math from a single CLI. No GPU required.

**Bob** — Materials and protein search engine. Queries GNoME, Materials
Project, UniProt, PDB. Feeds candidates to Bella.

**NSMace** — C++ MACE force field, compiled from source on install.
Native CPU performance for fast phonon screening.

## Install

```bash
cd Bella && bash install.sh
bella --help
```

## Cache Directories

The following populate automatically on first use — not included in repo:
- `~/.bella/cif_cache/` — downloaded CIF/PDB structures
- `~/.bella/phonon_runs/` — phonon calculation outputs
- `~/.bella/screening_runs/` — screening pipeline outputs
- `~/.bella/data/` — database downloads (UniProt, DisGeNET)

## License

AGPLv3 · OOM Commercial License for commercial use
https://ofmagnitude.com/license

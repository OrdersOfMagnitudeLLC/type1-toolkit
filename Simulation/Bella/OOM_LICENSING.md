# Bella: OOM Licensing Architecture
## DO NOT MODIFY THIS FILE WITHOUT EXPLICIT INSTRUCTION FROM THE AUTHOR

### Engine: DFT-FE (LGPL v2.1+)
- Source lives in ~/NS/Bella/engine/
- LGPL means: any modification to DFT-FE source files must also be LGPL
- RULE: Never modify files inside engine/src/ directly
- RULE: Never add OOM proprietary code inside engine/ directory
- RULE: DFT-FE source files are read-only reference: no edits, ever

### Bella Core (Proprietary: All Rights Reserved, OOM LLC)
- bella_chefsi_v2.py and future .cpp rewrite = proprietary algorithm
- bella_pyscf_adapter.py = proprietary
- Bob integration layer = proprietary
- Vacuum mask generation logic = proprietary
- License: AGPL3 + OOM commercial addendum
- RULE: These files never go inside engine/ directory

### Integration Layer (LGPL: clean boundary)
- BellaSubspaceSolver class = new file added to DFT-FE build
- Implements DFT-FE's internal solver interface
- Internally calls Bella CheFSI via function pointer / header include
- Lives in: ~/NS/Bella/integration/
- This file is LGPL by necessity (touches DFT-FE interface)
- RULE: Integration layer contains NO algorithm logic: only the handshake

### Vacuum Skip (Proprietary → LGPL boundary)
- Vacuum detection and mask generation: Bella proprietary code
- Mask is passed to DFT-FE via existing mesh/cell API: zero DFT-FE modification
- RULE: Vacuum logic lives in ~/NS/Bella/vacuum/ not in engine/

### What This Means In Practice
- OOM sells: Bella CheFSI algorithm + Bob + vacuum logic (proprietary)
- DFT-FE engine: open, LGPL, users can inspect
- Integration shim: open, LGPL, contributes back to DFT-FE community
- The moat is the algorithm. Not the wrapper.

### Red Lines (Never Cross These)
1. Never add proprietary algorithm code inside engine/
2. Never modify engine/src/ files
3. Never combine CheFSI algorithm logic with the integration shim in one file
4. Never remove this file

### Documentation Structure Rules
- findings/README.txt is for architecture notes only
- Benchmark results go in findings/<name>.txt only: never in README.txt
- This separation keeps architecture documentation clean and benchmark data organized

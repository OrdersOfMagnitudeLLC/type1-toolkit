# Bella 0.2.0: Feature & Command Reference

Last updated: 2026-08-14

Bella: Lightweight Universal Simulator. It coordinates the NSMace machine-learning force field, the SPARC DFT engine, the Bob discovery plugins, and the Materials Project API.

## Installation

```bash
cd /path/to/Bella
bash install.sh
```

- Installs the Python package (core deps only; `mace-torch` and `pyvista` are optional extras).
- Creates `~/.bella/` (`profiles`, `cache`, `cif_cache`, `bin`, `mace`).
- Writes a `~/.bella/.env` template for API keys.
- Compiles the NSMace C++ engine to `~/.bella/bin/NSMace` when `cmake` and `g++` are available; otherwise falls back to Python MACE.

## Welcome & first run

`bella` with no arguments shows the welcome screen and the command list. After the first run, a `~/.bella/.welcomed` marker is created and only a compact reminder plus the command list is shown.

`bella --help` also prints the welcome banner as the help header.

---

## Commands

All commands are invoked as `bella <command> [args]`.

### Simulation & screening

#### `run <path> [--engine {mace,sparc,both}] [--mpi-nodes N]`
Auto-detects the input and dispatches to the appropriate command.

- `<path>` may be a `.cif` file, `.pdb` file, `.xyz` file, or a folder.
- `--engine` selects the engine for `.xyz` files: `mace`, `sparc`, or `both` (default: `both`).
- `--mpi-nodes` is planning-only mode: estimates time/cost for an MPI run and prints a projection; it does not actually run in parallel.
- Headless-safe for `.cif` and `.pdb` batch inputs.
- Does **not** invoke the discovery pipeline; it runs a single structure.

#### `batch-screen <folder|file> [--screen-only]`
Screen one or many CIF files with NSMace and rank by energy.

- `--screen-only` skips force computation and runs the faster energy-only path.
- Prints a ranked `Materials Screening Results` table.
- Headless-safe.

#### `protein <pdb_file>`
Run NSMace on a single PDB protein file using the `--large-system` chunked path.

- Parses the PDB, counts elements, and prints the total energy and timing.
- Uses a 5-minute (300 s) timeout.
- Headless-safe.

All long-running commands print a `⏱  Estimated: ...` banner before computation. Suppress it with `--no-eta`.

#### `simulate <target>`
Lightweight universal simulation targets. Offline, CPU-first, 16GB RAM, no API keys.

- `diffusion`: 1D/2D diffusion PDE (alias for `bella sim --equations diffusion`).
- `wave`: 1D/2D wave PDE (alias for `bella sim --equations wave`).
- `reaction-diffusion`: 1D/2D Gray-Scott PDE (alias for `bella sim --equations reaction-diffusion`).
- `fusion-plasma` now runs a resistive nonlinear MHD time-stepper with SfePy (`Mesh`/`Field` setup + `Newton` nonlinear solve); supports `--preset iter`, `--grid N`, and `--duration T`.
- `abiogenesis` builds a coarse lipid/RNA/water system and reports vesicle self-assembly metrics; flags RunPod target if >50K atoms.
- `atmospheric` solves a linearized Navier-Stokes convective instability for a pressure/temperature/humidity column; supports `--region {tropical,arctic,urban}`, `--turbulence` (k-epsilon RANS solved with SfePy), `--radiation` (two-stream CO2 forcing), and `--co2`.
- `clean-water` Bob + NSMace search for membrane candidates; `--preset {seawater,brackish,greywater}`.
- `nitrogen-fixation` Bob + NSMace search for transition-metal catalysts.
- `carbon-capture` Bob + NSMace search for porous MOF/CO2 capture materials.
- `soil-microbiome` Bob search for beneficial soil bacteria enzymes.
- PDE presets support `--dim 1|2`, `--grid N`, `--duration T`, `--stochastic`, and `--jobs N`.
- All print an ETA before running; use `--no-eta` to suppress.
- Writes `findings/YYYY-MM-DD/sim_<target>_<timestamp>.json` for every run.
- Headless-safe.

#### `math <target>`
Mathematical / number-theory kill-tests.

- `riemann` enumerates primes up to `N` (default `10^8`), computes prime gaps, Chebyshev theta samples, and Mertens samples up to a 10M cap. Prints timing and a `<60s` kill-test pass flag.
- `riemann --resume` resumes from the latest `findings/riemann_checkpoint_<count>.json`; if none exists it starts fresh. Checkpoints are written every `10^9` primes with `(prime_count, max_gap, current_prime)` validation output.

#### `sim --equations "..."`
Custom PDE/ODE solver from an equation string. Supports diffusion, wave, and Gray-Scott reaction-diffusion in 1D or 2D; uses `py-pde` (`CartesianGrid`, `ScalarField`, `PDE`, `FieldCollection`) as the solver backbone.

- Examples: `bella sim --equations "du/dt = 0.1*laplacian(u)" --dim 1 --duration 10`
- `--stochastic` adds an Itô Langevin noise term to diffusion, reaction-diffusion, and coupled targets.
- `--jobs N` parallelizes the `py-pde` solve across `N` loky workers (`-1` for all cores).
- `--coupled {thermal-structural|em-thermal}` runs coupled multi-physics.
- Outputs `.npy` field files and `findings/sim_pde_*.json`.

### Search & discovery

#### `proteins <query> [--limit N] [--show]`
Search UniProt and RCSB PDB for proteins.

- `--limit N` caps the number of results (default: 20).
- `--show` downloads the AlphaFold PDB (if available) and opens the 3D protein viewer.
- Requires network access.
- `--show` requires a display/PyVista.

#### `fetch-screen <query>`
Fetch CIFs from the Materials Project and screen them.

- **Current behavior:** the MP CIF endpoint is not used; the command falls back to copying the first 5 `.cif` files from `Bob/data/cifs` into `/tmp/bella_mp` and then runs `batch-screen` on them.
- Headless-safe.

#### `search <query>`
Unified search across all loaded Bob plugins with threading, deduplication, and ranking by formation energy / bandgap.

- Prints a `Unified Search Results` table.
- Headless-safe.

#### `plugins {list|search} [query]`
- `list`: show all loaded plugins and their load status.
- `search <query>`: run the query through every plugin and show a combined result table.
- Headless-safe.

#### `discover <query> [options]`
The full automated discovery pipeline: **Bob search → CIF resolve → NSMace screen → SPARC DFT confirm → literature check → phonon stability check → findings output**.

- `--sparc-survivors N`: number of top NSMace survivors to run through SPARC (default: 5).
- `--sparc-ram-limit MB`: RAM cap for each SPARC worker in MB (default: half of system RAM; use `psutil` if available, otherwise 4000).
- `--sparc-timeout S`: per-material SPARC timeout in seconds (default: 1800).
- `--sparc-quality {screen,confirm}`: `screen` (fast, single-point, loose SCF) or `confirm` (relaxed, tighter SCF, default: `screen`).
- `--search-only`: stop after the Bob search, print the ranked table, and save `findings/search_*.json`.
- `--limit N`: Bob candidate cap. Default is 20 with `--search-only`, 50 otherwise.
- `--show`: open the 3D viewer / animation for confirmed materials; also fetches AlphaFold and previews protein hits.
- `--resume`: force resume from the latest `findings/checkpoint_*.json`.
- `--no-resume`: start a fresh run and ignore any checkpoints.

Pipeline stages:
1. **Bob search** across all material plugins (GNoME, Materials Project, local, etc.) in parallel.
2. **Deduplication & ranking** by source priority (MP first) and formation energy.
3. **CIF resolution** (MP API → local cache → Bob data → GNoME `by_id.zip` one-time download).
4. **NSMace screen** (8 workers) with SQLite caching in `cache/nsmace_cache.db`.
5. **SPARC confirmation** on the smallest `sparc_survivors` candidates with a 120 s per-material timeout and lanthanide PSP skip.
6. **POSCAR + Quantum ESPRESSO input export** for confirmed materials to `findings/exports/{formula}/`.
7. **Literature cross-check** (COD + Materials Project) for novelty scoring.
8. **Phonon stability** via `shifu_phonon.py` for systems with `n_atoms >= 4` and `<= 50`.
9. **Checkpoint/resume**: an initial checkpoint is written before phonon, then updated after every completed material.
10. **Findings JSON** saved to `findings/YYYY-MM-DD/discover_{query}_{timestamp}.json` and an audit JSON to `findings/YYYY-MM-DD/runs/run_{timestamp}.json`.

`--show` requires a display. Resume prompts require a TTY unless `--resume` or `--no-resume` is used.

### Visualization & analysis

#### `view <material_id> [--style {ball-stick,spacefill,wireframe,surface,ribbon}] [--protein <id>] [--headless]`
Display a material or protein in 3D.

- `<material_id>` is a formula or ID. Bella looks it up in `findings/**/*.json`, then in `cif_cache/<id>.cif` or `cif_cache/<id>.pdb`.
- `--style` sets the rendering style (`ball-stick`, `spacefill`, `wireframe`, `surface`, `ribbon`). `ribbon` is protein-only; for crystals it falls back to the default crystal view with a note.
- `--protein <id>` renders the material and an AlphaFold protein side by side.
- `--headless` renders off-screen and exits (no interactive window).
- Supports `--export-glb` to export a GLB file to `findings/YYYY-MM-DD/`.
- Requires PyVista, ASE, and a display (`PYVISTA_AVAILABLE`).
- In headless environments without a display, the command falls back or warns.

#### `benchmark [--sparc] [--phonon]`
Reproducible performance benchmark over five fixed Materials Project IDs:
`mp-1524805`, `mp-2351841`, `mp-971661`, `mp-971662`, `mp-988210`.

- Times CIF fetch, NSMace screen, SPARC (with `--sparc`), and phonon (with `--phonon`, requires `--sparc`).
- Prints a `Bella Benchmark` table and a total time.
- Does **not** write findings output.
- Headless-safe (no 3D output).

### Utilities

#### `cache-stats`
Show statistics for the NSMace SQLite cache (`cache/nsmace_cache.db`):
entry count, unique formulas, oldest/newest timestamps, and the 10 most recent entries.
Headless-safe.

#### `phonons <cif> [--suite] [--sparc-phonon] [--no-socket] [--np N] [--pressures P...] [--max-ram-gb N]`
Run the full phonon pipeline on a CIF file at the given pressures (default: 0.0 GPa).

- `--suite` runs multiple pressures sequentially with automatic **density restart cache**. The first run is cold; subsequent runs load the cached SPARC density and restart from a pre-converged state, giving **~5× faster** warm runs.
- `--sparc-phonon` uses SPARC DFT for the finite-displacement force matrix instead of MACE.
- Caches reference `.dens` and `.restart` files under `~/.bella/density_cache/<cif_hash>/`.
- Headless-safe.

#### `watch <query> [--interval S] [--sparc-survivors N] [--max-runs N] [--protein-query Q]`
Autonomous scheduled discovery loop that re-runs `discover`.

- `--interval S`: seconds between runs (default: 3600).
- `--sparc-survivors N`: passed to `discover` (default: 3).
- `--max-runs N`: stop after N runs; 0 means unlimited (default: 0).
- `--protein-query Q`: additionally run `proteins` each cycle and merge results.
- Saves each run to `findings/discover_{query}_{timestamp}.json`.
- Catch-up phonon pass for confirmed materials missing a phonon result.
- Headless-safe when not using `--show`.
- Press `Ctrl+C` to stop gracefully.

#### `status`
Show the Bella status dashboard:
NSMace binary availability, SPARC binary + PSPs, cache size, the last `discover` run summary, and today's dated findings folder.

- `--memory` shows total/available RAM and the Bella RAM limit.
- `--coverage` regenerates `docs/coverage_matrix.md` and prints the matrix.
- `--platform` prints a cross-platform install health check (OS, Python, RAM, CPU cores, optional dependencies, binaries, and feature availability).
- Headless-safe.

#### `chat`
Conversational split-terminal interface that dispatches `search`, `discover`, `proteins`, `combined`, `view`, `results`, and `status` tools via an LLM.

- Requires an LLM API key in `~/.bella/.env`. Supports Anthropic, OpenAI, OpenRouter, and custom OpenAI-compatible endpoints via `BELLA_LLM_PROVIDER`, `BELLA_LLM_API_KEY`, `BELLA_LLM_MODEL`, and `BELLA_LLM_BASE_URL`.
- Requires an interactive TTY.
- Not headless-safe.

#### `report [--format {csv,md,both}] [--out path]`
Aggregate all `findings/**/*.json` files and export a Markdown and/or CSV report.

- `--format` chooses `csv`, `md`, or `both` (default: `both`).
- `--out path` overrides the Markdown path; the CSV is written next to it.
- Aggregate report files are written to `findings/YYYY-MM-DD/`.
- Includes confirmed materials, protein screens, and run history.
- Headless-safe.

#### `query '<filter>'`
Search across all findings JSON with filter expressions joined by AND logic.

- Supported operators: `=`, `!=`, `>`, `<`, `>=`, `<=`.
- Example: `bella query 'novel=true phonon_stable=true'`.
- Headless-safe.

#### `help [command]`
Show the rich command reference, or detailed help for one command.
Headless-safe.

---

## Data & output layout

- `findings/discover_{query}_{timestamp}.json`: full `discover` results (`results`, `protein_results`, `binding_result`, `screened_count`, `confirmed_count`, `protein_count`, `timing`).
- `findings/search_{query}_{timestamp}.json`: `search` / `--search-only` outputs.
- `findings/runs/run_{timestamp}.json`: per-run audit trail (`materials`, stage counts, timing).
- `findings/checkpoint_{timestamp}.json`: resumable checkpoint state.
- `findings/exports/{formula}/`: VASP `POSCAR` and Quantum ESPRESSO `.pwi` inputs.
- `findings/animations/{formula}_{timestamp}.mp4`: animation exports from `view --animate`.
- `cif_cache/`: cached CIFs, AlphaFold PDBs, and the GNoME `gnome_by_id.zip` (~455 MB one-time download).
- `cache/nsmace_cache.db`: SQLite cache of NSMace energy/hash pairs.

---

## Display / headless behavior

| Feature | Headless-safe? | Notes |
|---------|---------------|-------|
| All tables / Rich console output | Yes | Works on any terminal; degrades gracefully to plain text. |
| `discover`, `watch`, `search`, `report`, `query` | Yes | No display required unless `--show` is used. |
| `view`, `proteins --show` | No | Requires PyVista + a display by default; use `view --headless` for off-screen rendering. |
| `discover --show` | No | Opens PyVista 3D viewer and/or animation for confirmed materials. |
| `chat` | No | Requires an interactive TTY (`prompt_toolkit` full-screen app). |
| `run` / `batch-screen` / `protein` | Yes | No 3D output by default. |
| `view --headless` | Yes | Renders off-screen and exits without opening a window. |

---

## Known limitations

- **GNoME CIF resolution:** If a material has no MP or local CIF, Bella downloads the ~455 MB `gnome_by_id.zip` once per process and extracts from it.
- **Phonon checks:** Skipped for unit cells with fewer than 4 atoms. `shifu_phonon.py` runs SPARC and parses `min ω` / `max ω` lines; full phonopy displacement/force-collection is not yet wired.
- **SPARC `screen` mode:** Uses a physics-based mesh spacing tuned for speed and a 120 s per-material timeout; not publication-grade.
- **SPARC lanthanides:** Materials containing lanthanides (La–Lu) are skipped because no pseudopotentials are bundled.
- **Protein animation:** Proteins are capped at 200 atoms for the 3D animation.
- **`fetch-screen`:** The Materials Project CIF endpoint is not currently used; the command falls back to local demo CIFs.
- **`chat`:** Requires an LLM API key (set in `~/.bella/.env`) and an interactive terminal.
- **Resume prompt:** When no TTY is present and neither `--resume` nor `--no-resume` is given, the resume prompt defaults to `n`.

---

## New / updated commands and flags (0.2.0)

### `bella run`
- `bella run <path>` auto-detects input (`.cif`, `.pdb`, `.xyz`, folder).
- `bella run --profile <name> [--dry-run]` runs a saved simulation profile; `--dry-run` prints the command without executing.

### `bella profile`
- `bella profile {list|show|create|edit|clone|delete} <name> [new_name]`.
- `bella profile create <name> --from-yaml PATH` imports an existing YAML profile.
- `bella profile create <name> --sim` creates a simulation profile (e.g. `riemann-deep`).
- `bella profile delete <name>` removes a user profile; built-in profiles cannot be deleted.
- Profile names are user-defined. Any name works except built-in names (`nitrogen-fixation`, `carbon-capture`, etc.).
- Profiles load from `~/.bella/profiles/` (user) and `profiles/` (built-in).

### `bella discover`
- `--profile <name>` is an alias for `--domain`; works with any loaded profile.
- `--max-ram-gb N` sets a RAM limit for the run (default: `BELLA_MAX_RAM_GB`).
- `--funnel-sizes S1,S2,S3` controls the candidate funnel (default: `5000,500,300`).
- If `mace-torch` is not installed, Bella prompts to install it; if declined, `discover` falls back to search-only mode.

### `bella math riemann`
- `--n N` upper limit (default: `10^8`).
- `--max-ram-gb N` RAM limit for the segmented sieve (default: `BELLA_MAX_RAM_GB`).
- `--mertens-cap M` cap for Mertens computation (default: `10^7`).
- `--resume` resumes from the latest checkpoint.

### `bella sim`
- `--stochastic` adds Itô Langevin noise.
- `--jobs N` parallel `py-pde` workers (`-1` for all cores).

### `bella validate`
- `--quick` skips SPARC and runs MACE-only sanity checks.
- `--pde-only` runs only PDE analytical-solution checks.
- `--quick-pde` runs PDE checks on a 16-point/16×16 grid for fast CI.
- `--materials-only` runs only material/phonon checks.
- `--proteins` runs protein structure checks (ACE2, H2O).
- Foam mechanics validation now included in full `bella validate` run:
  - **T50 binding**: BRD4/JQ1 anchor (r=4.2Å, logP=3.5, ΔG < 0)
  - **T71 diagnostic**: BRD4 D_score (Kd_opt=sqrt(C_h×C_d), D > 0)
  - **T73 hydrophobic floor**: BRD4 pocket (ΔG_floor < 0)
  - **Debye temperature**: Fe (B=168GPa, G=81GPa, θ_D 350–650K, known ~470K)
  - **QCD mass gap (T32)**: derived 1.521 GeV vs measured ~1.5 GeV (1.2–1.8 range)
  - **Toxicity membrane**: Aspirin (logP=1.19, MW=180, expect SAFE)
  - **Toxicity mito**: Chlorpromazine (cationic, MW=319, logP=4.9, expect TOXIC)
  - No SPARC or GPU required: pure foam physics checks.

### `bella report`
- `--format {csv,md,both}` chooses export format (default: `both`).
- `--out path` overrides the Markdown output path.

### `bella chat`
- `--message TEXT` (or positional `message`) for non-interactive single-turn mode.
- `--format {text,json,csv,md,pdf}` for non-interactive output.
- `--auto-confirm` confirms and saves a generated profile without prompting.
- Multi-provider LLM support via `~/.bella/.env`: `BELLA_LLM_PROVIDER` (`anthropic`, `openai`, `openrouter`, `custom`), `BELLA_LLM_API_KEY`, `BELLA_LLM_MODEL`, and `BELLA_LLM_BASE_URL` (custom endpoints).

### `bella status`
- `--memory` shows total/available RAM, the Bella RAM limit, and per-operation estimates.
- `--coverage` regenerates `docs/coverage_matrix.md` and prints the matrix.
- Displays today's `findings/YYYY-MM-DD/` subfolder and total JSON count.

### `bella adsorb`
- `bella adsorb <formula> <molecule> [--miller S] [--layers N] [--vacuum A] [--distance A] [--max-atoms N] [--max-ram-gb N]`.
- Supported molecules: `N2`, `H2O`, `CO2`, `H2` (1.5 Å standoff, shorter H–H bond), `Li` (ionic, single orientation, 2.0 Å standoff).
- `--max-atoms N` caps the surface supercell at N atoms.

### `bella neb`
- `bella neb <formula> <molecule> [--miller S] [--layers N] [--vacuum A] [--distance A] [--images N] [--max-atoms N] [--max-ram-gb N]`.
- NEB barrier for adsorbate dissociation on a surface.
- Supported molecules: `N2`, `H2O`.
- `--max-atoms N` caps the surface supercell at N atoms.

### `bella ui`
- `bella ui [--dry-run] [--port N]`.
- `--dry-run` verifies Streamlit is installed and exits.
- `--port N` sets the server port (default: `8501`).

### `bella create`
- `bella create [options]`: Inverse design: create candidate materials or small-molecule binders.
- **Status:** implemented

### `bella drugs`
- `bella drugs <uniprot_id> [--top N]`: Drug repurposing for a UniProt target via ChEMBL and AlphaFold.
- **Status:** implemented

### `bella design`
- `bella design --target TEXT --length N`: De novo protein design stub: LLM generates a FASTA and validates it.
- **Status:** implemented

### `bella signal`
- `bella signal [--download] [--dataset PATH] [--dry-run]`: Radio technosignature pipeline (Breakthrough Listen hit analysis).
- **Status:** implemented

### `bella suggest`
- `bella suggest <question>`: Plain-English explanation of why a pipeline step failed or how to extend Bella.
- **Status:** implemented

### `bella apply`
- `bella apply <diff_file> [--dry-run]`: Apply a unified-diff patch produced by `bella suggest`.
- **Status:** implemented

---

## Environment & configuration

- `BELLA_NSMACE_BIN`: override the NSMace binary path.
- `BELLA_SPARC_BIN`: override the SPARC binary path.
- `BELLA_PSPS_DIR`: override the SPARC pseudopotential directory.
- `BELLA_BOB_DIR`: override the Bob data directory (default: `~/.bella/bob`).
- `BELLA_MAX_RAM_GB`: RAM cap for Bella operations.
- `MATERIALS_PROJECT_API_KEY`: Materials Project API key.
- LLM configuration (for `bella chat`):
  - `BELLA_LLM_PROVIDER`: `anthropic` (default), `openai`, `openrouter`, `custom`
  - `BELLA_LLM_API_KEY`: API key for the chosen provider
  - `BELLA_LLM_MODEL`: model override (defaults: `claude-sonnet-4-6`, `gpt-4o`, `openrouter/auto`)
  - `BELLA_LLM_BASE_URL`: custom OpenAI-compatible endpoint (required for `custom`)
- NSMace resolution order: `BELLA_NSMACE_BIN` → `~/.bella/bin/NSMace` (compiled by `install.sh`) → `../NSMace/build/NSMace` (dev tree) → Python MACE fallback.
- SPARC fallback: `BELLA_SPARC_BIN` → `~/.bella/sparc-engine/lib/sparc` → `sparc-engine/lib/sparc` in the install directory.
- Plugins are loaded from the install `plugins/` directory.
- `install.sh` installs the Python package, creates `~/.bella/` and `~/.bella/.env`, and compiles the NSMace C++ engine when `cmake` + `g++` are available.

## T71: Hippocrates' Reading (Diagnostic Scanner)
Command: bella scan --example
Physics: Kd_optimal = sqrt(C_h * C_d), D_score threshold 0.3
Output: per-biomarker discrimination score + composite disease index
Use: point-of-care diagnostics from first principles, no lab required

## T72: Nasmyth's Lattice (Enamel Remineralization)
Command: bella enamel --supersaturation 2.0 --fluoride 0.2
Physics: r_crit = 2*gamma*V_m / (R*T*ln(S)), Gibbs-Thomson crystal nucleation
Output: critical nucleus radius, remineralization status, fluoride effect
Use: design topical gel formulations, first-principles cavity prevention
Combined: bella enamel --mode drug -> USAG-1 tooth regrowth pipeline

## T50: Schreiber's Boundary (Foam Binding Screen)
Command: bella t50 --pocket-radius 4.2 --domain hydrophobic
Physics: ΔG = -γ_water × r^α_pocket × SASA (Young-Laplace foam binding)
Calibration: BRD4/JQ1 anchor (r=4.2Å, Kd=33nM, logP=3.5)
Output: binding free energy, Kd estimate, domain-specific score
Use: first-principles drug binding prediction, no docking required

## T51: Gibbs-Thomson Melting
Command: bella t51 --formula Fe3Mn4
Physics: T_m(r) = T_bulk × (1 - 2γ/(r×ρ×L_f)), foam surface tension at nanoscale
Output: size-dependent melting point, thermal stability window
Use: nanomaterial thermal stability screening

## Life Walk: Foam Across All Scales
Command: bella life-walk
Physics: Young-Laplace equation applied across 12 orders of magnitude
Output: foam mechanics at each scale (protein → cell → tissue → organ → ecosystem → planet)
Use: pedagogical and cross-scale consistency check

## Foam Mechanics Validation
Command: bella validate (now includes foam suite)
Checks: T50 binding (BRD4/JQ1), T71 diagnostic (D_score), T73 hydrophobic floor,
        Debye temperature (Fe ~470K), QCD mass gap (T32, ~1.5 GeV),
        membrane toxicity (aspirin SAFE), mito toxicity (doxorubicin TOXIC)
No SPARC or GPU required: pure foam physics, milliseconds runtime.

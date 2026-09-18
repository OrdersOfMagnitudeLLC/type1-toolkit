# Bella Installation Guide

**No coding required after install.** Bella is a command-line tool that runs from a few shell commands.

## Prerequisites

- Python 3.10 or newer
- `pip` (usually included with Python)
- `git` (to clone the repository)
- At least 16 GB of RAM for MACE phonons; 32 GB recommended for SPARC confirm-quality runs

## Three-command install

1. **Clone the repository**

   ```bash
   git clone https://github.com/user/NS.git
   cd NS
   ```

2. **Run the install script**

   ```bash
   bash install.sh
   ```

   This installs Bella and Bob as editable Python packages and checks for SPARC.

3. **Verify the install**

   ```bash
   bella --help
   bob --help
   ```

   Both should print usage information.

## API key setup

Bella uses the Materials Project database to compare candidates against known compounds. Get a free API key at <https://materialsproject.org/api>.

Add the key to your shell:

```bash
export MATERIALS_PROJECT_API_KEY=your_key_here
```

To make it permanent, append it to `~/.bashrc`:

```bash
echo 'export MATERIALS_PROJECT_API_KEY=your_key_here' >> ~/.bashrc
source ~/.bashrc
```

## First run walkthrough

### 1. Validate Bella against a known material

```bash
bella validate
```

This runs Fe4N through MACE phonons and checks that the lattice constant and phonon stability match literature.

### 2. Run your first discovery

```bash
bella discover --domain nitrogen-fixation --generative --headless --sparc-quality confirm
```

Bella will:

1. Generate or fetch candidate formulas
2. Screen each with MACE
3. Confirm selected candidates with SPARC (if installed)
4. Run MACE phonons
5. Print BELLA RESULT blocks with confidence scores, uncertainty ranges, and MP benchmark comparisons

### 3. Generate a PDF report

```bash
bella report findings/discover_nitrogen_fixation_*.json
```

A `report_*.pdf` file is created in the current directory.

## Optional: install SPARC

For DFT confirmation, install SPARC from <https://github.com/SPARC-X/SPARC> and make sure `sparc` is on your `PATH`. If SPARC is missing, Bella skips DFT and uses MACE-only screening and phonons.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `bella: command not found` | Run `bash install.sh` again, or check that `~/.local/bin` is in your `PATH`. |
| `MATERIALS_PROJECT_API_KEY missing` | Set the environment variable and re-run. MP benchmark comparisons are skipped without it. |
| `SPARC not found` | Either install SPARC or use `--sparc-quality screen` instead of `confirm`. |
| `GPU not detected` | Bella falls back to CPU automatically. MACE is slower but still works. |
| `pip install fails` | Upgrade pip: `python3 -m pip install --upgrade pip` then re-run `bash install.sh`. |

## What next?

- **Want a GUI?** Run: `pip install streamlit && bella ui` and open `http://localhost:8501`.
- Read `docs/bella-ui.md` for the lightweight Streamlit UI.
- Read `docs/open-webui.md` to connect Bella to Open WebUI.
- Run `bella suggest "how do I add a domain"` for plain-English guidance.
- Run `bella --help` to see all commands.

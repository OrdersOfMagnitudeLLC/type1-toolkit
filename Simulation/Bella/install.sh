#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "  Bella — Lightweight Universal Simulator"
echo "  by Orders of Magnitude · ofmagnitude.com"
echo ""

echo "→ Installing Bob (materials search engine)..."
pip install -e "$SCRIPT_DIR/../Bob" --break-system-packages

echo "→ Installing Bella..."
pip install -e "$SCRIPT_DIR" --break-system-packages

echo "→ Creating user directories..."
mkdir -p ~/.bella/profiles ~/.bella/cache ~/.bella/cif_cache \
  ~/.bella/bin ~/.bella/mace \
  ~/.bella/phonon_runs ~/.bella/data ~/.bella/screening_runs

if [ ! -f ~/.bella/.env ]; then
  cat > ~/.bella/.env << 'EOF'
BELLA_LLM_PROVIDER=anthropic
BELLA_LLM_API_KEY=
BELLA_LLM_MODEL=claude-sonnet-4-6
EOF
  echo "→ Created ~/.bella/.env — add your API key to enable bella chat"
fi

# NSMace
if [ -f "$SCRIPT_DIR/NSMace/install.sh" ]; then
  echo "→ Compiling NSMace (native C++ accelerator)..."
  bash "$SCRIPT_DIR/NSMace/install.sh"
fi

echo ""
echo "  Done. Run: bella"

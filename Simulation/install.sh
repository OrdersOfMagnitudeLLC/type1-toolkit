#!/bin/bash
set -e

echo "Installing Bella + Bob..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "  -> Installing bella-sim from $SCRIPT_DIR/Bella"
pip install -e "$SCRIPT_DIR/Bella" --break-system-packages

echo "  -> Installing bob-search from $SCRIPT_DIR/Bob"
pip install -e "$SCRIPT_DIR/Bob" --break-system-packages

# SPARC
if ! command -v sparc &> /dev/null; then
    echo ""
    echo "SPARC not found. Install from: https://github.com/SPARC-X/SPARC"
    echo "Bella will use NSMace-only mode without SPARC."
fi

# API keys
echo ""
echo "Add to your ~/.bashrc:"
echo "  export MATERIALS_PROJECT_API_KEY=your_key_here"
echo "Get key free at: https://materialsproject.org/api"
echo ""
echo "Bella installed. Run: bella --help"

#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "  Bob — Materials Discovery Engine"
echo "  by Orders of Magnitude · ofmagnitude.com"
echo ""

echo "→ Installing Bob..."
pip install -e "$SCRIPT_DIR" --break-system-packages

echo ""
echo "  Done. Run: bob"

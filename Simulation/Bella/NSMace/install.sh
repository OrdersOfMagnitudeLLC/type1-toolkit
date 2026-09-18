#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
cp NSMace "$HOME/.bella/bin/NSMace"
echo "NSMace compiled and installed to ~/.bella/bin/NSMace"

#!/bin/bash
set -e
BIN="$1"
if [ -z "$BIN" ]; then
    echo "Usage: $0 <path/to/NSMace>"
    exit 1
fi

# Run 5 times and print energy/forces summary.
for i in 1 2 3 4 5; do
    "$BIN" | tail -n 3
done

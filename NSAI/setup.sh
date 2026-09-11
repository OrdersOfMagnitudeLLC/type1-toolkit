#!/bin/bash
set -e
GGUF="/home/lumiere/NS/deprecated/NSRun/models/qwen2.5-3b-instruct-q4_k_m.gguf"
NSM="/tmp/qwen3b_stream.nsm"
NSQUANT="/home/lumiere/NS/NSAI/NSQuant/nsquant"
NSRUN="/home/lumiere/NS/NSAI/nsrun_test"

if [ ! -f "$NSM" ]; then
  echo "[setup] NSM not found, regenerating..."
  $NSQUANT "$GGUF" "$NSM"
else
  echo "[setup] NSM found at $NSM"
fi

echo "[setup] Running coherence check..."
$NSRUN "$NSM"

chmod +x "$(readlink -f "$0")"

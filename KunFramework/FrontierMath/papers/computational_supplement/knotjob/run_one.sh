#!/bin/bash
# run_one.sh <pdfile> — serial KnotJob21 (4 invariants; -sl0 done separately, too slow at high crossings), 300s timeout, 4g heap
f="$1"
out="${f}_s0_s2_s3_sqo"
if [ -s "$out" ]; then echo "=== $f SKIP (done) ==="; cat "$out"; exit 0; fi
start=$(date +%s)
timeout 300 jdk21/bin/java -Xmx4g -jar KnotJob21.jar "$f" -s0 -s2 -s3 -sqo > "${f}.runlog" 2>&1
rc=$?
end=$(date +%s)
echo "=== $f rc=$rc $((end-start))s ==="
[ -s "$out" ] && cat "$out" || echo "(no/empty output file)"

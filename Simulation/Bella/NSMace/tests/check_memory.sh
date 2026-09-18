#!/bin/bash
echo "Checking RSS memory for 1000-atom system"
echo "Before run:"
cat /proc/$$/status | grep -i rss
${BELLA_HOME}/NSMace/NSMace tests/water1000.json > /tmp/water1000_mem.log 2>&1 &
PID=$!
sleep 2
echo "During run (PID $PID):"
cat /proc/$PID/status | grep -i rss
wait $PID
echo "After run:"
cat /proc/$$/status | grep -i rss

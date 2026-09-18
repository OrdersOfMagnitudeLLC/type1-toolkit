#!/bin/bash
echo "Testing batch-stdin mode with END_BATCH sentinel"

# Create test input
cat > /tmp/batch_test_input.txt << 'EOF'
3 8 0.0 0.0 0.119 1 0.0 0.763 -0.477 1 0.0 -0.763 -0.477
3 8 3.0 0.0 0.119 1 3.0 0.763 -0.477 1 3.0 -0.763 -0.477
END_BATCH
EOF

# Run NSMace in batch-stdin mode
${BELLA_HOME}/NSMace/NSMace --batch-stdin < /tmp/batch_test_input.txt

echo "Test completed"

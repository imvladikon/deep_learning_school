#!/bin/bash

# Test Script for Task 3: Non-Contrastive Learning (SimSiam)
# Quick test with reduced epochs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_ROOT="$PROJECT_ROOT/data/AudioMNIST/data"
EXP_BASE="$PROJECT_ROOT/experiments"

echo "========================================="
echo "TEST: Task 3 - Non-Contrastive Learning"
echo "========================================="
echo "Project root: $PROJECT_ROOT"
echo "Data root: $DATA_ROOT"
echo ""

# Check if data exists
if [ ! -d "$DATA_ROOT" ]; then
    echo "ERROR: Data not found at $DATA_ROOT"
    exit 1
fi

echo "========================================="
echo "Test 3: SimSiam (NCL)"
echo "========================================="
python "$PROJECT_ROOT/scripts/train_simsiam.py" \
    --batch-size 32 \
    --lr 3e-4 \
    --epochs 5 \
    --early-stop-patience 10 \
    --data-root "$DATA_ROOT" \
    --exp-dir "$EXP_BASE/test_simsiam" \
    --hidden-dim 256 \
    --num-workers 2 \
    --no-wandb

echo ""
echo "Test completed successfully!"
echo "Results: $EXP_BASE/test_simsiam"
echo "========================================="

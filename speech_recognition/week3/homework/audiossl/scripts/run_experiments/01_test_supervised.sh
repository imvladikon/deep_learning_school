#!/bin/bash

# Test Script for Task 1: Supervised Learning
# Quick test with reduced epochs to verify everything works

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_ROOT="$PROJECT_ROOT/data/AudioMNIST/data"
EXP_BASE="$PROJECT_ROOT/experiments"

echo "========================================="
echo "TEST: Task 1 - Supervised Learning"
echo "========================================="
echo "Project root: $PROJECT_ROOT"
echo "Data root: $DATA_ROOT"
echo ""

# Check if data exists
if [ ! -d "$DATA_ROOT" ]; then
    echo "ERROR: Data not found at $DATA_ROOT"
    echo "Please download AudioMNIST data first"
    exit 1
fi

# Test 1: Supervised with 1D encoder (waveform)
echo "========================================="
echo "Test 1.1: Supervised 1D (Waveform)"
echo "========================================="
python "$PROJECT_ROOT/scripts/train_supervised.py" \
    --encoder-type 1d \
    --batch-size 32 \
    --lr 3e-4 \
    --epochs 5 \
    --early-stop-patience 10 \
    --data-root "$DATA_ROOT" \
    --exp-dir "$EXP_BASE/test_supervised_1d" \
    --hidden-dim 256 \
    --num-workers 2 \
    --no-wandb

echo ""
echo "Test 1.1 completed successfully!"
echo ""

# Test 2: Supervised with 2D encoder (spectrogram)
echo "========================================="
echo "Test 1.2: Supervised 2D (Spectrogram)"
echo "========================================="
python "$PROJECT_ROOT/scripts/train_supervised.py" \
    --encoder-type 2d \
    --batch-size 32 \
    --lr 3e-4 \
    --epochs 5 \
    --early-stop-patience 10 \
    --data-root "$DATA_ROOT" \
    --exp-dir "$EXP_BASE/test_supervised_2d" \
    --hidden-dim 256 \
    --n-mels 64 \
    --num-workers 2 \
    --no-wandb

echo ""
echo "Test 1.2 completed successfully!"
echo ""

echo "========================================="
echo "All supervised tests passed!"
echo "Results saved to: $EXP_BASE/test_supervised_*"
echo "========================================="

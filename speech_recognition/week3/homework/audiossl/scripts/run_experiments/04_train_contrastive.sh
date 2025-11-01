#!/bin/bash

# Task 2: Contrastive Learning (InfoNCE) - Full Training
# Multi-format contrastive learning with 1D + 2D encoders

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_ROOT="$PROJECT_ROOT/data/AudioMNIST/data"
EXP_BASE="$PROJECT_ROOT/experiments/task2_contrastive"

echo "========================================="
echo "Task 2: Contrastive Learning (InfoNCE)"
echo "========================================="
echo "Project root: $PROJECT_ROOT"
echo "Data root: $DATA_ROOT"
echo "Experiment dir: $EXP_BASE"
echo ""

# Check if data exists
if [ ! -d "$DATA_ROOT" ]; then
    echo "ERROR: Data not found at $DATA_ROOT"
    exit 1
fi

echo "========================================="
echo "Training Contrastive Model (InfoNCE)"
echo "========================================="
echo "Architecture: Dual encoders (1D waveform + 2D spectrogram)"
echo "Loss: InfoNCE (temperature-scaled contrastive)"
echo ""

python "$PROJECT_ROOT/scripts/train_contrastive.py" \
    --batch-size 128 \
    --lr 3e-4 \
    --weight-decay 1e-4 \
    --epochs 50 \
    --early-stop-patience 15 \
    --data-root "$DATA_ROOT" \
    --exp-dir "$EXP_BASE/contrastive_infonce" \
    --hidden-dim 256 \
    --proj-dim 128 \
    --temperature 0.07 \
    --n-mels 64 \
    --n-fft 400 \
    --hop-length 160 \
    --num-workers 4 \
    --scheduler-patience 5 \
    --scheduler-factor 0.5 \
    --seed 42

echo ""
echo "Training completed!"
echo "Results: $EXP_BASE/contrastive_infonce"
echo ""

echo "========================================="
echo "Task 2 Complete!"
echo "========================================="
echo "Results saved to: $EXP_BASE"
echo ""
echo "Next steps:"
echo "  1. Check training curves: $EXP_BASE/contrastive_infonce/training_curves.png"
echo "  2. Run linear evaluation to assess embedding quality"
echo "  3. Compare with supervised baseline from Task 1"
echo "========================================="

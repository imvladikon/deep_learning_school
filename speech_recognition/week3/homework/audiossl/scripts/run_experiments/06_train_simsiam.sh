#!/bin/bash

# Task 3: Non-Contrastive Learning (SimSiam) - Full Training
# Multi-format NCL with predictor network and stop-gradient

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_ROOT="$PROJECT_ROOT/data/AudioMNIST/data"
EXP_BASE="$PROJECT_ROOT/experiments/task3_ncl"

echo "========================================="
echo "Task 3: Non-Contrastive Learning (SimSiam)"
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
echo "Training SimSiam Model"
echo "========================================="
echo "Architecture: Dual encoders + Predictor MLP"
echo "Loss: Negative cosine similarity with stop-gradient"
echo "Key: No negative pairs, no momentum encoder"
echo ""

python "$PROJECT_ROOT/scripts/train_simsiam.py" \
    --batch-size 128 \
    --lr 3e-4 \
    --weight-decay 1e-4 \
    --epochs 50 \
    --early-stop-patience 15 \
    --data-root "$DATA_ROOT" \
    --exp-dir "$EXP_BASE/simsiam" \
    --hidden-dim 256 \
    --proj-dim 128 \
    --pred-dim 64 \
    --n-mels 64 \
    --n-fft 400 \
    --hop-length 160 \
    --num-workers 4 \
    --scheduler-patience 5 \
    --scheduler-factor 0.5 \
    --seed 42

echo ""
echo "Training completed!"
echo "Results: $EXP_BASE/simsiam"
echo ""

echo "========================================="
echo "Task 3 Complete!"
echo "========================================="
echo "Results saved to: $EXP_BASE"
echo ""
echo "Next steps:"
echo "  1. Check training curves: $EXP_BASE/simsiam/training_curves.png"
echo "  2. Run linear evaluation"
echo "  3. Compare with InfoNCE (Task 2) and supervised (Task 1)"
echo "========================================="

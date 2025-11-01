#!/bin/bash

# Task 1: Supervised Learning (Full Training)
# Train supervised classifiers with 1D and 2D encoders

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_ROOT="$PROJECT_ROOT/data/AudioMNIST/data"
EXP_BASE="$PROJECT_ROOT/experiments/task1_supervised"

echo "========================================="
echo "Task 1: Supervised Learning"
echo "========================================="
echo "Project root: $PROJECT_ROOT"
echo "Data root: $DATA_ROOT"
echo "Experiment dir: $EXP_BASE"
echo ""

# Check if data exists
if [ ! -d "$DATA_ROOT" ]; then
    echo "ERROR: Data not found at $DATA_ROOT"
    echo "Please download AudioMNIST data first"
    exit 1
fi

# Training 1: Supervised with 1D encoder (waveform)
echo "========================================="
echo "Training 1.1: Supervised 1D (Waveform)"
echo "========================================="
python "$PROJECT_ROOT/scripts/train_supervised.py" \
    --encoder-type 1d \
    --batch-size 64 \
    --lr 3e-4 \
    --weight-decay 1e-4 \
    --epochs 40 \
    --early-stop-patience 12 \
    --data-root "$DATA_ROOT" \
    --exp-dir "$EXP_BASE/supervised_1d" \
    --hidden-dim 256 \
    --num-workers 4 \
    --scheduler-patience 5 \
    --scheduler-factor 0.5 \
    --seed 42

echo ""
echo "Training 1.1 completed!"
echo "Results: $EXP_BASE/supervised_1d"
echo ""

# Training 2: Supervised with 2D encoder (spectrogram)
echo "========================================="
echo "Training 1.2: Supervised 2D (Spectrogram)"
echo "========================================="
python "$PROJECT_ROOT/scripts/train_supervised.py" \
    --encoder-type 2d \
    --batch-size 64 \
    --lr 3e-4 \
    --weight-decay 1e-4 \
    --epochs 40 \
    --early-stop-patience 12 \
    --data-root "$DATA_ROOT" \
    --exp-dir "$EXP_BASE/supervised_2d" \
    --hidden-dim 256 \
    --n-mels 64 \
    --n-fft 400 \
    --hop-length 160 \
    --num-workers 4 \
    --scheduler-patience 5 \
    --scheduler-factor 0.5 \
    --seed 42

echo ""
echo "Training 1.2 completed!"
echo "Results: $EXP_BASE/supervised_2d"
echo ""

# Training 3: Supervised with both encoders
echo "========================================="
echo "Training 1.3: Supervised Both (1D + 2D)"
echo "========================================="
python "$PROJECT_ROOT/scripts/train_supervised.py" \
    --encoder-type both \
    --batch-size 64 \
    --lr 3e-4 \
    --weight-decay 1e-4 \
    --epochs 40 \
    --early-stop-patience 12 \
    --data-root "$DATA_ROOT" \
    --exp-dir "$EXP_BASE/supervised_both" \
    --hidden-dim 256 \
    --n-mels 64 \
    --n-fft 400 \
    --hop-length 160 \
    --num-workers 4 \
    --scheduler-patience 5 \
    --scheduler-factor 0.5 \
    --seed 42

echo ""
echo "Training 1.3 completed!"
echo "Results: $EXP_BASE/supervised_both"
echo ""

echo "========================================="
echo "Task 1 Complete!"
echo "========================================="
echo "Results saved to: $EXP_BASE"
echo ""
echo "To view results:"
echo "  - Training curves: $EXP_BASE/*/training_curves.png"
echo "  - Best models: $EXP_BASE/*/best_model.pt"
echo "  - Config & history: $EXP_BASE/*/config.json, history.json"
echo "========================================="

#!/bin/bash

# Run FIXED experiments based on DEEP_ANALYSIS_SUMMARY.md recommendations
#
# This script runs:
# 1. Contrastive training with FIXED hyperparameters
# 2. Joint training (SSL + supervised)
#
# Expected runtime: ~4-6 hours on GPU for full experiments
# Quick test mode: ~5-10 minutes

set -e  # Exit on error

echo "======================================================================"
echo "Running FIXED SSL Experiments"
echo "Based on: DEEP_ANALYSIS_SUMMARY.md"
echo "======================================================================"
echo ""

# Configuration
QUICK_TEST=${QUICK_TEST:-false}  # Set to true for quick testing
USE_WANDB=${USE_WANDB:-false}    # Set to true to enable W&B logging

if [ "$QUICK_TEST" = "true" ]; then
    echo "⚠️  QUICK TEST MODE ENABLED (2 epochs, small data)"
    echo ""
    QUICK_FLAG="--quick-test"
else
    echo "📊 FULL TRAINING MODE (100 epochs)"
    echo ""
    QUICK_FLAG=""
fi

if [ "$USE_WANDB" = "true" ]; then
    echo "📈 W&B logging ENABLED"
    echo ""
    WANDB_FLAG="--wandb"
else
    echo "📝 W&B logging DISABLED (local only)"
    echo ""
    WANDB_FLAG=""
fi

# Navigate to project root
cd "$(dirname "$0")/.."
echo "Working directory: $(pwd)"
echo ""

# ======================================================================
# Experiment 1: Contrastive with FIXED hyperparameters
# ======================================================================
echo "======================================================================"
echo "Experiment 1: Contrastive Training (FIXED)"
echo "======================================================================"
echo "Hyperparameters:"
echo "  - Temperature: 0.1 (was 0.5 → 5x DECREASE)"
echo "  - Batch Size:  128 (was 32 → 4x INCREASE)"
echo "  - Epochs:      100 (was 40 → 2.5x INCREASE)"
echo "  - Scheduler:   CosineAnnealing (was ReduceLROnPlateau)"
echo ""
echo "Expected: Validation loss should DECREASE (not flat!)"
echo ""
echo "Starting in 3 seconds..."
sleep 3

python3 scripts/train_contrastive_FIXED.py \
    --batch-size 128 \
    --temperature 0.1 \
    --epochs 100 \
    --scheduler cosine \
    --output-dir ./experiments/contrastive_FIXED \
    $QUICK_FLAG \
    $WANDB_FLAG \
    || { echo "❌ Experiment 1 FAILED"; exit 1; }

echo ""
echo "✅ Experiment 1 COMPLETE"
echo ""
echo "----------------------------------------------------------------------"
echo ""

# ======================================================================
# Experiment 2: Joint Training (SSL + Supervised)
# ======================================================================
echo "======================================================================"
echo "Experiment 2: Joint Training (SSL + Supervised)"
echo "======================================================================"
echo "Training mode: Encoder + Classifier simultaneously"
echo "Loss: α * L_contrastive + (1-α) * L_classification"
echo "Alpha schedule: cosine (start 70% SSL, end 30% SSL)"
echo ""
echo "Starting in 3 seconds..."
sleep 3

python3 scripts/train_joint_contrastive.py \
    --batch-size 128 \
    --temperature 0.1 \
    --epochs 100 \
    --alpha-schedule cosine \
    --initial-alpha 0.7 \
    --final-alpha 0.3 \
    --output-dir ./experiments/joint_contrastive \
    $QUICK_FLAG \
    $WANDB_FLAG \
    || { echo "❌ Experiment 2 FAILED"; exit 1; }

echo ""
echo "✅ Experiment 2 COMPLETE"
echo ""
echo "----------------------------------------------------------------------"
echo ""

# ======================================================================
# Summary
# ======================================================================
echo "======================================================================"
echo "ALL EXPERIMENTS COMPLETE! ✅"
echo "======================================================================"
echo ""
echo "Results saved to:"
echo "  1. ./experiments/contrastive_FIXED/"
echo "  2. ./experiments/joint_contrastive/"
echo ""
echo "Next steps:"
echo "  1. Check validation loss curves (should DECREASE, not flat!)"
echo "  2. Compare results with baseline (see ANALYSIS_README.md)"
echo "  3. If results are good, proceed with linear evaluation"
echo "  4. See DEEP_ANALYSIS_SUMMARY.md for further improvements"
echo ""
echo "======================================================================"
echo ""

# Optional: Print results summary
if [ -f "./experiments/contrastive_FIXED/results.json" ]; then
    echo "Experiment 1 Results (Contrastive FIXED):"
    cat ./experiments/contrastive_FIXED/results.json | python -m json.tool
    echo ""
fi

if [ -f "./experiments/joint_contrastive/results.json" ]; then
    echo "Experiment 2 Results (Joint Training):"
    cat ./experiments/joint_contrastive/results.json | python -m json.tool
    echo ""
fi

echo "✅ Done! See results above."
echo ""

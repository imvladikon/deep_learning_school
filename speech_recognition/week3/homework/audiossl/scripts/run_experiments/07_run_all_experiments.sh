#!/bin/bash

# Master Experiment Script
# Run all experiments for the homework
#
# Tasks:
#   1. Supervised learning (1D, 2D, both)
#   2. Contrastive learning (InfoNCE)
#   3. Non-contrastive learning (SimSiam)
#
# Total estimated time: 3-4 hours (depending on hardware)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
EXP_BASE="$PROJECT_ROOT/experiments"

echo "========================================="
echo "SSL Audio Homework - Full Experiments"
echo "========================================="
echo ""
echo "This will run ALL experiments:"
echo "  - Task 1: Supervised (3 variants)"
echo "  - Task 2: Contrastive (InfoNCE)"
echo "  - Task 3: Non-Contrastive (SimSiam)"
echo ""
echo "Total experiments: 5"
echo "Estimated time: 3-4 hours"
echo ""
echo "Results will be saved to: $EXP_BASE"
echo ""
echo "Press Ctrl+C to cancel, or wait 10 seconds to start..."
sleep 10

START_TIME=$(date +%s)

# Task 1: Supervised Learning
echo ""
echo "========================================="
echo "TASK 1: SUPERVISED LEARNING"
echo "========================================="
bash "$SCRIPT_DIR/02_train_supervised.sh"

# Task 2: Contrastive Learning
echo ""
echo "========================================="
echo "TASK 2: CONTRASTIVE LEARNING"
echo "========================================="
bash "$SCRIPT_DIR/04_train_contrastive.sh"

# Task 3: Non-Contrastive Learning
echo ""
echo "========================================="
echo "TASK 3: NON-CONTRASTIVE LEARNING"
echo "========================================="
bash "$SCRIPT_DIR/06_train_simsiam.sh"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))

echo ""
echo "========================================="
echo "ALL EXPERIMENTS COMPLETED! ✓"
echo "========================================="
echo ""
echo "Total time: ${HOURS}h ${MINUTES}m"
echo ""
echo "Results summary:"
echo "  Task 1 (Supervised):"
echo "    - 1D encoder: $EXP_BASE/task1_supervised/supervised_1d/"
echo "    - 2D encoder: $EXP_BASE/task1_supervised/supervised_2d/"
echo "    - Both encoders: $EXP_BASE/task1_supervised/supervised_both/"
echo ""
echo "  Task 2 (Contrastive):"
echo "    - InfoNCE: $EXP_BASE/task2_contrastive/contrastive_infonce/"
echo ""
echo "  Task 3 (Non-Contrastive):"
echo "    - SimSiam: $EXP_BASE/task3_ncl/simsiam/"
echo ""
echo "Next steps:"
echo "  1. Compare results across all experiments"
echo "  2. Create visualizations and tables"
echo "  3. Write analysis for Task 4"
echo "  4. Fill in the homework notebook"
echo ""
echo "To generate comparison report:"
echo "  python $PROJECT_ROOT/scripts/compare_results.py"
echo ""
echo "========================================="

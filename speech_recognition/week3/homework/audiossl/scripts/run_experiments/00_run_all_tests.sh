#!/bin/bash

# Master Test Script
# Run quick tests for all tasks to verify everything works
# Use this BEFORE running full experiments

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================="
echo "Running All Tests"
echo "========================================="
echo "This will run quick tests (5 epochs each)"
echo "to verify all training scripts work correctly"
echo ""
echo "Press Ctrl+C to cancel, or wait 5 seconds..."
sleep 5

# Test Task 1: Supervised
echo ""
echo "========================================="
echo "TESTING TASK 1: SUPERVISED"
echo "========================================="
bash "$SCRIPT_DIR/01_test_supervised.sh"

# Test Task 2: Contrastive
echo ""
echo "========================================="
echo "TESTING TASK 2: CONTRASTIVE"
echo "========================================="
bash "$SCRIPT_DIR/03_test_contrastive.sh"

# Test Task 3: SimSiam
echo ""
echo "========================================="
echo "TESTING TASK 3: NON-CONTRASTIVE"
echo "========================================="
bash "$SCRIPT_DIR/05_test_simsiam.sh"

echo ""
echo "========================================="
echo "ALL TESTS PASSED! ✓"
echo "========================================="
echo ""
echo "All training scripts are working correctly."
echo "You can now run the full experiments:"
echo ""
echo "  bash $SCRIPT_DIR/07_run_all_experiments.sh"
echo ""
echo "Or run individual tasks:"
echo "  bash $SCRIPT_DIR/02_train_supervised.sh    # Task 1"
echo "  bash $SCRIPT_DIR/04_train_contrastive.sh  # Task 2"
echo "  bash $SCRIPT_DIR/06_train_simsiam.sh      # Task 3"
echo ""
echo "========================================="

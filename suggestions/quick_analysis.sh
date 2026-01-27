#!/bin/bash
# Quick analysis script for AlphaEdit runs
# Usage: ./quick_analysis.sh <run_directory>

set -e

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <run_directory>"
    echo "Example: $0 results/AlphaEdit/run_050"
    exit 1
fi

RUN_DIR="$1"
RUN_NAME=$(basename "$RUN_DIR")

echo "========================================================================"
echo "Quick Analysis for: $RUN_DIR"
echo "========================================================================"
echo ""

# Check if run directory exists
if [ ! -d "$RUN_DIR" ]; then
    echo "Error: Directory $RUN_DIR does not exist"
    exit 1
fi

# Check if result files exist
NUM_FILES=$(find "$RUN_DIR" -name "*_edits-case_*.json" | wc -l)
if [ "$NUM_FILES" -eq 0 ]; then
    echo "Error: No edit result files found in $RUN_DIR"
    exit 1
fi

echo "Found $NUM_FILES edit result files"
echo ""

# Run basic analysis
echo "Running basic analysis..."
python3 analyze_failed_edits.py \
    --run-dir "$RUN_DIR" \
    --export "failed_edits_${RUN_NAME}.json" \
    --create-retry-dataset "retry_dataset_${RUN_NAME}.json" \
    --show-details 5

echo ""
echo "========================================================================"
echo ""

# Run deep diagnostic
echo "Running deep diagnostic..."
python3 diagnose_edit_failures.py \
    --run-dir "$RUN_DIR" \
    --output-report "diagnostic_${RUN_NAME}.json" \
    --output-recommendations "recommendations_${RUN_NAME}.json"

echo ""
echo "========================================================================"
echo "Analysis Complete!"
echo "========================================================================"
echo ""
echo "Generated Files:"
echo "  - failed_edits_${RUN_NAME}.json (all failed edits)"
echo "  - retry_dataset_${RUN_NAME}.json (for re-running failures)"
echo "  - diagnostic_${RUN_NAME}.json (detailed diagnostics)"
echo "  - recommendations_${RUN_NAME}.json (parameter recommendations)"
echo "  - recommendations_${RUN_NAME}.txt (human-readable guide)"
echo ""
echo "Next Steps:"
echo "  1. Read recommendations_${RUN_NAME}.txt for parameter tuning advice"
echo "  2. Examine specific cases: python3 examine_edit.py --run-dir $RUN_DIR --case-id <ID>"
echo "  3. Compare with other runs: python3 compare_runs.py --runs $RUN_DIR <other_runs>"
echo ""
echo "For detailed help, see FAILED_EDITS_GUIDE.md and ANALYSIS_SUMMARY.md"
echo "========================================================================"

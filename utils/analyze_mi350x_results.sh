#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Post-processing pipeline for Kimi K2.5 FP4 MI350X benchmark results.
#
# Steps:
#   1. process_result.py    – enrich each raw JSON with metadata & derived metrics
#   2. summarize.py         – render a markdown table from the processed results
#   3. export_csv.py        – export website-compatible CSV
#   4. plot_comparison.py   – MI350X vs MI355X comparison plots
#
# Usage:  bash utils/analyze_mi350x_results.sh
#         (run from the InferenceX repo root)
# =============================================================================

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

RAW_DIR="results_mi350x"
PROCESSED_DIR="agg_results_mi350x"

# ---- Common metadata (must match what the benchmark was run with) -----------

export RUNNER_TYPE=mi350x
export FRAMEWORK=vllm
export PRECISION=fp4
export SPEC_DECODING=false
export DISAGG=false
export MODEL_PREFIX=kimik2.5
export IMAGE="vllm/vllm-openai-rocm:v0.18.0"
export ISL=8192
export OSL=1024
export IS_MULTINODE=false
export TP=8
export EP_SIZE=1
export DP_ATTENTION=false

# ---- Step 1: Run process_result.py on each raw result ----------------------

mkdir -p "$PROCESSED_DIR"

echo "=== Step 1: Processing raw results ==="

for raw_json in "$RAW_DIR"/*.json; do
    [ -f "$raw_json" ] || continue

    basename_no_ext="${raw_json%.json}"          # e.g. results_mi350x/kimik2.5_...
    agg_output="agg_${basename_no_ext}.json"     # e.g. agg_results_mi350x/kimik2.5_...

    if [ -f "$agg_output" ]; then
        echo "  [skip] Already processed: $agg_output"
        continue
    fi

    echo "  Processing: $raw_json"
    export RESULT_FILENAME="$basename_no_ext"
    python3 utils/process_result.py
    echo "    -> $agg_output"
done

echo ""

# ---- Step 2: Summarize processed results ------------------------------------

echo "=== Step 2: Generating summary table ==="
echo ""
python3 utils/summarize.py "$PROCESSED_DIR"

# ---- Step 3: Export CSV (website-compatible) --------------------------------

CSV_OUTPUT="${PROCESSED_DIR}/results.csv"

echo ""
echo "=== Step 3: Exporting CSV ==="
python3 utils/export_csv.py "$PROCESSED_DIR" -o "$CSV_OUTPUT"
echo "CSV written to: $CSV_OUTPUT"

# ---- Step 4: Plot MI350X vs MI355X comparison -------------------------------

MI355X_CSV="${PROCESSED_DIR}/mi355x_results.csv"

if [ -f "$MI355X_CSV" ]; then
    echo ""
    echo "=== Step 4: Generating comparison plots ==="
    python3 utils/plot_comparison.py "$CSV_OUTPUT" "$MI355X_CSV" -o "$PROCESSED_DIR"
else
    echo ""
    echo "=== Step 4: Skipped (no MI355X CSV found at $MI355X_CSV) ==="
    echo "  Place the MI355X website CSV at $MI355X_CSV to enable comparison plots."
fi

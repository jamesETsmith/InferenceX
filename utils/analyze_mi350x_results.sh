#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Post-processing pipeline for Kimi K2.5 FP4 MI350X benchmark results.
#
# Thin wrapper around analyze_mi350x_results.py — all arguments are forwarded.
#
# Usage:  bash utils/analyze_mi350x_results.sh [--force] [--image IMAGE] ...
#         (run from the InferenceX repo root)
# =============================================================================

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
exec python3 "$REPO_DIR/utils/analyze_mi350x_results.py" "$@"

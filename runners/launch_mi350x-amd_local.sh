#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Local launcher for Kimi K2.5 FP4 benchmark on MI350X
#
# Run this directly on an allocated GPU node (no Slurm). It pulls the Docker
# image, then runs the benchmark inside the container for each concurrency
# level.
#
# Usage:  bash runners/run_kimik2.5_fp4_mi350x_local.sh
#         (run from the InferenceX repo root)
# =============================================================================

# ---- Configuration (edit these) ---------------------------------------------

IMAGE="vllm/vllm-openai-rocm:v0.18.0"
MODEL="amd/Kimi-K2.5-MXFP4"
TP=8
ISL=8192
OSL=1024
MAX_MODEL_LEN=16384
RANDOM_RANGE_RATIO=1.0
CONCURRENCIES=(4 8 16 32)

PORT=8000
CONTAINER_NAME="kimik2.5-fp4-mi350x-bench"
HF_CACHE_DIR="${HF_HOME:-/data/hf-cache}"

# ---- Derived paths (auto-detected) -----------------------------------------

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULT_DIR="${REPO_DIR}/results_mi350x"
mkdir -p "$RESULT_DIR" "$HF_CACHE_DIR"

echo "=== Kimi K2.5 FP4 MI350X Benchmark (local) ==="
echo "Image:   $IMAGE"
echo "Model:   $MODEL"
echo "TP:      $TP"
echo "ISL/OSL: ${ISL}/${OSL}"
echo "Concs:   ${CONCURRENCIES[*]}"
echo "Repo:    $REPO_DIR"
echo "Results: $RESULT_DIR"
echo ""

# ---- Pull Docker image -----------------------------------------------------

echo "Pulling Docker image: $IMAGE"
docker pull "$IMAGE"

# ---- Run benchmark for each concurrency level ------------------------------

for CONC in "${CONCURRENCIES[@]}"; do
    echo ""
    echo ">>> Running concurrency=$CONC (TP=$TP, ISL=$ISL, OSL=$OSL)"

    RESULT_FILENAME="kimik2.5_fp4_mi350x_tp${TP}_isl${ISL}_osl${OSL}_conc${CONC}"

    docker run --rm \
        --name "$CONTAINER_NAME" \
        --device=/dev/kfd \
        --device=/dev/dri \
        --group-add video \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        --shm-size=256g \
        --network=host \
        --entrypoint bash \
        -v "${REPO_DIR}:/workspace/" \
        -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
        -w /workspace/ \
        -e MODEL="$MODEL" \
        -e TP="$TP" \
        -e CONC="$CONC" \
        -e ISL="$ISL" \
        -e OSL="$OSL" \
        -e MAX_MODEL_LEN="$MAX_MODEL_LEN" \
        -e RANDOM_RANGE_RATIO="$RANDOM_RANGE_RATIO" \
        -e RESULT_FILENAME="$RESULT_FILENAME" \
        -e PORT="$PORT" \
        "$IMAGE" \
        benchmarks/single_node/kimik2.5_fp4_mi350x.sh

    # Move results from repo root (mounted as /workspace/) to results dir
    for f in "${REPO_DIR}/${RESULT_FILENAME}"*; do
        if [ -f "$f" ]; then
            mv "$f" "$RESULT_DIR/"
            echo "Saved: $RESULT_DIR/$(basename "$f")"
        fi
    done
done

echo ""
echo "=== All benchmark runs complete ==="
echo "Results in: $RESULT_DIR"
ls -la "$RESULT_DIR/"

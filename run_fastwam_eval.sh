#!/bin/bash
# =============================================================================
# FastWAM VLN Evaluation Launch Script
#
# Uses two processes:
# 1. FastWAM inference server (fastwam env, GPU)
# 2. Habitat evaluation client (internnaveval env, CPU + Habitat)
#
# Usage:
#   bash run_fastwam_eval.sh [checkpoint_path] [gpu_id] [port]
#
# Example:
#   bash run_fastwam_eval.sh /path/to/step_002000.pt 1 9527
#
# Stop head is auto-detected from the latest *.pt in the stop_head scan dir.
# Pass --stop_head_checkpoint explicitly to the server to override.
# Logs are written to auto_eval_logs/ (same convention as auto_eval_watcher.sh).
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT="${1:-}"
GPU_ID="${2:-1}"
PORT="${3:-9527}"
WAYPOINT_MODE="${4:-0}"   # set to 1 to enable polar-coordinate waypoint mode
CKPT_BASE_DIR="/apdcephfs_tj5/share_302528826/xxd/nav_vln_1e-4"

# Auto-detect latest checkpoint if not specified
if [ -z "${CHECKPOINT}" ]; then
    CHECKPOINT=$(find "${CKPT_BASE_DIR}" -name "step_*.pt" -path "*/weights/*" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | awk '{print $2}')
    if [ -z "${CHECKPOINT}" ]; then
        echo "ERROR: No checkpoint found in ${CKPT_BASE_DIR}"
        exit 1
    fi
    echo "Auto-detected latest checkpoint: ${CHECKPOINT}"
fi

FASTWAM_PYTHON="/apdcephfs_qy2/share_303214315/hunyuan/xxd/miniconda3/envs/fastwam/bin/python"
INTERNNAV_TORCHRUN="/apdcephfs_tj5/share_302528826/xxd/internnaveval/bin/torchrun"
INTERNNAV_DIR="/apdcephfs_tj5/share_302528826/xxd/InternNav"
LOG_DIR="${SCRIPT_DIR}/auto_eval_logs"
mkdir -p "${LOG_DIR}"

# Build per-run names (same convention as auto_eval_watcher.sh)
STEP_NAME=$(basename "${CHECKPOINT}" .pt)
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
EVAL_OUTPUT="/apdcephfs_tj5/share_302528826/xxd/fastwam_nav_eval_results/${STEP_NAME}_${TIMESTAMP}"
SERVER_LOG="${LOG_DIR}/server_${STEP_NAME}_${TIMESTAMP}.log"
CLIENT_LOG="${LOG_DIR}/client_${STEP_NAME}_${TIMESTAMP}.log"
export FASTWAM_EVAL_OUTPUT_PATH="${EVAL_OUTPUT}"
mkdir -p "${EVAL_OUTPUT}"

echo "============================================="
echo "FastWAM VLN Evaluation"
echo "============================================="
echo "Checkpoint: ${CHECKPOINT:-'(pretrained only)'}"
echo "GPU: ${GPU_ID}"
echo "Server port: ${PORT}"
echo "Waypoint mode: $([ "${WAYPOINT_MODE}" = "1" ] && echo "ON (polar-coordinate continuous)" || echo "OFF (discrete actions)")"
echo "Stop Head: auto-detect from /apdcephfs_tj5/share_302528826/xxd/nav_vln_1e-4/stop_head"
echo "Output dir: ${EVAL_OUTPUT}"
echo "Server log: ${SERVER_LOG}"
echo "Client log: ${CLIENT_LOG}"
echo ""

wait_for_server() {
    # Poll port until server is accepting connections (max 30 min)
    local max_wait=1800
    local waited=0
    while ! python3 -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1', ${PORT})); s.close()" 2>/dev/null; do
        sleep 5
        waited=$((waited + 5))
        if [ ${waited} -ge ${max_wait} ]; then
            echo "ERROR: Server did not open port ${PORT} within ${max_wait}s"
            return 1
        fi
        if ! kill -0 ${SERVER_PID} 2>/dev/null; then
            echo "ERROR: Server process died"
            return 1
        fi
        echo "  Waiting for server... (${waited}s elapsed)"
    done
    echo "Server is ready!"
    return 0
}

# --- Step 1: Start FastWAM inference server ---
echo "[1/2] Starting FastWAM inference server..."
# Stop head disabled: --stop_head_scan_dir "" prevents auto-detection
# Stop logic: moving_flag > 50% of waypoints → STOP (handled in traj_utils.py)
WAYPOINT_FLAG=""
if [ "${WAYPOINT_MODE}" = "1" ]; then
    WAYPOINT_FLAG="--waypoint_mode"
fi
CUDA_VISIBLE_DEVICES=${GPU_ID} ${FASTWAM_PYTHON} -u ${SCRIPT_DIR}/fastwam_server.py \
    --checkpoint "${CHECKPOINT}" \
    --stop_head_scan_dir "" \
    --port ${PORT} \
    --device cuda:0 \
    ${WAYPOINT_FLAG} \
    > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
echo "Server PID: ${SERVER_PID}"

# Wait until port is actually open
echo "Waiting for server to start..."
if ! wait_for_server; then
    echo "ERROR: Server failed to start! Check: ${SERVER_LOG}"
    kill ${SERVER_PID} 2>/dev/null || true
    exit 1
fi
echo ""

# --- Step 2: Start Habitat evaluation client ---
echo "[2/2] Starting Habitat evaluation..."
cd ${INTERNNAV_DIR}

${INTERNNAV_TORCHRUN} \
    --nproc_per_node=1 \
    --master_port=2345 \
    scripts/eval/eval.py \
    --config ${SCRIPT_DIR}/configs/fastwam_habitat_cfg.py \
    > "${CLIENT_LOG}" 2>&1

# Cleanup
echo ""
echo "Evaluation complete. Shutting down server..."
kill ${SERVER_PID} 2>/dev/null || true

# Print results if available
if [ -f "${EVAL_OUTPUT}/progress.json" ]; then
    echo "Results (${STEP_NAME}):"
    tail -5 "${EVAL_OUTPUT}/progress.json"
fi

echo "Done! Logs in: ${LOG_DIR}"

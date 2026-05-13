#!/bin/bash
# =============================================================================
# FastWAM VLN Auto-Eval Watcher (Habitat Full Evaluation)
#
# Automatically scans for new checkpoints every 30 minutes and runs full
# Habitat VLN evaluation using server-client architecture:
# - Server (fastwam env): FastWAM model inference on GPU
# - Client (internnaveval env): Habitat simulator + InternNav eval framework
#
# Usage:
#   bash auto_eval_watcher.sh [checkpoint_dir] [gpu_id]
#
# Example:
#   nohup bash auto_eval_watcher.sh /apdcephfs_tj5/share_302528826/xxd/nav_vln_1e-4 1 \
#       > watcher.log 2>&1 &
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT_BASE_DIR="${1:-/apdcephfs_tj5/share_302528826/xxd/nav_vln_1e-4}"
GPU_ID="${2:-1}"
SCAN_INTERVAL=1800  # 30 minutes
SERVER_PORT=9527

FASTWAM_PYTHON="/apdcephfs_tj5/share_302528826/xxd/fastwam_env/bin/python"
INTERNNAV_TORCHRUN="/apdcephfs_tj5/share_302528826/xxd/internnaveval/bin/torchrun"
INTERNNAV_DIR="/apdcephfs_tj5/share_302528826/xxd/InternNav"
EVAL_CONFIG="${SCRIPT_DIR}/configs/fastwam_habitat_cfg.py"
LOG_DIR="${SCRIPT_DIR}/auto_eval_logs"

mkdir -p "${LOG_DIR}"

LAST_CKPT=""
SERVER_PID=""
CLIENT_PID=""

find_latest_checkpoint() {
    local latest=""
    latest=$(find "${CKPT_BASE_DIR}" -name "step_*.pt" -path "*/weights/*" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | awk '{print $2}')
    echo "${latest}"
}

kill_current_eval() {
    # Kill client first, then server
    if [ -n "${CLIENT_PID}" ] && kill -0 ${CLIENT_PID} 2>/dev/null; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Killing Habitat client (PID=${CLIENT_PID})..."
        kill ${CLIENT_PID} 2>/dev/null
        wait ${CLIENT_PID} 2>/dev/null
        CLIENT_PID=""
    fi
    if [ -n "${SERVER_PID}" ] && kill -0 ${SERVER_PID} 2>/dev/null; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Killing FastWAM server (PID=${SERVER_PID})..."
        kill ${SERVER_PID} 2>/dev/null
        wait ${SERVER_PID} 2>/dev/null
        SERVER_PID=""
    fi
}

wait_for_server() {
    # Wait until server port is open
    local max_wait=900  # 15 minutes max (fastwam env on network FS, import is slow)
    local waited=0
    while ! python3 -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1', ${SERVER_PORT})); s.close()" 2>/dev/null; do
        sleep 5
        waited=$((waited + 5))
        if [ ${waited} -ge ${max_wait} ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Server failed to start within ${max_wait}s"
            return 1
        fi
        # Check if server process died
        if ! kill -0 ${SERVER_PID} 2>/dev/null; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Server process died"
            return 1
        fi
        echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Waiting for server... (${waited}s)"
    done
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Server is ready!"
    return 0
}

run_eval() {
    local ckpt_path="$1"
    local step_name=$(basename "${ckpt_path}" .pt)
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local server_log="${LOG_DIR}/server_${step_name}_${timestamp}.log"
    local client_log="${LOG_DIR}/client_${step_name}_${timestamp}.log"
    local output_path="/apdcephfs_tj5/share_302528826/xxd/fastwam_nav_eval_results/${step_name}_${timestamp}"

    # Use checkpoint directly from tj5 (no need to copy to /tmp since env and base checkpoints are all on tj5 now)
    local local_ckpt="${ckpt_path}"

    # Update output_path in config dynamically via env var
    export FASTWAM_EVAL_OUTPUT_PATH="${output_path}"
    mkdir -p "${output_path}"

    # --- Start FastWAM inference server ---
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting FastWAM server (GPU=${GPU_ID})..."
    DIFFSYNTH_MODEL_BASE_PATH=/apdcephfs_tj5/share_302528826/xxd/fastwam_checkpoints DIFFSYNTH_SKIP_DOWNLOAD=true \
    ${FASTWAM_PYTHON} -u ${SCRIPT_DIR}/fastwam_server.py \
        --checkpoint "${local_ckpt}" \
        --port ${SERVER_PORT} \
        --device cuda:${GPU_ID} \
        > "${server_log}" 2>&1 &
    SERVER_PID=$!
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Server PID: ${SERVER_PID}, Log: ${server_log}"

    # Wait for server to be ready
    if ! wait_for_server; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Server failed, aborting this eval."
        kill ${SERVER_PID} 2>/dev/null
        SERVER_PID=""
        return 1
    fi

    # --- Start Habitat evaluation client ---
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Habitat evaluation client..."
    cd ${INTERNNAV_DIR}

    ${INTERNNAV_TORCHRUN} \
        --nproc_per_node=1 \
        --master_port=2345 \
        scripts/eval/eval.py \
        --config ${EVAL_CONFIG} \
        > "${client_log}" 2>&1 &
    CLIENT_PID=$!
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Client PID: ${CLIENT_PID}, Log: ${client_log}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Output: ${output_path}"

    # Wait for client to finish
    wait ${CLIENT_PID} 2>/dev/null
    local exit_code=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Habitat eval finished (exit=${exit_code})"
    CLIENT_PID=""

    # Kill server after eval completes
    if [ -n "${SERVER_PID}" ] && kill -0 ${SERVER_PID} 2>/dev/null; then
        kill ${SERVER_PID} 2>/dev/null
        wait ${SERVER_PID} 2>/dev/null
    fi
    SERVER_PID=""

    # Print results if progress.json exists
    if [ -f "${output_path}/progress.json" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Results (${step_name}):"
        tail -5 "${output_path}/progress.json"
    fi

    return ${exit_code}
}

# Main loop
echo "============================================="
echo "FastWAM VLN Auto-Eval Watcher (Habitat)"
echo "============================================="
echo "Checkpoint dir: ${CKPT_BASE_DIR}"
echo "GPU: ${GPU_ID}"
echo "Scan interval: ${SCAN_INTERVAL}s ($(( SCAN_INTERVAL / 60 ))min)"
echo "Server port: ${SERVER_PORT}"
echo "Log dir: ${LOG_DIR}"
echo ""
echo "Watching for new checkpoints..."
echo ""

while true; do
    LATEST_CKPT=$(find_latest_checkpoint)

    if [ -n "${LATEST_CKPT}" ] && [ "${LATEST_CKPT}" != "${LAST_CKPT}" ]; then
        echo ""
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] *** New checkpoint detected: ${LATEST_CKPT}"

        # Kill current eval if running
        kill_current_eval

        # Start new eval
        run_eval "${LATEST_CKPT}"
        LAST_CKPT="${LATEST_CKPT}"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] No new checkpoint. Latest: $(basename "${LATEST_CKPT:-none}")"
    fi

    sleep ${SCAN_INTERVAL}
done

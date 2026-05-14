"""
FastWAM Navigation Inference Server (Multi-Client).

Supports multiple simultaneous client connections via threading.
Each client gets independent agent state (episode tracking), but shares
the same underlying model for GPU inference (protected by a lock).

Usage:
    CUDA_VISIBLE_DEVICES=1 python fastwam_server.py \
        --checkpoint /path/to/checkpoint.pt \
        --port 9527

Protocol (JSON over TCP, length-prefixed):
    Request:  {"command": "step", "rgb": base64_jpg, "rgb_down": base64_jpg, "instruction": "..."}
    Response: {"actions": [1, 2, 1, 1, ...], "elapsed": 0.5}

    Request:  {"command": "reset"}
    Response: {"status": "ok"}

    Request:  {"command": "shutdown"}
    Response: {"status": "bye"}
"""

import argparse
import base64
import io
import json
import os
import socket
import struct
import sys
import threading
import time

import numpy as np
from PIL import Image

# Add FastWAM source to path
FASTWAM_SRC = "/apdcephfs_tj5/share_302528826/xxd/FastWAM/src"
sys.path.insert(0, FASTWAM_SRC)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastwam_agent import FastWAMNavAgent


def parse_args():
    parser = argparse.ArgumentParser(description="FastWAM VLN Inference Server (Multi-Client)")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to trained checkpoint")
    parser.add_argument("--model_config", type=str,
                        default="/apdcephfs_tj5/share_302528826/xxd/FastWAM/configs/model/fastwam_nav.yaml")
    parser.add_argument("--port", type=int, default=9527, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--action_horizon", type=int, default=32)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--re_infer_interval", type=int, default=1)
    parser.add_argument("--max_clients", type=int, default=4, help="Max simultaneous clients")
    # Standalone stop head (optional)
    parser.add_argument("--stop_head_checkpoint", type=str, default="",
                        help="Path to standalone stop head checkpoint (.pt). "
                             "Leave empty to auto-detect the latest .pt in "
                             "--stop_head_scan_dir (or disable if dir is empty).")
    parser.add_argument("--stop_head_scan_dir", type=str,
                        default="/apdcephfs_tj5/share_302528826/xxd/nav_vln_1e-4/stop_head",
                        help="Directory to scan for the latest stop head checkpoint "
                             "when --stop_head_checkpoint is not specified.")
    parser.add_argument("--stop_head_vae_path", type=str,
                        default="/tmp/fastwam_checkpoints",
                        help="VAE path for stop head (DIFFSYNTH_MODEL_BASE_PATH)")
    parser.add_argument("--stop_head_threshold", type=float, default=0.5,
                        help="Stop head probability threshold (default 0.5)")
    parser.add_argument("--stop_head_ensemble", action="store_true", default=True,
                        help="If set, STOP when EITHER moving_flag OR stop head triggers. "
                             "If not set, stop head is the sole stop signal.")
    parser.add_argument("--waypoint_mode", action="store_true", default=False,
                        help="If set, use polar-coordinate continuous waypoint inference instead "
                             "of discrete LEFT/RIGHT/FORWARD actions. "
                             "Returns {'action': [x, y]} per step, or {'action': [0]} for STOP.")
    return parser.parse_args()


def find_latest_stop_head(scan_dir: str) -> str:
    """
    Scan scan_dir recursively for the newest *.pt file.
    Returns the path, or empty string if none found.
    """
    import glob
    candidates = glob.glob(os.path.join(scan_dir, "**", "*.pt"), recursive=True)
    if not candidates:
        return ""
    latest = max(candidates, key=os.path.getmtime)
    return latest


def send_msg(conn, data: dict):
    """Send a JSON message with length prefix."""
    msg = json.dumps(data).encode("utf-8")
    conn.sendall(struct.pack(">I", len(msg)) + msg)


def recv_msg(conn) -> dict:
    """Receive a JSON message with length prefix."""
    raw_len = _recv_n(conn, 4)
    if not raw_len:
        return None
    msg_len = struct.unpack(">I", raw_len)[0]
    raw_msg = _recv_n(conn, msg_len)
    if not raw_msg:
        return None
    return json.loads(raw_msg.decode("utf-8"))


def _recv_n(conn, n):
    """Receive exactly n bytes."""
    data = b""
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def handle_client(conn, addr, agent, infer_lock):
    """
    Handle a single client connection in its own thread.

    Each client gets independent episode state via agent.reset()/step(),
    but GPU inference is serialized through infer_lock.
    """
    client_id = f"{addr[0]}:{addr[1]}"
    print(f"[{client_id}] Client connected")

    # Each client has its own episode counter
    episode_step = 0

    try:
        while True:
            request = recv_msg(conn)
            if request is None:
                print(f"[{client_id}] Client disconnected")
                break

            command = request.get("command", "step")

            if command == "shutdown":
                send_msg(conn, {"status": "bye"})
                print(f"[{client_id}] Shutdown requested")
                break

            elif command == "reset":
                episode_step = 0
                send_msg(conn, {"status": "ok"})

            elif command == "step":
                # Decode RGB image (forward camera)
                rgb_b64 = request.get("rgb")
                instruction = request.get("instruction", "")

                if rgb_b64:
                    img_bytes = base64.b64decode(rgb_b64)
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    rgb = np.array(img)
                else:
                    rgb = np.zeros((480, 640, 3), dtype=np.uint8)

                # Decode downward camera if available
                rgb_down = None
                rgb_down_b64 = request.get("rgb_down")
                if rgb_down_b64:
                    img_bytes = base64.b64decode(rgb_down_b64)
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    rgb_down = np.array(img)

                # Run inference (serialized through lock for GPU safety)
                t0 = time.time()
                obs = {"rgb": rgb, "rgb_down": rgb_down, "instruction": instruction}
                with infer_lock:
                    result = agent.step(obs)
                elapsed = time.time() - t0

                import numpy as _np
                raw_action = result["action"]
                response_waypoint = None
                if isinstance(raw_action, _np.ndarray):
                    # Waypoint mode: [r, theta] float array — polar coordinates for GO_TOWARD_POINT
                    response_waypoint = raw_action.tolist()
                    actions = [1]  # non-STOP marker; client uses GO_TOWARD_POINT
                elif (isinstance(raw_action, list)
                      and len(raw_action) == 2
                      and isinstance(raw_action[0], float)):
                    # Already a list [r, theta]
                    response_waypoint = raw_action
                    actions = [1]
                elif isinstance(raw_action, list) and raw_action == [0]:
                    # Waypoint STOP signal
                    actions = [0]
                else:
                    # Discrete mode or STOP: raw_action is [int] or int
                    actions = raw_action if isinstance(raw_action, list) else [raw_action]
                episode_step += 1
                response = {"actions": actions, "elapsed": elapsed}
                if response_waypoint is not None:
                    response["waypoint"] = response_waypoint
                if "trajectory" in result:
                    response["trajectory"] = result["trajectory"]
                    # Log trajectory summary
                    traj = result["trajectory"]
                    if traj and len(traj) > 0:
                        import numpy as _np
                        traj_arr = _np.array(traj)
                        endpoint = traj_arr[-1]
                        dist = _np.linalg.norm(endpoint[:2])
                        stop_prob_str = ""
                        if "stop_prob" in result:
                            response["stop_prob"] = result["stop_prob"]
                            stop_prob_str = f" stop_prob={result['stop_prob']:.3f}"
                        print(f"[{client_id}] step={episode_step} traj_endpoint=({endpoint[0]:.3f}, {endpoint[1]:.3f}, {endpoint[2]:.3f}) "
                              f"dist={dist:.3f}m actions={actions}{stop_prob_str}")
                elif "stop_prob" in result:
                    response["stop_prob"] = result["stop_prob"]
                send_msg(conn, response)

            else:
                send_msg(conn, {"error": f"Unknown command: {command}"})

    except Exception as e:
        print(f"[{client_id}] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()
        print(f"[{client_id}] Connection closed")


def main():
    args = parse_args()

    # Create agent (shared model, but step() must be called with lock)
    print("=" * 60)
    print("FastWAM VLN Inference Server (Multi-Client)")
    print("=" * 60)
    print(f"Loading model (device={args.device})...")

    # Auto-detect latest stop head checkpoint if not explicitly provided
    stop_head_ckpt = args.stop_head_checkpoint
    if not stop_head_ckpt and args.stop_head_scan_dir:
        stop_head_ckpt = find_latest_stop_head(args.stop_head_scan_dir)
        if stop_head_ckpt:
            print(f"[StopHead] Auto-detected latest checkpoint: {stop_head_ckpt}")
        else:
            print(f"[StopHead] No checkpoints found in {args.stop_head_scan_dir}, stop head disabled.")

    agent = FastWAMNavAgent(
        checkpoint_path=args.checkpoint,
        model_config_path=args.model_config,
        device=args.device,
        action_horizon=args.action_horizon,
        num_inference_steps=args.num_inference_steps,
        re_infer_interval=args.re_infer_interval,
        stop_head_checkpoint_path=stop_head_ckpt or None,
        stop_head_vae_path=args.stop_head_vae_path,
        stop_head_threshold=args.stop_head_threshold,
        stop_head_ensemble=args.stop_head_ensemble,
        waypoint_mode=args.waypoint_mode,
    )
    agent.reset()
    print("Model loaded successfully.")

    # Lock for serializing GPU inference
    infer_lock = threading.Lock()

    # Start server
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(args.max_clients)
    print(f"Server listening on {args.host}:{args.port} (max {args.max_clients} clients)")
    print("Waiting for client connections...")

    try:
        while True:
            conn, addr = server.accept()
            t = threading.Thread(
                target=handle_client,
                args=(conn, addr, agent, infer_lock),
                daemon=True,
            )
            t.start()
    except KeyboardInterrupt:
        print("\nServer shutting down.")
    finally:
        server.close()


if __name__ == "__main__":
    main()

"""
FastWAM Client Agent for InternNav evaluation framework.

Runs in the internnaveval environment. Sends observations to the FastWAM
inference server and receives discrete actions back.

This file should be placed alongside the InternNav eval code.
"""

import base64
import io
import json
import socket
import struct
import time

import numpy as np
from PIL import Image

from internnav.agent.base import Agent
from internnav.configs.agent import AgentCfg


def _send_msg(sock, data: dict):
    """Send a JSON message with length prefix."""
    msg = json.dumps(data).encode("utf-8")
    sock.sendall(struct.pack(">I", len(msg)) + msg)


def _recv_msg(sock) -> dict:
    """Receive a JSON message with length prefix."""
    raw_len = b""
    while len(raw_len) < 4:
        chunk = sock.recv(4 - len(raw_len))
        if not chunk:
            return None
        raw_len += chunk
    msg_len = struct.unpack(">I", raw_len)[0]
    raw_msg = b""
    while len(raw_msg) < msg_len:
        chunk = sock.recv(msg_len - len(raw_msg))
        if not chunk:
            return None
        raw_msg += chunk
    return json.loads(raw_msg.decode("utf-8"))


@Agent.register('fastwam_nav')
class FastWAMClientAgent(Agent):
    """
    Lightweight agent client that communicates with FastWAM inference server.
    No heavy dependencies (torch, fastwam) needed in this environment.
    """

    def __init__(self, config: AgentCfg):
        super().__init__(config)
        settings = config.model_settings
        self.server_host = settings.get("server_host", "127.0.0.1")
        self.server_port = settings.get("server_port", 9527)
        self.max_retries = settings.get("max_retries", 20)
        self.episode_step = 0

        # Connect to FastWAM server
        self._connect()

    def _connect(self):
        """Connect to the FastWAM inference server."""
        for attempt in range(self.max_retries):
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect((self.server_host, self.server_port))
                print(f"Connected to FastWAM server at {self.server_host}:{self.server_port}")
                return
            except ConnectionRefusedError:
                print(f"Connection attempt {attempt+1}/{self.max_retries} failed, retrying in 3s...")
                time.sleep(3)
        raise RuntimeError(f"Cannot connect to FastWAM server at {self.server_host}:{self.server_port}")

    def _encode_image(self, rgb: np.ndarray) -> str:
        """Encode RGB image as base64 JPEG string."""
        img = Image.fromarray(rgb)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def reset(self, reset_index=None):
        """Reset agent for a new episode."""
        self.episode_step = 0
        # Notify server
        _send_msg(self.sock, {"command": "reset"})
        resp = _recv_msg(self.sock)
        if resp is None:
            # Reconnect if connection lost
            self._connect()
            _send_msg(self.sock, {"command": "reset"})
            _recv_msg(self.sock)

    def _waypoint_to_discrete(self, waypoint, turn_angle_deg=15.0):
        """
        Convert continuous waypoint [x_target, y_target] to a single discrete Habitat action.

        FastWAM local frame: x = lateral (positive = right), y = forward (positive = forward).
        Returns one of: 0=STOP, 1=FORWARD, 2=LEFT, 3=RIGHT.
        """
        import math
        x, y = float(waypoint[0]), float(waypoint[1])
        if abs(x) < 1e-6 and abs(y) < 1e-6:
            return 0  # STOP — zero-length waypoint
        # Angle from forward axis (positive = rightward, negative = leftward)
        angle_deg = math.degrees(math.atan2(x, y))
        half = turn_angle_deg / 2.0
        if abs(angle_deg) <= half:
            return 1  # FORWARD
        elif angle_deg > 0:
            return 3  # RIGHT
        else:
            return 2  # LEFT

    def step(self, obs):
        """
        Agent step: send observation to server, get exactly 1 action back.

        Args:
            obs: list of dicts, each with 'rgb', 'rgb_down', 'depth', 'instruction'.

        Returns:
            list of dicts with 'action' and 'ideal_flag'.
        """
        obs = obs[0]  # Single env
        rgb = obs['rgb']
        rgb_down = obs.get('rgb_down', None)
        instruction = obs.get('instruction', '')

        # Request new inference from server every step
        rgb_b64 = self._encode_image(rgb)
        request = {
            "command": "step",
            "rgb": rgb_b64,
            "instruction": instruction,
        }
        # Include downward camera if available
        if rgb_down is not None:
            request["rgb_down"] = self._encode_image(rgb_down)

        try:
            _send_msg(self.sock, request)
            response = _recv_msg(self.sock)
        except (BrokenPipeError, ConnectionResetError):
            print("Connection lost, reconnecting...")
            self._connect()
            _send_msg(self.sock, request)
            response = _recv_msg(self.sock)

        if response is None:
            print("ERROR: No response from server, returning STOP")
            return [{'action': [0], 'ideal_flag': True}]

        actions = response.get("actions", [0])
        waypoint = response.get("waypoint", None)  # [x, y] in waypoint mode, else None
        elapsed = response.get("elapsed", 0)
        trajectory = response.get("trajectory", None)
        stop_prob = response.get("stop_prob", None)

        # Log full trajectory (all 32 points) and action info
        if trajectory and len(trajectory) > 0:
            import numpy as _np
            traj_arr = _np.array(trajectory)
            endpoint = traj_arr[-1]
            dist = _np.linalg.norm(endpoint[:2])
            stop_prob_str = f" stop_prob={stop_prob:.3f}" if stop_prob is not None else ""
            waypoint_str = f" waypoint=({waypoint[0]:.3f},{waypoint[1]:.3f})" if waypoint else ""
            # Print summary
            print(f"[FastWAM] step={self.episode_step} actions={actions} in {elapsed:.2f}s | "
                  f"traj_endpoint=({endpoint[0]:.3f}, {endpoint[1]:.3f}, {endpoint[2]:.3f}) "
                  f"dist={dist:.3f}m{waypoint_str}{stop_prob_str}")
            # Print all 32 trajectory points
            traj_str = " ".join([f"({t[0]:.2f},{t[1]:.2f},{t[2]:.2f})" for t in traj_arr[:, :3]])
            print(f"[FastWAM] traj_points: [{traj_str}]")
            # Print moving_flag if 4D
            if traj_arr.shape[1] >= 4:
                flags = traj_arr[:, 3]
                n_stop = int(_np.sum(flags < 0.5))
                print(f"[FastWAM] moving_flags: {n_stop}/32 stopped, values=[{','.join([f'{f:.2f}' for f in flags])}]")
        else:
            stop_prob_str = f" stop_prob={stop_prob:.3f}" if stop_prob is not None else ""
            print(f"[FastWAM] step={self.episode_step} actions={actions} in {elapsed:.2f}s{stop_prob_str}")

        if waypoint is not None:
            # Waypoint mode: server sends [x_target, y_target] in agent-local frame.
            # actions[0]==0 means STOP signal; otherwise convert waypoint to discrete action.
            if actions[0] == 0:
                action = 0  # STOP
            else:
                action = self._waypoint_to_discrete(waypoint)
            print(f"[FastWAM] waypoint=({waypoint[0]:.3f},{waypoint[1]:.3f}) → discrete={action}")
        else:
            # Discrete mode: server returns a Habitat int action directly.
            action = actions[0] if actions else 0

        self.episode_step += 1
        result = {'action': [action], 'ideal_flag': True}
        if trajectory is not None:
            result['trajectory'] = trajectory
        if stop_prob is not None:
            result['stop_prob'] = stop_prob
        if waypoint is not None:
            result['waypoint'] = waypoint
        return [result]

    def __del__(self):
        try:
            self.sock.close()
        except:
            pass

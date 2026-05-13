"""
Trajectory to discrete action conversion utilities for FastWAM VLN evaluation.

Converts continuous (x, y, theta) trajectories output by FastWAM
into discrete navigation actions (STOP/FORWARD/LEFT/RIGHT).

Coordinate System (from training data):
    - x = lateral (positive = right)
    - y = forward (positive = forward / agent's facing direction)
    - theta = heading change (radians)
    - Agent initially faces along +y axis (yaw = pi/2 in standard math convention)
"""

from typing import Optional

import numpy as np


# Discrete action codes (matching Habitat)
STOP = 0
FORWARD = 1
LEFT = 2
RIGHT = 3


def normalize_angle(angle):
    """Normalize angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def fastwam_traj_to_actions(
    trajectory: np.ndarray,
    step_size: float = 0.25,
    turn_angle_deg: float = 15,
    lookahead: int = 4,
    goal_threshold: float = 0.2,
    moving_flag_threshold: float = 0.5,
    max_actions: int = 50,
) -> list:
    """
    Convert FastWAM output trajectory [T, 3] or [T, 4] to discrete actions.

    Supports two formats:
    - [T, 3]: (x, y, theta) — no stop signal, always navigates toward endpoint
    - [T, 4]: (x, y, theta, moving_flag) — STOP if more than 3/4 of waypoints
              have moving_flag < threshold

    Coordinate convention (matching training):
        - x = lateral (positive right)
        - y = forward (positive forward, agent's initial facing direction)
        - theta = heading change
        - Agent starts at origin facing +y direction

    Args:
        trajectory: [T, 3] or [T, 4] array relative to current pose.
        step_size: Distance per FORWARD action (meters). Default 0.25m.
        turn_angle_deg: Angle per LEFT/RIGHT action (degrees). Default 15.
        lookahead: Number of waypoints to look ahead for target selection.
        goal_threshold: Distance to goal to consider reached (meters).
        moving_flag_threshold: For 4D model, threshold below which moving_flag means "stop".
        max_actions: Maximum number of discrete actions to generate.

    Returns:
        List[int]: Sequence of discrete actions (STOP=0, FORWARD=1, LEFT=2, RIGHT=3).
    """
    if trajectory is None or len(trajectory) == 0:
        return [STOP]

    # Check if 4D model (x, y, theta, moving_flag)
    has_moving_flag = trajectory.shape[1] >= 4
    if has_moving_flag:
        moving_flags = trajectory[:, 3]  # [T]
        # STOP if more than 3/4 of the waypoints have moving_flag < threshold
        stop_steps = np.sum(moving_flags < moving_flag_threshold)
        stop_ratio = stop_steps / len(moving_flags)
        if stop_ratio > 0.75:
            print(f"[MoveFlag] stop_steps={stop_steps}/{len(moving_flags)} ({stop_ratio:.1%}) > 75% → STOP")
            return [STOP]
        else:
            print(f"[MoveFlag] stop_steps={stop_steps}/{len(moving_flags)} ({stop_ratio:.1%}) ≤ 75% → go")

    # Extract 2D trajectory points (x, y)
    # In FastWAM's local frame: x = lateral (right), y = forward
    traj_2d = trajectory[:, :2].copy()  # [T, 2] - (x, y) in local frame

    goal = traj_2d[-1]

    turn_angle_rad = np.deg2rad(turn_angle_deg)

    actions = []
    pos = np.array([0.0, 0.0])  # Start at origin (x=0, y=0)
    # Agent initially faces +y direction → yaw = pi/2 in standard math (atan2 convention)
    yaw = np.pi / 2.0

    while np.linalg.norm(pos - goal) > goal_threshold and len(actions) < max_actions:
        # Find nearest trajectory point to current position
        dists = np.linalg.norm(traj_2d - pos, axis=1)
        nearest_idx = np.argmin(dists)

        # Look ahead to find target point
        target_idx = min(nearest_idx + lookahead, len(traj_2d) - 1)
        target = traj_2d[target_idx]

        # Compute direction to target in standard math convention
        target_dir = target - pos
        if np.linalg.norm(target_dir) < 1e-6:
            break

        # Target yaw: atan2(y, x) gives angle from +x axis
        # Since our coordinate is x=lateral, y=forward, this naturally works
        target_yaw = np.arctan2(target_dir[1], target_dir[0])

        # Compute required turn (positive = counter-clockwise = LEFT in Habitat)
        delta_yaw = normalize_angle(target_yaw - yaw)
        n_turns = int(round(delta_yaw / turn_angle_rad))

        # Add turn actions
        # Positive delta_yaw = turn left (counter-clockwise)
        # Negative delta_yaw = turn right (clockwise)
        if n_turns > 0:
            actions.extend([LEFT] * n_turns)
        elif n_turns < 0:
            actions.extend([RIGHT] * (-n_turns))

        # Update yaw after turning
        yaw = normalize_angle(yaw + n_turns * turn_angle_rad)

        # Move forward one step in current facing direction
        next_pos = pos + step_size * np.array([np.cos(yaw), np.sin(yaw)])

        # If moving forward makes us farther from goal, stop moving
        if np.linalg.norm(next_pos - goal) > np.linalg.norm(pos - goal):
            break

        actions.append(FORWARD)
        pos = next_pos

    # If we reached the goal (loop ended naturally), append STOP
    if np.linalg.norm(pos - goal) <= goal_threshold:
        actions.append(STOP)

    # If no actions generated, output STOP
    if len(actions) == 0:
        actions = [STOP]

    return actions


def fastwam_traj_to_waypoint(
    trajectory: np.ndarray,
    moving_flag_threshold: float = 0.5,
    stop_ratio_threshold: float = 0.90,
) -> Optional[np.ndarray]:
    """
    Convert FastWAM trajectory to a single continuous waypoint using polar coordinates.

    Instead of mapping to discrete LEFT/RIGHT/FORWARD actions, this function uses the
    model's theta output (3rd dimension) directly as a polar angle, combined with the
    Euclidean distance of the first waypoint as radius, to compute a target (x, y)
    position in the agent-local frame.

    Polar coordinate convention:
        r     = ||trajectory[0, :2]||  — distance of first predicted point from origin
        theta = trajectory[0, 2]       — polar angle (radians), 3rd model output dimension
        x_target = r * cos(theta)
        y_target = r * sin(theta)

    Stop condition:
        Returns None (STOP) if more than `stop_ratio_threshold` (default 90%) of the
        32 moving_flags are below `moving_flag_threshold`.

    Args:
        trajectory: [T, 3] or [T, 4] array in agent-local frame.
                    dim 0 = x (lateral, positive right)
                    dim 1 = y (forward, positive forward)
                    dim 2 = theta (polar angle, radians)
                    dim 3 = moving_flag (optional, 0=stop 1=move)
        moving_flag_threshold: Value below which a moving_flag counts as "stop". Default 0.5.
        stop_ratio_threshold: Fraction of stop votes required to trigger STOP. Default 0.90.

    Returns:
        np.ndarray([x_target, y_target]) — continuous target in agent-local frame, or
        None — if the stop condition is triggered.
    """
    if trajectory is None or len(trajectory) == 0:
        return None  # STOP

    # Stop check: >stop_ratio_threshold of moving_flags say stop
    if trajectory.shape[1] >= 4:
        moving_flags = trajectory[:, 3]
        stop_steps = np.sum(moving_flags < moving_flag_threshold)
        stop_ratio = stop_steps / len(moving_flags)
        if stop_ratio > stop_ratio_threshold:
            print(f"[Waypoint] stop_steps={stop_steps}/{len(moving_flags)} "
                  f"({stop_ratio:.1%}) > {stop_ratio_threshold:.0%} → STOP")
            return None  # STOP
        else:
            print(f"[Waypoint] stop_steps={stop_steps}/{len(moving_flags)} "
                  f"({stop_ratio:.1%}) ≤ {stop_ratio_threshold:.0%} → move")

    # Polar coordinate conversion using first waypoint
    first = trajectory[0]
    x0, y0 = float(first[0]), float(first[1])
    r = np.sqrt(x0 ** 2 + y0 ** 2)
    theta = float(first[2])  # 3rd model output dimension used as polar angle

    x_target = r * np.cos(theta)
    y_target = r * np.sin(theta)

    print(f"[Waypoint] first=({x0:.3f}, {y0:.3f}) r={r:.3f} theta={theta:.3f} "
          f"→ target=({x_target:.3f}, {y_target:.3f})")
    return np.array([x_target, y_target], dtype=np.float32)

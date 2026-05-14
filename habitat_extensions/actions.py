"""
Custom Habitat task actions for FastWAM VLN evaluation.

Registers GoTowardPoint — a continuous waypoint action parameterized by
polar coordinates (r, theta) that teleports the agent along a straight line
toward the specified point, respecting the navmesh.

This file is self-contained (no VLN_CE dependency). It must be imported
before the Habitat environment is instantiated so that:
  1. @registry.register_task_action takes effect, and
  2. The GoTowardPointActionConfig is stored in the Hydra ConfigStore.
"""

from dataclasses import dataclass, field
from typing import Any, List

import numpy as np
import habitat_sim
from gym import spaces
from hydra.core.config_store import ConfigStore
from habitat.config.default_structured_configs import ActionConfig
from habitat.core.registry import registry
from habitat.core.simulator import Observations
from habitat.tasks.nav.nav import TeleportAction


# ---------------------------------------------------------------------------
# Helper functions (extracted from OmniNav's VLN_CE/habitat_extensions/utils)
# ---------------------------------------------------------------------------

def rtheta_to_global_coordinates(
    sim,
    r: float,
    theta: float,
    y_delta: float = 0.0,
    dimensionality: int = 2,
) -> List[float]:
    """Convert polar (r, theta) relative to agent forward into a world-frame position.

    theta is a rotation angle around the UP axis (positive = rightward turn).
    The returned position is NOT validated for navigability.
    """
    assert dimensionality in [2, 3]
    scene_node = sim.get_agent(0).scene_node
    forward_ax = (
        np.array(scene_node.absolute_transformation().rotation_scaling())
        @ habitat_sim.geo.FRONT
    )
    agent_state = sim.get_agent_state()
    rotation = habitat_sim.utils.quat_from_angle_axis(theta, habitat_sim.geo.UP)
    move_ax = habitat_sim.utils.quat_rotate_vector(rotation, forward_ax)
    position = agent_state.position + (move_ax * r)
    position[1] += y_delta
    if dimensionality == 2:
        return [position[0], position[2]]
    return position


def compute_heading_to(pos_from, pos_to):
    """Compute quaternion heading from pos_from toward pos_to in the XZ plane.

    Returns:
        (quat_list [x,y,z,w], scalar_angle_radians)
    """
    import quaternion as quat_module
    from habitat.utils.geometry_utils import quaternion_to_list

    delta_x = pos_to[0] - pos_from[0]
    delta_z = pos_to[-1] - pos_from[-1]
    xz_angle = np.arctan2(delta_x, delta_z)
    xz_angle = (xz_angle + np.pi) % (2 * np.pi)
    q = quaternion_to_list(quat_module.from_euler_angles([0.0, xz_angle, 0.0]))
    return q, xz_angle


# ---------------------------------------------------------------------------
# Action config dataclass — registered in Hydra ConfigStore
# ---------------------------------------------------------------------------

@dataclass
class GoTowardPointActionConfig(ActionConfig):
    """Config for the GoTowardPoint continuous waypoint action.

    rotate_agent: if True the agent's heading is updated to face the target
                  even when blocked by a wall.
    """
    type: str = "GoTowardPoint"
    rotate_agent: bool = True


# Register in Hydra ConfigStore so it can be listed in vln_r2r_xxd.yaml
_cs = ConfigStore.instance()
_cs.store(
    package="habitat.task.actions.go_toward_point",
    group="habitat/task/actions",
    name="go_toward_point",
    node=GoTowardPointActionConfig,
)


# ---------------------------------------------------------------------------
# Habitat task action — registered via @registry
# ---------------------------------------------------------------------------

@registry.register_task_action
class GoTowardPoint(TeleportAction):
    """Waypoint action parameterized by (r, theta) in polar coordinates.

    r     — distance in metres from the agent's current position
    theta — heading offset in radians around the UP axis
              (0 = straight ahead; positive = rightward)

    The agent is teleported as far as possible along the straight line toward
    the target, stopping at the navmesh boundary on collision.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # rotate_agent defaults to True if not in config
        self._rotate_agent = getattr(self._config, "rotate_agent", True)

    def step(
        self,
        *args: Any,
        r: float,
        theta: float,
        **kwargs: Any,
    ) -> Observations:
        y_delta = kwargs.get("y_delta", 0.0)
        pos = rtheta_to_global_coordinates(
            self._sim, r, theta, y_delta=y_delta, dimensionality=3
        )

        agent_pos = self._sim.get_agent_state().position
        new_pos = np.array(self._sim.step_filter(agent_pos, pos))
        new_rot = self._sim.get_agent_state().rotation

        if np.any(np.isnan(new_pos)) or not self._sim.is_navigable(new_pos):
            new_pos = agent_pos
            if self._rotate_agent:
                new_rot, _ = compute_heading_to(agent_pos, pos)
        else:
            new_pos = np.array(self._sim.pathfinder.snap_point(new_pos))
            if np.any(np.isnan(new_pos)) or not self._sim.is_navigable(new_pos):
                new_pos = agent_pos
            if self._rotate_agent:
                new_rot, _ = compute_heading_to(agent_pos, pos)

        assert np.all(np.isfinite(new_pos))
        return self._sim.get_observations_at(
            position=new_pos, rotation=new_rot, keep_agent_at_new_pose=True
        )

    @property
    def action_space(self) -> spaces.Dict:
        coord_range = self.COORDINATE_MAX - self.COORDINATE_MIN
        return spaces.Dict(
            {
                "r": spaces.Box(
                    low=np.array([0.0]),
                    high=np.array([np.sqrt(2 * (coord_range ** 2))]),
                    dtype=float,
                ),
                "theta": spaces.Box(
                    low=np.array([0.0]),
                    high=np.array([2 * np.pi]),
                    dtype=float,
                ),
            }
        )

"""
Guidance Algorithm
Converts gate observations + vehicle state into a desired position/velocity
target to feed the autopilot.

Architecture:
  GuidanceAlgorithm
    ├── CourseMap          — ordered list of known/estimated gate positions
    ├── WaypointSequencer  — selects next waypoint; advances when gate is passed
    └── ObstacleAvoider    — modifies waypoint to steer around obstacles
"""

import math
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from anduril_gp.core.state_estimator import VehicleState
from anduril_gp.perception.perception import GateObservation, Obstacle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class Waypoint:
    position_ned: np.ndarray          # (N, E, D) metres
    target_speed: float = 8.0         # m/s through this waypoint
    label: str = ''

    def __repr__(self):
        n, e, d = self.position_ned
        return f"Waypoint({self.label} N={n:.1f} E={e:.1f} D={d:.1f} v={self.target_speed:.1f})"


@dataclass
class GuidanceOutput:
    """Desired state for the autopilot to track."""
    target_position: np.ndarray       # NED metres
    target_velocity: np.ndarray       # NED m/s (feedforward)
    target_yaw: float                 # radians
    target_speed: float               # scalar m/s
    waypoint_index: int = 0
    course_complete: bool = False


# ---------------------------------------------------------------------------
# Course Map
# ---------------------------------------------------------------------------

class CourseMap:
    """
    Holds the ordered list of gate waypoints.
    Gates can be pre-loaded from a config file or built up live from vision.
    """

    def __init__(self):
        self.waypoints: List[Waypoint] = []

    def load_from_list(self, positions: List[Tuple], speed: float = 8.0):
        """positions: list of (N, E, D) tuples."""
        self.waypoints = [
            Waypoint(np.array(p, dtype=float), target_speed=speed, label=f"Gate_{i}")
            for i, p in enumerate(positions)
        ]
        logger.info(f"CourseMap loaded {len(self.waypoints)} waypoints.")

    def update_gate_from_observation(self, obs: GateObservation,
                                      vehicle_state: VehicleState,
                                      gate_index: int):
        """
        Refine a gate's position using a visual observation.
        obs.bearing_body is a unit vector in body NED;
        rotate to world NED and project at obs.range_estimate.
        """
        R = vehicle_state.body_to_ned_R()
        bearing_ned = R @ obs.bearing_body
        pos_ned = vehicle_state.position_ned() + bearing_ned * obs.range_estimate

        if gate_index < len(self.waypoints):
            # EMA update
            alpha = 0.3
            self.waypoints[gate_index].position_ned = (
                alpha * pos_ned + (1 - alpha) * self.waypoints[gate_index].position_ned
            )
        else:
            self.waypoints.append(Waypoint(pos_ned, label=f"Gate_{gate_index}"))


# ---------------------------------------------------------------------------
# Waypoint Sequencer
# ---------------------------------------------------------------------------

class WaypointSequencer:
    """Advances through the course map as waypoints are reached."""

    # A gate is "passed" when the drone is within this radius AND has moved past it
    CAPTURE_RADIUS_M = 3.0

    def __init__(self, course_map: CourseMap):
        self.course_map = course_map
        self.current_index = 0

    def current_waypoint(self) -> Optional[Waypoint]:
        wps = self.course_map.waypoints
        if not wps or self.current_index >= len(wps):
            return None
        return wps[self.current_index]

    def update(self, state: VehicleState) -> bool:
        """
        Check if current waypoint has been passed; advance if so.
        Returns True if course is complete.
        """
        wp = self.current_waypoint()
        if wp is None:
            return True

        dist = float(np.linalg.norm(wp.position_ned - state.position_ned()))
        if dist < self.CAPTURE_RADIUS_M:
            logger.info(f"Waypoint captured: {wp}")
            self.current_index += 1
            if self.current_index >= len(self.course_map.waypoints):
                logger.info("Course complete!")
                return True
        return False

    @property
    def is_complete(self) -> bool:
        return self.current_index >= len(self.course_map.waypoints)


# ---------------------------------------------------------------------------
# Obstacle Avoider
# ---------------------------------------------------------------------------

class ObstacleAvoider:
    """
    Modifies the direct waypoint vector to steer around detected obstacles.
    Uses a simple potential-field repulsion approach.
    """

    REPULSION_GAIN = 5.0      # m²  — scales repulsion force
    REPULSION_RANGE = 6.0     # m   — max range obstacles affect guidance

    def avoid(self, desired_pos: np.ndarray,
              vehicle_pos: np.ndarray,
              obstacles: List[Obstacle],
              state: VehicleState) -> np.ndarray:
        """
        Returns modified desired_pos with obstacle repulsion applied.
        Obstacles are in body NED frame; we rotate to world NED.
        """
        if not obstacles:
            return desired_pos

        R = state.body_to_ned_R()
        repulsion = np.zeros(3)

        for obs in obstacles:
            obs_world = vehicle_pos + R @ obs.position_body
            diff = vehicle_pos - obs_world
            dist = np.linalg.norm(diff)
            if 0 < dist < self.REPULSION_RANGE:
                strength = self.REPULSION_GAIN / (dist ** 2)
                repulsion += (diff / dist) * strength

        return desired_pos + repulsion


# ---------------------------------------------------------------------------
# Guidance Algorithm (main class)
# ---------------------------------------------------------------------------

class GuidanceAlgorithm:
    """
    Main guidance loop. Call compute() at ~50 Hz.

    Inputs:  VehicleState, gate observations, obstacles
    Outputs: GuidanceOutput (desired position + velocity for autopilot)
    """

    LOOKAHEAD_M = 5.0       # pure-pursuit lookahead distance

    def __init__(self, course_map: Optional[CourseMap] = None):
        self.course_map   = course_map or CourseMap()
        self.sequencer    = WaypointSequencer(self.course_map)
        self.avoider      = ObstacleAvoider()

    # ------------------------------------------------------------------

    def compute(self, state: VehicleState,
                gates: List[GateObservation],
                obstacles: List[Obstacle]) -> GuidanceOutput:

        # 1. Advance sequencer
        complete = self.sequencer.update(state)

        # 2. Get current target waypoint
        wp = self.sequencer.current_waypoint()
        if wp is None or complete:
            return GuidanceOutput(
                target_position=state.position_ned(),
                target_velocity=np.zeros(3),
                target_yaw=state.yaw,
                target_speed=0.0,
                course_complete=True,
            )

        # 3. Refine gate position from vision if we have a good observation
        if gates:
            best = max(gates, key=lambda g: g.confidence)
            if best.confidence > 0.5:
                self.course_map.update_gate_from_observation(
                    best, state, self.sequencer.current_index
                )

        # 4. Pure-pursuit: pick a lookahead point on the line to the waypoint
        pos       = state.position_ned()
        to_wp     = wp.position_ned - pos
        dist      = np.linalg.norm(to_wp)
        direction = to_wp / max(dist, 0.1)

        lookahead = min(self.LOOKAHEAD_M, dist)
        target_pos = pos + direction * lookahead

        # 5. Obstacle avoidance modifies target position
        target_pos = self.avoider.avoid(target_pos, pos, obstacles, state)

        # 6. Target velocity (feedforward in direction of travel)
        speed    = wp.target_speed
        vel_ff   = direction * speed

        # 7. Desired yaw points toward gate
        target_yaw = math.atan2(to_wp[1], to_wp[0])

        return GuidanceOutput(
            target_position=target_pos,
            target_velocity=vel_ff,
            target_yaw=target_yaw,
            target_speed=speed,
            waypoint_index=self.sequencer.current_index,
            course_complete=False,
        )

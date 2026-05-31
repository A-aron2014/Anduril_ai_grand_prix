"""
Pure Pursuit Racing Guidance

Responsibilities:
- Select current target gate
- Generate look-ahead waypoint
- Schedule speed
- Avoid obstacles
- Produce GuidanceOutput
"""

from dataclasses import dataclass
import numpy as np


# --------------------------------------------------
# Output Structure
# --------------------------------------------------

@dataclass
class GuidanceOutput:
    target_position: np.ndarray
    target_velocity: np.ndarray
    target_yaw: float
    course_complete: bool = False


# --------------------------------------------------
# Waypoint
# --------------------------------------------------

@dataclass
class Waypoint:
    position_ned: np.ndarray
    speed_mps: float


# --------------------------------------------------
# Course Map
# --------------------------------------------------

class CourseMap:

    def __init__(self):
        self.waypoints = []

    def load_from_list(self, positions, speed):
        self.waypoints = [
            Waypoint(np.array(p, dtype=float), speed)
            for p in positions
        ]


# --------------------------------------------------
# Gate Sequencer
# --------------------------------------------------

class GateSequencer:

    def __init__(self, course_map):
        self.course_map = course_map
        self.index = 0
        self.pass_distance = 1.5

    def current_waypoint(self):

        if self.index >= len(self.course_map.waypoints):
            return None

        return self.course_map.waypoints[self.index]

    def update(self, drone_pos):

        wp = self.current_waypoint()

        if wp is None:
            return

        distance = np.linalg.norm(
            wp.position_ned - drone_pos
        )

        if distance < self.pass_distance:
            self.index += 1

    def complete(self):

        return self.index >= len(
            self.course_map.waypoints
        )


# --------------------------------------------------
# Guidance Algorithm
# --------------------------------------------------

class GuidanceAlgorithm:

    def __init__(self, course_map):

        self.course_map = course_map
        self.sequencer = GateSequencer(course_map)

        self.lookahead_distance = 3.0
        self.max_speed = 12.0
        self.min_speed = 4.0

        self.obstacle_radius = 3.0

    # ------------------------------------------
    # Main Compute
    # ------------------------------------------

    def compute(
        self,
        state,
        gates,
        obstacles
    ):

        position = state.position_ned()

        self.sequencer.update(position)

        if self.sequencer.complete():

            return GuidanceOutput(
                target_position=position,
                target_velocity=np.zeros(3),
                target_yaw=state.yaw,
                course_complete=True
            )

        wp = self.sequencer.current_waypoint()

        target = self._lookahead_target(
            position,
            wp.position_ned
        )

        speed = self._compute_speed()

        velocity = self._desired_velocity(
            position,
            target,
            speed,
            obstacles
        )

        yaw = np.arctan2(
            velocity[1],
            velocity[0]
        )

        return GuidanceOutput(
            target_position=target,
            target_velocity=velocity,
            target_yaw=yaw,
            course_complete=False
        )

    # ------------------------------------------
    # Lookahead
    # ------------------------------------------

    def _lookahead_target(
        self,
        position,
        gate_pos
    ):

        vec = gate_pos - position

        norm = np.linalg.norm(vec)

        if norm < 1e-6:
            return gate_pos

        direction = vec / norm

        return (
            gate_pos
            + direction * self.lookahead_distance
        )

    # ------------------------------------------
    # Speed Scheduler
    # ------------------------------------------

    def _compute_speed(self):

        current_idx = self.sequencer.index

        if current_idx >= (
            len(self.course_map.waypoints) - 1
        ):
            return self.min_speed

        current_wp = self.course_map.waypoints[current_idx]
        next_wp = self.course_map.waypoints[current_idx + 1]

        if current_idx == 0:
            return current_wp.speed_mps

        prev_wp = self.course_map.waypoints[current_idx - 1]

        v1 = (
            current_wp.position_ned
            - prev_wp.position_ned
        )

        v2 = (
            next_wp.position_ned
            - current_wp.position_ned
        )

        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)

        dot = np.clip(
            np.dot(v1, v2),
            -1.0,
            1.0
        )

        angle = np.degrees(
            np.arccos(dot)
        )

        if angle < 20:
            return 12.0

        if angle < 60:
            return 8.0

        return 5.0

    # ------------------------------------------
    # Velocity Command
    # ------------------------------------------

    def _desired_velocity(
        self,
        position,
        target,
        speed,
        obstacles
    ):

        force = target - position

        # obstacle avoidance
        for obs in obstacles:

            obs_pos = np.array(obs.position)

            vec = position - obs_pos

            dist = np.linalg.norm(vec)

            if (
                dist < self.obstacle_radius
                and dist > 0.1
            ):

                repulsion = (
                    vec
                    / (dist ** 2)
                )

                force += repulsion * 5.0

        norm = np.linalg.norm(force)

        if norm < 1e-6:
            return np.zeros(3)

        direction = force / norm

        return direction * speed
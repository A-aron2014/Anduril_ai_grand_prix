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
    target_altitude: float = 0.0
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

    def update(self, drone_pos) -> bool:
        """
        Advance to the next gate if within pass_distance.
        Returns True if the sequencer just advanced this call
        (i.e. the active leg changed), so callers can rebuild
        any per-leg state (like a line segment).
        """
        wp = self.current_waypoint()
        if wp is None:
            return False

        distance = np.linalg.norm(wp.position_ned - drone_pos)

        if distance < self.pass_distance:
            self.index += 1
            return True

        return False

    def leg_endpoints(self, drone_pos):
        """
        Returns (start, end) NED positions for the *current* leg —
        start is the previous gate (or drone_pos if this is the first
        leg, since there's no 'previous gate' before the course begins),
        end is the current target waypoint.
        """
        wp = self.current_waypoint()
        if wp is None:
            return None, None

        if self.index == 0:
            start = drone_pos
        else:
            start = self.course_map.waypoints[self.index - 1].position_ned

        return start, wp.position_ned

    def complete(self):
        return self.index >= len(self.course_map.waypoints)


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

        advanced = self.sequencer.update(position)

        if self.sequencer.complete():
            return GuidanceOutput(
                target_position=position,
                target_velocity=np.zeros(3),
                target_yaw=state.yaw,
                course_complete=True
            )

        if advanced or not hasattr(self, '_line_dir_n'):
            start, end = self.sequencer.leg_endpoints(position)
            self._build_line_segment(
                (start[0], start[1], start[2]),
                end[0], end[1],
                wp_down=end[2],
            )

        wp = self.sequencer.current_waypoint()

        target = self._lookahead_target(position, wp.position_ned)
        speed = self._compute_speed()
        velocity = self._desired_velocity(position, target, speed, obstacles)

        yaw = self._line_following_yaw(position[0], position[1])
        target_alt = self._line_following_altitude(position[0], position[1], position[2])

        return GuidanceOutput(
            target_position=target,
            target_velocity=velocity,
            target_yaw=yaw,
            target_altitude=target_alt,   # NEW FIELD — see dataclass change below
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
    
   # ------------------------------------------------------------------
    # Straight-Line Path Following (quadcopter, attitude-rate native)
    # ------------------------------------------------------------------

    def _build_line_segment(self, pos: tuple, wp_n: float, wp_e: float, wp_down: float = None):
        """
        Define a straight-line segment from pos (start) to (wp_n, wp_e),
        now also tracking altitude endpoints so _line_following_altitude
        can interpolate target altitude based on progress along the leg,
        not just snap to the next gate's altitude on arrival.

        pos : (north, east, down) — segment start point, full 3D
        wp_n, wp_e : target gate N/E
        wp_down : target gate altitude (down, NED). If omitted, holds
                  the origin's altitude — useful for the very first leg
                  before a target is known.
        """
        self._line_origin_n = pos[0]
        self._line_origin_e = pos[1]
        self._line_origin_down = pos[2]

        self._line_target_down = wp_down if wp_down is not None else pos[2]

        dn = wp_n - pos[0]
        de = wp_e - pos[1]
        norm = np.sqrt(dn * dn + de * de)

        if norm > 1e-3:
            self._line_dir_n = dn / norm
            self._line_dir_e = de / norm
        else:
            self._line_dir_n, self._line_dir_e = 1.0, 0.0

        self._line_length_horizontal = norm

    def _line_following_yaw(self, n: float, e: float, k_path: float = 1.0) -> float:
        """
        Returns a desired yaw (rad) that blends 'point at the line' with
        'point at the gate', correcting cross-track error instead of
        always pointing straight at the target (which lets disturbances
        bow the path into a curve).

        k_path: higher = snaps back onto the line harder. Start at 0.6;
        raise toward ~1.5 if the drone cuts corners, lower if it
        oscillates across the line.
        """
        ex = n - self._line_origin_n
        ey = e - self._line_origin_e

        # Cross-track distance: component of error perpendicular to line_dir
        cross = -self._line_dir_e * ex + self._line_dir_n * ey

        # atan2 over the line direction, then rotate the heading toward the
        # line by an amount proportional to how far off it we are.
        chi_line = np.atan2(self._line_dir_e, self._line_dir_n)
        correction = np.atan(k_path * cross)
        yaw = chi_line - correction

        yaw = (yaw + np.pi)%(2*np.pi)-np.pi #need to normalize pi
        return yaw
    
    def _line_following_altitude(self, n: float, e: float, d: float) -> float:
        """
        Returns a target altitude (down, NED) by interpolating along the
        current line segment based on progress, rather than snapping
        straight to the next gate's altitude. This gives the same
        "stay on the line" behavior vertically that _line_following_yaw
        gives horizontally — if the drone is only 20% of the way along
        the leg, it should be heading toward 20% of the way through the
        altitude change, not already at the target gate's altitude.

        Falls back to the line's end-point altitude if the segment has
        ~zero length (shouldn't happen in practice, but avoids div-by-zero).
        """
        # Vector from line origin to current position, in 3D this time
        ex = n - self._line_origin_n
        ey = e - self._line_origin_e

        # Project (ex, ey) onto the horizontal line direction to get
        # how far along the leg we are, in metres.
        along_dist = ex * self._line_dir_n + ey * self._line_dir_e

        # Total horizontal leg length, recovered from origin/target down
        # — stored alongside the line direction when the segment is built.
        leg_length = self._line_length_horizontal

        if leg_length < 1e-3:
            return self._line_target_down

        # Fraction of the way along the leg, clamped to [0, 1] so we
        # don't extrapolate past the gate if lookahead/overshoot occurs.
        progress = max(0.0, min(1.0, along_dist / leg_length))

        return (
            self._line_origin_down
            + progress * (self._line_target_down - self._line_origin_down)
        )
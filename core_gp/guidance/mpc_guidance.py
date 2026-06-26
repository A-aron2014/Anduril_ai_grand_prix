"""
Linear MPC Trajectory Guidance
-------------------------------
Replaces the straight-line / lookahead guidance with a receding-horizon
linear-quadratic trajectory tracker.

Design rationale
-----------------
The simulator's onboard flight controller is the "black box": given a
position/velocity target it already closes the position -> attitude ->
thrust loop internally. Guidance's only job is to hand it a smooth,
dynamically-consistent 3D reference (position + velocity) each tick --
not to run a second feedback loop on top of it. We model translational
motion as a plain double integrator (position/velocity states,
acceleration as the virtual control input), which is exactly the
abstraction the onboard controller is built to absorb, and solve a
finite-horizon LQ tracking problem against a reference sampled along
the gate course.

There is no QP solver in this environment (no cvxpy/osqp/casadi), so
the problem is solved in closed form via the standard condensed/batch
MPC formulation -- a one-time (per horizon/dt configuration) matrix
factorization, then a matrix-vector product per control tick. Velocity
and acceleration limits are enforced by clamping after the solve
rather than as inequality constraints in the QP itself.

This also fixes two issues in the old straight-line guidance:
  - vertical velocity is part of the same 6-state tracked trajectory,
    not hardcoded to zero.
  - the reference path is a single global polyline through all gates,
    so there is no per-leg origin reset / discontinuity at each gate
    transition -- progress is tracked as one continuous arc length.
"""

import math
import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

from .guidance import GuidanceOutput


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MPCConfig:
    dt: float = 0.05          # control tick, matches the ~20 Hz fly_gates loop
    horizon: int = 16         # ~0.8 s lookahead

    # State tracking weights (position, velocity). q_vel was 0.5 -- so cheap
    # relative to q_pos that the solve preferred to smooth a velocity error
    # out over the whole horizon rather than spend control effort canceling
    # it quickly (observed: a multi-m/s residual climb taking 5+ seconds and
    # >20m to arrest). Raising it makes the tracker treat "moving at the
    # wrong speed" as costly as "being in the wrong place" rather than a
    # distant second priority.
    q_pos: float = 6.0
    q_vel: float = 2.5
    q_pos_terminal: float = 8.0
    q_vel_terminal: float = 5.0

    # Control effort weight (acceleration). Was 0.08 -- too cheap relative
    # to q_pos, which let the solve happily slam into the velocity clamps
    # (observed: vertical channel pinned at max_vert_speed for >1s while
    # overshooting the leg's altitude target by ~10m) instead of trading
    # off a slower, smoother approach.
    r_acc: float = 0.2

    # Output limits -- enforced by clamping, not by the QP. max_accel is
    # split into horizontal/vertical clips (thrust and tilt are separate
    # actuators -- the vehicle doesn't have one shared acceleration budget
    # across axes the way a single combined-vector clip implies). A single
    # combined clip was confirmed (U0_RAW diagnostic, 2026-06-25) to starve
    # the vertical channel for seconds at the start of every race leg: the
    # 0->8m/s horizontal speed-up dominates the vector norm (-19 to -20 m/s2
    # raw) and forces the same scale-down onto the vertical component even
    # while it's legitimately growing (the solve correctly wanted more and
    # more vertical correction, +3.9 up to +66.7, while clipping squeezed
    # the applied value down to near nothing) -- altitude drifted
    # uncontrolled for 3-5s every leg start as a direct result.
    max_horiz_speed: float = 14.0
    max_vert_speed: float = 5.0
    max_accel_horiz: float = 8.0
    max_accel_vert: float = 8.0

    # How many ticks ahead (of the MPC's own predicted trajectory) to pull
    # the position target from, so it's a real lead point consistent with
    # the commanded velocity rather than "current position plus an
    # infinitesimal step". 6 steps * dt=0.05 = 0.3s lookahead.
    position_lookahead_steps: int = 6

    # Course geometry
    pass_distance: float = 1.5
    min_speed: float = 3.0

    # Obstacle avoidance (soft bias on the reference, not a hard constraint)
    obstacle_radius: float = 3.0
    obstacle_gain: float = 6.0


# ---------------------------------------------------------------------------
# Global course polyline (arc-length parameterized)
# ---------------------------------------------------------------------------

class GatePath:
    """
    Single polyline from the start point through every gate, in arc-length
    order. Sampling a continuous arc-length coordinate (rather than
    rebuilding a fresh line segment at every gate transition) is what gives
    the MPC reference trajectory continuity across the whole course.
    """

    def __init__(self, start_ned, gates_sorted, cruise_speed: float, min_speed: float):
        pts = [np.asarray(start_ned, dtype=float)]
        for g in gates_sorted:
            pts.append(np.array([g['north'], g['east'], g['down']], dtype=float))
        self.points = np.array(pts)                       # (M+1, 3)

        seg_vec = np.diff(self.points, axis=0)             # (M, 3)
        self.seg_len = np.linalg.norm(seg_vec, axis=1)
        self.seg_len = np.maximum(self.seg_len, 1e-6)
        self.seg_dir = seg_vec / self.seg_len[:, None]
        self.cum_len = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.total_len = float(self.cum_len[-1])

        # Per-segment cruise speed: slow down ahead of sharp turns using the
        # angle between consecutive segments (same heuristic the old speed
        # scheduler used per-tick; here it's precomputed once for the whole
        # course since the polyline itself doesn't change).
        self.seg_speed = np.full(len(self.seg_len), cruise_speed)
        for i in range(1, len(self.seg_dir)):
            dot = float(np.clip(np.dot(self.seg_dir[i - 1], self.seg_dir[i]), -1.0, 1.0))
            angle_deg = math.degrees(math.acos(dot))
            if angle_deg < 20:
                self.seg_speed[i] = cruise_speed
            elif angle_deg < 60:
                self.seg_speed[i] = max(min_speed, cruise_speed * 0.65)
            else:
                self.seg_speed[i] = max(min_speed, cruise_speed * 0.4)

    def project(self, pos):
        """
        Arc length of the closest point on the polyline to `pos`, plus
        diagnostics: lateral_dist is the perpendicular distance from pos to
        that closest point (how far off the line we are), and raw_t is the
        *unclamped* scalar projection onto the winning segment -- project()
        itself clips this to [0, length] to pick the closest point, which
        silently freezes progress at a segment's start whenever the drone
        has overshot off the front of it (raw_t < 0, e.g. climbing/descending
        away from the line faster than it advances along it). Comparing
        raw_t to the clamped value exposes that case instead of hiding it.
        """
        pos = np.asarray(pos, dtype=float)
        best_s, best_d2, best_raw_t, best_idx = 0.0, np.inf, 0.0, 0
        for i in range(len(self.seg_dir)):
            p0, d, length = self.points[i], self.seg_dir[i], self.seg_len[i]
            raw_t = float(np.dot(pos - p0, d))
            t = float(np.clip(raw_t, 0.0, length))
            closest = p0 + d * t
            d2 = float(np.sum((pos - closest) ** 2))
            if d2 < best_d2:
                best_d2 = d2
                best_s = self.cum_len[i] + t
                best_raw_t = raw_t
                best_idx = i
        lateral_dist = math.sqrt(best_d2)
        return best_s, lateral_dist, best_raw_t, best_idx

    def sample(self, s: float):
        """Position, tangent direction, and scheduled speed at arc length s."""
        s = min(max(s, 0.0), self.total_len)
        idx = int(np.clip(np.searchsorted(self.cum_len, s, side='right') - 1,
                           0, len(self.seg_dir) - 1))
        t = s - self.cum_len[idx]
        pos = self.points[idx] + self.seg_dir[idx] * t
        return pos, self.seg_dir[idx], float(self.seg_speed[idx])


# ---------------------------------------------------------------------------
# Closed-form finite-horizon LQ tracker (condensed/batch MPC)
# ---------------------------------------------------------------------------

class LinearMPC:
    """
    Double-integrator translational model:
        state x = [n, e, d, vn, ve, vd]
        control u = [an, ae, ad]
    Solved once in condensed form (X = Sx @ x0 + Su @ U) so each control
    tick is just a couple of matrix-vector products -- no QP solver
    dependency, no per-tick matrix factorization.
    """

    STATE_DIM = 6
    CTRL_DIM = 3

    def __init__(self, cfg: MPCConfig):
        self.cfg = cfg
        dt, N = cfg.dt, cfg.horizon

        A = np.eye(6)
        A[0, 3] = A[1, 4] = A[2, 5] = dt
        B = np.zeros((6, 3))
        B[0, 0] = B[1, 1] = B[2, 2] = 0.5 * dt * dt
        B[3, 0] = B[4, 1] = B[5, 2] = dt
        self.A, self.B = A, B

        A_pows = [np.eye(6)]
        for _ in range(N):
            A_pows.append(A_pows[-1] @ A)

        Sx = np.vstack(A_pows[1:N + 1])                   # (6N, 6)
        Su = np.zeros((6 * N, 3 * N))
        for k in range(N):
            for j in range(k + 1):
                Su[k * 6:(k + 1) * 6, j * 3:(j + 1) * 3] = A_pows[k - j] @ B

        q_diag = np.array([cfg.q_pos] * 3 + [cfg.q_vel] * 3)
        q_term = np.array([cfg.q_pos_terminal] * 3 + [cfg.q_vel_terminal] * 3)
        Qbar_diag = np.tile(q_diag, N)
        Qbar_diag[-6:] = q_term
        Qbar = np.diag(Qbar_diag)
        Rbar = np.diag(np.full(3 * N, cfg.r_acc))

        gram = Su.T @ Qbar @ Su + Rbar
        # M maps the open-loop error (Sx@x0 - x_ref) directly to the optimal
        # control sequence: U* = -M @ (Sx@x0 - x_ref).
        self._M = np.linalg.solve(gram, Su.T @ Qbar)       # (3N, 6N)
        self.Sx, self.Su = Sx, Su
        self.N = N

    def solve(self, x0: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
        """
        x0    : (6,)   current state
        x_ref : (N, 6) reference state for each of the next N ticks
        Returns the full optimal control sequence U, shape (3*N,) --
        U[:3] is the immediate commanded acceleration; the rest lets the
        caller read the MPC's own predicted trajectory (via predict()) for
        a dynamically-consistent lookahead position, not just one step.
        """
        e0 = self.Sx @ x0 - x_ref.reshape(-1)
        return -self._M @ e0

    def predict(self, x0: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Predicted state trajectory for control sequence U, shape (N, 6)."""
        return (self.Sx @ x0 + self.Su @ U).reshape(self.N, 6)


# ---------------------------------------------------------------------------
# Guidance facade -- same call shape as GuidanceAlgorithm.compute()
# ---------------------------------------------------------------------------

class MPCGuidance:
    """
    Drop-in replacement for GuidanceAlgorithm: compute(state, gates, obstacles)
    -> GuidanceOutput. Builds the global GatePath lazily on the first call
    (it needs the drone's start position), then each tick:
      1. projects current position onto the path to get arc-length progress
      2. samples a reference trajectory N steps ahead along the path
      3. solves the closed-form LQ tracker for the next acceleration
      4. clamps to velocity/accel limits and returns position+velocity
         for the *next* tick (dynamically consistent with each other,
         since both come from propagating the same control through the
         same double-integrator model).
    """

    def __init__(self, course_map, config: MPCConfig = None):
        self.course_map = course_map
        self.cfg = config or MPCConfig()
        self.mpc = LinearMPC(self.cfg)

        self.path: GatePath = None
        self.progress_s: float = 0.0
        self.cruise_speed = (
            course_map.waypoints[0].speed_mps if course_map.waypoints else 8.0
        )
        self._last_ref_pos = None
        self._last_ref_vel = None
        self._last_lateral_dist = 0.0
        self._last_raw_t = 0.0
        self._last_proj_idx = 0
        self._last_u0_raw = np.zeros(3)
        self._last_accel_mag = 0.0
        self._last_horiz_clipped = False
        self._last_vert_clipped = False

    # ------------------------------------------------------------------

    def _build_path(self, start_ned):
        gates_sorted_pos = [
            {"north": wp.position_ned[0], "east": wp.position_ned[1], "down": wp.position_ned[2]}
            for wp in self.course_map.waypoints
        ]
        self.path = GatePath(start_ned, gates_sorted_pos, self.cruise_speed, self.cfg.min_speed)
        self.progress_s = 0.0

        points_str = " -> ".join(f"({p[0]:.1f},{p[1]:.1f},{p[2]:.1f})" for p in self.path.points)
        logger.info(
            f"MPCGuidance: built path with {len(self.path.seg_dir)} legs, "
            f"total_len={self.path.total_len:.1f}m, cruise_speed={self.cruise_speed:.1f}m/s\n"
            f"  points: {points_str}"
        )

        if self.path.seg_dir.shape[0] > 0:
            first_leg_len = self.path.seg_len[0]
            if first_leg_len > 500.0:
                logger.error(
                    f"MPCGuidance: first leg is {first_leg_len:.0f}m long (start "
                    f"{tuple(round(x, 1) for x in start_ned)} -> first waypoint "
                    f"{tuple(round(x, 1) for x in self.path.points[1])}). This almost "
                    f"certainly means gate telemetry and drone position telemetry are "
                    f"in different coordinate frames this session (e.g. gates reported "
                    f"in world coordinates while drone position is zeroed at spawn) -- "
                    f"not a real course. Try a full simulator restart, not just sim_reset."
                )

    def status_string(self) -> str:
        """Human-readable progress for logging: which leg, distance to the
        next waypoint along the path, and overall course progress."""
        if self.path is None:
            return "path not built yet"
        n_legs = len(self.path.seg_dir)
        idx = int(np.clip(
            np.searchsorted(self.path.cum_len, self.progress_s, side='right') - 1,
            0, n_legs - 1,
        ))
        dist_to_next_wp = max(0.0, self.path.cum_len[idx + 1] - self.progress_s)
        next_wp = self.path.points[idx + 1]
        ref_str = ""
        if self._last_ref_pos is not None:
            rp, rv = self._last_ref_pos, self._last_ref_vel
            ref_str = (
                f" on_line_now=({rp[0]:.1f},{rp[1]:.1f},{rp[2]:.1f}) "
                f"line_vel=({rv[0]:.1f},{rv[1]:.1f},{rv[2]:.1f})"
            )
        # raw_t < 0 means we've overshot off the *front* of the current
        # segment (e.g. climbing/descending away from the line faster than
        # we advance along it) -- project() clamps that to 0, which is why
        # course_progress can sit frozen even while the drone is clearly
        # moving. lateral_dist is the straight-line distance off the path
        # itself, independent of arc-length bookkeeping -- the direct
        # "how far from where we want to be" number to compare against POS.
        dev_str = (
            f" LATERAL_DEV={self._last_lateral_dist:.1f}m"
            f" RAW_T={self._last_raw_t:+.1f}m(leg{self._last_proj_idx + 1})"
        )
        # u0_raw is the LQ solve's actual chosen acceleration *before*
        # clipping -- compare its sign against line_vel above to tell a
        # real wrong-direction solve apart from a clipping artifact.
        # Horizontal/vertical clip flags are reported separately since
        # they're now applied independently (see max_accel_horiz/vert).
        u0 = self._last_u0_raw
        accel_str = (
            f" U0_RAW=({u0[0]:+.1f},{u0[1]:+.1f},{u0[2]:+.1f}) "
            f"ACCEL_MAG={self._last_accel_mag:.1f}m/s2"
            f"{'(H-CLIP)' if self._last_horiz_clipped else ''}"
            f"{'(V-CLIP)' if self._last_vert_clipped else ''}"
        )
        return (
            f"leg={idx + 1}/{n_legs} dist_to_next_wp={dist_to_next_wp:.1f}m "
            f"next_wp=({next_wp[0]:.1f},{next_wp[1]:.1f},{next_wp[2]:.1f}) "
            f"course_progress={self.progress_s:.1f}/{self.path.total_len:.1f}m"
            f"{ref_str}{dev_str}{accel_str}"
        )

    def _reference_horizon(self) -> np.ndarray:
        cfg = self.cfg
        s_k = self.progress_s
        x_ref = np.zeros((cfg.horizon, 6))
        for k in range(cfg.horizon):
            pos_k, dir_k, speed_k = self.path.sample(s_k)
            x_ref[k, 0:3] = pos_k
            x_ref[k, 3:6] = dir_k * speed_k
            s_k += speed_k * cfg.dt
        return x_ref

    def _obstacle_bias(self, x_ref: np.ndarray, state, obstacles) -> np.ndarray:
        if not obstacles:
            return x_ref
        cfg = self.cfg
        R = state.body_to_ned_R()
        own_pos = state.position_ned()
        bias = np.zeros(3)
        for obs in obstacles:
            obs_pos_ned = own_pos + R @ np.asarray(obs.position_body)
            vec = own_pos - obs_pos_ned
            dist = float(np.linalg.norm(vec))
            if 0.1 < dist < cfg.obstacle_radius:
                bias += (vec / dist) * (cfg.obstacle_gain / (dist ** 2))
        if np.linalg.norm(bias) < 1e-9:
            return x_ref
        # Decay the bias further into the horizon -- the obstacle's
        # position is only known *now*, not propagated.
        decay = np.exp(-np.arange(cfg.horizon) / max(cfg.horizon / 3, 1))
        x_ref[:, 0:3] += bias[None, :] * decay[:, None]
        return x_ref

    # ------------------------------------------------------------------

    def compute(self, state, gates, obstacles) -> GuidanceOutput:
        cfg = self.cfg
        position = state.position_ned()

        if self.path is None:
            self._build_path(position)
            self._last_leg_idx = 0

        raw_s, lateral_dist, raw_t, proj_idx = self.path.project(position)
        self._last_lateral_dist = lateral_dist
        self._last_raw_t = raw_t
        self._last_proj_idx = proj_idx
        # Monotonic progress: never snap backwards onto an earlier part of
        # the polyline (can happen near sharp turns where two segments pass
        # close to each other) -- only allow tiny jitter, not a real regression.
        self.progress_s = max(raw_s, self.progress_s - 0.25)

        leg_idx = int(np.clip(
            np.searchsorted(self.path.cum_len, self.progress_s, side='right') - 1,
            0, len(self.path.seg_dir) - 1,
        ))
        if leg_idx > self._last_leg_idx:
            wp = self.path.points[leg_idx]
            logger.info(
                f"MPCGuidance: waypoint {leg_idx} reached at "
                f"({wp[0]:.1f},{wp[1]:.1f},{wp[2]:.1f}) -- advancing to leg {leg_idx + 1}/{len(self.path.seg_dir)}"
            )
            self._last_leg_idx = leg_idx

        if self.progress_s >= self.path.total_len - cfg.pass_distance:
            return GuidanceOutput(
                target_position=position,
                target_velocity=np.zeros(3),
                target_yaw=state.yaw,
                target_altitude=position[2],
                course_complete=True,
            )

        x_ref = self._reference_horizon()
        x_ref = self._obstacle_bias(x_ref, state, obstacles)

        # The point on the line being tracked *right now* (first horizon
        # step) -- exposed via status_string() so logging can show it's a
        # real, moving point on the built path, not a static/frozen target.
        self._last_ref_pos = x_ref[0, 0:3].copy()
        self._last_ref_vel = x_ref[0, 3:6].copy()

        x0 = np.array([state.north, state.east, state.down,
                       state.vn, state.ve, state.vd])

        U = self.mpc.solve(x0, x_ref)
        u0_raw = U[:3].copy()

        # Diagnostics: the *unclipped* optimal acceleration the LQ solve
        # actually wants. If u0_raw's vertical component disagrees in sign
        # with the mild feedforward in line_vel/_last_ref_vel, that's the
        # solve itself choosing the wrong direction -- not a downstream
        # clipping or sign-convention artifact.
        self._last_u0_raw = u0_raw.copy()
        self._last_accel_mag = float(np.linalg.norm(u0_raw))

        # Clip horizontal and vertical acceleration independently --
        # thrust and tilt are separate actuators, not one shared budget.
        # A single combined-vector clip here previously meant a large
        # horizontal speed-up (e.g. ramping to cruise at the start of a
        # leg, regularly -19..-20 m/s2 raw) consumed nearly the entire
        # clip allowance and squeezed the vertical component down to near
        # zero even while it was legitimately large and growing, leaving
        # altitude to drift uncontrolled for seconds at every leg start.
        u0 = u0_raw.copy()
        horiz_mag = float(np.linalg.norm(u0[:2]))
        self._last_horiz_clipped = horiz_mag > cfg.max_accel_horiz
        if self._last_horiz_clipped:
            u0[:2] *= cfg.max_accel_horiz / horiz_mag
        self._last_vert_clipped = abs(u0[2]) > cfg.max_accel_vert
        if self._last_vert_clipped:
            u0[2] = math.copysign(cfg.max_accel_vert, u0[2])

        x1 = self.mpc.A @ x0 + self.mpc.B @ u0

        horiz_speed = float(np.linalg.norm(x1[3:5]))
        if horiz_speed > cfg.max_horiz_speed:
            x1[3:5] *= cfg.max_horiz_speed / horiz_speed
        x1[5] = float(np.clip(x1[5], -cfg.max_vert_speed, cfg.max_vert_speed))

        target_velocity = x1[3:6]

        # The sim's autopilot needs position+velocity sent together to
        # respond at all (velocity-only produced zero motion), but pairing
        # a *huge* velocity feedforward with a position that's only an
        # infinitesimal step from "where you already are" is an internally
        # inconsistent reference for a position-tracking controller. Use a
        # real lead point a few ticks into the MPC's own predicted
        # trajectory instead -- "where the plan says to be shortly",
        # consistent with the commanded velocity, the way a pure-pursuit
        # lookahead point is.
        lookahead_idx = min(cfg.position_lookahead_steps, cfg.horizon) - 1
        x_pred = self.mpc.predict(x0, U)
        target_position = x_pred[lookahead_idx, 0:3]

        # Heading from the path's own tangent direction at the current
        # progress point (x_ref[0, 3:6], already computed above as
        # _last_ref_vel), not the one-step MPC velocity command. The
        # latter goes near-zero -- and atan2 of a near-zero vector is
        # noise-dominated -- whenever the solve is spending its effort on
        # a vertical correction instead of horizontal motion, which was
        # causing target_yaw to flip across nearly the full +-180 deg
        # range tick to tick. The path tangent is leg-constant and stable
        # by comparison.
        ref_vel = x_ref[0, 3:6]
        target_yaw = (
            math.atan2(ref_vel[1], ref_vel[0])
            if np.linalg.norm(ref_vel[:2]) > 0.3
            else state.yaw
        )

        return GuidanceOutput(
            target_position=target_position,
            target_velocity=target_velocity,
            target_yaw=target_yaw,
            target_altitude=target_position[2],
            course_complete=False,
        )

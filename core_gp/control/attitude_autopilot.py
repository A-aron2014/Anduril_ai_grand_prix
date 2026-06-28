"""
Attitude Autopilot
------------------
Closes the position/velocity -> attitude/thrust loop ourselves and drives
SET_ATTITUDE_TARGET (rate mode), instead of handing position/velocity
targets to the sim's own controller via SET_POSITION_TARGET_LOCAL_NED.

This exists because SET_POSITION_TARGET_LOCAL_NED is confirmed broken in
this simulator: streaming it faster than ~1 message/1-2s causes an
unbounded climb (proven across 7 isolated tests -- content-independent,
cadence-triggered, never reaches a terminal velocity) regardless of what
position/velocity is actually commanded. SET_ATTITUDE_TARGET, streamed at
the same rate, behaves like a normal physical system (thrust above hover
accelerates the vehicle up to a bounded terminal velocity where drag
balances thrust, then holds). See attitude_target_test.py /
rate_threshold_test.py for the evidence trail.

Cascade (outer to inner), all gains empirically tunable -- there is no
physical units conversion (thrust-to-newtons, mass, etc. are unknown), so
every stage's output is in the units the next stage expects directly:
  1. position error (NED)        -> added to guidance's target_velocity
  2. velocity error (NED)        -> desired body-frame tilt (pitch/roll,
                                     rad) for horizontal, thrust delta for
                                     vertical
  3. tilt angle error (body)     -> desired body rate (rad/s)
  4. yaw error                   -> desired yaw rate (rad/s)

GAINS BELOW ARE UNVERIFIED STARTING GUESSES -- this has not been flight
tested in the sim yet. Expect to retune kp_tilt/kp_rate/hover_thrust
against real telemetry.
"""

import math
import time
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from core.state_estimator import VehicleState
from guidance.guidance import GuidanceOutput
from control.autopilot import PIDController

logger = logging.getLogger(__name__)


@dataclass
class AttitudeAutopilotConfig:
    # Position -> velocity correction (added on top of guidance's own
    # target_velocity). NOT just a small trim: confirmed on a real run
    # (2026-06-27, after fixing target_position to come from the path
    # reference x_ref instead of the MPC's own forward-simulated x_pred --
    # see mpc_guidance.py compute()) that guidance.target_velocity is
    # itself nearly indistinguishable from the vehicle's *actual* velocity
    # whenever the LQ solve's raw correction is clipped (max_accel_horiz),
    # which is true on almost every tick of a cruise leg. E.g. one sample:
    # target_velocity_e=2.27 vs ACTUAL_VEL_e=2.2 -- a 0.07 m/s difference,
    # while the real cross-track position error (err_e) was -1.9m. At the
    # old kp_horiz=0.3 that error only contributed -0.57 m/s to vel_cmd,
    # nowhere near enough to drive the inner tilt loop hard within a ~3s
    # leg (roll only reached -0.29 of a -0.5 limit by the gate). Since
    # err_e is now a real, uncorrupted cross-track measurement (not true
    # before the x_ref fix), it's safe to lean on it much harder here --
    # raised so the same -1.9m error produces a -1.5 m/s vel_cmd
    # contribution instead.
    pos_kp_horiz: float = 0.8
    pos_kp_vert:  float = 0.5

    # Velocity error -> tilt angle (horizontal) / thrust delta (vertical).
    # No D-term: it amplifies telemetry jitter into the next loop down
    # without any filtering, which was a likely contributor to the
    # chattering seen in the first (pre-sign-fix) hold_test.py run.
    # tilt_ki was 0.0 -- a P-only loop can't cancel a steady-state velocity
    # bias at all, which is exactly what was observed: HOLD's horizontal
    # speed crept monotonically from 0.7 to 1.2+ m/s over a 3s window and
    # never converged. 0.03 wasn't enough either -- speed still hit 4+ m/s
    # within the 3s settle window (2026-06-25 runs), consistent with a
    # real, fairly large steady-state bias (cf. the "~0.3rad spawn-tilt
    # offset" noted below -- double max_tilt) that a weak integral takes
    # too long to walk out. Raised further; if HOLD still won't settle
    # after this, max_tilt=0.15 -- not the gain -- is the likely ceiling,
    # since the output saturates there well before the bias is cancelled.
    #
    # Confirmed (2026-06-25 race log): max_tilt=0.15 IS the ceiling, not
    # just for HOLD but for RACE too. At hover_thrust=0.27, the max
    # horizontal thrust component available at 0.15rad saturation is only
    # thrust*sin(0.15) =~ 0.04 (=~1.4 m/s2 of forward accel) -- nowhere
    # near enough to reach or hold an 8 m/s cruise reference against drag.
    # U0_RAW stayed correctly signed and grew monotonically (-19.9 to
    # -32.5+ m/s2) the entire leg while pitch correctly drove nose-down,
    # yet ACTUAL_VEL decayed through zero and reversed -- the commanded
    # direction was right the whole time, the vehicle just never had
    # enough lean authority to act on it once residual momentum bled off.
    # Raised to a value in line with typical racing-drone lean angles.
    #
    # Tried raising further to 0.75 (2026-06-27) on the theory that roll
    # pinning at this limit while ACTUAL_VEL's east component kept climbing
    # meant the controller just needed more authority. Disproved by the
    # next run: with more headroom, pitch climbed even higher (0.50 ->
    # 0.65+) and the east runaway got WORSE, not better -- LATERAL_DEV hit
    # 7.6m before even finishing leg 1 (vs 6.4m at the old limit), and the
    # post-leg HOLD settle speed went from 13.1 to 17.3 m/s. More tilt
    # authority made the outcome worse, which rules out "authority
    # ceiling" -- reverted to 0.5. The pattern (east drift growing in
    # lockstep with pitch, not with roll) now points at pitch itself (or
    # the forward speed it produces) coupling into lateral motion in this
    # sim, the same family of bug as the documented pitch->vertical
    # coupling (see pitch_vertical_coupling_test.py) but on the lateral
    # axis instead. Needs the same kind of isolated test before touching
    # this gain again.
    tilt_kp: float = 0.02
    tilt_ki: float = 0.07
    tilt_kd: float = 0.0
    max_tilt: float = 0.5           # rad (~28.6 deg)

    # Right (lateral) channel gets its own, stronger gains. Four
    # consecutive race runs (2026-06-27) on this course -- all flown at
    # ~180deg heading -- showed the same repeatable eastward drift: roll
    # stayed near zero while pitch climbed through the leg, yet east
    # position kept creeping the same direction every time, ending up
    # 1.5-2m off-line by the time it reached a 2.72m gate (more than half
    # the gate's clearance). The shared forward/right gains weren't
    # cancelling it inside the ~20m/3s leg before the gate arrived.
    # Mitigation, not a confirmed root-cause fix -- the actual source
    # (sim-side pitch/thrust coupling, a residual trim, or something
    # else) was never isolated via a dedicated test. A stronger, faster
    # lateral loop fights the symptom regardless of the exact mechanism;
    # forward's gains are untouched since that channel tracks cleanly.
    tilt_kp_right: float = 0.05
    tilt_ki_right: float = 0.18

    # Were 0.03/0.02 -- too gentle to kill a multi-m/s vertical velocity
    # error quickly: observed taking 5+ seconds and >20m of altitude
    # excursion to walk thrust down and arrest a ~3 m/s residual climb
    # carried over from a hold phase. kp=0.12 reaches max_thrust_delta on
    # an error of ~2.5 m/s in a single tick instead of crawling toward it.
    thrust_kp: float = 0.12
    thrust_ki: float = 0.05         # self-trims hover_thrust if it's off
    thrust_kd: float = 0.0
    # Was 0.4 -- a starting guess that turned out ~0.13 too high. Every
    # HOLD phase across many logs settles vertical velocity near zero with
    # thrust around 0.26-0.28, never near 0.4. Because a fresh
    # AttitudeAutopilot (integral=0) is constructed at the start of every
    # flight phase, and the velocity error right at a phase boundary is
    # usually near zero (masking the bias from the PID), thrust starts
    # AT hover_thrust with no correction applied yet -- so a too-high
    # default meant every single phase transition (TAKEOFF->HOLD->RACE,
    # or into this test) started with ~0.13 of unwanted excess lift and
    # climbed for several seconds before the slow integral term walked it
    # back down. This was misread as a pitch-coupling effect (pitch is
    # also elevated right after a phase transition, but it's correlation,
    # not cause -- a flat leg with zero pitch authority would show this
    # same thrust-too-high climb at the start of every phase too, since
    # the discrepancy is in this constant, not in any attitude coupling).
    hover_thrust: float = 0.27
    max_thrust_delta: float = 0.3
    min_thrust: float = 0.05
    max_thrust: float = 0.9

    # Tilt angle -> body rate (inner loop). Raising this (2.0->4.0 kp,
    # 0.6->2.0 max rate) to correct the ~0.3rad spawn-tilt offset faster
    # made the divergence WORSE, not better (2026-06-21 runs) -- reverted.
    # The instability isn't loop-speed, it's something else; investigating
    # with per-tick attitude logging before touching these again.
    attitude_kp: float = 2.0
    max_body_rate: float = 0.6      # rad/s

    # Yaw
    yaw_kp: float = 0.5
    yaw_ki: float = 0.0
    yaw_kd: float = 0.0
    max_yaw_rate: float = 0.5       # rad/s

    max_horiz_vel: float = 14.0
    max_vert_vel: float = 5.0


class AttitudeAutopilot:
    """
    compute(state, guidance) -> (roll_rate, pitch_rate, yaw_rate, thrust)
    for FlightController._send_attitude_target. One instance per flight
    phase (fresh PID state) -- create a new one rather than reusing across
    takeoff/racing the way MPCGuidance instances are already handled.
    """

    def __init__(self, config: Optional[AttitudeAutopilotConfig] = None):
        self.cfg = config or AttitudeAutopilotConfig()
        c = self.cfg

        # integral_limit bounds the *accumulated* integral state, not just
        # the PID's output -- PIDController's own default (100.0) is a
        # generic fallback that was never scaled to any of these channels'
        # actual output ranges. For tilt_right specifically (output capped
        # at +-max_tilt=0.5, ki=0.18) the output saturates once the
        # integral alone reaches ~0.5/0.18=2.8, but nothing stopped it
        # climbing toward 100 from there -- confirmed on a real run
        # (2026-06-27): roll pinned at -0.50 for 20+ consecutive ticks
        # (over a full second) while U0_RAW's east component blew out to
        # -199 m/s2, then a second run diverged the opposite direction
        # once the wound-up integral finally unwound past zero. Classic
        # windup: raising tilt_ki_right without rescaling this limit to
        # match is what caused both. Limit is ~1.5x the integral value
        # that alone saturates that channel's output -- enough headroom
        # for the I-term to do its job, not enough to wind up for seconds
        # past saturation before it can respond to a sign reversal.
        self._pid_forward = PIDController(c.tilt_kp, c.tilt_ki, c.tilt_kd,
                                           -c.max_tilt, c.max_tilt,
                                           integral_limit=1.5 * c.max_tilt / c.tilt_ki)
        self._pid_right = PIDController(c.tilt_kp_right, c.tilt_ki_right, c.tilt_kd,
                                         -c.max_tilt, c.max_tilt,
                                         integral_limit=1.5 * c.max_tilt / c.tilt_ki_right)
        self._pid_vert = PIDController(c.thrust_kp, c.thrust_ki, c.thrust_kd,
                                        -c.max_thrust_delta, c.max_thrust_delta,
                                        integral_limit=1.5 * c.max_thrust_delta / c.thrust_ki)
        self._pid_yaw = PIDController(c.yaw_kp, c.yaw_ki, c.yaw_kd,
                                       -c.max_yaw_rate, c.max_yaw_rate)

    def reset(self):
        for pid in (self._pid_forward, self._pid_right, self._pid_vert, self._pid_yaw):
            pid.reset()

    def compute(self, state: VehicleState, guidance: GuidanceOutput):
        cfg = self.cfg
        now = time.monotonic()
        pos = state.position_ned()
        vel = state.velocity_ned()

        # --- Position error folded into guidance's own velocity feedforward ---
        err_n = guidance.target_position[0] - pos[0]
        err_e = guidance.target_position[1] - pos[1]
        err_d = guidance.target_position[2] - pos[2]

        vel_cmd_n = guidance.target_velocity[0] + cfg.pos_kp_horiz * err_n
        vel_cmd_e = guidance.target_velocity[1] + cfg.pos_kp_horiz * err_e
        vel_cmd_d = guidance.target_velocity[2] + cfg.pos_kp_vert * err_d

        horiz = math.hypot(vel_cmd_n, vel_cmd_e)
        if horiz > cfg.max_horiz_vel:
            scale = cfg.max_horiz_vel / horiz
            vel_cmd_n *= scale
            vel_cmd_e *= scale
        vel_cmd_d = float(np.clip(vel_cmd_d, -cfg.max_vert_vel, cfg.max_vert_vel))

        vel_err_n = vel_cmd_n - vel[0]
        vel_err_e = vel_cmd_e - vel[1]
        vel_err_d = vel_cmd_d - vel[2]

        # --- NED horizontal velocity error -> body-frame forward/right ---
        yaw = state.yaw
        cy, sy = math.cos(yaw), math.sin(yaw)
        forward_err = vel_err_n * cy + vel_err_e * sy
        right_err = -vel_err_n * sy + vel_err_e * cy

        # Right error: positive roll (right-side-down) tilts thrust LEFT,
        # so accelerating rightward needs NEGATIVE roll. Missing this
        # negation was confirmed (2026-06-21 hold_test.py run) to cause
        # positive feedback: the vehicle drifted essentially monotonically
        # to +103m east over 10s while roll_rate sat saturated at -2.0,
        # i.e. the correction was being applied in the wrong direction.
        #
        # Pitch previously carried the same negation "to match
        # fly_forward()'s FORWARD_PITCH convention" -- an old script,
        # never independently re-verified against this attitude-rate
        # cascade the way roll's negation was. Confirmed wrong
        # (2026-06-25 race log): pitch correctly tracked an increasingly
        # nose-down pitch_des the whole leg (the rest of the cascade was
        # doing exactly what it intended), yet ACTUAL_VEL's north
        # component decayed toward zero in lockstep with |pitch| growing
        # instead of accelerating toward the gate, while U0_RAW stayed
        # strongly, correctly signed throughout -- the vehicle was
        # decelerating in proportion to its own forward-intended tilt,
        # not just under-accelerating against drag (which would plateau,
        # not decay through zero from a near-stop). Removed the negation.
        pitch_des = self._pid_forward.update(forward_err, now)
        roll_des = -self._pid_right.update(right_err, now)

        # --- Vertical velocity error -> thrust delta around hover ---
        thrust_delta = -self._pid_vert.update(vel_err_d, now)
        thrust = float(np.clip(cfg.hover_thrust + thrust_delta,
                                cfg.min_thrust, cfg.max_thrust))

        # --- Tilt angle error -> body rate (inner attitude loop) ---
        # Wrapped to [-pi, pi] same as yaw below: an unwrapped error here
        # would (and did) flip sign incorrectly once attitude error exceeded
        # pi, driving the vehicle to keep rotating the same way into a tumble
        # instead of taking the short way back.
        #
        # pitch_rate is negated, roll_rate is NOT -- confirmed via two
        # separate hold_test.py telemetry runs (2026-06-21) that this sim's
        # SET_ATTITUDE_TARGET body rate sign is inverted from the ATTITUDE
        # angle sign for pitch (sending -0.6 made pitch climb, not fall) but
        # NOT for roll (sending +0.6 made roll climb in the same direction
        # as commanded, i.e. already correct). Applying the same negation to
        # both axes was wrong: it fixed pitch but turned roll's negative
        # feedback into positive feedback, producing the 180-degree roll
        # divergence seen when both were negated. Don't assume the two axes
        # share a sign convention in this sim -- they don't.
        roll_rate = float(np.clip(cfg.attitude_kp * self._wrap_pi(roll_des - state.roll),
                                   -cfg.max_body_rate, cfg.max_body_rate))
        pitch_rate = -float(np.clip(cfg.attitude_kp * self._wrap_pi(pitch_des - state.pitch),
                                     -cfg.max_body_rate, cfg.max_body_rate))

        # --- Yaw ---
        # Negated for the same reason as pitch: hold_test.py telemetry
        # (2026-06-21) showed yaw drifting monotonically ~2 rad over a 10s
        # hold despite a fixed target_yaw, with commanded yaw_rate positive
        # the entire time yaw was decreasing -- the same inverted-sign
        # signature pitch had, never reversing on its own.
        yaw_err = self._wrap_pi(guidance.target_yaw - yaw)
        yaw_rate = -self._pid_yaw.update(yaw_err, now)

        return roll_rate, pitch_rate, yaw_rate, thrust

    @staticmethod
    def _wrap_pi(angle: float) -> float:
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

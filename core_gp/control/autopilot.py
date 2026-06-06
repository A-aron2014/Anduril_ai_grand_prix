"""
Autopilot
Translates GuidanceOutput into MAVLink control commands.
Uses PID feedback on position/velocity error in NED frame,
commanding SET_POSITION_TARGET_LOCAL_NED at up to 100 Hz.

The simulator's on-board stabiliser handles attitude/throttle internally,
so we operate at the position/velocity level (outer-loop only).
"""

import math
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from core.state_estimator import VehicleState
from core_gp.guidance.guidance import GuidanceOutput
from core_gp.comms.mavlink_bridge import ControlCommand

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PID Controller (single axis)
# ---------------------------------------------------------------------------

class PIDController:
    """Generic 1-D PID with anti-windup and output clamping."""

    def __init__(self, kp: float, ki: float, kd: float,
                 output_min: float = -np.inf,
                 output_max: float = np.inf,
                 integral_limit: float = 100.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        self.integral_limit = integral_limit
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time: Optional[float] = None

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None

    def update(self, error: float, now: Optional[float] = None) -> float:
        now = now or time.monotonic()
        dt = (now - self._prev_time) if self._prev_time else 0.02
        dt = max(1e-4, min(dt, 0.5))

        self._integral = float(np.clip(
            self._integral + error * dt,
            -self.integral_limit, self.integral_limit
        ))
        derivative = (error - self._prev_error) / dt

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        output = float(np.clip(output, self.output_min, self.output_max))

        self._prev_error = error
        self._prev_time = now
        return output


# ---------------------------------------------------------------------------
# Autopilot
# ---------------------------------------------------------------------------

@dataclass
class AutopilotConfig:
    # Position-loop gains
    pos_kp:  float = 1.2
    pos_ki:  float = 0.05
    pos_kd:  float = 0.3

    # Velocity feedforward blend
    vel_ff_weight: float = 0.8     # 0 = pure feedback, 1 = pure feedforward

    # Output limits
    max_horiz_vel: float = 15.0    # m/s horizontal command
    max_vert_vel:  float = 5.0     # m/s vertical command
    max_yaw_rate:  float = 1.5     # rad/s

    # Yaw PID
    yaw_kp: float = 1.0
    yaw_ki: float = 0.0
    yaw_kd: float = 0.1


class Autopilot:
    """
    Outer-loop position controller.

    Typical call sequence (50–100 Hz):
        cmd = autopilot.compute(state, guidance_output)
        bridge.send_command(cmd)

    Sends SET_POSITION_TARGET_LOCAL_NED with:
      - target position (NED)
      - target velocity (NED feedforward)
      - target yaw
    """

    def __init__(self, config: Optional[AutopilotConfig] = None):
        self.cfg = config or AutopilotConfig()

        self._pid_n = PIDController(self.cfg.pos_kp, self.cfg.pos_ki, self.cfg.pos_kd,
                                    -self.cfg.max_horiz_vel, self.cfg.max_horiz_vel)
        self._pid_e = PIDController(self.cfg.pos_kp, self.cfg.pos_ki, self.cfg.pos_kd,
                                    -self.cfg.max_horiz_vel, self.cfg.max_horiz_vel)
        self._pid_d = PIDController(self.cfg.pos_kp, self.cfg.pos_ki, self.cfg.pos_kd,
                                    -self.cfg.max_vert_vel,  self.cfg.max_vert_vel)
        self._pid_yaw = PIDController(self.cfg.yaw_kp, self.cfg.yaw_ki, self.cfg.yaw_kd,
                                      -self.cfg.max_yaw_rate, self.cfg.max_yaw_rate)

    def reset(self):
        for pid in [self._pid_n, self._pid_e, self._pid_d, self._pid_yaw]:
            pid.reset()

    def compute(self, state: VehicleState, guidance: GuidanceOutput) -> ControlCommand:
        now = time.monotonic()
        pos  = state.position_ned()
        tpos = guidance.target_position
        tvel = guidance.target_velocity

        # Position errors
        err_n = tpos[0] - pos[0]
        err_e = tpos[1] - pos[1]
        err_d = tpos[2] - pos[2]

        # PID outputs (velocity corrections)
        vn_fb = self._pid_n.update(err_n, now)
        ve_fb = self._pid_e.update(err_e, now)
        vd_fb = self._pid_d.update(err_d, now)

        # Blend feedforward + feedback
        w = self.cfg.vel_ff_weight
        vn = w * tvel[0] + (1 - w) * vn_fb
        ve = w * tvel[1] + (1 - w) * ve_fb
        vd = w * tvel[2] + (1 - w) * vd_fb

        # Clamp
        horiz = math.sqrt(vn**2 + ve**2)
        if horiz > self.cfg.max_horiz_vel:
            scale = self.cfg.max_horiz_vel / horiz
            vn, ve = vn * scale, ve * scale
        vd = float(np.clip(vd, -self.cfg.max_vert_vel, self.cfg.max_vert_vel))

        # Yaw control
        yaw_err = self._wrap_pi(guidance.target_yaw - state.yaw)
        yaw_rate = self._pid_yaw.update(yaw_err, now)

        # MAVLink type_mask: use position + velocity + yaw rate
        # Bit 0=ignore pos, 1=ignore vel, 2=ignore accel, 3=ignore pos_z, ...
        # We send position + velocity (feedforward), ignore acceleration
        TYPE_MASK = 0b0000_0111_1111_0000  # send pos + vel, ignore accel + yaw pos, use yaw rate

        cmd = ControlCommand(
            target_north=tpos[0],
            target_east=tpos[1],
            target_down=tpos[2],
            target_vn=vn,
            target_ve=ve,
            target_vd=vd,
            target_yaw=guidance.target_yaw,
            target_yaw_rate=yaw_rate,
            type_mask=TYPE_MASK,
            use_attitude=False,
        )
        return cmd

    @staticmethod
    def _wrap_pi(angle: float) -> float:
        """Wrap angle to [-π, π]."""
        while angle > math.pi:  angle -= 2 * math.pi
        while angle < -math.pi: angle += 2 * math.pi
        return angle

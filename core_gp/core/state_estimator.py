"""
State Estimator
Fuses MAVLink telemetry (attitude, IMU) with visual odometry to maintain
a best-estimate of vehicle state in the NED frame.

The simulator provides attitude and IMU directly, so we do NOT need a full
EKF for orientation. We use a complementary filter for position when GPS/
groundtruth position is available, and fall back to dead-reckoning + VO
when it isn't.
"""

import math
import time
import threading
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from core_gp.comms.mavlink_bridge import TelemetryState
from core_gp.perception.perception import VisualOdometryResult

logger = logging.getLogger(__name__)


@dataclass
class VehicleState:
    """
    Unified vehicle state in NED frame (origin = arm point).
    All angles in radians, distances in metres, velocities in m/s.
    """
    # --- Position (NED) ---
    north: float = 0.0
    east: float = 0.0
    down: float = 0.0

    # --- Velocity (NED) ---
    vn: float = 0.0
    ve: float = 0.0
    vd: float = 0.0

    # --- Attitude (ZYX Euler, radians) ---
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    # --- Angular rates (rad/s, body frame) ---
    p: float = 0.0  # roll rate
    q: float = 0.0  # pitch rate
    r: float = 0.0  # yaw rate

    # --- Derived ---
    speed: float = 0.0           # |v| m/s
    altitude_agl: float = 0.0   # above arm point (positive up) = -down

    # --- Meta ---
    timestamp: float = field(default_factory=time.time)
    position_source: str = 'mavlink'   # 'mavlink' | 'dead_reckoning' | 'vo_aided'

    def position_ned(self) -> np.ndarray:
        return np.array([self.north, self.east, self.down])

    def velocity_ned(self) -> np.ndarray:
        return np.array([self.vn, self.ve, self.vd])

    def euler_angles(self) -> np.ndarray:
        return np.array([self.roll, self.pitch, self.yaw])

    def body_to_ned_R(self) -> np.ndarray:
        """3×3 rotation matrix: body → NED."""
        cr, sr = math.cos(self.roll),  math.sin(self.roll)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        cy, sy = math.cos(self.yaw),   math.sin(self.yaw)
        return np.array([
            [cp*cy,  sr*sp*cy - cr*sy,  cr*sp*cy + sr*sy],
            [cp*sy,  sr*sp*sy + cr*cy,  cr*sp*sy - sr*cy],
            [-sp,    sr*cp,              cr*cp           ],
        ])


class StateEstimator:
    """
    Maintains VehicleState by fusing:
      1. MAVLink ATTITUDE  → roll, pitch, yaw, angular rates (primary, 120 Hz)
      2. MAVLink LOCAL_POSITION_NED / HIGHRES_IMU → position & velocity
      3. Visual Odometry   → position correction when MAVLink position is unavailable

    Call update_from_mavlink() at the telemetry rate (~120 Hz).
    Call update_from_vo()      at the camera rate (~30 Hz).
    """

    # Complementary filter weight for VO position blending
    VO_ALPHA = 0.05   # weight given to VO correction (0 = ignore, 1 = fully trust VO)
    IMU_ALPHA= 0.05   # weight given to the IMU integration of Gyro and Accel. (0 = ignore, 1 = fully trust)
    LPF_ALPHA= 0.05

    def __init__(self):
        self._state = VehicleState()
        self._lock = threading.Lock()
        self._last_update = time.monotonic()
        self._have_mavlink_position = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------


    def get_state(self) -> VehicleState:
        with self._lock:
            return VehicleState(**self._state.__dict__)
        
    def predict_state(self, imu: TelemetryState, dt):
            s = self._state
            #No need to predict this I don't think
            # # Attitude — direct from simulator (authoritative)
            # s.roll       = imu.roll
            # s.pitch      = imu.pitch
            # s.yaw        = imu.yaw
            # s.p          = imu.rollspeed
            # s.q          = imu.pitchspeed
            # s.r          = imu.yawspeed

            # s.roll += s.p * dt
            # s.pitch += s.q*dt
            # s.yaw += s.r*dt


            s.north += imu.vn*dt
            s.east  += imu.ve*dt
            s.down  += imu.vd*dt
            s.vn    += imu.ax*dt
            s.ve    += imu.ay*dt
            s.vd    += imu.az*dt
            s.position_source = 'preditction'
            self._have_mavlink_position = False

#Include Covariance, Measurement uncertainty and process uncertainty
    def update_from_mavlink(self, telem: TelemetryState,P,Q,R_meas):
        """Called by MAVLink bridge callback (~120 Hz)."""
        now = time.monotonic()
        with self._lock:
            s = self._state

            # Attitude — direct from simulator (authoritative)
            s.roll       = telem.roll
            s.pitch      = telem.pitch
            s.yaw        = telem.yaw
            s.p          = telem.rollspeed
            s.q          = telem.pitchspeed
            s.r          = telem.yawspeed

            # Position & velocity
            if any([telem.north, telem.east, telem.down,
                    telem.vn, telem.ve, telem.vd]):
                s.north = telem.north
                s.east  = telem.east
                s.down  = telem.down
                s.vn    = telem.vn
                s.ve    = telem.ve
                s.vd    = telem.vd
                s.position_source = 'mavlink'
                self._have_mavlink_position = True
            else:
                # Dead-reckoning fallback
                dt = now - self._last_update
                s.north += s.vn * dt
                s.east  += s.ve * dt
                s.down  += s.vd * dt

                # Propagate velocity from IMU (body → NED)
                R = s.body_to_ned_R()
                a_body = np.array([telem.ax, telem.ay, telem.az])
                a_ned  = R @ a_body
                a_ned[2] += 9.81   # remove gravity (NED: +Z is down, gravity is +)
                s.vn += a_ned[0] * dt
                s.ve += a_ned[1] * dt
                s.vd += a_ned[2] * dt
                s.position_source = 'dead_reckoning'

            # Derived quantities
            s.speed       = math.sqrt(s.vn**2 + s.ve**2 + s.vd**2)
            s.altitude_agl = -s.down
            s.timestamp   = time.time()
            self._last_update = now

    def update_from_vo(self, vo: VisualOdometryResult):
        """
        Blend VO delta into position estimate. Only meaningful when
        dead-reckoning (no MAVLink position) or as a soft correction.
        """
        if not vo.valid:
            return
        with self._lock:
            s = self._state
            if s.position_source == 'dead_reckoning':
                # VO delta is in camera frame — rotate to NED using current yaw
                cy, sy = math.cos(s.yaw), math.sin(s.yaw)
                dn = vo.delta_position[2] * cy - vo.delta_position[0] * sy
                de = vo.delta_position[2] * sy + vo.delta_position[0] * cy
                # Complementary blend
                s.north += self.VO_ALPHA * dn
                s.east  += self.VO_ALPHA * de
                s.position_source = 'vo_aided'

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def distance_to(self, target_ned: np.ndarray) -> float:
        pos = self._state.position_ned()
        return float(np.linalg.norm(target_ned - pos))

    def bearing_to(self, target_ned: np.ndarray) -> float:
        """Yaw angle (rad) from current position to target, in NED plane."""
        s = self._state
        dn = target_ned[0] - s.north
        de = target_ned[1] - s.east
        return math.atan2(de, dn)
    

   
    def _low_pass_filter(z_old, u_new, lpf_alpha):
        if z_old is None:
            return u_new
        return lpf_alpha * z_old + (1-lpf_alpha) * u_new
"""
MAVLink Bridge — handles all MAVLink2 communication over UDP.
Receives: HEARTBEAT, ATTITUDE, HIGHRES_IMU, TIMESYNC
Sends:    SET_POSITION_TARGET_LOCAL_NED, SET_ATTITUDE_TARGET, HEARTBEAT
"""

import asyncio
import threading
import time
import struct
import logging
from dataclasses import dataclass, field
from typing import Optional, Callable

try:
    from pymavlink import mavutil
except ImportError:
    mavutil = None  # Graceful degradation for development/testing

logger = logging.getLogger(__name__)


@dataclass
class TelemetryState:
    """Snapshot of the latest telemetry from the simulator (NED frame)."""
    # Position (meters, NED from arm point)
    north: float = 0.0
    east: float = 0.0
    down: float = 0.0

    # Velocity (m/s, NED)
    vn: float = 0.0
    ve: float = 0.0
    vd: float = 0.0

    # Attitude (radians)
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    # Angular rates (rad/s)
    rollspeed: float = 0.0
    pitchspeed: float = 0.0
    yawspeed: float = 0.0

    # IMU
    ax: float = 0.0
    ay: float = 0.0
    az: float = 0.0

    timestamp_ns: int = 0
    last_heartbeat: float = field(default_factory=time.time)
    connected: bool = False


@dataclass
class ControlCommand:
    """
    A control command to be sent to the simulator.
    Use EITHER position-target OR attitude-target per cycle.
    """
    # SET_POSITION_TARGET_LOCAL_NED fields (NED, m/s, m/s²)
    target_north: float = 0.0
    target_east: float = 0.0
    target_down: float = 0.0
    target_vn: float = 0.0
    target_ve: float = 0.0
    target_vd: float = 0.0
    target_yaw: float = 0.0          # radians
    target_yaw_rate: float = 0.0     # rad/s
    type_mask: int = 0b0000_1111_1111_1000  # position only by default

    # SET_ATTITUDE_TARGET fields
    roll_cmd: float = 0.0
    pitch_cmd: float = 0.0
    yaw_cmd: float = 0.0
    thrust_cmd: float = 0.5  # 0–1
    use_attitude: bool = False


class MAVLinkBridge:
    """
    Async-friendly MAVLink bridge. Spawns a reader thread that
    continuously processes incoming messages and populates TelemetryState.

    Usage
    -----
        bridge = MAVLinkBridge(host='127.0.0.1', port=14550)
        bridge.connect()
        state = bridge.get_state()
        bridge.send_command(cmd)
        bridge.close()
    """

    HEARTBEAT_INTERVAL = 0.5  # seconds (≥ 2 Hz required)
    COMMAND_RATE_HZ = 50       # 50 Hz command loop

    def __init__(self, host: str = '127.0.0.1', port: int = 14550,
                 telemetry_callback: Optional[Callable[[TelemetryState], None]] = None):
        self.host = host
        self.port = port
        self._state = TelemetryState()
        self._lock = threading.Lock()
        self._running = False
        self._connection = None
        self._telemetry_callback = telemetry_callback

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self):
        if mavutil is None:
            raise RuntimeError("pymavlink not installed. Run: pip install pymavlink")
        conn_str = f"udpin:{self.host}:{self.port}"
        logger.info(f"Connecting MAVLink on {conn_str}")
        self._connection = mavutil.mavlink_connection(conn_str)
        self._running = True
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._reader_thread.start()
        self._heartbeat_thread.start()
        logger.info("MAVLink bridge started.")

    def close(self):
        self._running = False
        if self._connection:
            self._connection.close()

    # ------------------------------------------------------------------
    # State Access
    # ------------------------------------------------------------------

    def get_state(self) -> TelemetryState:
        with self._lock:
            # Return a shallow copy so callers don't race
            return TelemetryState(**self._state.__dict__)

    def is_connected(self) -> bool:
        with self._lock:
            return self._state.connected

    # ------------------------------------------------------------------
    # Command Sending
    # ------------------------------------------------------------------

    def send_command(self, cmd: ControlCommand):
        if self._connection is None:
            return
        if cmd.use_attitude:
            self._send_attitude_target(cmd)
        else:
            self._send_position_target(cmd)

    def _send_position_target(self, cmd: ControlCommand):
        self._connection.mav.set_position_target_local_ned_send(
            int(time.time() * 1000) & 0xFFFFFFFF,  # time_boot_ms
            1, 1,                                   # target_system, target_component
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            cmd.type_mask,
            cmd.target_north, cmd.target_east, cmd.target_down,
            cmd.target_vn, cmd.target_ve, cmd.target_vd,
            0, 0, 0,                                # acceleration (unused)
            cmd.target_yaw, cmd.target_yaw_rate,
        )

    def _send_attitude_target(self, cmd: ControlCommand):
        import math
        # Convert Euler → quaternion (ZYX)
        cr, sr = math.cos(cmd.roll_cmd / 2), math.sin(cmd.roll_cmd / 2)
        cp, sp = math.cos(cmd.pitch_cmd / 2), math.sin(cmd.pitch_cmd / 2)
        cy, sy = math.cos(cmd.yaw_cmd / 2), math.sin(cmd.yaw_cmd / 2)
        q = [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
        self._connection.mav.set_attitude_target_send(
            int(time.time() * 1000) & 0xFFFFFFFF,
            1, 1,
            0b00000000,          # body rates ignored
            q,
            0, 0, 0,             # body roll/pitch/yaw rate
            cmd.thrust_cmd,
        )

    # ------------------------------------------------------------------
    # Internal Loops
    # ------------------------------------------------------------------

    def _heartbeat_loop(self):
        while self._running:
            if self._connection:
                self._connection.mav.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_GCS,
                    mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                    0, 0, 0,
                )
            time.sleep(self.HEARTBEAT_INTERVAL)

    def _reader_loop(self):
        while self._running:
            if self._connection is None:
                time.sleep(0.01)
                continue
            msg = self._connection.recv_match(blocking=True, timeout=1.0)
            if msg is None:
                continue
            self._dispatch(msg)

    def _dispatch(self, msg):
        t = msg.get_type()
        with self._lock:
            if t == 'HEARTBEAT':
                self._state.connected = True
                self._state.last_heartbeat = time.time()

            elif t == 'ATTITUDE':
                self._state.roll = msg.roll
                self._state.pitch = msg.pitch
                self._state.yaw = msg.yaw
                self._state.rollspeed = msg.rollspeed
                self._state.pitchspeed = msg.pitchspeed
                self._state.yawspeed = msg.yawspeed

            elif t == 'HIGHRES_IMU':
                self._state.ax = msg.xacc
                self._state.ay = msg.yacc
                self._state.az = msg.zacc
                self._state.vn = msg.xgyro   # placeholder: simulator may differ
                self._state.timestamp_ns = msg.time_usec * 1000

            elif t == 'LOCAL_POSITION_NED':
                self._state.north = msg.x
                self._state.east = msg.y
                self._state.down = msg.z
                self._state.vn = msg.vx
                self._state.ve = msg.vy
                self._state.vd = msg.vz

        if self._telemetry_callback:
            self._telemetry_callback(self.get_state())

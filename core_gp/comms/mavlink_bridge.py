"""
MAVLink Bridge
===============

Responsibilities
----------------
Receives:
    HEARTBEAT
    ATTITUDE
    HIGHRES_IMU
    LOCAL_POSITION_NED
    ODOMETRY
    TIMESYNC
    ENCAPSULATED_DATA (race + track info)

Sends:
    HEARTBEAT
    SET_POSITION_TARGET_LOCAL_NED
    SET_ATTITUDE_TARGET
    ARM / DISARM

Architecture
-------------
- Reader thread updates authoritative telemetry cache
- Publisher thread emits coherent telemetry snapshots
- External estimator consumes synchronized state
"""

import logging
import math
import struct
import threading
import time

from dataclasses import dataclass, field
from typing import Callable, Optional

from pymavlink import mavutil

logger = logging.getLogger(__name__)


# ============================================================
# Custom encapsulated message IDs
# ============================================================

ENCAPSULATED_RACE_STATUS_MSG_ID = 1
ENCAPSULATED_TRACK_INFO_MSG_ID = 2


# ============================================================
# Telemetry
# ============================================================

@dataclass
class TelemetryState:
    """Unified telemetry snapshot."""

    # Position (NED)
    north: float = 0.0
    east: float = 0.0
    down: float = 0.0

    # Velocity (NED)
    vn: float = 0.0
    ve: float = 0.0
    vd: float = 0.0

    # Attitude
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    # Angular rates
    rollspeed: float = 0.0
    pitchspeed: float = 0.0
    yawspeed: float = 0.0

    # IMU accel
    ax: float = 0.0
    ay: float = 0.0
    az: float = 0.0

    # IMU gyro
    gx: float = 0.0
    gy: float = 0.0
    gz: float = 0.0

    # Meta
    timestamp_ns: int = 0
    last_heartbeat: float = field(default_factory=time.time)
    connected: bool = False

    position_source: str = "unknown"


@dataclass
class RaceStatus:
    sim_boot_time_ms: int = 0
    race_start_boot_time_ms: int = -1
    race_finish_time_ns: int = -1
    active_gate_index: int = 0
    last_gate_race_time: float = 0.0


@dataclass
class ControlCommand:

    # Position target
    target_north: float = 0.0
    target_east: float = 0.0
    target_down: float = 0.0

    target_vn: float = 0.0
    target_ve: float = 0.0
    target_vd: float = 0.0

    target_yaw: float = 0.0
    target_yaw_rate: float = 0.0

    type_mask: int = (
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
    )

    # Attitude mode
    use_attitude: bool = False
    roll_cmd: float = 0.0
    pitch_cmd: float = 0.0
    yaw_cmd: float = 0.0
    thrust_cmd: float = 0.5


# ============================================================
# Bridge
# ============================================================

class MAVLinkBridge:

    HEARTBEAT_INTERVAL = 0.5
    TELEMETRY_PUBLISH_HZ = 100

    def __init__(
        self,
        host='127.0.0.1',
        port=14550,
        telemetry_callback: Optional[
            Callable[[TelemetryState], None]
        ] = None,
    ):

        self.host = host
        self.port = port

        self._connection = None
        self._running = False

        self._lock = threading.Lock()

        self._state = TelemetryState()
        self._race_status = RaceStatus()

        self._telemetry_callback = telemetry_callback

        self.target_system = None
        self.target_component = None

        # Track map chunks
        self.track_chunks = {}
        self.expected_num_track_chunks = {}

    # ========================================================
    # Connection
    # ========================================================

    def connect(self):

        conn_str = f"udpin:{self.host}:{self.port}"

        logger.info(
            f"Connecting MAVLink on {conn_str}"
        )

        self._connection = mavutil.mavlink_connection(
            conn_str
        )

        logger.info(
            "Waiting for simulator heartbeat..."
        )

        self._connection.wait_heartbeat()

        self.target_system = (
            self._connection.target_system
        )

        self.target_component = (
            self._connection.target_component
        )

        logger.info(
            f"Connected to "
            f"system={self.target_system}, "
            f"component={self.target_component}"
        )

        self._running = True

        threading.Thread(
            target=self._reader_loop,
            daemon=True
        ).start()

        threading.Thread(
            target=self._heartbeat_loop,
            daemon=True
        ).start()

        threading.Thread(
            target=self._publisher_loop,
            daemon=True
        ).start()

    def close(self):
        self._running = False

        if self._connection:
            self._connection.close()

    # ========================================================
    # Public API
    # ========================================================

    def get_state(self) -> TelemetryState:
        with self._lock:
            return TelemetryState(
                **self._state.__dict__
            )

    def get_race_status(self):
        with self._lock:
            return RaceStatus(
                **self._race_status.__dict__
            )

    def is_connected(self):

        with self._lock:
            dt = (
                time.time() -
                self._state.last_heartbeat
            )

            return (
                self._state.connected
                and dt < 2.0
            )

    # ========================================================
    # Command Interface
    # ========================================================

    def arm(self):

        self._connection.mav.command_long_send(
            self.target_system,
            self.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1,
            0, 0, 0, 0, 0, 0
        )

    def send_command(
        self,
        cmd: ControlCommand
    ):

        if cmd.use_attitude:
            self._send_attitude(cmd)
        else:
            self._send_position(cmd)

    def _send_position(
        self,
        cmd: ControlCommand
    ):

        now_ms = (
            int(time.time() * 1000)
            & 0xFFFFFFFF
        )

        self._connection.mav.set_position_target_local_ned_send(
            now_ms,
            self.target_system,
            self.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            cmd.type_mask,

            cmd.target_north,
            cmd.target_east,
            cmd.target_down,

            cmd.target_vn,
            cmd.target_ve,
            cmd.target_vd,

            0,
            0,
            0,

            cmd.target_yaw,
            cmd.target_yaw_rate,
        )

    def _send_attitude(
        self,
        cmd: ControlCommand
    ):

        cr = math.cos(cmd.roll_cmd / 2)
        sr = math.sin(cmd.roll_cmd / 2)

        cp = math.cos(cmd.pitch_cmd / 2)
        sp = math.sin(cmd.pitch_cmd / 2)

        cy = math.cos(cmd.yaw_cmd / 2)
        sy = math.sin(cmd.yaw_cmd / 2)

        q = [
            cr*cp*cy + sr*sp*sy,
            sr*cp*cy - cr*sp*sy,
            cr*sp*cy + sr*cp*sy,
            cr*cp*sy - sr*sp*cy,
        ]

        self._connection.mav.set_attitude_target_send(
            int(time.time() * 1000),

            self.target_system,
            self.target_component,

            mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE,

            q,
            0,
            0,
            0,
            cmd.thrust_cmd,
        )

    # ========================================================
    # Threads
    # ========================================================

    def _heartbeat_loop(self):

        while self._running:

            self._connection.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_GCS,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                0,
                0,
                0,
            )

            time.sleep(
                self.HEARTBEAT_INTERVAL
            )

    def _publisher_loop(self):

        dt = (
            1.0 /
            self.TELEMETRY_PUBLISH_HZ
        )

        while self._running:

            if self._telemetry_callback:

                self._telemetry_callback(
                    self.get_state()
                )

            time.sleep(dt)

    def _reader_loop(self):

        while self._running:

            msg = self._connection.recv_match(
                blocking=True,
                timeout=0.02
            )

            if msg is None:
                continue

            self._dispatch(msg)

    # ========================================================
    # Message Dispatch
    # ========================================================

    def _dispatch(self, msg):

        t = msg.get_type()

        with self._lock:

            if t == "HEARTBEAT":

                self._state.connected = True
                self._state.last_heartbeat = (
                    time.time()
                )

            elif t == "ATTITUDE":

                self._state.roll = msg.roll
                self._state.pitch = msg.pitch
                self._state.yaw = msg.yaw

                self._state.rollspeed = (
                    msg.rollspeed
                )

                self._state.pitchspeed = (
                    msg.pitchspeed
                )

                self._state.yawspeed = (
                    msg.yawspeed
                )

            elif t == "HIGHRES_IMU":

                self._state.ax = msg.xacc
                self._state.ay = msg.yacc
                self._state.az = msg.zacc

                self._state.gx = msg.xgyro
                self._state.gy = msg.ygyro
                self._state.gz = msg.zgyro

                self._state.timestamp_ns = (
                    msg.time_usec * 1000
                )

            elif t == "LOCAL_POSITION_NED":

                self._state.north = msg.x
                self._state.east = msg.y
                self._state.down = msg.z

                self._state.vn = msg.vx
                self._state.ve = msg.vy
                self._state.vd = msg.vz

                self._state.position_source = (
                    "local_position_ned"
                )

            elif t == "ODOMETRY":

                self._state.north = msg.x
                self._state.east = msg.y
                self._state.down = msg.z

                self._state.vn = msg.vx
                self._state.ve = msg.vy
                self._state.vd = msg.vz

                self._state.position_source = (
                    "odometry"
                )

            elif t == "ENCAPSULATED_DATA":

                self._handle_encapsulated_data(
                    msg
                )

    # ========================================================
    # Encapsulated Data
    # ========================================================

    def _handle_encapsulated_data(
        self,
        msg
    ):

        raw_payload = bytes(msg.data)

        if len(raw_payload) == 0:
            return

        data_type = raw_payload[0]

        if (
            data_type ==
            ENCAPSULATED_RACE_STATUS_MSG_ID
        ):
            self._on_race_status(
                raw_payload
            )

    def _on_race_status(
        self,
        raw_payload
    ):

        (
            _,
            sim_boot_time_ms,
            race_start_boot_time_ms,
            race_finish_time_ns,
            active_gate_index,
            last_gate_race_time
        ) = struct.unpack_from(
            "<BQqqIq",
            raw_payload
        )

        self._race_status = RaceStatus(
            sim_boot_time_ms,
            race_start_boot_time_ms,
            race_finish_time_ns,
            active_gate_index,
            last_gate_race_time
        )



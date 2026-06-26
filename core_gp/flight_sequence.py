"""
flight_sequence.py
------------------
Basic flight workflow for the Anduril AI Grand Prix sim:
  1. Connect & wait for heartbeat
  2. Arm
  3. Takeoff to a target altitude (NED -5 m)
  4. Fly forward at a fixed speed
  5. Query and print gate locations from track data
"""

import math
import struct
import time
import threading
import logging
import guidance.guidance as guidance
from pymavlink import mavutil
import numpy as np
from core.state_estimator import VehicleState
from control.attitude_autopilot import AttitudeAutopilot
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


# ---------------------------------------------------------------------------
# Gate Store
# ---------------------------------------------------------------------------

class GateStore:
    """Thread-safe store for gate positions received from the sim."""

    def __init__(self):
        self._lock = threading.Lock()
        self._gates: list[dict] = []
        self.received = threading.Event()

    def set_gates(self, gates: list[dict]):
        with self._lock:
            self._gates = gates
        self.received.set()
        logger.info(f"GateStore: received {len(gates)} gates.")

    def get_gates(self) -> list[dict]:
        with self._lock:
            return list(self._gates)

    def __len__(self):
        with self._lock:
            return len(self._gates)


# ---------------------------------------------------------------------------
# GateAwareMAVLinkRX
# ---------------------------------------------------------------------------

from mavlink_rx import MAVLinkRX

class GateAwareMAVLinkRX(MAVLinkRX):
    """Extends MAVLinkRX to forward parsed track data into a GateStore."""

    def __init__(self, mavlink_connection, data, gate_store: GateStore):
        super().__init__(mavlink_connection, data)
        self._gate_store = gate_store

    @classmethod
    def create_mavlink_rx(cls, mavlink_connection, data, gate_store: GateStore = None):
        rx = cls(mavlink_connection, data, gate_store)
        rx.thread = threading.Thread(
            target=rx.mavlink_receive_loop,
            daemon=False
        )
        rx.is_running = True
        rx.thread.start()
        return rx

    def on_track_data(self, payload: bytes):
        num_gates, = struct.unpack_from("<H", payload)
        payload = payload[2:]
        gates = []
        for _ in range(num_gates):
            (gate_id,
             pos_n, pos_e, pos_d,
             ow, ox, oy, oz,
             width, height) = struct.unpack_from("<Hfffffffff", payload)
            payload = payload[38:]
            gates.append({
                "gate_id": gate_id,
                "north":   pos_n,
                "east":    pos_e,
                "down":    pos_d,
                "orient":  (ow, ox, oy, oz),
                "width":   width,
                "height":  height,
            })
        self._gate_store.set_gates(gates)


# ---------------------------------------------------------------------------
# FlightController — operates directly on sim_conn + shared_data.
#
# FIX #4 (revised): Do NOT open a MAVLinkBridge (second UDP socket) here.
# UDP delivers each datagram to exactly one socket; a second socket on the
# same port starves sim_conn of heartbeats and all other messages.
# Instead, send MAVLink messages directly through the existing sim_conn and
# read telemetry from the shared_data dict that MAVLinkRX populates.
# ---------------------------------------------------------------------------

class FlightController:
    """
    Controls the drone using the shared sim_conn pymavlink connection
    and the shared_data dict populated by MAVLinkRX.
    No second UDP socket is opened.
    """

    FORWARD_SPEED   = 0.1   # m/s north during forward flight (used by fly_forward only)

    # NOTE: SET_POSITION_TARGET_LOCAL_NED is confirmed broken in this sim --
    # streaming it faster than ~1 message/1-2s causes an unbounded climb to
    # 150m+ regardless of what's actually commanded (proven across 7
    # isolated tests: hold_test.py, launch_profile_test.py,
    # single_shot_test.py, frozen_target_test.py, low_rate_test.py,
    # rate_threshold_test.py, attitude_target_test.py). SET_ATTITUDE_TARGET
    # at the same rate behaves like a normal physical system instead
    # (bounded terminal velocity). _run_mpc_phase and hold_position() now
    # close the position/velocity -> attitude/thrust loop themselves via
    # AttitudeAutopilot and drive _send_attitude_target. SEND_TYPE_MASK and
    # _send_position_target are kept only for the diagnostic test scripts
    # above that demonstrate the bug -- do not wire them back into the
    # live flight path.
    SEND_TYPE_MASK = (
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
    )

    def __init__(self, sim_conn, shared_data: dict, system_boot_ms: int = None):
        self.conn = sim_conn
        self.data = shared_data
        # system_boot_ms anchors time_boot_ms to boot rather than Unix epoch.
        # If not supplied, record "now" as boot time (good enough for a fresh run).
        self._system_boot_ms = system_boot_ms if system_boot_ms is not None else int(time.time() * 1000)

    # ------------------------------------------------------------------
    # Telemetry helpers — read from shared_data written by MAVLinkRX
    # ------------------------------------------------------------------

    def _pos(self):
        """Return (north, east, down) from the latest LOCAL_POSITION_NED."""
        return (
            self.data.get('pos_x', 0.0),
            self.data.get('pos_y', 0.0),
            self.data.get('pos_z', 0.0),
        )

    def _vel(self):
        """Return (vn, ve, vd) from the latest LOCAL_POSITION_NED."""
        return (
            self.data.get('vel_x', 0.0),
            self.data.get('vel_y', 0.0),
            self.data.get('vel_z', 0.0),
        )

    def _yaw(self) -> float:
        """Return yaw (rad) from the latest ATTITUDE message."""
        return self.data.get('yaw', 0.0)

    def _pitch(self) -> float:
        """Return pitch (rad) from the latest ATTITUDE message."""
        return self.data.get('pitch', 0.0)

    def _roll(self) -> float:
        """Return roll (rad) from the latest ATTITUDE message."""
        return self.data.get('roll', 0.0)

    def _wait_for_telemetry(self, timeout_s: float = 5.0):
        """Block until shared_data contains at least one position update."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            n, e, d = self._pos()
            if n != 0.0 or e != 0.0 or d != 0.0:
                return
            time.sleep(0.05)
        logger.warning("_wait_for_telemetry: timed out — proceeding with last known state.")

    # ------------------------------------------------------------------
    # MAVLink send helpers
    # ------------------------------------------------------------------

    def _send_position_target(self, type_mask, x, y, z, vx, vy, vz):
        # time_boot_ms must fit in uint32 — use ms since boot, not Unix epoch
        now_boot_ms = int(time.time() * 1000) - self._system_boot_ms

        logger.info(f"type_mask={type_mask}")
        self.conn.mav.set_position_target_local_ned_send(
            now_boot_ms,
            self.conn.target_system,
            self.conn.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            type_mask,
            x, y, z,
            vx, vy, vz,
            0.0, 0.0, 0.0,  # acceleration — ignored
            self._yaw(),
            0.0,            # yaw rate — ignored
        )

    def _send_attitude_target(self, roll_rate, pitch_rate, yaw_rate, thrust):
        now_boot_ms = int(time.time() * 1000) - self._system_boot_ms
        # ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE = rate mode:
        # ignore the quaternion, control body angular rates + thrust directly.
        self.conn.mav.set_attitude_target_send(
            now_boot_ms,
            self.conn.target_system,
            self.conn.target_component,
            mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE,
            [1.0, 0.0, 0.0, 0.0],  # quaternion ignored in rate mode
            roll_rate,
            pitch_rate,
            yaw_rate,
            thrust,
        )

    # ------------------------------------------------------------------
    # High-level flight commands
    # ------------------------------------------------------------------

    def send_sim_reset(self):
        MAVLINK_CMD_SIM_RESET = 31000
        logger.info("Sending sim reset command...")
        self.conn.mav.command_long_send(
            self.conn.target_system,
            self.conn.target_component,
            MAVLINK_CMD_SIM_RESET,
            0, 0, 0, 0, 0, 0, 0, 0
        )

    def arm(self):
        logger.info("Arming...")
        self.conn.mav.command_long_send(
            self.conn.target_system,
            self.conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1, 0, 0, 0, 0, 0, 0
        )
        # Wait until the heartbeat confirms the drone is armed.
        # Without this we may start sending flight commands before the
        # autopilot is ready, which is why takeoff was silently ignored.
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            if self.data.get('armed', False):
                logger.info("Arm confirmed via heartbeat.")
                return
            bm  = self.data.get('base_mode', '?')
            cm  = self.data.get('custom_mode', '?')
            st  = self.data.get('system_status', '?')
            logger.info(f"  Waiting for arm... base_mode={bm} custom_mode={cm} system_status={st}")
            time.sleep(0.5)
        logger.warning("Arm confirmation timed out — proceeding anyway.")

    def _run_mpc_phase(self, guidance, timeout_s: float, phase_label: str) -> bool:
        """
        Drives one MPCGuidance instance to completion. Each tick: read
        telemetry, ask guidance for the next position+velocity reference,
        then run it through AttitudeAutopilot (our own position/velocity ->
        attitude/thrust loop) and send the result via _send_attitude_target.
        SET_POSITION_TARGET_LOCAL_NED is not used here -- see the note on
        SEND_TYPE_MASK above for why.

        Returns True if the phase's guidance reported course_complete,
        False if it timed out.
        """
        attitude_autopilot = AttitudeAutopilot()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            n, e, cur_d = self._pos()
            vn, ve, vd = self._vel()
            state = VehicleState(
                north=n, east=e, down=cur_d,
                vn=vn, ve=ve, vd=vd,
                roll=self._roll(), pitch=self._pitch(), yaw=self._yaw(),
            )

            output = guidance.compute(state=state, gates=[], obstacles=[])

            if output.course_complete:
                logger.info(f"{phase_label}: complete ({guidance.status_string()}).")
                # Send an explicit hold immediately instead of just returning —
                # otherwise the vehicle keeps whatever rate/thrust was last
                # commanded for however long it takes the next phase to
                # construct its guidance and send its first command.
                self.hold_position()
                return True

            roll_rate, pitch_rate, yaw_rate, thrust = attitude_autopilot.compute(state, output)
            self._send_attitude_target(roll_rate, pitch_rate, yaw_rate, thrust)

            logger.info(
                f"[{phase_label}] POS=({n:.1f},{e:.1f},{cur_d:.1f}) "
                f"VEL_CMD=({output.target_velocity[0]:.1f},"
                f"{output.target_velocity[1]:.1f},"
                f"{output.target_velocity[2]:.1f}) "
                f"TARGET_POS=({output.target_position[0]:.1f},"
                f"{output.target_position[1]:.1f},"
                f"{output.target_position[2]:.1f}) "
                f"ACTUAL_VEL=({vn:.1f},{ve:.1f},{vd:.1f}) "
                f"ATT=(roll={state.roll:.2f},pitch={state.pitch:.2f},yaw={state.yaw:.2f}) "
                f"ATT_CMD=(roll_rate={roll_rate:.2f},pitch_rate={pitch_rate:.2f},"
                f"yaw_rate={yaw_rate:.2f},thrust={thrust:.2f}) "
                f"YAW_TARGET={math.degrees(output.target_yaw):.1f} "
                f"MPC: {guidance.status_string()}"
            )
            time.sleep(0.05)

        logger.warning(f"{phase_label}: timeout reached.")
        return False

    def takeoff_mpc(self, climb_alt_m: float = 0.5, timeout_s: float = 15.0):
        """
        Climbs straight up to climb_alt_m AGL using the same MPC controller
        fly_gates uses for racing — takeoff is just a course with a single
        waypoint directly above the arm point. Replaces the old hand-tuned
        thrust-formula takeoff()/settle loop, whose floor on the thrust term
        (max(0.05, ...)) could never command a brake or descent and would
        run away indefinitely once climbing too fast.
        """
        from guidance.mpc_guidance import MPCGuidance, MPCConfig
        from guidance.guidance import CourseMap

        logger.info(f"Taking off to {climb_alt_m} m AGL via MPC...")
        self._wait_for_telemetry()

        n, e, d = self._pos()
        target_down = d - climb_alt_m   # NED: climbing decreases 'down'

        course_map = CourseMap()
        course_map.load_from_list([(n, e, target_down)], speed=1.5)
        # Default pass_distance (1.5 m) is tuned for gates, not a single
        # point target — with a short climb (e.g. 0.5 m) it would report
        # course_complete on the very first tick. Use a tight tolerance
        # instead so takeoff actually climbs before declaring done.
        cfg = MPCConfig(pass_distance=min(0.15, climb_alt_m / 2))
        guidance = MPCGuidance(course_map=course_map, config=cfg)

        self._run_mpc_phase(guidance, timeout_s, "TAKEOFF")

    def fly_forward(self, duration_s: float = 5.0, target_down: float = None):
        _, _, cur_d = self._pos()
        target_d    = target_down if target_down is not None else cur_d

        logger.info(f"Flying forward for {duration_s} s (attitude control)...")

        # Attitude rate mode: negative pitch_rate = nose down = forward motion.
        # Thrust is adjusted by a P-controller to hold the takeoff altitude.
        # 0.6 ≈ hover thrust per reference controller; KP tuned conservatively.
        HOVER_THRUST = 0.4
        FORWARD_PITCH = -0.1   # rad/s, gentle forward tilt (~6 deg/s)
        THRUST_KP    = 0.08    # thrust per metre of altitude error

        t0 = time.monotonic()
        while time.monotonic() - t0 < duration_s:
            _, _, cur_d = self._pos()
            # Positive error → drone too low → need more thrust
            alt_error_m = (-cur_d) - (-target_d)   # current_alt - target_alt
            thrust = max(0.3, min(0.9, HOVER_THRUST + alt_error_m * THRUST_KP))
            self._send_attitude_target(0.0, FORWARD_PITCH, 0.0, thrust)
            n, e, cur_d = self._pos()
            vn, _, _ = self._vel()
            logger.info(f"  N={n:.1f}  E={e:.1f}  Alt={-cur_d:.1f} m  Vn={vn:.1f} m/s  thrust={thrust:.2f}")
            time.sleep(0.05)

        logger.info("Forward flight complete — holding position.")
        self.hold_position()

    def hold_position(self, duration_s: float = 0.5, settle_speed_mps: float = 0.3,
                       max_settle_s: float = 3.0):
        """
        Actively holds the position measured at call time by streaming
        attitude/thrust commands through a fresh AttitudeAutopilot.
        Unlike the old position-target version, this is not fire-and-forget:
        there is no sim-side controller persisting our last setpoint, so
        holding only happens for as long as this loop keeps running.

        Runs for at least duration_s, then keeps going until speed drops
        below settle_speed_mps (bounded by max_settle_s) -- a fixed 0.5s
        was letting callers (e.g. fly_gates building its path) start the
        next phase with several m/s of residual velocity still on the
        vehicle (observed: a ~3 m/s carried-over climb that took the next
        phase 5+ seconds and >20m of altitude to fight back under control),
        which is the caller's problem to inherit, not this loop's to ignore.
        """
        hold_n, hold_e, hold_d = self._pos()
        hold_output = guidance.GuidanceOutput(
            target_position=np.array([hold_n, hold_e, hold_d]),
            target_velocity=np.zeros(3),
            target_yaw=self._yaw(),
        )
        attitude_autopilot = AttitudeAutopilot()
        min_deadline = time.monotonic() + duration_s
        max_deadline = time.monotonic() + max_settle_s
        while time.monotonic() < max_deadline:
            n, e, d = self._pos()
            vn, ve, vd = self._vel()
            state = VehicleState(
                north=n, east=e, down=d,
                vn=vn, ve=ve, vd=vd,
                roll=self._roll(), pitch=self._pitch(), yaw=self._yaw(),
            )
            roll_rate, pitch_rate, yaw_rate, thrust = attitude_autopilot.compute(state, hold_output)
            self._send_attitude_target(roll_rate, pitch_rate, yaw_rate, thrust)
            speed = math.sqrt(vn * vn + ve * ve + vd * vd)
            logger.info(
                f"HOLD: pos=({n:.2f},{e:.2f},{d:.2f}) vel=({vn:.2f},{ve:.2f},{vd:.2f}) speed={speed:.2f} "
                f"att=(roll={state.roll:.3f},pitch={state.pitch:.3f},yaw={state.yaw:.3f}) "
                f"ATT_CMD=(roll_rate={roll_rate:.2f},pitch_rate={pitch_rate:.2f},"
                f"yaw_rate={yaw_rate:.2f},thrust={thrust:.2f})"
            )
            now = time.monotonic()
            if now >= min_deadline and speed < settle_speed_mps:
                return
            time.sleep(0.05)
        logger.warning(
            f"hold_position: speed still {speed:.2f} m/s after {max_settle_s}s settle "
            f"window -- proceeding anyway."
        )


    def fly_gates(self, gate_store: 'GateStore', timeout_per_gate_s: float = 30.0):
            """
            Navigate the full gate sequence. MPCGuidance owns trajectory
            generation (a continuous reference across the whole course, not
            a fresh line segment per gate); _run_mpc_phase runs that
            reference through AttitudeAutopilot and drives the vehicle via
            SET_ATTITUDE_TARGET, since SET_POSITION_TARGET_LOCAL_NED is
            confirmed broken in this sim (see the note on SEND_TYPE_MASK).
            """
            from guidance.mpc_guidance import MPCGuidance
            from guidance.guidance import CourseMap

            gates = gate_store.get_gates()
            if not gates:
                logger.warning("fly_gates: No gate data available — aborting.")
                return

            gates_sorted = sorted(gates, key=lambda g: g['gate_id'])

            # Sanity check before committing to a 100+ m/s climb: gate
            # telemetry and drone position telemetry should be in the same
            # local frame, so gate 0 should be within a sane distance of
            # where the drone actually is. If it isn't, the two are coming
            # from different coordinate frames this session (seen in
            # practice: gate north/east in the thousands while the drone
            # sits at ~(0,0,0)) and flying toward it just chases a phantom
            # target as fast as the velocity clamps allow.
            n, e, d = self._pos()
            g0 = gates_sorted[0]
            dist_to_gate0 = math.sqrt(
                (g0['north'] - n) ** 2 + (g0['east'] - e) ** 2 + (g0['down'] - d) ** 2
            )
            if dist_to_gate0 > 500.0:
                logger.error(
                    f"fly_gates: gate 0 is {dist_to_gate0:.0f}m from current position "
                    f"({n:.1f},{e:.1f},{d:.1f}) vs gate0=({g0['north']:.1f},{g0['east']:.1f},{g0['down']:.1f}). "
                    f"This is almost certainly a coordinate-frame mismatch between gate "
                    f"and drone telemetry, not a real course -- aborting rather than "
                    f"flying toward a phantom target. Try a full simulator restart."
                )
                self.hold_position()
                return

            # CourseMap wants (north, east, down) tuples + a cruise speed.
            # 'down' here is already the sim-corrected altitude value used
            # by the rest of fly_gates (-pos_d + 1.5), so guidance and
            # actuation share one consistent definition of gate altitude.
            course_map = CourseMap()
            course_map.load_from_list(
                [(g['north'], g['east'], g['down']) for g in gates_sorted],
                speed=8.0,   # cruise speed_mps fed into the speed scheduler
            )
            guidance = MPCGuidance(course_map=course_map)

            self._run_mpc_phase(guidance, timeout_per_gate_s * len(gates_sorted), "RACE")
            self.hold_position()


# ---------------------------------------------------------------------------
# Gate query helper
# ---------------------------------------------------------------------------

def print_gates(gate_store: GateStore, timeout_s: float = 10.0):
    logger.info(f"Waiting up to {timeout_s} s for track/gate data...")
    received = gate_store.received.wait(timeout=timeout_s)

    if not received:
        logger.warning("No gate data received within timeout.")
        logger.warning("  → Check that your MAVLinkRX is running and the sim is active.")
        return

    gates = gate_store.get_gates()
    print(f"\n{'='*55}")
    print(f"  TRACK DATA  —  {len(gates)} gates")
    print(f"{'='*55}")
    print(f"  {'ID':>4}  {'North':>8}  {'East':>8}  {'Down':>7}  {'W':>5}  {'H':>5}")
    print(f"  {'-'*50}")
    for g in gates:
        print(f"  {g['gate_id']:>4}  "
              f"{g['north']:>8.2f}  "
              f"{g['east']:>8.2f}  "
              f"{g['down']:>7.2f}  "
              f"{g['width']:>5.2f}  "
              f"{g['height']:>5.2f}")
    print(f"{'='*55}\n")
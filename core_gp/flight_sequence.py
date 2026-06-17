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

from pymavlink import mavutil

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

    TAKEOFF_ALT_M   = 0.1   # metres above arm point (positive up)
    TAKEOFF_SPEED   = 0.1   # m/s upward during climb
    FORWARD_SPEED   = 0.1   # m/s north during forward flight
    ALT_THRESHOLD_M = 0.01   # metres — "close enough" to target alt

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

    def takeoff(self, alt_m: float = None) -> float:
        alt_m       = alt_m     or self.TAKEOFF_ALT_M
        target_down = -alt_m  # NED: up is negative

        logger.info(f"Taking off to {alt_m} m AGL (attitude control)...")
        self._wait_for_telemetry()

        # Attitude rate mode: zero body rates (hold level), thrust above hover to climb.
        # 0.6 ≈ hover per reference; 0.75 should produce a steady climb.
        CLIMB_THRUST = 0.7

        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            self._send_attitude_target(0.0, 0.0, 0.0, CLIMB_THRUST)
            _, _, d = self._pos()
            current_alt = -d
            logger.info(f"  Alt: {current_alt:.2f} m  (target {alt_m:.1f} m)")
            if current_alt >= alt_m - self.ALT_THRESHOLD_M:
                logger.info("Takeoff complete.")
                break
            time.sleep(0.05)
        else:
            logger.warning("Takeoff timed out — proceeding anyway.")
            return target_down

        # Settle: hold altitude until vertical velocity is calm before starting gate flight.
        # Without this, the drone enters fly_gates with 7+ m/s upward velocity and
        # oscillates wildly around the first gate's altitude target.
        logger.info("Settling — waiting for vertical velocity < 0.3 m/s...")
        settle_end = time.monotonic() + 4.0
        while time.monotonic() < settle_end:
            _, _, vd = self._vel()
            _, _, d  = self._pos()
            alt_err  = (-d) - alt_m
            vert_vel = -vd
            thrust   = max(0.05, min(0.85, 0.13 - alt_err * 0.15 - vert_vel * 0.25))
            self._send_attitude_target(0.0, 0.0, 0.0, thrust)
            if abs(vd) < 0.3:
                logger.info("Vertical velocity settled.")
                break
            time.sleep(0.05)
        return target_down

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

    def hold_position(self):
        n, e, d = self._pos()
        type_mask = (
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
        )
        for _ in range(10):
            self._send_position_target(type_mask, n, e, d, 0.0, 0.0, 0.0)
            time.sleep(0.05)

    def fly_gates(self, gate_store: 'GateStore', timeout_per_gate_s: float = 30.0):
        """Navigate through every race gate in order using attitude rate control.

        Gate 'down' values in this sim are altitudes (positive = above ground),
        not NED Z — so target_alt = gate['down'] directly.
        """
        gates = gate_store.get_gates()
        if not gates:
            logger.warning("fly_gates: No gate data available — aborting.")
            return

        # Empirically: T≈0.10-0.13 maintains altitude at level flight. 0.13 is the setpoint.
        # MIN below 0.13 → drone descends; MAX 0.85 → fast climb for large altitude gaps.
        HOVER_THRUST       = 0.13
        THRUST_KP          = 0.15   # thrust per metre of altitude error (P)
        THRUST_KD          = 0.25   # thrust per m/s of vertical velocity (D — prevents overshoot)
        MAX_THRUST         = 0.85
        MIN_THRUST         = 0.05   # below hover so drone can actually descend when too high
        YAW_KP             = 2.0    # rad/s per radian of yaw error
        MAX_YAW_RATE       = 0.8
        GATE_RADIUS_M      = 1.5    # horizontal radius to count gate as passed
        GATE_HALF_HEIGHT   = .75   # half of 2.72m gate — altitude must be within this
        ALT_LEASH_M        = 5.0    # forward pitch suspended when this far from target alt
        MAX_SPEED_MPS      = 20.0   # hard speed cap — high enough to stay in fwd mode through gates
        DECEL_SPEED        = 2.0    # below this, stop active backward pitch (prevent reverse)
        FORWARD_PITCH      = -0.1   # rad/s: nose-down → forward
        # LEVEL_PITCH is applied only when altitude gap is large (between gates 2-5).
        # NOT applied for overspeed or misaligned — that caused gate-0 oscillation.
        LEVEL_PITCH        = +0.1   # rad/s: nose-up → active deceleration + allows climb

        for gate in sorted(gates, key=lambda g: g['gate_id']):
            gid      = gate['gate_id']
            g_n      = gate['north']
            g_e      = gate['east']
            gate_alt   = gate['down']   # actual gate center altitude — used for pass check
            target_alt = max(gate_alt, 0.3)  # altitude to fly; floor at 0.3m

            logger.info(f"--- Gate {gid}: N={g_n:.1f} E={g_e:.1f} alt={target_alt:.1f}m ---")

            deadline = time.monotonic() + timeout_per_gate_s
            while time.monotonic() < deadline:
                n, e, cur_d = self._pos()
                cur_alt = -cur_d  # NED Z → altitude

                dn   = g_n - n
                de   = g_e - e
                dist = math.sqrt(dn * dn + de * de)

                if dist < GATE_RADIUS_M and abs(cur_alt - gate_alt) < GATE_HALF_HEIGHT:
                    logger.info(f"  Gate {gid} passed! pos=({n:.1f},{e:.1f},{cur_alt:.1f}m)")
                    break

                # Bearing to gate in NED frame: atan2(east_delta, north_delta)
                desired_yaw = math.atan2(de, dn)
                current_yaw = self._yaw()
                yaw_err = (desired_yaw - current_yaw + math.pi) % (2 * math.pi) - math.pi
                # NOTE: positive yaw_rate = counterclockwise in this sim (sign flipped vs NED)
                yaw_rate = max(-MAX_YAW_RATE, min(MAX_YAW_RATE, -yaw_err * YAW_KP))

                # Altitude PD-controller: P on position error, D on vertical velocity.
                # D-term brakes upward momentum before overshooting the target altitude.
                alt_err = cur_alt - target_alt
                vn, ve, vd = self._vel()
                vert_vel = -vd  # NED z is down-positive; flip so upward = positive
                thrust  = max(MIN_THRUST, min(MAX_THRUST,
                              HOVER_THRUST - alt_err * THRUST_KP - vert_vel * THRUST_KD))

                speed_h  = math.sqrt(vn * vn + ve * ve)
                aligned  = abs(yaw_err) < math.radians(30)
                near_alt = abs(alt_err) < ALT_LEASH_M
                go       = aligned and near_alt and speed_h < MAX_SPEED_MPS
                if go:
                    pitch_rate = FORWARD_PITCH          # nose-down: fly toward gate
                elif not near_alt and speed_h > DECEL_SPEED:
                    pitch_rate = LEVEL_PITCH            # nose-up: shed speed while gaining alt
                else:
                    pitch_rate = 0.0                    # coast: turning or near speed cap

                self._send_attitude_target(0.0, pitch_rate, yaw_rate, thrust)
                logger.info(
                    f"  G{gid}: pos=({n:.1f},{e:.1f},{cur_alt:.1f}m) dist={dist:.1f}m "
                    f"yaw={math.degrees(current_yaw):.0f}° yawerr={math.degrees(yaw_err):.0f}° "
                    f"spd={speed_h:.1f} vz={vert_vel:.1f} T={thrust:.2f} "
                    f"{'fwd' if go else ('lvl' if not near_alt and speed_h > DECEL_SPEED else 'cst')}"
                )
                time.sleep(0.05)
            else:
                logger.warning(f"Gate {gid} navigation timed out!")

        logger.info("All gates navigated.")
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
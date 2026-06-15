"""
flight_sequence.py
------------------
Basic flight workflow for the Anduril AI Grand Prix sim:
  1. Connect & wait for heartbeat
  2. Arm
  3. Takeoff to a target altitude (NED -5 m)
  4. Fly forward at a fixed speed
  5. Query and print gate locations from track data

Plug this into your existing main() or run standalone.
"""

import struct
import time
import threading
import logging

from pymavlink import mavutil
from comms.mavlink_bridge import MAVLinkBridge, ControlCommand, TelemetryState

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


# ---------------------------------------------------------------------------
# Gate Store  —  populated by the chunked ENCAPSULATED_DATA / track packets
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
# Patch MAVLinkRX to populate GateStore
# ---------------------------------------------------------------------------
# Import your existing MAVLinkRX and subclass it so we can capture gate data
# without touching the original file.

from mavlink_rx import MAVLinkRX  # adjust import path as needed

class GateAwareMAVLinkRX(MAVLinkRX):
    """Extends MAVLinkRX to forward parsed track data into a GateStore."""

    def __init__(self, mavlink_connection, data, gate_store: GateStore):
        super().__init__(mavlink_connection, data)
        self._gate_store = gate_store

    def on_track_data(self, payload: bytes):
        """
        Called once all chunks for a transfer are assembled.
        Parses gate positions and loads them into GateStore.

        Packet layout (little-endian):
          [2B] num_gates
          per gate (38 bytes each):
            [2B] gate_id
            [4B] pos_ned_x (north)
            [4B] pos_ned_y (east)
            [4B] pos_ned_z (down)
            [4B] orient_w
            [4B] orient_x
            [4B] orient_y
            [4B] orient_z
            [4B] width
            [4B] height
        """
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
                "gate_id":    gate_id,
                "north":      pos_n,
                "east":       pos_e,
                "down":       pos_d,
                "orient":     (ow, ox, oy, oz),
                "width":      width,
                "height":     height,
            })

        self._gate_store.set_gates(gates)


# ---------------------------------------------------------------------------
# Flight Controller
# ---------------------------------------------------------------------------

class FlightController:
    """
    Thin wrapper around MAVLinkBridge that adds a
    blocking arm-takeoff-command workflow.
    """

    TAKEOFF_ALT_M   = 5.0    # metres above arm point (positive up)
    TAKEOFF_SPEED   = 2.0    # m/s upward during climb
    FORWARD_SPEED   = 4.0    # m/s north during forward flight
    ALT_THRESHOLD_M = 0.4    # metres — "close enough" to target alt

    def __init__(self, bridge: MAVLinkBridge):
        self.bridge = bridge

    # ------------------------------------------------------------------
    # High-level sequence
    # ------------------------------------------------------------------

    def connect(self):
        """Connect and block until heartbeat."""
        self.bridge.connect()
        logger.info("Waiting for connection...")
        for _ in range(40):
            if self.bridge.is_connected():
                logger.info("Connected.")
                return
            time.sleep(0.25)
        raise RuntimeError("No heartbeat after 10 s — is the sim running?")

    def arm(self):
        """Send arm command and wait for the drone to report armed."""
        logger.info("Arming...")
        self.bridge.arm()
        # Give the sim a moment to process the arm command
        time.sleep(1.0)
        logger.info("Arm command sent.")

    def takeoff(self, alt_m: float = None, speed_mps: float = None):
        """
        Command a vertical climb to alt_m (above arm point).
        Blocks until within ALT_THRESHOLD_M of target altitude.
        """
        alt_m    = alt_m    or self.TAKEOFF_ALT_M
        speed_mps = speed_mps or self.TAKEOFF_SPEED
        target_down = -alt_m   # NED: down is negative

        logger.info(f"Taking off to {alt_m} m AGL (down={target_down})...")

        cmd = ControlCommand(
            # Stay at current N/E, climb to target altitude
            target_north = self.bridge.get_state().north,
            target_east  = self.bridge.get_state().east,
            target_down  = target_down,

            # Climb at specified speed, no lateral movement
            target_vn = 0.0,
            target_ve = 0.0,
            target_vd = -speed_mps,   # NED: negative = up

            target_yaw = 0.0,

            # Ignore acceleration; control position + velocity
            type_mask = (
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
            ),
        )

        # Send at ~20 Hz while climbing
        deadline = time.monotonic() + 30.0   # safety timeout
        while time.monotonic() < deadline:
            self.bridge.send_command(cmd)
            state = self.bridge.get_state()
            current_alt = -state.down          # positive up
            logger.info(f"  Alt: {current_alt:.2f} m  (target {alt_m:.1f} m)")

            if current_alt >= alt_m - self.ALT_THRESHOLD_M:
                logger.info("Takeoff complete.")
                return

            time.sleep(0.05)

        logger.warning("Takeoff timed out — proceeding anyway.")

    def fly_forward(self, speed_mps: float = None, duration_s: float = 5.0):
        """
        Command forward flight (positive north) at speed_mps for duration_s seconds.
        Maintains current altitude.
        """
        speed_mps = speed_mps or self.FORWARD_SPEED
        state     = self.bridge.get_state()
        hold_down = state.down   # keep current altitude

        logger.info(f"Flying forward at {speed_mps} m/s for {duration_s} s...")

        cmd = ControlCommand(
            # Don't constrain position — let velocity drive it
            target_north = 0.0,
            target_east  = 0.0,
            target_down  = hold_down,

            target_vn = speed_mps,
            target_ve = 0.0,
            target_vd = 0.0,

            target_yaw = state.yaw,

            # Ignore position X/Y; control velocity + altitude hold
            type_mask = (
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
            ),
        )

        t0 = time.monotonic()
        while time.monotonic() - t0 < duration_s:
            self.bridge.send_command(cmd)
            state = self.bridge.get_state()
            logger.info(
                f"  N={state.north:.1f}  E={state.east:.1f}  "
                f"Alt={-state.down:.1f} m  Vn={state.vn:.1f} m/s"
            )
            time.sleep(0.05)

        logger.info("Forward flight complete — holding position.")
        self.hold_position()

    def hold_position(self):
        """Command a position hold at current location."""
        state = self.bridge.get_state()
        cmd = ControlCommand(
            target_north = state.north,
            target_east  = state.east,
            target_down  = state.down,
            target_vn    = 0.0,
            target_ve    = 0.0,
            target_vd    = 0.0,
            target_yaw   = state.yaw,
            type_mask    = (
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
            ),
        )
        for _ in range(10):
            self.bridge.send_command(cmd)
            time.sleep(0.05)


# ---------------------------------------------------------------------------
# Gate query helper
# ---------------------------------------------------------------------------

def print_gates(gate_store: GateStore, timeout_s: float = 10.0):
    """
    Wait for gate data to arrive, then pretty-print it.
    Gate data arrives automatically once the sim sends the
    DATA_TRANSMISSION_HANDSHAKE + ENCAPSULATED_DATA chunks.
    """
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


# ---------------------------------------------------------------------------
# Entry Point  —  replace / call from your existing main()
# ---------------------------------------------------------------------------

def run_basic_flight():
    """
    Standalone entry point.
    Mirrors your existing main() setup but wires in the gate store
    and flight controller.
    """
    from pymavlink import mavutil as mu

    SIM_IP   = "127.0.0.1"
    SIM_PORT = 14550

    # --- Gate store shared across threads ---
    gate_store = GateStore()

    # --- MAVLink connection (shared between RX and bridge) ---
    conn = mu.mavlink_connection(f"udpin:{SIM_IP}:{SIM_PORT}")
    conn.wait_heartbeat()
    logger.info(f"Heartbeat from system={conn.target_system}")

    # --- Start gate-aware RX thread ---
    shared_data = {}
    mavlink_rx = GateAwareMAVLinkRX.create_mavlink_rx(conn, shared_data, gate_store)

    # --- Flight bridge (uses same UDP port via MAVLinkBridge internal conn) ---
    bridge = MAVLinkBridge(host=SIM_IP, port=SIM_PORT)
    fc     = FlightController(bridge)

    try:
        # 1. Connect
        fc.connect()

        # 2. Print gates as soon as they arrive (non-blocking — runs in background)
        gate_thread = threading.Thread(
            target=print_gates,
            args=(gate_store, 15.0),
            daemon=True
        )
        gate_thread.start()

        # 3. Arm
        fc.arm()

        # 4. Takeoff to 5 m
        fc.takeoff(alt_m=5.0)

        # 5. Wait for gate data if not yet received (nice to have before racing)
        gate_thread.join(timeout=5.0)

        # 6. Fly forward for 5 seconds
        fc.fly_forward(speed_mps=4.0, duration_s=5.0)

    except KeyboardInterrupt:
        logger.info("Interrupted — holding position.")
        fc.hold_position()
    finally:
        bridge.close()
        mavlink_rx.get_thread_for_join().join(timeout=1.0)


if __name__ == "__main__":
    run_basic_flight()
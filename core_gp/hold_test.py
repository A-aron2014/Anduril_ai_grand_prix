"""
Isolation test: arm, then hold at ground level with an explicit
zero-velocity command for N seconds -- nothing else. No takeoff, no
guidance, no MPC. If altitude/velocity runs away anyway, the cause is
upstream of all of our flight logic (sim-side behavior, missing mode
setup, etc.), not anything in flight_sequence.py/guidance/mpc_guidance.

Run directly: python hold_test.py
"""

import time
import logging

from pymavlink import mavutil

from setup import setup_components
from flight_sequence import FlightController

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Matches PyAIPilotExample/controller.py's VELOCITY_POSITION_MASK exactly --
# the official sample never sends position, only velocity (+ ignores accel,
# yaw, yaw_rate). Our own SEND_TYPE_MASK only ignores acceleration and always
# sends position+velocity together, which is the one concrete difference
# from the known-reference message format left to rule out.
OFFICIAL_VELOCITY_ONLY_MASK = (
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
)


def main():
    SIM_IP = "127.0.0.1"
    SIM_PORT = 14550

    system_boot_ms = int(time.time() * 1000)
    shared_data = {}

    components = setup_components(shared_data, system_boot_ms, SIM_IP, SIM_PORT)
    sim_conn = components['sim_conn']

    fc = FlightController(sim_conn, shared_data, system_boot_ms)

    fc.send_sim_reset()
    time.sleep(6.0)

    fc.arm()

    fc._wait_for_telemetry()
    n, e, d = fc._pos()
    logger.info(f"Post-arm position (should be ~ground level): ({n:.2f},{e:.2f},{d:.2f})")

    logger.info("Holding with a velocity-only zero command for 10s, matching "
                "PyAIPilotExample's exact type_mask (position/accel/yaw all "
                "IGNORE'd) -- no takeoff, no guidance, nothing else.")
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        cur_n, cur_e, cur_d = fc._pos()
        vn, ve, vd = fc._vel()
        fc._send_position_target(
            type_mask=OFFICIAL_VELOCITY_ONLY_MASK,
            x=0.0, y=0.0, z=0.0,   # ignored per type_mask, matching the official sample
            vx=0.0, vy=0.0, vz=0.0,
        )
        logger.info(
            f"HOLD_TEST: pos=({cur_n:.2f},{cur_e:.2f},{cur_d:.2f}) "
            f"vel=({vn:.2f},{ve:.2f},{vd:.2f}) "
            f"race_started={shared_data.get('race_started')} "
            f"active_gate={shared_data.get('active_gate')} "
            f"race_finished={shared_data.get('race_finished')} "
            f"base_mode={shared_data.get('base_mode')} "
            f"custom_mode={shared_data.get('custom_mode')}"
        )
        time.sleep(0.05)

    logger.info("Hold test complete.")


if __name__ == '__main__':
    main()

"""
Regression test for takeoff_mpc(): arm, climb to a target altitude via
the MPC + AttitudeAutopilot path, then hold and log telemetry throughout
the climb and the post-climb hold so a real altitude/position overshoot
or drift is visible, not just "did it finish."

This is the first real test of takeoff_mpc() since the AttitudeAutopilot
sign-convention fixes (see project_attitude_axis_sign_conventions memory) --
takeoff_mpc/_run_mpc_phase share the exact same actuation path hold_test.py
already validated, but this is a moving target (climbing to climb_alt_m)
rather than a static hold, and a fresh AttitudeAutopilot/MPCGuidance
instance, so it's worth checking in isolation before chaining into
fly_gates().

Run directly: python takeoff_test.py
"""

import time
import logging

from setup import setup_components
from flight_sequence import FlightController

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def main():
    SIM_IP = "127.0.0.1"
    SIM_PORT = 14550

    system_boot_ms = int(time.time() * 1000)
    shared_data = {}

    components = setup_components(shared_data, system_boot_ms, SIM_IP, SIM_PORT)
    sim_conn   = components['sim_conn']
    ts_loop    = components['ts_loop']
    mavlink_rx = components['mavlink_rx']
    vision_rx  = components['vision_rx']

    fc = FlightController(sim_conn, shared_data, system_boot_ms)

    try:
        fc.send_sim_reset()
        time.sleep(6.0)

        fc.arm()
        fc._wait_for_telemetry()
        n, e, d = fc._pos()
        roll, pitch, yaw = fc._roll(), fc._pitch(), fc._yaw()
        logger.info(f"Post-arm position (should be ~ground level): ({n:.2f},{e:.2f},{d:.2f})")
        logger.info(f"Post-arm attitude (should be ~level): roll={roll:.3f} pitch={pitch:.3f} yaw={yaw:.3f}")

        logger.info("Climbing to 0.5m AGL via takeoff_mpc()...")
        fc.takeoff_mpc(climb_alt_m=0.5, timeout_s=15.0)

        n, e, d = fc._pos()
        vn, ve, vd = fc._vel()
        logger.info(
            f"Post-takeoff: pos=({n:.2f},{e:.2f},{d:.2f}) alt={-d:.2f}m "
            f"vel=({vn:.2f},{ve:.2f},{vd:.2f})"
        )

        logger.info("Holding for 5s to check post-climb stability...")
        fc.hold_position(duration_s=5.0)

        n, e, d = fc._pos()
        vn, ve, vd = fc._vel()
        logger.info(
            f"Takeoff test complete. Final pos=({n:.2f},{e:.2f},{d:.2f}) alt={-d:.2f}m "
            f"vel=({vn:.2f},{ve:.2f},{vd:.2f})"
        )

    except KeyboardInterrupt:
        print("Interrupted — holding position.")
        fc.hold_position()

    finally:
        for component in (ts_loop, mavlink_rx, vision_rx):
            try:
                t = component.get_thread_for_join()
                if t is not None:
                    t.join(timeout=1.0)
            except Exception:
                pass
        print("Exited cleanly.")


if __name__ == '__main__':
    main()

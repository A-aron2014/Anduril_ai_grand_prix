"""
Pure-observation test: arm, then send NOTHING -- no hold, no velocity-zero,
no position target, nothing -- and just log telemetry for a long window.

hold_test.py proved that an explicit, correctly-formed stop command
(position=current, velocity=0,0,0) is being sent and is having zero effect
on the climb for at least 5s post-arm. Before guessing at another trigger
condition, this script answers two open questions with no interference
from our own commands:
  1. Does the climb ever decelerate and settle on its own, and when?
  2. Does base_mode/custom_mode ever change (a real "control handed back"
     signal), or is it static the whole time?

Run directly: python launch_profile_test.py
Let it run the full duration -- do NOT Ctrl-C early, we need the full profile.
"""

import time
import logging

from setup import setup_components
from flight_sequence import FlightController

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def main(duration_s: float = 60.0):
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

        logger.info(f"Observing telemetry for {duration_s}s with ZERO commands sent -- "
                    "no hold, no setpoints, nothing.")

        last_base_mode = shared_data.get('base_mode')
        last_custom_mode = shared_data.get('custom_mode')
        peak_speed = 0.0
        peak_alt = 0.0

        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            n, e, d = fc._pos()
            vn, ve, vd = fc._vel()
            alt = -d
            speed = (vn**2 + ve**2 + vd**2) ** 0.5
            peak_speed = max(peak_speed, speed)
            peak_alt = max(peak_alt, alt)

            base_mode = shared_data.get('base_mode')
            custom_mode = shared_data.get('custom_mode')
            if base_mode != last_base_mode or custom_mode != last_custom_mode:
                logger.warning(
                    f"MODE CHANGE: base_mode {last_base_mode}->{base_mode} "
                    f"custom_mode {last_custom_mode}->{custom_mode}"
                )
                last_base_mode, last_custom_mode = base_mode, custom_mode

            logger.info(
                f"PROFILE: t={duration_s - (deadline - time.monotonic()):.1f}s "
                f"pos=({n:.1f},{e:.1f},{d:.1f}) alt={alt:.1f}m "
                f"vel=({vn:.1f},{ve:.1f},{vd:.1f}) speed={speed:.1f} "
                f"peak_alt={peak_alt:.1f} peak_speed={peak_speed:.1f} "
                f"base_mode={base_mode} custom_mode={custom_mode} "
                f"active_gate={shared_data.get('active_gate')}"
            )
            time.sleep(0.1)

        logger.info(f"Profile complete. peak_alt={peak_alt:.1f}m peak_speed={peak_speed:.1f}m/s")

    except KeyboardInterrupt:
        print("Interrupted.")

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

"""
Isolation test #7: does the cadence-triggered runaway also happen on the
OTHER documented control channel, SET_ATTITUDE_TARGET, or is it specific
to SET_POSITION_TARGET_LOCAL_NED?

Six prior tests (hold_test.py, launch_profile_test.py, single_shot_test.py,
frozen_target_test.py, low_rate_test.py, rate_threshold_test.py) showed
SET_POSITION_TARGET_LOCAL_NED streamed faster than ~1 message per 1-2s
causes an unbounded, content-independent climb to 150m+ within seconds,
with zero response to position/velocity hold commands. The technical
spec's "Example Control Session" describes streaming control commands as
the normal intended flow with no caveat about this.

This sends SET_ATTITUDE_TARGET at 20Hz (well above the unsafe threshold
found for position targets) commanding a fixed, sub-hover-ish thrust with
zero rates, and just watches whether the climb is physically sane (bounded,
roughly proportional to commanded thrust) or exhibits the same unbounded
runaway -- which would mean the bug is in core control handling generally,
not specific to position-target parsing.

Run directly: python attitude_target_test.py
Let it run the full duration -- do NOT Ctrl-C early.
"""

import time
import logging

from pymavlink import mavutil

from setup import setup_components
from flight_sequence import FlightController

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

THRUST = 0.5  # sub-max, roughly hover-ish per existing controller.py reference values
DURATION_S = 10.0
SEND_INTERVAL_S = 0.05  # 20 Hz -- well above the unsafe threshold for position targets


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

        logger.info(f"Streaming SET_ATTITUDE_TARGET at {1/SEND_INTERVAL_S:.0f}Hz, "
                    f"thrust={THRUST}, zero rates, for {DURATION_S}s.")

        peak_speed = 0.0
        peak_alt = 0.0
        next_send = time.monotonic()
        deadline = time.monotonic() + DURATION_S

        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_send:
                fc._send_attitude_target(
                    roll_rate=0.0, pitch_rate=0.0, yaw_rate=0.0, thrust=THRUST,
                )
                next_send += SEND_INTERVAL_S

            n, e, d = fc._pos()
            vn, ve, vd = fc._vel()
            alt = -d
            speed = (vn**2 + ve**2 + vd**2) ** 0.5
            peak_speed = max(peak_speed, speed)
            peak_alt = max(peak_alt, alt)

            logger.info(
                f"ATTITUDE: pos=({n:.1f},{e:.1f},{d:.1f}) alt={alt:.1f}m "
                f"vel=({vn:.1f},{ve:.1f},{vd:.1f}) speed={speed:.1f} "
                f"peak_alt={peak_alt:.1f} peak_speed={peak_speed:.1f} "
                f"base_mode={shared_data.get('base_mode')} custom_mode={shared_data.get('custom_mode')}"
            )
            time.sleep(0.05)

        logger.info(f"Done. peak_alt={peak_alt:.1f}m peak_speed={peak_speed:.1f}m/s "
                    f"-- {'RAN AWAY' if peak_alt > 20.0 else 'BOUNDED/SANE'}")

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

"""
Isolation test #5: arm, then send the hold setpoint at a much lower rate
-- once every 2s instead of every 50ms -- for 5 sends (10s total), with
telemetry logged continuously in between so we can see exactly what
happens around each send.

frozen_target_test.py proved content/freshness isn't the trigger: even a
perfectly unchanging setpoint streamed at 20Hz ran away to 255m in 10s.
single_shot_test.py proved one message alone is harmless. This test
checks whether it's specifically about RATE: if sending well below
whatever threshold the sim needs to "lock on" to a stream avoids the
runaway, that's a usable workaround (rate-limit our own setpoints).

Run directly: python low_rate_test.py
Let it run the full duration -- do NOT Ctrl-C early.
"""

import time
import logging

from pymavlink import mavutil

from setup import setup_components
from flight_sequence import FlightController

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

HOLD_TYPE_MASK = (
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
)

SEND_INTERVAL_S = 2.0
NUM_SENDS = 5


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

        n0, e0, d0 = fc._pos()
        logger.info(f"Sending hold setpoint at ({n0:.2f},{e0:.2f},{d0:.2f}) "
                    f"once every {SEND_INTERVAL_S}s, {NUM_SENDS} times total.")

        peak_speed = 0.0
        peak_alt = 0.0
        next_send = time.monotonic()
        sends_done = 0
        deadline = time.monotonic() + SEND_INTERVAL_S * NUM_SENDS + 2.0

        while time.monotonic() < deadline:
            if sends_done < NUM_SENDS and time.monotonic() >= next_send:
                fc._send_position_target(
                    type_mask=HOLD_TYPE_MASK,
                    x=n0, y=e0, z=d0,
                    vx=0.0, vy=0.0, vz=0.0,
                )
                sends_done += 1
                next_send += SEND_INTERVAL_S
                logger.warning(f"--- SENT setpoint #{sends_done}/{NUM_SENDS} ---")

            n, e, d = fc._pos()
            vn, ve, vd = fc._vel()
            alt = -d
            speed = (vn**2 + ve**2 + vd**2) ** 0.5
            peak_speed = max(peak_speed, speed)
            peak_alt = max(peak_alt, alt)

            logger.info(
                f"LOW_RATE: pos=({n:.1f},{e:.1f},{d:.1f}) alt={alt:.1f}m "
                f"vel=({vn:.1f},{ve:.1f},{vd:.1f}) speed={speed:.1f} "
                f"peak_alt={peak_alt:.1f} peak_speed={peak_speed:.1f} "
                f"base_mode={shared_data.get('base_mode')} custom_mode={shared_data.get('custom_mode')}"
            )
            time.sleep(0.1)

        logger.info(f"Done. peak_alt={peak_alt:.1f}m peak_speed={peak_speed:.1f}m/s "
                    f"-- {'RAN AWAY' if peak_alt > 1.0 else 'STAYED PUT'}")

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

"""
Isolation test #3: arm, send exactly ONE SET_POSITION_TARGET_LOCAL_NED
(hold at current position, zero velocity), then send NOTHING further --
just observe telemetry for a long window.

launch_profile_test.py proved the sim does NOT launch on its own: with
zero commands sent, the drone sat at pos=(0,0,0) for 25+ seconds straight,
no movement, no mode change. hold_test.py separately showed the climb
starts within ~50ms of the FIRST setpoint message being sent, regardless
of its type_mask/content. Those two facts together mean the runaway is
triggered by *sending the message*, not by what's in it.

This test isolates whether that's a one-shot trigger (climbs on its own
after a single message, independent of further input) or something tied
to the continuous stream (only climbs while we keep sending).

Run directly: python single_shot_test.py
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


def main(observe_s: float = 30.0):
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
        logger.info(f"Pre-send position: ({n:.2f},{e:.2f},{d:.2f})")
        logger.info("Sending exactly ONE hold setpoint (pos=current, vel=0,0,0), "
                    "then going silent -- no more messages of any kind.")
        fc._send_position_target(
            type_mask=HOLD_TYPE_MASK,
            x=n, y=e, z=d,
            vx=0.0, vy=0.0, vz=0.0,
        )

        last_base_mode = shared_data.get('base_mode')
        last_custom_mode = shared_data.get('custom_mode')
        peak_speed = 0.0
        peak_alt = 0.0

        deadline = time.monotonic() + observe_s
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
                f"SINGLE_SHOT: pos=({n:.1f},{e:.1f},{d:.1f}) alt={alt:.1f}m "
                f"vel=({vn:.1f},{ve:.1f},{vd:.1f}) speed={speed:.1f} "
                f"peak_alt={peak_alt:.1f} peak_speed={peak_speed:.1f} "
                f"base_mode={base_mode} custom_mode={custom_mode}"
            )
            time.sleep(0.1)

        logger.info(f"Done. peak_alt={peak_alt:.1f}m peak_speed={peak_speed:.1f}m/s "
                    f"-- {'CLIMBED ON ITS OWN' if peak_alt > 1.0 else 'STAYED PUT'}")

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

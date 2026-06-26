"""
pitch_vertical_coupling_test.py
--------------------------------
Isolates whether pitch (commanded for horizontal acceleration) is coupling
into unwanted vertical motion in this sim, independent of guidance/MPC.

Context: every race-leg log so far shows 'down' drifting by 15-25m over the
first 3-5s of a leg -- exactly the window where pitch is largest (ramping up
to speed toward cruise). The U0_RAW/H-CLIP/V-CLIP diagnostics in
MPCGuidance.compute() already showed the vertical channel gets full,
correctly-signed acceleration authority the whole time and the drift still
happens -- ruling out the MPC. This test rules MPC IN or OUT definitively by
building a leg with a large horizontal distance (so pitch ramps up the same
way) but ZERO required altitude change (same start/end 'down'), then
watching whether 'down'/vd drift anyway. If they do, the cause is pitch
itself (or forward-speed) coupling into vertical motion downstream of
guidance -- in AttitudeAutopilot or the sim's actuator response -- not in
trajectory planning. We don't have access to the sim's actuator model, so
this empirical isolation is the only way to localize it without that.

Run directly: python pitch_vertical_coupling_test.py
"""

import time
import logging

from setup import setup_components
from flight_sequence import FlightController
from guidance.mpc_guidance import MPCGuidance, MPCConfig
from guidance.guidance import CourseMap

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

HORIZ_DISTANCE_M = 20.0   # matches the magnitude of a real leg's speed-up
CRUISE_SPEED_MPS = 8.0    # same as fly_gates' cruise speed -- reproduce the same pitch ramp


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
        fc.takeoff_mpc(climb_alt_m=0.5)
        fc.hold_position()

        n, e, d = fc._pos()
        logger.info(
            f"Building flat horizontal-only leg: ({n:.1f},{e:.1f},{d:.1f}) -> "
            f"({n - HORIZ_DISTANCE_M:.1f},{e:.1f},{d:.1f})  -- same 'down' both ends, "
            f"zero altitude change commanded."
        )

        course_map = CourseMap()
        course_map.load_from_list(
            [(n - HORIZ_DISTANCE_M, e, d)],
            speed=CRUISE_SPEED_MPS,
        )
        guidance = MPCGuidance(course_map=course_map, config=MPCConfig())

        logger.info("Running flat leg -- watch ATT(pitch) vs POS(down)/ACTUAL_VEL(vd) below.")
        fc._run_mpc_phase(guidance, timeout_s=15.0, phase_label="PITCH_VS_VERTICAL")

        n2, e2, d2 = fc._pos()
        vn, ve, vd = fc._vel()
        logger.info(
            f"Flat-leg test complete. start_down={d:.2f} end_down={d2:.2f} "
            f"down_drift={d2 - d:+.2f}m (expected ~0 if pitch isn't coupling into vertical) "
            f"final_vel=({vn:.2f},{ve:.2f},{vd:.2f})"
        )

        fc.hold_position()

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

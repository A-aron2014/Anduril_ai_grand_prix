"""
pitch_lateral_coupling_test.py
--------------------------------
Isolates whether pitch (commanded for horizontal acceleration) is coupling
into unwanted lateral (east) motion in this sim, independent of guidance/MPC.

Context: every race-leg log so far shows 'east' drifting several meters over
a single ~20m/3s leg -- and raising max_tilt (2026-06-27) to give the lateral
(roll) loop more correction authority made the drift WORSE, not better,
because it also let pitch climb higher. That rules out "not enough roll
authority" and points at pitch itself (or the forward speed/pitch angle it
produces) coupling into lateral motion downstream of guidance -- the same
family of bug as the already-confirmed pitch->vertical coupling
(pitch_vertical_coupling_test.py), but on the east axis instead of down.

This test rules MPC/guidance IN or OUT the same way the vertical test did:
build a leg with a large horizontal (south) distance, so pitch ramps up the
same way a race leg does, but ZERO required east change (same start/end
'east'), then watch whether 'east'/ve drifts anyway despite the lateral loop
actively fighting it the whole time. If it does, the cause is pitch/forward-
speed coupling into lateral motion in AttitudeAutopilot or the sim's actuator
response -- not in trajectory planning.

Run directly: python pitch_lateral_coupling_test.py
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
            f"Building flat south-only leg: ({n:.1f},{e:.1f},{d:.1f}) -> "
            f"({n - HORIZ_DISTANCE_M:.1f},{e:.1f},{d:.1f})  -- same 'east' both ends, "
            f"zero lateral displacement commanded."
        )

        course_map = CourseMap()
        course_map.load_from_list(
            [(n - HORIZ_DISTANCE_M, e, d)],
            speed=CRUISE_SPEED_MPS,
        )
        guidance = MPCGuidance(course_map=course_map, config=MPCConfig())

        logger.info("Running flat leg -- watch ATT(pitch) vs POS(east)/ACTUAL_VEL(ve) below.")
        fc._run_mpc_phase(guidance, timeout_s=15.0, phase_label="PITCH_VS_LATERAL")

        n2, e2, d2 = fc._pos()
        vn, ve, vd = fc._vel()
        logger.info(
            f"Flat-leg test complete. start_east={e:.2f} end_east={e2:.2f} "
            f"east_drift={e2 - e:+.2f}m (expected ~0 if pitch isn't coupling into lateral) "
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

from time import time
import logging


def _vision_shadow_compare_loop(shared_data, gate_store, is_running, period_s: float = 1.0):
    """
    Shadow mode: logs the camera-derived gate pose alongside the
    sim-provided ground-truth gate position, transformed into the same
    NED frame, so detection quality can be judged against real footage
    once the sim is running. Vision is NOT fed into guidance yet — state
    estimation/fusion is a follow-up step; this is purely for validation.
    """
    import time as _time
    import numpy as np
    from core.state_estimator import VehicleState

    start = _time.monotonic()
    warned_no_frames = False

    while is_running():
        _time.sleep(period_s)

        if not warned_no_frames and shared_data.get('vision_frame_count', 0) == 0 and _time.monotonic() - start > 5.0:
            logging.getLogger(__name__).warning(
                "VISION_SHADOW: no camera frames received after 5s — check the sim is "
                "streaming on UDP 5600 and nothing else is bound to that port."
            )
            warned_no_frames = True

        # position_body is now populated for every confident detection, not
        # just clean PnP solves (see perception.py) -- log both, tagged
        # with pose_valid, instead of only ever seeing the high-fidelity
        # subset (which was also previously the only thing reaching
        # guidance, see flight_sequence.py).
        if shared_data.get('vision_gate_position_body') is None:
            continue

        gates = gate_store.get_gates()
        if not gates:
            continue

        # active_gate is published by FlightController._run_mpc_phase as
        # the MPC's current leg index -- previously nothing ever wrote
        # this, so it silently stayed at the default of 0 and every
        # comparison below was against gate 0's truth position regardless
        # of which gate the drone had actually progressed to.
        active_idx = shared_data.get('active_gate', 0)
        gates_sorted = sorted(gates, key=lambda g: g['gate_id'])
        truth_gate = gates_sorted[min(active_idx, len(gates_sorted) - 1)]

        state = VehicleState(
            north=shared_data.get('pos_x', 0.0),
            east=shared_data.get('pos_y', 0.0),
            down=shared_data.get('pos_z', 0.0),
            roll=shared_data.get('roll', 0.0),
            pitch=shared_data.get('pitch', 0.0),
            yaw=shared_data.get('yaw', 0.0),
        )

        position_body = np.asarray(shared_data['vision_gate_position_body'])
        vision_gate_ned = state.position_ned() + state.body_to_ned_R() @ position_body
        truth_ned = np.array([truth_gate['north'], truth_gate['east'], truth_gate['down']])

        error = float(np.linalg.norm(vision_gate_ned - truth_ned))
        logging.getLogger(__name__).info(
            f"VISION_SHADOW: active_gate={active_idx} gate_id={truth_gate['gate_id']} "
            f"pose_valid={shared_data.get('vision_gate_pose_valid', False)} "
            f"vision_gate_ned=({vision_gate_ned[0]:.2f},{vision_gate_ned[1]:.2f},{vision_gate_ned[2]:.2f}) "
            f"truth_gate_ned=({truth_ned[0]:.2f},{truth_ned[1]:.2f},{truth_ned[2]:.2f}) "
            f"error_m={error:.2f} confidence={shared_data.get('vision_gate_confidence', 0.0):.2f}"
        )


def main():
    import time
    import threading
    from setup import setup_components
    from flight_sequence import FlightController, GateStore, print_gates

    SIM_IP   = "127.0.0.1"
    SIM_PORT = 14550

    system_boot_ms = int(time.time() * 1000)
    shared_data = {}

    # One connection shared by everything — no second socket opened anywhere
    components = setup_components(shared_data, system_boot_ms, SIM_IP, SIM_PORT)
    sim_conn   = components['sim_conn']
    controller = components['controller']
    ts_loop    = components['ts_loop']
    vision_rx  = components['vision_rx']
    mavlink_rx = components['mavlink_rx']

    # Wire gate data into the existing RX instance via monkey-patch
    gate_store = GateStore()
    
    def _on_track_data(payload: bytes):
        import struct
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
                "north": pos_n, "east": pos_e , "down": pos_d ,
                "orient": (ow, ox, oy, oz),
                "width": width, "height": height,
            })
        gate_store.set_gates(gates)

    mavlink_rx.on_track_data = _on_track_data

    # FlightController uses sim_conn + shared_data directly — no second UDP socket
    fc = FlightController(sim_conn, shared_data, system_boot_ms)
    stop_event = threading.Event()
    try:
        threading.Thread(
            target=print_gates,
            args=(gate_store, 15.0),
            daemon=True
        ).start()

        threading.Thread(
            target=_vision_shadow_compare_loop,
            args=(shared_data, gate_store, lambda: not stop_event.is_set()),
            daemon=True
        ).start()

        fc.send_sim_reset()
        time.sleep(6.0)          # sim needs ~5s after reset before race can start

        fc.arm()
        fc.takeoff_mpc(climb_alt_m=0.5)   # stay low — gate 0 center is at altitude ~0m
        fc.fly_gates(gate_store)

    except KeyboardInterrupt:
        print("Interrupted — holding position.")
        fc.hold_position()
    finally:
        stop_event.set()
        # get_thread_for_join() may return None if a component never started
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
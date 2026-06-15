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
                "north": pos_n, "east": pos_e, "down": pos_d,
                "orient": (ow, ox, oy, oz),
                "width": width, "height": height,
            })
        gate_store.set_gates(gates)

    mavlink_rx.on_track_data = _on_track_data

    # FlightController uses sim_conn + shared_data directly — no second UDP socket
    fc = FlightController(sim_conn, shared_data, system_boot_ms)

    try:
        threading.Thread(
            target=print_gates,
            args=(gate_store, 15.0),
            daemon=True
        ).start()

        fc.arm()
        target_down = fc.takeoff(alt_m=5.0)
        fc.fly_forward(speed_mps=4.0, duration_s=5.0, target_down=target_down)

    except KeyboardInterrupt:
        print("Interrupted — holding position.")
        fc.hold_position()
    finally:
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
def main():
    import time
    from setup import setup_components
    from comms.mavlink_bridge import ControlCommand
    from flight_sequence import FlightController, GateStore, GateAwareMAVLinkRX, print_gates
    import threading
    from pymavlink import mavutil

    SIM_IP   = "127.0.0.1"
    SIM_PORT = 14550

    system_boot_ms = int(time.time() * 1000)
    shared_data = {}

    # --- One connection, everything shares it ---
    components = setup_components(shared_data, system_boot_ms, SIM_IP, SIM_PORT)
    sim_conn   = components['sim_conn']
    controller = components['controller']
    ts_loop    = components['ts_loop']
    vision_rx  = components['vision_rx']
    # Don't use mavlink_rx from components — replace with gate-aware version
    mavlink_rx = components['mavlink_rx']

    # --- Gate store wired into the existing RX thread ---
    gate_store = GateStore()

    # Monkey-patch on_track_data onto the existing mavlink_rx instance
    # so we don't open a second connection or second thread
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

    mavlink_rx.on_track_data = _on_track_data  # override in place

    # --- Flight controller uses sim_conn directly (no second UDP socket) ---
    fc = FlightController(sim_conn)

    try:
        # Kick off gate printing in background — data arrives whenever sim sends it
        threading.Thread(
            target=print_gates,
            args=(gate_store, 15.0),
            daemon=True
        ).start()

        # Arm → Takeoff → Forward
        fc.arm()
        fc.takeoff(alt_m=5.0)
        fc.fly_forward(speed_mps=4.0, duration_s=5.0)

    except KeyboardInterrupt:
        print("Interrupted — holding position.")
        fc.hold_position()
    finally:
        ts_loop.get_thread_for_join().join(timeout=1.0)
        mavlink_rx.get_thread_for_join().join(timeout=1.0)
        vision_rx.get_thread_for_join().join(timeout=1.0)
        print("Exited cleanly.")

if __name__ == '__main__':
    main()
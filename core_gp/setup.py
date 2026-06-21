import logging
import os
import time
import numpy as np
from pymavlink import mavutil
from timesync import TimeSync
from mavlink_rx import MAVLinkRX
from controller import Controller
from comms.vision_stream import VisionStreamReceiver
from perception.perception import PerceptionManager

logger = logging.getLogger(__name__)

DEBUG_FRAME_DIR = os.path.join(os.path.dirname(__file__), "debug_frames")

def setup_components(shared_data, system_boot_ms, server_ip, server_udp_port, vision_port=5600):
    # -------------------------------
    # Mavlink Connection
    # -------------------------------
    # Start a connection listening on a UDP port
    sim_conn = mavutil.mavlink_connection('udpin:%s:%s' % (server_ip, server_udp_port,))
    print("Waiting for heartbeat...", flush=True)
    sim_conn.wait_heartbeat()
    print(f"Connected to system: {sim_conn.target_system}", flush=True)

    # -------------------------------
    # Setup Mavlink msg receiver
    # -------------------------------
    print("Setting up MAVLink rx...", flush=True)
    mavlink_rx = MAVLinkRX.create_mavlink_rx(sim_conn, shared_data)

    # -------------------------------
    # Timesync request Loop
    # -------------------------------
    print("Setting up Timesync loop...", flush=True)
    ts_loop = TimeSync.create_timesync(sim_conn, shared_data)

    # -------------------------------
    # Vision: camera frames -> gate/obstacle/VO perception
    # -------------------------------
    print("Setting up vision stream + perception...", flush=True)
    perception = PerceptionManager()

    # Frame-level diagnostics — without this, nothing logs at all unless a
    # PnP pose succeeds, which gives no signal if frames simply aren't
    # arriving or the HSV thresholds don't match this sim's actual gate
    # colour (the most likely first failure mode on real footage).
    diag = {'frame_count': 0, 'last_log': 0.0, 'dumped': False}
    os.makedirs(DEBUG_FRAME_DIR, exist_ok=True)
    detector = perception.gate_detector

    def _on_frame(frame, sim_time_ns):
        body_velocity = np.array([
            shared_data.get('vel_x', 0.0),
            shared_data.get('vel_y', 0.0),
            shared_data.get('vel_z', 0.0),
        ])
        speed_mps = float(np.linalg.norm(body_velocity))
        perception.update(frame, body_velocity, speed_mps)

        diag['frame_count'] += 1
        shared_data['vision_frame_count'] = diag['frame_count']
        shared_data['vision_gates_in_frame'] = len(perception.latest_gates)

        # Dump the first real frame to disk -- open it directly to see the
        # actual gate colour against the actual background, rather than
        # guessing HSV bounds blind.
        if not diag['dumped']:
            try:
                import cv2
                path = os.path.join(DEBUG_FRAME_DIR, "first_frame.png")
                cv2.imwrite(path, frame)
                logger.info(f"VISION: saved first frame to {path}")
            except Exception as e:
                logger.warning(f"VISION: failed to save debug frame: {e}")
            diag['dumped'] = True

        gate = perception.next_gate()
        if gate is not None:
            shared_data['vision_gate_pose_valid']     = gate.pose_valid
            shared_data['vision_gate_position_body']  = gate.position_body
            shared_data['vision_gate_bearing_body']   = gate.bearing_body
            shared_data['vision_gate_range']          = gate.range_estimate
            shared_data['vision_gate_relative_yaw']   = gate.relative_yaw
            shared_data['vision_gate_confidence']     = gate.confidence
            shared_data['vision_gate_time_ns']        = sim_time_ns
        else:
            shared_data['vision_gate_pose_valid'] = False

        now = time.monotonic()
        if now - diag['last_log'] >= 1.0:
            diag['last_log'] = now
            hsv = detector.last_dominant_hsv
            hsv_str = f"({hsv[0]:.0f},{hsv[1]:.0f},{hsv[2]:.0f})" if hsv else "none"
            if not perception.latest_gates:
                logger.info(
                    f"VISION: frame_count={diag['frame_count']} gates_in_frame=0 "
                    f"mask_px={detector.last_mask_pixel_count} "
                    f"raw_contours={detector.last_raw_contour_count} "
                    f"area_filtered={detector.last_area_filtered_count} "
                    f"quads={detector.last_quad_count} "
                    f"dominant_colourful_hsv={hsv_str} "
                    f"(configured band: {detector.hsv_lower.tolist()}-{detector.hsv_upper.tolist()})"
                )
            else:
                best = perception.latest_gates[0]
                logger.info(
                    f"VISION: frame_count={diag['frame_count']} "
                    f"gates_in_frame={len(perception.latest_gates)} "
                    f"best_confidence={best.confidence:.2f} "
                    f"pose_valid={best.pose_valid} "
                    f"range_est={best.range_estimate:.1f}m "
                    f"dominant_colourful_hsv={hsv_str}"
                )

    vision_rx = VisionStreamReceiver(port=vision_port, on_frame_callback=_on_frame)
    vision_rx.start()

    # -------------------------------
    # Main control loop
    # -------------------------------
    controller = Controller(sim_conn, shared_data, system_boot_ms)

    return {
        'vision_rx': vision_rx,
        'perception': perception,
        'mavlink_rx': mavlink_rx,
        'ts_loop': ts_loop,
        'sim_conn': sim_conn,
        'controller': controller
    }
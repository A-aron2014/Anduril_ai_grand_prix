import time
import threading

from pymavlink import mavutil

TIMESYNC_REQUEST_HZ = 10
HEARTBEAT_HZ        = 2

class TimeSync:

    def __init__(self, mavlink_connection, data):
        self.mavlink_conn = mavlink_connection
        self.data = data
        self.thread = None
        self.is_running = False

    @classmethod
    def create_timesync(cls, mavlink_connection, data):
        ts = cls(mavlink_connection, data)
        ts.thread = threading.Thread(
            target=ts.timesync_loop,
            daemon = False
        )
        ts.is_running = True
        ts.thread.start()
        return ts

    def get_thread_for_join(self):
        self.is_running = False
        return self.thread

    def timesync_loop(self):
        hb_interval = 1.0 / HEARTBEAT_HZ
        ts_interval = 1.0 / TIMESYNC_REQUEST_HZ
        last_hb = 0.0
        while self.is_running:
            now_s = time.monotonic()
            # Heartbeat: autopilots go silent/failsafe without a steady 1 Hz
            # heartbeat from the GCS side.
            if now_s - last_hb >= hb_interval:
                self.mavlink_conn.mav.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_GCS,
                    mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                    0,  # base_mode
                    0,  # custom_mode
                    mavutil.mavlink.MAV_STATE_ACTIVE,
                )
                last_hb = now_s
            now_ns = int(time.time_ns())
            self.mavlink_conn.mav.timesync_send(now_ns, 0)
            time.sleep(ts_interval)

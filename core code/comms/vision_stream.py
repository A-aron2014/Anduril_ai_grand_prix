"""
Vision Stream Receiver
Receives chunked JPEG frames over UDP (port 5600) and reconstructs them.
Camera spec: 640x360, 30 Hz, pinhole (fx=fy=320, cx=320, cy=180), 20° up-tilt.
"""

import socket
import struct
import threading
import time
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Callable
import numpy as np

logger = logging.getLogger(__name__)

# Header format: frame_id(u32), chunk_id(u16), total_chunks(u16),
#                jpeg_size(u32), payload_size(u32), sim_time_ns(u64)
HEADER_FORMAT = '<IHHIIQ'
HEADER_SIZE = 24  # bytes


@dataclass
class CameraIntrinsics:
    """Pinhole camera model (no distortion)."""
    width: int = 640
    height: int = 360
    fx: float = 320.0
    fy: float = 320.0
    cx: float = 320.0
    cy: float = 180.0
    tilt_deg: float = 20.0   # camera tilted 20° upward from body X-axis (NED)

    @property
    def K(self) -> np.ndarray:
        """3×3 intrinsic matrix."""
        return np.array([
            [self.fx, 0,       self.cx],
            [0,       self.fy, self.cy],
            [0,       0,       1      ],
        ], dtype=np.float64)

    @property
    def body_to_cam_R(self) -> np.ndarray:
        """
        Rotation from body (NED) to camera frame.
        Camera is tilted 20° nose-up (negative pitch in NED).
        Then you must rotate into your image library's convention (OpenCV: x right, y down, z forward).
        """
        import math
        tilt = math.radians(self.tilt_deg)
        # Rotation about body Y-axis (pitch up)
        R_tilt = np.array([
            [ math.cos(tilt), 0, math.sin(tilt)],
            [0,               1, 0              ],
            [-math.sin(tilt), 0, math.cos(tilt)],
        ])
        # NED → OpenCV: x→right, y→down, z→forward
        R_ned_to_cv = np.array([
            [0,  1,  0],
            [0,  0,  1],
            [1,  0,  0],
        ], dtype=np.float64)
        return R_ned_to_cv @ R_tilt


class FrameBuffer:
    """Accumulates chunks for a single frame_id."""

    def __init__(self, frame_id: int, total_chunks: int, jpeg_size: int, sim_time_ns: int):
        self.frame_id = frame_id
        self.total_chunks = total_chunks
        self.jpeg_size = jpeg_size
        self.sim_time_ns = sim_time_ns
        self.chunks: Dict[int, bytes] = {}
        self.created_at = time.monotonic()

    def add_chunk(self, chunk_id: int, data: bytes):
        self.chunks[chunk_id] = data

    def is_complete(self) -> bool:
        return len(self.chunks) == self.total_chunks

    def assemble(self) -> bytes:
        return b''.join(self.chunks[i] for i in range(self.total_chunks))

    def is_stale(self, timeout: float = 0.5) -> bool:
        return (time.monotonic() - self.created_at) > timeout


class VisionStreamReceiver:
    """
    Listens on UDP port 5600, reassembles JPEG frames, and delivers
    decoded numpy images (H×W×3, BGR) to a registered callback.

    Usage
    -----
        def on_frame(frame: np.ndarray, sim_time_ns: int):
            ...   # runs in the receiver thread

        vsr = VisionStreamReceiver(on_frame_callback=on_frame)
        vsr.start()
        ...
        vsr.stop()
    """

    def __init__(self, host: str = '0.0.0.0', port: int = 5600,
                 on_frame_callback: Optional[Callable] = None,
                 max_buffer_age: float = 0.5):
        self.host = host
        self.port = port
        self.on_frame_callback = on_frame_callback
        self.max_buffer_age = max_buffer_age
        self.intrinsics = CameraIntrinsics()
        self._running = False
        self._buffers: Dict[int, FrameBuffer] = {}
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_sim_time: int = 0
        self._frame_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        self._running = True
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        self._sock.bind((self.host, self.port))
        self._sock.settimeout(1.0)
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        logger.info(f"Vision stream listening on {self.host}:{self.port}")

    def stop(self):
        self._running = False
        self._sock.close()

    def get_latest_frame(self) -> Optional[tuple]:
        """Returns (frame_np, sim_time_ns) or None. Thread-safe."""
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy(), self._latest_sim_time

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _recv_loop(self):
        while self._running:
            try:
                data, _ = self._sock.recvfrom(65535)
            except socket.timeout:
                self._purge_stale()
                continue
            except OSError:
                break

            if len(data) < HEADER_SIZE:
                continue

            header = struct.unpack_from(HEADER_FORMAT, data, 0)
            frame_id, chunk_id, total_chunks, jpeg_size, payload_size, sim_time_ns = header
            payload = data[HEADER_SIZE:HEADER_SIZE + payload_size]

            if frame_id not in self._buffers:
                self._buffers[frame_id] = FrameBuffer(frame_id, total_chunks, jpeg_size, sim_time_ns)

            buf = self._buffers[frame_id]
            buf.add_chunk(chunk_id, payload)

            if buf.is_complete():
                jpeg_bytes = buf.assemble()
                del self._buffers[frame_id]
                self._decode_and_deliver(jpeg_bytes, sim_time_ns)

            self._purge_stale()

    def _decode_and_deliver(self, jpeg_bytes: bytes, sim_time_ns: int):
        try:
            import cv2
            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                return
        except Exception as e:
            logger.warning(f"Frame decode error: {e}")
            return

        with self._lock:
            self._latest_frame = frame
            self._latest_sim_time = sim_time_ns
            self._frame_count += 1

        if self.on_frame_callback:
            try:
                self.on_frame_callback(frame, sim_time_ns)
            except Exception as e:
                logger.error(f"Frame callback error: {e}")

    def _purge_stale(self):
        stale = [fid for fid, buf in self._buffers.items() if buf.is_stale(self.max_buffer_age)]
        for fid in stale:
            del self._buffers[fid]

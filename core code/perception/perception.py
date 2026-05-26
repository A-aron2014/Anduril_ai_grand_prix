"""
Perception — Gate Detection, Obstacle Avoidance, and Visual Odometry.
All coordinates are returned in NED body frame unless noted.
Camera intrinsics: fx=fy=320, cx=320, cy=180, 640x360, 20° up-tilt.
"""

import math
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GateObservation:
    """A detected gate in the image, projected to a bearing + range estimate."""
    pixel_center: Tuple[float, float]   # (u, v) in image
    pixel_width: float                  # apparent width in pixels
    bearing_body: np.ndarray            # unit vector in body-NED frame
    range_estimate: float               # metres (from apparent size heuristic)
    confidence: float                   # 0–1
    gate_id: int = -1                   # assigned by tracker


@dataclass
class Obstacle:
    """A detected obstacle in body-NED frame."""
    position_body: np.ndarray           # (fwd, right, down) metres
    radius: float                       # estimated radius, metres
    confidence: float


@dataclass
class VisualOdometryResult:
    """Incremental pose change estimated from consecutive frames."""
    delta_position: np.ndarray          # (dx, dy, dz) NED metres
    delta_yaw: float                    # radians
    num_features: int
    valid: bool


# ---------------------------------------------------------------------------
# Gate Detector
# ---------------------------------------------------------------------------

class GateDetector:
    """
    Detects race gates from a BGR camera frame.

    Strategy (swappable):
      1. HSV colour segmentation for high-visibility gate markers.
      2. Contour fitting to find rectangular gate opening.
      3. Project gate centre pixel → body-frame bearing using camera intrinsics.
      4. Estimate range from apparent gate width (requires known gate_real_width_m).

    Replace segment_and_find_contours() with a neural detector if available.
    """

    # Pinhole intrinsics (matches spec)
    FX = FY = 320.0
    CX, CY = 320.0, 180.0
    TILT_DEG = 20.0                     # camera tilted upward

    def __init__(self, gate_real_width_m: float = 2.0,
                 min_contour_area: float = 500.0):
        self.gate_real_width_m = gate_real_width_m
        self.min_contour_area = min_contour_area
        self._next_id = 0

        # Pre-compute camera→body rotation (inverse of body→cam)
        tilt = math.radians(self.TILT_DEG)
        # Camera tilted nose-up: rotate image plane rays back to body NED
        self._R_cam_to_body = np.array([
            [math.cos(tilt),  0, math.sin(tilt)],
            [0,               1, 0             ],
            [-math.sin(tilt), 0, math.cos(tilt)],
        ])

    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[GateObservation]:
        """Main entry: frame is H×W×3 BGR. Returns list of gate observations."""
        try:
            import cv2
        except ImportError:
            logger.error("OpenCV not available")
            return []

        contours = self._segment_and_find_contours(frame, cv2)
        gates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_contour_area:
                continue
            obs = self._contour_to_observation(cnt, cv2)
            if obs:
                gates.append(obs)

        # Sort by range (closest first)
        gates.sort(key=lambda g: g.range_estimate)
        return gates

    def _segment_and_find_contours(self, frame, cv2):
        """
        HSV segmentation for brightly coloured gate markers.
        Tune the HSV ranges for the actual gate colours in the sim.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Example: orange gate markers
        lower = np.array([5, 150, 150])
        upper = np.array([25, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def _contour_to_observation(self, contour, cv2) -> Optional[GateObservation]:
        rect = cv2.minAreaRect(contour)
        (cx_px, cy_px), (w_px, h_px), _ = rect
        pix_width = max(w_px, h_px)

        if pix_width < 10:
            return None

        # Pixel → normalised camera ray (OpenCV convention: x right, y down, z fwd)
        ray_cam = np.array([
            (cx_px - self.CX) / self.FX,
            (cy_px - self.CY) / self.FY,
            1.0,
        ])
        ray_cam /= np.linalg.norm(ray_cam)

        # Rotate to body NED
        ray_body = self._R_cam_to_body @ ray_cam
        ray_body /= np.linalg.norm(ray_body)

        # Range from apparent width: R = fx * real_width / pixel_width
        range_est = self.FX * self.gate_real_width_m / pix_width

        confidence = min(1.0, pix_width / 100.0)

        obs = GateObservation(
            pixel_center=(cx_px, cy_px),
            pixel_width=pix_width,
            bearing_body=ray_body,
            range_estimate=range_est,
            confidence=confidence,
            gate_id=self._next_id,
        )
        self._next_id += 1
        return obs


# ---------------------------------------------------------------------------
# Obstacle Detector
# ---------------------------------------------------------------------------

class ObstacleDetector:
    """
    Simple depth-from-motion obstacle detector.
    Uses optical-flow divergence to flag regions that are approaching fast.
    Replace with a stereo or ML depth estimator for better performance.
    """

    def __init__(self, danger_threshold_m: float = 3.0):
        self.danger_threshold_m = danger_threshold_m
        self._prev_gray = None

    def detect(self, frame: np.ndarray, body_velocity: np.ndarray) -> List[Obstacle]:
        """
        frame: H×W×3 BGR
        body_velocity: (vx, vy, vz) in body NED (m/s)
        Returns list of Obstacle in body-NED frame.
        """
        try:
            import cv2
        except ImportError:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        obstacles = []

        if self._prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self._prev_gray, gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            # Regions with low flow magnitude while moving = close/static objects
            speed = np.linalg.norm(body_velocity)
            if speed > 0.5:
                mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                low_flow = (mag < 1.0).astype(np.uint8) * 255
                cnts, _ = cv2.findContours(low_flow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in cnts:
                    area = cv2.contourArea(cnt)
                    if area < 200:
                        continue
                    M = cv2.moments(cnt)
                    if M['m00'] == 0:
                        continue
                    cu = M['m10'] / M['m00']
                    cv_ = M['m01'] / M['m00']
                    # Very rough: assume obstacle is at danger_threshold_m
                    ray = np.array([(cu - 320) / 320, (cv_ - 180) / 320, 1.0])
                    ray /= np.linalg.norm(ray)
                    pos = ray * self.danger_threshold_m
                    obstacles.append(Obstacle(
                        position_body=pos,
                        radius=1.0,
                        confidence=min(1.0, area / 5000),
                    ))

        self._prev_gray = gray
        return obstacles


# ---------------------------------------------------------------------------
# Visual Odometry
# ---------------------------------------------------------------------------

class VisualOdometry:
    """
    Frame-to-frame feature-based visual odometry.
    Uses ORB features + essential matrix decomposition (monocular → scale ambiguous).
    Scale recovered from IMU velocity magnitude.
    """

    def __init__(self):
        self._prev_gray = None
        self._prev_kp = None
        self._prev_des = None
        self._K = np.array([[320, 0, 320], [0, 320, 180], [0, 0, 1]], dtype=np.float64)

    def update(self, frame: np.ndarray, speed_mps: float) -> VisualOdometryResult:
        """
        frame: H×W×3 BGR
        speed_mps: scalar speed from IMU for scale recovery
        """
        try:
            import cv2
        except ImportError:
            return VisualOdometryResult(np.zeros(3), 0.0, 0, False)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(nfeatures=300)
        kp, des = orb.detectAndCompute(gray, None)

        result = VisualOdometryResult(np.zeros(3), 0.0, len(kp), False)

        if (self._prev_gray is not None and des is not None
                and self._prev_des is not None and len(kp) >= 8):
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(self._prev_des, des)
            matches = sorted(matches, key=lambda x: x.distance)[:50]

            if len(matches) >= 8:
                pts1 = np.float32([self._prev_kp[m.queryIdx].pt for m in matches])
                pts2 = np.float32([kp[m.trainIdx].pt for m in matches])

                E, mask = cv2.findEssentialMat(pts1, pts2, self._K,
                                               method=cv2.RANSAC, prob=0.999, threshold=1.0)
                if E is not None:
                    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, self._K, mask=mask)
                    # t is unit vector — scale from IMU
                    delta_pos = (t.flatten() * speed_mps * (1 / 30.0))  # assume 30 Hz
                    yaw_delta = math.atan2(R[1, 0], R[0, 0])
                    result = VisualOdometryResult(delta_pos, yaw_delta, len(matches), True)

        self._prev_gray = gray
        self._prev_kp = kp
        self._prev_des = des
        return result


# ---------------------------------------------------------------------------
# Perception Manager (facade)
# ---------------------------------------------------------------------------

class PerceptionManager:
    """
    Top-level perception facade. Call update() each frame.
    Stores latest gate observations, obstacles, and VO delta.
    """

    def __init__(self):
        self.gate_detector = GateDetector()
        self.obstacle_detector = ObstacleDetector()
        self.visual_odometry = VisualOdometry()

        self.latest_gates: List[GateObservation] = []
        self.latest_obstacles: List[Obstacle] = []
        self.latest_vo: Optional[VisualOdometryResult] = None

    def update(self, frame: np.ndarray,
               body_velocity: np.ndarray,
               speed_mps: float):
        """Call at ~30 Hz as frames arrive."""
        self.latest_gates = self.gate_detector.detect(frame)
        self.latest_obstacles = self.obstacle_detector.detect(frame, body_velocity)
        self.latest_vo = self.visual_odometry.update(frame, speed_mps)

    def next_gate(self) -> Optional[GateObservation]:
        """Returns the closest high-confidence gate, or None."""
        viable = [g for g in self.latest_gates if g.confidence > 0.3]
        return viable[0] if viable else None

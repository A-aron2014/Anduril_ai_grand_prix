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
    bearing_body: np.ndarray            # unit vector in body-FRD frame (forward, right, down)
    range_estimate: float               # metres (from apparent size heuristic, or PnP if pose_valid)
    confidence: float                   # 0–1
    gate_id: int = -1                   # assigned by tracker

    # Populated when solvePnP succeeds on a clean 4-corner quad fit; gives
    # the full relative gate pose instead of just a bearing ray, which is
    # what guidance actually needs to localize gates from vision alone.
    pose_valid: bool = False
    position_body: Optional[np.ndarray] = None   # (fwd, right, down) metres, gate center
    relative_yaw: float = 0.0                     # gate-plane heading relative to body, radians


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
                 gate_real_height_m: float = None,
                 min_contour_area: float = 500.0):
        self.gate_real_width_m = gate_real_width_m
        # Gates are often square in practice; default height to width if
        # not given separately.
        self.gate_real_height_m = gate_real_height_m if gate_real_height_m is not None else gate_real_width_m
        self.min_contour_area = min_contour_area
        self._next_id = 0

        # HSV band for "bright orange" — wider than a tight guess since we
        # have no real footage to calibrate against yet. Hue 0-30 covers
        # red-orange through orange-yellow (OpenCV hue is 0-179, red=0);
        # saturation/value floors of 80 are permissive enough to admit a
        # mildly-lit orange against a desaturated gray background while
        # still excluding gray (low saturation by definition).
        self.hsv_lower = np.array([0, 80, 80])
        self.hsv_upper = np.array([30, 255, 255])

        # Stage-by-stage diagnostics, updated every detect() call, so a
        # caller can tell *where* detection is failing (no colour match at
        # all vs. a match that's just too small/non-quad) instead of just
        # seeing "0 gates" with no further information.
        self.last_mask_pixel_count = 0
        self.last_raw_contour_count = 0
        self.last_area_filtered_count = 0
        self.last_quad_count = 0
        self.last_dominant_hsv: Optional[Tuple[float, float, float]] = None

        # Camera (OpenCV: x=right, y=down, z=forward) -> body-FRD
        # (forward, right, down). This is a permutation (camera axes don't
        # share the body's axis *order*, only a common origin) composed
        # with the upward pitch tilt -- NOT just a tilt rotation applied
        # in place, which would silently mislabel "forward" as "right".
        # Anchor check: a dead-center pixel (camera ray straight down the
        # lens, (0,0,1)) must map to a body ray that is mostly forward
        # with an upward (negative-down) component, since the camera is
        # tilted nose-up relative to the body.
        tilt = math.radians(self.TILT_DEG)
        self._R_cam_to_body = np.array([
            [0,             math.sin(tilt),  math.cos(tilt)],
            [1,             0,                0             ],
            [0,             math.cos(tilt), -math.sin(tilt)],
        ])

    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[GateObservation]:
        """Main entry: frame is H×W×3 BGR. Returns list of gate observations."""
        try:
            import cv2
        except ImportError:
            logger.error("OpenCV not available")
            return []

        self._update_dominant_hsv_diagnostic(frame, cv2)

        contours = self._segment_and_find_contours(frame, cv2)
        self.last_raw_contour_count = len(contours)
        self.last_quad_count = 0

        gates = []
        area_filtered = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_contour_area:
                continue
            area_filtered += 1
            obs = self._contour_to_observation(cnt, cv2)
            if obs:
                gates.append(obs)
        self.last_area_filtered_count = area_filtered

        # Sort by range (closest first)
        gates.sort(key=lambda g: g.range_estimate)
        return gates

    def _update_dominant_hsv_diagnostic(self, frame, cv2):
        """
        Reports the dominant colour of non-gray pixels in the frame,
        independent of the configured hsv_lower/hsv_upper band -- this is
        what should be used to actually tune that band, instead of
        guessing. Gray/background pixels (low saturation) are excluded so
        the result reflects whatever *is* colourful in frame (gate markers,
        if visible).
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        colourful = sat > 60
        if not np.any(colourful):
            self.last_dominant_hsv = None
            return
        h = float(np.median(hsv[:, :, 0][colourful]))
        s = float(np.median(hsv[:, :, 1][colourful]))
        v = float(np.median(hsv[:, :, 2][colourful]))
        self.last_dominant_hsv = (h, s, v)

    def _segment_and_find_contours(self, frame, cv2):
        """
        HSV segmentation for brightly coloured gate markers.
        Tune hsv_lower/hsv_upper for the actual gate colours in the sim --
        last_dominant_hsv (set by detect()) reports the actual colourful
        pixels in frame to calibrate against.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        self.last_mask_pixel_count = int(np.count_nonzero(mask))

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

        # Rotate to body frame (forward, right, down)
        ray_body = self._R_cam_to_body @ ray_cam
        ray_body /= np.linalg.norm(ray_body)

        # Range from apparent width — fallback heuristic, overridden below
        # if a clean 4-corner PnP solve succeeds.
        range_est = self.FX * self.gate_real_width_m / pix_width
        confidence = min(1.0, pix_width / 100.0)

        # Bearing-ray fallback position -- always available from the pixel
        # centroid alone, so callers (e.g. the MPC's gate-centroid feedback)
        # have *something* to correct toward even when the quad/PnP solve
        # below fails. This matters most up close, right before crossing a
        # gate, where the contour clips against the frame edge and breaks
        # the 4-corner fit -- exactly when a position correction is most
        # valuable and was previously silently dropped (position_body
        # stayed None whenever pose_valid was False).
        position_body = ray_body * range_est
        relative_yaw = 0.0
        pose_valid = False

        quad = self._contour_to_quad(contour, cv2)
        if quad is not None:
            self.last_quad_count += 1
            pnp_result = self._solve_pnp(quad, cv2)
            if pnp_result is not None:
                position_body, relative_yaw = pnp_result
                range_est = float(np.linalg.norm(position_body))
                pose_valid = True
                confidence = min(1.0, confidence + 0.3)   # PnP solve is a stronger signal than width alone

        obs = GateObservation(
            pixel_center=(cx_px, cy_px),
            pixel_width=pix_width,
            bearing_body=ray_body,
            range_estimate=range_est,
            confidence=confidence,
            gate_id=self._next_id,
            pose_valid=pose_valid,
            position_body=position_body,
            relative_yaw=relative_yaw,
        )
        self._next_id += 1
        return obs

    # ------------------------------------------------------------------
    # PnP-based relative pose (preferred over the width-heuristic range
    # estimate whenever the gate's 4 corners can be cleanly extracted)
    # ------------------------------------------------------------------

    def _contour_to_quad(self, contour, cv2) -> Optional[np.ndarray]:
        """
        Approximates the contour to a quadrilateral and returns its 4
        corners ordered (top-left, top-right, bottom-right, bottom-left)
        in pixel coordinates, or None if the contour isn't usably
        rectangular (occlusion, blur, segmentation noise, an actually
        round/irregular blob, etc.) — callers fall back to the
        width-heuristic bearing/range estimate.

        Falls back to the contour's minimum-area bounding box when
        approxPolyDP doesn't land on exactly 4 points. In practice this
        happens routinely at close range: the gate fills (or clips
        against) the frame edge and antialiasing/segmentation noise adds
        extra vertices, breaking the clean corner fit right when the
        drone is closest to the gate and a position correction matters
        most. Gated on extent (contour area vs. its bounding-box area)
        so a genuinely round/irregular blob is still rejected instead of
        being handed to PnP as a fake rectangle.
        """
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float64)
        else:
            rect = cv2.minAreaRect(contour)
            (_, _), (w_px, h_px), _ = rect
            box_area = max(w_px * h_px, 1e-6)
            extent = cv2.contourArea(contour) / box_area
            if extent < 0.7:
                return None
            pts = cv2.boxPoints(rect).astype(np.float64)

        s = pts.sum(axis=1)
        diff = pts[:, 0] - pts[:, 1]
        ordered = np.zeros((4, 2), dtype=np.float64)
        ordered[0] = pts[np.argmin(s)]      # top-left: smallest x+y
        ordered[2] = pts[np.argmax(s)]      # bottom-right: largest x+y
        ordered[1] = pts[np.argmax(diff)]   # top-right: largest x-y
        ordered[3] = pts[np.argmin(diff)]   # bottom-left: smallest x-y
        return ordered

    def _solve_pnp(self, image_corners: np.ndarray, cv2) -> Optional[Tuple[np.ndarray, float]]:
        """
        image_corners: (4,2) pixel coordinates, ordered TL/TR/BR/BL,
        matching the model points below.

        Returns (position_body, relative_yaw):
          position_body — gate center relative to the drone, in body-FRD
                          (forward, right, down) metres.
          relative_yaw  — gate-plane heading relative to body forward,
                          radians. 0 = gate faced square-on; nonzero means
                          approaching the gate at an angle.
        Returns None if the solve is degenerate or fails.
        """
        hw = self.gate_real_width_m / 2.0
        hh = self.gate_real_height_m / 2.0
        model_points = np.array([
            [-hw, -hh, 0.0],   # top-left
            [ hw, -hh, 0.0],   # top-right
            [ hw,  hh, 0.0],   # bottom-right
            [-hw,  hh, 0.0],   # bottom-left
        ], dtype=np.float64)

        K = np.array([
            [self.FX, 0,        self.CX],
            [0,        self.FY, self.CY],
            [0,        0,        1.0   ],
        ], dtype=np.float64)

        # IPPE (not IPPE_SQUARE) — gates aren't guaranteed to be square,
        # width/height are independent fields in the track data, and
        # IPPE_SQUARE's square assumption gives a silently wrong pose on
        # a rectangular target.
        pnp_flag = getattr(cv2, 'SOLVEPNP_IPPE', cv2.SOLVEPNP_ITERATIVE)
        ok, rvec, tvec = cv2.solvePnP(
            model_points, image_corners, K, None, flags=pnp_flag
        )
        if not ok:
            return None

        t_cam = tvec.reshape(3)
        R_gate_to_cam, _ = cv2.Rodrigues(rvec)

        position_body = self._R_cam_to_body @ t_cam
        R_gate_to_body = self._R_cam_to_body @ R_gate_to_cam

        # Gate plane's own forward axis (its local Z, the direction you'd
        # fly straight through it) expressed in body coordinates — gives
        # how square-on the approach is.
        gate_normal_body = R_gate_to_body[:, 2]
        relative_yaw = math.atan2(gate_normal_body[1], gate_normal_body[0])

        return position_body, relative_yaw


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
        frame: HxWx3 BGR
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

"""
GuidanceStateMachine.py
-----------------------
A clean state machine for UAS target search & track guidance.

States:
    TAKEOFF  - Orbit home position, climb to h_trim
    TRANSIT  - Straight-line guidance toward last known target position
    SEARCH   - Expanding orbit (or lawnmower) around last known position
    TRACK    - Orbit around target centroid; re-centers each update

Transitions:
    TAKEOFF  → TRANSIT  : altitude within tol of h_trim
    TRANSIT  → SEARCH   : horizontal dist to waypoint < r_capture
    SEARCH   → TRACK    : one or more targets found in FOV
    TRACK    → TRANSIT  : target lost for > loss_timeout seconds
                          (returns to last known position to re-acquire)
    TRACK    → SEARCH   : orbit radius exceeded r_search_max
                          (shouldn't happen in TRACK, but guards edge cases)

Usage
-----
    gsm = GuidanceStateMachine(params)
    ...
    control_objectives = gsm.update(state_est, wind_angles_est, found_targets, dt)

`control_objectives` is the dict expected by OrbitGuidance / straight_line_guidance.
Internally the machine calls your existing guidance functions so you don't need to
rip out any existing infrastructure.
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from types import SimpleNamespace
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------
class GuidanceState(Enum):
    TAKEOFF = auto()
    TRANSIT = auto()
    SEARCH  = auto()
    TRACK   = auto()


# ---------------------------------------------------------------------------
# Parameters dataclass  (all distances in metres, angles in radians)
# ---------------------------------------------------------------------------
@dataclass
class GuidanceMachineParams:
    # Altitude
    h_trim: float = 1805.0          # target cruise altitude (m, positive up)
    h_tol:  float = 50.0            # altitude tolerance to leave TAKEOFF (m)

    # Transit / waypoint capture
    r_capture: float = 150.0        # horizontal dist to WP that triggers SEARCH (m)

    # Search – expanding orbit
    orbit_speed:    float = 18.0    # commanded airspeed in orbit (m/s)
    orbit_r_init:   float = 200.0   # initial search orbit radius (m)
    orbit_r_step:   float = 100.0   # radius increment per completed revolution (m)
    orbit_r_max:    float = 2000.0  # give up and return to TRANSIT beyond this (m)
    orbit_flag:     int   = 1       # 1 = CW, -1 = CCW (passed to OrbitGuidance)
    orbit_gains: object   = field(default_factory=lambda: SimpleNamespace(kr=0.01, kz=0.001))

    # Home position (used for TAKEOFF orbit, NED metres)
    home_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))

    # Track
    loss_timeout: float = 5.0       # seconds without a detection before → TRANSIT
    track_r:      float = 300.0     # orbit radius around target centroid (m)

    # Straight-line guidance gains
    chi_inf: float = np.deg2rad(70)
    kpath:   float = 0.05

    SimpleNamespace
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _horizontal_dist(pos_a: np.ndarray, pos_b: np.ndarray) -> float:
    """Euclidean distance in the North-East plane (ignores altitude)."""
    return float(np.hypot(pos_a[0] - pos_b[0], pos_a[1] - pos_b[1]))


def _target_centroid(found_targets) -> Optional[np.ndarray]:
    """Return NE centroid of found_targets, or None if list empty."""
    if not found_targets:
        return None
    x_c = np.mean([t.x for t in found_targets])
    y_c = np.mean([t.y for t in found_targets])
    return np.array([x_c, y_c])


def _wrap_angle(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------
class GuidanceStateMachine:
    """
    Drop-in guidance state machine.  Call `.update()` every simulation step;
    it returns a `control_objectives` SimpleNamespace compatible with
    SLCWithFeedForwardAutopilot / SimpleSLCAutopilot.
    """

    def __init__(self, params: GuidanceMachineParams,
                 orbit_guidance_fn,
                 straight_line_guidance_fn):
        """
        Parameters
        ----------
        params                  : GuidanceMachineParams
        orbit_guidance_fn       : callable  – your OrbitGuidance function
        straight_line_guidance_fn: callable – your straight_line_guidance function
        """
        self.p   = params
        self._orbit_guid = orbit_guidance_fn
        self._sl_guid    = straight_line_guidance_fn

        # ---- mutable state ----
        self.state: GuidanceState = GuidanceState.TAKEOFF

        self.last_known_pos: np.ndarray = params.home_pos.copy()  # NED
        self.transit_wp:     np.ndarray = params.home_pos.copy()  # where we're heading

        # Search orbit
        self.search_orbit_r:   float = params.orbit_r_init
        self.search_orbit_center: np.ndarray = params.home_pos.copy()
        self._orbit_heading_ref: Optional[float] = None   # heading when orbit started
        self._orbit_laps_heading: Optional[float] = None  # used to detect full rev

        # Track
        self._loss_timer: float = 0.0
        self.track_center: np.ndarray = params.home_pos.copy()

        # Logging (read by your plotter / animator)
        self.state_log: List[GuidanceState] = []
        self.orbit_r_log: List[float] = []

    def  update(self,
               aircraft_state: np.ndarray,
               wind_angles:    np.ndarray,
               found_targets:      list,
               dt:                 float):
        """
        Run one guidance step.
 
        Parameters
        ----------
        aircraft_state_est : full 12-state estimated aircraft state (NED)
        wind_angles_est    : [Va, alpha, beta] estimated wind angles
        found_targets      : list of MovingTarget objects currently in FOV
        dt                 : timestep (s)
 
        Returns
        -------
        control_objectives : SimpleNamespace passed to your autopilot
        """
        pos        = aircraft_state[0:3]   # NED position (z negative up)
        h_agl      = -pos[2]                   # altitude (positive up)
        Va_est     = wind_angles[0]
        full_state = aircraft_state        # keep reference for OrbitGuidance
 
        # ---- state transitions ----------------------------------------
        if self.state == GuidanceState.TAKEOFF:
            self._transition_from_takeoff(h_agl, pos)
 
        elif self.state == GuidanceState.TRANSIT:
            self._transition_from_transit(pos)
 
        elif self.state == GuidanceState.SEARCH:
            self._transition_from_search(found_targets, pos, Va_est, dt)
 
        elif self.state == GuidanceState.TRACK:
            self._transition_from_track(found_targets, dt, pos)
 
        # ---- log ----------------------------------------------------------
        self.state_log.append(self.state)
        self.orbit_r_log.append(self.search_orbit_r)
 
        # ---- compute guidance command ------------------------------------
        return self._compute_guidance(pos, Va_est, wind_angles, full_state)
 
    @property
    def state_name(self) -> str:
        return self.state.name
 
    # ------------------------------------------------------------------
    # Transition logic
    # ------------------------------------------------------------------
    def _transition_from_takeoff(self, h_agl: float, pos: np.ndarray):
        if abs(h_agl - self.p.h_trim) < self.p.h_tol:
            self._enter_transit(pos, self.last_known_pos)
 
    def _transition_from_transit(self, pos: np.ndarray):
        dist = _horizontal_dist(pos, self.transit_wp)
        if dist < self.p.r_capture:
            print(f"[GSM] TRANSIT → SEARCH  (dist={dist:.1f}m to WP)")
            self._enter_search(pos)
 
    def _transition_from_search(self, found_targets, pos, Va_est, dt):
        # Target acquired?
        if found_targets:
            centroid = _target_centroid(found_targets)
            print(f"[GSM] SEARCH → TRACK  ({len(found_targets)} target(s) found)")
            self._enter_track(centroid, pos)
            return
 
        # Check if we've completed a full revolution → expand radius
        self._check_orbit_lap(pos, Va_est, dt)
 
        # Orbit too large → give up and re-transit (shouldn't normally happen)
        if self.search_orbit_r > self.p.orbit_r_max:
            print(f"[GSM] SEARCH orbit radius exceeded max ({self.p.orbit_r_max}m) → TRANSIT")
            self._enter_transit(pos, self.last_known_pos)
 
    def _transition_from_track(self, found_targets, dt, pos: np.ndarray):
        if found_targets:
            # Update centroid continuously
            centroid = _target_centroid(found_targets)
            ned_centroid = np.array([centroid[0], centroid[1], -self.p.h_trim])
            self.track_center = ned_centroid
            self._loss_timer = 0.0
        else:
            self._loss_timer += dt
            if self._loss_timer >= self.p.loss_timeout:
                print(f"[GSM] TRACK → TRANSIT  (lost target for {self._loss_timer:.1f}s)")
                self.last_known_pos = self.track_center.copy()
                self._loss_timer    = 0.0
                self._enter_transit(pos, self.last_known_pos)
 
    # ------------------------------------------------------------------
    # Entry helpers
    # ------------------------------------------------------------------
    def _enter_transit(self, pos: np.ndarray, wp: np.ndarray):
        """
        Fix the straight-line start point and direction when entering TRANSIT.
        pos : current aircraft NED position
        wp  : target waypoint NED position
        """
        self.transit_wp          = wp.copy()
        self.transit_line_origin = pos.copy()
 
        dir_vec = np.array([wp[0] - pos[0], wp[1] - pos[1], 0.0])
        norm    = np.linalg.norm(dir_vec)
        self.transit_line_dir = dir_vec / norm if norm > 1e-3 else np.array([1.0, 0.0, 0.0])
 
        self.state = GuidanceState.TRANSIT
        print(f"[GSM] → TRANSIT  origin=({pos[0]:.0f},{pos[1]:.0f})  "
              f"wp=({wp[0]:.0f},{wp[1]:.0f})  dist={norm:.0f}m")
 
    def _enter_search(self, pos: np.ndarray):
        """Configure expanding orbit centred on current position."""
        self.search_orbit_center = np.array([pos[0], pos[1], -self.p.h_trim])
        self.search_orbit_r      = self.p.orbit_r_init
        self._orbit_heading_ref  = None
        self._orbit_laps_heading = None
        self._orbit_travelled    = 0.0
        self.state               = GuidanceState.SEARCH
        print(f"[GSM] → SEARCH  center=({pos[0]:.0f},{pos[1]:.0f})  r={self.search_orbit_r:.0f}m")
 
    def _enter_track(self, centroid: np.ndarray, pos: np.ndarray):
        """Begin orbiting the detected target centroid."""
        self.track_center = np.array([centroid[0], centroid[1], -self.p.h_trim])
        self._loss_timer  = 0.0
        self.state        = GuidanceState.TRACK
 
    # ------------------------------------------------------------------
    # Lap detector (expanding orbit)
    # ------------------------------------------------------------------
    def _check_orbit_lap(self, pos: np.ndarray, Va_est: float, dt: float):
        """
        Detect when the aircraft completes a full revolution around the search
        orbit centre.  Uses the angle from centre to aircraft position; a full
        revolution is detected when that angle crosses the reference angle for
        the second time (i.e. after having passed through 2π of travel).
        """
        cx, cy = self.search_orbit_center[0], self.search_orbit_center[1]
        current_angle = np.arctan2(pos[1] - cy, pos[0] - cx)
 
        if self._orbit_heading_ref is None:
            # First step in this orbit – record start angle
            self._orbit_heading_ref  = current_angle
            self._orbit_laps_heading = current_angle
            self._orbit_travelled    = 0.0
            return
 
        # Accumulate angular travel (handle wrap-around)
        d_angle = _wrap_angle(current_angle - self._orbit_laps_heading)
        self._orbit_travelled    += abs(d_angle)
        self._orbit_laps_heading  = current_angle
 
        if self._orbit_travelled >= 2 * np.pi:
            # Completed one revolution – expand radius
            self.search_orbit_r = min(
                self.search_orbit_r + self.p.orbit_r_step,
                self.p.orbit_r_max
            )
            self._orbit_travelled    = 0.0
            self._orbit_heading_ref  = current_angle
            print(f"[GSM] SEARCH orbit expanded → r={self.search_orbit_r:.0f}m")
 
    # ------------------------------------------------------------------
    # Guidance command generation
    # ------------------------------------------------------------------
    def _compute_guidance(self,
                          pos:             np.ndarray,
                          Va_est:          float,
                          wind_angles_est: np.ndarray,
                          full_state:      np.ndarray) -> SimpleNamespace:
        """
        Return control_objectives for the current state.
 
        control_objectives fields expected by SLCWithFeedForwardAutopilot:
            Va_c    – commanded airspeed
            h_c     – commanded altitude (positive up)
            chi_c   – commanded course angle (rad)
 
        full_state is the complete 12-element estimated state vector,
        passed directly to OrbitGuidance which needs velocity components [6:9].
        """
        if self.state == GuidanceState.TAKEOFF:
            result = self._orbit_guid(
                full_state,
                self.p.orbit_speed,
                self.p.orbit_r_init,
                np.array([full_state[0], full_state[1], -self.p.h_trim]),
                self.p.orbit_flag,
                self.p.orbit_gains,
            )
            result_arr = np.asarray(result, dtype=float).flatten()
            if not hasattr(self, '_orbit_array_printed'):
                print(f"[GSM] OrbitGuidance output array (len={len(result_arr)}): {result_arr}")
                self._orbit_array_printed = True
            result_arr[1] = self.p.h_trim
            return result_arr
        
        elif self.state == GuidanceState.TRANSIT:
            # pos_line is FIXED at the position when TRANSIT was entered.
            # dir_line is FIXED unit vector toward the waypoint.
            # Only pos (current aircraft position) updates each step.
            return self._sl_guid(
                self.transit_line_origin,
                self.transit_line_dir,
                pos,
                self.p.kpath,
                self.p.chi_inf,
                Va_est,
            )
 
        elif self.state == GuidanceState.SEARCH:
            return self._orbit_guid(
                full_state,
                self.p.orbit_speed,
                self.search_orbit_r,
                self.search_orbit_center,
                self.p.orbit_flag,
                self.p.orbit_gains,
            )
 
        elif self.state == GuidanceState.TRACK:
            return self._orbit_guid(
                full_state,
                self.p.orbit_speed,
                self.p.track_r,
                self.track_center,
                self.p.orbit_flag,
                self.p.orbit_gains,
            )
"""
DrawAircraftWithTargets.py
--------------------------
Extends DrawAircraft to overlay:
  - A FOV circle projected onto the ground plane
  - Moving target dots with short history trails
  - A guidance state label (TAKEOFF / TRANSIT / SEARCH / TRACK)
  - The search / track orbit circle

Drop-in replacement for DrawAircraft — same constructor signature,
same .update() call, just add the extra keyword arguments.

Usage
-----
    from DrawAircraftWithTargets import DrawAircraftWithTargets

    drawer = DrawAircraftWithTargets(pts, fov_radius_fn=build_aircraft_fov)

    # inside your sim loop:
    drawer.update(
        time           = time_iter[i],
        aircraft_state = aircraft_array[:, i],
        targets        = targets,           # full list of MovingTarget objects
        found_targets  = found_targets,     # subset currently in FOV
        gsm            = gsm,               # GuidanceStateMachine instance
    )
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# Max history length shown as a trail behind each target
TRAIL_LEN = 60


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_circle_ned(center_n, center_e, radius, n_pts=64):
    """
    Return (N-array, E-array, D-array) for a circle in the ground plane (D=0).
    """
    theta = np.linspace(0, 2 * np.pi, n_pts)
    ns = center_n + radius * np.cos(theta)
    es = center_e + radius * np.sin(theta)
    ds = np.zeros(n_pts)
    return ns, es, ds


def _ned_to_xyz_arr(ns, es, ds):
    """Vectorised NED → display XYZ (X=East, Y=North, Z=-Down)."""
    return es, ns, -ds


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DrawAircraftWithTargets:
    """
    Real-time 3-D aircraft animator with FOV, targets, and GSM state overlay.

    Parameters
    ----------
    pts : dict
        Aircraft wireframe geometry (from DefineTTwistor).
    scale : float
        Scale factor for body-frame wireframe (default 10).
    axis_half_range : tuple
        Half-widths of view window (East, North, Alt) in metres.
    fov_radius_fn : callable or None
        Function ``fov_radius_fn(aircraft_state_est_0_3) -> (origin, radius)``.
        Pass your existing ``build_aircraft_fov``.  If None, FOV is not drawn.
    trail_len : int
        Number of past positions shown per target trail.
    """

    SCALE = 10

    def __init__(self, pts,
                 scale=10,
                 axis_half_range=(500, 500, 200),
                 fov_radius_fn=None,
                 trail_len=TRAIL_LEN):

        self.pts          = pts
        self.scale        = scale
        self._axis_half   = np.array(axis_half_range, dtype=float)
        self._fov_fn      = fov_radius_fn
        self._trail_len   = trail_len

        # figure / axes
        self._fig     = None
        self._ax      = None
        self._ax2d    = None   # inset 2-D top-down view

        # aircraft handles
        self._handles = {}
        self._axis_vec = None

        # overlay handles (created lazily)
        self._fov_handle    = None   # FOV circle line
        self._orbit_handle  = None   # search/track orbit circle
        self._tgt_dots      = {}     # target_id -> dot handle (3d)
        self._tgt_trails    = {}     # target_id -> trail handle (3d)
        self._tgt_found_ring= {}     # target_id -> highlight ring handle
        self._state_text    = None   # 3-D text annotation

        # 2-D inset handles
        self._inset_ac      = None
        self._inset_fov     = None
        self._inset_orbit   = None
        self._inset_dots    = {}
        self._inset_trails  = {}
        self._inset_found   = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, time, aircraft_state,
               targets=None,
               found_targets=None,
               gsm=None,
               aircraft_state_est=None):
        """
        Draw or update everything.

        Parameters
        ----------
        time              : float  - current sim time (s)
        aircraft_state    : array  - full 12-state vector (true state)
        targets           : list of MovingTarget  - all targets in world
        found_targets     : list of MovingTarget  - subset currently in FOV
        gsm               : GuidanceStateMachine instance (for state label & orbit)
        aircraft_state_est: array  - estimated state (used for FOV centre if provided)
        """
        targets       = targets       or []
        found_targets = found_targets or []
        found_ids     = {t.id for t in found_targets}

        state = np.asarray(aircraft_state, dtype=float)
        pn, pe, pd       = state[0], state[1], state[2]
        phi, theta, psi  = state[3], state[4], state[5]

        # Use estimated state for FOV if available, else true state
        pos_for_fov = np.asarray(aircraft_state[:3]) if aircraft_state is not None \
                      else np.array([pn, pe, pd])

        if self._fig is None:
            self._init_figure(pn, pe, pd, phi, theta, psi)

        # --- aircraft body ---
        self._update_bodies(pn, pe, pd, phi, theta, psi)
        self._update_axis(pe, pn, pd)

        # --- FOV circle ---
        if self._fov_fn is not None:
            self._update_fov(pos_for_fov)

        # --- targets ---
        if targets:
            self._update_targets(targets, found_ids)

        # --- orbit circle (search / track) ---
        if gsm is not None:
            self._update_orbit(gsm)
            self._update_state_label(pn, pe, pd, gsm, time)

        # --- 2-D inset ---
        self._update_inset(pn, pe, pos_for_fov, targets, found_ids, gsm)

        plt.pause(0.001)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_figure(self, pn, pe, pd, phi, theta, psi):
        self._fig = plt.figure(20, figsize=(12, 8))
        self._fig.clf()

        # # Main 3-D axes (left 2/3)
        # self._ax = self._fig.add_axes([0.0, 0.0, 0.65, 1.0], projection='3d')

        # 2-D top-down inset (right 1/3)
        self._ax2d = self._fig.add_axes([0.67, 0.1, 0.30, 0.80])
        self._ax2d.set_aspect('equal')
        self._ax2d.set_title('Top-down view', fontsize=9)
        self._ax2d.set_xlabel('East (m)', fontsize=8)
        self._ax2d.set_ylabel('North (m)', fontsize=8)
        self._ax2d.tick_params(labelsize=7)
        self._ax2d.grid(True, alpha=0.3)

        # ax = self._ax

        # # Aircraft wireframe
        # for key in ('fuse', 'wing', 'tailwing', 'tail'):
        #     xyz = self._transform(self.pts[key], pn, pe, pd, phi, theta, psi)
        #     (h,) = ax.plot(xyz[0], xyz[1], xyz[2], 'k', linewidth=1.2)
        #     self._handles[key] = h

        # ax.set_title('Aircraft + Targets', fontsize=10)
        # ax.set_xlabel('East')
        # ax.set_ylabel('North')
        # ax.set_zlabel('Alt (m)')
        # ax.view_init(elev=35, azim=225)

        # hr = self._axis_half
        # self._axis_vec = np.array([
        #     pe - hr[0], pe + hr[0],
        #     pn - hr[1], pn + hr[1],
        #     -pd - hr[2], -pd + hr[2],
        # ])
        # self._apply_axis()
        # ax.grid(True)

        # Legend entries
        legend_elements = [
            mpatches.Patch(facecolor='none', edgecolor='k', label='Aircraft'),
            mpatches.Patch(facecolor='cyan',  alpha=0.25,   label='FOV'),
            mpatches.Patch(facecolor='red',                 label='Target (undetected)'),
            mpatches.Patch(facecolor='lime',                label='Target (in FOV)'),
            mpatches.Patch(facecolor='none', edgecolor='orange', linestyle='--', label='Search/track orbit'),
        ]
        # ax.legend(handles=legend_elements, loc='upper left', fontsize=7)

        # 2-D inset: aircraft position dot
        (self._inset_ac,) = self._ax2d.plot(pe, pn, 'k^', markersize=6, zorder=5, label='Aircraft')
        self._ax2d.legend(fontsize=7, loc='upper right')

    # ------------------------------------------------------------------
    # Aircraft body
    # ------------------------------------------------------------------

    def _update_bodies(self, pn, pe, pd, phi, theta, psi):
        for key in ('fuse', 'wing', 'tailwing', 'tail'):
            xyz = self._transform(self.pts[key], pn, pe, pd, phi, theta, psi)
            h = self._handles[key]
            h.set_data(xyz[0], xyz[1])
            h.set_3d_properties(xyz[2])

    # ------------------------------------------------------------------
    # FOV circle
    # ------------------------------------------------------------------

    def _update_fov(self, pos_for_fov):
        origin, r = self._fov_fn(pos_for_fov)
        ns, es, ds = _make_circle_ned(origin[0], origin[1], r)
        xs, ys, zs = _ned_to_xyz_arr(ns, es, ds)

        if self._fov_handle is None:
            # 3-D circle
            (self._fov_handle,) = self._ax.plot(
                xs, ys, zs, color='cyan', alpha=0.6, linewidth=1.5,
                linestyle='-', label='FOV'
            )
            # Filled disc on ground (semi-transparent)
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            verts = [list(zip(xs, ys, zs))]
            self._fov_fill = Poly3DCollection(verts, alpha=0.08, facecolor='cyan',
                                              edgecolor='none')
            self._ax.add_collection3d(self._fov_fill)

            # 2-D inset FOV circle
            self._inset_fov = plt.Circle(
                (origin[1], origin[0]), r,
                color='cyan', alpha=0.25, fill=True, linewidth=1
            )
            self._ax2d.add_patch(self._inset_fov)
        else:
            self._fov_handle.set_data(xs, ys)
            self._fov_handle.set_3d_properties(zs)

            # Update fill (cheap: remove + re-add)
            self._fov_fill.remove()
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            verts = [list(zip(xs, ys, zs))]
            self._fov_fill = Poly3DCollection(verts, alpha=0.08, facecolor='cyan',
                                              edgecolor='none')
            self._ax.add_collection3d(self._fov_fill)

            # 2-D inset
            self._inset_fov.center = (origin[1], origin[0])
            self._inset_fov.set_radius(r)

    # ------------------------------------------------------------------
    # Targets
    # ------------------------------------------------------------------

    def _update_targets(self, targets, found_ids):
        for tgt in targets:
            tid = tgt.id
            color     = 'lime' if tid in found_ids else 'red'
            zorder_3d = 6      if tid in found_ids else 4

            # ---- 3-D dot ----
            xe, yn = tgt.x, tgt.y
            if tid not in self._tgt_dots:
                (dot,) = self._ax.plot(
                    [xe], [yn], [0.0],
                    'o', color=color, markersize=5, zorder=zorder_3d
                )
                self._tgt_dots[tid] = dot
            else:
                dot = self._tgt_dots[tid]
                dot.set_data([xe], [yn])
                dot.set_3d_properties([0.0])
                dot.set_color(color)

            # ---- 3-D trail ----
            trail_n = list(tgt.history_y)[-self._trail_len:]
            trail_e = list(tgt.history_x)[-self._trail_len:]
            trail_z = [0.0] * len(trail_n)
            if tid not in self._tgt_trails:
                (tr,) = self._ax.plot(
                    trail_e, trail_n, trail_z,
                    '-', color=color, alpha=0.35, linewidth=0.8
                )
                self._tgt_trails[tid] = tr
            else:
                tr = self._tgt_trails[tid]
                tr.set_data(trail_e, trail_n)
                tr.set_3d_properties(trail_z)
                tr.set_color(color)

            # ---- found ring (larger highlight when in FOV) ----
            if tid in found_ids:
                if tid not in self._tgt_found_ring:
                    (ring,) = self._ax.plot(
                        [xe], [yn], [0.0],
                        'o', mfc='none', mec='lime',
                        markersize=12, linewidth=1.5, zorder=7
                    )
                    self._tgt_found_ring[tid] = ring
                else:
                    ring = self._tgt_found_ring[tid]
                    ring.set_data([xe], [yn])
                    ring.set_3d_properties([0.0])
                    ring.set_visible(True)
            elif tid in self._tgt_found_ring:
                self._tgt_found_ring[tid].set_visible(False)

            # ---- 2-D inset dot ----
            if tid not in self._inset_dots:
                (d2,) = self._ax2d.plot(
                    [xe], [yn], 'o', color=color, markersize=4, zorder=5
                )
                self._inset_dots[tid] = d2
            else:
                self._inset_dots[tid].set_data([xe], [yn])
                self._inset_dots[tid].set_color(color)

            # ---- 2-D trail ----
            if tid not in self._inset_trails:
                (t2,) = self._ax2d.plot(
                    trail_e, trail_n, '-', color=color, alpha=0.3, linewidth=0.7
                )
                self._inset_trails[tid] = t2
            else:
                self._inset_trails[tid].set_data(trail_e, trail_n)
                self._inset_trails[tid].set_color(color)

    # ------------------------------------------------------------------
    # Orbit circle  (search or track)
    # ------------------------------------------------------------------

    def _update_orbit(self, gsm):
        from GuidanceStateMachine import GuidanceState

        # Only draw orbit during SEARCH or TRACK
        if gsm.state in (GuidanceState.SEARCH, GuidanceState.TRACK):
            if gsm.state == GuidanceState.SEARCH:
                center = gsm.search_orbit_center
                radius = gsm.search_orbit_r
                col    = 'orange'
            else:
                center = gsm.track_center
                radius = gsm.p.track_r
                col    = 'magenta'

            ns, es, ds = _make_circle_ned(center[0], center[1], radius)
            xs, ys, zs = _ned_to_xyz_arr(ns, es, ds)
            zs = -center[2] * np.ones_like(zs)   # draw at flight altitude

            if self._orbit_handle is None:
                (self._orbit_handle,) = self._ax.plot(
                    xs, ys, zs, '--', color=col, linewidth=1.2, alpha=0.7
                )
                self._inset_orbit = plt.Circle(
                    (center[1], center[0]), radius,
                    color=col, alpha=0.15, fill=True,
                    linewidth=1.2, linestyle='--'
                )
                self._ax2d.add_patch(self._inset_orbit)
            else:
                self._orbit_handle.set_data(xs, ys)
                self._orbit_handle.set_3d_properties(zs)
                self._orbit_handle.set_color(col)
                self._orbit_handle.set_visible(True)

                self._inset_orbit.center = (center[1], center[0])
                self._inset_orbit.set_radius(radius)
                self._inset_orbit.set_color(col)
                self._inset_orbit.set_visible(True)
        else:
            # Hide orbit ring when not in SEARCH/TRACK
            if self._orbit_handle is not None:
                self._orbit_handle.set_visible(False)
            if self._inset_orbit is not None:
                self._inset_orbit.set_visible(False)

    # ------------------------------------------------------------------
    # State label
    # ------------------------------------------------------------------

    def _update_state_label(self, pn, pe, pd, gsm, time):
        label = (
            f"State: {gsm.state_name}\n"
            f"t = {time:.1f}s\n"
            f"orbit r = {gsm.search_orbit_r:.0f}m"
        )
        if self._state_text is None:
            self._state_text = self._ax.text2D(
                0.02, 0.95, label,
                transform=self._ax.transAxes,
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.6)
            )
        else:
            self._state_text.set_text(label)

    # ------------------------------------------------------------------
    # 2-D inset update
    # ------------------------------------------------------------------

    def _update_inset(self, pn, pe, pos_for_fov, targets, found_ids, gsm):
        # Aircraft marker
        self._inset_ac.set_data([pe], [pn])

        # Auto-zoom the inset around the aircraft + targets
        all_e = [pe] + [t.x for t in targets]
        all_n = [pn] + [t.y for t in targets]
        margin = 600
        self._ax2d.set_xlim(min(all_e) - margin, max(all_e) + margin)
        self._ax2d.set_ylim(min(all_n) - margin, max(all_n) + margin)

    # ------------------------------------------------------------------
    # Axis tracking  (identical to original DrawAircraft)
    # ------------------------------------------------------------------

    def _update_axis(self, pe, pn, pd):
        flag, new_vec = self._in_view(self._axis_vec, pn, pe, pd)
        if flag:
            self._axis_vec = new_vec
            self._apply_axis()

    def _apply_axis(self):
        v = self._axis_vec
        self._ax.set_xlim(v[0], v[1])
        self._ax.set_ylim(v[2], v[3])
        self._ax.set_zlim(v[4], v[5])

    # ------------------------------------------------------------------
    # Geometry helpers  (identical to original DrawAircraft)
    # ------------------------------------------------------------------

    def _transform(self, body_pts, pn, pe, pd, phi, theta, psi):
        ned = self.scale * self._rotate(body_pts, phi, theta, psi)
        ned = self._translate(ned, pn, pe, pd)
        return self._ned_to_xyz(ned)

    @staticmethod
    def _rotate(pts, phi, theta, psi):
        R_roll  = np.array([[1,0,0],[0,np.cos(phi),-np.sin(phi)],[0,np.sin(phi),np.cos(phi)]])
        R_pitch = np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
        R_yaw   = np.array([[np.cos(psi),-np.sin(psi),0],[np.sin(psi),np.cos(psi),0],[0,0,1]])
        return R_yaw @ R_pitch @ R_roll @ pts

    @staticmethod
    def _translate(pts, pn, pe, pd):
        return pts + np.array([[pn],[pe],[pd]])

    @staticmethod
    def _ned_to_xyz(ned):
        return np.array([[0,1,0],[1,0,0],[0,0,-1]]) @ ned

    @staticmethod
    def _in_view(axis_vec, pn, pe, pd):
        xp, yp, zp = pe, pn, -pd
        flag = 0
        axis_new = axis_vec.copy()
        if xp < axis_vec[0]:
            flag=1; dx=axis_vec[1]-axis_vec[0]; axis_new[0:2]=axis_vec[0:2]-dx
        elif xp > axis_vec[1]:
            flag=2; dx=axis_vec[1]-axis_vec[0]; axis_new[0:2]=axis_vec[0:2]+dx
        if yp < axis_new[2]:
            flag=3; dy=axis_vec[3]-axis_vec[2]; axis_new[2:4]=axis_vec[2:4]-dy
        elif yp > axis_new[3]:
            flag=4; dy=axis_vec[3]-axis_vec[2]; axis_new[2:4]=axis_vec[2:4]+dy
        if zp < axis_new[4]:
            flag=5; dz=axis_vec[5]-axis_vec[4]; axis_new[4:6]=axis_vec[4:6]-dz
        elif zp > axis_new[5]:
            flag=6; dz=axis_vec[5]-axis_vec[4]; axis_new[4:6]=axis_vec[4:6]+dz
        return flag, axis_new

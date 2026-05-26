"""
DrawAircraft - Python class equivalent of the MATLAB DrawAircraft.m function.

Renders a 3-D aircraft wireframe that follows the aircraft in real time.
The view window tracks the aircraft position, keeping it centred.

Usage
-----
# Create once (initialises the figure on first call)
drawer = DrawAircraft(pts)

# Call every time step inside your simulation loop
drawer.update(time, aircraft_state)

Parameters
----------
pts : dict
    Wireframe point arrays with keys 'fuse', 'wing', 'tailwing', 'tail'.
    Each value is an ndarray of shape (3, n_pts) in the body NED frame,
    as produced by DefineTTwistor.

aircraft_state : array-like, length 12
    [ pn, pe, pd, phi, theta, psi, u, v, w, p, q, r ]
    positions in metres, angles in radians, velocities in m/s and rad/s.

time : float
    Current simulation time (seconds).
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 – required for 3-D projection


class DrawAircraft:
    """
    Stateful aircraft drawing class.  Mirrors the MATLAB persistent-variable
    pattern by storing figure/axis/handle state on the instance.

    Parameters
    ----------
    pts : dict
        Aircraft wireframe geometry (from DefineTTwistor).
    scale : float, optional
        Scale factor applied to the body-frame wireframe (default 10,
        matching the MATLAB SCALE constant).
    axis_half_range : tuple of 3 floats, optional
        Half-widths of the view window in the East, North, and Alt directions.
        Default is (200, 200, 100), giving ±200 m East/North and ±100 m Alt.
    """

    SCALE = 10  # matches MATLAB constant

    def __init__(self, pts, scale=10, axis_half_range=(200, 200, 100)):
        self.pts = pts
        self.scale = scale
        self._axis_half = np.array(axis_half_range, dtype=float)

        # Persistent state (equivalent to MATLAB persistent variables)
        self._fig = None
        self._ax  = None
        self._handles = {}          # 'fuse', 'wing', 'tailwing', 'tail'
        self._axis_vec = None       # [xmin, xmax, ymin, ymax, zmin, zmax]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, time, aircraft_state):
        """
        Draw or update the aircraft at the given state.

        Parameters
        ----------
        time : float
        aircraft_state : array-like, length 12
        """
        state = np.asarray(aircraft_state, dtype=float)
        pn, pe, pd       = state[0], state[1], state[2]
        phi, theta, psi  = state[3], state[4], state[5]

        if self._fig is None:
            self._init_figure(pn, pe, pd, phi, theta, psi)
        else:
            self._update_bodies(pn, pe, pd, phi, theta, psi)
            self._update_axis(pe, pn, pd)

    # ------------------------------------------------------------------
    # Initialisation (first call)
    # ------------------------------------------------------------------

    def _init_figure(self, pn, pe, pd, phi, theta, psi):
        self._fig = plt.figure(20)
        self._fig.clf()
        self._ax = self._fig.add_subplot(111, projection='3d')
        ax = self._ax

        for key in ('fuse', 'wing', 'tailwing', 'tail'):
            xyz = self._transform(self.pts[key], pn, pe, pd, phi, theta, psi)
            (h,) = ax.plot(xyz[0], xyz[1], xyz[2], 'k')
            self._handles[key] = h

        ax.set_title('Aircraft')
        ax.set_xlabel('East')
        ax.set_ylabel('North')
        ax.set_zlabel('-Down')
        ax.view_init(elev=47, azim=32)

        # Initial axis window centred on aircraft
        hr = self._axis_half
        self._axis_vec = np.array([
            pe - hr[0], pe + hr[0],
            pn - hr[1], pn + hr[1],
            -pd - hr[2], -pd + hr[2],
        ])
        self._apply_axis()
        ax.grid(True)
        plt.pause(0.001)

    # ------------------------------------------------------------------
    # Per-step update
    # ------------------------------------------------------------------

    def _update_bodies(self, pn, pe, pd, phi, theta, psi):
        for key in ('fuse', 'wing', 'tailwing', 'tail'):
            xyz = self._transform(self.pts[key], pn, pe, pd, phi, theta, psi)
            h = self._handles[key]
            h.set_data(xyz[0], xyz[1])
            h.set_3d_properties(xyz[2])
        plt.pause(0.001)

    def _update_axis(self, pe, pn, pd):
        """Re-centre the view window on the aircraft (sliding window)."""
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
    # Geometry helpers
    # ------------------------------------------------------------------

    def _transform(self, body_pts, pn, pe, pd, phi, theta, psi):
        """Scale, rotate, translate body pts then convert NED → display XYZ."""
        ned = self.scale * self._rotate(body_pts, phi, theta, psi)
        ned = self._translate(ned, pn, pe, pd)
        return self._ned_to_xyz(ned)

    @staticmethod
    def _rotate(pts, phi, theta, psi):
        R_roll = np.array([
            [1,          0,           0         ],
            [0,  np.cos(phi), -np.sin(phi)],
            [0,  np.sin(phi),  np.cos(phi)],
        ])
        R_pitch = np.array([
            [ np.cos(theta), 0, np.sin(theta)],
            [             0, 1,             0],
            [-np.sin(theta), 0, np.cos(theta)],
        ])
        R_yaw = np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi),  np.cos(psi), 0],
            [          0,            0, 1],
        ])
        R = R_yaw @ R_pitch @ R_roll
        return R @ pts

    @staticmethod
    def _translate(pts, pn, pe, pd):
        return pts + np.array([[pn], [pe], [pd]])

    @staticmethod
    def _ned_to_xyz(ned):
        """NED → matplotlib display frame (X=East, Y=North, Z=-Down)."""
        R = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, -1],
        ])
        return R @ ned

    # ------------------------------------------------------------------
    # View-tracking helper  (mirrors in_view sub-function)
    # ------------------------------------------------------------------

    @staticmethod
    def _in_view(axis_vec, pn, pe, pd):
        """
        Check whether the aircraft is inside the current axis window.
        If not, shift the window by one full width in the offending direction.

        Returns
        -------
        flag : int   0 = in view, non-zero = which boundary was exceeded
        axis_new : ndarray  updated axis vector [xmin,xmax,ymin,ymax,zmin,zmax]
        """
        xp = pe        # display X = East
        yp = pn        # display Y = North
        zp = -pd       # display Z = -Down = altitude

        flag = 0
        axis_new = axis_vec.copy()

        # East (X)
        if xp < axis_vec[0]:
            flag = 1
            dx = axis_vec[1] - axis_vec[0]
            axis_new[0:2] = axis_vec[0:2] - dx
        elif xp > axis_vec[1]:
            flag = 2
            dx = axis_vec[1] - axis_vec[0]
            axis_new[0:2] = axis_vec[0:2] + dx

        # North (Y)
        if yp < axis_new[2]:
            flag = 3
            dy = axis_vec[3] - axis_vec[2]
            axis_new[2:4] = axis_vec[2:4] - dy
        elif yp > axis_new[3]:
            flag = 4
            dy = axis_vec[3] - axis_vec[2]
            axis_new[2:4] = axis_vec[2:4] + dy

        # Altitude (Z)
        if zp < axis_new[4]:
            flag = 5
            dz = axis_vec[5] - axis_vec[4]
            axis_new[4:6] = axis_vec[4:6] - dz
        elif zp > axis_new[5]:
            flag = 6
            dz = axis_vec[5] - axis_vec[4]
            axis_new[4:6] = axis_vec[4:6] + dz

        return flag, axis_new


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # Import the geometry class from the companion module if available,
    # otherwise fall back to a minimal stub so this file is self-contained.
    try:
        #from define_ttwistor import DefineTTwistor
        import DefineTTwistor
        pts = DefineTTwistor().pts
    except ImportError:
        # Minimal stub: single-point wireframes so the class runs standalone
        pts = {k: np.zeros((3, 2)) for k in ('fuse', 'wing', 'tailwing', 'tail')}

    drawer = DrawAircraft(pts)

    # Simulate a simple climbing turn
    dt = 0.05
    t  = 0.0
    pn, pe, pd   = 0.0, 0.0, -50.0
    phi, theta, psi = 0.0, 0.05, 0.0
    Va = 15.0

    for _ in range(300):
        psi += 0.02
        pn  += Va * np.cos(psi) * dt
        pe  += Va * np.sin(psi) * dt
        pd  -= Va * np.sin(theta) * dt

        state = [pn, pe, pd, phi, theta, psi, Va, 0, 0, 0, 0, 0.02]
        drawer.update(t, state)
        t += dt

    plt.show()
"""
AnimateSimulation - Python class equivalent of the MATLAB AnimateSimulation function.

Original MATLAB code created by Eric W. Frew, March 7, 2013, for ASEN 3128.
Based on code originally created by Randy Beard and Tim McLain.

This class animates a 3D aircraft simulation given a time vector and state array,
as would be output from an ODE solver (e.g., scipy.integrate.solve_ivp).

State array columns:
    xarray[:, 0:3]  -> position (pn, pe, pd) in NED frame
    xarray[:, 3:6]  -> Euler angles (phi, theta, psi)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class AnimateSimulation:
    """
    Animates a 3D aircraft flight simulation.

    Parameters
    ----------
    tout : array-like
        Time vector from simulation (currently unused but included for future use).
    xarray : ndarray, shape (m, n)
        State array where each row is a time step.
        Columns 0:3 are NED position (pn, pe, pd).
        Columns 3:6 are Euler angles (phi, theta, psi) in radians.
    """

    def __init__(self, tout, xarray):
        self.tout = np.asarray(tout)
        self.xarray = np.asarray(xarray)
        self._setup_axes()
        self._define_aircraft_pts()
        self._run()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_axes(self):
        """Compute axis limits and scale from the simulation path."""
        xarray = self.xarray
        # Stack north, east, and altitude (-down) columns
        path = np.column_stack([xarray[:, 0], xarray[:, 1], -xarray[:, 2]])

        max_p = path.max(axis=0)
        min_p = path.min(axis=0)
        range_p = max_p - min_p
        cent_p = (max_p + min_p) / 2.0
        max_dim = range_p.max()

        self.scale_plot = max_dim / 50.0

        axis_min = cent_p - 1.05 * max_dim * np.ones(3)
        axis_max = cent_p + 1.05 * max_dim * np.ones(3)

        del_z = 1.05 * max(0.2 * max_dim, range_p[2])
        axis_min[2] = cent_p[2] - del_z
        axis_max[2] = cent_p[2] + del_z

        self.axis_min = axis_min
        self.axis_max = axis_max

    # ------------------------------------------------------------------
    # Aircraft geometry definition
    # ------------------------------------------------------------------

    def _define_aircraft_pts(self):
        """Define the aircraft wireframe points (equivalent to DefineAircraftPts)."""
        fuse_h = 1.0
        fuse_w = 1.0
        fuse_l1 = 3.0
        fuse_l2 = 1.0
        fuse_l3 = 10.0

        wing_l = 1.5
        wing_w = 9.0

        tailwing_l = 1.0
        tailwing_w = 4.0

        tail_h = 3.0

        # Vertices defined as (N, E, D)
        V = np.array([
            [fuse_l1,                  0,              0],           # 1
            [fuse_l2,          fuse_w / 2,    -fuse_h / 2],          # 2
            [fuse_l2,         -fuse_w / 2,    -fuse_h / 2],          # 3
            [fuse_l2,         -fuse_w / 2,     fuse_h / 2],          # 4
            [fuse_l2,          fuse_w / 2,     fuse_h / 2],          # 5
            [-fuse_l3,                 0,              0],           # 6
            [0,            wing_w / 2,              0],              # 7
            [-wing_l,      wing_w / 2,              0],              # 8
            [-wing_l,     -wing_w / 2,              0],              # 9
            [0,           -wing_w / 2,              0],              # 10
            [-fuse_l3 + tailwing_l,  tailwing_w / 2,  0],           # 11
            [-fuse_l3,               tailwing_w / 2,  0],           # 12
            [-fuse_l3,              -tailwing_w / 2,  0],           # 13
            [-fuse_l3 + tailwing_l, -tailwing_w / 2,  0],           # 14
            [-fuse_l3 + tailwing_l,              0,   0],           # 15
            [-fuse_l3,                           0,  -tail_h],      # 16
        ])

        # Indices are 0-based (subtract 1 from MATLAB indices)
        self.pts = {
            'fuse': V[[0, 1, 5, 2, 0, 4, 5, 4, 0]].T,          # V(1,2,6,3,1,5,6,5,1)
            'wing': V[[6, 7, 8, 9, 6]].T,                        # V(7,8,9,10,7)
            'tailwing': V[[10, 11, 12, 13, 10]].T,               # V(11,12,13,14,11)
            'tail': V[[5, 14, 15, 5]].T,                          # V(6,15,16,6)
        }

    # ------------------------------------------------------------------
    # Rotation / translation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rotation_matrix(phi, theta, psi):
        """Body-to-inertial rotation matrix (ZYX Euler convention)."""
        R_roll = np.array([
            [1,          0,           0],
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
        return R_yaw @ R_pitch @ R_roll

    @staticmethod
    def _rotate(pts, phi, theta, psi):
        """Rotate a set of NED points by Euler angles."""
        R = AnimateSimulation._rotation_matrix(phi, theta, psi)
        return R @ pts

    @staticmethod
    def _translate(pts, pn, pe, pd):
        """Translate a set of NED points by (pn, pe, pd)."""
        return pts + np.array([[pn], [pe], [pd]])

    @staticmethod
    def _ned_to_xyz(ned):
        """
        Convert NED coordinates to the matplotlib XYZ display frame.
        MATLAB convention: X=East, Y=North, Z=-Down.
        """
        R = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, -1],
        ])
        return R @ ned

    # ------------------------------------------------------------------
    # Draw helpers
    # ------------------------------------------------------------------

    def _transform_body(self, body_pts, pos, att):
        """
        Apply scale, rotation, translation, and NED->XYZ conversion to body points.

        Parameters
        ----------
        body_pts : ndarray, shape (3, n)
        pos : array-like, length 3  (pn, pe, pd)
        att : array-like, length 3  (phi, theta, psi)

        Returns
        -------
        xyz : ndarray, shape (3, n)
        """
        pn, pe, pd = pos
        phi, theta, psi = att

        ned = self.scale_plot * self._rotate(body_pts, phi, theta, psi)
        ned = self._translate(ned, pn, pe, pd)
        return self._ned_to_xyz(ned)

    def _init_aircraft_handles(self, pos, att):
        """Draw the initial aircraft wireframe and return plot handles."""
        handles = {}
        for key in ('fuse', 'wing', 'tailwing', 'tail'):
            xyz = self._transform_body(self.pts[key], pos, att)
            (h,) = self.ax.plot(xyz[0], xyz[1], xyz[2], 'k')
            handles[key] = h
        return handles

    def _update_aircraft(self, pos, att):
        """Update aircraft wireframe handles to a new position/attitude."""
        for key in ('fuse', 'wing', 'tailwing', 'tail'):
            xyz = self._transform_body(self.pts[key], pos, att)
            h = self.aircraft_handles[key]
            h.set_data(xyz[0], xyz[1])
            h.set_3d_properties(xyz[2])

    def _ned_pos_to_xyz(self, pos):
        """Convert a single NED position vector to display XYZ."""
        return self._ned_to_xyz(np.array(pos).reshape(3, 1)).flatten()

    def _init_lines(self, pos):
        """Draw initial projection dashed lines and return handles."""
        xyz_pos = self._ned_pos_to_xyz(pos)

        # Vertical line to XY (ground) plane
        bot = self._ned_to_xyz(np.array([[pos[0]], [pos[1]], [-self.axis_min[2]]])).flatten()
        (h1,) = self.ax.plot(
            [xyz_pos[0], bot[0]], [xyz_pos[1], bot[1]], [xyz_pos[2], bot[2]], 'ko-.'
        )

        # Horizontal line to YZ (side) wall
        side = self._ned_to_xyz(np.array([[self.axis_max[0]], [pos[1]], [pos[2]]])).flatten()
        (h2,) = self.ax.plot(
            [xyz_pos[0], side[0]], [xyz_pos[1], side[1]], [xyz_pos[2], side[2]], 'ko-.'
        )
        return h1, h2

    def _update_lines(self, pos):
        """Update projection line handles."""
        xyz_pos = self._ned_pos_to_xyz(pos)

        bot = self._ned_to_xyz(np.array([[pos[0]], [pos[1]], [-self.axis_min[2]]])).flatten()
        self.line_handle1.set_data([xyz_pos[0], bot[0]], [xyz_pos[1], bot[1]])
        self.line_handle1.set_3d_properties([xyz_pos[2], bot[2]])

        side = self._ned_to_xyz(np.array([[self.axis_max[0]], [pos[1]], [pos[2]]])).flatten()
        self.line_handle2.set_data([xyz_pos[0], side[0]], [xyz_pos[1], side[1]])
        self.line_handle2.set_3d_properties([xyz_pos[2], side[2]])

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def _run(self):
        """Set up the figure, draw the static path, and animate the aircraft."""
        xarray = self.xarray
        axis_min = self.axis_min
        axis_max = self.axis_max

        fig = plt.figure(21)
        fig.clf()
        self.ax = fig.add_subplot(111, projection='3d')
        ax = self.ax

        # Static 3-D path and projections (East=X, North=Y, Alt=-Down=Z)
        north = xarray[:, 0]
        east  = xarray[:, 1]
        alt   = -xarray[:, 2]

        ax.plot(east, north, alt, 'b--', label='Path')
        ax.plot(east, north, axis_min[2] * np.ones_like(north), 'g--', label='Ground projection')
        ax.plot(east, axis_max[0] * np.ones_like(north), alt, 'g--', label='Side projection')

        # Initial state
        pos0 = xarray[0, 0:3]
        att0 = xarray[0, 3:6]

        self.aircraft_handles = self._init_aircraft_handles(pos0, att0)
        self.line_handle1, self.line_handle2 = self._init_lines(pos0)

        ax.set_title('Aircraft')
        ax.set_xlabel('East')
        ax.set_ylabel('North')
        ax.set_zlabel('-Down')
        ax.view_init(elev=47, azim=32)
        ax.set_xlim(axis_min[1], axis_max[1])
        ax.set_ylim(axis_min[0], axis_max[0])
        ax.set_zlim(axis_min[2], axis_max[2])
        ax.grid(True)

        plt.pause(0.001)

        # Animation loop
        m = xarray.shape[0]
        for i in range(1, m):
            pos = xarray[i, 0:3]
            att = xarray[i, 3:6]
            self._update_aircraft(pos, att)
            self._update_lines(pos)
            plt.pause(0.001)

        plt.show()


# ------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------
if __name__ == '__main__':
    # Generate a simple helical trajectory for demonstration
    t = np.linspace(0, 20, 400)
    radius = 100.0
    altitude_rate = 5.0

    pn    =  radius * np.cos(0.3 * t)
    pe    =  radius * np.sin(0.3 * t)
    pd    = -altitude_rate * t          # negative = climbing

    phi   =  np.zeros_like(t)
    theta =  np.deg2rad(5) * np.ones_like(t)
    psi   =  0.3 * t + np.pi / 2

    xarray = np.column_stack([pn, pe, pd, phi, theta, psi])

    AnimateSimulation(tout=t, xarray=xarray)
"""
PlotSimulationWithCommands - Python equivalent of the MATLAB function of the same name.

Produces a comprehensive set of diagnostic plots for a UAV simulation, including:
  - Fig 3:  Inertial position components (pn, pe, pd)
  - Fig 4:  Euler angles (roll, pitch, yaw)
  - Fig 5:  Body-frame velocities (u, v, w)
  - Fig 6:  Body-frame angular rates (p, q, r)
  - Fig 7:  Control surface deflections (de, da, dr, dt)
  - Fig 8:  3-D flight path
  - Fig 9:  Full 9×2 state-and-command dashboard

Dependencies
------------
The wind-angle and flight-path-angle helper functions must be supplied
externally (or the stub implementations in this file can be replaced):
  - TransformFromInertialToBody(wind_inertial, euler_angles) -> (3,) array
  - AirRelativeVelocityVectorToWindAngles(v_body) -> [Va, beta, alpha]
  - FlightPathAnglesFromState(state_12) -> [something, chi, gamma]

Usage
-----
plotter = PlotSimulationWithCommands()
plotter.plot(time, aircraft_state_array, control_input_array,
             background_wind_array, state_command_array, color='b')

Arrays are expected as (rows × time_steps) ndarrays, matching the MATLAB
convention where each column is one time step.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ---------------------------------------------------------------------------
# Stub helpers – replace with real implementations as needed
# ---------------------------------------------------------------------------

def TransformFromInertialToBody(wind_inertial, euler_angles):
    """
    Rotate a vector from the inertial NED frame to the body frame.

    Parameters
    ----------
    wind_inertial : array-like, shape (3,)
    euler_angles  : array-like, shape (3,)   [phi, theta, psi]

    Returns
    -------
    wind_body : ndarray, shape (3,)
    """
    phi, theta, psi = euler_angles
    R_roll = np.array([
        [1,           0,            0],
        [0,  np.cos(phi),  np.sin(phi)],
        [0, -np.sin(phi),  np.cos(phi)],
    ])
    R_pitch = np.array([
        [ np.cos(theta), 0, -np.sin(theta)],
        [             0, 1,              0],
        [ np.sin(theta), 0,  np.cos(theta)],
    ])
    R_yaw = np.array([
        [ np.cos(psi), np.sin(psi), 0],
        [-np.sin(psi), np.cos(psi), 0],
        [           0,           0, 1],
    ])
    R_body_from_inertial = R_roll @ R_pitch @ R_yaw
    return R_body_from_inertial @ np.asarray(wind_inertial, dtype=float)


def AirRelativeVelocityVectorToWindAngles(v_air_body):
    """
    Convert air-relative velocity in body frame to wind angles.

    Returns [Va, beta, alpha].
    """
    ur, vr, wr = v_air_body
    Va    = np.sqrt(ur**2 + vr**2 + wr**2)
    beta  = np.arcsin(vr / Va) if Va > 1e-6 else 0.0
    alpha = np.arctan2(wr, ur)
    return np.array([Va, beta, alpha])


def FlightPathAnglesFromState(state_12):
    """
    Compute flight-path angles from the 12-element state vector.

    Returns [something, chi, gamma] where:
      chi   = course angle (rad)
      gamma = flight-path angle (rad)
    """
    pn, pe, pd       = state_12[0], state_12[1], state_12[2]
    phi, theta, psi  = state_12[3], state_12[4], state_12[5]
    u, v, w          = state_12[6], state_12[7], state_12[8]

    # Inertial velocity
    R_roll = np.array([
        [1,           0,            0],
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
    vel_inertial = R @ np.array([u, v, w])

    Vg    = np.linalg.norm(vel_inertial)
    chi   = np.arctan2(vel_inertial[1], vel_inertial[0])
    gamma = np.arctan2(-vel_inertial[2], np.sqrt(vel_inertial[0]**2 + vel_inertial[1]**2))
    return np.array([0.0, chi, gamma])


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PlotSimulationWithCommands:
    """
    Generates the full suite of diagnostic plots for a UAV simulation run.

    All plots use matplotlib figure numbers that mirror the original MATLAB
    figure numbers (3–9) so results can be compared side-by-side.
    """

    def __init__(self):
        self._legend_handles = []
        self._legend_labels  = []

    def plot(self,
             time,
             aircraft_state_array,
             control_input_array,
             background_wind_array,
             state_command_array,
             color='b',
             label=None):
        """
        Produce all diagnostic plots.

        Parameters
        ----------
        time : array-like, shape (N,)
        aircraft_state_array  : ndarray, shape (12, N)
        control_input_array   : ndarray, shape (4,  N)
        background_wind_array : ndarray, shape (3+, N)
        state_command_array   : ndarray, shape (12, N)
        color : str
            Matplotlib color/linestyle string, e.g. 'b', 'r--'.
        """
        self._legend_handles = []
        self._legend_labels  = []
        t   = np.asarray(time, dtype=float)
        st  = np.asarray(aircraft_state_array, dtype=float)
        ci  = np.asarray(control_input_array,  dtype=float)
        bw  = np.asarray(background_wind_array, dtype=float)
        sc  = np.asarray(state_command_array,   dtype=float)

        self._plot_position(t, st, color,label)
        self._plot_euler(t, st, color,label)
        self._plot_velocity(t, st, color,label)
        self._plot_angular_rate(t, st, color,label)
        self._plot_controls(t, ci, color,label)
        self._plot_3d_path(st, color)
        self._plot_dashboard(t, st, ci, bw, sc, color,label)

    # ------------------------------------------------------------------
    # Fig 3 – Inertial position
    # ------------------------------------------------------------------
    def _plot_position(self, t, st, col,label):
        #fig = plt.figure(3)
        labels = ['$x_E$', '$y_E$', '$z_E$']
        rows   = [st[0], st[1], st[2]]
        for i, (lbl, data) in enumerate(zip(labels, rows), start=1):
            fig, ax = self._get_or_create_axes(3, 3, 1, i)
            #ax = fig.add_subplot(3, 1, i)
            ax.plot(t, data, col, label = label)
            ax.set_ylabel(lbl)
            if i == 1:
                ax.set_title('Inertial Position Components')
            if i == 3:
                ax.set_xlabel('time [sec]')
            ax.legend()
    # ------------------------------------------------------------------
    # Fig 4 – Euler angles
    # ------------------------------------------------------------------
    def _plot_euler(self, t, st, col,label):
        #fig = plt.figure(4)
        labels = ['roll [deg]', 'pitch [deg]', 'yaw [deg]']
        rows   = [st[3], st[4], st[5]]
        for i, (lbl, data) in enumerate(zip(labels, rows), start=1):
            #ax = fig.add_subplot(3, 1, i)
            fig, ax = self._get_or_create_axes(4, 3, 1, i)
            ax.plot(t, np.rad2deg(data), col, label=label)
            ax.set_ylabel(lbl)
            if i == 1:
                ax.set_title('Euler Angles')
            if i == 3:
                ax.set_xlabel('time [sec]')
            ax.legend()
    # ------------------------------------------------------------------
    # Fig 5 – Body velocities
    # ------------------------------------------------------------------
    def _plot_velocity(self, t, st, col,label):
        #fig = plt.figure(5)
        labels = ['$u^E$', '$v^E$', '$w^E$']
        rows   = [st[6], st[7], st[8]]
        for i, (lbl, data) in enumerate(zip(labels, rows), start=1):
            # ax = fig.add_subplot(3, 1, i)
            fig, ax = self._get_or_create_axes(5, 3, 1, i)
            ax.plot(t, data, col, label=label)
            ax.set_ylabel(lbl)
            if i == 1:
                ax.set_title('Inertial Velocity in Body Coordinates')
            if i == 3:
                ax.set_xlabel('time [sec]')
            ax.legend()
    # ------------------------------------------------------------------
    # Fig 6 – Body angular rates
    # ------------------------------------------------------------------
    def _plot_angular_rate(self, t, st, col,label):
        #fig = plt.figure(6)
        labels = ['p [deg/sec]', 'q [deg/sec]', 'r [deg/sec]']
        rows   = [st[9], st[10], st[11]]
        for i, (lbl, data) in enumerate(zip(labels, rows), start=1):
            #ax = fig.add_subplot(3, 1, i)
            fig, ax = self._get_or_create_axes(6, 3, 1, i)
            ax.plot(t, np.rad2deg(data), col, label=label)
            ax.set_ylabel(lbl)
            if i == 1:
                ax.set_title('Inertial Angular Velocity in Body Coordinates')
            if i == 3:
                ax.set_xlabel('time [sec]')
            ax.legend()
    # ------------------------------------------------------------------
    # Fig 7 – Control surfaces
    # ------------------------------------------------------------------
    def _plot_controls(self, t, ci, col,label):
        #fig = plt.figure(7)
        labels = ['de [deg]', 'da [deg]', 'dr [deg]', 'dt']
        scale  = [True, True, True, False]   # True = convert rad→deg
        for i, (lbl, rad) in enumerate(zip(labels, scale), start=1):
            #ax = fig.add_subplot(4, 1, i)
            fig, ax = self._get_or_create_axes(7, 4, 1, i)
            data = np.rad2deg(ci[i - 1]) if rad else ci[i - 1]
            ax.plot(t, data, col,label=label)
            ax.set_ylabel(lbl)
            if i == 1:
                ax.set_title('Control Surfaces')
            if i == 4:
                ax.set_xlabel('time [sec]')
            ax.legend()

    # ------------------------------------------------------------------
    # Fig 8 – 3-D flight path
    # ------------------------------------------------------------------
    def _plot_3d_path(self, st, col):
        fig = plt.figure(8)
        if not fig.axes:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.axes[0]

        ax.plot(st[0], st[1], -st[2], col)
        ax.plot([st[0, 0]], [st[1, 0]], [-st[2, 0]],
                'ks', markerfacecolor='g', label='Start')
        ax.plot([st[0, -1]], [st[1, -1]], [-st[2, -1]],
                'ko', markerfacecolor='r', label='End')
        ax.set_xlabel('North')
        ax.set_ylabel('East')
        ax.set_zlabel('Altitude (-Down)')
        ax.legend()

    # ------------------------------------------------------------------
    # Wind-angle computations (vectorised)
    # ------------------------------------------------------------------
    def _compute_wind_angles(self, st, bw):
        N = st.shape[1]
        Va_arr    = np.zeros(N)
        beta_arr  = np.zeros(N)
        alpha_arr = np.zeros(N)
        chi_arr   = np.zeros(N)
        gamma_arr = np.zeros(N)

        for i in range(N):
            wind_body  = TransformFromInertialToBody(bw[0:3, i], st[3:6, i])
            v_air      = st[6:9, i] - wind_body
            wa         = AirRelativeVelocityVectorToWindAngles(v_air)
            Va_arr[i]    = wa[0]
            beta_arr[i]  = np.rad2deg(wa[1])
            alpha_arr[i] = np.rad2deg(wa[2])

            fa = FlightPathAnglesFromState(st[:, i])
            chi_arr[i]   = np.rad2deg(fa[1])
            gamma_arr[i] = np.rad2deg(fa[2])

        return Va_arr, alpha_arr, beta_arr, chi_arr, gamma_arr

    # ------------------------------------------------------------------
    # Fig 9 – Full 9×2 dashboard
    # ------------------------------------------------------------------
    def _plot_dashboard(self, t, st, ci, bw, sc, col, label):

        Va_arr, alpha_arr, beta_arr, chi_arr, gamma_arr = \
            self._compute_wind_angles(st, bw)

        fig = plt.figure(9, figsize=(14, 22))  # wider for legend

        if not fig.axes:
            gs = GridSpec(9, 2, figure=fig, hspace=0.6, wspace=0.35)
            for row in range(9):
                for col_idx in range(2):
                    fig.add_subplot(gs[row, col_idx])

        axes = fig.axes

        # --- store legend handles ---
        handles = self._legend_handles
        labels  = self._legend_labels
        def gplot(ax, y, ylabel, y_cmd=None):
            line, = ax.plot(t, y, col, label=label)

            if label not in labels:
                handles.append(line)
                labels.append(label)

            if y_cmd is not None:
                cmd_line, = ax.plot(t, y_cmd, 'g--', label='Command')
                if 'Command' not in labels:
                    handles.append(cmd_line)
                    labels.append('Command')

            ax.set_ylabel(ylabel, rotation=0, labelpad=25)

        # --- plotting ---
        gplot(axes[0],  st[0],                       '$p_n$')
        gplot(axes[1],  Va_arr,                      '$V_a$',     sc[3])
        gplot(axes[2],  st[1],                       '$p_e$')
        gplot(axes[3],  alpha_arr,                   r'$\alpha$', np.rad2deg(sc[4]))
        gplot(axes[4],  -st[2],                      '$h$',       sc[2])
        gplot(axes[5],  beta_arr,                    r'$\beta$',  np.rad2deg(sc[5]))
        gplot(axes[6],  np.rad2deg(st[3]),           r'$\phi$',   np.rad2deg(sc[6]))
        gplot(axes[7],  np.rad2deg(st[9]),           '$p$',       np.rad2deg(sc[9]))
        gplot(axes[8],  np.rad2deg(st[4]),           r'$\theta$', np.rad2deg(sc[7]))
        gplot(axes[9],  np.rad2deg(st[10]),          '$q$',       np.rad2deg(sc[10]))
        gplot(axes[10], np.rad2deg(st[5]),           r'$\psi$')
        gplot(axes[11], np.rad2deg(st[11]),          '$r$',       np.rad2deg(sc[11]))
        gplot(axes[12], chi_arr,                     r'$\chi$',   np.rad2deg(sc[8]))
        gplot(axes[13], gamma_arr,                   r'$\gamma$')
        gplot(axes[14], np.rad2deg(ci[0]),           'de')
        gplot(axes[15], np.rad2deg(ci[1]),           'da')
        gplot(axes[16], np.rad2deg(ci[2]),           'dr')
        gplot(axes[17], ci[3],                       'dt')

        # # --- Global legend on the RIGHT ---
        # if len(self._legend_handles) > 0:
        #     fig.legend(self._legend_handles, self._legend_labels,
        #             loc='center left',
        #             bbox_to_anchor=(0.82, 0.5))

        fig.suptitle('State Variables and Commands', fontsize=13, y=0.995)

        # Make room for legend
        fig.subplots_adjust(right=0.78)


    def _get_or_create_axes(self, fig_num, nrows, ncols, index, projection=None):
        fig = plt.figure(fig_num)
        n_axes_needed = nrows * ncols
        if len(fig.axes) < n_axes_needed:
            ax = fig.add_subplot(nrows, ncols, index, projection=projection)
        else:
            ax = fig.axes[index - 1]
        return fig, ax

    # ------------------------------------------------------------------
    # Show all figures
    # ------------------------------------------------------------------
    @staticmethod
    def show():
        
        plt.show()


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    N = 500
    t = np.linspace(0, 50, N)

    # Synthetic sinusoidal state trajectory
    st = np.zeros((12, N))
    st[0]  =  50  * t / t[-1]                     # pn  increasing
    st[1]  =  30  * np.sin(0.1 * t)               # pe  sinusoidal
    st[2]  = -100 * np.ones(N)                     # pd  constant altitude
    st[3]  =  np.deg2rad(10) * np.sin(0.2 * t)    # phi
    st[4]  =  np.deg2rad(5)  * np.ones(N)         # theta
    st[5]  =  0.05 * t                             # psi (yawing)
    st[6]  =  15   * np.ones(N)                   # u
    st[7]  =  0.5  * np.sin(0.15 * t)             # v
    st[8]  =  0.2  * np.ones(N)                   # w
    st[9]  =  np.deg2rad(2) * np.sin(0.3 * t)     # p
    st[10] =  np.deg2rad(1) * np.ones(N)          # q
    st[11] =  np.deg2rad(3) * np.sin(0.1 * t)     # r

    ci = np.zeros((4, N))
    ci[0] = np.deg2rad(-5)  * np.ones(N)          # de
    ci[1] = np.deg2rad(2)   * np.sin(0.2 * t)     # da
    ci[2] = np.deg2rad(1)   * np.sin(0.1 * t)     # dr
    ci[3] = 0.6             * np.ones(N)           # dt

    bw = np.zeros((3, N))   # calm wind

    sc = np.zeros((12, N))
    sc[2] =  100  * np.ones(N)   # altitude command
    sc[3] =  15   * np.ones(N)   # Va command
    sc[6] =  np.deg2rad(10) * np.sin(0.2 * t)   # phi command

    plotter = PlotSimulationWithCommands()
    plotter.plot(t, st, ci, bw, sc, color='b')
    plotter.show()
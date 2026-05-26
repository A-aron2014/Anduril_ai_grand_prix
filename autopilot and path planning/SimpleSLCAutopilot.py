"""
SimpleSLCAutopilot - Python class equivalent of the MATLAB SimpleSLCAutopilot.m function.

A successive-loop-closure (SLC) autopilot for a fixed-wing UAV.

Original MATLAB code by RWB (2010), modified by EWF (2013, 2023) for ASEN 5519.

Architecture
------------
Lateral:
  sideslip_hold  -> delta_r  (rudder,  PI)
  course_hold    -> phi_c    (roll cmd, PI)
  roll_hold      -> delta_a  (aileron, PID)

Longitudinal:
  altitude_state_machine -> (delta_t, theta_c)
    mode 1 – Take-off:        fixed throttle + fixed pitch
    mode 2 – Climb:           fixed throttle + airspeed_with_pitch_hold
    mode 3 – Altitude hold:   airspeed_with_throttle_hold + altitude_hold
    mode 4 – Descend:         zero throttle + airspeed_with_pitch_hold
  pitch_hold     -> delta_e  (elevator, PD)

Usage
-----
ap = SimpleSLCAutopilot(control_gains)

# call once per time step:
control_input, x_command = ap.update(time, aircraft_state, wind_angles, control_objectives)

Inputs
------
control_gains : object or dict
    Must have all fields produced by CalculateControlGains, plus:
      .Ts          – autopilot sample time (s)
      .u_trim      – trim control vector [de_trim, da_trim, dr_trim, dt_trim]

time : float
    Current simulation time (s).

aircraft_state : array-like, shape (12,)
    [pn, pe, pd, phi, theta, psi, u, v, w, p, q, r]

wind_angles : array-like, shape (3,)
    [Va, beta, alpha]

control_objectives : array-like, shape (5,)
    [h_c, h_dot_c, chi_c, chi_dot_ff, Va_c]

Outputs
-------
control_input : ndarray, shape (4,)
    [de, da, dr, dt]  (rad, rad, rad, dimensionless)

x_command : ndarray, shape (12,)
    Commanded/desired state vector for logging.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _sat(value, upper, lower):
    """Saturate value between lower and upper limits."""
    return float(np.clip(value, lower, upper))


def _unwrap_angle(angle):
    """Wrap angle to (-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# Integrator helper
# ---------------------------------------------------------------------------

class _PIState:
    """Stores the integrator and previous error for a single PI/PID loop."""
    def __init__(self):
        self.integrator = 0.0
        self.error_d1   = 0.0

    def reset(self):
        self.integrator = 0.0
        self.error_d1   = 0.0


# ---------------------------------------------------------------------------
# Individual control loops
# ---------------------------------------------------------------------------

class _CourseHold:
    """PI course (heading) hold -> phi_c command."""

    def __init__(self):
        self._s = _PIState()

    def reset(self):
        self._s.reset()

    def update(self, chi_c, chi, r,flag, gains):
        if flag:
            self._s.reset()
        s = self._s
        error = _unwrap_angle(chi_c - chi)

        s.integrator += (gains.Ts / 2.0) * (error + s.error_d1)   # trapezoidal

        up = gains.Kp_course * error
        ui = gains.Ki_course * s.integrator

        phi_c = _sat(up + ui, gains.max_roll, -gains.max_roll)

        # anti-wind-up
        if gains.Ki_course != 0:
            phi_c_unsat = up + ui
            s.integrator += (gains.Ts / gains.Ki_course) * (phi_c - phi_c_unsat)

        s.error_d1 = error
        return phi_c


class _RollHold:
    """PID roll hold -> delta_a (aileron)."""

    def __init__(self):
        self._s = _PIState()

    def reset(self):
        self._s.reset()

    def update(self, phi_c, phi, p,flag, gains):
        if flag:
            self._s.reset()
        s = self._s
        error = phi_c - phi

        s.integrator += (gains.Ts / 2.0) * (error + s.error_d1)

        up = gains.Kp_roll * error
        ud = -gains.Kd_roll * p
        ui = gains.Ki_roll * s.integrator

        delta_a = _sat(up + ud + ui, gains.max_da, -gains.max_da)

        # anti-wind-up
        if gains.Ki_roll != 0:
            delta_a_unsat = up + ud + ui
            s.integrator += (gains.Ts / gains.Ki_roll) * (delta_a - delta_a_unsat)

        s.error_d1 = error
        return delta_a


class _SideslipHold:
    """PI sideslip hold -> delta_r (rudder)."""

    def __init__(self):
        self._s = _PIState()

    def reset(self):
        self._s.reset()

    def update(self, beta, flag, gains):
        if flag:
            self._s.reset()
        s = self._s
        error = -beta   # command is zero sideslip

        s.integrator += (gains.Ts / 2.0) * (error + s.error_d1)

        up = gains.Kp_beta * error
        ui = gains.Ki_beta * s.integrator

        delta_r = _sat(up + ui, gains.max_dr, -gains.max_dr)

        # anti-wind-up
        if gains.Ki_beta != 0:
            delta_r_unsat = up + ui
            s.integrator += (gains.Ts / gains.Ki_beta) * (delta_r - delta_r_unsat)

        s.error_d1 = error
        return delta_r


class _PitchHold:
    """PD pitch hold -> delta_e (elevator). No integrator."""

    def update(self, theta_c, theta, q, gains):
        up = gains.Kp_pitch * (theta_c - theta)
        ud = -gains.Kd_pitch * q
        return _sat(up + ud, gains.max_de, -gains.max_de)


class _AirspeedWithPitchHold:
    """PI airspeed hold via pitch -> theta_c."""

    def __init__(self):
        self._s = _PIState()

    def reset(self):
        self._s.reset()

    def update(self, Va_c, Va,flag, gains):
        if flag:
            self._s.reset()
        s = self._s
        error = Va_c - Va

        s.integrator += (gains.Ts / 2.0) * (error + s.error_d1)

        up = gains.Kp_speed_pitch * error
        ui = gains.Ki_speed_pitch * s.integrator

        theta_c = _sat(up + ui, gains.max_pitch, -gains.max_pitch)

        # anti-wind-up
        if gains.Ki_speed_pitch != 0:
            theta_c_unsat = up + ui
            s.integrator += (gains.Ts / gains.Ki_speed_pitch) * (theta_c - theta_c_unsat)

        s.error_d1 = error
        return theta_c


class _AirspeedWithThrottleHold:
    """PI airspeed hold via throttle -> delta_t."""

    def __init__(self):
        self._s = _PIState()

    def reset(self):
        self._s.reset()

    def update(self, Va_c, Va,flag, gains):
        if flag:
            self._s.reset()
        s = self._s
        error = Va_c - Va

        s.integrator += (gains.Ts / 2.0) * (error + s.error_d1)

        up = gains.Kp_speed_throttle * error
        ui = gains.Ki_speed_throttle * s.integrator

        u_trim_t = _get(gains, 'u_trim')[3]
        delta_t = _sat(u_trim_t + up + ui, 1.0, 0.0)

        # anti-wind-up
        if gains.Ki_speed_throttle != 0:
            delta_t_unsat = u_trim_t + up + ui
            s.integrator += (gains.Ts / gains.Ki_speed_throttle) * (delta_t - delta_t_unsat)

        s.error_d1 = error
        return delta_t


class _AltitudeHold:
    """PI altitude hold -> theta_c."""

    def __init__(self):
        self._s = _PIState()

    def reset(self):
        self._s.reset()

    def update(self, h_c, h, flag, gains):
        if flag:
            self._s.reset()
        s = self._s
        error = h_c - h

        s.integrator += (gains.Ts / 2.0) * (error + s.error_d1)

        up = gains.Kp_height * error
        ui = gains.Ki_height * s.integrator

        theta_c = _sat(up + ui, gains.max_pitch, -gains.max_pitch)

        # anti-wind-up
        if gains.Ki_height != 0:
            theta_c_unsat = up + ui
            s.integrator += (gains.Ts / gains.Ki_height) * (theta_c - theta_c_unsat)

        s.error_d1 = error
        return theta_c


# ---------------------------------------------------------------------------
# Altitude state machine
# ---------------------------------------------------------------------------

class _AltitudeStateMachine:
    """
    Manages four altitude modes:
      1 – Take-off
      2 – Climb
      3 – Altitude hold
      4 – Descend
    """

    MODE_TAKEOFF   = 1
    MODE_CLIMB     = 2
    MODE_ALT_HOLD  = 3
    MODE_DESCEND   = 4

    def __init__(self):
        self._mode = 0
        self._reset_flag = True
        self._speed_pitch   = _AirspeedWithPitchHold()
        self._speed_throttle = _AirspeedWithThrottleHold()
        self._alt_hold       = _AltitudeHold()

    def reset(self):
        self._mode = 0
        self._reset_flag = True
        self._speed_pitch.reset()
        self._speed_throttle.reset()
        self._alt_hold.reset()

    def update(self, h_c, h, Va_c, Va, gains):
        """Returns (delta_t, theta_c, mode)."""
        error_height = h_c - h

        if h < gains.takeoff_height:
            # ---- Take-off ----
            if self._mode != self.MODE_TAKEOFF:
                print('Altitude mode: Take Off')
                self._mode = self.MODE_TAKEOFF
            delta_t = gains.climb_throttle
            theta_c = gains.takeoff_pitch

        elif -error_height < -gains.height_hold_limit:
            # ---- Climb ----
            if self._mode != self.MODE_CLIMB:
                print('Altitude mode: Climb')
                self._mode = self.MODE_CLIMB
                self._reset_flag = True
                self._speed_pitch.reset()
            else:
                self._reset_flag = False
            delta_t = gains.climb_throttle
            theta_c = self._speed_pitch.update(Va_c, Va,self._reset_flag, gains)

        elif abs(error_height) <= gains.height_hold_limit:
            # ---- Altitude hold ----
            if self._mode != self.MODE_ALT_HOLD:
                print('Altitude mode: Altitude Hold')
                self._mode = self.MODE_ALT_HOLD
                self._reset_flag = True
                self._speed_throttle.reset()
                self._alt_hold.reset()
            else:
                self._reset_flag = False
            delta_t = self._speed_throttle.update(Va_c, Va,self._reset_flag, gains)
            theta_c = self._alt_hold.update(h_c, h,self._reset_flag, gains)

        else:
            # ---- Descend ----
            if self._mode != self.MODE_DESCEND:
                print('Altitude mode: Descend')
                self._mode = self.MODE_DESCEND
                self._reset_flag = True
                self._speed_pitch.reset()
            else:
                self._reset_flag = False
            delta_t = 0.0
            theta_c = self._speed_pitch.update(Va_c, Va, self._reset_flag, gains)

        return delta_t, theta_c, self._mode


# ---------------------------------------------------------------------------
# Attribute accessor (supports both object and dict gains)
# ---------------------------------------------------------------------------

def _get(gains, name):
    if isinstance(gains, dict):
        return gains[name]
    return getattr(gains, name)


# ---------------------------------------------------------------------------
# Flight-path angle helper (matches stub in plot_simulation_with_commands.py)
# ---------------------------------------------------------------------------

def _flight_path_angles_from_state(state):
    """
    Returns [Vg, chi, gamma] from the 12-element state vector.
    Mirrors FlightPathAnglesFromState used in the MATLAB code.
    """
    phi, theta, psi = state[3], state[4], state[5]
    u, v, w         = state[6], state[7], state[8]

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
    R = R_yaw @ R_pitch @ R_roll
    vel_i = R @ np.array([u, v, w])

    Vg    = np.linalg.norm(vel_i)
    chi   = np.arctan2(vel_i[1], vel_i[0])
    gamma = np.arctan2(-vel_i[2],
                       np.sqrt(vel_i[0]**2 + vel_i[1]**2))
    return np.array([Vg, chi, gamma])


# ---------------------------------------------------------------------------
# Main autopilot class
# ---------------------------------------------------------------------------

class SimpleSLCAutopilot:
    """
    Successive-loop-closure autopilot for a fixed-wing UAV.

    Parameters
    ----------
    control_gains : object or dict
        All gain fields from CalculateControlGains, plus:
          .Ts      – sample time (s)
          .u_trim  – trim inputs [de_trim, da_trim, dr_trim, dt_trim]
    """

    def __init__(self, control_gains):
        self.gains = control_gains

        self._course_hold    = _CourseHold()
        self._roll_hold      = _RollHold()
        self._sideslip_hold  = _SideslipHold()
        self._pitch_hold     = _PitchHold()
        self._alt_machine    = _AltitudeStateMachine()

        self._initialized = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, time, aircraft_state, wind_angles, control_objectives):
        """
        Compute control inputs for one time step.

        Parameters
        ----------
        time : float
        aircraft_state : array-like, shape (12,)
        wind_angles : array-like, shape (3,)   [Va, beta, alpha]
        control_objectives : array-like, shape (5,)
            [h_c, h_dot_c, chi_c, chi_dot_ff, Va_c]

        Returns
        -------
        control_input : ndarray, shape (4,)   [de, da, dr, dt]
        x_command     : ndarray, shape (12,)
        """
        gains = self.gains
        st    = np.asarray(aircraft_state, dtype=float).flatten()
        wa    = np.asarray(wind_angles,    dtype=float).flatten()
        co    = np.asarray(control_objectives, dtype=float).flatten()

        # State extraction
        phi, theta, psi = st[3], st[4], st[5]
        p, q, r         = st[9], st[10], st[11]
        h               = -st[2]

        Va, beta, alpha  = wa[0], wa[1], wa[2]

        h_c        = co[0]
        h_dot_c    = co[1]
        chi_c      = co[2]
        chi_dot_ff = co[3]
        Va_c       = co[4]

        # Reset flag: true on first call (t == 0) – matches MATLAB flag==1
        flag = (time == 0) or (not self._initialized)
        if flag:
            self._reset_all()
            self._initialized = True

        # Flight-path angles
        flight_angles = _flight_path_angles_from_state(st)
        chi = flight_angles[1]

        # ---- Lateral ----
        delta_r = self._sideslip_hold.update(beta,flag, gains)
        phi_c   = self._course_hold.update(chi_c, chi, r,flag, gains)
        delta_a = self._roll_hold.update(phi_c, phi, p,flag, gains)

        # ---- Longitudinal ----
        delta_t, theta_c, alt_mode = self._alt_machine.update(
            h_c, h, Va_c, Va, gains)
        delta_e = self._pitch_hold.update(theta_c, theta, q, gains)

        # ---- Assemble outputs ----
        u_trim = np.asarray(_get(gains, 'u_trim'), dtype=float).flatten()
        control_input = np.array([
            u_trim[0] + delta_e,
            u_trim[1] + delta_a,
            u_trim[2] + delta_r,
            delta_t,
        ])

        x_command = np.array([
            0.0,      # pn
            0.0,      # pe
            h_c,      # h
            Va_c,     # Va
            0.0,      # alpha
            0.0,      # beta
            phi_c,    # phi
            theta_c,  # theta
            chi_c,    # chi
            0.0,      # p
            0.0,      # q
            0.0,      # r
        ])

        return control_input, x_command

    # ------------------------------------------------------------------
    # Internal reset
    # ------------------------------------------------------------------

    def _reset_all(self):
        self._course_hold.reset()
        self._roll_hold.reset()
        self._sideslip_hold.reset()
        self._alt_machine.reset()
        # pitch_hold has no state to reset


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    from types import SimpleNamespace

    gains = SimpleNamespace(
        g                  = 9.81,
        Ts                 = 0.01,
        u_trim             = np.array([0.0, 0.0, 0.0, 0.5]),
        max_roll           = np.deg2rad(45),
        max_roll_rate      = np.deg2rad(45),
        max_pitch          = np.deg2rad(30),
        max_da             = np.deg2rad(30),
        max_dr             = np.deg2rad(30),
        max_de             = np.deg2rad(20),
        Kp_roll            =  8.0,
        Kd_roll            =  0.5,
        Ki_roll            =  1.0,
        Kp_course          =  3.0,
        Ki_course          =  1.5,
        Kp_beta            =  1.0,
        Ki_beta            =  0.5,
        Kd_beta            =  0.0,
        Kp_pitch           =  5.0,
        Kd_pitch           =  0.8,
        Kp_height          =  0.05,
        Ki_height          =  0.01,
        Kp_speed_pitch     = -0.5,
        Ki_speed_pitch     = -0.1,
        Kp_speed_throttle  =  2.0,
        Ki_speed_throttle  =  1.0,
        takeoff_height     = 10.0,
        takeoff_pitch      = np.deg2rad(6),
        height_hold_limit  = 5.0,
        climb_throttle     = 0.75,
        Kpitch_DC          = 0.9,
    )

    ap = SimpleSLCAutopilot(gains)

    # Straight-and-level at 50 m altitude, Va = 15 m/s
    state = np.array([0, 0, -50, 0, np.deg2rad(3), 0,
                      15, 0, 0.8, 0, 0, 0], dtype=float)
    wind_angles = np.array([15.0, 0.0, np.deg2rad(3)])
    objectives  = np.array([50.0, 0.0, 0.0, 0.0, 15.0])

    for i in range(10):
        t = i * gains.Ts
        ci, xc = ap.update(t, state, wind_angles, objectives)
        print(f't={t:.2f}  de={np.rad2deg(ci[0]):6.2f}°  '
              f'da={np.rad2deg(ci[1]):6.2f}°  '
              f'dr={np.rad2deg(ci[2]):6.2f}°  '
              f'dt={ci[3]:.3f}')
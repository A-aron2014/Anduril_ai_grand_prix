import numpy as np


class SLCWithFeedForwardAutopilot:
    """
    Autopilot for mavsim - Python conversion of MATLAB SLCWithFeedForwardAutopilot.

    Relies on external helper functions:
        - FlightPathAnglesFromState(aircraft_state) -> [Vg, chi, gamma]

    Usage:
        autopilot = SLCWithFeedForwardAutopilot(control_gain_struct)
        control_input, x_command = autopilot.update(time, aircraft_state, wind_angles, control_objectives)
    """

    def __init__(self, control_gain_struct):
        self.gains = control_gain_struct

        # Persistent state for each sub-controller (replaces MATLAB persistent variables)
        self._roll_hold_state       = _IntegratorState()
        self._sideslip_state        = _IntegratorState()
        self._airspeed_pitch_state  = _IntegratorState()
        self._airspeed_throttle_state = _IntegratorState()
        self._altitude_hold_state   = _IntegratorState()

        # Altitude state machine
        self._alt_mode   = 0
        self._reset_flag = 1

        self._initialized = False

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def update(self, time, aircraft_state, wind_angles, control_objectives,
               flight_path_angles_fn=None):
        """
        Parameters
        ----------
        time               : float
        aircraft_state     : array-like, length >= 12
        wind_angles        : array-like [Va, beta, alpha]
        control_objectives : array-like [h_c, h_dot_c, chi_c, chi_dot_ff, Va_c]
        flight_path_angles_fn : callable(aircraft_state) -> [Vg, chi, gamma]
                               If None, you must supply chi another way.

        Returns
        -------
        control_input : np.ndarray [delta_e, delta_a, delta_r, delta_t]
        x_command     : np.ndarray [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r]
        """
        aircraft_state     = np.asarray(aircraft_state, dtype=float).flatten()
        wind_angles        = np.asarray(wind_angles, dtype=float).flatten()
        control_objectives = np.asarray(control_objectives, dtype=float).flatten()

        flag = 1 if (time == 0 or not self._initialized) else 0
        if flag:
            self._initialized = True
            self._reset_all_integrators()

        # --- Unpack state ---
        pn    = aircraft_state[0]
        pe    = aircraft_state[1]
        h     = -aircraft_state[2]       # altitude (positive up)

        phi   = aircraft_state[3]
        theta = aircraft_state[4]
        psi   = aircraft_state[5]

        euler_angles  = aircraft_state[3:6]
        velocity_body = aircraft_state[6:9]   # [u, v, w]
        omega_body    = aircraft_state[9:12]  # [p, q, r]

        Va   = wind_angles[0]
        beta = wind_angles[1]
        alpha= wind_angles[2]

        h_c        = control_objectives[0]
        h_dot_c    = control_objectives[1]
        chi_c      = control_objectives[2]
        chi_dot_ff = control_objectives[3]
        Va_c       = control_objectives[4]

        # Course angle from external function
        if flight_path_angles_fn is not None:
            flight_angles = flight_path_angles_fn(aircraft_state)
            chi = float(flight_angles[1])
        else:
            # Fallback: use yaw as course (valid for wings-level, no wind)
            chi = psi

        # ==============================================================
        # Lateral autopilot
        # ==============================================================
        chi_dot_c = (self.gains.Kff_course_rate * chi_dot_ff
                     + self.gains.Kp_course_rate * _unwrap_angle(chi_c - chi))

        phi_des, q_des, r_des = self._coordinated_turn_rates(chi_dot_c, Va)

        phi_c = phi_des
        delta_a, p_c = self._roll_hold(phi_c, euler_angles[0], omega_body[0], flag)

        delta_r = self._sideslip_hold(beta, r_des, omega_body[2], flag)

        # ==============================================================
        # Longitudinal autopilot
        # ==============================================================
        delta_t, theta_c, alt_mode = self._altitude_state_machine(h_c, h, Va_c, Va, flag)

        delta_e = self._pitch_hold(theta_c, theta, q_des, omega_body[1])

        # ==============================================================
        # Outputs
        # ==============================================================
        u_trim = np.asarray(self.gains.u_trim, dtype=float).flatten()

        control_input = np.array([
            u_trim[0] + delta_e,
            u_trim[1] + delta_a,
            u_trim[2] + delta_r,
            delta_t
        ])

        x_command = np.array([
            0.0,    # pn
            0.0,    # pe
            h_c,    # h
            Va_c,   # Va
            0.0,    # alpha
            0.0,    # beta
            phi_c,  # phi
            theta_c,# theta
            chi_c,  # chi
            p_c,    # p
            q_des,  # q
            r_des,  # r
        ])

        return control_input, x_command

    # ------------------------------------------------------------------
    # Sub-controller methods
    # ------------------------------------------------------------------

    def _coordinated_turn_rates(self, chi_dot_c, V):
        """Desired roll, pitch rate, and yaw rate for a coordinated turn."""
        phi_des = np.arctan2(chi_dot_c * V, self.gains.g)
        phi_des = _sat(phi_des, self.gains.max_roll, -self.gains.max_roll)

        q_des = chi_dot_c * np.sin(phi_des)
        r_des = chi_dot_c * np.cos(phi_des)
        return phi_des, q_des, r_des

    def _roll_hold(self, phi_c, phi, p, flag):
        """
        Regulate roll using aileron.
        PI on roll angle error -> desired roll rate p_c.
        Feedforward + P on roll rate error -> delta_a.
        """
        s = self._roll_hold_state
        if flag:
            s.reset()

        error = phi_c - phi

        s.integrator += (self.gains.Ts / 2.0) * (error + s.error_d1)

        up = self.gains.Kp_roll * error
        ui = self.gains.Ki_roll * s.integrator

        p_c = _sat(up + ui, self.gains.max_roll_rate, -self.gains.max_roll_rate)

        # Anti-windup
        if self.gains.Ki_roll != 0:
            p_c_unsat = up + ui
            s.integrator += (self.gains.Ts / self.gains.Ki_roll) * (p_c - p_c_unsat)

        delta_a_ff = self.gains.Kff_da * p_c
        ud         = self.gains.Kd_roll * (p_c - p)

        delta_a = _sat(delta_a_ff + ud, self.gains.max_da, -self.gains.max_da)

        s.error_d1 = error
        return delta_a, p_c

    def _sideslip_hold(self, beta, r_des, r, flag):
        """Regulate sideslip using rudder (PI on beta + P/FF on yaw rate)."""
        s = self._sideslip_state
        if flag:
            s.reset()

        error   = -beta
        error_r = r_des - r

        s.integrator += (self.gains.Ts / 2.0) * (error + s.error_d1)

        up         = self.gains.Kp_beta * error
        ui         = self.gains.Ki_beta * s.integrator
        ud         = self.gains.Kd_beta * error_r
        delta_r_ff = self.gains.Kff_dr * r_des

        delta_r = _sat(delta_r_ff + up + ud + ui, self.gains.max_dr, -self.gains.max_dr)

        # Anti-windup
        if self.gains.Ki_beta != 0:
            delta_r_unsat = delta_r_ff + up + ud + ui
            s.integrator += (self.gains.Ts / self.gains.Ki_beta) * (delta_r - delta_r_unsat)

        s.error_d1 = error
        return delta_r

    def _altitude_state_machine(self, h_c, h, Va_c, Va, flag):
        """
        State machine that selects throttle strategy and pitch command
        based on current vs. commanded altitude.

        States:
            1 - Take Off
            2 - Climb
            3 - Altitude Hold
            4 - Descend
        """
        if flag:
            self._alt_mode   = 0
            self._reset_flag = 1

        error_height = h_c - h

        if h < self.gains.takeoff_height:                          # Take-off
            if self._alt_mode != 1:
                print("Altitude mode: Take Off")
                self._alt_mode = 1
            delta_t = self.gains.climb_throttle
            theta_c = self.gains.takeoff_pitch

        elif -error_height < -self.gains.height_hold_limit:        # Climb
            if self._alt_mode != 2:
                print("Altitude mode: Climb")
                self._alt_mode   = 2
                self._reset_flag = 1
            else:
                self._reset_flag = 0
            delta_t = self.gains.climb_throttle
            theta_c = self._airspeed_with_pitch_hold(Va_c, Va, self._reset_flag)

        elif abs(error_height) <= self.gains.height_hold_limit:    # Altitude hold
            if self._alt_mode != 3:
                print("Altitude mode: Altitude Hold")
                self._alt_mode   = 3
                self._reset_flag = 1
            else:
                self._reset_flag = 0
            delta_t = self._airspeed_with_throttle_hold(Va_c, Va, self._reset_flag)
            theta_c = self._altitude_hold(h_c, h, self._reset_flag)

        else:                                                       # Descend
            if self._alt_mode != 4:
                print("Altitude mode: Descend")
                self._alt_mode   = 4
                self._reset_flag = 1
            else:
                self._reset_flag = 0
            delta_t = 0.0
            theta_c = self._airspeed_with_pitch_hold(Va_c, Va, self._reset_flag)

        return delta_t, theta_c, self._alt_mode

    def _pitch_hold(self, theta_c, theta, q_c, q):
        """Regulate pitch using elevator (PD + feedforward on pitch rate)."""
        uff     = self.gains.Kff_de  * q_c
        up      = self.gains.Kp_pitch * (theta_c - theta)
        ud      = self.gains.Kd_pitch * (q_c - q)
        delta_e = _sat(uff + up + ud, self.gains.max_de, -self.gains.max_de)
        return delta_e

    def _airspeed_with_pitch_hold(self, Va_c, Va, flag):
        """Regulate airspeed using pitch angle (PI controller)."""
        s = self._airspeed_pitch_state
        if flag:
            s.reset()

        error = Va_c - Va
        s.integrator += (self.gains.Ts / 2.0) * (error + s.error_d1)

        up = self.gains.Kp_speed_pitch * error
        ui = self.gains.Ki_speed_pitch * s.integrator

        theta_c = _sat(up + ui, self.gains.max_pitch, -self.gains.max_pitch)

        if self.gains.Ki_speed_pitch != 0:
            theta_c_unsat = up + ui
            s.integrator += (self.gains.Ts / self.gains.Ki_speed_pitch) * (theta_c - theta_c_unsat)

        s.error_d1 = error
        return theta_c

    def _airspeed_with_throttle_hold(self, Va_c, Va, flag):
        """Regulate airspeed using throttle (PI controller)."""
        s = self._airspeed_throttle_state
        if flag:
            s.reset()

        u_trim = np.asarray(self.gains.u_trim, dtype=float).flatten()
        error  = Va_c - Va
        s.integrator += (self.gains.Ts / 2.0) * (error + s.error_d1)

        up = self.gains.Kp_speed_throttle * error
        ui = self.gains.Ki_speed_throttle * s.integrator

        delta_t = _sat(u_trim[3] + up + ui, 1.0, 0.0)

        if self.gains.Ki_speed_throttle != 0:
            delta_t_unsat = u_trim[3] + up + ui
            s.integrator += (self.gains.Ts / self.gains.Ki_speed_throttle) * (delta_t - delta_t_unsat)

        s.error_d1 = error
        return delta_t

    def _altitude_hold(self, h_c, h, flag):
        """Regulate altitude using pitch angle (PI controller)."""
        s = self._altitude_hold_state
        if flag:
            s.reset()

        error = h_c - h
        s.integrator += (self.gains.Ts / 2.0) * (error + s.error_d1)

        up = self.gains.Kp_height * error
        ui = self.gains.Ki_height * s.integrator

        theta_c = _sat(up + ui, self.gains.max_pitch, -self.gains.max_pitch)

        if self.gains.Ki_height != 0:
            theta_c_unsat = up + ui
            s.integrator += (self.gains.Ts / self.gains.Ki_height) * (theta_c - theta_c_unsat)

        s.error_d1 = error
        return theta_c

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reset_all_integrators(self):
        self._roll_hold_state.reset()
        self._sideslip_state.reset()
        self._airspeed_pitch_state.reset()
        self._airspeed_throttle_state.reset()
        self._altitude_hold_state.reset()


# ----------------------------------------------------------------------
# Helper classes / module-level functions
# ----------------------------------------------------------------------

class _IntegratorState:
    """Holds integrator and previous-error state for a single PI/PID loop."""
    def __init__(self):
        self.integrator = 0.0
        self.error_d1   = 0.0

    def reset(self):
        self.integrator = 0.0
        self.error_d1   = 0.0


def _sat(value, upper, lower):
    """Saturation / clamp function."""
    return float(np.clip(value, lower, upper))


def _unwrap_angle(angle):
    """Wrap angle to (-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

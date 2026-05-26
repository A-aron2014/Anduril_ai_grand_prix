import numpy as np

# ---------------------------------------------------------------------------
# Persistent state (replaces MATLAB persistent variables)
# ---------------------------------------------------------------------------
_phat      = None
_qhat      = None
_rhat      = None
_press_stat = None
_press_dyn  = None
_pn_hat    = None
_pe_hat    = None
_s_x       = None
_s_y       = None
_s_z       = None
_chi_hat   = None


def SimpleEstimator(time, gps_sensor, inertial_sensors, sensor_params):
    """
    Simple low-pass filter based estimator.

    Runs at the IMU rate; GPS position is only updated at the GPS rate.

    Parameters
    ----------
    time : float
        Current simulation time [s]
    gps_sensor : np.ndarray, shape (5,)
        [pn, pe, ph, Vg, chi]
    inertial_sensors : np.ndarray, shape (8,)
        [y_accel(3), y_gyro(3), y_abs_pressure, y_dyn_pressure]
    sensor_params : SimpleNamespace
        Sensor parameters from SensorParametersTtwistor

    Returns
    -------
    aircraft_state_est : np.ndarray, shape (12,)
        Estimated aircraft state [pn, pe, pd, phi, theta, psi, u, v, w, p, q, r]
    wind_inertial_est : np.ndarray, shape (3,)
        Estimated inertial wind [wn, we, wd]
    """
    global _phat, _qhat, _rhat
    global _press_stat, _press_dyn
    global _pn_hat, _pe_hat, _chi_hat
    global _s_x, _s_y, _s_z

    import stdatmo
    from GNC_Sim import WindAnglesToAirRelativeVelocityVector

    h_ground = sensor_params.h_ground
    density  = stdatmo.std_atmo(h_ground).rho

    Ts_imu = sensor_params.Ts_imu
    Ts_gps = sensor_params.Ts_gps
    g      = sensor_params.g

    # -----------------------------------------------------------------------
    # Angular velocity — high bandwidth low-pass filter
    # -----------------------------------------------------------------------
    a_omega     = 1000.0
    alpha_omega = np.exp(-a_omega * Ts_imu)

    _phat = _low_pass_filter(_phat, inertial_sensors[3], alpha_omega)
    _qhat = _low_pass_filter(_qhat, inertial_sensors[4], alpha_omega)
    _rhat = _low_pass_filter(_rhat, inertial_sensors[5], alpha_omega)

    # -----------------------------------------------------------------------
    # Height from absolute pressure
    # -----------------------------------------------------------------------
    a_h     = 0.7                          # STUDENT: tune this
    alpha_h = np.exp(-a_h * Ts_imu)

    _press_stat = _low_pass_filter(_press_stat, inertial_sensors[6], alpha_h)

    # h = press_stat / (density * g) + h_ground
    hhat = _press_stat / (density * g) + h_ground

    # -----------------------------------------------------------------------
    # Airspeed from dynamic pressure
    # -----------------------------------------------------------------------
    a_Va     = 0.6                        # STUDENT: tune this
    alpha_Va = np.exp(-a_Va * Ts_imu)

    _press_dyn = _low_pass_filter(_press_dyn, inertial_sensors[7], alpha_Va)

    # Va = sqrt(2 * press_dyn / density)
    Va = np.sqrt(np.maximum(2.0 * _press_dyn / density, 0.0))

    # -----------------------------------------------------------------------
    # Position from GPS (updated only at GPS rate)
    # -----------------------------------------------------------------------
    a_gps     = 0.25 #1                         # STUDENT: tune this
    alpha_gps = np.exp(-a_gps * Ts_gps)

    if _pn_hat is None:
        _pn_hat = gps_sensor[0]
    elif np.isclose(time % Ts_gps, 0.0, atol=Ts_imu / 2.0):
        _pn_hat = _low_pass_filter(_pn_hat, gps_sensor[0], alpha_gps)

    if _pe_hat is None:
        _pe_hat = gps_sensor[1]
    elif np.isclose(time % Ts_gps, 0.0, atol=Ts_imu / 2.0):
        _pe_hat = _low_pass_filter(_pe_hat, gps_sensor[1], alpha_gps)

    if _chi_hat is None:
        _chi_hat = gps_sensor[4]
    elif np.isclose(time % Ts_gps, 0.0, atol=Ts_imu / 2.0):
        _chi_hat = _low_pass_filter(_chi_hat, gps_sensor[4], alpha_gps)

    # -----------------------------------------------------------------------
    # Orientation from accelerometers
    # -----------------------------------------------------------------------
    a_acc     = 0.25                        # STUDENT: tune this
    alpha_acc = np.exp(-a_acc * Ts_imu)

    _s_x = _low_pass_filter(_s_x, inertial_sensors[0], alpha_acc)
    _s_y = _low_pass_filter(_s_y, inertial_sensors[1], alpha_acc)
    _s_z = _low_pass_filter(_s_z, inertial_sensors[2], alpha_acc)

    # roll and pitch from accelerometer readings
    roll_hat  = np.arctan2(-_s_y, -_s_z)
    pitch_hat = np.arcsin(np.clip(_s_x / g, -1.0, 1.0))
    yaw_hat   = _chi_hat   # approximate: psi ~ chi when sideslip ~ 0

    # -----------------------------------------------------------------------
    # Reconstruct body velocity from Va, pitch, yaw
    # -----------------------------------------------------------------------
    wind_angles_est = np.array([Va, 0.0, pitch_hat])   # [Va, beta~0, alpha~pitch]
    vel_body_est    = WindAnglesToAirRelativeVelocityVector(wind_angles_est)

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------
    aircraft_state_est = np.array([
        _pn_hat,                  # 0  pn
        _pe_hat,                  # 1  pe
        -hhat,                    # 2  pd = -h
        roll_hat,                 # 3  phi
        pitch_hat,                # 4  theta
        yaw_hat,                  # 5  psi
        vel_body_est[0],          # 6  u
        vel_body_est[1],          # 7  v
        vel_body_est[2],          # 8  w
        _phat,                    # 9  p
        _qhat,                    # 10 q
        _rhat,                    # 11 r
    ])

    wind_inertial_est = np.zeros(3)

    return aircraft_state_est, wind_inertial_est


def reset_simple_estimator():
    """Reset all persistent state between simulations."""
    global _phat, _qhat, _rhat, _press_stat, _press_dyn
    global _pn_hat, _pe_hat, _chi_hat, _s_x, _s_y, _s_z
    _phat = _qhat = _rhat = None
    _press_stat = _press_dyn = None
    _pn_hat = _pe_hat = _chi_hat = None
    _s_x = _s_y = _s_z = None


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _low_pass_filter(y_old, u_new, alpha):
    """First-order low-pass filter. Initializes on first call."""
    if y_old is None:
        return u_new
    return alpha * y_old + (1.0 - alpha) * u_new

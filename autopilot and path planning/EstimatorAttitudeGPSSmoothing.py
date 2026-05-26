import numpy as np

# ---------------------------------------------------------------------------
# Persistent state (replaces MATLAB persistent variables)
# ---------------------------------------------------------------------------
_phat       = None
_qhat       = None
_rhat       = None
_press_stat  = None
_press_dyn   = None
_phi_hat    = None
_theta_hat  = None
_P_est      = None
_xhat_gps   = None
_P_gps      = None


def EstimatorAttitudeGPSSmoothing(time, gps_sensor, inertial_sensors, sensor_params):
    """
    Extended Kalman Filter based estimator with attitude EKF and GPS smoothing.

    Runs at IMU rate; GPS measurement update only occurs at GPS rate.

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
    global _phi_hat, _theta_hat, _P_est
    global _xhat_gps, _P_gps

    import stdatmo
    from GNC_Sim import TransformFromInertialToBody, WindAnglesToAirRelativeVelocityVector

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
    hhat = _press_stat / (density * g) + h_ground #Student Complete

    # -----------------------------------------------------------------------
    # Airspeed from dynamic pressure
    # -----------------------------------------------------------------------
    a_Va     = 0.9                         # STUDENT: tune this
    alpha_Va = np.exp(-a_Va * Ts_imu)

    _press_dyn = _low_pass_filter(_press_dyn, inertial_sensors[7], alpha_Va)
    Va = np.sqrt(np.maximum(2.0 * _press_dyn / density, 0.0))

    # -----------------------------------------------------------------------
    # Attitude EKF
    # -----------------------------------------------------------------------
    Q = 0.01 * ((np.pi / 180.0) ** 2) * np.eye(2)
    R = (sensor_params.sig_accel ** 2) * np.eye(3)

    if _phi_hat is None:
        _phi_hat   = 0.0
        _theta_hat = 0.0
        _P_est     = ((30.0 * np.pi / 180.0) ** 2) * np.eye(2)
    else:
        # --- Propagate ---
        xdot, A = _attitude_filter_update(_phi_hat, _theta_hat, _phat, _qhat, _rhat)
        _phi_hat   = _phi_hat   + xdot[0] * Ts_imu
        _theta_hat = _theta_hat + xdot[1] * Ts_imu
        _P_est     = _P_est + Ts_imu * (A @ _P_est + _P_est @ A.T + Q)

        # --- Measurement update ---
        zhat, H = _attitude_filter_measurement(_phi_hat, _theta_hat,
                                                _phat, _qhat, _rhat, Va, g)
        L      = _P_est @ H.T @ np.linalg.inv(R + H @ _P_est @ H.T)
        _P_est = (np.eye(2) - L @ H) @ _P_est
        xhat   = np.array([_phi_hat, _theta_hat]) + L @ (inertial_sensors[0:3] - zhat)
        _phi_hat   = xhat[0]
        _theta_hat = xhat[1]

    # -----------------------------------------------------------------------
    # GPS smoothing EKF
    # state: xhat_gps = [pn, pe, Vg, chi, wn, we, psi]
    # -----------------------------------------------------------------------
    Qgps = np.diag([
        10.0 ** 2,              # pn
        10.0 ** 2,              # pe
        2.0 ** 2,               # Vg
        (5.0 * np.pi / 180.0) ** 2,  # chi
        25.0,                   # wn
        25.0,                   # we
        (5.0 * np.pi / 180.0) ** 2,  # psi
    ])

    sv = sensor_params.sig_gps_v
    Rgps = np.diag([
        sensor_params.sig_gps[0] ** 2,
        sensor_params.sig_gps[1] ** 2,
        sv ** 2,
        (sv / 20.0) ** 2,
        sv ** 2,
        sv ** 2,
    ])

    if _xhat_gps is None:
        _xhat_gps = np.array([
            gps_sensor[0],   # pn
            gps_sensor[1],   # pe
            gps_sensor[3],   # Vg
            gps_sensor[4],   # chi
            0.0,             # wn
            0.0,             # we
            gps_sensor[4],   # psi ~ chi initially
        ])
        _P_gps = 10.0 * Qgps
    else:
        # --- Propagate ---
        xdot_gps, A_gps = _gps_smoothing_update(
            _xhat_gps, Va, _qhat, _rhat, _phi_hat, _theta_hat, g
        )
        _xhat_gps = _xhat_gps + xdot_gps * Ts_imu
        _P_gps    = _P_gps + Ts_imu * (A_gps @ _P_gps + _P_gps @ A_gps.T + Qgps)

        # --- GPS measurement update (at GPS rate only) ---
        if np.isclose(time % Ts_gps, 0.0, atol=Ts_imu / 2.0):
            zhat_gps, H_gps = _gps_smoothing_measurement(_xhat_gps, Va)
            ygps = np.array([
                gps_sensor[0],   # pn
                gps_sensor[1],   # pe
                gps_sensor[3],   # Vg
                gps_sensor[4],   # chi
                0.0,             # wn (not measured)
                0.0,             # we (not measured)
            ])

            L_gps  = _P_gps @ H_gps.T @ np.linalg.inv(Rgps + H_gps @ _P_gps @ H_gps.T)
            _P_gps = (np.eye(7) - L_gps @ H_gps) @ _P_gps

            yerr_gps = ygps - zhat_gps

            # Wrap chi error to [-pi, pi]
            while yerr_gps[3] >  np.pi:
                yerr_gps[3] -= 2.0 * np.pi
            while yerr_gps[3] < -np.pi:
                yerr_gps[3] += 2.0 * np.pi

            _xhat_gps = _xhat_gps + L_gps @ yerr_gps

    # -----------------------------------------------------------------------
    # Reconstruct body velocity and output
    # -----------------------------------------------------------------------
    wind_body_est = TransformFromInertialToBody(
        np.array([_xhat_gps[4], _xhat_gps[5], 0.0]),
        np.array([_phi_hat, _theta_hat, _xhat_gps[6]])
    )
    air_rel_est  = np.array([Va * np.cos(_theta_hat), 0.0, Va * np.sin(_theta_hat)])
    vel_body_est = air_rel_est + wind_body_est

    aircraft_state_est = np.array([
        _xhat_gps[0],       # 0  pn
        _xhat_gps[1],       # 1  pe
        -hhat,              # 2  pd = -h
        _phi_hat,           # 3  phi
        _theta_hat,         # 4  theta
        _xhat_gps[6],       # 5  psi
        vel_body_est[0],    # 6  u
        vel_body_est[1],    # 7  v
        vel_body_est[2],    # 8  w
        _phat,              # 9  p
        _qhat,              # 10 q
        _rhat,              # 11 r
    ])

    wind_inertial_est = np.array([_xhat_gps[4], _xhat_gps[5], 0.0])

    return aircraft_state_est, wind_inertial_est


def reset_estimator():
    """Reset all persistent state between simulations."""
    global _phat, _qhat, _rhat, _press_stat, _press_dyn
    global _phi_hat, _theta_hat, _P_est, _xhat_gps, _P_gps
    _phat = _qhat = _rhat = None
    _press_stat = _press_dyn = None
    _phi_hat = _theta_hat = _P_est = None
    _xhat_gps = _P_gps = None


# ---------------------------------------------------------------------------
# Helper: low-pass filter
# ---------------------------------------------------------------------------
def _low_pass_filter(y_old, u_new, alpha):
    if y_old is None:
        return u_new
    return alpha * y_old + (1.0 - alpha) * u_new

#TODO Come back and check Xdot and Jacobian for correctness
# ---------------------------------------------------------------------------
# Attitude EKF: process model
# state: x = [phi, theta]
# ---------------------------------------------------------------------------
def _attitude_filter_update(phi, theta, p, q, r):
    """
    Returns xdot and Jacobian A for attitude EKF propagation.

    x = [phi, theta]
    xdot = f(x, p, q, r)
    """
    cp = np.cos(phi)
    sp = np.sin(phi)
    ct = np.cos(theta)
    tt = np.tan(theta)

    # phi_dot   = p + q*sin(phi)*tan(theta) + r*cos(phi)*tan(theta)
    # theta_dot = q*cos(phi)               - r*sin(phi)
    phi_dot   = p + q * sp * tt + r * cp * tt
    theta_dot = q * cp          - r * sp

    xdot = np.array([phi_dot, theta_dot])

    # Jacobian A = d(xdot)/d([phi, theta])
    # d(phi_dot)/d(phi)     = q*cos(phi)*tan(theta) - r*sin(phi)*tan(theta)
    # d(phi_dot)/d(theta)   = q*sin(phi)/cos^2(theta) + r*cos(phi)/cos^2(theta)
    # d(theta_dot)/d(phi)   = -q*sin(phi) - r*cos(phi)
    # d(theta_dot)/d(theta) = 0
    A = np.array([
        [q * cp * tt - r * sp * tt,   (q * sp + r * cp) / (ct ** 2)], #(q * sp + r * cp) / (ct ** 2)
        [-q * sp - r * cp,             0.0],
    ])

    return xdot, A

#TODO Come back and check Zhat and Jacobian for correctness
# ---------------------------------------------------------------------------
# Attitude EKF: measurement model
# measurement: y = [ax, ay, az] (specific force from accelerometers)
# ---------------------------------------------------------------------------
def _attitude_filter_measurement(phi, theta, p, q, r, Va, g):
    """
    Returns predicted accelerometer measurement zhat and Jacobian H.
    """
    cp = np.cos(phi)
    sp = np.sin(phi)
    ct = np.cos(theta)
    st = np.sin(theta)

    # Predicted specific force (accelerometer model)
    # ax =  q*Va*sin(theta)       + g*sin(theta)
    # ay = -r*Va*cos(theta)       + p*Va*sin(theta) - g*cos(theta)*sin(phi)
    # az = -q*Va*cos(theta)                         - g*cos(theta)*cos(phi)
    zhat = np.array([
         q * Va * st        + g * st,
        r * Va * ct        - p * Va * st - g * ct * sp,
        -q * Va * ct                      - g * ct * cp,
    ])

    # Jacobian H = d(zhat)/d([phi, theta])
    H = np.array([
        [0.0,                               q * Va * ct + g * ct],
        [-g * cp * ct,                      -r * Va * st - p * Va * ct + g * st * sp],
        [ g * sp * ct,                      (q*Va+g*ct)*st],#q * Va * st - g * ct * (-cp)
    ])
    # Note: last row corrected: d(az)/d(theta) = q*Va*sin(theta) + g*sin(theta)*cos(phi)
    H[2, 1] = q * Va * st + g * st * cp

    return zhat, H


# ---------------------------------------------------------------------------
# GPS smoothing EKF: process model
# state: xh = [pn, pe, Vg, chi, wn, we, psi]
# ---------------------------------------------------------------------------
def _gps_smoothing_update(xh, Va, q, r, roll, pitch, g):
    """
    Returns xdot and Jacobian A for GPS smoothing EKF propagation.
    """
    psi    = xh[6]
    Vg     = xh[2]
    chi    = xh[3]
    wn     = xh[4]
    we     = xh[5]

    cp = np.cos(roll)
    sp = np.sin(roll)
    ct = np.cos(pitch)

    psi_dot = q * sp / ct + r * cp / ct

    # Vg_dot from wind triangle kinematics
    Vg_dot = (
        (Va * np.sin(psi) + we) * (Va * psi_dot * np.cos(psi))
        - (Va * np.cos(psi) + wn) * (Va * psi_dot * np.sin(psi))
    ) / Vg

    xdot = np.array([
        Vg * np.cos(chi),          # pn_dot
        Vg * np.sin(chi),          # pe_dot
        Vg_dot,                    # Vg_dot
        g / Vg * np.tan(roll),     # chi_dot
        0.0,                       # wn_dot = 0 (constant wind assumed)
        0.0,                       # we_dot = 0
        psi_dot,                   # psi_dot
    ])

    # Jacobian A = d(xdot)/d(xh)  [7x7]
    A = np.zeros((7, 7))

    # d(pn_dot)/d(Vg), d(pn_dot)/d(chi)
    A[0, 2] =  np.cos(chi)
    A[0, 3] = -Vg * np.sin(chi)

    # d(pe_dot)/d(Vg), d(pe_dot)/d(chi)
    A[1, 2] =  np.sin(chi)
    A[1, 3] =  Vg * np.cos(chi)

    # d(Vg_dot)/d(Vg), d(Vg_dot)/d(wn), d(Vg_dot)/d(we), d(Vg_dot)/d(psi)
    A[2, 2] = -Vg_dot / Vg
    A[2, 4] = -Va * psi_dot * np.sin(psi) / Vg
    A[2, 5] =  Va * psi_dot * np.cos(psi) / Vg
    A[2, 6] = (
        (Va * np.cos(psi) * Va * psi_dot * np.cos(psi)
         + (Va * np.sin(psi) + we) * Va * (-np.sin(psi)) * psi_dot
         - (-Va * np.sin(psi) * Va * psi_dot * np.sin(psi)
            + (Va * np.cos(psi) + wn) * Va * np.cos(psi) * psi_dot))
        / Vg
    )

    # d(chi_dot)/d(Vg)
    A[3, 2] = -g * np.tan(roll) / (Vg ** 2)

    return xdot, A


# ---------------------------------------------------------------------------
# GPS smoothing EKF: measurement model
# measurements: y = [pn, pe, Vg, chi, wn_meas, we_meas]  (6x1)
# ---------------------------------------------------------------------------
def _gps_smoothing_measurement(xh, Va):
    """
    Returns predicted GPS measurement zhat and Jacobian H.

    Note: wn and we are not directly measured by GPS; those rows of ygps
    are set to zero in the calling function and the corresponding rows of
    H are zero — they act as a soft prior on wind.
    """
    psi = xh[6]
    Vg  = xh[2]
    chi = xh[3]
    wn  = xh[4]
    we  = xh[5]

    zhat = np.array([
        xh[0],    # pn
        xh[1],    # pe
        Vg,       # Vg
        chi,      # chi
        0.0,      # wn measurement placeholder
        0.0,      # we measurement placeholder
    ])

    # H = d(zhat)/d(xh)  [6x7]
    H = np.zeros((6, 7))
    H[0, 0] = 1.0   # d(pn)/d(pn)
    H[1, 1] = 1.0   # d(pe)/d(pe)
    H[2, 2] = 1.0   # d(Vg)/d(Vg)
    H[3, 3] = 1.0   # d(chi)/d(chi)
    # rows 4 and 5 remain zero (wind not directly observed)

    return zhat, H

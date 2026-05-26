import numpy as np
from GNC_Sim import (TransformFromInertialToBody,
                        AirRelativeVelocityVectorToWindAngles,
                        AeroForcesAndMoments_BodyState_WindCoeffs)
import stdatmo

def InertialSensors(aircraft_state, aircraft_surfaces, wind_inertial,
                    aircraft_parameters, sensor_params):
    """
    Simulates IMU sensor measurements (accelerometers, rate gyros,
    absolute pressure, dynamic pressure).

    Parameters
    ----------
    aircraft_state : np.ndarray, shape (12,)
        [pn, pe, pd, phi, theta, psi, u, v, w, p, q, r]
    aircraft_surfaces : np.ndarray, shape (4,)
        Control surface deflections [de, da, dr, dt]
    wind_inertial : np.ndarray, shape (3,)
        Wind velocity in inertial frame [wn, we, wd]
    aircraft_parameters : object
        Aircraft parameters
    sensor_params : SimpleNamespace
        Sensor parameters from SensorParametersTtwistor

    Returns
    -------
    inertial_sensors : np.ndarray, shape (8,)
        [y_accel (3,), y_gyro (3,), y_abs_pressure, y_dyn_pressure]
    """


    omega_body  = aircraft_state[9:12]
    roll        = aircraft_state[3]
    pitch       = aircraft_state[4]
    height      = -aircraft_state[2]              # h = -pd

    density     = stdatmo.std_atmo(height).rho

    wind_body       = TransformFromInertialToBody(wind_inertial, aircraft_state[3:6])
    air_rel_body    = aircraft_state[6:9] - wind_body
    wind_angles     = AirRelativeVelocityVectorToWindAngles(air_rel_body)
    Va              = wind_angles[0]

    fa_body, _ = AeroForcesAndMoments_BodyState_WindCoeffs(
        aircraft_state, aircraft_surfaces, wind_inertial, density, aircraft_parameters
    )

    # --- Accelerometers ---
    y_accel = fa_body.flatten() / sensor_params.m \
              + sensor_params.sig_accel * np.random.randn(3)

    # --- Rate gyros ---
    y_gyro = omega_body + sensor_params.sig_gyro * np.random.randn(3)

    # --- Absolute pressure ---
    y_abs_pressure = (density * sensor_params.g * (height - sensor_params.h_ground)
                      + sensor_params.bias_abs_press
                      + sensor_params.sig_abs_press * np.random.randn())

    # --- Dynamic pressure ---
    y_dyn_pressure = (0.5 * density * Va * Va
                      + sensor_params.bias_dyn_press
                      + sensor_params.sig_dyn_press * np.random.randn())

    inertial_sensors = np.concatenate([y_accel, y_gyro,
                                        [y_abs_pressure],
                                        [y_dyn_pressure]])
    return inertial_sensors

import numpy as np
from types import SimpleNamespace


def SensorParametersTtwistor(aircraft_parameters):
    """
    Returns a SimpleNamespace of sensor parameters for the TTwistor aircraft.

    Parameters
    ----------
    aircraft_parameters : object
        Aircraft parameters object with attributes g and m.

    Returns
    -------
    sensor_params : SimpleNamespace
    """
    sensor_params = SimpleNamespace()

    sensor_params.Ts_imu = 0.1   # [s] IMU sample time
    sensor_params.Ts_gps = 1.0   # [s] GPS sample time

    sensor_params.g = aircraft_parameters.g
    sensor_params.m = aircraft_parameters.m

    # --- Accelerometer ---
    # ADXL325: sigma_accel = 0.0025g
    sensor_params.sig_accel = 0.0025 * sensor_params.g   # [m/s^2]

    # --- Rate gyro ---
    # ADXRS540: sigma_gyro = 0.13 deg/s
    sensor_params.sig_gyro = 0.13 * np.pi / 180.0        # [rad/s]

    # --- Absolute pressure sensor ---
    # Freescale Semiconductor MP3H6115A
    sensor_params.h_ground      = 1655.0    # [m] ground elevation
    sensor_params.bias_abs_press = 0.125    # [kPa]
    sensor_params.sig_abs_press  = 0.01     # [kPa]

    # --- Dynamic pressure sensor ---
    # Freescale Semiconductor MPXV5004G
    sensor_params.bias_dyn_press = 0.02     # [kPa]
    sensor_params.sig_dyn_press  = 0.002    # [kPa]

    # --- GPS sensor ---
    sensor_params.sig_gps   = np.array([0.21, 0.21, 0.4])  # [m] position noise std
    sensor_params.k_gps     = 1.0 / 1100.0
    sensor_params.k_gps_exp = np.exp(-sensor_params.Ts_gps * sensor_params.k_gps)
    sensor_params.sig_gps_v = 0.01  # [m/s] velocity noise std

    return sensor_params

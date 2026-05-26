import numpy as np

# Module-level persistent state (replaces MATLAB persistent variables)
_gps_noise = None


def GPSSensor(aircraft_state, sensor_params):
    """
    Simulates GPS sensor measurements with exponentially correlated noise.

    Parameters
    ----------
    aircraft_state : np.ndarray, shape (12,)
        Full aircraft state vector [pn, pe, pd, phi, theta, psi, u, v, w, p, q, r]
    sensor_params : SimpleNamespace
        Sensor parameters from SensorParametersTtwistor

    Returns
    -------
    gps_sensor : np.ndarray, shape (5,)
        [pn, pe, ph, Vg, chi]
    """
    global _gps_noise

    # --- Import here to avoid circular dependency ---
    #from GNC_Sim import FlightPathAnglesFromState
    from RunHW6 import FlightPathAnglesFromState
    # --- Initialize persistent GPS noise ---
    if _gps_noise is None:
        _gps_noise = np.zeros(3)

    # --- Exponentially correlated position noise ---
    _gps_noise = (sensor_params.k_gps_exp * _gps_noise
                  + sensor_params.sig_gps * np.random.randn(3))

    # --- Position measurements ---
    pn = aircraft_state[0] + _gps_noise[0]
    pe = aircraft_state[1] + _gps_noise[1]
    ph = -aircraft_state[2] + _gps_noise[2]   # h = -pd

    # --- Velocity measurements ---
    flight_path_angles = FlightPathAnglesFromState(aircraft_state)

    Vg  = flight_path_angles[0] + sensor_params.sig_gps_v * np.random.randn()
    chi = flight_path_angles[1] + sensor_params.sig_gps_v * np.random.randn() / Vg

    gps_sensor = np.array([pn, pe, ph, Vg, chi])
    return gps_sensor


def reset_gps_sensor():
    """Reset persistent state (call between simulations)."""
    global _gps_noise
    _gps_noise = None

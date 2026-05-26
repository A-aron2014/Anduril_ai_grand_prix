import numpy as np
from dataclasses import dataclass

@dataclass
class Atmosphere:
    rho: np.ndarray      # density [kg/m^3]
    a: np.ndarray        # speed of sound [m/s]
    T: np.ndarray        # temperature [K]
    P: np.ndarray        # pressure [Pa]
    nu: np.ndarray       # kinematic viscosity [m^2/s]
    ZorH: np.ndarray     # geometric or geopotential altitude [m]


def std_atmo(
    H,
    dT=0.0,
    *,
    geometric=False,
    units="SI"
) -> Atmosphere:
    """
    1976 Standard Atmosphere (up to ~86 km)

    Parameters
    ----------
    H : array_like
        Altitude [m] (geopotential by default)
    dT : array_like or scalar
        Temperature offset [K]
    geometric : bool
        If True, H is geometric altitude
    units : {"SI", "US"}
        Input/output units

    Returns
    -------
    Atmosphere
    """

    H = np.asarray(H, dtype=float)
    dT = np.asarray(dT, dtype=float)

    # -----------------------------
    # Unit handling (boundary only)
    # -----------------------------
    if units.upper() == "US":
        H = H * 0.3048
        dT = dT * 5.0 / 9.0

    # -----------------------------
    # Constants
    # -----------------------------
    R = 287.05287          # J/(kg*K)
    gamma = 1.4
    g0 = 9.80665           # m/s^2
    RE = 6_356_766.0       # Earth radius [m]
    Bs = 1.458e-6          # Sutherland constant
    S = 110.4              # K

    # -----------------------------
    # Atmosphere layers
    # -----------------------------
    layers = np.array([
        [-0.0065, 288.15,   0.0,     101325.0],
        [ 0.0,    216.65, 11000.0,    22632.04],
        [ 0.001,  216.65, 20000.0,     5474.88],
        [ 0.0028, 228.65, 32000.0,      868.02],
        [ 0.0,    270.65, 47000.0,      110.91],
        [-0.0028, 270.65, 51000.0,       66.94],
        [-0.002,  214.65, 71000.0,        3.96],
        [ 0.0,    186.95, 84852.0,        0.37],
    ])

    K, T0, H0, P0 = layers.T

    # -----------------------------
    # Geometric ↔ geopotential
    # -----------------------------
    if geometric:
        H_geop = (RE * H) / (RE + H)
    else:
        H_geop = H

    # -----------------------------
    # Allocate outputs
    # -----------------------------
    T = np.zeros_like(H_geop)
    P = np.zeros_like(H_geop)

    # -----------------------------
    # Layer-wise computation
    # -----------------------------
    for i in range(len(layers)):
        in_layer = (
            (H_geop >= H0[i]) &
            (H_geop < (H0[i+1] if i+1 < len(layers) else 90_000))
        )

        if not np.any(in_layer):
            continue

        if K[i] == 0.0:
            T[in_layer] = T0[i]
            P[in_layer] = P0[i] * np.exp(
                -g0 * (H_geop[in_layer] - H0[i]) / (R * T0[i])
            )
        else:
            T_ratio = 1 + K[i] * (H_geop[in_layer] - H0[i]) / T0[i]
            T[in_layer] = T0[i] * T_ratio
            P[in_layer] = P0[i] * T_ratio ** (-g0 / (K[i] * R))

    # -----------------------------
    # Apply temperature offset
    # -----------------------------
    T += dT

    # -----------------------------
    # Derived properties
    # -----------------------------
    rho = P / (R * T)
    a = np.sqrt(gamma * R * T)
    nu = (Bs * T**1.5 / (T + S)) / rho

    # -----------------------------
    # Return altitude
    # -----------------------------
    if geometric:
        ZorH = H_geop
    else:
        ZorH = RE * H_geop / (RE - H_geop)

    # -----------------------------
    # Output unit conversion
    # -----------------------------
    if units.upper() == "US":
        rho /= 515.3788
        a /= 0.3048
        T *= 1.8
        P /= 47.88026
        nu /= 0.09290304
        ZorH /= 0.3048

    return Atmosphere(rho=rho, a=a, T=T, P=P, nu=nu, ZorH=ZorH)

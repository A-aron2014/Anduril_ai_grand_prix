"""
CalculateControlGainsSimpleSLC_Nondim_Ttwistor - Python class equivalent of the
MATLAB function of the same name.

Determines control gains required for a SimpleSLC autopilot given nondimensional
aircraft coefficients, a trim definition, and trim variables.

Inputs
------
aircraft_parameters : AircraftParameters (or any object/dict with the fields below)
    .g          - gravitational acceleration (m/s^2)
    .S          - reference wing area (m^2)
    .b          - wingspan (m)
    .c          - mean aerodynamic chord (m)
    .m          - aircraft mass (kg)
    .Iy         - moment of inertia about y-axis (kg·m^2)
    .Clp        - roll damping derivative
    .Clda       - aileron roll control derivative
    .CYbeta     - side-force due to sideslip
    .CYdr       - side-force due to rudder
    .Cmq        - pitch damping derivative
    .Cmalpha    - pitch moment due to AoA
    .Cmde       - pitch moment due to elevator
    .CL0        - lift coefficient at zero AoA
    .CLalpha    - lift curve slope
    .CLde       - lift due to elevator
    .CDmin      - minimum drag coefficient
    .K          - induced drag factor
    .CLmin      - CL at minimum drag
    .Sprop      - propeller disk area (m^2)
    .Cprop      - propeller thrust coefficient
    .kmotor     - motor constant

trim_definition : array-like, length >= 3
    [Va, ?, altitude]   (Va = trim airspeed m/s, index 2 = altitude m)

trim_variables : array-like, length >= 3
    [alpha, de, dt]     (AoA rad, elevator deflection rad, throttle 0-1)

Outputs
-------
control_gains  : ControlGains  (dataclass with all gain fields)
linear_terms   : LinearTerms   (dataclass with a_phi1/2, a_beta1/2, a_theta1/2/3, a_v1/2)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Simple standard atmosphere (returns density in kg/m^3)
# ---------------------------------------------------------------------------

def stdatmo(altitude_m: float) -> float:
    """
    Returns air density (kg/m^3) for a given altitude (metres) using the
    International Standard Atmosphere (troposphere only, valid to ~11 km).
    """
    T0 = 288.15      # K,    sea-level temperature
    P0 = 101325.0    # Pa,   sea-level pressure
    L  = 0.0065      # K/m,  lapse rate
    R  = 287.058     # J/(kg·K)
    g0 = 9.80665     # m/s^2

    T = T0 - L * altitude_m
    P = P0 * (T / T0) ** (g0 / (L * R))
    rho = P / (R * T)
    return rho


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ControlGains:
    # gravity
    g: float = 0.0

    # limits
    max_roll:      float = 0.0
    max_roll_rate: float = 0.0
    max_pitch:     float = 0.0
    max_da:        float = 0.0
    max_dr:        float = 0.0
    max_de:        float = 0.0

    # roll hold
    Kp_roll: float = 0.0
    Kd_roll: float = 0.0
    Ki_roll: float = 0.0

    # course hold
    Kp_course: float = 0.0
    Ki_course: float = 0.0

    # sideslip hold
    Kp_beta: float = 0.0
    Ki_beta: float = 0.0
    Kd_beta: float = 0.0

    # pitch hold
    Kp_pitch: float = 0.0
    Kd_pitch: float = 0.0

    # height hold
    Kp_height: float = 0.0
    Ki_height: float = 0.0

    # height state-machine parameters
    Kpitch_DC:         float = 0.0
    takeoff_height:    float = 0.0
    takeoff_pitch:     float = 0.0
    height_hold_limit: float = 0.0
    climb_throttle:    float = 0.0

    # airspeed from pitch
    Kp_speed_pitch: float = 0.0
    Ki_speed_pitch: float = 0.0

    # airspeed from throttle
    Kp_speed_throttle: float = 0.0
    Ki_speed_throttle: float = 0.0


@dataclass
class LinearTerms:
    a_phi1:   float = 0.0
    a_phi2:   float = 0.0
    a_beta1:  float = 0.0
    a_beta2:  float = 0.0   # NOTE: MATLAB had a typo "a_bet2"; corrected here
    a_theta1: float = 0.0
    a_theta2: float = 0.0
    a_theta3: float = 0.0
    a_v1:     float = 0.0
    a_v2:     float = 0.0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CalculateControlGains:
    """
    Computes autopilot control gains for a SimpleSLC autopilot using
    nondimensional aircraft coefficients.

    Usage
    -----
    result = CalculateControlGains(aircraft_parameters, trim_definition, trim_variables)
    gains  = result.control_gains
    lterms = result.linear_terms

    Parameters
    ----------
    aircraft_parameters : object or dict
        Must expose the aerodynamic/inertia fields listed in the module docstring.
        Both attribute-style objects and dict-style objects are supported.
    trim_definition : array-like, length 3
        [Va, (unused), altitude_m]
    trim_variables : array-like, length 3
        [alpha_rad, de_rad, throttle]
    """

    def __init__(self, aircraft_parameters, trim_definition, trim_variables):
        self._ap = aircraft_parameters
        self._trim_def = np.asarray(trim_definition, dtype=float)
        self._trim_var = np.asarray(trim_variables, dtype=float)

        self.control_gains = ControlGains()
        self.linear_terms  = LinearTerms()

        self._compute()

    # ------------------------------------------------------------------
    # Helper: unified attribute access for dict or object
    # ------------------------------------------------------------------

    def _get(self, name: str):
        ap = self._ap
        if isinstance(ap, dict):
            return ap[name]
        return getattr(ap, name)

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def _compute(self):
        cg = self.control_gains
        lt = self.linear_terms

        g       = self._get('g')
        cg.g    = g

        Va      = self._trim_def[0]
        alt     = self._trim_def[2]
        density = stdatmo(alt)

        S = self._get('S')
        b = self._get('b')
        c = self._get('c')
        m = self._get('m')
        Iy = self._get('Iy')

        # ---- Control limits ----
        cg.max_roll      = np.deg2rad(45)
        cg.max_roll_rate = np.deg2rad(45)
        cg.max_pitch     = np.deg2rad(30)
        cg.max_da        = np.deg2rad(30)
        cg.max_dr        = np.deg2rad(30)
        cg.max_de        = np.deg2rad(20)

        QS = 0.5 * density * Va**2 * S

        # ================================================================
        # Lateral / directional
        # ================================================================

        # ---- Roll hold ----
        zeta_roll = 1.0 #Places poles at -2 -> 0.3
        e_phi_max = cg.max_roll

        a_phi1 = -QS * b * self._get('Clp') * b / (2.0 * Va)
        a_phi2 =  QS * b * self._get('Clda')

        cg.Kp_roll = 3.0 * (cg.max_da / e_phi_max) * np.sign(a_phi2)
        wn_roll    = np.sqrt(abs(a_phi2 * cg.Kp_roll))

        print(f'The desired Zeta is {-2/wn_roll}')

        cg.Kd_roll = (2.0 * zeta_roll * wn_roll - a_phi1) / a_phi2
        cg.Ki_roll = cg.Kd_roll*0.001 # student-selected default - choosing to be a magnitude lower than Kp

        # Closed-loop denominator coefficients (for reference / analysis)
        # den_phi2 = [1, a_phi1 + a_phi2*Kd, a_phi2*Kp, a_phi2*Ki]
        self._den_phi2 = np.array([
            1,
            a_phi1 + a_phi2 * cg.Kd_roll,
            a_phi2 * cg.Kp_roll,
            a_phi2 * cg.Ki_roll,
        ])

        # ---- Course hold ----
        wn_chi  = (1.0 / 5.0) * wn_roll   # student-selected bandwidth ratio - Choosing 5 to start
        zeta_chi = 0.7

        cg.Kp_course = 2.0 * zeta_chi * wn_chi * Va / g
        cg.Ki_course = wn_chi**2 * Va / g

        # ---- Sideslip hold ----
        e_beta_max = 2.0 * cg.max_roll
        zeta_beta  = 1.0

        a_beta1 = -density * Va * S * self._get('CYbeta') / (2.0 * m)
        a_beta2 =  density * Va * S * self._get('CYdr')   / (2.0 * m)

        cg.Kp_beta = (cg.max_dr / e_beta_max) * np.sign(a_beta2)
        cg.Ki_beta = (1.0 / a_beta2) * ((a_beta1 + a_beta2 * cg.Kp_beta) / (2.0 * zeta_beta))**2
        cg.Kd_beta = 0.8  # student-selected default

        wn_beta = np.sqrt(a_beta2 * cg.Ki_beta)   # stored for reference

        # ================================================================
        # Longitudinal
        # ================================================================

        # ---- Pitch hold ----
        e_theta_max = 2.0 * cg.max_pitch
        zeta_pitch  = 0.3

        a_theta1 = -density * Va * c * S * self._get('Cmq') * c / (4.0 * Iy)
        a_theta2 = -density * Va**2 * c * S * self._get('Cmalpha') / (2.0 * Iy)
        a_theta3 =  density * Va**2 * c * S * self._get('Cmde')    / (2.0 * Iy)

        cg.Kp_pitch = (cg.max_de / e_theta_max) * np.sign(a_theta3)
        wn_pitch    = np.sqrt(a_theta2 + abs(cg.Kp_pitch * a_theta3))

        cg.Kd_pitch = (2.0 * zeta_pitch * wn_pitch - a_theta1) / a_theta3

        # ---- Height hold ----
        Kpitch_DC  = a_theta3 * cg.Kp_pitch / (a_theta3 * cg.Kp_pitch + a_theta2)
        wn_height  = (1.0 / 5.0) * wn_pitch   # student-selected bandwidth ratio
        zeta_height = 0.7

        cg.Kp_height  = 2.0 * zeta_height * wn_height / (Kpitch_DC * Va)
        cg.Ki_height  = wn_height**2 / (Kpitch_DC * Va)

        # ---- Height state-machine parameters ----
        cg.Kpitch_DC         = Kpitch_DC
        cg.takeoff_height    = 1675.0
        cg.takeoff_pitch     = np.deg2rad(6.0)
        cg.height_hold_limit = 25.0
        cg.climb_throttle    = 0.75

        # ---- Airspeed plant coefficients ----
        alpha = self._trim_var[0]
        de    = self._trim_var[1]
        dt    = self._trim_var[2]

        CL_trim = (self._get('CL0')
                   + self._get('CLalpha') * alpha
                   + self._get('CLde')    * de)

        CD_trim = (self._get('CDmin')
                   + self._get('K') * (CL_trim - self._get('CLmin'))**2)

        dCDdCL  = 2.0 * self._get('K') * (CL_trim - self._get('CLmin'))
        CDalpha = dCDdCL * self._get('CLalpha')   # available for downstream use

        Sprop  = self._get('Sprop')
        Cprop  = self._get('Cprop')
        km     = self._get('kmotor')

        a_v1 = ((density * Va * S / m) * CD_trim
                - density * Sprop * Cprop
                  * (2.0 * (dt - 1.0) * Va + (km - 2.0 * km * dt)) / m)

        a_v2 = (density * Sprop * Cprop
                * (  (2.0 * dt - 1.0) * Va**2
                   + (km - 4.0 * km * dt) * Va
                   + 2.0 * km**2 * dt) / m)

        # ---- Airspeed from pitch ----
        wn_airspeed_p  = (1.0 / 10.0) * wn_pitch   # student-selected
        zeta_airspeed_p = 0.7

        cg.Kp_speed_pitch = (a_v1 - 2.0 * zeta_airspeed_p * wn_airspeed_p) / (Kpitch_DC * g)
        cg.Ki_speed_pitch = -wn_airspeed_p**2 / (Kpitch_DC * g)

        # ---- Airspeed from throttle ----
        wn_airspeed_t   = (1.0 / 10.0) * wn_pitch   # student-selected
        zeta_airspeed_t = 0.3

        cg.Kp_speed_throttle = (2.0 * zeta_airspeed_t * wn_airspeed_t - a_v1) / a_v2
        cg.Ki_speed_throttle = wn_airspeed_t**2 / a_v2

        # ================================================================
        # Linear terms
        # ================================================================
        lt.a_phi1   = a_phi1
        lt.a_phi2   = a_phi2
        lt.a_beta1  = a_beta1
        lt.a_beta2  = a_beta2
        lt.a_theta1 = a_theta1
        lt.a_theta2 = a_theta2
        lt.a_theta3 = a_theta3
        lt.a_v1     = a_v1
        lt.a_v2     = a_v2

    # ------------------------------------------------------------------
    # Convenience repr
    # ------------------------------------------------------------------

    def __repr__(self):
        return (f"CalculateControlGains(\n"
                f"  control_gains={self.control_gains},\n"
                f"  linear_terms={self.linear_terms}\n"
                f")")


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == '__main__':

    # Minimal aircraft parameter object using a simple namespace
    from types import SimpleNamespace

    ap = SimpleNamespace(
        g        = 9.81,
        S        = 0.55,      # m^2
        b        = 2.9,       # m
        c        = 0.19,      # m
        m        = 2.0,       # kg
        Iy       = 0.135,     # kg·m^2

        # Lateral / directional
        Clp      = -0.321,
        Clda     =  0.110,
        CYbeta   = -0.220,
        CYdr     =  0.107,

        # Longitudinal
        Cmq      = -8.794,
        Cmalpha  = -0.713,
        Cmde     = -0.993,
        CL0      =  0.285,
        CLalpha  =  3.450,
        CLde     =  0.360,
        CDmin    =  0.043,
        K        =  0.070,
        CLmin    =  0.0,

        # Propulsion
        Sprop    =  0.2027,
        Cprop    =  1.0,
        kmotor   = 80.0,
    )

    trim_definition = [15.0, 0.0, 1350.0]   # [Va m/s, unused, altitude m]
    trim_variables  = [0.05, -0.02, 0.55]   # [alpha rad, de rad, throttle]

    result = CalculateControlGains(ap, trim_definition, trim_variables)

    print("=== Control Gains ===")
    for field_name, value in vars(result.control_gains).items():
        print(f"  {field_name:25s} = {value:.6f}")

    print("\n=== Linear Terms ===")
    for field_name, value in vars(result.linear_terms).items():
        print(f"  {field_name:10s} = {value:.6f}")
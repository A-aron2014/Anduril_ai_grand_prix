from dataclasses import dataclass, field
import numpy as np
from InertiaTerms import inertia_terms

@dataclass
class AircraftParameters:
    # Constants
    g: float = 9.81  # [m/s^2]

    # Geometry
    S: float = 0.6282  # [m^2]
    b: float = 3.067   # [m]
    c: float = 0.208   # [m]
    AR: float = field(init=False)

    # Mass properties
    m: float = 5.74    # [kg]
    W: float = field(init=False)

    # Inertia
    Ix: float = field(init=False)
    Iy: float = field(init=False)
    Iz: float = field(init=False)
    Ixz: float = field(init=False)
    inertia_matrix: np.ndarray = field(init=False)
    gamma : np.ndarray = field(init=False)
    # Drag model
    CDmin: float = 0.0240
    CLmin: float = 0.2052
    K: float = 0.0549
    e: float = field(init=False)
    CD0: float = field(init=False)
    K1: float = field(init=False)
    CDpa: float = field(init=False)

    # Engine
    Sprop: float = 0.0707  # [m^2]
    Cprop: float = 1.0
    kmotor: float = 30.0

    # Zero-angle aero
    CL0: float = 0.2219
    Cm0: float = 0.0519
    CY0: float = 0.0
    Cl0: float = 0.0
    Cn0: float = 0.0

    # Longitudinal derivatives
    CLalpha: float = 6.196683
    Cmalpha: float = -1.634010
    CLq: float = 10.137584
    Cmq: float = -24.376066
    CLalphadot: float = 0.0
    Cmalphadot: float = 0.0

    # Lateral-directional derivatives
    CYbeta: float = -0.367231
    Clbeta: float = -0.080738
    Cnbeta: float = 0.080613
    CYp: float = -0.064992
    Clp: float = -0.686618
    Cnp: float = -0.039384
    Clr: float = 0.119718
    Cnr: float = -0.052324
    CYr: float = 0.213412

    # Control derivatives
    CLde: float = 0.006776
    Cmde: float = -0.06
    CYda: float = -0.000754
    Clda: float = -0.02
    Cnda: float = -0.000078
    CYdr: float = 0.003056
    Cldr: float = 0.000157
    Cndr: float = -0.000856

    def __post_init__(self):
        # Derived quantities
        self.AR = self.b ** 2 / self.S
        self.W = self.m * self.g

        # Inertia conversion
        SLUGFT2_TO_KGM2 = 14.5939 / (3.2804 ** 2)
        self.Ix = SLUGFT2_TO_KGM2 * 4106 / 12**2 / 32.2
        self.Iy = SLUGFT2_TO_KGM2 * 3186 / 12**2 / 32.2
        self.Iz = SLUGFT2_TO_KGM2 * 7089 / 12**2 / 32.2
        self.Ixz = SLUGFT2_TO_KGM2 * 323.5 / 12**2 / 32.2

        self.inertia_matrix = np.array([
            [ self.Ix,  0.0,     -self.Ixz],
            [ 0.0,      self.Iy,  0.0     ],
            [-self.Ixz, 0.0,      self.Iz ]
        ])

        # Drag coefficients
        self.e = 1.0 / (self.K * self.AR * np.pi)
        self.CD0 = self.CDmin + self.K * self.CLmin**2
        self.K1 = -2.0 * self.K * self.CLmin
        self.CDpa = self.CD0

        #Gamma
        self.gamma = inertia_terms(self)

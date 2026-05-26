import numpy as np

def inertia_terms(aircraft_parameters):
    """
    Compute inertia terms from aircraft parameters.
    
    Args:
        aircraft_parameters: dict with keys 'Ix', 'Iy', 'Iz', 'Ixz'
        
    Returns:
        numpy array with [Gam1, Gam2, Gam3, Gam4, Gam5, Gam6, Gam7, Gam8]
    """
    Jx = aircraft_parameters.Ix
    Jy = aircraft_parameters.Iy
    Jz = aircraft_parameters.Iz
    Jxz = aircraft_parameters.Ixz
    
    Gamma = Jx * Jz - Jxz**2
    
    Gam1 = Jxz * (Jx - Jy + Jz) / Gamma
    Gam2 = (Jz * (Jz - Jy) + Jxz**2) / Gamma
    Gam3 = Jz / Gamma
    Gam4 = Jxz / Gamma
    Gam5 = (Jz - Jx) / Jy
    Gam6 = Jxz / Jy
    Gam7 = ((Jx - Jy) * Jx + Jxz**2) / Gamma
    Gam8 = Jx / Gamma
    
    return np.array([Gam1, Gam2, Gam3, Gam4, Gam5, Gam6, Gam7, Gam8])
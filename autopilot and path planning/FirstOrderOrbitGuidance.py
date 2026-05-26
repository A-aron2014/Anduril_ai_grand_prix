import numpy as np
from RunHW6 import FlightPathAnglesFromState
def first_order_orbit_guidance(t, pos, orbit_speed, orbit_radius, orbit_center, orbit_flag, orbit_gains, debug=False):
# %
# % STUDENTS complete this function to simulate the first order model with
# % their orbit guidance algorithm
# %

#vel = [10; 0; 0]; 

    d = np.sqrt((pos[0] - orbit_center[0])**2 + (pos[1] - orbit_center[1])**2)
    phi = np.atan2(pos[1] - orbit_center[1],pos[0] - orbit_center[0])
    chi = phi + orbit_flag*(np.pi/2 + np.atan2(orbit_gains.kr*(d-orbit_radius),orbit_radius))

    pn = orbit_speed*np.cos(chi)
    pe = orbit_speed*np.sin(chi)
    #h_dot  = orbit_gains.kz * (-orbit_center[2] - pos[2])
    return np.array([pn,pe,0])


def orbit_guidance(aircraft_state,orbit_speed, orbit_radius, orbit_center, orbit_flag, orbit_gains):
    h_commanded = -orbit_center[2] #aircraft_state[2]
    d = np.sqrt((aircraft_state[0] - orbit_center[0])**2 + (aircraft_state[1] - orbit_center[1])**2)
    phi = np.atan2(aircraft_state[1] - orbit_center[1],aircraft_state[0] - orbit_center[0])
    
    flight_path_angles =  FlightPathAnglesFromState(aircraft_state)

    # #no wind
    course_rate  = orbit_flag*orbit_speed/orbit_radius

    while phi - flight_path_angles[1] < -np.pi:
        phi = phi + 2*np.pi
    while phi - flight_path_angles[1] > np.pi:
        phi = phi - 2*np.pi

    chi_commanded  = phi + orbit_flag*(np.pi/2 + np.atan2(orbit_gains.kr*(d-orbit_radius),orbit_radius))

    return np.array([h_commanded, 0, chi_commanded, course_rate, flight_path_angles[0]])
"""
RunHW6.py - Python equivalent of the MATLAB RunHW6.m script.

Eric W. Frew / ASEN 5519
Original MATLAB: 3/2/23

This is the top-level simulation runner for HW6. It:
  1. Loads aircraft parameters (ttwistor)
  2. Calculates trim state and control inputs
  3. Calculates autopilot control gains
  4. Sets guidance commands
  5. Sets initial conditions
  6. Runs the simulation loop (control sampled at Ts, dynamics via ODE45)
  7. Plots results
  8. Optionally animates the flight

Students replace the stub functions with their own implementations from HW3.

Dependencies (must be provided separately):
  - ttwistor.py            : aircraft_parameters definition
  - CalculateTrimVariables : trim solver
  - TrimConditionFromDefinitionAndVariables : state/control from trim vars
  - CalculateControlGainsSimpleSLC_Nondim_Ttwistor (already translated)
  - SimpleSLCAutopilot     : autopilot law
  - AircraftEOM            : equations of motion (for scipy ODE solver)
  - TransformFromInertialToBody (already in plot_simulation_with_commands.py)
  - WindAnglesFromVelocityBody
  - PlotSimulationWithCommands (already translated)
  - DrawAircraft / AnimateSimulation / DefineTTwistor (already translated)
"""

import numpy as np
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# Import translated modules (adjust paths/names as needed for your project)
# ---------------------------------------------------------------------------
from ttwistor import AircraftParameters
from CalculateControlGainSimpleSLC import CalculateControlGains
from PlotSimulationWithCommands import PlotSimulationWithCommands
from DefineTTwistor import DefineTTwistor
from DrawAircraft import DrawAircraft
from AnimateSimulation import AnimateSimulation
from scipy.optimize import minimize
from ttwistor import AircraftParameters
from SimpleSLCAutopilot import SimpleSLCAutopilot
import stdatmo
from InertiaTerms import inertia_terms
# ---------------------------------------------------------------------------
# Stub functions – STUDENTS REPLACE WITH THEIR HW3 IMPLEMENTATIONS
# ---------------------------------------------------------------------------

def TrimConditionFromDefinitionAndVariables(trim_variables, trim_definition):
    '''
    Docstring for CalcTrimForSLUF
    Assuming no wind here, therefore no induced sideslip or flight 
    
    :param trim_definition: definition for SLUF [Va, gamma, h] There is no Radius for a straight line(R = infinity)
    :param trim_variables: Trim Variables for SLUF [alpha, beta, phi, de,da,dr,dt]
    :param inertial wind vector for determining body velocity

    returns aircraft state vector, control surface vector
    '''
    # #For SLUF there is no turning, no change in height(climb rate is 0), course angle X is 0, angular velocity is a 0 vector, 
    # #with no wind and flying in a straight line, theta = 0 so gamma_a = alpha
    Va,gamma,h = trim_definition
    alpha, de, dt, = trim_variables

    #gamma = theta - alpha but gamma = 0
    theta = gamma + alpha

    wind_angles = np.array([Va,0,alpha])
    euler_angles = np.array([0,theta,0])
    v_body = WindAnglesToAirRelativeVelocityVector(wind_angles)

    u,v,w = v_body

    p=q=r=0.0
    phi=psi=0.0

    x = np.array([0, 0, -h, phi, theta, psi, u, v, w, p, q, r])
    u = np.array([de,0,0,dt])
    
    return x,u

def calc_cost_val_for_SLUF(trim_variables, trim_definition, aircraft_parameters):
    '''
    Docstring for Calc_Cost_Val_For_FLUF
    
    :param trim_definition: Description
    :param trim_variables: Description
    :param aircraft_parameters: Description

    return cost J(x_tv|x_tf, ap)
    '''
    Va,gamma,h = trim_definition
    x_ideal, u_ideal = TrimConditionFromDefinitionAndVariables(trim_variables, trim_definition)
    time = []
    x_dot_est = AircraftEOM(time,aircraft_state=x_ideal,aircraft_surfaces=u_ideal, aircraft_parameters=aircraft_parameters)
    x_dot_est[2] = 0.0   # enforce straight-and-level
    x_dot_ideal = np.array([ x_dot_est[0], x_dot_est[1], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # print(f'The estimated state variables are{x_dot_est}')
    e_trim = x_dot_ideal - x_dot_est
    e_trim_angular = e_trim[3:12]
    cost = np.linalg.norm(e_trim_angular)**2

    return cost

def CalculateTrimVariables(trim_definition, aircraft_parameters):
    dt0 = 0.1
    de0 = 0.0
    alpha0 = 0.0

    max_angle = 45*np.pi/180

    bounds = [(-max_angle, max_angle),
               (-max_angle, max_angle),
               (0,1)]

    x0 = np.array([alpha0,de0,dt0])
    #use the cost function to select the optimal aircraft state and control states
    result = minimize(calc_cost_val_for_SLUF, x0=x0,args = (trim_definition,aircraft_parameters), method='SLSQP', bounds=bounds,options={'ftol':1e-9, 'disp' : True})

    if result.success:
        print(f'Optimization successful!')
        # print(f'  Alpha:    {np.rad2deg(result.x[0]):.3f}°')
        # print(f'  Delta_e:  {np.rad2deg(result.x[1]):.3f}°')
        print(f'  Alpha:    {result.x[0]:.3f}')
        print(f'  Delta_e:  {result.x[1]:.3f}')
        print(f'  Throttle: {result.x[2]:.3f}')
        print(f'  Final cost: {result.fun:.2e}')
        trim_variables = result.x
        cost = result.fun
        #aircraft_state, control_input = TrimConditionFromDefinitionAndVariables(trim_variables,trim_definition)
        return trim_variables, cost
    else:
        raise ValueError(f"Optimization failed: {result.message}")

def AeroForcesAndMoments_BodyState_WindCoeffs(aircraft_state, aircraft_surfaces, density, aircraft_parameters):
    #Recall aircraft_state is x = [x_E, y_E, z_E, phi, theta, psi, u_E, v_E, w_E, p, q, r]
    #np array of wind components in the Inertial Frame

    inertial_position = aircraft_state[0:3] #x_E, y_E, z_E
    euler_angles = aircraft_state[3:6] #phi, theta, psi
    vel_of_aircraft_in_body = aircraft_state[6:9] 
    #inertial_velocity = aircraft_state[6:9] 
    angular_velocity = aircraft_state[9:12] #p, q, r

    #Rotate velocity into body coordinates
    wind = WindAnglesFromVelocityBody(vel_of_aircraft_in_body)
    Va = wind[0]
    beta = wind[1]
    alpha = wind[2]

    p_hat = (angular_velocity[0]*aircraft_parameters.b)/(2*Va)
    q_hat = (angular_velocity[1]*aircraft_parameters.c)/(2*Va)
    r_hat = (angular_velocity[2]*aircraft_parameters.b)/(2*Va)

    #Calculating Pressure
    Q = 0.5 * density * Va**2
    P= Q * aircraft_parameters.S

    #Calculating Coefficients
    C_L  = aircraft_parameters.CL0 + aircraft_parameters.CLalpha * alpha  + aircraft_parameters.CLq * q_hat + aircraft_parameters.CLde * aircraft_surfaces[0]
    C_D = aircraft_parameters.CDmin + aircraft_parameters.K * (C_L - aircraft_parameters.CLmin)**2
    C_X = -C_D * np.cos(alpha) + C_L * np.sin(alpha)
    C_Y = aircraft_parameters.CYbeta * beta + aircraft_parameters.CYp * p_hat + aircraft_parameters.CYr * r_hat + aircraft_parameters.CYda * aircraft_surfaces[1] + aircraft_parameters.CYdr * aircraft_surfaces[2]
    C_Z = -C_D * np.sin(alpha) - C_L * np.cos(alpha)


    C_l = aircraft_parameters.Clbeta * beta + aircraft_parameters.Clp*p_hat + aircraft_parameters.Clr*r_hat + aircraft_parameters.Clda*aircraft_surfaces[1] + aircraft_parameters.Cldr * aircraft_surfaces[2]


    C_m = aircraft_parameters.Cm0 + aircraft_parameters.Cmalpha * alpha + aircraft_parameters.Cmq * q_hat + aircraft_parameters.Cmde * aircraft_surfaces[0]


    C_n = aircraft_parameters.Cnbeta * beta + aircraft_parameters.Cnp * p_hat + aircraft_parameters.Cnr * r_hat + aircraft_parameters.Cnda*aircraft_surfaces[1] + aircraft_parameters.Cndr * aircraft_surfaces[2]


    C_T = 2 * (aircraft_parameters.Sprop/aircraft_parameters.S)*aircraft_parameters.Cprop * (aircraft_surfaces[3]/Va**2) * (Va + aircraft_surfaces[3] * (aircraft_parameters.kmotor - Va)) * (aircraft_parameters.kmotor - Va)

    #Calculating X, Y, And Z Forces on the aircraft due to aerodynamics
    f = P* np.array([ [C_X],
                  [C_Y],
                  [C_Z]
                  ]).reshape((3,1))

    #For this Thrust is considered a part of the aerodynamic forces. but only in the X direction in the body Frame
    T = P*C_T
    f[0]+=T

    #Need to Calculate Rotational Forces - Moments
    #Calculate G  = [ L M N ]
    C_moments = np.array([ [aircraft_parameters.b*C_l],
                            [aircraft_parameters.c*C_m],
                            [aircraft_parameters.b*C_n]
                            ])
    G = P * C_moments
    #There is no propeller torque due to the dual propellers
    return f, G

def AircraftForcesAndMoments(aircraft_state, aircraft_surfaces, density, aircraft_parameters):
    '''
    [aircraft forces, aircraft moments] =
    AircraftForcesAndMoments(aircraft state, aircraft surfaces,
    wind inertial, density, aircraft parameters)
    Create the above function that takes as input the aircraft state, the control input vector,
    the inertial wind velocity in inertial coordinates, the air density, and the aircraft parameters
    structure and returns the total force and moment acting on the aircraft expressed in body
    coordinates. The total force includes the aerodynamic force, propulsive force, and weight.
    The total moment includes aerodynamic and propulsive moments. The output of the function
    should be two vectors, one for the force and one for the moment.
    '''
    #Recall aircraft_state is x = [x_E, y_E, z_E, phi, theta, psi, u_E, v_E, w_E, p, q, r]
    #euler_angles = np.zeros([3,1])
    euler_angles = aircraft_state[3:6] #phi, theta, psi
    #Aerodynamic Forces
    f = np.zeros([3,1])
    f_g = f.copy()
    f_g[2] = aircraft_parameters.m * aircraft_parameters.g #Mass * gravity expressed in the inertial Frame
    #Need to Rotate Gravity forces to the body frame
    gravity_inertial_to_body = TransformFromInertialToBody(f_g,euler_angles)
    f_aero, G = AeroForcesAndMoments_BodyState_WindCoeffs(aircraft_state,aircraft_surfaces, density,aircraft_parameters)

   # print("Gravity body:", gravity_inertial_to_body.flatten())

    f = f_aero + gravity_inertial_to_body

    return f,G
def AircraftEOM(time, aircraft_state, aircraft_surfaces, aircraft_parameters):
    '''
    [xdot] = AircraftEOM(time,aircraft state,aircraft surfaces,wind inertial,
    aircraft parameters) Implement the full equations of motion by returning the derivative
    of the state vector, ˙x. I strongly recommend writing this function with vector equations,
    not by typing out the differential equation for each separate term. Note, the input time is
    needed for the Matlab simulation tools. Although it is not used, it needs to be included.
'''
    #                               0    1    2    3    4      5    6    7    8   9  10 11
    #Recall aircraft_state is x = [x_E, y_E, z_E, phi, theta, psi, u, v, w, p, q, r]
    x = aircraft_state#.flatten() #12x1 state vector
    H = -x[2]
    atmosphere = stdatmo.std_atmo(H)
    density = atmosphere.rho

    #print(f'The density is: {density}')
    inertial_position = x[0:3] #x_E, y_E, z_E
    euler_angles = x[3:6] #phi, theta, psi
    v_B = x[6:9] #u_E, v_E, w_E
    w_B = x[9:12] #p, q, r
    #Calculate the Forces and Moments acting on the aircraft
    f_B, G_B = AircraftForcesAndMoments(x,aircraft_surfaces,density,aircraft_parameters)

    #Equation 1
    position_dot = TransformFromBodyToInertial(v_B,euler_angles)

    #Define Rate of change of the euler angles
    T = np.array([[1, np.sin(x[3])*np.tan(x[4]), np.cos(x[3]) * np.tan(x[4])],
                    [0,     np.cos(x[3]),               -np.sin(x[3])],
                    [0, np.sin(x[3])*(1/np.cos(x[4])), np.cos(x[3])*(1/np.cos(x[4]))]  
                  ])
    #Equation 2
    o_dot = T @ w_B

    #Equation 3
    w = w_B
    v = v_B
    f = f_B.flatten()

    V_dot = f/aircraft_parameters.m - np.cross(w, v)
    #V_dot = f/aircraft_parameters.m - np.cross(w, v)


    #equation 4
    # I_inv = np.linalg.inv(aircraft_parameters.inertia_matrix)
    # w = w_B.reshape((3,1))
    # h = aircraft_parameters.inertia_matrix @ w
    # G = G_B
    # omega_cross = np.cross(w.flatten(),h.flatten()).reshape((3,1))
    # w_dot = (I_inv @ (G-omega_cross)).flatten()

    # Extract scalar values properly
    G_0 = float(G_B[0,0])
    G_1 = float(G_B[1,0])
    G_2 = float(G_B[2,0])

    p_dot = aircraft_parameters.gamma[0] *w_B[0]*w_B[1] - aircraft_parameters.gamma[1]*w_B[1]*w_B[2] + aircraft_parameters.gamma[2]*G_0 + aircraft_parameters.gamma[3]*G_2
    q_dot = aircraft_parameters.gamma[4]*w_B[0]*w_B[2] - aircraft_parameters.gamma[5]*(w_B[0]**2 - w_B[2]**2) + (1/aircraft_parameters.Iy)*G_1
    r_dot = aircraft_parameters.gamma[6]*w_B[0]*w_B[1] - aircraft_parameters.gamma[0]*w_B[1]*w_B[2] + aircraft_parameters.gamma[3]*G_0 + aircraft_parameters.gamma[7]*G_2

    w_dot = np.array([p_dot, q_dot, r_dot]).reshape((3,))

    x_dot = np.hstack((position_dot,o_dot,V_dot,w_dot))
    return x_dot

def EOM_wrapper(t,x,u,w, aircraft_parameters):
    """
    Wrapper for Aircraft EOM to work with solve_ivp
    
    Args:
        t: time (required by solve_ivp but not used)
        x: state vector (12,)
        u: control input vector (4,)
        w: wind vector (3,)
        aircraft_parameters: aircraft parameter structure
    
    Returns:
        x_dot: derivative of state vector (12,)
    """

    x_dot = AircraftEOM(t,x, u,w,aircraft_parameters)

    return x_dot

def WindAnglesFromVelocityBody(v_body):
    """
    returns [Va, alpha, beta] from body-frame air-relative velocity.
    Replace with your implementation.
    """
    #extract elements from input vector
    u,v,w = v_body
    #Airspeed V_a = the magnitude of V_body
    Va = np.sqrt(u**2 + v**2 + w**2)
    #Handles windspeed near zero
    if Va <= 1e-6:
        raise ValueError("Airspeed too small to compute wind angles!")
    alpha = np.arctan2(w, u)
    beta  = np.arcsin(v / Va) if Va > 1e-6 else 0.0
    return np.array([Va, beta, alpha])

def WindAnglesToAirRelativeVelocityVector(wind_angles):
    """Calculate the aircraft air relative velocity vector in body coordinates  from the airspeed, side slip, and angle of attack(Wind Angles)
        Return a column vector
    """
    V_a, beta, alpha = wind_angles
    #Inverting the velocity to wind angles caculations
    u = V_a*np.cos(alpha)*np.cos(beta)
    v = V_a*np.sin(beta)
    w = V_a*np.sin(alpha)*np.cos(beta)
    
    velocity_body = np.array([u,v,w])
    return velocity_body

def TransformFromInertialToBody(vector_inertial, euler_angles):
    """Body-from-inertial rotation (ZYX Euler)."""
    """For a vector given in inertial coordinates, determine the components in body coordinates."""
    inertial_to_body_rotation = RotationMatrix321(euler_angles)

    vector_body = inertial_to_body_rotation @ vector_inertial

    return vector_body
def TransformFromBodyToInertial(vector_body, euler_angles):
    """For a vector given in body coordinates, determine the components in inertial coordinates."""
    body_to_inertial_rotation = RotationMatrix321(euler_angles)
    #Multiply the rotation matrix transpose with the vector relative to the Body Frame
    vector_inertial = body_to_inertial_rotation.T @ vector_body

    return vector_inertial


def RotationMatrix321(euler_angles):
    """Given the Euler Angles calculate and return the Rotation matrix in 3-2-1 Format"""
    #Extract angles from the input vector
    phi = float(euler_angles[0])
    theta = float(euler_angles[1])
    psi = float(euler_angles[2])

    #Define Rotation Matrix 1
    R3 = np.array([[np.cos(psi), np.sin(psi), 0],
                   [-np.sin(psi), np.cos(psi),0],
                   [0,          0,      1]])
    #Define Rotation Matrix 2 from Yaw rotation to Pitch Rotation
    R2 = np.array([[np.cos(theta), 0,  -np.sin(theta)],
                   [0,          1,          0],
                   [np.sin(theta), 0,   np.cos(theta)]])
    #Define Rotation Matrix 3 from Pitch Rotation to Roll Rotation
    R1 = np.array([[1,      0,          0],
                   [0,  np.cos(phi), np.sin(phi)],
                   [0, -np.sin(phi), np.cos(phi)]])
    full_rotation = R1 @ R2 @ R3
    return full_rotation

def FlightPathAnglesFromState(aircraft_state):
    wind_angles = WindAnglesFromVelocityBody(aircraft_state[6:9])
    Vg = wind_angles[0]# - wind if there were any
    gamma_a = aircraft_state[4] - wind_angles[2] # Theta - Alpha
    gamma = gamma_a #Because again no wind
    chi = aircraft_state[5] #Because no sideslip angle/wind

    return np.array([Vg,chi,gamma])


# ---------------------------------------------------------------------------
# Main simulation runner
# ---------------------------------------------------------------------------

class RunHW6:
    """
    Top-level simulation runner equivalent to RunHW6.m.

    Parameters
    ----------
    aircraft_parameters : object
        Loaded from ttwistor (or equivalent Python module).
    animate : bool, optional
        Set True to run the 3-D animation after plotting (slow – off by default,
        mirroring the commented-out section in the MATLAB script).
    """

    def __init__(self, aircraft_parameters, animate=False):
        self.ap      = aircraft_parameters
        self.animate = animate

    def run(self):
        ap = self.ap

        # ----------------------------------------------------------------
        # Trim
        # ----------------------------------------------------------------
        V_trim     = 18.0
        h_trim     = 1805.0
        gamma_trim = 0.0
        trim_definition = np.array([V_trim, gamma_trim, h_trim])

        # STUDENTS: replace stubs with HW3 functions
        trim_variables, fval = CalculateTrimVariables(trim_definition, ap)
        aircraft_state_trim, control_input_trim = \
            TrimConditionFromDefinitionAndVariables(trim_variables, trim_definition)

        print(f"Trim found with residual fval = {fval:.6e}")

        # ----------------------------------------------------------------
        # Control gains
        # ----------------------------------------------------------------

        gains_result = CalculateControlGains(ap, trim_definition, trim_variables)
        control_gain_struct = gains_result.control_gains
        linear_terms        = gains_result.linear_terms

        print("Control gains calculated.")
        print(f"  wn_roll  = {np.sqrt(abs(linear_terms.a_phi2 * control_gain_struct.Kp_roll)):.4f} rad/s")

        # Store trim control input inside gain struct for autopilot use
        control_gain_struct.u_trim = control_input_trim

        # ----------------------------------------------------------------
        # Guidance commands
        # ----------------------------------------------------------------
        h_c        = h_trim                  # commanded altitude (m)
        h_dot_c    = 0.0                     # commanded altitude rate (m/s)
        chi_c      = np.deg2rad(40.0)        # commanded course (rad)
        chi_dot_ff = 0.0                     # course rate feedforward (rad/s)
        Va_c       = V_trim                  # commanded airspeed (m/s)

        # ----------------------------------------------------------------
        # Initial conditions
        # ----------------------------------------------------------------
        aircraft_state0        = aircraft_state_trim.copy()
        aircraft_state0[2]     = -1655.0            # pd: climb mode starts at h=1675
        aircraft_state0[3]     = np.deg2rad(0.0)    # phi; set to 180° when confident

        control_input0  = control_input_trim.copy()
        wind_inertial   = np.zeros(3)

        # ----------------------------------------------------------------
        # Simulation parameters
        # ----------------------------------------------------------------
        Ts     = 0.1    # control sample period (s)
        Tfinal = 100.0  # total simulation time (s)
        control_gain_struct.Ts = Ts

        n_ind = int(Tfinal / Ts)

        # Pre-allocate storage  (MATLAB: aircraft_array, control_array, …)
        n_states   = len(aircraft_state0)
        n_controls = len(control_input0)
        n_wind     = 3
        n_cmd      = 12

        aircraft_array = np.zeros((n_states,   n_ind + 1))
        control_array  = np.zeros((n_controls, n_ind + 1))
        wind_array     = np.zeros((n_wind,     n_ind + 1))
        x_command      = np.zeros((n_cmd,      n_ind + 1))
        wind_angles_arr= np.zeros((3,          n_ind + 1))
        time_iter      = np.zeros(n_ind + 1)

        aircraft_array[:, 0] = aircraft_state0
        control_array[:, 0]  = control_input0
        time_iter[0]         = 0.0

        # ----------------------------------------------------------------
        # Simulation loop
        # ----------------------------------------------------------------
        print(f"Running simulation: Ts={Ts}s, Tfinal={Tfinal}s, {n_ind} steps …")
        autopilot = SimpleSLCAutopilot(control_gain_struct)
        for i in range(1, n_ind + 1):
            t_start = Ts * (i - 1)
            t_end   = Ts * i
            tspan   = (t_start, t_end)

            wind_array[:, i - 1] = wind_inertial

            # Wind angles at current step
            wind_body = TransformFromInertialToBody(
                wind_inertial, aircraft_array[3:6, i - 1]
            )
            air_rel_vel_body = aircraft_array[6:9, i - 1] - wind_body
            wind_angles_arr[:, i - 1] = WindAnglesFromVelocityBody(air_rel_vel_body)

            # Guidance objectives vector
            control_objectives = np.array([
                h_c,
                h_dot_c,
                chi_c,
                chi_dot_ff,
                Va_c,
            ])

            # # Autopilot
            # control_slc, x_c_slc = SimpleSLCAutopilot(
            #     t_start,
            #     aircraft_array[:, i - 1],
            #     wind_angles_arr[:, i - 1],
            #     control_objectives,
            #     control_gain_struct,
            # )
            control_slc, x_c_slc = autopilot.update(
                                                t_start,
                                                aircraft_array[:, i - 1],
                                                wind_angles_arr[:, i - 1],
                                                control_objectives,
                                            )
            control_array[:, i - 1] = control_slc
            x_command[:, i - 1]     = x_c_slc
            x_command[4, i - 1]     = trim_variables[0]   # alpha command = trim alpha

            # Aircraft dynamics  (scipy equivalent of ode45)
            sol = solve_ivp(
                fun=lambda t, y: AircraftEOM(
                    t, y, control_array[:, i - 1], ap
                ),
                t_span=tspan,
                y0=aircraft_array[:, i - 1],
                method='RK45',
                dense_output=False,
                rtol=1e-6,
                atol=1e-8,
            )

            aircraft_array[:, i] = sol.y[:, -1]
            time_iter[i]         = sol.t[-1]
            wind_array[:, i]     = wind_inertial
            control_array[:, i]  = control_array[:, i - 1]
            x_command[:, i]      = x_command[:, i - 1]

            if i % 100 == 0:
                print(f"  Step {i}/{n_ind}  t={time_iter[i]:.1f}s  "
                      f"h={-aircraft_array[2, i]:.1f}m")

        print("Simulation complete.")

        # ----------------------------------------------------------------
        # Store results as instance attributes for inspection
        # ----------------------------------------------------------------
        self.time_iter      = time_iter
        self.aircraft_array = aircraft_array
        self.control_array  = control_array
        self.wind_array     = wind_array
        self.x_command      = x_command

        # ----------------------------------------------------------------
        # Plotting
        # ----------------------------------------------------------------

        plotter = PlotSimulationWithCommands()
        plotter.plot(
            time_iter,
            aircraft_array,
            control_array,
            wind_array,
            x_command,
            color='b',
        )
        plotter.show()

        # ----------------------------------------------------------------
        # Optional animation  (slow – disabled by default)
        # Students: set animate=True once autopilot is tuned
        # ----------------------------------------------------------------
        if self.animate:


            pts    = DefineTTwistor().pts
            drawer = DrawAircraft(pts)

            for aa in range(len(time_iter)):
                drawer.update(time_iter[aa], aircraft_array[:, aa])

            # Full-path animation (equivalent to AnimateSimulation call)

            AnimateSimulation(tout=time_iter, xarray=aircraft_array.T)

        return self


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # Load aircraft parameters from ttwistor module.
    # Uncomment and adjust the import once ttwistor.py is available.
    #
    # from ttwistor import aircraft_parameters
    aircraft_parameters = AircraftParameters()
    sim = RunHW6(aircraft_parameters, animate=True)
    sim.run()

    print("RunHW6: ready to run.")
    print("To execute:")
    print("  from ttwistor import aircraft_parameters")
    print("  from run_hw6 import RunHW6")
    print("  RunHW6(aircraft_parameters, animate=False).run()")
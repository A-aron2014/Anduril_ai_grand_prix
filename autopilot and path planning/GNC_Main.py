import numpy as np
import matplotlib.pyplot as plt
import seaborn
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from ttwistor import AircraftParameters
import stdatmo
from InertiaTerms import inertia_terms
import control

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Homework 1 Functions
def AirRelativeVelocityVectorToWindAngles(velocity_body):
    """Given the air relative velocity vector in body coordinates, return the wind angles in a column vector"""
    #extract elements from input vector
    #u,v,w = velocity_body#.flatten()
    #Airspeed V_a = the magnitude of V_body
    #V_a = np.sqrt(u**2 + v**2 + w**2)
    V_a = np.linalg.norm(velocity_body) #Simplify compute time and clean up code a bit
    #Handles windspeed near zero
    if V_a <= 1e-6:
        raise ValueError("Airspeed too small to compute wind angles!")

    #need to calculate Alpha
    alpha = np.atan2(velocity_body[2],velocity_body[0])

    #clipping guarantees no floating point error causes values exceeding +- 1
    beta = np.arcsin(np.clip(velocity_body[1]/V_a, -1.0,1.0))

    wind_angles = np.array([V_a, beta,alpha])
    return wind_angles
def WindAnglesToAirRelativeVelocityVector(wind_angles):
    """Calculate the aircraft air relative velocity vector in body coordinates  from the airspeed, side slip, and angle of attack(Wind Angles)
        Return a column vector
    """
    V_a, beta, alpha = wind_angles#.flatten()
    #Inverting the velocity to wind angles caculations
    u = V_a*np.cos(alpha)*np.cos(beta)
    v = V_a*np.sin(beta)
    w = V_a*np.sin(alpha)*np.cos(beta)
    
    velocity_body = np.array([u,v,w])
    return velocity_body

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

def TransformFromInertialToBody(vector_inertial, euler_angles):
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
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Homework 2 Functions
def AeroForcesAndMoments_BodyState_WindCoeffs(aircraft_state, aircraft_surfaces, wind_inertial, density, aircraft_parameters):
    """
    Docstring for AeroForcesAndMoments_BodyState_WindCoeffs
    :param aircraft_state: Aircraft State Vector - 12 Vars
    :param aircraft_surfaces: Control Input Vector to the Aircraft [delta_e, delta_a, delta_r, delta_t]
    :param wind_inertial: Inertial Wind Velocity in the Inertial Frame
    :param density: Scalar Air Density around Aircraft
    :param aircraft_parameters: 
    :returns aerodynamic force matrix and moment matrix
    Create the above function that takes as input the aircraft state, the control input vector,
    the inertial wind velocity in inertial coordinates, the air density, and the aircraft parameters
    structure and returns the aerodynamic force and moment acting on the aircraft expressed in
    body coordinates. For this function the propulsive force is considered part of the aerodynamic
    force, and it DOES NOT include weight. The moment includes aerodynamic and propulsive
    moments. The output of the function should be two vectors, one for the force and one for
    the moment.
    """
    #Recall aircraft_state is x = [x_E, y_E, z_E, phi, theta, psi, u_E, v_E, w_E, p, q, r]
    #np array of wind components in the Inertial Frame
    #W = wind_inertial
    inertial_position = aircraft_state[0:3] #x_E, y_E, z_E
    euler_angles = aircraft_state[3:6] #phi, theta, psi
    body_velocity = aircraft_state[6:9] 
    #inertial_velocity = aircraft_state[6:9] 
    angular_velocity = aircraft_state[9:12] #p, q, r
    vel_of_aircraft_in_body = body_velocity - TransformFromInertialToBody(wind_inertial, euler_angles)

    #Rotate velocity into body coordinates
    wind = AirRelativeVelocityVectorToWindAngles(vel_of_aircraft_in_body)
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

def AircraftForcesAndMoments(aircraft_state, aircraft_surfaces,wind_inertial, density, aircraft_parameters):
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
    # I think this means include Gravity and Prpulsive Force
    #Recall aircraft_state is x = [x_E, y_E, z_E, phi, theta, psi, u_E, v_E, w_E, p, q, r]
    #euler_angles = np.zeros([3,1])
    euler_angles = aircraft_state[3:6] #phi, theta, psi
    #Aerodynamic Forces
    f = np.zeros([3,1])
    f_g = f.copy()
    f_g[2] = aircraft_parameters.m * aircraft_parameters.g #Mass * gravity expressed in the inertial Frame
    #Need to Rotate Gravity forces to the body frame
    gravity_inertial_to_body = TransformFromInertialToBody(f_g,euler_angles)
    f_aero, G = AeroForcesAndMoments_BodyState_WindCoeffs(aircraft_state,aircraft_surfaces,wind_inertial, density,aircraft_parameters)

   # print("Gravity body:", gravity_inertial_to_body.flatten())

    f = f_aero + gravity_inertial_to_body

    return f,G

def AircraftEOM(time, aircraft_state, aircraft_surfaces,wind_inertial, aircraft_parameters):
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
    f_B, G_B = AircraftForcesAndMoments(x,aircraft_surfaces,wind_inertial,density,aircraft_parameters)

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

    # g_i = np.array([0.0, 0.0, aircraft_parameters.g])  # NED
    # g_b = TransformFromInertialToBody(g_i, euler_angles)

    V_dot = f/aircraft_parameters.m - np.cross(w, v) #+ g_b
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

def PlotSimulation(time, aircraft_state_array, control_input_array, col):

    #change back from radians to degrees for plotting
    rad2degree = 180/np.pi
        # State labels
    pos_labels   = ['x_E [m]', 'y_E [m]', 'z_E [m]']
    eul_labels   = [r'$\phi$ [deg]', r'$\theta$ [deg]', r'$\psi$ [deg]']
    vel_labels   = ['u_E [m/s]', 'v_E [m/s]', 'w_E [m/s]']
    ang_labels   = ['p [deg/s]', 'q [deg/s]', 'r [deg/s]']
    ctrl_labels  = [r'$\delta_e$ [deg]', r'$\delta_a$ [deg]', r'$\delta_r$ [deg]', r'$\delta_t$ [%]']
    
    # ---------------- ALL 12 STATES IN ONE FIGURE ----------------
    fig1, axs1 = plt.subplots(6, 2, num=1, figsize=(12, 14))
    fig1.suptitle('Aircraft States vs Time', fontsize=16, fontweight='bold')
    
    # Column 1: Position and Euler angles
    for i in range(3):
        axs1[i, 0].plot(time, aircraft_state_array[i, :], col, linewidth=1.5)
        axs1[i, 0].set_ylabel(pos_labels[i], fontsize=10)
        axs1[i, 0].grid(True, alpha=0.3)
    
    for i in range(3):
        axs1[i+3, 0].plot(time, aircraft_state_array[i+3, :]*rad2degree, col, linewidth=1.5)
        axs1[i+3, 0].set_ylabel(eul_labels[i], fontsize=10)
        axs1[i+3, 0].grid(True, alpha=0.3)
    
    # Column 2: Velocities and angular rates
    for i in range(3):
        axs1[i, 1].plot(time, aircraft_state_array[i+6, :], col, linewidth=1.5)
        axs1[i, 1].set_ylabel(vel_labels[i], fontsize=10)
        axs1[i, 1].grid(True, alpha=0.3)
    
    for i in range(3):
        axs1[i+3, 1].plot(time, aircraft_state_array[i+9, :]*rad2degree, col, linewidth=1.5)
        axs1[i+3, 1].set_ylabel(ang_labels[i], fontsize=10)
        axs1[i+3, 1].grid(True, alpha=0.3)
    
    # Set x-labels for bottom row
    axs1[5, 0].set_xlabel('Time [s]', fontsize=10)
    axs1[5, 1].set_xlabel('Time [s]', fontsize=10)
    
    plt.tight_layout()
    
    # ---------------- CONTROL INPUTS ----------------
    fig2, axs2 = plt.subplots(4, 1, num=2, figsize=(10, 10))
    fig2.suptitle('Control Inputs vs Time', fontsize=16, fontweight='bold')
    
    for i in range(4):
        if i < 3:  # Control surfaces in degrees
            axs2[i].plot(time, control_input_array[i, :]*rad2degree, col, linewidth=1.5)
        else:  # Throttle in percentage or original units
            axs2[i].plot(time, control_input_array[i, :], col, linewidth=1.5)
        axs2[i].set_ylabel(ctrl_labels[i], fontsize=10)
        axs2[i].grid(True, alpha=0.3)
    
    axs2[-1].set_xlabel('Time [s]', fontsize=10)
    plt.tight_layout()
    
    # ---------------- 3D TRAJECTORY ----------------
    fig3 = plt.figure(num=3, figsize=(12, 9))
    ax3 = fig3.add_subplot(111, projection='3d')
    fig3.suptitle('3D Aircraft Trajectory', fontsize=16, fontweight='bold')
    
    x = aircraft_state_array[0, :]
    y = aircraft_state_array[1, :]
    z = -aircraft_state_array[2, :]  # Positive height upward
    
    ax3.plot(x, y, z, col, linewidth=2, label='Trajectory')
    ax3.scatter(x[0], y[0], z[0], c='green', marker='o', s=100, label='Start', edgecolors='black')
    ax3.scatter(x[-1], y[-1], z[-1], c='red', marker='X', s=150, label='End', edgecolors='black')
    
    ax3.set_xlabel('x_E [m]', fontsize=11)
    ax3.set_ylabel('y_E [m]', fontsize=11)
    ax3.set_zlabel('Altitude [m]', fontsize=11)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    return

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

def EOM_wrapper_Pulsed(t,x,u,w, aircraft_parameters):
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
     
    if t<0.1:
        single_pulse = 0.261799
    else:
        single_pulse = 0

    u[0] += single_pulse 
        
    x_dot = AircraftEOM(t,x, u,w,aircraft_parameters)

    return x_dot#.flatten()

def EOM_wrapper_Aieleron_DoublePulsed(t,x,u,w, aircraft_parameters):
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
     
    if t<0.1:
        pulse = .261799
    elif t>0.1 and t<0.2:
        pulse = -.523598
    else:
        pulse = 0

    u[1] += pulse #aileron
        
    x_dot = AircraftEOM(t,x, u,w,aircraft_parameters)

    return x_dot#.flatten()

def EOM_wrapper_Rudder_DoublePulsed(t,x,u,w, aircraft_parameters):
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
     
    if t<0.1:
        pulse = .261799
    elif t>0.1 and t<0.2:
        pulse = -.523598
    else:
        pulse = 0

    u[2] += pulse #rudder
        
    x_dot = AircraftEOM(t,x, u,w,aircraft_parameters)

    return x_dot
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Homework 3 Funcitons
#Problem 1.1
def calc_state_vars_for_SLUF(trim_variables, trim_definition):
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
    #v_body += TransformFromInertialToBody(wind_inertial,euler_angles)
    u,v,w = v_body

    p=q=r=0.0
    phi=psi=0.0

    x = np.array([0, 0, -h, phi, theta, psi, u, v, w, p, q, r])
    u = np.array([de,0,0,dt])
    
    return x,u

#Problem 1.2
def calc_cost_val_for_SLUF(trim_variables, trim_definition,wind_inertial, aircraft_parameters):
    '''
    Docstring for Calc_Cost_Val_For_FLUF
    
    :param trim_definition: Description
    :param trim_variables: Description
    :param aircraft_parameters: Description

    return cost J(x_tv|x_tf, ap)
    '''
    Va,gamma,h = trim_definition
    x_ideal, u_ideal = calc_state_vars_for_SLUF(trim_variables, trim_definition)
    time = []
    x_dot_est = AircraftEOM(time,aircraft_state=x_ideal,aircraft_surfaces=u_ideal,wind_inertial=wind_inertial, aircraft_parameters=aircraft_parameters)
    x_dot_est[2] = 0.0   # enforce straight-and-level
    x_dot_ideal = np.array([ x_dot_est[0], x_dot_est[1], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # print(f'The estimated state variables are{x_dot_est}')
    e_trim = x_dot_ideal - x_dot_est
    e_trim_angular = e_trim[3:12]
    cost = np.linalg.norm(e_trim_angular)**2

    #print(f'The cost found for SLUF is {cost}')

    return cost
#Problem 1.3
def calculate_trim(trim_definition,wind_inertial, aircraft_parameters):
    '''
    Docstring for Optimize_Aircraft_States
    
    :param trim_definition: 
    :param aircraft_parameters: 
    :param wind inertial velocity
    
    return optimal aircraft state and control vectors to be in the defined trim condition
    '''

    dt0 = 0.1
    de0 = 0.0
    alpha0 = 0.0

    max_angle = 45*np.pi/180

    bounds = [(-max_angle, max_angle),
               (-max_angle, max_angle),
               (0,1)]

    x0 = np.array([alpha0,de0,dt0])
    #use the cost function to select the optimal aircraft state and control states
    result = minimize(calc_cost_val_for_SLUF, x0=x0,args = (trim_definition,wind_inertial,aircraft_parameters), method='SLSQP', bounds=bounds,options={'ftol':1e-9, 'disp' : True})

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
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#problem 2
def calc_state_vars_for_coordinated_turn(trim_variables, trim_definition):
    '''
    Docstring for CalcTrimForSLUF
    Assuming no wind here, therefore no induced sideslip or flight 
    
    :param trim_definition: definition for SLUF [Va, gamma, h] There is no Radius for a straight line(R = infinity)
    :param trim_variables: Trim Variables for SLUF [alpha, beta, phi, de,da,dr,dt]
    :param wind_inertial: wind velocity vector

    returns aircraft state vector, control surface vector
    '''
    #For SLUF there is no turning, no change in height(climb rate is 0), course angle X is 0, angular velocity is a 0 vector, 
    #with no wind and flying in a straight line, theta = 0 so gamma_a = alpha
    Va,gamma,h,R = trim_definition
    alpha, de, dt, phi, beta, da, dr = trim_variables
    #gamma = theta - alpha but gamma = 0
    theta = gamma + alpha

    wind_vec = np.array([Va,beta,alpha])

    u,v,w = WindAnglesToAirRelativeVelocityVector(wind_vec)
    Chi_dot = (Va/R)*np.cos(gamma)
    #print(f"Chi Dot: {Chi_dot}")
    x = np.array([0, 0, -h, phi, theta,0 , u, v, w, -Chi_dot*np.sin(theta),Chi_dot*np.sin(phi)*np.cos(theta), Chi_dot*np.cos(phi)*np.cos(theta)])

    u = np.array([de,da,dr,dt])
    
    return x,u

def calc_cost_val_for_coord_turn(trim_variables,trim_definition,aircraft_parameters):
    '''
    Docstring for Calc_Cost_Val_For_FLUF
    
    :param trim_definition: Description
    :param trim_variables: Description
    :param aircraft_parameters: Description

    return cost J(x_tv|x_tf, ap)
    '''

    Va,gamma,h,R = trim_definition
    Chi_dot = (Va/R)*np.cos(gamma)
    Y = 0 #The side force, should be related to sideslip angle?
    x_ideal, u_ideal = calc_state_vars_for_coordinated_turn(trim_variables,trim_definition)
    wind = np.zeros(3)
    time =[]
    x_dot_est = AircraftEOM(time,aircraft_state=x_ideal,aircraft_surfaces=u_ideal, aircraft_parameters=aircraft_parameters)
    x_dot_ideal = np.array([ x_dot_est[0], x_dot_est[1], -Va*np.sin(gamma), 0, 0, Chi_dot, 0, 0, 0, 0, 0, 0])
    e_trim = x_dot_ideal - x_dot_est
    e_trim_angular = e_trim[3:12]
    cost = np.linalg.vector_norm(e_trim_angular)**2 + Y**2
    return cost

def calculate_trim_coord_turn(trim_definition, aircraft_parameters):
    '''
    Docstring for Optimize_Aircraft_States
    
    :param trim_definition: Description
    :param aircraft_parameters: Description
    
    return optimal aircraft state and control vectors to be in the defined trim condition
    '''
    dt0 = 0.2
    de0 = 0
    alpha0 = 0
    phi0 = 0
    beta0 = 0
    da0 = 0
    dr0 = 0

    max_angle = 45*np.pi/180

    bounds = [(-max_angle, max_angle),  #alpha
               (-max_angle, max_angle), #de
               (0,1),                   #dt
               (-max_angle, max_angle), #phi
               (-max_angle, max_angle), #beta
               (-max_angle, max_angle), #da
               (-max_angle, max_angle)] #dr
    
    x0 = np.array([alpha0,de0,dt0,phi0,beta0,da0,dr0])
    #use the cost function to select the optimal aircraft state and control states
    result = minimize(calc_cost_val_for_coord_turn, x0=x0,args = (trim_definition,aircraft_parameters), method='SLSQP', bounds=bounds,options={'ftol':1e-9, 'disp' : True})

    if result.success:
        print(f'Optimization successful!')
        print(f'  Alpha:    {result.x[0]:.3f}')
        print(f'  Delta_e:  {result.x[1]:.3f}')
        print(f'  Throttle: {result.x[2]:.3f}')
        print(f'  Roll: {result.x[3]:.3f}')
        print(f'  Sideslip: {result.x[4]:.3f}')
        print(f'  Delta_a: {result.x[5]:.3f}')
        print(f'  Delta_r: {result.x[6]:.3f}')
        print(f'  Final cost: {result.fun:.2e}')
        trim_variables = result.x
        aircraft_state, control_input = calc_state_vars_for_coordinated_turn(trim_variables,trim_definition)
        return aircraft_state,control_input
    else:
        raise ValueError(f"Optimization failed: {result.message}")

def debug_trim_forces(trim_variables, trim_definition, aircraft_parameters):
    """
    Diagnostic function to see what's happening with forces and moments
    """
    Va, gamma, h = trim_definition
    alpha, de, dt = trim_variables
    
    # Get state and controls
    x, u = calc_state_vars_for_SLUF(trim_variables, trim_definition)
    
    print("\n=== TRIM STATE ===")
    print(f"Alpha: {alpha}")
    print(f"Theta: {x[4]}")
    print(f"Delta_e: {de}")
    print(f"Throttle: {dt}")
    print(f"State: {x}")
    print(f"Controls: {u}")
    
    # Get atmosphere
    H = -x[2]
    atm = stdatmo.std_atmo(H)
    rho = atm.rho
    print(f"\n=== ATMOSPHERE ===")
    print(f"Altitude: {H:.1f} m")
    print(f"Density: {rho:.4f} kg/m³")
    
    # Get forces and moments
    wind = np.zeros(3)
    f_total, G_total = AircraftForcesAndMoments(x, u, wind, rho, aircraft_parameters)
    
    # Also get just aero forces (without gravity)
    f_aero, G_aero = AeroForcesAndMoments_BodyState_WindCoeffs(x, u, wind, rho, aircraft_parameters)
    
    print("\n=== FORCES (Body Frame) ===")
    print(f"Total Force: {f_total.flatten()}")
    print(f"Aero Force:  {f_aero.flatten()}")
    print(f"Weight (should be ~{aircraft_parameters.m * aircraft_parameters.g:.1f} N in -Z inertial)")
    
    # Check if forces balance
    # For level flight, need lift ≈ weight
    # In body frame with small alpha, Z-force ≈ -weight
    print("\n=== FORCE BALANCE CHECK ===")
    print(f"Z-force (body): {f_total[2,0]:.2f} N")
    print(f"Expected (weight in body): ~{aircraft_parameters.m * aircraft_parameters.g * np.cos(x[4]):.2f} N")
    print(f"X-force (body): {f_total[0,0]:.2f} N (should be ~0 for steady flight)")
    
    print("\n=== MOMENTS (Body Frame) ===")
    print(f"Moments: {G_total.flatten()}")
    print(f"Pitch moment M: {G_total[1,0]:.4f} N·m (should be ~0 for trim)")
    
    # Get state derivative
    x_dot = AircraftEOM(0, x, u, wind, aircraft_parameters)
    print("\n=== STATE DERIVATIVES ===")
    print(f"x_dot: {x_dot}")
    print(f"Key derivatives that should be ~0:")
    print(f"  u_E_dot: {x_dot[6]:.4f}")
    print(f"  w_E_dot: {x_dot[8]:.4f}")
    print(f"  q_dot:   {x_dot[10]:.4f}")
    
    return

def build_ss_models(trim_definition,wind_inertial, aircraft_parameters):
    #p∗ = q∗ = r∗ = β∗ = ϕ∗ = δ∗_a = δ∗_r = 0.
    wind = wind_inertial
    Va,gamma,h = trim_definition
    atmo = stdatmo.std_atmo(h)
    rho = atmo.rho
    m = aircraft_parameters.m
    S = aircraft_parameters.S
    c = aircraft_parameters.c
    b = aircraft_parameters.b
    #calculate inertial terms
    intertia = inertia_terms(aircraft_parameters)

    trim_variables, cost = calculate_trim(trim_definition=trim_definition,wind_inertial=wind_inertial,aircraft_parameters=aircraft_parameters)
    aircraft_state, control_inputs = calc_state_vars_for_SLUF(trim_variables,trim_definition)
    angular_velocity = aircraft_state[9:12] #p, q, r
    x,y,z,phi,theta,psi,u,v,w,p,q,r = aircraft_state
    delta_e,delta_a,delta_r,delta_t = control_inputs

    inertial_position = aircraft_state[0:3] #x_E, y_E, z_E
    euler_angles = aircraft_state[3:6] #phi, theta, psi
    body_velocity = aircraft_state[6:9] 
    #inertial_velocity = aircraft_state[6:9] 
    angular_velocity = aircraft_state[9:12] #p, q, r
    vel_of_aircraft_in_body = body_velocity #- TransformFromInertialToBody(wind_inertial, euler_angles)
    #vel_of_aircraft_in_body = TransformFromInertialToBody(body_velocity,euler_angles) - TransformFromInertialToBody(wind_inertial, euler_angles)

    #Rotate velocity into body coordinates
    wind_angles = AirRelativeVelocityVectorToWindAngles(vel_of_aircraft_in_body)

    alpha_trim = wind_angles[2]
    Va_trim = wind_angles[0]

    C_L  = aircraft_parameters.CL0 + aircraft_parameters.CLalpha * alpha_trim + aircraft_parameters.CLde * control_inputs[0]
    C_D = aircraft_parameters.CDmin + aircraft_parameters.K * (C_L - aircraft_parameters.CLmin)**2

    CD_dalpha = 2*aircraft_parameters.K*(C_L - aircraft_parameters.CLmin)*aircraft_parameters.CLalpha
    CD_dq = 2*aircraft_parameters.K*(C_L - aircraft_parameters.CLmin)*aircraft_parameters.CLq
    CD_de = 2*aircraft_parameters.K*(C_L - aircraft_parameters.CLmin)*aircraft_parameters.CLde

    Cx = -np.cos(alpha_trim)*C_D + np.sin(alpha_trim)*C_L
    Cz = -np.sin(alpha_trim)*C_D - np.cos(alpha_trim)*C_L 


    Cxalpha = -np.cos(alpha_trim)*(CD_dalpha) + np.sin(alpha_trim)*C_D + np.sin(alpha_trim)*aircraft_parameters.CLalpha + np.cos(alpha_trim)*C_L
    Cxq = -np.cos(alpha_trim)*(CD_dq) + np.sin(alpha_trim)*aircraft_parameters.CLq
    Cxde = -np.cos(alpha_trim)*(CD_de) + np.sin(alpha_trim)*aircraft_parameters.CLde

    Czalpha = -np.sin(alpha_trim)*(CD_dalpha) - np.cos(alpha_trim)*C_D - np.cos(alpha_trim)*aircraft_parameters.CLalpha + np.sin(alpha_trim)*C_L
    Czq = -np.sin(alpha_trim)*(CD_dq) - np.cos(alpha_trim)*aircraft_parameters.CLq
    Czde = -np.sin(alpha_trim)*(CD_de) - np.cos(alpha_trim)*aircraft_parameters.CLde

    #Calculate the Longitudinal State Space Matrices
    Xu = (u*rho*S/m)*Cx - (rho*S*w*Cxalpha/(2*m)) + ((rho*aircraft_parameters.Sprop*aircraft_parameters.Cprop*delta_t)/m)*(((aircraft_parameters.kmotor*u)/Va_trim)*(1-2*delta_t) + 2*u*(delta_t-1))

    Xw = (w*rho*S/m)*(Cx) + rho*S*Cxalpha*u/(2*m) + ((rho*aircraft_parameters.Sprop*aircraft_parameters.Cprop*delta_t/m)) * ((aircraft_parameters.kmotor*w/Va_trim)*(1-delta_t)+2*w*(delta_t-1))
    
    Xq = -w + (rho * Va_trim * S * Cxq * c)/(4*m)
    
    Xde = rho*Va_trim**2 * S * Cxde/(2*m)
    
    Xdt = (rho * aircraft_parameters.Sprop*aircraft_parameters.Cprop/m) * (Va_trim*(aircraft_parameters.kmotor-Va_trim) + 2*control_inputs[3]*((aircraft_parameters.kmotor-Va_trim)**2))

    Zu = (u*rho*S/m) * Cz - (rho*S*Czalpha*w/(2*m))
    
    Zw = (w*rho*S/m)*Cz + (rho*S*Czalpha*u/(2*m))
    
    Zq = u + (rho*Va_trim*S*Czq*c/(4*m))
    
    Zde = rho*(Va_trim**2) * S * Czde / (2*m)

    Mu = ((u*rho*S*c)/aircraft_parameters.Iy)*(aircraft_parameters.Cm0 + aircraft_parameters.Cmalpha*alpha_trim + aircraft_parameters.Cmde*control_inputs[0]) - (rho*S*c*aircraft_parameters.Cmalpha*w/(2*aircraft_parameters.Iy))


    Mw = (w*rho*S*c/aircraft_parameters.Iy)*(aircraft_parameters.Cm0 + aircraft_parameters.Cmalpha*alpha_trim + aircraft_parameters.Cmde*control_inputs[0]) + (rho*S*c*aircraft_parameters.Cmalpha*u)/(2*aircraft_parameters.Iy)
    
    Mq = (rho*Va_trim * S *c**2 * aircraft_parameters.Cmq)/(4*aircraft_parameters.Iy)
    Mde = (rho*Va_trim**2*S*c*aircraft_parameters.Cmde)/(2*aircraft_parameters.Iy)

    A_lon = np.array([[Xu,              Va_trim*Xw,       Xq,-aircraft_parameters.g*np.cos(theta), 0],
                  [Zu/Va_trim,                  Zw,       Zq/Va_trim,-aircraft_parameters.g*np.sin(theta)/Va_trim, 0],
                  [Mu,                  Mw*Va_trim,       Mq,               0,                     0],
                  [0,                   0,        1,                0,                     0],
                  [np.sin(theta), -Va_trim*np.cos(theta), 0, u*np.cos(theta)+w*np.sin(theta),      0]
                  ])
    
    B_lon = np.array([[Xde/Va_trim, Xdt/Va_trim],
                      [Zde/Va_trim,  0],
                      [Mde,  0],
                      [0,    0],
                      [0,    0]])
    
    #==================================================================================================================================================
    #Lateral State Space Equaations


    Yv = (rho*S*v/(m))*(aircraft_parameters.CY0) + (rho*S*aircraft_parameters.CYbeta/(2*m))*np.sqrt(u**2 + w**2)
    
    Yp = w + (rho*Va_trim*S*b/(4*m))*aircraft_parameters.CYp
    Yr = -u + (rho*Va_trim*S*b/(4*m))*aircraft_parameters.CYr

    Yda = (rho*Va_trim**2 * S/(2*m)) * aircraft_parameters.CYda
    Ydr = (rho*Va_trim**2 * S/(2*m))*aircraft_parameters.CYdr
    Lv = (rho*S*b*v)*(aircraft_parameters.Cl0) + (rho*S*aircraft_parameters.Clbeta/2)*np.sqrt(u**2 + w**2)
    Lp = (rho*Va_trim*S*b**2/4)*aircraft_parameters.Clp
    Lr = (rho*Va_trim*S*b**2)/4 * aircraft_parameters.Clr
    Lda = (rho*(Va_trim**2)*S*b/2)*aircraft_parameters.Clda
    Ldr = ((rho*(Va_trim**2)*S*b)/2) * aircraft_parameters.Cldr
    Nv = rho*S*b*v*(aircraft_parameters.Cn0) + (rho*S*b*aircraft_parameters.Cnbeta/2)*np.sqrt(u**2 + w**2)
    Np = ((rho*Va_trim*S*b**2)/4)*aircraft_parameters.Cnp
    Nr = ((rho*Va_trim*S*b**2)/4)*aircraft_parameters.Cnr
    Nda = ((rho*(Va_trim**2)*S*b)/2)*aircraft_parameters.Cnda
    Ndr = ((rho*(Va_trim**2)*S*b)/2)*aircraft_parameters.Cndr

    A_lat = np.array([[Yv, Yp/Va_trim,              Yr/Va_trim, aircraft_parameters.g*np.cos(theta)*np.cos(phi)/Va_trim, 0],
                      [Lv*Va_trim, Lp,              Lr,                      0,                          0],
                      [Nv*Va_trim, Np,              Nr,                      0,                          0],
                      [0,   1,   np.cos(phi)*np.tan(theta),          0,                          0],
                      [0,   0,  np.cos(phi)*(1/np.cos(theta)),       0,                          0]
                      ])
    B_lat = np.array([ 
        [Yda/Va_trim, Ydr/Va_trim],
        [Lda, Ldr],
        [Nda, Ndr],
        [0,0],
        [0,0]
    ])

    #Calculate the Lateral State Space Matrices
    print(f'The Controll Surface Inputs are: {control_inputs}')
    print(f'The aircraft States are: {aircraft_state}')
    troll = -1/aircraft_parameters.Clp
    print(f'roll period is: {troll}')
    tspi = -aircraft_parameters.Clbeta/(aircraft_parameters.Cnr*aircraft_parameters.Clbeta - aircraft_parameters.Cnbeta*aircraft_parameters.Clr)
    print(f'T spiral is:{tspi}')
    return A_lon, B_lon, A_lat, B_lat

def test_moments_at_prof_trim():
    """Verify moments using professor's trim values"""
    # Professor's trim
    prof_trim = np.array([0.0242, 0.1982, 0.2059, 0.0814, 0.0010, 0.0161, -0.0903])

    h = 200
    Va = 20.2
    gamma = 0
    R = 500
    
    trim_def = [Va, gamma, h, R]  # your values
    wind = np.zeros(3)
    
    x, u = calc_state_vars_for_coordinated_turn(prof_trim, trim_def, wind)
    
    # Calculate moments
    H = -x[2]
    atm = stdatmo.std_atmo(H)
    f, G = AircraftForcesAndMoments(x, u, wind, atm.rho, aircraft_parameters)
    
    print("Moments at professor's trim:")
    print(f"L = {G[0,0]:.6f}")
    print(f"M = {G[1,0]:.6f}")
    print(f"N = {G[2,0]:.6f}")
    
    # These should be nearly zero for trim
    x_dot = AircraftEOM(0, x, u, wind, aircraft_parameters)
    print("\nAngular accelerations (should be ~0):")
    print(f"p_dot = {x_dot[9]:.6e}")
    print(f"q_dot = {x_dot[10]:.6e}")
    print(f"r_dot = {x_dot[11]:.6e}")

    # ADD THIS:
    print("State velocities (body frame):")
    print(f"u = {x[6]:.4f}, v = {x[7]:.4f}, w = {x[8]:.4f}")
    
    # Reconstruct wind angles
    euler = x[3:6]
    v_air_rel = x[6:9] - TransformFromInertialToBody(wind, euler)
    Va_check, beta_check, alpha_check = AirRelativeVelocityVectorToWindAngles(v_air_rel)
    
    print(f"Reconstructed: Va={Va_check:.4f}, alpha={alpha_check:.4f}, beta={beta_check:.4f}")
    print(f"Expected: Va={trim_def[0]:.4f}, alpha={prof_trim[0]:.4f}, beta={prof_trim[4]:.4f}")

    # Print inertia values
    print("Inertia values:")
    print(f"Ix = {aircraft_parameters.Ix:.6f}")
    print(f"Iy = {aircraft_parameters.Iy:.6f}")
    print(f"Iz = {aircraft_parameters.Iz:.6f}")
    print(f"Ixz = {aircraft_parameters.Ixz:.6f}")
    
    # Get angular rates and moments
    p, q, r = x[9:12]
    print(f"\nAngular rates: p={p:.6f}, q={q:.6f}, r={r:.6f}")
    
    # Calculate wind angles for coefficient calculation
    euler = x[3:6]
    v_air_rel = x[6:9] - TransformFromInertialToBody(wind, euler)
    Va, beta, alpha = AirRelativeVelocityVectorToWindAngles(v_air_rel)
    
    # Calculate q_hat
    q_hat = (q * aircraft_parameters.c) / (2 * Va)
    print(f"\nq_hat = {q_hat:.6f}")
    
    # Calculate C_m breakdown
    print(f"\nC_m coefficient breakdown:")
    print(f"  Cm0 = {aircraft_parameters.Cm0:.6f}")
    print(f"  Cmalpha = {aircraft_parameters.Cmalpha:.6f}, alpha = {alpha:.6f}")
    print(f"  Cmalpha*alpha = {aircraft_parameters.Cmalpha * alpha:.6f}")
    print(f"  Cmq = {aircraft_parameters.Cmq:.6f}, q_hat = {q_hat:.6f}")
    print(f"  Cmq*q_hat = {aircraft_parameters.Cmq * q_hat:.6f}")
    print(f"  Cmde = {aircraft_parameters.Cmde:.6f}, de = {u[0]:.6f}")
    print(f"  Cmde*de = {aircraft_parameters.Cmde * u[0]:.6f}")
    
    C_m = (aircraft_parameters.Cm0 + 
           aircraft_parameters.Cmalpha * alpha + 
           aircraft_parameters.Cmq * q_hat + 
           aircraft_parameters.Cmde * u[0])
    print(f"  Total C_m = {C_m:.6f}")
    
    # Calculate dimensional moment
    Q = 0.5 * atm.rho * Va**2
    P = Q * aircraft_parameters.S
    M = P * aircraft_parameters.c * C_m
    
    print(f"\nDimensional pitching moment:")
    print(f"  Q = {Q:.6f}")
    print(f"  P = Q*S = {P:.6f}")
    print(f"  M = P*c*C_m = {M:.6f}")
    
    # Calculate q_dot
    term1 = aircraft_parameters.gamma[4] * p * r
    term2 = aircraft_parameters.gamma[5] * (p**2 - r**2)
    term3 = M / aircraft_parameters.Iy
    
    print(f"\nq_dot breakdown:")
    print(f"  gamma[4]*p*r = {term1:.6e}")
    print(f"  -gamma[5]*(p²-r²) = {-term2:.6e}")
    print(f"  M/Iy = {term3:.6e}")
    print(f"  Total q_dot = {term1 - term2 + term3:.6e}")
    
def verify_final_trim():
    your_trim = np.array([0.023, 0.231, 0.211, 0.083, 0.001, 0.016, -0.092])
    prof_trim = np.array([0.0242, 0.1982, 0.2059, 0.0814, 0.0010, 0.0161, -0.0903])
    
    trim_def = [20.2, 0, 1000, 300]  # Use your actual values
    wind = np.zeros(3)

    Va, gamma, h, R = trim_def
    Chi_dot = (Va/R) * np.cos(gamma)
    
    print("=" * 60)
    print("YOUR TRIM:")
    x_yours, u_yours = calc_state_vars_for_coordinated_turn(your_trim, trim_def, wind)
    H = -x_yours[2]
    atm = stdatmo.std_atmo(H)
    
    x_dot_yours = AircraftEOM(0, x_yours, u_yours, wind, aircraft_parameters)
    print(f"Angular accelerations: p_dot={x_dot_yours[9]:.6e}, q_dot={x_dot_yours[10]:.6e}, r_dot={x_dot_yours[11]:.6e}")
    x_dot_ideal = np.array([x_dot_yours[0], x_dot_yours[1], -Va*np.sin(gamma), 
                        0, 0, Chi_dot, 
                        0, 0, 0, 
                        0, 0, 0])
    
    f, G = AircraftForcesAndMoments(x_yours, u_yours, wind, atm.rho, aircraft_parameters)
    print(f"Moments: L={G[0,0]:.6e}, M={G[1,0]:.6e}, N={G[2,0]:.6e}")
    
    print("\n" + "=" * 60)
    print("PROFESSOR'S TRIM:")
    x_prof, u_prof = calc_state_vars_for_coordinated_turn(prof_trim, trim_def, wind)
    x_dot_prof = AircraftEOM(0, x_prof, u_prof, wind, aircraft_parameters)
    print(f"Angular accelerations: p_dot={x_dot_prof[9]:.6e}, q_dot={x_dot_prof[10]:.6e}, r_dot={x_dot_prof[11]:.6e}")
    
    f_prof, G_prof = AircraftForcesAndMoments(x_prof, u_prof, wind, atm.rho, aircraft_parameters)
    print(f"Moments: L={G_prof[0,0]:.6e}, M={G_prof[1,0]:.6e}, N={G_prof[2,0]:.6e}")

    # ADD THESE PRINT STATEMENTS:
    print("\nIdeal state derivatives:")
    print(f"  x_dot_ideal = {x_dot_ideal}")
    print("\nEstimated state derivatives:")
    print(f"  x_dot_estimated = {x_dot_yours}")
    print("\nErrors (ideal - estimated):")
    error = x_dot_ideal - x_dot_yours
    print(f"  error = {error}")
    print(f"\nError in angular accelerations specifically:")
    print(f"  p_dot error: {error[9]:.6e}")
    print(f"  q_dot error: {error[10]:.6e}")
    print(f"  r_dot error: {error[11]:.6e}")
    
    print(f"\nAngular accelerations: p_dot={x_dot_yours[9]:.6e}, q_dot={x_dot_yours[10]:.6e}, r_dot={x_dot_yours[11]:.6e}")
    
    f, G = AircraftForcesAndMoments(x_yours, u_yours, wind, atm.rho, aircraft_parameters)
    print(f"Moments: L={G[0,0]:.6e}, M={G[1,0]:.6e}, N={G[2,0]:.6e}")
    
    print("\n" + "=" * 60)
    print("PROFESSOR'S TRIM:")
#-----------------------------------------------------------------
#Homework 6 code
#-----------------------------------------------------------------
def FlightPathAnglesFromState(aircraft_state):
    wind_angles = AirRelativeVelocityVectorToWindAngles(aircraft_state[6:9])
    Vg = wind_angles[0]# - wind if there were any
    gamma_a = aircraft_state[4] - wind_angles[2] # Theta - Alpha
    gamma = gamma_a #Because again no wind
    chi = aircraft_state[5] #Because no sideslip angle/wind

    return np.array([Vg,chi,gamma])

#-----------------------------------------------------------------
#Homework 8 code
#-----------------------------------------------------------------

def nl_guidance_model(t,state, wind_inertial, control_objective,parameter_struct):
    #State = [pn,pe,chi,chi_dot, h, h_dot, Va]

    #Control_objectives = [h_c, h_dot_c, chi_c, chi_dot_c, Va_c]
    bcd = parameter_struct.bcd
    bc = parameter_struct.bc
    bhd = parameter_struct.bhd
    bh = parameter_struct.bh
    bVa = parameter_struct.bVa
    temp = np.array([-np.sin(state[2]),np.cos(state[2])])
    inside = (1/state[-1]) * wind_inertial[0:2].T @ temp

    yaw = state[2] - np.asin(inside)

    pdot = state[-1]*np.array([np.cos(yaw),np.sin(yaw)]) + wind_inertial[0:2]

    chi_dot = state[3]

    chi_dot_dot = bcd*(control_objective[3] - chi_dot) + bc*(control_objective[2] - state[2])

    h_dot = state[5]

    h_dot_dot = bhd*(control_objective[1] - h_dot) + bh*(control_objective[0] - state[4])

    Va_dot = bVa* (control_objective[4] - state[-1])

    x_dot = np.concatenate([pdot, [chi_dot, chi_dot_dot, h_dot, h_dot_dot, Va_dot]])

    return x_dot

def straight_line_guidance(pos_line, dir_line,pos_ac, kpath,chi_inf,Va_c):
    #Course angle Command
    chi_q = np.atan2(dir_line[1],dir_line[0])
    e_ac = pos_ac - pos_line
    epy = -np.sin(chi_q)*e_ac[0] + np.cos(chi_q)*e_ac[1]

    chi_c = chi_q - chi_inf*(2/np.pi)*np.atan(kpath*epy)
    chi_dot_c = 0

    #Height Command
    ki = np.array([0,0,1])
    qki = np.linalg.cross(dir_line,ki)

    n_vec = qki/np.linalg.norm(qki)
    s_vec = e_ac - (e_ac.T*n_vec)*n_vec

    h_c = -pos_line[2] + np.linalg.norm(s_vec[0:2]*dir_line[2])/np.linalg.norm(dir_line[0:2])

    q_line = dir_line/np.linalg.norm(dir_line) # turn into a unit vector
    h_dot_c = -q_line[2]*Va_c #*0 #to eliminate steady state error according to Frew

    return np.array([h_c,h_dot_c,chi_c,chi_dot_c,Va_c])

def first_order_straight_guidance(t,pos,pos_line,dir_line,kpath,chi_inf,Vtrim):
     control_objectives = straight_line_guidance(pos_line, dir_line,pos,kpath,chi_inf,Vtrim)

     kh = 0.1
     h_c = control_objectives[0]
     h_dot_c = control_objectives[1]
     z_vel = -(h_dot_c + kh*(h_c+pos[2]))

     chi_c = control_objectives[2]

     if (z_vel**2 > control_objectives[4]**2):
          print(f"Climb Rate too high")
          z_vel = -h_dot_c
     aircraft_speed = np.sqrt(control_objectives[4]**2 - z_vel**2)
     vel = np.array([aircraft_speed*np.cos(chi_c), aircraft_speed*np.sin(chi_c), z_vel])
     return vel

#-----------------------------------------------------------------
#Homework 9 code
#-----------------------------------------------------------------
def calculate_wind_inertial(inertial_velocities, euler_angles, angular_rates, wind_angles):
    #Each input will contain n rows
    #Return inertial wind vector
    wind_inertial = []
    for n,angles in enumerate(wind_angles):
        arv = WindAnglesToAirRelativeVelocityVector(angles)
        air_rel_v_body = TransformFromBodyToInertial(arv,euler_angles[n,:])
        wind_inertial.append(inertial_velocities[n,:] - air_rel_v_body)

    wind_inertial = np.array(wind_inertial)
    return wind_inertial



if __name__ == '__main__':

    aircraft_parameters = AircraftParameters()

   
    #Problem 4
    h = 1800
    Va = 18
    gamma = 0
    gamma = np.deg2rad(gamma)
    # atmo = stdatmo.std_atmo(H=h)
    # density = atmo.rho

    w = np.array([0.0,0.0,0.0])#ground wind 
    trim_def = np.array([Va,gamma,h])
    A_lon,B_lon,A_lat,B_lat = build_ss_models(trim_def,w, aircraft_parameters)

    print(f'The longitudinal A matrix: {A_lon}')
    print(f'The longitudinal B Matrix: {B_lon}')
    print(f'The latitudinal A matrix: {A_lat}')
    print(f'The latitudinal B matrix: {B_lat}')
    evals_lon,e_vecs_lon = np.linalg.eig(A_lon)
    evals_lat,e_vecs_lat = np.linalg.eig(A_lat)

    print(f'Longitudinal eigen values are: {evals_lon}')
    print(f'Longitudinal eigen vectors are: {e_vecs_lon}')

    print(f'Lateral eigen values are: {evals_lat}')
    print(f'Lateral eigen vectors are: {e_vecs_lat}')

    eval1,eval2,eval3,eval4,eval5 = evals_lon

    eval1_lat,eval2_lat,eval3_lat,eval4_lat,eval5_lat = evals_lat

    print(eval2.imag)

    wn_short_period = np.sqrt(eval2.real**2 + eval2.imag**2)
    print(f'The Short Period mode natural Freq is: {wn_short_period}')

    damp_short_period = - eval2.real/wn_short_period
    print(f'The Short Period mode damping is: {damp_short_period}')

    wn_phugoid = np.sqrt(eval4.real**2 + eval4.imag**2)
    print(f'The Phugoid mode natural Freq is: {wn_phugoid}')

    damp_phugoid = - eval4.real/wn_phugoid
    print(f'The Phugoid mode damping is: {damp_phugoid}')

    wn_dutch = np.sqrt(eval4_lat.real**2 + eval4_lat.imag**2)
    print(f'The Dutch mode natural Freq is: {wn_dutch}')

    damp_dutch = - eval4_lat.real/wn_dutch
    print(f'The Dutch roll mode damping is: {damp_dutch}')

    time_period_roll =-1 / eval2_lat.real
    print(f'The period of the roll using eigenvalue is: {time_period_roll}')

    #Can use L_p from the State Space calc actually
    time_period_spiral = -1/eval3_lat.real
    print(f'The time period of the sprial mode using eigenval is: {time_period_spiral}')

    #Homework 5 

    #Phugoid Initial conditions: 
    evec_lon1, evec_lon2, evec_lon3, evec_lon4, evec_long5 = e_vecs_lon
    evec_lat1, evec_lat2, evec_lat3, evec_lat4, evec_lat5 = e_vecs_lat

    phugoid_evec = evec_lon4
    print(-46.667755753*phugoid_evec)
    k1 = -46.667755753 # Scaling factor for Problem 1.1
    #determine Perturbations
    x0_1 = np.real(k1*phugoid_evec)
    print(f'The scaled initial conditions based on the Phugoid Mode eigenvector are: {x0_1}')
    aircraft_state, control_inputs = calculate_trim(trim_definition=trim_def,wind_inertial=w, aircraft_parameters=aircraft_parameters)

    print(f'The aircraft states based on the trim conditions are:{aircraft_state}')
    print(f'The contrtol inputs based on the trim conditions are: {control_inputs}')

    t_span = (0,250)
    t_eval = np.linspace(0,250,1000)
    dt = 0.1
    linear_time = np.arange(0,250,dt)
    full_time = int(250/dt)
    x_not = np.array([aircraft_state[6],aircraft_state[8],aircraft_state[10],aircraft_state[4],aircraft_state[2]])
    # #apply Perturbations
    aircraft_state[2] += x0_1[4] #h
    aircraft_state[4] += x0_1[3] #Theta
    aircraft_state[10] += x0_1[2] #q
    aircraft_state[8] += x0_1[1] #w
    aircraft_state[6] += x0_1[0] #u
    linear_state = np.zeros((5,len(linear_time)))
    linear_state[:,0] = x_not + x0_1 #Perturbed initial linear state

    print(f'The shape of the B matrix is:{B_lon.shape}')
    print(f'The shape of the control inputs matrix is: {control_inputs.shape}')
    #The linear phugoid approximation is x_dot = Ax + Bu
    control_input_array = np.tile(control_inputs.reshape(4,1),(1,len(linear_time)))
    states = np.zeros((12,len(linear_time)))
    states[:,0] = aircraft_state

    linearized_u_longitudinal = np.zeros(2)
    x_trim = aircraft_state.copy()
    print(states.shape)
    #states[:,0] = aircraft_state
    #Linearized Approximation of motion as it propagates
    sigma = np.real(eval4.real)
    for k in range(1,full_time):
        x_dot = sigma*linear_state[:,k-1]
        #x_dot = A_lon@linear_state[:,k-1] + B_lon@linearized_u_longitudinal
        linear_state[:,k] = linear_state[:,k-1] + x_dot*dt

    # #Full State Reconstruction
    for k in range(1,full_time):
        states[:,k] = x_trim.copy()
        states[6,k]  += linear_state[0,k]  # u
        states[8,k]  += linear_state[1,k]  # w
        states[10,k] += linear_state[2,k]  # q
        states[4,k]  += linear_state[3,k]  # theta
        states[2,k]  += linear_state[4,k]  # h


    PlotSimulation(time=linear_time,aircraft_state_array=states,control_input_array=control_input_array,col = 'b-')
    #This is the nonlinear phugoid calculation
    x0 = aircraft_state.flatten()   # initial state must be 1D
    sol = solve_ivp(
        EOM_wrapper,
        t_span,
        x0,
        t_eval=t_eval,
        method='RK45',
        args=(control_inputs,w,aircraft_parameters)
    )
    time = sol.t
    aircraft_state_array = sol.y
    control_input_array = np.tile(control_inputs.reshape(4,1),(1,len(time)))
    PlotSimulation(time=time,aircraft_state_array=aircraft_state_array,control_input_array=control_input_array,col = 'b-')

    # #Homework 5 Problem 1.2
    # #Plotting the Short Period Mode
    # print(60*"=")
    # print("Homework Prolem 1.2: Short Period Mode Modeling")
    # # sp_evec = evec_lon2
    # # print(6436.93878*sp_evec)
    # # k2 = -6436.93878 # Scaling factor for Problem 1.1
    # sp_evec = evec_lon2

    # desired_theta = 2 * np.pi/180  # 5 deg/s
    # alpha2 = desired_theta / np.real(sp_evec[3])
    # print(-112.3457751*sp_evec)
    # k2 = -112.3457751 # Scaling factor for Problem 1.1
    # #determine Perturbations
    # x0_2 = np.real(alpha2*sp_evec)
    # t_span = (0,3)
    # t_eval = np.linspace(0,3,1000)

    # linear_time = np.arange(0,3,0.1)
    # full_time = int(3/0.1)

    # print(f'The scaled initial conditions based on the Phugoid Mode eigenvector are: {x0_2}')
    # aircraft_state, control_inputs = calculate_trim(trim_definition=trim_def,wind_inertial=w, aircraft_parameters=aircraft_parameters)
    # #apply Perturbations
    # aircraft_state[2] += x0_2[4] #h
    # aircraft_state[4] += x0_2[3] #Theta
    # aircraft_state[10] += x0_2[2] #q
    # aircraft_state[8] += x0_2[1] #w
    # aircraft_state[6] += x0_2[0] #u
    # linear_state = np.zeros((5,len(linear_time)))
    # linear_state[:,0] = x_not + x0_2 #Perturbed initial linear state

    # print(f'The shape of the B matrix is:{B_lon.shape}')
    # print(f'The shape of the control inputs matrix is: {control_inputs.shape}')
    # #The linear phugoid approximation is x_dot = Ax + Bu
    # control_input_array = np.tile(control_inputs.reshape(4,1),(1,len(linear_time)))
    # states = np.zeros((12,len(linear_time)))
    # states[:,0] = aircraft_state

    # linearized_u_longitudinal = np.zeros(2)

    # print(states.shape)
    # # for k in range(1,full_time):
    # #     x_dot = A_lon@linear_state[:,k-1] 
    # sigma = np.real(eval2.real)
    # for k in range(1,full_time):
    #     x_dot = sigma*linear_state[:,k-1]
    #     #x_dot = A_lon@linear_state[:,k-1] + B_lon@linearized_u_longitudinal
    #     linear_state[:,k] = linear_state[:,k-1] + x_dot*dt
    #     linear_state[:,k] = linear_state[:,k-1] + x_dot*dt
    
    # #Full State Reconstruction
    # for k in range(full_time):
    #     states[:,k] = x_trim.copy()
    #     states[6,k]  += linear_state[0,k]  # u
    #     states[8,k]  += linear_state[1,k]  # w
    #     states[10,k] += linear_state[2,k]  # q
    #     states[4,k]  += linear_state[3,k]  # theta
    #     states[2,k]  += linear_state[4,k]  # h

    # PlotSimulation(time=linear_time,aircraft_state_array=states,control_input_array=control_input_array,col = 'b-')
    # #This is the nonlinear phugoid calculation
    # x0 = aircraft_state.flatten()   # initial state must be 1D
    # sol = solve_ivp(
    #     EOM_wrapper,
    #     t_span,
    #     x0,
    #     t_eval=t_eval,
    #     method='RK45',
    #     args=(control_inputs,w,aircraft_parameters)
    # )
    # time = sol.t
    # aircraft_state_array = sol.y
    # control_input_array = np.tile(control_inputs.reshape(4,1),(1,len(time)))
    # PlotSimulation(time=time,aircraft_state_array=aircraft_state_array,control_input_array=control_input_array,col = 'b-')

    # #Homework 5 Problem 1.3
    # #Plotting the Short Period Mode

    # # sp_evec = evec_lon2
    # # print(6436.93878*sp_evec)
    # # k3 = -6436.93878 # Scaling factor for Problem 1.1
    # sp_evec = evec_lon2

    # desired_theta = 2 * np.pi/180  # 5 deg/s
    # alpha3 = desired_theta / np.real(sp_evec[3])
    # #determine Perturbations
    # x0_3 = np.real(alpha3*sp_evec)
    # t_span = (0,25)
    # t_eval = np.linspace(0,25,1000)

    # linear_time = np.arange(0,25,0.1)
    # full_time = int(25/0.1)

    # print(f'The scaled initial conditions based on the Phugoid Mode eigenvector are: {x0_3}')
    # aircraft_state, control_inputs = calculate_trim(trim_definition=trim_def,wind_inertial=w, aircraft_parameters=aircraft_parameters)
    # #apply Perturbations
    # aircraft_state[2] += x0_3[4] #h
    # aircraft_state[4] += x0_3[3] #Theta
    # aircraft_state[10] += x0_3[2] #q
    # aircraft_state[8] += x0_3[1] #w
    # aircraft_state[6] += x0_3[0] #u
    # linear_state = np.zeros((5,len(linear_time)))
    # linear_state[:,0] = x_not + x0_3 #Perturbed initial linear state

    # print(f'The shape of the B matrix is:{B_lon.shape}')
    # print(f'The shape of the control inputs matrix is: {control_inputs.shape}')
    # #The linear phugoid approximation is x_dot = Ax + Bu
    # control_input_array = np.tile(control_inputs.reshape(4,1),(1,len(linear_time)))
    # states = np.zeros((12,len(linear_time)))
    # states[:,0] = aircraft_state

    # # for k in range(1,full_time):
    # #     x_dot = A_lon@linear_state[:,k-1]
    # sigma = np.real(eval2.real)
    # for k in range(1,full_time):
    #     x_dot = sigma*linear_state[:,k-1]
    #     #x_dot = A_lon@linear_state[:,k-1] + B_lon@linearized_u_longitudinal
    #     linear_state[:,k] = linear_state[:,k-1] + x_dot*dt
    #     linear_state[:,k] = linear_state[:,k-1] + x_dot*dt
    
    # #Full State Reconstruction
    # for k in range(full_time):
    #     states[:,k] = x_trim.copy()
    #     states[6,k]  += linear_state[0,k]  # u
    #     states[8,k]  += linear_state[1,k]  # w
    #     states[10,k] += linear_state[2,k]  # q
    #     states[4,k]  += linear_state[3,k]  # theta
    #     states[2,k]  += linear_state[4,k]  # h

    # PlotSimulation(time=linear_time,aircraft_state_array=states,control_input_array=control_input_array,col = 'b-')
    # #This is the nonlinear phugoid calculation
    # x0 = aircraft_state.flatten()   # initial state must be 1D
    # sol = solve_ivp(
    #     EOM_wrapper,
    #     t_span,
    #     x0,
    #     t_eval=t_eval,
    #     method='RK45',
    #     args=(control_inputs,w,aircraft_parameters)
    # )
    # time = sol.t
    # aircraft_state_array = sol.y
    # control_input_array = np.tile(control_inputs.reshape(4,1),(1,len(time)))
    # PlotSimulation(time=time,aircraft_state_array=aircraft_state_array,control_input_array=control_input_array,col = 'b-')
    #Homework 5 Problem 1.4
    #Phugoid Initial conditions: 
    # evec_lon1, evec_lon2, evec_lon3, evec_lon4, evec_long5 = e_vecs_lon
    # evec_lat1, evec_lat2, evec_lat3, evec_lat4, evec_lat5 = e_vecs_lat

    phugoid_evec = evec_lon4

    desired_theta = 25 * np.pi/180  # 5 deg/s
    alpha4 = desired_theta / np.real(phugoid_evec[3])
    #determine Perturbations
    x0_4 = np.real(alpha4*phugoid_evec)
    print(f'The scaled initial conditions based on the Phugoid Mode eigenvector are: {x0_4}')
    aircraft_state, control_inputs = calculate_trim(trim_definition=trim_def,wind_inertial=w, aircraft_parameters=aircraft_parameters)

    print(f'The aircraft states based on the trim conditions are:{aircraft_state}')
    print(f'The contrtol inputs based on the trim conditions are: {control_inputs}')

    t_span = (0,250)
    t_eval = np.linspace(0,250,1000)

    linear_time = np.arange(0,250,0.1)
    full_time = int(250/0.1)
    # x_not = np.array([aircraft_state[6],aircraft_state[8],aircraft_state[10],aircraft_state[4],aircraft_state[2]])
    # #apply Perturbations
    aircraft_state[2] += x0_4[4] #h
    aircraft_state[4] += x0_4[3] #Theta
    aircraft_state[10] += x0_4[2] #q
    aircraft_state[8] += x0_4[1] #w
    aircraft_state[6] += x0_4[0] #u
    linear_state = np.zeros((5,len(linear_time)))
    linear_state[:,0] = x_not + x0_4 #Perturbed initial linear state

    # print(f'The shape of the B matrix is:{B_lon.shape}')
    # print(f'The shape of the control inputs matrix is: {control_inputs.shape}')
    # #The linear phugoid approximation is x_dot = Ax + Bu
    control_input_array = np.tile(control_inputs.reshape(4,1),(1,len(linear_time)))
    # linear_state = np.zeros((5,len(linear_time)))
    states = np.zeros((12,len(linear_time)))
    states[:,0] = aircraft_state

    print(states.shape)
    # for k in range(1,full_time):
    #     x_dot = A_lon@linear_state[:,k-1]
    sigma = np.real(eval4.real)
    for k in range(1,full_time):
        x_dot = sigma*linear_state[:,k-1]
        #x_dot = A_lon@linear_state[:,k-1] + B_lon@linearized_u_longitudinal
        linear_state[:,k] = linear_state[:,k-1] + x_dot*dt
        linear_state[:,k] = linear_state[:,k-1] + x_dot*dt
    
    #Full State Reconstruction
    for k in range(1,full_time):
        states[:,k] = x_trim.copy()
        states[6,k]  += linear_state[0,k]  # u
        states[8,k]  += linear_state[1,k]  # w
        states[10,k] += linear_state[2,k]  # q
        states[4,k]  += linear_state[3,k]  # theta
        states[2,k]  += linear_state[4,k]  # h

    PlotSimulation(time=linear_time,aircraft_state_array=states,control_input_array=control_input_array,col = 'b-')
    #This is the nonlinear phugoid calculation
    x0 = aircraft_state.flatten()   # initial state must be 1D
    sol = solve_ivp(
        EOM_wrapper,
        t_span,
        x0,
        t_eval=t_eval,
        method='RK45',
        args=(control_inputs,w,aircraft_parameters)
    )
    time = sol.t
    aircraft_state_array = sol.y
    control_input_array = np.tile(control_inputs.reshape(4,1),(1,len(time)))
    PlotSimulation(time=time,aircraft_state_array=aircraft_state_array,control_input_array=control_input_array,col = 'b-')

    #========================================================================================================================================================
    #Homework5 Problem 2.1

    # phugoid_evec = evec_lon4
    # # print(2675*phugoid_evec)
    # # k1 = -2675 # Scaling factor for Problem 1.1
    # # #determine Perturbations
    # x0_1 = np.real(phugoid_evec)
    # print(f'The scaled initial conditions based on the Phugoid Mode eigenvector are: {x0_1}')
    aircraft_state, control_inputs = calculate_trim(trim_definition=trim_def,wind_inertial=w, aircraft_parameters=aircraft_parameters)

    print(f'The aircraft states based on the trim conditions are:{aircraft_state}')
    print(f'The contrtol inputs based on the trim conditions are: {control_inputs}')

    t_span = (0,250)
    t_eval = np.linspace(0,250,1000)
    #This is the nonlinear phugoid calculation
    x0 = aircraft_state.flatten()   # initial state must be 1D
    sol = solve_ivp(
        EOM_wrapper_Pulsed,
        t_span,
        x0,
        t_eval=t_eval,
        method='RK45',
        args=(control_inputs,w,aircraft_parameters)
    )
    time = sol.t
    aircraft_state_array = sol.y
    control_input_array = np.tile(control_inputs.reshape(4,1),(1,len(time)))
    PlotSimulation(time=time,aircraft_state_array=aircraft_state_array,control_input_array=control_input_array,col = 'b-')

    #Homework5 Problem 2.2
    #Doublet Pulse to the Aileron
    aircraft_state, control_inputs = calculate_trim(trim_definition=trim_def,wind_inertial=w, aircraft_parameters=aircraft_parameters)

    print(f'The aircraft states based on the trim conditions are:{aircraft_state}')
    print(f'The contrtol inputs based on the trim conditions are: {control_inputs}')

    t_span = (0,250)
    t_eval = np.linspace(0,250,1000)

    #This is the nonlinear phugoid calculation
    x0 = aircraft_state.flatten()   # initial state must be 1D
    sol = solve_ivp(
        EOM_wrapper_Aieleron_DoublePulsed,
        t_span,
        x0,
        t_eval=t_eval,
        method='RK45',
        args=(control_inputs,w,aircraft_parameters)
    )
    time = sol.t
    aircraft_state_array = sol.y
    control_input_array = np.tile(control_inputs.reshape(4,1),(1,len(time)))
    PlotSimulation(time=time,aircraft_state_array=aircraft_state_array,control_input_array=control_input_array,col = 'b-')

    #Homework5 Problem 2.3
    #Doublet Pulse to the Rudder
    aircraft_state, control_inputs = calculate_trim(trim_definition=trim_def,wind_inertial=w, aircraft_parameters=aircraft_parameters)

    print(f'The aircraft states based on the trim conditions are:{aircraft_state}')
    print(f'The contrtol inputs based on the trim conditions are: {control_inputs}')

    t_span = (0,250)
    t_eval = np.linspace(0,250,1000)

    #This is the nonlinear phugoid calculation
    x0 = aircraft_state.flatten()   # initial state must be 1D
    sol = solve_ivp(
        EOM_wrapper_Rudder_DoublePulsed,
        t_span,
        x0,
        t_eval=t_eval,
        method='RK45',
        args=(control_inputs,w,aircraft_parameters)
    )
    time = sol.t
    aircraft_state_array = sol.y
    control_input_array = np.tile(control_inputs.reshape(4,1),(1,len(time)))
    PlotSimulation(time=time,aircraft_state_array=aircraft_state_array,control_input_array=control_input_array,col = 'b-')
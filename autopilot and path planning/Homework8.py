import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ------------------------------------------------------------------
# Import your converted / provided Python modules here
# ------------------------------------------------------------------

from FirstOrderOrbitGuidance import first_order_orbit_guidance,orbit_guidance
from GNC_Sim import TransformFromInertialToBody, AirRelativeVelocityVectorToWindAngles,AircraftEOM,EOM_wrapper,calculate_trim,calc_cost_val_for_SLUF,calc_state_vars_for_SLUF,WindAnglesToAirRelativeVelocityVector,AirRelativeVelocityVectorToWindAngles,TransformFromBodyToInertial
from PlotSimulationWithCommands import PlotSimulationWithCommands
from SLCWithFeedForwardAutopilot import SLCWithFeedForwardAutopilot
from SimpleSLCAutopilot import SimpleSLCAutopilot
from CalculateControlGainSimpleSLC import CalculateControlGains
from PlotSimulationWithCommands import PlotSimulationWithCommands
from DefineTTwistor import DefineTTwistor
from DrawAircraft import DrawAircraft
from AnimateSimulation import AnimateSimulation
from scipy.optimize import minimize
from ttwistor import AircraftParameters
import stdatmo
from InertiaTerms import inertia_terms
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D

def run_problem_1(aircraft_parameters):
        ap      = aircraft_parameters
        # ----------------------------------------------------------------
        # Trim
        # ----------------------------------------------------------------
        V_trim     = 18.0
        h_trim     = 1805.0
        gamma_trim = 0.0
        trim_definition = np.array([V_trim, gamma_trim, h_trim])
        # ----------------------------------------------------------------
        # Simulation parameters
        # ----------------------------------------------------------------
        Ts     = 0.1    # control sample period (s)
        Tfinal = 300.0  # total simulation time (s)

        n_ind = int(Tfinal / Ts)

        #For problem 1 I am running my basic kinematic guidance model 
        wind_inertial = np.array([0,0,0])
        parameter_struct = parameters()
        Tfinal = 300
        t_span = (0,Tfinal)
        t_eval = np.linspace(0,Tfinal,3000)

        chi_trim = 0
        Va_trim = 18

        # ----------------------------------------------------------------
        # Guidance commands
        # ----------------------------------------------------------------
        h_c        = h_trim                  # commanded altitude (m)
        h_dot_c    = 0.0                     # commanded altitude rate (m/s)
        chi_c      = np.deg2rad(30)                      # commanded course (rad)
        chi_dot_ff = 0.0                     # course rate feedforward (rad/s)
        Va_c       = V_trim                # commanded airspeed (m/s)
        control_objective0 = np.array([h_c, 0, chi_c,0,Va_c])
        x0 = np.array([0,0,0,0,h_trim,0, Va_trim])

        sol = solve_ivp(nl_guidance_model,t_span=t_span,y0=x0,method='RK45',t_eval=t_eval, args=(wind_inertial,control_objective0,parameter_struct) )
        state_array = sol.y
        #------------------------------------------------------------------------------------------------------------------------------------------------
        fig,(axs1,axs2,axs3) = plt.subplots(3,1, figsize=(12,8))
        print(f"The Calculated Chi for  nl guidanc: {state_array[2]}")
        axs1.plot(sol.t, state_array[2],color='red', linestyle='dashed',label="nonlinear model 1a")
        axs1.set_xlabel('time')
        axs1.set_ylabel(r'$\chi$')

        axs2.plot(sol.t, state_array[4,:],color='red', linestyle='dashed',label="nonlinear model 1a")
        axs2.set_xlabel('time')
        axs2.set_ylabel('h')

        axs3.plot(sol.t, state_array[6,:],color='red', linestyle='dashed',label="nonlinear model 1a")
        axs3.set_xlabel('time')
        axs3.set_ylabel('V_a')

        # STUDENTS: replace stubs with HW3 functions
        trim_variables, fval = calculate_trim(trim_definition, wind_inertial,ap)
        aircraft_state_trim, control_input_trim = \
            calc_state_vars_for_SLUF(trim_variables, trim_definition)

        print(f"Trim found with residual fval = {fval:.6e}")
        # ----------------------------------------------------------------
        # Initial conditions
        # ----------------------------------------------------------------
        aircraft_state0        = aircraft_state_trim.copy()
        print(f"The initial aircraft state: {aircraft_state0}")
        aircraft_state0[2]     = -1805.0            # pd: climb mode starts at h=1675
        aircraft_state0[3]     = np.deg2rad(0.0)    # phi; set to 180° when confident

        control_input0  = control_input_trim.copy()
        wind_inertial   = np.zeros(3)

        # ----------------------------------------------------------------
        # Control gains
        # ----------------------------------------------------------------

        Ts = 0.1
        control_gain_struct= load_gains(r'data\ttwistor_gains_feed.mat',
                                        struct_name='control_gain_struct')
        control_gain_struct.Ts = Ts
        autopilot = SLCWithFeedForwardAutopilot(control_gain_struct)
        control_gain_struct.Ts = Ts
        # Optionally zero out gains if guidance only gives rate, not angle:
        control_gain_struct.Kp_course_rate  = 1
        control_gain_struct.Kff_course_rate = 1

        # Pre-allocate storage  (MATLAB: aircraft_array, control_array, …)
        n_states   = len(aircraft_state0)
        n_controls = len(control_input0)
        n_wind     = 3
        n_cmd      = 12

        aircraft_array = np.zeros((n_states,   n_ind + 1))
        aircraft_Va = np.zeros(n_ind + 1)
        aircraft_Va[0] = Va_trim
        #aircraft_Va[0] = np.linalg.norm(aircraft_array[6:9, 0])  # set initial Va from trim state

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
        # autopilot = SLCWithFeedForwardAutopilot(control_gain_struct)
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
            wind_angles_arr[:, i - 1] = AirRelativeVelocityVectorToWindAngles(air_rel_vel_body)

            # Guidance objectives vector
            control_objectives = np.array([
                h_c,
                h_dot_c,
                chi_c,
                chi_dot_ff,
                Va_c,
            ])

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
            sol_true = solve_ivp(
                fun=lambda t, y: EOM_wrapper(
                    t, y, control_array[:, i - 1],wind_inertial, ap
                ),
                t_span=tspan,
                y0=aircraft_array[:, i - 1],
                method='RK45',
                dense_output=False,
                rtol=1e-6,
                atol=1e-8,
            )

            aircraft_array[:, i] = sol_true.y[:, -1]
            time_iter[i]         = sol_true.t[-1]
            wind_array[:, i]     = wind_inertial
            control_array[:, i]  = control_array[:, i - 1]
            x_command[:, i]      = x_command[:, i - 1]

            aircraft_Va[i] = np.linalg.norm(aircraft_array[6:9, i])

            if i % 100 == 0:
                print(f"  Step {i}/{n_ind}  t={time_iter[i]:.1f}s  "
                        f"h={-aircraft_array[2, i]:.1f}m")
        # Quick sanity check on the transform
        test_v = TransformFromBodyToInertial(aircraft_array[6:9, 1], aircraft_array[3:6, 1])
        print(f"test v_inertial: {test_v}")
        print(f"euler angles at step 1: {aircraft_array[3:6, 1]}")
        print(f"aircraft_array[6:9, 1] = {aircraft_array[6:9, 1]}")   # should be ~[18, 0, 0]
        print(f"aircraft_array[6:9, 100] = {aircraft_array[6:9, 100]}")
        print(f"aircraft_state_trim = {aircraft_state_trim}")
        print(f"The estimated Air Relative Velocity Calcs: {aircraft_Va}")
        print("Simulation complete.")
        chi_array = np.zeros(n_ind + 1)
        for i in range(n_ind + 1):
            v_inertial = TransformFromBodyToInertial(
                aircraft_array[6:9, i],   # [u, v, w]
                aircraft_array[3:6, i]    # [phi, theta, psi]
            )
            chi_array[i] = np.arctan2(v_inertial[1], v_inertial[0])  # atan2(v_east, v_north
        chi_array = np.zeros(n_ind + 1)
        for i in range(n_ind + 1):
            v_body = aircraft_array[6:9, i].reshape(3)      # ensure 1D
            euler  = aircraft_array[3:6, i].reshape(3)      # ensure 1D
            v_inertial = TransformFromBodyToInertial(v_body, euler)
            v_inertial = np.array(v_inertial).flatten()     # ensure result is 1D
            chi_array[i] = np.arctan2(v_inertial[1], v_inertial[0])

        print(f"chi_array sample: {chi_array[:5]}")

        axs1.plot(time_iter, chi_array,label="true EOM")
        # axs1.plot(time_iter, aircraft_array[5,:],label="true EOM")
        print(f"The Calculated Chi for full EOM: {chi_array}")
        axs2.plot(time_iter, -aircraft_array[2,:],label="true EOM")

        axs3.plot(time_iter, aircraft_Va,label="true EOM")
        fig.suptitle("Guidance Variables Vs. Time")
        axs1.legend()
        axs2.legend()
        axs3.legend()
        plt.legend()
        plt.show()
from types import SimpleNamespace

def load_gains(filepath, struct_name='control_gain_struct'):
    raw = sio.loadmat(filepath, squeeze_me=True)
    
    # Pull out just the struct you want
    struct = raw[struct_name]
    
    gains = SimpleNamespace()
    for field in struct.dtype.names:
        val = struct[field].item()  # extract scalar from 0-d array
        if isinstance(val, np.ndarray):
            setattr(gains, field, val)
        else:
            setattr(gains, field, float(val))
    
    # if Ts is not None:
    #     gains.Ts = Ts
    
    return gains

def run_problem_2(aircraft_parameters):
        ap      = aircraft_parameters
        # ----------------------------------------------------------------
        # Trim
        # ----------------------------------------------------------------
        V_trim     = 18.0
        h_trim     = 1805.0
        gamma_trim = 0.0
        trim_definition = np.array([V_trim, gamma_trim, h_trim])
        # ----------------------------------------------------------------
        # Simulation parameters
        # ----------------------------------------------------------------

        pos_line = np.array([200,0,-h_trim])
        dir_line = np.array([1.0,2.0,0.0])
        init_pos_ac = np.array([0.0,0.0,-1650])
        chi_inf = np.deg2rad(70) # From Book rec
        kpath = 0.05

        Ts     = 0.1    # control sample period (s)
        Tfinal = 300.0  # total simulation time (s)

        n_ind = int(Tfinal / Ts)

        #For problem 1 I am running my basic kinematic guidance model 
        wind_inertial = np.array([0,0,0])
        parameter_struct = parameters()
        Tfinal = 300
        t_span = (0,Tfinal)
        t_eval = np.linspace(0,Tfinal,3000)

        chi_trim = 0
        Va_trim = 18

        # ----------------------------------------------------------------
        # Guidance commands
        # ----------------------------------------------------------------
        h_c        = h_trim + 20                  # commanded altitude (m)
        h_dot_c    = 0.0                     # commanded altitude rate (m/s)
        chi_c      = 0                       # commanded course (rad)
        chi_dot_ff = 0.0                     # course rate feedforward (rad/s)
        Va_c       = V_trim                  # commanded airspeed (m/s)
        control_objective0 = np.array([h_trim, 0, 0,0,Va_trim])
        x0_p2 = init_pos_ac

        sol2 = solve_ivp(
        fun=lambda t, pos: first_order_straight_guidance(
            t, pos, pos_line, dir_line, kpath, chi_inf, Va_trim
        ),
        t_span=t_span,
        y0=x0_p2,
        method='RK45',
        t_eval=t_eval
        )
        # After solving, extract position history
        pn = sol2.y[0, :]
        pe = sol2.y[1, :]
        pd = sol2.y[2, :]

        # Reconstruct velocity at each timestep by re-calling the guidance function
        n_pts = len(sol2.t)
        chi_array = np.zeros(n_pts)
        Va_array  = np.zeros(n_pts)
        h_array   = -pd  # h = -pd (NED convention)

        for i in range(n_pts):
            pos_i = sol2.y[:, i]
            vel   = first_order_straight_guidance(sol2.t[i], pos_i, pos_line, dir_line, kpath, chi_inf, Va_trim)
            vn, ve, vd = vel
            Va_array[i]  = np.linalg.norm(vel)                # total speed
            chi_array[i] = np.arctan2(ve, vn)                 # course angle from velocity components
        # #------------------------------------------------------------------------------------------------------------------------------------------------
        # fig,(axs1,axs2,axs3) = plt.subplots(3,1, figsize=(12,8))
        # axs1.plot(sol2.t, chi_array, label="straight line guidance")
        # axs2.plot(sol2.t, h_array,   label="straight line guidance")
        # axs3.plot(sol2.t, Va_array,  label="straight line guidance")

        # axs1.set_xlabel('time')
        # axs1.set_ylabel(r'$\chi$')

        # axs2.set_xlabel('time')
        # axs2.set_ylabel('h')

        # axs3.set_xlabel('time')
        # axs3.set_ylabel('V_a')
        # fig3 = plt.figure(figsize=(10, 8))
        # ax3d = fig3.add_subplot(111, projection='3d')

        # # Aircraft path
        # pn = sol2.y[0, :]
        # pe = sol2.y[1, :]
        # h  = -sol2.y[2, :]  # convert pd to h (positive up)

        # ax3d.plot(pe, pn, h, label="aircraft path", color='blue')

        # # Draw the desired line for reference
        # t_line  = np.linspace(0, 10*Tfinal, 100)
        # pn_line = pos_line[0] + dir_line[0] * t_line
        # pe_line = pos_line[1] + dir_line[1] * t_line
        # h_line  = -pos_line[2] + (-dir_line[2]) * t_line  # convert NED to altitude

        # ax3d.plot(pe_line, pn_line, h_line, 'r--', label="desired path")

        # # Mark start point
        # ax3d.scatter(pe[0], pn[0], h[0], color='green', s=50, label="start")

        # ax3d.set_xlabel('East (m)')
        # ax3d.set_ylabel('North (m)')
        # ax3d.set_zlabel('Altitude (m)')
        # ax3d.set_title('3D Flight Path')
        # ax3d.legend()

        # plt.tight_layout()
        # axs1.legend()
        # axs2.legend()
        # axs3.legend()
        # plt.legend()
        # plt.show()
        return sol2, chi_array, h_array,Va_array
def run_problem_3(aircraft_parameters):
    # ==================================================================
    # Aircraft parameters
    # ==================================================================
    aircraft_parameters = AircraftParameters()   

    # ==================================================================
    # Trim state
    # ==================================================================
    Va_trim    = 18.0        # m/s
    h_trim     = 1805.0      # m
    gamma_trim = 0.0         # rad
    trim_definition = np.array([Va_trim, gamma_trim, h_trim])
    # ----------------------------------------------------------------
    # Simulation parameters
    # ----------------------------------------------------------------

    pos_line = np.array([200,0,-h_trim])
    dir_line = np.array([1.0,2.0,0.0])
    init_pos_ac = np.array([0.0,0.0,-1650])
    chi_inf = np.deg2rad(70) # From Book rec
    kpath = 0.05

    Ts     = 0.1    # control sample period (s)
    Tfinal = 300.0  # total simulation time (s)

    n_ind = int(Tfinal / Ts)

    #For problem 1 I am running my basic kinematic guidance model 
    wind_inertial = np.array([0,0,0])
    parameter_struct = parameters()
    Tfinal = 300
    t_span = (0,Tfinal)
    t_eval = np.linspace(0,Tfinal,3000)

    chi_trim = 0


    # ----------------------------------------------------------------
    # Guidance commands
    # ----------------------------------------------------------------
    h_c        = h_trim                  # commanded altitude (m)
    h_dot_c    = 0.0                     # commanded altitude rate (m/s)
    chi_c      = 0                       # commanded course (rad)
    chi_dot_ff = 0.0                     # course rate feedforward (rad/s)
    Va_c       = Va_trim                  # commanded airspeed (m/s)
    control_objective0 = np.array([h_trim, 0, 0,0,Va_trim])
    x0_p2 = init_pos_ac

    # ==================================================================
    # Simulation parameters
    # ==================================================================
    Ts     = 0.1      # control sample time (s)
    Tfinal = 300.0    # total simulation time (s)
    n_ind = int(Tfinal / Ts)

    # ==================================================================
    # Main simulation loop
    # ==================================================================

    wind_array     = np.zeros((3,          n_ind + 1))
    x_command      = np.zeros((12,         n_ind + 1))
    time_iter      = np.zeros(n_ind + 1)
    x0 = np.array([0,0,0,0,1650,0, Va_trim])
    guide_array     = np.zeros((7,n_ind + 1))
    guide_array[:,0] = x0
    control_guide_array = np.zeros((5,         n_ind + 1))

    # Initial conditions
    time_iter[0]         = 0.0

    for i in range(1, n_ind + 1):
        t_start = Ts * (i - 1)
        t_end   = Ts * i
        tspan   = (t_start, t_end)
        wind_array[:, i - 1] = wind_inertial
        current_state = guide_array[:,i-1]
        pos_ac= np.array([current_state[0],current_state[1],-current_state[4]])
        #straight_line_guidance(pos_line, dir_line,pos_ac, kpath,chi_inf,Va_c):
        control_objectives = straight_line_guidance(pos_line,dir_line,pos_ac, kpath,chi_inf,Va_c)


        sol = solve_ivp(nl_guidance_model,t_span=tspan,y0=current_state,method='RK45', args=(wind_inertial,control_objectives,parameter_struct) )

        guide_array[:, i]           = sol.y[:, -1]
        time_iter[i]                = sol.t[-1]
        wind_array[:, i]            = wind_inertial
        control_guide_array[:, i]   = control_objectives

    # # --- 2D time-history plots ---
    # fig, (axs1, axs2, axs3) = plt.subplots(3, 1, figsize=(12, 8))  # Fix 4

    # axs1.plot(time_iter, guide_array[2, :], color='g', linestyle='--', label=r'$\chi$')  # Fix 5
    # axs1.set_xlabel('time (s)')
    # axs1.set_ylabel(r'$\chi$ (rad)')
    # axs1.legend()

    # axs2.plot(time_iter, guide_array[4, :], color='g', linestyle='--', label='h')
    # axs2.set_xlabel('time (s)')
    # axs2.set_ylabel('h (m)')
    # axs2.legend()

    # axs3.plot(time_iter, guide_array[6, :], color='g', linestyle='--', label='Va')
    # axs3.set_xlabel('time (s)')
    # axs3.set_ylabel('Va (m/s)')
    # fig.suptitle("Guidance variables vs Time")
    # axs3.legend()
    # plt.tight_layout()

    # # --- 3D trajectory plot ---
    # from mpl_toolkits.mplot3d import Axes3D
    # pn = guide_array[0, :]
    # pe = guide_array[1, :]
    # h  = guide_array[4, :]

    # fig2 = plt.figure(figsize=(10, 8))
    # ax3d = fig2.add_subplot(111, projection='3d')
    # ax3d.plot(pe, pn, h, color='blue', label='aircraft path')

    # # Draw desired line
    # t_line  = np.linspace(0, Va_trim * Tfinal, 200)
    # pn_line = pos_line[0] + dir_line[0] * t_line
    # pe_line = pos_line[1] + dir_line[1] * t_line
    # h_line  = np.full_like(t_line, h_trim)
    # ax3d.plot(pe_line, pn_line, h_line, 'r--', label='desired path')
    # ax3d.scatter(pe[0], pn[0], h[0], color='green', s=50, label='start')

    # ax3d.set_xlabel('East (m)')
    # ax3d.set_ylabel('North (m)')
    # ax3d.set_zlabel('Altitude (m)')
    # ax3d.set_title('3D Flight Path - Straight Line Following')
    # ax3d.legend()
    # plt.tight_layout()
    # plt.show()

    return time_iter, guide_array
#@dataclass
class parameters():
    def __init__(self):
        self.bcd=0.8
        self.bc=0.9
        self.bhd=1.45
        self.bh=0.075
        self.bVa=1

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


def run_problem_4(aircraft_parameters):
     
    # ==================================================================
    # Flags / mode constants  (replaces MATLAB top-level defines)
    # ==================================================================
    SLC  = 1

    FEED = 2

    ANIMATE_FLAG = True   # Set True to show animation after simulation
    CONTROL_FLAG = FEED    # Set to SLC or FEED

    # ==================================================================
    # Aircraft parameters
    # ==================================================================
    aircraft_parameters = AircraftParameters()   # <-- replace with your loader

    # ==================================================================
    # Trim state
    # ==================================================================
    Va_trim     = 18.0        # m/s
    h_trim     = 1805.0      # m
    gamma_trim = 0.0         # rad
    trim_definition = np.array([Va_trim, gamma_trim, h_trim])

    wind_inertial = np.zeros(3)
    # STUDENTS: replace these two calls with your HW 3/4 versions
    trim_variables, fval = calculate_trim(trim_definition,wind_inertial, aircraft_parameters)
    aircraft_state_trim, control_input_trim = calc_state_vars_for_SLUF(trim_variables, trim_definition)

    # ==================================================================
    # Control gains
    # ==================================================================
    if CONTROL_FLAG == FEED:
        control_gain_struct = load_gains(r'data\ttwistor_gains_feed.mat', 
                                    struct_name='control_gain_struct')

        print('printing gains')
        print(control_gain_struct)

        print("====================================")
        print("AUTOPILOT: SLC with Feedforward")
        print()
    else:
        control_gain_struct = load_gains(r'data\ttwistor_gains_slc.mat', 
                                    struct_name='control_gain_struct')
        print('printing gains')
        print(control_gain_struct)
        print("====================================")
        print("AUTOPILOT: Simple SLC")
        print()

    # ==================================================================
    # Initial conditions
    # ==================================================================
    aircraft_state0 = aircraft_state_trim.copy()
    aircraft_state0[2] = -1655.0   # altitude: climb mode starts at h = 1675 m
    aircraft_state0[3] = 0.0       # phi = 0

    # ==================================================================
    # Simulation parameters
    # ==================================================================
    Ts     = 0.1      # control sample time (s)
    Tfinal = 300.0    # total simulation time (s)
    #control_gain_struct = #CalculateControlGains()
    control_gain_struct.Ts = Ts   # attach sample time to gain struct

    n_ind = int(Tfinal / Ts)

    pos_line = np.array([200,0,-h_trim])
    dir_line = np.array([1.0,2.0,0.0])
    init_pos_ac = np.array([0.0,0.0,-1650])
    chi_inf = np.deg2rad(70) # From Book rec
    kpath = 0.05

    # ----------------------------------------------------------------
    # Guidance commands
    # ----------------------------------------------------------------
    h_c        = h_trim                  # commanded altitude (m)
    h_dot_c    = 0.0                     # commanded altitude rate (m/s)
    chi_c      = 0                       # commanded course (rad)
    chi_dot_ff = 0.0                     # course rate feedforward (rad/s)
    Va_c       = Va_trim                  # commanded airspeed (m/s)
    control_objective0 = np.array([h_trim, 0, 0,0,Va_trim])
    x0_p2 = init_pos_ac
    # --- First-order guidance preview ---
    control_objectives = straight_line_guidance(pos_line,dir_line,init_pos_ac, kpath,chi_inf,Va_c)

    # --- Instantiate autopilots ---
    autopilot_feed = SLCWithFeedForwardAutopilot(control_gain_struct)
    autopilot_slc  = SimpleSLCAutopilot(control_gain_struct)   # when available

        # --- Run FEED ---
    CONTROL_FLAG = FEED
    control_gain_struct_feed = load_gains(r'data\ttwistor_gains_feed.mat',
                                            struct_name='control_gain_struct')
    control_gain_struct_feed.Ts = Ts
    autopilot_feed = SLCWithFeedForwardAutopilot(control_gain_struct_feed)

    # ==================================================================
    # Main simulation loop
    # ==================================================================

    n_states  = len(aircraft_state_trim)
    n_controls= len(control_input_trim)

    aircraft_array = np.zeros((n_states,   n_ind + 1))
    control_array  = np.zeros((n_controls, n_ind + 1))
    wind_array     = np.zeros((3,          n_ind + 1))
    x_command      = np.zeros((12,         n_ind + 1))
    time_iter      = np.zeros(n_ind + 1)

    # Initial conditions
    aircraft_state0 = aircraft_state_trim.copy()
    aircraft_state0[2] = -1655.0    # start below h_trim so climb mode fires
    aircraft_state0[3] = 0.0        # phi = 0

    aircraft_array[:, 0] = aircraft_state0
    control_array[:,  0] = control_input_trim

    x0 = np.array([0,0,0,0,1650,0, Va_trim])
    guide_array     = np.zeros((7,n_ind + 1))
    guide_array[:,0] = x0
    control_guide_array = np.zeros((5,         n_ind + 1))

    time_iter[0]         = 0.0

    for i in range(1, n_ind + 1):
        t_start = Ts * (i - 1)
        t_now = Ts * (i - 1)
        t_end   = Ts * i
        t_span   = (t_start, t_end)
        wind_array[:, i - 1] = wind_inertial
        # Use actual aircraft position from EOM state, not guide_array
        pn_now = aircraft_array[0, i - 1]
        pe_now = aircraft_array[1, i - 1]
        h_now  = -aircraft_array[2, i - 1]   # h = -pd
        pos_ac = np.array([pn_now, pe_now, -h_now])   # back to NED (pd = -h)

        #straight_line_guidance(pos_line, dir_line,pos_ac, kpath,chi_inf,Va_c):
        control_objectives = straight_line_guidance(pos_line,dir_line,pos_ac, kpath,chi_inf,Va_c)

        wind_array[:, i - 1] = wind_inertial

        # --- Wind angles ---
        wind_body = TransformFromInertialToBody(wind_inertial, aircraft_array[3:6, i - 1])
        air_rel_vel_body = aircraft_array[6:9, i - 1] - wind_body
        wind_angles_i = AirRelativeVelocityVectorToWindAngles(air_rel_vel_body)

        # Optionally zero out gains if guidance only gives rate, not angle:
        control_gain_struct.Kp_course_rate  = 1
        control_gain_struct.Kff_course_rate = 1

        control_out, x_c_out = autopilot_feed.update(
            t_now, aircraft_array[:, i - 1], wind_angles_i,
            control_objectives)

        control_array[:, i - 1] = control_out
        x_command[:, i - 1]     = x_c_out
        x_command[4, i - 1]     = trim_variables[0]   # alpha command = trim alpha

        # --- Aircraft dynamics (ODE integration over one Ts step) ---
        sol = solve_ivp(
            fun=lambda t, y: AircraftEOM(
                t, y, control_array[:, i - 1],wind_inertial, aircraft_parameters),
            t_span=t_span,
            y0=aircraft_array[:, i - 1],
            method='RK45',
            rtol=1e-6, atol=1e-9
        )

        aircraft_array[:, i] = sol.y[:, -1]
        time_iter[i]          = sol.t[-1]
        wind_array[:, i]      = wind_inertial
        control_array[:, i]   = control_array[:, i - 1]
        x_command[:, i]       = x_command[:, i - 1]
    # ==================================================================
    # Post-processing
    # ==================================================================
    
    # Compute chi and Va from full EOM results
    chi_array    = np.zeros(n_ind + 1)
    aircraft_Va  = np.zeros(n_ind + 1)
    aircraft_Va[0] = np.linalg.norm(aircraft_array[6:9, 0])
    
    for i in range(n_ind + 1):
        v_inertial      = TransformFromBodyToInertial(
                            aircraft_array[6:9, i],
                            aircraft_array[3:6, i])
        v_inertial      = np.array(v_inertial).flatten()
        chi_array[i]    = np.arctan2(v_inertial[1], v_inertial[0])
        aircraft_Va[i]  = np.linalg.norm(aircraft_array[6:9, i])

    # # ==================================================================
    # # Plot 1: chi, h, Va vs time
    # # ==================================================================
    # fig, (axs1, axs2, axs3) = plt.subplots(3, 1, figsize=(12, 8))

    # axs1.plot(time_iter, chi_array, color='b', label=r'$\chi$ EOM')
    # axs1.set_xlabel('time (s)')
    # axs1.set_ylabel(r'$\chi$ (rad)')
    # axs1.legend()

    # axs2.plot(time_iter, -aircraft_array[2, :], color='b', label='h EOM')
    # axs2.set_xlabel('time (s)')
    # axs2.set_ylabel('h (m)')
    # axs2.legend()

    # axs3.plot(time_iter, aircraft_Va, color='b', label='Va EOM')
    # axs3.set_xlabel('time (s)')
    # axs3.set_ylabel('Va (m/s)')
    # axs3.legend()
    # fig.suptitle("Guidance Vars Vs. Time")

    # plt.tight_layout()

    # # ==================================================================
    # # Plot 2: 3D trajectory
    # # ==================================================================
    # from mpl_toolkits.mplot3d import Axes3D

    # pn = aircraft_array[0, :]
    # pe = aircraft_array[1, :]
    # h  = -aircraft_array[2, :]   # h = -pd

    # fig2  = plt.figure(figsize=(10, 8))
    # ax3d  = fig2.add_subplot(111, projection='3d')

    # ax3d.plot(pe, pn, h, color='blue', label='aircraft path')
    # ax3d.scatter(pe[0], pn[0], h[0], color='green', s=50, label='start')

    # # Draw desired line for reference
    # dir_line_norm = dir_line / np.linalg.norm(dir_line)
    # t_line  = np.linspace(0, np.linalg.norm([pn.max()-pn.min(),
    #                                           pe.max()-pe.min()]) * 1.2, 200)
    # pn_line = pos_line[0] + dir_line_norm[0] * t_line
    # pe_line = pos_line[1] + dir_line_norm[1] * t_line
    # h_line  = np.full_like(t_line, h_trim)
    # ax3d.plot(pe_line, pn_line, h_line, 'r--', label='desired path')

    # # Equal aspect ratio
    # max_range = np.array([pe.max()-pe.min(),
    #                       pn.max()-pn.min(),
    #                       h.max()-h.min()]).max() / 2.0
    # mid_pe = (pe.max() + pe.min()) / 2.0
    # mid_pn = (pn.max() + pn.min()) / 2.0
    # mid_h  = (h.max()  + h.min())  / 2.0
    # ax3d.set_xlim(mid_pe - max_range, mid_pe + max_range)
    # ax3d.set_ylim(mid_pn - max_range, mid_pn + max_range)
    # ax3d.set_zlim(mid_h  - max_range, mid_h  + max_range)

    # ax3d.set_xlabel('East (m)')
    # ax3d.set_ylabel('North (m)')
    # ax3d.set_zlabel('Altitude (m)')
    # ax3d.set_title('3D Flight Path - Straight Line Following (Full EOM)')
    # ax3d.legend()

    # plt.tight_layout()
    # plt.show()

    return time_iter, aircraft_array, chi_array, aircraft_Va

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

# ==================================================================
# Entry point
# ==================================================================
if __name__ == '__main__':

# --- First-order guidance preview ---
# circ_pos = run_first_order_guidance()

# # --- Instantiate autopilots ---
# autopilot_feed = SLCWithFeedForwardAutopilot(control_gain_struct)
# autopilot_slc  = SimpleSLCAutopilot(control_gain_struct)   # when available

# --- Run full simulation ---
# time_iter, aircraft_array, control_array, wind_array, x_command = run_simulation(
#     aircraft_state_trim, control_input_trim,
#     control_gain_struct, trim_variables,
#     autopilot_feed, autopilot_slc,CONTROL_FLAG)

# # --- Plot results ---
# color = 'm' if CONTROL_FLAG == FEED else 'b'

# plot_sim = PlotSimulationWithCommands()
# plot_sim.plot(
#     time_iter, aircraft_array, control_array, wind_array, x_command, color)
# plot_sim.show()
# plot_orbit_circle_on_figure(8, circ_pos)
# plot_orbit_tracking_error(time_iter, aircraft_array, color)

# plt.show()

# --- Run FEED ---
    ap = AircraftParameters()
    #run_problem_1(ap)
    Tfinal = 300
    Va_trim = 18
    h_trim = 1805
    pos_line = np.array([200,0,-1805])
    dir_line = np.array([1.0,2.0,0.0])
# ==================================================================
# Run all problems
# ==================================================================
sol2, chi_array2, h_array2, Va_array2 = run_problem_2(ap)
time_iter3, guide_array3              = run_problem_3(ap)
time_iter4, aircraft_array4, chi_array4, aircraft_Va4 = run_problem_4(ap)

# ==================================================================
# Plot 1: Overlay chi, h, Va vs time - ONE figure, shared axes
# ==================================================================
fig, (axs1, axs2, axs3) = plt.subplots(3, 1, figsize=(12, 8))

# Problem 2
axs1.plot(sol2.t, chi_array2,  color='red',   linestyle='--', label=r'p2 $\chi$')
axs2.plot(sol2.t, h_array2,    color='red',   linestyle='--', label='p2 h')
axs3.plot(sol2.t, Va_array2,   color='red',   linestyle='--', label='p2 Va')

# Problem 3
axs1.plot(time_iter3, guide_array3[2, :], color='green',  linestyle='--', label=r'p3 $\chi$')
axs2.plot(time_iter3, guide_array3[4, :], color='green',  linestyle='--', label='p3 h')
axs3.plot(time_iter3, guide_array3[6, :], color='green',  linestyle='--', label='p3 Va')

# Problem 4
axs1.plot(time_iter4, chi_array4,              color='blue', linestyle='--', label=r'p4 $\chi$')
axs2.plot(time_iter4, -aircraft_array4[2, :],  color='blue', linestyle='--', label='p4 h')
axs3.plot(time_iter4, aircraft_Va4,            color='blue', linestyle='--', label='p4 Va')

axs1.set_xlabel('time (s)')
axs1.set_ylabel(r'$\chi$ (rad)')
axs1.legend()

axs2.set_xlabel('time (s)')
axs2.set_ylabel('h (m)')
axs2.legend()

axs3.set_xlabel('time (s)')
axs3.set_ylabel('Va (m/s)')
axs3.legend()

fig.suptitle('Guidance Variables vs Time')
plt.tight_layout()

# ==================================================================
# Plot 2: Overlay 3D trajectories - ONE figure, shared axes
# ==================================================================
from mpl_toolkits.mplot3d import Axes3D

fig2 = plt.figure(figsize=(10, 8))
ax3d = fig2.add_subplot(111, projection='3d')

# Problem 2
pn2 = sol2.y[0, :]
pe2 = sol2.y[1, :]
h2  = -sol2.y[2, :]
ax3d.plot(pe2, pn2, h2, color='red',   linestyle='--', label='p2 path')

# Problem 3
pn3 = guide_array3[0, :]
pe3 = guide_array3[1, :]
h3  = guide_array3[4, :]
ax3d.plot(pe3, pn3, h3, color='green', linestyle='--', label='p3 path')

# Problem 4
pn4 = aircraft_array4[0, :]
pe4 = aircraft_array4[1, :]
h4  = -aircraft_array4[2, :]
ax3d.plot(pe4, pn4, h4, color='blue',  linestyle='--', label='p4 path')

# Start points
ax3d.scatter(pe2[0], pn2[0], h2[0], color='green', s=50, label='start')

# Desired line
dir_line_norm = dir_line / np.linalg.norm(dir_line)
all_pn = np.concatenate([pn2, pn3, pn4])
all_pe = np.concatenate([pe2, pe3, pe4])
t_line  = np.linspace(0, np.linalg.norm([all_pn.max()-all_pn.min(),
                                          all_pe.max()-all_pe.min()]) * 1.2, 200)
pn_line = pos_line[0] + dir_line_norm[0] * t_line
pe_line = pos_line[1] + dir_line_norm[1] * t_line
h_line  = np.full_like(t_line, float(h_trim))
ax3d.plot(pe_line, pn_line, h_line, color='black', linestyle='-', label='desired path')

# Equal aspect ratio across all trajectories
all_h = np.concatenate([h2, h3, h4])
max_range = np.array([all_pe.max()-all_pe.min(),
                      all_pn.max()-all_pn.min(),
                      all_h.max()-all_h.min()]).max() / 2.0
mid_pe = (all_pe.max() + all_pe.min()) / 2.0
mid_pn = (all_pn.max() + all_pn.min()) / 2.0
mid_h  = (all_h.max()  + all_h.min())  / 2.0
ax3d.set_xlim(mid_pe - max_range, mid_pe + max_range)
ax3d.set_ylim(mid_pn - max_range, mid_pn + max_range)
ax3d.set_zlim(mid_h  - max_range, mid_h  + max_range)

ax3d.set_xlabel('East (m)')
ax3d.set_ylabel('North (m)')
ax3d.set_zlabel('Altitude (m)')
ax3d.set_title('3D Flight Path Comparison')
ax3d.legend()

plt.tight_layout()
plt.show()
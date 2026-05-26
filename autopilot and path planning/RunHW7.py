"""
run_hw7.py
Python conversion of RunHW7.m

Dependencies (you must supply Python versions of these):
    - ttwistor                  : loads aircraft_parameters (dict or object)
    - ttwistor_gains_feed       : loads control_gain_struct for feedforward
    - ttwistor_gains_slc        : loads control_gain_struct for SLC
    - CalculateTrimVariables(trim_definition, aircraft_parameters)
    - TrimConditionFromDefinitionAndVariables(trim_variables, trim_definition)
    - FirstOrderOrbitGuidance(t, y, orbit_speed, orbit_radius, orbit_center,
                              orbit_flag, orbit_gains)
    - OrbitGuidance(pos, orbit_speed, orbit_radius, orbit_center,
                    orbit_flag, orbit_gains)            [STUDENTS complete]
    - TransformFromInertialToBody(wind_inertial, euler)
    - AirRelativeVelocityVectorToWindAngles(air_rel_vel_body)
    - AircraftEOM(t, y, control_input, wind_inertial, aircraft_parameters)
    - PlotSimulationWithCommands(time_iter, aircraft_array, control_array,
                                 wind_array, x_command, color)
    - slc_autopilot.SLCWithFeedForwardAutopilot  (converted class)
    - SimpleSLCAutopilot                           (to be converted)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ------------------------------------------------------------------
# Import your converted / provided Python modules here
# ------------------------------------------------------------------

from FirstOrderOrbitGuidance import first_order_orbit_guidance,orbit_guidance
from GNC_Sim import TransformFromInertialToBody, AirRelativeVelocityVectorToWindAngles,AircraftEOM,calculate_trim,calc_cost_val_for_SLUF,calc_state_vars_for_SLUF
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
V_trim     = 18.0        # m/s
h_trim     = 1805.0      # m
gamma_trim = 0.0         # rad
trim_definition = np.array([V_trim, gamma_trim, h_trim])

# STUDENTS: replace these two calls with your HW 3/4 versions
trim_variables, fval = calculate_trim(trim_definition, aircraft_parameters)
aircraft_state_trim, control_input_trim = calc_state_vars_for_SLUF(trim_variables, trim_definition)


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
# Orbit guidance parameters
# ==================================================================
orbit_speed  = 18.0
orbit_radius = 200.0
orbit_center = np.array([1000.0, 1000.0, -1805.0])
orbit_flag   = 1

class OrbitGains:
    kr = 0.2    # <-- STUDENTS set if needed
    kz = 0.001   # <-- STUDENTS set if needed

orbit_gains = OrbitGains()


# ==================================================================
# Initial conditions
# ==================================================================
aircraft_state0 = aircraft_state_trim.copy()
aircraft_state0[2] = -1655.0   # altitude: climb mode starts at h = 1675 m
aircraft_state0[3] = 0.0       # phi = 0

# control_input0 = control_input_trim.copy()

wind_inertial = np.zeros(3)


# ==================================================================
# Simulation parameters
# ==================================================================
Ts     = 0.1      # control sample time (s)
Tfinal = 300.0    # total simulation time (s)
#control_gain_struct = #CalculateControlGains()
control_gain_struct.Ts = Ts   # attach sample time to gain struct

n_ind = int(Tfinal / Ts)


# ==================================================================
# First-Order Orbit Guidance (first-order model, open-loop check)
# ==================================================================
def run_first_order_guidance():
    """
    Replicates the ode45 call for FirstOrderOrbitGuidance and plots result.
    """
    t_span = (0.0, Tfinal)
    t_eval = np.arange(0.0, Tfinal + Ts, Ts)
    y0     = np.array([0.0, 0.0, -h_trim]) #Good initial Pos, needs rest of conditions

    problem1 = first_order_orbit_guidance(0,y0,orbit_speed, orbit_radius, orbit_center, orbit_flag, orbit_gains)
    print(f'the values for problem1 are: {problem1}')

    sol = solve_ivp(
        fun=lambda t, y: first_order_orbit_guidance(
            t, y, orbit_speed, orbit_radius, orbit_center, orbit_flag, orbit_gains),
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-6, atol=1e-9
    )

    # 3-D path plot (Figure 20 equivalent)
    fig = plt.figure(20)
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot(sol.y[0], sol.y[1], -sol.y[2])

    # Draw desired orbit circle
    angles   = np.linspace(0, 2 * np.pi, 361)
    circ_pos = (orbit_center[:, None]
                + orbit_radius * np.vstack([np.cos(angles), np.sin(angles), np.zeros_like(angles)]))
    ax.plot(circ_pos[0], circ_pos[1], -circ_pos[2], '--')

    ax.set_xlabel('North (m)')
    ax.set_ylabel('East (m)')
    ax.set_zlabel('Altitude (m)')
    ax.set_title('First-Order Orbit Guidance')
    plt.tight_layout()

    return circ_pos


# ==================================================================
# Main simulation loop
# ==================================================================
def run_simulation(aircraft_state_trim, control_input_trim,
                   control_gain_struct, trim_variables,
                   autopilot_feed, autopilot_slc,control_flag):
    """
    Parameters
    ----------
    aircraft_state_trim : np.ndarray, shape (12,)
    control_input_trim  : np.ndarray, shape (4,)
    control_gain_struct : gains object with .Ts set
    trim_variables      : np.ndarray  (used for alpha command)
    autopilot_feed      : SLCWithFeedForwardAutopilot instance
    autopilot_slc       : SimpleSLCAutopilot instance (or None)

    Returns
    -------
    time_iter     : np.ndarray, shape (n_ind+1,)
    aircraft_array: np.ndarray, shape (12, n_ind+1)
    control_array : np.ndarray, shape (4,  n_ind+1)
    wind_array    : np.ndarray, shape (3,  n_ind+1)
    x_command     : np.ndarray, shape (12, n_ind+1)
    """
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
    time_iter[0]         = 0.0

    for i in range(1, n_ind + 1):
        t_now = Ts * (i - 1)
        t_span = (Ts * (i - 1), Ts * i)

        wind_array[:, i - 1] = wind_inertial

        # --- Wind angles ---
        wind_body = TransformFromInertialToBody(wind_inertial, aircraft_array[3:6, i - 1])
        air_rel_vel_body = aircraft_array[6:9, i - 1] - wind_body
        wind_angles_i = AirRelativeVelocityVectorToWindAngles(air_rel_vel_body)

        # --- Guidance (control objectives) ---
        # STUDENTS: uncomment OrbitGuidance once implemented
        # control_objectives = OrbitGuidance(
        #     aircraft_array[0:3, i-1], orbit_speed, orbit_radius,
        #     orbit_center, orbit_flag, orbit_gains)

        # control_objectives = orbit_guidance(
        #     aircraft_array[:, i-1], orbit_speed, orbit_radius,
        #      orbit_center, orbit_flag, orbit_gains)
        #print(f"orbit_guidance output: {control_objectives}")
        #
        # Optionally zero out gains if guidance only gives rate, not angle:
        control_gain_struct.Kp_course_rate  = 0
        control_gain_struct.Kff_course_rate = 1
        # print(f"aircraft_state[2] = {aircraft_array[2, 0]}")  # should be -1655 (negative pd)
        # print(f"h computed = {-aircraft_array[2, 0]}")         # should be 1655
        # print(f"aircraft_state_trim = {aircraft_state_trim}")
        # print(f"control_input_trim  = {control_input_trim}")
        # Placeholder while OrbitGuidance is not yet implemented
        control_objectives = np.array([1805.0, 0.0, 45.0 * np.pi / 180.0, 18/600, V_trim])

        # --- Autopilot ---
        if CONTROL_FLAG == FEED:
            control_out, x_c_out = autopilot_feed.update(
                t_now, aircraft_array[:, i - 1], wind_angles_i,
                control_objectives)
        else:
            control_out, x_c_out = autopilot_slc.update(
                t_now, aircraft_array[:, i - 1], wind_angles_i,
                control_objectives)

        control_array[:, i - 1] = control_out
        x_command[:, i - 1]     = x_c_out
        x_command[4, i - 1]     = trim_variables[0]   # alpha command = trim alpha

        # --- Aircraft dynamics (ODE integration over one Ts step) ---
        sol = solve_ivp(
            fun=lambda t, y: AircraftEOM(
                t, y, control_array[:, i - 1], aircraft_parameters),
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

    return time_iter, aircraft_array, control_array, wind_array, x_command


# ==================================================================
# Plotting helpers
# ==================================================================
def plot_orbit_circle_on_figure(fig_num, circ_pos):
    """Overlay the desired orbit circle on an existing 3-D figure."""
    fig = plt.figure(fig_num)
    ax  = fig.axes[0] if fig.axes else fig.add_subplot(111, projection='3d')
    ax.plot(circ_pos[0], circ_pos[1], -circ_pos[2], '--', label='Desired orbit')


def plot_orbit_tracking_error(time_iter, aircraft_array, color='m'):
    """
    Figure 11: distance from desired orbit vs. time.
    Equivalent to the dist_from_circ loop in MATLAB.
    """
    dist_from_circ = np.zeros(len(time_iter))
    for j in range(len(time_iter)):
        err_pos = aircraft_array[0:3, j] - orbit_center
        dist_from_center  = np.linalg.norm(err_pos)
        dist_from_circ[j] = dist_from_center - orbit_radius

    #fig, ax = plt.subplots(num=11)
    # ax.plot(time_iter, dist_from_circ, color=color)
    # ax.set_title('Distance from Desired Orbit vs. Time')
    # ax.set_ylabel('Tracking Error [m]')
    # ax.set_xlabel('Time [sec]')

    fig = plt.figure(11)
    if not fig.axes:
        ax = fig.add_subplot(111)
        ax.set_title('Distance from Desired Orbit vs. Time')
        ax.set_ylabel('Tracking Error [m]')
        ax.set_xlabel('Time [sec]')
    else:
        ax = fig.axes[0]
    plt.tight_layout()


# ==================================================================
# Entry point
# ==================================================================
if __name__ == '__main__':

    # --- First-order guidance preview ---
    circ_pos = run_first_order_guidance()

    # --- Instantiate autopilots ---
    autopilot_feed = SLCWithFeedForwardAutopilot(control_gain_struct)
    autopilot_slc  = SimpleSLCAutopilot(control_gain_struct)   # when available

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
    CONTROL_FLAG = FEED
    control_gain_struct_feed = load_gains(r'data\ttwistor_gains_feed.mat',
                                           struct_name='control_gain_struct')
    control_gain_struct_feed.Ts = Ts
    autopilot_feed = SLCWithFeedForwardAutopilot(control_gain_struct_feed)

    time_iter_f, aircraft_array_f, control_array_f, wind_array_f, x_command_f = run_simulation(
        aircraft_state_trim, control_input_trim,
        control_gain_struct_feed, trim_variables,
        autopilot_feed, None,control_flag=FEED)

    # --- Run SLC ---
    CONTROL_FLAG = SLC
    control_gain_struct_slc = load_gains(r'data\ttwistor_gains_slc.mat',
                                          struct_name='control_gain_struct')
    control_gain_struct_slc.Ts = Ts
    autopilot_slc = SimpleSLCAutopilot(control_gain_struct_slc)

    time_iter_s, aircraft_array_s, control_array_s, wind_array_s, x_command_s = run_simulation(
        aircraft_state_trim, control_input_trim,
        control_gain_struct_slc, trim_variables,
        None, autopilot_slc,control_flag = SLC)

    # --- Plot both overlaid ---
    plot_sim = PlotSimulationWithCommands()
    plot_sim.plot(time_iter_f, aircraft_array_f, control_array_f, wind_array_f, x_command_f, 'm')
    plot_sim.plot(time_iter_s, aircraft_array_s, control_array_s, wind_array_s, x_command_s, 'b')
    plot_sim.show()

    plot_orbit_circle_on_figure(8, circ_pos)
    plot_orbit_tracking_error(time_iter_f, aircraft_array_f, 'm')
    plot_orbit_tracking_error(time_iter_s, aircraft_array_s, 'b')

    plt.show()
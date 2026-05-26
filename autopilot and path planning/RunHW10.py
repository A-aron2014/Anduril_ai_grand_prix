import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# Import your converted modules
# ---------------------------------------------------------------------------
from ttwistor import AircraftParameters
from SensorParametersTtwistor import SensorParametersTtwistor
from GPSSensor import GPSSensor, reset_gps_sensor
from InertialSensors import InertialSensors
from SimpleEstimator import SimpleEstimator, reset_simple_estimator
from EstimatorAttitudeGPSSmoothing import EstimatorAttitudeGPSSmoothing, reset_estimator
from GNC_Sim import (
    TransformFromInertialToBody,
    AirRelativeVelocityVectorToWindAngles,
    AircraftEOM,
    calculate_trim,
    calc_state_vars_for_SLUF,
)
from SLCWithFeedForwardAutopilot import SLCWithFeedForwardAutopilot
from SimpleSLCAutopilot import SimpleSLCAutopilot
from PlotSimulationWithCommands import PlotSimulationWithCommands

# Replace with your orbit guidance implementation
from FirstOrderOrbitGuidance import orbit_guidance as OrbitGuidance
from FirstOrderOrbitGuidance import first_order_orbit_guidance

import scipy.io as sio
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------
SLC    = 2
FEED   = 1
SIMPLE = 1
SMOOTH = 2

ANIMATE_FLAG      = True   # Set True to show animation
CONTROL_FLAG      = FEED    # FEED or SLC
ESTIM_FLAG        = SMOOTH  # SIMPLE or SMOOTH
ESTIM_CONTROL_FLAG = True  # True = control from estimated state


def load_gains(filepath, struct_name='control_gain_struct'):
    """Load MATLAB .mat gains file into a SimpleNamespace."""
    raw    = sio.loadmat(filepath, squeeze_me=True)
    struct = raw[struct_name]
    gains  = SimpleNamespace()
    for field in struct.dtype.names:
        val = struct[field].item()
        setattr(gains, field, float(val) if not isinstance(val, np.ndarray) else val)
    return gains


def run():
    # -----------------------------------------------------------------------
    # Aircraft parameters
    # -----------------------------------------------------------------------
    aircraft_parameters = AircraftParameters()
    sensor_params       = SensorParametersTtwistor(aircraft_parameters)

    # -----------------------------------------------------------------------
    # Trim
    # -----------------------------------------------------------------------
    V_trim     = 18.0
    h_trim     = 1805.0
    gamma_trim = 0.0
    trim_definition = np.array([V_trim, gamma_trim, h_trim])

    wind_inertial_trim = np.zeros(3)

    # STUDENTS: replace with your HW3/4 versions
    trim_variables, fval = calculate_trim(trim_definition, wind_inertial_trim,
                                          aircraft_parameters)
    aircraft_state_trim, control_input_trim = calc_state_vars_for_SLUF(
        trim_variables, trim_definition
    )

    # -----------------------------------------------------------------------
    # Load control gains
    # -----------------------------------------------------------------------
    if CONTROL_FLAG == FEED:
        control_gain_struct = load_gains('data/ttwistor_gains_feed.mat')
        print('\n====================================')
        print('AUTOPILOT: SLC with Feedforward\n')
    else:
        control_gain_struct = load_gains('data/ttwistor_gains_slc.mat')
        print('\n====================================')
        print('AUTOPILOT: Simple SLC\n')

    # -----------------------------------------------------------------------
    # Guidance parameters
    # -----------------------------------------------------------------------
    gvf_speed  = 18.0
    gvf_radius = 500.0
    gvf_center = np.array([5000.0, 5000.0, -1805.0])
    gvf_flag   = 1
    gvf_gains  = SimpleNamespace(kr=0.01, kz=0.001)

    # -----------------------------------------------------------------------
    # Initial conditions
    # -----------------------------------------------------------------------
    aircraft_state0        = aircraft_state_trim.copy()
    aircraft_state0[2]     = -1655.0   # start below h_trim so climb mode fires
    aircraft_state0[3]     = 0.0       # phi = 0
    control_input0         = control_input_trim.copy()
    wind_inertial          = np.array([0.0, 10.0, 0.0])
    wind_body = TransformFromInertialToBody(wind_inertial,aircraft_state0[3:6])
    aircraft_state0[6:9] = aircraft_state0[6:9] + wind_body

    # -----------------------------------------------------------------------
    # Simulation parameters
    # -----------------------------------------------------------------------
    Ts     = sensor_params.Ts_imu   # 0.1 s
    Tfinal = 200.0 #1000.0
    control_gain_struct.Ts = Ts

    n_ind = int(Tfinal / Ts)

    n_states   = len(aircraft_state0)
    n_controls = len(control_input0)

    aircraft_array     = np.zeros((n_states,   n_ind + 1))
    control_array      = np.zeros((n_controls, n_ind + 1))
    wind_array         = np.zeros((3,          n_ind + 1))
    x_command          = np.zeros((12,         n_ind + 1))
    wind_angles        = np.zeros((3,          n_ind + 1))
    gps_sensor_arr     = np.zeros((5,          n_ind + 1))
    inertial_arr       = np.zeros((8,          n_ind + 1))
    aircraft_state_est = np.zeros((n_states,   n_ind + 1))
    wind_inertial_est  = np.zeros((3,          n_ind + 1))
    wind_angles_est    = np.zeros((3,          n_ind + 1))
    time_iter          = np.zeros(n_ind + 1)

    aircraft_array[:, 0] = aircraft_state0
    control_array[:, 0]  = control_input0
    time_iter[0]         = 0.0

    # Reset persistent sensor/estimator state
    reset_gps_sensor()
    reset_simple_estimator()
    reset_estimator()

    # -----------------------------------------------------------------------
    # Instantiate autopilots
    # -----------------------------------------------------------------------
    autopilot_feed = SLCWithFeedForwardAutopilot(control_gain_struct)
    autopilot_slc  = SimpleSLCAutopilot(control_gain_struct)

    # -----------------------------------------------------------------------
    # Main simulation loop
    # -----------------------------------------------------------------------
    print(f'Running simulation: Ts={Ts}s, Tfinal={Tfinal}s, {n_ind} steps ...')

    for i in range(1, n_ind + 1):
        t_start = Ts * (i - 1)
        t_end   = Ts * i
        t_span  = (t_start, t_end)

        wind_array[:, i - 1] = wind_inertial

        # --- Wind angles ---
        wind_body       = TransformFromInertialToBody(wind_inertial,
                                                       aircraft_array[3:6, i - 1])
        air_rel_vel     = aircraft_array[6:9, i - 1] - wind_body
        wind_angles[:, i - 1] = AirRelativeVelocityVectorToWindAngles(air_rel_vel)

        # -------------------------------------------------------------------
        # Sensor measurements
        # -------------------------------------------------------------------
        gps_sensor_arr[:, i - 1] = GPSSensor(aircraft_array[:, i - 1], sensor_params)
        inertial_arr[:, i - 1]   = InertialSensors(
            aircraft_array[:, i - 1], control_array[:, i - 1],
            wind_inertial, aircraft_parameters, sensor_params
        )

        # -------------------------------------------------------------------
        # Estimator
        # -------------------------------------------------------------------
        if ESTIM_FLAG == SIMPLE:
            aircraft_state_est[:, i - 1], wind_inertial_est[:, i - 1] = \
                SimpleEstimator(time_iter[i - 1],
                                gps_sensor_arr[:, i - 1],
                                inertial_arr[:, i - 1],
                                sensor_params)
        else:
            aircraft_state_est[:, i - 1], wind_inertial_est[:, i - 1] = \
                EstimatorAttitudeGPSSmoothing(time_iter[i - 1],
                                              gps_sensor_arr[:, i - 1],
                                              inertial_arr[:, i - 1],
                                              sensor_params)

        # Wind angles from estimated state
        wind_body_est = TransformFromInertialToBody(
            wind_inertial_est[:, i - 1], aircraft_state_est[3:6, i - 1]
        )
        air_rel_est = aircraft_state_est[6:9, i - 1] - wind_body_est
        wind_angles_est[:, i - 1] = AirRelativeVelocityVectorToWindAngles(air_rel_est)

        # Select state for control
        if ESTIM_CONTROL_FLAG:
            state_con      = aircraft_state_est[:, i - 1]
            wind_angles_con = wind_angles_est[:, i - 1]
        else:
            state_con      = aircraft_array[:, i - 1]
            wind_angles_con = wind_angles[:, i - 1]

        # -------------------------------------------------------------------
        # Guidance  — STUDENTS replace with HW7 guidance algorithm
        # -------------------------------------------------------------------
        # control_objectives = OrbitGuidance(
        #     state_con[0:3], gvf_speed, gvf_radius, gvf_center, gvf_flag, gvf_gains
        # )
        control_objectives = OrbitGuidance(
            state_con, gvf_speed, gvf_radius, gvf_center, gvf_flag, gvf_gains
        )        

        # -------------------------------------------------------------------
        # Autopilot
        # -------------------------------------------------------------------
        if CONTROL_FLAG == FEED:
            control_out, x_c_out = autopilot_feed.update(
                t_start, state_con, wind_angles_con, control_objectives
            )
        else:
            control_out, x_c_out = autopilot_slc.update(
                t_start, state_con, wind_angles_con, control_objectives
            )

        control_array[:, i - 1] = control_out
        x_command[:, i - 1]     = x_c_out
        x_command[4, i - 1]     = trim_variables[0]   # alpha command = trim alpha

        # -------------------------------------------------------------------
        # Aircraft dynamics
        # -------------------------------------------------------------------
        sol = solve_ivp(
            fun=lambda t, y: AircraftEOM(
                t, y, control_array[:, i - 1], wind_inertial, aircraft_parameters
            ),
            t_span=t_span,
            y0=aircraft_array[:, i - 1],
            method='RK45',
            rtol=1e-6, atol=1e-9,
        )

        aircraft_array[:, i]     = sol.y[:, -1]
        time_iter[i]             = sol.t[-1]
        wind_array[:, i]         = wind_inertial
        control_array[:, i]      = control_array[:, i - 1]
        x_command[:, i]          = x_command[:, i - 1]
        aircraft_state_est[:, i] = aircraft_state_est[:, i - 1]
        wind_inertial_est[:, i]  = wind_inertial_est[:, i - 1]

        if i % 500 == 0:
            print(f'  Step {i}/{n_ind}  t={time_iter[i]:.1f}s  '
                  f'h={-aircraft_array[2, i]:.1f}m')

    print('Simulation complete.')

    # -----------------------------------------------------------------------
    # Plotting
    # -----------------------------------------------------------------------
    plotter = PlotSimulationWithCommands()
    plotter.plot(time_iter, aircraft_array, control_array, wind_array, x_command, 'b',label = 'True')
    plotter.plot(time_iter, aircraft_state_est, control_array,
                 wind_inertial_est, x_command, 'r--',label='Estimate')
    
    fig = plt.figure(9)

    # Collect ALL handles from axes
    handles = []
    labels = []

    for ax in fig.axes:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi)
                labels.append(li)

    # Place legend on right
    fig.subplots_adjust(right=0.78)

    fig.legend(handles, labels,
            loc='center left',
            bbox_to_anchor=(0.82, 0.5))

    # Orbit circle
    angles    = np.deg2rad(np.arange(0, 361))
    circ_pn   = gvf_center[0] + gvf_radius * np.cos(angles)
    circ_pe   = gvf_center[1] + gvf_radius * np.sin(angles)
    circ_h    = -gvf_center[2] * np.ones_like(angles)
    fig8 = plt.figure(8)
    ax8  = fig8.add_subplot(111, projection='3d')
    ax8.plot(circ_pn, circ_pe, circ_h, 'k:')
    ax8.legend()
    # Wind estimate plot
    fig11, axs = plt.subplots(3, 1, figsize=(10, 8), num=11)
    # labels = ['wn [m/s]', 'we [m/s]', 'wd [m/s]']
    # for k in range(3):
    #     axs[k].plot(time_iter, wind_inertial_est[k, :], 'b',label='Estimate')
    #     axs[k].axhline(wind_inertial[k], color='g', linestyle='--', label='Modeled')
    #     axs[k].set_ylabel(labels[k])
    # axs[0].set_title('Wind Velocity vs. Time')
    # axs[2].set_xlabel('Time [sec]')
    labels = ['wn [m/s]', 'we [m/s]', 'wd [m/s]']

    for k in range(3):
        axs[k].plot(time_iter, wind_inertial_est[k, :], 'b', label='Estimate')
        axs[k].plot(time_iter, wind_array[k, :], 'k', label='True')
        axs[k].axhline(wind_inertial[k], color='g', linestyle='--', label='Modeled')

        axs[k].set_ylabel(labels[k])
        axs[k].legend() 

    axs[0].set_title('Wind Velocity vs. Time')
    axs[2].set_xlabel('Time [sec]')
    plt.tight_layout()

    # Estimator error plot
    estimator_error = aircraft_state_est - aircraft_array
    fig12, axs2 = plt.subplots(3, 1, figsize=(10, 8), num=12)
    pos_labels = ['X Pos [m]', 'Y Pos [m]', 'Z Pos [m]']
    for k in range(3):
        axs2[k].plot(time_iter, estimator_error[k, :], 'b', label='Error')
        axs2[k].set_ylabel(pos_labels[k])
        axs2[k].legend()  
    axs2[0].set_title('Estimator Position Error')
    axs2[2].set_xlabel('Time [sec]')
    plt.tight_layout()
    plt.legend()
    plt.show()

    return time_iter, aircraft_array, aircraft_state_est, wind_inertial_est


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    run()

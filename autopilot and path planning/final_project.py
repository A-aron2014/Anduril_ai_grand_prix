import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import  Rectangle
from dataclasses import dataclass
from typing import List, Tuple

#Simulation Libraries
import DefineTTwistor
import DrawAircraft
import AnimateSimulation

import GNC_Sim as uas

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

from SLCWithFeedForwardAutopilot import SLCWithFeedForwardAutopilot
from SimpleSLCAutopilot import SimpleSLCAutopilot
from PlotSimulationWithCommands import PlotSimulationWithCommands

# Replace with your orbit guidance implementation
from FirstOrderOrbitGuidance import orbit_guidance as OrbitGuidance
from FirstOrderOrbitGuidance import first_order_orbit_guidance

import scipy.io as sio
from types import SimpleNamespace



#Claude help for animation 
from DrawAircraftWithTargets import DrawAircraftWithTargets

# ============================================================
# 1.  IMPORTS  (add at top of your main file)
# ============================================================
from GuidanceStateMachine import GuidanceStateMachine, GuidanceMachineParams, GuidanceState

# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------
SLC    = 2
FEED   = 1
SIMPLE = 1
SMOOTH = 2

ANIMATE_FLAG      = True   # Set True to show animation
CONTROL_FLAG      = FEED    # FEED or SLC
ESTIM_FLAG        = SIMPLE  # SIMPLE or SMOOTH
ESTIM_CONTROL_FLAG = False  # True = control from estimated state
np.random.seed(42)

#LYAPUNOV    = True


def load_gains(filepath, struct_name='control_gain_struct'):
    """Load MATLAB .mat gains file into a SimpleNamespace."""
    raw    = sio.loadmat(filepath, squeeze_me=True)
    struct = raw[struct_name]
    gains  = SimpleNamespace()
    for field in struct.dtype.names:
        val = struct[field].item()
        setattr(gains, field, float(val) if not isinstance(val, np.ndarray) else val)
    return gains
@dataclass


class MovingTarget:
    """Represents a moving point-source target."""
    id: int
    x: float  # current pixel coordinates
    y: float
    z: float
    vx: float  # velocity (pixels/frame)
    vy: float
    #snr: float
    #signal_level: float
    conf: float
    
    # Track history for visualization
    history_x: List[float] = None
    history_y: List[float] = None
    
    def __post_init__(self):
        if self.history_x is None:
            self.history_x = [self.x]
        if self.history_y is None:
            self.history_y = [self.y]
        conf = 1.0 #hard write confidence in detection to 1
    
    def update_position(self, dt, image_bounds: Tuple[int, int] = (10000, 10000)):
        """
        Update position with velocity and handle boundary conditions.
        
        Args:
            dt: Time step (frames)
            image_bounds: (width, height) of image
        """
        # Update position
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        # Store history
        self.history_x.append(self.x)
        self.history_y.append(self.y)
        
        # Bounce off boundaries since we can't let them leave the frame
        width, height = image_bounds
        if self.x < 20 or self.x > width - 20:
            self.vx *= -1  # Reverse direction
            self.x = np.clip(self.x, 20, width - 20)
        
        if self.y < 20 or self.y > height - 20:
            self.vy *= -1
            self.y = np.clip(self.y, 20, height - 20)


    
def generate_moving_targets(image_width,image_height, num_targets) -> List[MovingTarget]:
    """
    Generate targets with random initial positions and velocities.
    
    Returns:
        List of MovingTarget objects
    """
    targets = []
    min_edge = 20
    max_separation = 10
    
    # Velocity ranges (pixels/frame)
    # Slow to moderate speeds for visualization
    v_min = 0.25   # 1 pixel/frame
    v_max = 1.0   # 5 pixels/frame
    
    #for i, snr in enumerate(self.snr_levels):
        # Try to place target
        #   for attempt in range(1000):
            # Random initial position

    for tgt in range(num_targets):
        x = np.random.uniform(min_edge, image_width - min_edge)
        y = np.random.uniform(min_edge, image_height - min_edge)
        z = -1650
        # Check separation from existing targets
        valid = True
        for existing in targets:
            dist = np.sqrt((x - existing.x)**2 + (y - existing.y)**2)
            if dist < max_separation:
                valid = False
                break
        
        if valid:
            # Random velocity
            speed = np.random.uniform(v_min, v_max)
            angle = np.random.uniform(0, 2*np.pi)
            vx = speed * np.cos(angle)
            vy = speed * np.sin(angle)
            
            # Calculate signal level
            #signal_level = pipeline.calculate_signal_level(self.snr_levels)
            #self.true_signal_level = signal_level  # add this line
            target = MovingTarget(
                id=tgt,
                x=x,
                y=y,
                z=z,
                vx=vx,
                vy=vy,
                conf = 1.0
            )
            targets.append(target)
    return targets

def generate_clustered_moving_targets(image_width, image_height, num_targets,
                            num_clusters=3,
                            cluster_std=80.0,
                            min_edge=50,
                            seed=None):
    """
    Generate clustered moving targets for multi-target tracking evaluation.
    """

    if seed is not None:
        np.random.seed(seed)

    targets = []

    z = -1650

    # -----------------------------
    # 1. Generate cluster centers
    # -----------------------------
    cluster_centers = []
    for _ in range(num_clusters):
        cx = np.random.uniform(min_edge, image_width - min_edge)
        cy = np.random.uniform(min_edge, image_height - min_edge)
        cluster_centers.append(np.array([cx, cy]))

    # -----------------------------
    # 2. Assign targets to clusters
    # -----------------------------
    for i in range(num_targets):

        # pick a cluster
        c = cluster_centers[np.random.randint(0, num_clusters)]

        # Gaussian spread around cluster center
        x, y = np.random.normal(loc=c, scale=cluster_std)

        # keep inside bounds
        x = np.clip(x, min_edge, image_width - min_edge)
        y = np.clip(y, min_edge, image_height - min_edge)

        # velocity (correlated within cluster = more realistic)
        speed = np.random.uniform(0.25, 1.0)
        angle = np.random.uniform(0, 2*np.pi)

        vx = speed * np.cos(angle)
        vy = speed * np.sin(angle)

        targets.append(MovingTarget(
            id=i,
            x=x,
            y=y,
            z=z,
            vx=vx,
            vy=vy,
            conf=1.0
        ))

    return targets


def build_aircraft_fov(aircraft_loc):
    #make radius of FOV a function of height
    r = -aircraft_loc[2]//10
    origin = [aircraft_loc[0],aircraft_loc[1],0.0]
    return origin, r

def find_targets_in_fov(origin,r,target_list):
    #Since this is mimicking a TO-MHT or PHD Filter style tracker we assume that the UAV "Knows" how many targets are within it's FOV


    #Convert targets in the list to polar coordinates that are relative to the origin of the FOV
    found_targets = []
    for targets in target_list:
        dist = np.hypot(targets.x-origin[0],targets.y-origin[1])
        if dist <= r :
            found_targets.append(targets)

    return found_targets

def find_target_centroid(target_list):
    x_c = sum(t.x for t in target_list) / len(target_list)
    y_c = sum(t.y for t in  target_list) / len(target_list)

    return np.array([x_c,y_c,-1650])

#-------------------------------------------------------------------------
#Final Project Work - Lyapunov Vector Fields
#-------------------------------------------------------------------------
class OrbitField:
    def __init__(self, k_r, k_theta, R):
        self.k_r = k_r
        self.k_theta = k_theta
        self.R = R

    def compute(self, p,center):
        p = np.array(p[:2])
        c = np.array(center[:2])

        dx = p - c
        r = np.linalg.norm(dx) + 1e-6
        r_hat = dx / r
        t_hat = np.array([-r_hat[1], r_hat[0]])

        radial = -self.k_r * (r - self.R) * r_hat
        tangent = self.k_theta * t_hat
        return radial + tangent
    
class TrackField:
    def __init__(self, k_t):
        self.k_t = k_t

    def compute(self, p, target_pos):
        p = np.array(p[:2])
        t = np.array(target_pos[:2])


        return -self.k_t * (p-t)
    
class CoverageField:
    def __init__(self, k_J, sigma):
        self.k_J = k_J
        self.sigma = sigma

    def compute(self, p, targets):
        p = np.array(p[:2])

        gradient = np.zeros(2)

        for t in targets:
            ti = np.array([t.x,t.y])

            d = ti - p
            dist2 = np.dot(d,d)

            weight = np.exp(-dist2/(self.sigma**2))

            gradient += weight*d
        return self.k_J*gradient
    
class ConfidenceModel:
    def compute_weight(self, detections):
        if len(detections)==0:
            return 0.0
        confidence = sum(d.conf for d in detections)/len(detections)

        return np.clip(confidence,0.0,1.0)
    
def normalize(v,eps=1e-6):
    n = np.linalg.norm(v)
    return v/(n+eps)
    
class GuidanceMixer:
    def __init__(self,orbit,track,coverage,confidence_model):
        self.orbit = orbit
        self.track = track
        self.coverage = coverage
        self.conf_model = confidence_model

    def compute(self, state, targets, target_est, orbit_center):
        p = state[:3]

        v_orbit = self.orbit.compute(p, orbit_center)
        v_cov = self.coverage.compute(p, targets)

        w_t = self.conf_model.compute_weight(targets)

        if target_est is not None:
            v_track = self.track.compute(p, target_est)
        else:
            v_track = np.zeros(2)

        #Normalize components
        # v_orbit = normalize(v_orbit)
        # v_track = normalize(v_track)
        # v_cov   = normalize(v_cov)
       # if len(targets) == 0:
       #     w_cov = 3.0
       #     w_orbit = 0.3
       # else: 
        w_cov = 1.0
        w_orbit = 1.0

        v = w_orbit*v_orbit + w_t*v_track + w_cov*v_cov

        return v, w_t
    
def project_to_speed(v,V_cmd):
    return V_cmd * v / (np.linalg.norm(v) + 1e-6)

def lyapunov_guidance(state, targets, gsm, mixer, params):

    # aircraft planner position
    p_ac = state[0:2]

    #target estimates
    if len(targets) > 0:
        target_est = find_target_centroid(targets)
    else:
        target_est = gsm.last_known_pos[:2]

    if gsm.state == GuidanceState.TRANSIT:
        if gsm.last_known_pos is not None:
            p_t = gsm.last_known_pos[:2]
            v_xy = p_t - p_ac
            return project_to_speed(v_xy, params.V_cmd)
        
    #Orbit center 
    if gsm.state == GuidanceState.SEARCH:
        orbit_center = gsm.p.home_pos[:2] #could turn this into a wandering center too
    else:
        orbit_center = target_est

    # compute the vector field
    v_xy, w_t = mixer.compute(
        state = state,
        targets = targets,
        target_est = target_est,
        orbit_center = orbit_center
    )

    #normalize to commanded speed
    speed = np.linalg.norm(v_xy)
    if speed > 1e-6:
        v_xy = params.V_cmd*v_xy/speed
    else:
        v_xy = np.zeros(2)

    return v_xy

def run(first_scan,LYAPUNOV, targets):
    
    targets_found = False #initiazlize as false, change to true when targets are located
    global prev_num_targets
    animate = True
    last_known_target_pos = find_target_centroid(targets)
    if first_scan:
        first_scan = False


        #generate initial centroid for targets

        

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
    trim_variables, fval = uas.calculate_trim(trim_definition, wind_inertial_trim,
                                          aircraft_parameters)
    aircraft_state_trim, control_input_trim = uas.calc_state_vars_for_SLUF(
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
    # Initial conditions
    # -----------------------------------------------------------------------
    aircraft_state0        = aircraft_state_trim.copy()
    aircraft_state0[2]     = -1655.0   # start below h_trim so climb mode fires
    aircraft_state0[3]     = 0.0       # phi = 0
    control_input0         = control_input_trim.copy()
    wind_inertial          = np.array([0.0, 10.0, 0.0])
    wind_body = uas.TransformFromInertialToBody(wind_inertial,aircraft_state0[3:6])
    aircraft_state0[6:9] = aircraft_state0[6:9] + wind_body

    
    current_target_pos = last_known_target_pos
    gvf_center = np.array([current_target_pos[0], current_target_pos[1], -1805.0])   # keep for plotting the reference circle

    gsm_params = GuidanceMachineParams(
        h_trim        = h_trim,
        h_tol         = 50.0,
        r_capture     = 150.0,
        orbit_speed   = 18.0,
        orbit_r_init  = 200.0,
        orbit_r_step  = 100.0,
        orbit_r_max   = 2000.0,
        orbit_flag    = 1,
        home_pos      = current_target_pos,
        loss_timeout  = 5.0,
        track_r       = 300.0,
        chi_inf       = np.deg2rad(70),
        kpath         = 0.05,
    )


    #------------------------------------------------------------------------
    # Define Lyapunov Parameters
    #------------------------------------------------------------------------
    lyap_params = SimpleNamespace(
    k_r = 1.0,
    k_theta = 1.0,
    k_t = 1.0,
    k_J = 1.0,
    R_search = gsm_params.orbit_r_init,
    R_track  = gsm_params.track_r,
    sigma    = 600.0,
    V_cmd    = 18.0
    )

    # Initialise with last known target position (from your target list)
    gsm_params.home_pos = current_target_pos

    gsm = GuidanceStateMachine(
        params                    = gsm_params,
        orbit_guidance_fn         = OrbitGuidance,           # pass directly, no lambda
        straight_line_guidance_fn = uas.straight_line_guidance,
    )


    orbit_field = OrbitField(k_r = lyap_params.k_r, k_theta=lyap_params.k_theta, R = gsm_params.orbit_r_init)
    track_field = TrackField(k_t = lyap_params.k_t)
    coverage_field = CoverageField(k_J=lyap_params.k_J, sigma = lyap_params.sigma)
    confidence_model = ConfidenceModel()

    mixer = GuidanceMixer(
        orbit=orbit_field,
        track = track_field,
        coverage=  coverage_field,
        confidence_model=confidence_model
    )
    # Seed the state machine with the first target's position so TRANSIT
    # knows where to go as soon as we climb out.
    if targets:
        gsm.last_known_pos = np.array([targets[0].x, targets[0].y, 0.0])
        gsm.transit_wp     = gsm.last_known_pos.copy()


    chi_inf = np.deg2rad(70) # From Book rec
    kpath = 0.05
    # -----------------------------------------------------------------------
    # Simulation parameters
    # -----------------------------------------------------------------------
    Ts     = sensor_params.Ts_imu   # 0.1 s
    Tfinal = 1000.0
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


    # -----------------------------------------------------------------------
    # Logging vars for analysis 
    # -----------------------------------------------------------------------
    num_targets_tracked = np.zeros(n_ind+1)
    state_log = np.zeros(n_ind+1)
    x_log = np.zeros(n_ind+1)
    y_log = np.zeros(n_ind+1)
    target_history = []

    state_map = {
        GuidanceState.SEARCH: 0,
        GuidanceState.TRANSIT: 1,
        GuidanceState.TRACK: 2
    }

    Va_min = 2.0   # m/s — below this the estimator result is meaningless anyway

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
        #Before anything else update target positions. 
        for tgt in targets:
            tgt.update_position(Ts)

        t_start = Ts * (i - 1)
        t_end   = Ts * i
        t_span  = (t_start, t_end)

        wind_array[:, i - 1] = wind_inertial

        # --- Wind angles ---
        wind_body       = uas.TransformFromInertialToBody(wind_inertial,
                                                       aircraft_array[3:6, i - 1])
        air_rel_vel     = aircraft_array[6:9, i - 1] - wind_body
        # wind_angles[:, i - 1] = uas.AirRelativeVelocityVectorToWindAngles(air_rel_vel)
        air_rel_vel = aircraft_array[6:9, i - 1] - wind_body
        Va_true_mag = float(np.linalg.norm(air_rel_vel))
        
        if Va_true_mag > Va_min:
            wind_angles[:, i - 1] = uas.AirRelativeVelocityVectorToWindAngles(air_rel_vel)
        else:
            wind_angles[:, i - 1] = wind_angles[:, i - 2] if i > 1 else np.zeros(3)


        state_con      = aircraft_array[:, i - 1]
        wind_angles_con = wind_angles[:, i - 1]

        # -------------------------------------------------------------------
        # FOV  &  target detection
        # -------------------------------------------------------------------
        fov_origin, fov_radius = build_aircraft_fov(aircraft_array[0:3, i-1])
        found_targets          = find_targets_in_fov(fov_origin, fov_radius, targets)

        # Safety: only compute centroid when there are detections
        if found_targets and gsm.state in [GuidanceState.SEARCH,GuidanceState.TRACK]:
            est_target_center = find_target_centroid(found_targets)
            gsm.last_known_pos = np.array([est_target_center[0], est_target_center[1],
                                            -gsm_params.h_trim])
            num_targets_tracked[i] = len(found_targets)
        state_map[i] = state_map.get(gsm.state,-1)
        x_log[i] = aircraft_array[0,i-1]
        y_log[i] = aircraft_array[1,i-1]
        target_snapshot= [(t.x,t.y) for t in targets]
        target_history.append(target_snapshot)


        # -------------------------------------------------------------------
        # Guidance  — state machine drives everything
        # -------------------------------------------------------------------
        
        if LYAPUNOV:

            #Update for memory and state transitions only
            gsm.update(
            aircraft_state = aircraft_array[:, i-1],
            wind_angles    = wind_angles_est[:, i-1],
            found_targets      = found_targets,
            dt                 = Ts
            )


            if gsm.state == GuidanceState.SEARCH:
                orbit_field.R = gsm.search_orbit_r

            else:
                orbit_field.R = gsm_params.track_r

            v_xy = lyapunov_guidance(
                state = aircraft_array[:,i-1],
                targets=found_targets,
                gsm= gsm,
                mixer= mixer,
                params= lyap_params
            )

            h_cmd = gsm_params.h_trim
            h_dot_cmd = 0.0
            chi_cmd = np.arctan2(v_xy[1],v_xy[0])
            chi_dot_cmd = 0.0
            Va_cmd = lyap_params.V_cmd

            control_objectives = np.array([h_cmd, h_dot_cmd, chi_cmd, chi_dot_cmd, Va_cmd])
        else:

            control_objectives = gsm.update(
                aircraft_state = aircraft_array[:, i-1],
                wind_angles    = wind_angles_est[:, i-1],
                found_targets      = found_targets,
                dt                 = Ts,
            )
        
        # Optional: print state transitions every N steps
        if i % 100 == 0:
            print(f"  [GSM] state={gsm.state_name}  orbit_r={gsm.search_orbit_r:.0f}m  "
                f"targets_in_fov={len(found_targets)}")


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
            fun=lambda t, y: uas.AircraftEOM(
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
            
        # #---------------------------------------------------------------------
        # #Search for targets
        # #---------------------------------------------------------------------
        # fov_origin, fov_radius = build_aircraft_fov(aircraft_state_est[0:3,i-1])

        # found_targets = find_targets_in_fov(fov_origin, fov_radius, targets)
        
        # est_target_center = find_target_centroid(found_targets)

        # current_target_pos = est_target_center

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
    # circ_pn   = gvf_center[0] + gvf_radius * np.cos(angles)
    # circ_pe   = gvf_center[1] + gvf_radius * np.sin(angles)
    circ_pn   = gvf_center[0] + gsm.search_orbit_r * np.cos(angles)
    circ_pe   = gvf_center[1] + gsm.search_orbit_r * np.sin(angles)
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


    # ============================================================
    # 4.  AFTER THE LOOP  (add to your plotting section)
    # ============================================================

    # Plot state machine history
    state_numeric = [s.value for s in gsm.state_log]
    state_labels  = {1: 'TAKEOFF', 2: 'TRANSIT', 3: 'SEARCH', 4: 'TRACK'}

    fig_gsm, ax_gsm = plt.subplots(figsize=(12, 3), num=20)
    ax_gsm.step(time_iter[:len(state_numeric)], state_numeric, where='post', color='steelblue')
    ax_gsm.set_yticks(list(state_labels.keys()))
    ax_gsm.set_yticklabels(list(state_labels.values()))
    ax_gsm.set_xlabel('Time [s]')
    ax_gsm.set_title('Guidance state vs. time')
    ax_gsm.grid(True, alpha=0.3)

    # Plot search orbit radius over time
    fig_orb, ax_orb = plt.subplots(figsize=(12, 3), num=21)
    ax_orb.plot(time_iter[:len(gsm.orbit_r_log)], gsm.orbit_r_log, color='darkorange')
    ax_orb.set_xlabel('Time [s]')
    ax_orb.set_ylabel('Search orbit radius [m]')
    ax_orb.set_title('Expanding search orbit')
    ax_orb.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.legend()
    plt.show()
    # Remove the duplicate FOV block that was at the bottom of the loop
    # (it's now handled inside the guidance section above)
   
    if animate:
        pts    = DefineTTwistor.DefineTTwistor().pts

        # Use the extended drawer — same interface, extra keyword args
        drawer = DrawAircraftWithTargets(
            pts,
            scale           = 10,
            axis_half_range = (500, 500, 300),   # wider window to see targets
            fov_radius_fn   = build_aircraft_fov,
            trail_len       = 60,
        )

        # ------------------------------------------------------------------
        # Replay loop
        # ------------------------------------------------------------------
        # We need targets at each historical timestep.
        # Since MovingTarget stores full history, we reconstruct positions
        # from the history lists rather than re-simulating.
        # ------------------------------------------------------------------

        n_frames = len(time_iter)

        # Pre-build per-frame target snapshots from the stored history
        # (history_x / history_y are appended once per Ts step in the sim loop)
        def get_targets_at_frame(targets_list, frame_idx):
            """Return a lightweight list of (id, x, y) at a given frame index."""
            snaps = []
            for tgt in targets_list:
                idx = min(frame_idx, len(tgt.history_x) - 1)
                snaps.append((tgt.id, tgt.history_x[idx], tgt.history_y[idx]))
            return snaps

        # Rebuild fov + found list per frame using the same logic as the sim
        class _TgtSnap:
            """Minimal duck-type of MovingTarget for the drawer."""
            def __init__(self, tid, x, y, hx, hy):
                self.id = tid; self.x = x; self.y = y
                self.history_x = hx; self.history_y = hy

        for aa in range(0, n_frames, 1):   # step every frame; use ::2 to speed up
            # Reconstruct snapshot targets at this frame
            snap_targets = []
            for tgt in targets:
                idx = min(aa, len(tgt.history_x) - 1)
                # Trail up to current frame
                hx = tgt.history_x[:idx+1]
                hy = tgt.history_y[:idx+1]
                snap_targets.append(_TgtSnap(tgt.id, hx[idx], hy[idx], hx, hy))

            # Recompute FOV membership
            #est_pos = aircraft_state_est[:3, aa]
            est_pos = aircraft_array[:3,aa]
            fov_origin, fov_radius = build_aircraft_fov(est_pos)
            snap_found = find_targets_in_fov(fov_origin, fov_radius, snap_targets)

            drawer.update(
                time            = time_iter[aa],
                aircraft_state  = aircraft_array[:, aa],
                targets         = snap_targets,
                found_targets   = snap_found,
                gsm             = gsm,
                aircraft_state_est = est_pos,
            )

        # ------------------------------------------------------------------
        # Full-path AnimateSimulation still works unchanged if you want it:
        # AnimateSimulation(tout=time_iter, xarray=aircraft_array.T)
        # ------------------------------------------------------------------


        # ============================================================
        # One-line swap in your sim loop for real-time animation
        # (optional — only if ANIMATE_FLAG is True during the sim):
        # ============================================================
        #
        # Replace:
        #   if ANIMATE_FLAG:
        #       drawer = DrawAircraft(pts)
        #       drawer.update(time_iter[i], aircraft_array[:, i])
        #
        # With:
        #   if ANIMATE_FLAG:
        #       drawer.update(
        #           time            = time_iter[i],
        #           aircraft_state  = aircraft_array[:, i],
        #           targets         = targets,
        #           found_targets   = found_targets,        # already computed above
        #           gsm             = gsm,
        #           aircraft_state_est = aircraft_state_est[:, i],
        #       )
        #
        # And before the loop, replace the DrawAircraft construction:
        #   pts    = DefineTTwistor().pts
        #   drawer = DrawAircraftWithTargets(
        #       pts,
        #       fov_radius_fn   = build_aircraft_fov,
        #       axis_half_range = (500, 500, 300),
        #   )



    #return time_iter, aircraft_array, aircraft_state_est, wind_inertial_est
    return {
        "time":time_iter,
        "x": x_log,
        "y": y_log,
        "targets_tracked": num_targets_tracked,
        "state": state_log,
        "targets": target_history
    }

def plot_tracking_performance(results_list, labels):
    plt.figure()

    for results, label in zip(results_list, labels):
        plt.plot(results["time"], results["targets_tracked"], label=label)

    plt.xlabel("Time [s]")
    plt.ylabel("Targets in FOV")
    plt.title("Target Tracking Performance")
    plt.legend()
    plt.grid()
def plot_trajectories(results_list, labels):
    fig, axs = plt.subplots(1, len(results_list), figsize=(15,5))

    if len(results_list) == 1:
        axs = [axs]

    for ax, results, label in zip(axs, results_list, labels):
        x = results["x"]
        y = results["y"]

        ax.plot(x, y, 'b-', label="UAS Path")

        # Plot targets (final positions)
        targets = results["targets"][-1]
        tx = [t[0] for t in targets]
        ty = [t[1] for t in targets]

        ax.scatter(tx, ty, c='r', s=20, label="Targets")

        ax.set_title(label)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.axis('equal')
        ax.grid()

    plt.tight_layout()
def plot_trajectory_with_fov(results, fov_radius=180):
    plt.figure()

    x = results["x"]
    y = results["y"]

    plt.plot(x, y, 'b-', label="UAS Path")

    # draw FOV circles every N steps
    for k in range(0, len(x), 200):
        circle = plt.Circle((x[k], y[k]), fov_radius,
                            fill=False, alpha=0.1, color='blue')
        plt.gca().add_patch(circle)

    # targets
    targets = results["targets"][-1]
    tx = [t[0] for t in targets]
    ty = [t[1] for t in targets]

    plt.scatter(tx, ty, c='r', s=20, label="Targets")

    plt.axis('equal')
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title("Trajectory with FOV Coverage")
    plt.legend()
    plt.grid()

def plot_state_machine(results):
    plt.figure()

    plt.plot(results["time"], results["state"])

    plt.yticks([0,1,2], ["SEARCH","TRANSIT","TRACK"])
    plt.xlabel("Time [s]")
    plt.ylabel("State")
    plt.title("Guidance State Evolution")
    plt.grid()

def plot_vector_field(mixer, targets, target_est, orbit_center,
                      xlim, ylim, resolution=25):

    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)

    X, Y = np.meshgrid(x, y)

    U = np.zeros_like(X)
    V = np.zeros_like(Y)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            state = np.array([X[i,j], Y[i,j], 0,0,0,0])
            v, _ = mixer.compute(state, targets, target_est, orbit_center)

            U[i,j] = v[0]
            V[i,j] = v[1]

    plt.figure()
    plt.quiver(X, Y, U, V)

    # plot targets
    tx = [t.x for t in targets]
    ty = [t.y for t in targets]
    plt.scatter(tx, ty, c='r')

    plt.title("Lyapunov Vector Field")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.axis('equal')
if __name__ == "__main__":
    import copy
    first_scan = True
    #run(first_scan)
    targets = [] #Hold the current set of targets currently found
    targets = generate_moving_targets(1250,1250,9)

    res_orbit = run(first_scan=True,LYAPUNOV=False,targets=copy.deepcopy(targets))
    res_lyap  = run(first_scan=True,LYAPUNOV = True,targets=copy.deepcopy(targets))

    plot_tracking_performance(
        [res_orbit, res_lyap],
        ["Orbit", "Lyapunov"]
    )

    plot_trajectories(
        [res_orbit, res_lyap],
        ["Orbit", "Lyapunov"]
    )

    plot_state_machine(res_lyap)
    plot_trajectory_with_fov(res_lyap)
    plt.show()


    clustered_targets = []
    clustered_targets = generate_clustered_moving_targets(1250,1250,9)

    res_orbit2 = run(first_scan=True,LYAPUNOV=False,targets=copy.deepcopy(clustered_targets))
    res_lyap2  = run(first_scan=True,LYAPUNOV = True,targets=copy.deepcopy(clustered_targets))

    plot_tracking_performance(
        [res_orbit2, res_lyap2],
        ["Orbit", "Lyapunov"]
    )

    plot_trajectories(
        [res_orbit2, res_lyap2],
        ["Orbit", "Lyapunov"]
    )

    plot_state_machine(res_lyap2)
    plot_trajectory_with_fov(res_lyap2)
    plt.show()
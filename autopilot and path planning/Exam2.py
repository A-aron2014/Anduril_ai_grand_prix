import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

import GNC_Sim



#loading wind information from Homework 9
#--------------------------------------------------------------------------
filepath  = 'Homework9\RaavenWindData.mat'
raw_data = sio.loadmat(filepath, squeeze_me=True)

print(f"The raw data is: {raw_data}")

time = raw_data['Time']

Va = raw_data['Va']

agl = raw_data['agl']

aircraft_velocity_inertial = raw_data['aircraft_velocity_inertial'].T
print(aircraft_velocity_inertial.shape)

alpha = raw_data['alpha']
print("alpha shape")
print(alpha.shape)

beta = raw_data['beta']
print("beta shape")
print(beta.shape)
lat = raw_data['lat']

lon = raw_data['lon']

pitch = raw_data['pitch']

roll = raw_data['roll']

yaw = raw_data['yaw']
alpha = np.deg2rad(alpha)
beta = np.deg2rad(beta)
roll = np.deg2rad(roll)
pitch = np.deg2rad(pitch)
yaw = np.deg2rad(yaw)

time_sec = time * 60

phi_dot   = np.gradient(roll, time_sec)
theta_dot = np.gradient(pitch, time_sec)
psi_dot   = np.gradient(yaw, time_sec)

p = phi_dot - psi_dot*np.sin(pitch)

q = theta_dot * np.cos(roll) + psi_dot * np.sin(roll) * np.cos(pitch)

r = -theta_dot * np.sin(roll) + psi_dot * np.cos(roll) * np.cos(pitch)

angular_rates = np.column_stack((p,q,r))

euler_angles = np.column_stack((roll,pitch,yaw))

wind_angles = np.column_stack((Va, beta, alpha))

#Calculating air relative body velocity
air_rel_body = []

for angles in wind_angles:
    arv = GNC_Sim.WindAnglesToAirRelativeVelocityVector(angles)
    air_rel_body.append(arv)

air_rel_body = np.array(air_rel_body)

#calculating air relative velocity
air_rel_inertial = []

for n in range(len(time)):
    v_inertial = GNC_Sim.TransformFromBodyToInertial(
        air_rel_body[n,:],
        euler_angles[n,:]
    )
    air_rel_inertial.append(v_inertial)

air_rel_inertial = np.array(air_rel_inertial)

#Calculating inertial wind
wind_inertial = GNC_Sim.calculate_wind_inertial(aircraft_velocity_inertial, euler_angles, angular_rates,wind_angles)

Vw = []
Chi_W = []#For each element take the phase angle of the planar wind

sensitivities = []
for elements in wind_inertial:
    Vw.append(np.linalg.norm(elements[:2]))#Take the norm of the 1st and 2nd elements only
    Chi_W.append(np.atan2(elements[1],elements[0]))

Vw = np.array(Vw) #This is the wind speed in the planar direction
Chi_W = np.array(Chi_W)

fig1,ax1 = plt.subplots(3,1, figsize = (8,12))

mask = time <= 24

t_plot = time[mask]
Vw_plot = Vw[mask]
Chi_W_plot = Chi_W[mask]
Wz_plot = wind_inertial[mask,2]
#Problem 2.1
ax1[0].plot(t_plot, Vw_plot, label = r'$V_W$')
ax1[0].set_xlabel('Time')
ax1[0].set_ylabel('Planar Wind Speed')
ax1[0].legend()
ax1[1].plot(t_plot, Chi_W_plot, label = r'$\chi_W$')
ax1[1].set_xlabel('Time')
ax1[1].set_ylabel('Planar Wind Angle')
ax1[1].legend()
ax1[2].plot(t_plot, Wz_plot, label = r'$W_z$')
ax1[2].set_xlabel('Time')
ax1[2].set_ylabel('Planar Wind Vertical')
ax1[2].legend()
fig1.suptitle("Planar Wind Components Vs. Time")
plt.legend()
plt.tight_layout()
plt.show()

#--------------------------------------------------------------------
#Exam 2 Problem 2
#--------------------------------------------------------------------

def calc_planar_wind_sensitivity(euler_angles, wind_angles, v_inertial):

    Va_body = GNC_Sim.WindAnglesToAirRelativeVelocityVector(wind_angles)
    Va_inertial = GNC_Sim.TransformFromBodyToInertial(Va_body,euler_angles)

    wind = v_inertial - Va_inertial

    Vw = np.linalg.norm(wind[:2])
    Chi_w = np.atan2(wind[1],wind[0])
    Wz = wind[2]
    #need to return planar wind speed, planar wind direction and vertical wind measurements
    return np.array([Vw, Chi_w, Wz])

phi0 = 0
theta0 = np.deg2rad(4)
psi0 = 0

Va0 = 18
beta0 = np.deg2rad(-2)
alpha0 = np.deg2rad(6)

v_init = np.array([18,0,2])
pert = 0.1
phi_perturb = np.deg2rad(pert)#1 degree perturbation
theta_perturb = np.deg2rad(pert)#1 degree perturbation
psi_perturb = np.deg2rad(pert)#1 degree perturbation

Va_perturb = pert
beta_perturb = np.deg2rad(pert)#1 degree perturbation
alpha_perturb = np.deg2rad(pert)#1 degree perturbation

euler0 = np.array([phi0,theta0,psi0])
wind_angles0 = np.array([Va0,beta0,alpha0])

f0 = calc_planar_wind_sensitivity(euler0,wind_angles0, v_init)

euler0_dphi = np.array([phi0+phi_perturb,theta0,psi0])

euler0_dtheta = np.array([phi0,theta0+theta_perturb,psi0])

euler0_dpsi = np.array([phi0,theta0,psi0+psi_perturb])

wind_angles0_dVa = np.array([Va0+Va_perturb,beta0,alpha0])

wind_angles0_dbeta = np.array([Va0,beta0+beta_perturb,alpha0])

wind_angles0_dalpha = np.array([Va0,beta0,alpha0+alpha_perturb])
#Calculate Sensitivity to change in Phi
d_phi = calc_planar_wind_sensitivity(euler0_dphi,wind_angles0,v_init)

Phi_sensitivity = (d_phi-f0) / phi_perturb
print("="*30)
print("Problem 2.2")
print(f"The System sensitivity to Phi is: {Phi_sensitivity}")
#Calculate Sensitivity to change in Theta
d_theta = calc_planar_wind_sensitivity(euler0_dtheta,wind_angles0,v_init)
Theta_sensitivity = (d_theta-f0) / theta_perturb
print(f"The System sensitivity to Theta is: {Theta_sensitivity}")
#Calculate Sensitivity to change in Psi
d_psi = calc_planar_wind_sensitivity(euler0_dpsi,wind_angles0,v_init)
Psi_sensitivity = (d_psi-f0) / psi_perturb
print(f"The System sensitivity to Psi is: {Psi_sensitivity}")
#Calculate Sensitivity to change in Va
d_Va = calc_planar_wind_sensitivity(euler0,wind_angles0_dVa,v_init)
Va_sensitivity = (d_Va - f0)  / Va_perturb
print(f"The System sensitivity to Va is: {Va_sensitivity}")
#Calculate Sensitivity to change in Beta
d_beta = calc_planar_wind_sensitivity(euler0,wind_angles0_dbeta,v_init)
Beta_sensitivity = (d_beta -f0)  / beta_perturb
print(f"The System sensitivity to Beta is: {Beta_sensitivity}")
#Calculate Sensitivity to change in Alpha
d_alpha = calc_planar_wind_sensitivity(euler0,wind_angles0_dalpha,v_init)
Alpha_sensitivity = (d_alpha -f0 )  / alpha_perturb
print(f"The System sensitivity to Alpha is: {Alpha_sensitivity}")




#Problem 2.3 

def compute_all_sensitivities(euler, wind_angles, v_inertial):
    
    d_angle = np.deg2rad(0.1)
    d_Va = 0.1

    phi, theta, psi = euler
    Va, beta, alpha = wind_angles

    f0 = calc_planar_wind_sensitivity(euler, wind_angles, v_inertial)

    sensitivities = []

    # --- φ ---
    f = calc_planar_wind_sensitivity(
        np.array([phi + d_angle, theta, psi]),
        wind_angles,
        v_inertial
    )
    sensitivities.append((np.array(f) - np.array(f0)) / d_angle)

    # --- θ ---
    f = calc_planar_wind_sensitivity(
        np.array([phi, theta + d_angle, psi]),
        wind_angles,
        v_inertial
    )
    sensitivities.append((np.array(f) - np.array(f0)) / d_angle)

    # --- ψ ---
    f = calc_planar_wind_sensitivity(
        np.array([phi, theta, psi + d_angle]),
        wind_angles,
        v_inertial
    )
    sensitivities.append((np.array(f) - np.array(f0)) / d_angle)

    # --- Va ---
    f = calc_planar_wind_sensitivity(
        euler,
        np.array([Va + d_Va, beta, alpha]),
        v_inertial
    )
    sensitivities.append((np.array(f) - np.array(f0)) / d_Va)

    # --- β ---
    f = calc_planar_wind_sensitivity(
        euler,
        np.array([Va, beta + d_angle, alpha]),
        v_inertial
    )
    sensitivities.append((np.array(f) - np.array(f0)) / d_angle)

    # --- α ---
    f = calc_planar_wind_sensitivity(
        euler,
        np.array([Va, beta, alpha + d_angle]),
        v_inertial
    )
    sensitivities.append((np.array(f) - np.array(f0)) / d_angle)

    return np.array(sensitivities)  # shape (6,3)

for i in range(len(euler_angles)):
    v_inertial = aircraft_velocity_inertial[i,:]
    sensitivities.append(compute_all_sensitivities(euler=euler_angles[i],wind_angles=wind_angles[i], v_inertial=v_inertial))
all_sensitivities = np.array(sensitivities)
all_sensitivities = all_sensitivities[mask]

mean = np.mean(all_sensitivities, axis=0)
std  = np.std(all_sensitivities, axis=0)

vars = ["phi", "theta", "psi", "Va", "beta", "alpha"]
outputs = ["Vw", "Chi", "Wz"]

print("="*60)
print("Problem 2.3")

print("\nMean Sensitivities:")
for i, var in enumerate(vars):
    print(f"{var}: {dict(zip(outputs, mean[i]))}")

print("\nStd Dev Sensitivities:")
for i, var in enumerate(vars):
    print(f"{var}: {dict(zip(outputs, std[i]))}")
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from GNC_Sim import TransformFromBodyToInertial, TransformFromInertialToBody, WindAnglesToAirRelativeVelocityVector,AirRelativeVelocityVectorToWindAngles

#----------------------------------------------
#Problem 1
#----------------------------------------------
euler = np.array([6,9,-75])
aircraft_vel = np.array([21,-1,4])
euler_rads = np.deg2rad(euler)
W_E = np.array([0,6,0])

W_E_B = TransformFromInertialToBody(W_E,euler_rads)

print(f" The wind inertial converted to body coordinates is {W_E_B}")

air_relative_velocity = aircraft_vel - W_E_B

wind_angles = AirRelativeVelocityVectorToWindAngles(air_relative_velocity)

print(f"The wind angles are: {wind_angles}")


W_E_B_faulty = np.array([12.2,-0.2,2.3])

faulty_aire_rel_vel = aircraft_vel - W_E_B_faulty

wind_angles_faulty = AirRelativeVelocityVectorToWindAngles(faulty_aire_rel_vel)

print(f"The Wind Angles from the faulty sensor are: {wind_angles_faulty}")
#----------------------------------------------
#Problem 2
#----------------------------------------------

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


#----------------------
#Problem 2.1
#----------------------

air_rel_body = []

for angles in wind_angles:
    arv = WindAnglesToAirRelativeVelocityVector(angles)
    air_rel_body.append(arv)

air_rel_body = np.array(air_rel_body)
fig, axs = plt.subplots(3, 1, figsize=(12,8))

axs[0].plot(time, air_rel_body[:,0])
axs[0].set_ylabel("u (body) [m\s]")

axs[1].plot(time, air_rel_body[:,1])
axs[1].set_ylabel("v (body) [m\s]")

axs[2].plot(time, air_rel_body[:,2])
axs[2].set_ylabel("w (body) [m\s]")

plt.xlabel("Time (min)")
plt.suptitle("Air-Relative Velocity (Body Frame)")
#plt.show()


#----------------------
#Problem 2.2
#----------------------
air_rel_inertial = []

for n in range(len(time)):
    v_inertial = TransformFromBodyToInertial(
        air_rel_body[n,:],
        euler_angles[n,:]
    )
    air_rel_inertial.append(v_inertial)

air_rel_inertial = np.array(air_rel_inertial)
fig, axs = plt.subplots(3, 1, figsize=(12,8))

axs[0].plot(time, air_rel_inertial[:,0])
axs[0].set_ylabel("Vx [m\s]")

axs[1].plot(time, air_rel_inertial[:,1])
axs[1].set_ylabel("Vy [m\s]")

axs[2].plot(time, air_rel_inertial[:,2])
axs[2].set_ylabel("Vz m\s")

plt.xlabel("Time (min)")
plt.suptitle("Air-Relative Velocity (Inertial Frame)")
#plt.show()
#----------------------
#Problem 2.3
#----------------------
wind_inertial = calculate_wind_inertial(aircraft_velocity_inertial, euler_angles, angular_rates,wind_angles)
fig, axs = plt.subplots(3, 1, figsize=(12,8))

axs[0].plot(time, wind_inertial[:,0])
axs[0].set_ylabel("Wx [m\s]")

axs[1].plot(time, wind_inertial[:,1])
axs[1].set_ylabel("Wy [m\s]")

axs[2].plot(time, wind_inertial[:,2])
axs[2].set_ylabel("Wz [m\s]")

plt.suptitle("Inertial Wind Vector")
plt.xlabel("Time (min)")
#plt.show()


#-------------------------------
#Problem 2.4
#-------------------------------

idx = np.arange(0, len(time), 120)

from matplotlib.collections import LineCollection

# Create line segments
points = np.array([lon, lat]).T.reshape(-1,1,2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Create colored line
lc = LineCollection(segments, cmap='plasma')
#lc.set_array(t_norm)
lc.set_array(time)
lc.set_linewidth(2)

fig, ax = plt.subplots(figsize=(12,8))
ax.add_collection(lc)

ax.set_xlim(lon.min()-0.002, lon.max()+0.002)
ax.set_ylim(lat.min()-0.002, lat.max()+0.002)

cbar = plt.colorbar(lc, ax=ax)
cbar.set_label("Time [min]")
ax.set_xlabel('longitude')
ax.set_ylabel('latitude')

ax.quiver(
    lon[idx],
    lat[idx],
    wind_inertial[idx,0],
    wind_inertial[idx,1],
    scale=1000,  # tune this
    color="deepskyblue",
    width = 0.002
)

ax.set_title("Flight Path Colored by Time")
ax.set_aspect('equal')

plt.show()


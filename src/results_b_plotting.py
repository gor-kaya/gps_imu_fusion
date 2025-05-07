import numpy as np
import matplotlib.pyplot as plt
import math

from kalman_filters import ExtendedKalmanFilter as EKF
from utils import lla_to_enu, normalize_angles

from pathlib import Path

# USEFUL FUNCTIONS

def ddmm_to_dec(coordinates):
    # Coordinate is a np.array [Lon, Lat, Alt] with:
    # Lon in DMM.MMMM format,
    # Lat in DDMM.MMMM format,
    # Alt in MSL
    # This functions returns [Lon, Lat, Alt] with Lat and Lon in DD.DDDDDD format,
    # Alt is kept the same.
    decimal_coordinates = []
    for point in coordinates:
        decimal_coordinates.append([
            float(point[0][slice(0, 1)])+float(point[0][slice(2,9)])/60,
            (float(point[1][slice(0, 2)])+float(point[1][slice(2,9)])/60),
            float(point[2])
        ])
    return(decimal_coordinates)

def get_sec(time_str):
    """Get seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)

def data_unpacking(pond_data_dir):
    observed_trajectory_lla = [] # [longitude(dec), latitude(dec), altitude(MSL)] x N
    observed_yaws = [] # [yaw_angle(rad)] x N
    observed_yaw_rates = [] # [yaw_rate(rad/s)] x N
    observed_forward_velocities = [] # [forward_velocity(m/s)] x N
    timestamps = [] # [HH:MM:SS.SS]
    
    with open(pond_data_dir):
        pond_data = np.genfromtxt(pond_data_dir, dtype=str, 
                                encoding=None, delimiter=",")
    
    for line in pond_data:
        # Here the observed trajectory consists of strings:
        observed_trajectory_lla.append([
            line[2][:-1].strip(),
            line[1][:-1].strip(),
            line[3].strip()
        ])
        
        # observed velocity is taken as SOG from GPS (given in knots, converted to m/s)
        observed_forward_velocities.append(0.5144*float(line[4]))
        # Assuming that euler x is what we need, converted to rad
        observed_yaws.append(np.deg2rad(float(line[6])))
        
        # Assuming that gyro z is what we need, already in rad/s
        observed_yaw_rates.append(float(line[11]))
        
        timestamps.append(get_sec(line[0]))
    
    # Converting DDMM.MMMM trajectory to decimal
    observed_trajectory_lla = np.array(ddmm_to_dec(observed_trajectory_lla)).T
    observed_yaws = np.array(observed_yaws)
    observed_yaw_rates = np.array(observed_yaw_rates)
    observed_forward_velocities = np.array(observed_forward_velocities)
    timestamps = np.array(timestamps)
    unpacked_data = (observed_trajectory_lla, observed_yaws, observed_yaw_rates, observed_forward_velocities, timestamps)
    return unpacked_data

def filtering(raw_data):
    observed_trajectory_lla, observed_yaws, observed_yaw_rates, observed_forward_velocities, timestamps = raw_data

    elapsed = np.array(timestamps) - timestamps[0]
    N = len(elapsed)

    ###### KALMAN FILTER STARTS HERE ######
    xy_obs_noise_std = 2.5  # standard deviation of observation noise of x and y in meter
    forward_velocity_noise_std = 0.3 # standard deviation of forward velocity in m/s
    yaw_rate_noise_std = 0.8 # standard deviation of yaw rate in rad/s
    
    
    # initial state x_0 (first 2d position is initialised with the first GPS observation
    # yaw estimation is random)
    initial_yaw_std = np.pi
    initial_yaw = observed_yaws[0] + np.random.normal(0, initial_yaw_std)

    origin = observed_trajectory_lla[:, 0]
    observed_trajectory_xyz = lla_to_enu(observed_trajectory_lla, origin)

    x = np.array([
        observed_trajectory_xyz[0, 0],
        observed_trajectory_xyz[1, 0],
        initial_yaw
    ])

    # covariance for initial state estimation error (Sigma_0)
    P = np.array([
        [xy_obs_noise_std ** 2., 0., 0.],
        [0., xy_obs_noise_std ** 2., 0.],
        [0., 0., initial_yaw_std ** 2.]
    ])

    # Measurement error covariance Q
    Q = np.array([
        [xy_obs_noise_std ** 2., 0.],
        [0., xy_obs_noise_std ** 2.]
    ])

    # State transition noise covariance R
    R = np.array([
        [forward_velocity_noise_std ** 2., 0., 0.],
        [0., forward_velocity_noise_std ** 2., 0.],
        [0., 0., yaw_rate_noise_std ** 2.]
    ])

    # initialize Kalman filter
    kf = EKF(x, P)

    # array to store estimated 2d pose [x, y, theta]
    mu_x = [x[0],]
    mu_y = [x[1],]
    mu_theta = [x[2],]

    # array to store estimated error variance of 2d pose
    var_x = [P[0, 0],]
    var_y = [P[1, 1],]
    var_theta = [P[2, 2],]

    t_last = 0.
    for t_idx in range(1, N):
        t = elapsed[t_idx]
        dt = t - t_last
        
        if observed_yaw_rates[t_idx] == 0:
            observed_yaw_rates[t_idx] = 0.01
            
        # get control input `u = [v, omega] + noise`
        u = np.array([
            observed_forward_velocities[t_idx],
            observed_yaw_rates[t_idx]
        ])
        
        # because velocity and yaw rate are multiplied with `dt` in state transition function,
        # its noise covariance must be multiplied with `dt**2.`
        R_ = R * (dt ** 2.)
        
        # propagate!
        kf.propagate(u, dt, R)
        
        # get measurement `z = [x, y] + noise`
        z = np.array([
            observed_trajectory_xyz[0, t_idx],
            observed_trajectory_xyz[1, t_idx]
        ])
        
        # update!
        kf.update(z, Q)
        
        # save estimated state to analyze later
        mu_x.append(kf.x[0])
        mu_y.append(kf.x[1])
        mu_theta.append(normalize_angles(kf.x[2]))
        
        # save estimated variance to analyze later
        var_x.append(kf.P[0, 0])
        var_y.append(kf.P[1, 1])
        var_theta.append(kf.P[2, 2])
        
        t_last = t
        

    mu_x = np.array(mu_x)
    mu_y = np.array(mu_y)
    mu_theta = np.array(mu_theta)

    var_x = np.array(var_x)
    var_y = np.array(var_y)
    var_theta = np.array(var_theta)

    return(mu_x, mu_y, mu_theta)


def main():
    data_dir = Path(r"path_data")
    pond_data_files = [
        data_dir / "Scenario_B1.txt",
        data_dir / "Scenario_B2.txt",
        data_dir / "Scenario_B3.txt",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, pond_data_file in enumerate(pond_data_files):
        unpacked_data = data_unpacking(str(pond_data_file))
        observed_trajectory_lla, observed_yaws, observed_yaw_rates, observed_forward_velocities, timestamps = unpacked_data

        mu_x, mu_y, mu_theta = filtering(unpacked_data)

        origin = observed_trajectory_lla[:, 0]
        observed_trajectory_xyz = lla_to_enu(observed_trajectory_lla, origin)

        xs, ys, _ = observed_trajectory_xyz

        ax = axes[idx]
        ax.plot(xs, ys, lw=0, marker='.', markersize=4, alpha=1., label='observed trajectory')
        ax.plot(mu_x, mu_y, lw=2, label='estimated [EKF] trajectory', color='r')
        ax.set_title(f'Scenario B{idx + 1}', fontsize=16)
        ax.set_xlabel('X [m]', fontsize=14)
        ax.set_ylabel('Y [m]', fontsize=14)
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=14)

    fig.subplots_adjust(wspace=0.3, top=0.85)  # Controls spacing between subplots and top legend
    fig.savefig("ekf_trajectories.svg", format='svg', bbox_inches='tight')
    fig.savefig("ekf_trajectories.png", format='png', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
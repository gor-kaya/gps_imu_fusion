import numpy as np
import matplotlib.pyplot as plt
import math
import pykitti

from kalman_filters import ExtendedKalmanFilter as EKF
from utils import lla_to_enu, normalize_angles

from pathlib import Path


def data_unpacking(dataset):

    gt_trajectory_lla = []  # [longitude(deg), latitude(deg), altitude(meter)] x N
    gt_yaws = []  # [yaw_angle(rad),] x N
    gt_yaw_rates= []  # [vehicle_yaw_rate(rad/s),] x N
    gt_forward_velocities = []  # [vehicle_forward_velocity(m/s),] x N

    for oxts_data in dataset.oxts:
        packet = oxts_data.packet
        gt_trajectory_lla.append([
            packet.lon,
            packet.lat,
            packet.alt
            ])
        gt_yaws.append(packet.yaw)
        gt_yaw_rates.append(packet.wz)
        gt_forward_velocities.append(packet.vf)

    gt_trajectory_lla = np.array(gt_trajectory_lla).T
    gt_yaws = np.array(gt_yaws)
    gt_yaw_rates = np.array(gt_yaw_rates)
    gt_forward_velocities = np.array(gt_forward_velocities)
    timestamps = np.array(dataset.timestamps)
    unpacked_data = (gt_trajectory_lla, gt_yaws, gt_yaw_rates, gt_forward_velocities, timestamps)
    return unpacked_data

def filtering(raw_data):
    gt_trajectory_lla, gt_yaws, gt_yaw_rates, gt_forward_velocities, timestamps = raw_data

    elapsed = np.array(timestamps) - timestamps[0]
    ts = [t.total_seconds() for t in elapsed]
    N = len(ts)

    origin = gt_trajectory_lla[:, 0]  # set the initial position to the origin
    gt_trajectory_xyz = lla_to_enu(gt_trajectory_lla, origin)

    ###### KALMAN FILTER STARTS HERE ######
    xy_obs_noise_std = 5.0  # standard deviation of observation noise of x and y in meter
    forward_velocity_noise_std = 0.3 # standard deviation of forward velocity in m/s
    yaw_rate_noise_std = 0.02 # standard deviation of yaw rate in rad/s
    
    ### NOISE ###
    xy_obs_noise = np.random.normal(0.0, xy_obs_noise_std, (2, N))  # gen gaussian noise
    obs_trajectory_xyz = gt_trajectory_xyz.copy()
    obs_trajectory_xyz[:2, :] += xy_obs_noise  # add the noise to ground-truth positions

    yaw_rate_noise = np.random.normal(0.0, yaw_rate_noise_std, (N,))  # gen gaussian noise
    obs_yaw_rates = gt_yaw_rates.copy()
    obs_yaw_rates += yaw_rate_noise  # add the noise to ground-truth positions

    forward_velocity_noise = np.random.normal(0.0, forward_velocity_noise_std, (N,))  # gen gaussian noise
    obs_forward_velocities = gt_forward_velocities.copy()
    obs_forward_velocities += forward_velocity_noise  # add the noise to ground-truth positions

    
    # initial state x_0 (first 2d position is initialised with the first GPS observation
    # yaw estimation is random)
    initial_yaw_std = np.pi
    initial_yaw = gt_yaws[0] + np.random.normal(0, initial_yaw_std)


    x = np.array([
        obs_trajectory_xyz[0, 0],
        obs_trajectory_xyz[1, 0],
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
        t = ts[t_idx]
        dt = t - t_last
        
        if obs_yaw_rates[t_idx] == 0:
            obs_yaw_rates[t_idx] = 0.01
            
        # get control input `u = [v, omega] + noise`
        u = np.array([
            obs_forward_velocities[t_idx],
            obs_yaw_rates[t_idx]
        ])
        
        # because velocity and yaw rate are multiplied with `dt` in state transition function,
        # its noise covariance must be multiplied with `dt**2.`
        R_ = R * (dt ** 2.)
        
        # propagate!
        kf.propagate(u, dt, R)
        
        # get measurement `z = [x, y] + noise`
        z = np.array([
            obs_trajectory_xyz[0, t_idx],
            obs_trajectory_xyz[1, t_idx]
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

    true_state = gt_trajectory_xyz
    observed_state = obs_trajectory_xyz
    estimated_state = (mu_x, mu_y, mu_theta)
    return(true_state, observed_state, estimated_state)


def main():
    kitti_root_dir = r'kitti_directory_location'
    kitti_date_1 = '2011_09_26'
    kitti_drive_1 = '0009'

    kitti_date_2 = '2011_09_26'
    kitti_drive_2 = '0093'

    dataset_1 = pykitti.raw(kitti_root_dir, kitti_date_1, kitti_drive_1)
    dataset_2 = pykitti.raw(kitti_root_dir, kitti_date_2, kitti_drive_2)

    datasets = [dataset_1, dataset_2]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for idx, (dataset, ax) in enumerate(zip(datasets, axes)):
        unpacked_data = data_unpacking(dataset)
        gt_trajectory_xyz, obs_trajectory_xyz, estimated_state = filtering(unpacked_data)
        mu_x, mu_y, mu_theta = estimated_state

        xs, ys, _ = gt_trajectory_xyz
        ax.plot(xs, ys, lw=2, label='ground-truth trajectory')
        xs, ys, _ = obs_trajectory_xyz
        ax.plot(xs, ys, lw=0, marker='.', markersize=4, alpha=1., label='observed trajectory')
        ax.plot(mu_x, mu_y, lw=2, label='estimated trajectory', color='r')

        if idx == 0:
            ax.set_title('Scenario A - Dataset 2', fontsize=16)

        if idx == 1:
            ax.set_title('Scenario A - Dataset 6', fontsize=16)
        ax.set_xlabel('X [m]', fontsize=14)
        ax.set_ylabel('Y [m]', fontsize=14)
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=14)

    fig.subplots_adjust(wspace=0.3, top=0.85)
    # fig.suptitle("KITTI Trajectories Comparison", fontsize=16)
    fig.savefig("kitti_trajectories.png", format='png', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
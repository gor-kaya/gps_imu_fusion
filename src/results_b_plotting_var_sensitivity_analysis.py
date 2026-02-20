import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from pathlib import Path

from kalman_filters import ExtendedKalmanFilter as EKF
from utils import lla_to_enu, normalize_angles, ddmm_to_dec, get_sec, data_unpacking, rmse
from utils.plots import plot_trajectory, plot_variances_xy, plot_variances_theta, plot_residuals


@dataclass
class FilterResult:
    mu_x: np.ndarray
    mu_y: np.ndarray
    mu_theta: np.ndarray
    var_x: np.ndarray
    var_y: np.ndarray
    var_theta: np.ndarray
    residuals: np.ndarray


@dataclass
class TuningParams:
    """Kalman filter tuning parameters."""
    xy_obs_noise_std: float  # standard deviation of observation noise of x and y in meter
    initial_yaw_std: float   # standard deviation of initial yaw estimation error in rad
    forward_velocity_noise_std: float  # standard deviation of forward velocity in m/s
    yaw_rate_noise_std: float  # standard deviation of yaw rate in rad/s

    def to_tuple(self) -> tuple[float, float, float, float]:
        """Convert to tuple format for backward compatibility."""
        return (self.xy_obs_noise_std, self.initial_yaw_std, 
                self.forward_velocity_noise_std, self.yaw_rate_noise_std)


BASE_TUNING = TuningParams(
    xy_obs_noise_std=2.5,
    initial_yaw_std=np.pi / 2,
    forward_velocity_noise_std=0.2,
    yaw_rate_noise_std=0.1
)

def filtering(raw_data, tuning_params=None, seed: int | None = None):
    observed_trajectory_lla, observed_yaws, observed_yaw_rates, observed_forward_velocities, timestamps = raw_data

    elapsed = np.array(timestamps) - timestamps[0]
    N = len(elapsed)

    ###### KALMAN FILTER STARTS HERE ######
    # Handle tuning_params: accept TuningParams dataclass or tuple, default to BASE_TUNING
    if tuning_params is None:
        tuning_params = BASE_TUNING
    
    # Convert TuningParams dataclass to tuple if needed
    if isinstance(tuning_params, TuningParams):
        xy_obs_noise_std, initial_yaw_std, forward_velocity_noise_std, yaw_rate_noise_std = tuning_params.to_tuple()
    else:
        # Assume tuple format for backward compatibility
        xy_obs_noise_std, initial_yaw_std, forward_velocity_noise_std, yaw_rate_noise_std = tuning_params
    
    # initial state x_0 (first 2d position is initialised with the first GPS observation
    # yaw estimation is random)
    if seed is not None:
        rng = np.random.default_rng(seed)
        initial_yaw = observed_yaws[0] + rng.normal(0.0, initial_yaw_std)
    else:
        initial_yaw = observed_yaws[0] + np.random.normal(0.0, initial_yaw_std)

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
    print("Initial P, Q, R:", P, Q, R)

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

    residuals = []
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
        # R_ = R * (dt ** 2.)
        
        # propagate!
        kf.propagate(u, dt, R)
        
        # get measurement `z = [x, y] + noise`
        z = np.array([
            observed_trajectory_xyz[0, t_idx],
            observed_trajectory_xyz[1, t_idx]
        ])
        
        # update!
        residual = kf.update(z, Q)
        residuals.append(residual)
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

    residuals = np.array(residuals)

    return FilterResult(
        mu_x=mu_x,
        mu_y=mu_y,
        mu_theta=mu_theta,
        var_x=var_x,
        var_y=var_y,
        var_theta=var_theta,
        residuals=residuals,
    )


def run_scenario(pond_data_file: Path, tuning_params: TuningParams = BASE_TUNING, seed: int | None = None) -> FilterResult:
    """
    Run filtering on a single scenario file with given tuning parameters.
    
    Args:
        pond_data_file: Path to the scenario data file
        tuning_params: Tuning parameters to use (defaults to BASE_TUNING)
        seed: Random seed for reproducibility (optional)
    
    Returns:
        FilterResult with filtering results
    """
    unpacked_data = data_unpacking(str(pond_data_file))
    return filtering(unpacked_data, tuning_params=tuning_params, seed=seed)


def main(run_sensitivity: bool = False):
    """
    Main function to run filtering and visualization.
    
    Args:
        run_sensitivity: If True, run sensitivity analysis instead of standard plotting
    """
    # Fixed seed for reproducibility
    SEED = 2
    
    data_dir = Path(r"C:\Users\isb20183\Documents\GitHub\gps_imu_fusion\data\pond_test")
    pond_data_files = [
        data_dir / "Scenario_B1.txt",
        data_dir / "Scenario_B2.txt",
        data_dir / "Scenario_B3.txt",
    ]

    if run_sensitivity:
        # TODO: Implement sensitivity analysis
        # run_sensitivity_analysis(pond_data_files, BASE_TUNING, seed=SEED)
        print("Sensitivity analysis not yet implemented")
        return

    # Standard plotting mode
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig_var, axes_var = plt.subplots(2, 3, figsize=(18, 10))
    fig_res, axes_res = plt.subplots(1, 3, figsize=(18, 6))


    for idx, pond_data_file in enumerate(pond_data_files):
        # Run filtering with baseline tuning
        result = run_scenario(pond_data_file, tuning_params=BASE_TUNING, seed=SEED)
        
        # Get observed trajectory for plotting
        unpacked_data = data_unpacking(str(pond_data_file))
        observed_trajectory_lla, _, _, _, _ = unpacked_data
        origin = observed_trajectory_lla[:, 0]
        observed_trajectory_xyz = lla_to_enu(observed_trajectory_lla, origin)

        xs, ys, _ = observed_trajectory_xyz
        scenario_label = f"Scenario B{idx + 1}"

        # Plot results
        plot_trajectory(axes[idx], xs, ys, result.mu_x, result.mu_y, scenario_label)
        plot_variances_xy(axes_var[0, idx], result.var_x, result.var_y, scenario_label)
        plot_variances_theta(axes_var[1, idx], result.var_theta, scenario_label)
        plot_residuals(axes_res[idx], result.residuals, scenario_label)
        
        # Calculate and print mean residuals
        mean_residual_x = np.mean(result.residuals[:, 0])
        mean_residual_y = np.mean(result.residuals[:, 1])
        
        print(f"Mean residuals for Scenario B{idx + 1}:")
        print(f"  Mean Residual X: {mean_residual_x:.4f} m")
        print(f"  Mean Residual Y: {mean_residual_y:.4f} m")
        
       

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=14)
    
    fig.subplots_adjust(wspace=0.3, top=0.85)  # Controls spacing between subplots and top legend
    fig.savefig("ekf_trajectories.svg", format='svg', bbox_inches='tight')
    fig.savefig("ekf_trajectories.png", format='png', bbox_inches='tight')
    
    fig_var.subplots_adjust(wspace=0.3, top=0.85)
    fig_var.savefig("ekf_variances.svg", format='svg', bbox_inches='tight')
    fig_var.savefig("ekf_variances.png", format='png', bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    # Set to True to run sensitivity analysis instead of standard plotting
    main(run_sensitivity=False)
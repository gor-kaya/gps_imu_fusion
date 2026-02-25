import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from pathlib import Path

from kalman_filters import ExtendedKalmanFilter as EKF
from utils import lla_to_enu, normalize_angles, ddmm_to_dec, get_sec, data_unpacking, rmsd, pct_diff_series
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

# Parameter sensitivity analysis boundaries
# Format: {param_name: (lower_bound, upper_bound, baseline_value)}
PARAM_BOUNDARIES = {
    'xy_obs_noise_std': (1.0, 7.5, BASE_TUNING.xy_obs_noise_std),
    'initial_yaw_std': (0.63, 1.5 * np.pi, BASE_TUNING.initial_yaw_std),
    'forward_velocity_noise_std': (0.08, 0.6, BASE_TUNING.forward_velocity_noise_std),
    'yaw_rate_noise_std': (0.04, 0.3, BASE_TUNING.yaw_rate_noise_std),
}

# Mapping of parameter names to their LaTeX notation for plotting
PARAM_LATEX_NAMES = {
    'xy_obs_noise_std': r'$\sigma_{x_0^1}^{2}=\sigma_{x_0^2}^{2}$',
    'initial_yaw_std': r'$\sigma_{x_0^3}^{2}$',
    'forward_velocity_noise_std': r'$\sigma_{u_0^1}^{2}$',
    'yaw_rate_noise_std': r'$\sigma_{u_0^2}^{2}$',
}

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


def run_sensitivity_analysis(pond_data_files: list[Path], seed: int | None = None):
    """
    Run sensitivity analysis on tuning parameters and generate CSV reports.
    
    Varies each parameter between its lower and upper boundaries and compares
    the resulting variances against the baseline tuning.
    
    Args:
        pond_data_files: List of paths to scenario data files
        seed: Random seed for reproducibility
    """
    import csv
    
    for scenario_idx, pond_data_file in enumerate(pond_data_files):
        scenario_name = f"Scenario_B{scenario_idx + 1}"
        print(f"\nRunning sensitivity analysis for {scenario_name}...")
        
        # Get baseline result
        baseline_result = run_scenario(pond_data_file, tuning_params=BASE_TUNING, seed=seed)
        
        # Prepare rows for CSV
        rows = []
        
        # Iterate through each parameter using global boundaries
        for param_name, (lower_bound, upper_bound, baseline_val) in PARAM_BOUNDARIES.items():
            for boundary_type, boundary_value in [('lower', lower_bound), ('upper', upper_bound)]:
                # Create perturbed tuning
                perturbed_tuning = TuningParams(
                    xy_obs_noise_std=BASE_TUNING.xy_obs_noise_std,
                    initial_yaw_std=BASE_TUNING.initial_yaw_std,
                    forward_velocity_noise_std=BASE_TUNING.forward_velocity_noise_std,
                    yaw_rate_noise_std=BASE_TUNING.yaw_rate_noise_std,
                )
                
                # Update the specific parameter
                if param_name == 'xy_obs_noise_std':
                    perturbed_tuning.xy_obs_noise_std = boundary_value
                elif param_name == 'initial_yaw_std':
                    perturbed_tuning.initial_yaw_std = boundary_value
                elif param_name == 'forward_velocity_noise_std':
                    perturbed_tuning.forward_velocity_noise_std = boundary_value
                elif param_name == 'yaw_rate_noise_std':
                    perturbed_tuning.yaw_rate_noise_std = boundary_value
                
                # Run filtering with perturbed tuning
                perturbed_result = run_scenario(pond_data_file, tuning_params=perturbed_tuning, seed=seed)
                
                # Calculate metrics for each variance type
                row = {
                    'Parameter': param_name,
                    'Boundary': boundary_type,
                }
                
                # For each variance type (var_x, var_y, var_theta)
                for var_type, var_name in [('var_x', 'x'), ('var_y', 'y'), ('var_theta', 'theta')]:
                    baseline_var = getattr(baseline_result, var_type)
                    perturbed_var = getattr(perturbed_result, var_type)
                    
                    # Calculate RMSD
                    rmsd_val = rmsd(baseline_var, perturbed_var)
                    
                    # Calculate percentage differences
                    pct_diffs = pct_diff_series(baseline_var, perturbed_var)
                    
                    # Filter out infinite values for max/min calculations
                    valid_pct_diffs = pct_diffs[np.isfinite(pct_diffs)]
                    
                    if len(valid_pct_diffs) > 0:
                        max_pct = np.max(valid_pct_diffs)
                        min_pct = np.min(valid_pct_diffs)
                    else:
                        max_pct = 0.0
                        min_pct = 0.0
                    
                    row[f'RMSD_var_{var_name}'] = rmsd_val
                    row[f'Max%_var_{var_name}'] = max_pct
                    row[f'Min%_var_{var_name}'] = min_pct
                
                rows.append(row)
        
        # Write results to CSV file
        csv_filename = f"sensitivity_analysis_{scenario_name}.csv"
        fieldnames = ['Parameter', 'Boundary', 
                      'RMSD_var_x', 'Max%_var_x', 'Min%_var_x',
                      'RMSD_var_y', 'Max%_var_y', 'Min%_var_y',
                      'RMSD_var_theta', 'Max%_var_theta', 'Min%_var_theta']
        
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"Results saved to {csv_filename}")
        
        # Print table to console
        print(f"\n{scenario_name} Sensitivity Analysis Results:")
        print("-" * 150)
        print(f"{'Parameter':<30} {'Boundary':<10} {'RMSD_x':<12} {'Max%_x':<12} {'Min%_x':<12} "
              f"{'RMSD_y':<12} {'Max%_y':<12} {'Min%_y':<12} {'RMSD_th':<12} {'Max%_th':<12} {'Min%_th':<12}")
        print("-" * 150)
        for row in rows:
            print(f"{row['Parameter']:<30} {row['Boundary']:<10} "
                  f"{row['RMSD_var_x']:<12.4f} {row['Max%_var_x']:<12.2f} {row['Min%_var_x']:<12.2f} "
                  f"{row['RMSD_var_y']:<12.4f} {row['Max%_var_y']:<12.2f} {row['Min%_var_y']:<12.2f} "
                  f"{row['RMSD_var_theta']:<12.4f} {row['Max%_var_theta']:<12.2f} {row['Min%_var_theta']:<12.2f}")


def create_tornado_graphs(scenarios: list[str] = None):
    """
    Create consolidated tornado graphs from sensitivity analysis CSV results.
    
    Generates one figure per scenario with a 3x3 grid showing all metrics and variance types.
    Each row represents a variance type (var_x, var_y, var_theta).
    Each column represents a metric (RMSD, Max%, Min%).
    
    Args:
        scenarios: List of scenario names (e.g., ['Scenario_B1', 'Scenario_B2', 'Scenario_B3'])
                   If None, uses default scenarios
    """
    if scenarios is None:
        scenarios = ['Scenario_B1', 'Scenario_B2', 'Scenario_B3']
    
    # Metrics to create tornado graphs for
    metrics = {
        'RMSD': ('RMSD_{var_type}', 'RMSD'),
        'Max% dif': ('Max%_{var_type}', 'Max % dif'),
        'Min% dif': ('Min%_{var_type}', 'Min % dif'),
    }
    
    variance_types = ['var_x', 'var_y', 'var_theta']
    
    for scenario_name in scenarios:
        csv_filename = f"sensitivity_analysis_{scenario_name}.csv"
        
        # Read the CSV file
        df = pd.read_csv(csv_filename)
        
        # Create one large figure with 3x3 grid (rows=variance types, columns=metrics)
        fig, axes = plt.subplots(3, 3, figsize=(20, 14))
        fig.suptitle(f'{scenario_name} - Sensitivity Analysis', fontsize=18)
        
        # Iterate through metrics (columns) and variance types (rows)
        for col_idx, (metric_name, (col_pattern, metric_label)) in enumerate(metrics.items()):
            for row_idx, var_type in enumerate(variance_types):
                ax = axes[row_idx, col_idx]
                
                # Extract parameters and their impact ranges
                parameters = []
                lower_impacts = []
                upper_impacts = []
                
                for param_name in PARAM_BOUNDARIES.keys():
                    param_data = df[df['Parameter'] == param_name]
                    lower_row = param_data[param_data['Boundary'] == 'lower'].iloc[0]
                    upper_row = param_data[param_data['Boundary'] == 'upper'].iloc[0]
                    
                    # Get metric values for this variance type
                    metric_col = col_pattern.format(var_type=var_type)
                    lower_value = lower_row[metric_col]
                    upper_value = upper_row[metric_col]
                    
                    lower_impacts.append(lower_value)
                    upper_impacts.append(upper_value)
                    
                    # Use LaTeX parameter names for labels
                    param_short = PARAM_LATEX_NAMES[param_name]
                    parameters.append(param_short)
                
                # Calculate impact ranges (upper impact - lower impact)
                impact_ranges = np.array(upper_impacts) - np.array(lower_impacts)
                
                # Sort by magnitude of impact
                sorted_indices = np.argsort(np.abs(impact_ranges))[::-1]
                sorted_parameters = [parameters[i] for i in sorted_indices]
                sorted_lower = [lower_impacts[i] for i in sorted_indices]
                sorted_upper = [upper_impacts[i] for i in sorted_indices]
                
                # Create tornado plot
                y_pos = np.arange(len(sorted_parameters))
                
                # Plot bars for lower and upper bounds (all positive values)
                left_values = np.array(sorted_lower)
                right_values = np.array(sorted_upper)
                
                # Create offset positions for side-by-side bars
                bar_height = 0.35
                offset = bar_height / 2
                
                # Plot lower bounds (red)
                ax.barh(y_pos - offset, left_values, bar_height, color='#e74c3c', alpha=0.75, label='Lower Bound' if col_idx == 0 and row_idx == 0 else '')
                
                # Plot upper bounds (green)
                ax.barh(y_pos + offset, right_values, bar_height, color='#2ecc71', alpha=0.75, label='Upper Bound' if col_idx == 0 and row_idx == 0 else '')
                
                # Customize plot
                ax.set_yticks(y_pos)
                ax.set_yticklabels(sorted_parameters, fontsize=11)
                
                # Set xlabel with units for RMSD based on variance type and metric
                if metric_name == 'RMSD':
                    if var_type == 'var_theta':
                        xlabel = f'RMSD [$rad^2$] of {var_type}'
                    else:  # var_x or var_y
                        xlabel = f'RMSD [$m^2$] of {var_type}'
                else:
                    xlabel = f'{metric_label} of {var_type}'
                
                ax.set_xlabel(xlabel, fontsize=10)
                
                ax.axvline(0, color='black', linestyle='-', linewidth=0.8)
                ax.grid(axis='x', alpha=0.3, linewidth=0.5)
        
        # Add shared legend for all subplots
        handles = [plt.Rectangle((0, 0), 1, 1, fc='#e74c3c', alpha=0.75),
                   plt.Rectangle((0, 0), 1, 1, fc='#2ecc71', alpha=0.75)]
        labels = ['Lower Bound', 'Upper Bound']
        fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.05, 1.0),
                  ncol=2, fontsize=12, frameon=True, fancybox=True, shadow=True)
        
        plt.tight_layout(rect=[0.04, 0, 1, 1])
        plot_filename = f"tornado_graph_{scenario_name}_consolidated.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Consolidated tornado graph saved: {plot_filename}")
        plt.close()  # Close to free memory




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
        run_sensitivity_analysis(pond_data_files, seed=SEED)
        
        # Generate tornado graphs from the CSV results
        print("\nGenerating tornado graphs...")
        create_tornado_graphs()
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
    main(run_sensitivity=True)
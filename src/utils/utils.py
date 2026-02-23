import numpy as np

def normalize_angles(angles):
    """
    Args:
        angles (float or numpy.array): angles in radian (= [a1, a2, ...], shape of [n,])
    Returns:
        numpy.array or float: angles in radians normalized b/w/ -pi and +pi (same shape w/ angles)
    """
    angles = (angles + np.pi) % (2 * np.pi ) - np.pi
    return angles

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
        observed_yaws.append(np.deg2rad(90 - float(line[6])))
        
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

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate Root Mean Square Error between two arrays.
    
    Args:
        a: First array (baseline variances)
        b: Second array (perturbed variances)
    
    Returns:
        RMSE value
    """
    if len(a) != len(b):
        raise ValueError(f"Arrays must have the same length: {len(a)} vs {len(b)}")
    if len(a) == 0:
        return 0.0
    return np.sqrt(np.mean((a - b) ** 2))


def pct_diff_series(baseline: np.ndarray, new: np.ndarray) -> np.ndarray:
    """
    Compute relative change (%) at each element:
        (new - baseline) / baseline * 100

    Args:
        baseline: Baseline array (e.g., variances from baseline tuning)
        new: Array from new tuning (same shape as baseline)

    Returns:
        NumPy array of relative changes in percent, same shape as inputs.
    """
    if baseline.shape != new.shape:
        raise ValueError(f"Shapes must match: {baseline.shape} vs {new.shape}")

    baseline = baseline.astype(float)
    new = new.astype(float)

    out = np.empty_like(baseline, dtype=float)
    mask = baseline != 0.0

    # Where baseline is nonzero, compute standard relative change
    out[mask] = 100.0 * (new[mask] - baseline[mask]) / baseline[mask]

    # Where baseline is zero:
    # - if new is also zero, define change as 0%
    # - otherwise, mark as inf to indicate undefined/infinite relative change
    zero_mask = ~mask
    both_zero = zero_mask & (new == 0.0)
    out[both_zero] = 0.0
    out[zero_mask & ~both_zero] = float("inf")

    return out
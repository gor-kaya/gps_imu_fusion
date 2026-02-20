import matplotlib.pyplot as plt
def plot_variances(var_x, var_y, var_theta, title="State Estimate Variances"):
    plt.figure(figsize=(10, 6))
    plt.plot(var_x, label='Variance X')
    plt.plot(var_y, label='Variance Y')
    plt.plot(var_theta, label='Variance Theta')
    plt.xlabel('Time Step')
    plt.ylabel('Variance')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_trajectory(ax, xs, ys, mu_x, mu_y, scenario_label: str) -> None:
    ax.plot(xs, ys, lw=0, marker='.', markersize=4, alpha=1.0, label='observed trajectory')
    ax.plot(mu_x, mu_y, lw=2, label='estimated [EKF] trajectory', color='r')
    ax.set_title(scenario_label, fontsize=16)
    ax.set_xlabel('X [m]', fontsize=14)
    ax.set_ylabel('Y [m]', fontsize=14)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid()


def plot_variances_xy(ax, var_x, var_y, scenario_label: str) -> None:
    ax.plot(var_x, label='Variance X')
    ax.plot(var_y, label='Variance Y')
    ax.set_title(f"{scenario_label} - X/Y Variance")
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Variance (m²)')
    ax.legend()
    ax.grid()
    ax.set_xlim([0, len(var_x)])


def plot_variances_theta(ax, var_theta, scenario_label: str) -> None:
    ax.plot(var_theta, label='Variance Theta', color='g')
    ax.set_title(f"{scenario_label} - Theta Variance")
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Variance (rad²)')
    ax.legend()
    ax.grid()
    ax.set_xlim([0, len(var_theta)])


def plot_residuals(ax, residuals, scenario_label: str) -> None:
    ax.plot(residuals[:, 0], label='Residual X')
    ax.plot(residuals[:, 1], label='Residual Y')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Residual')
    ax.set_title(f"{scenario_label} - residuals")
    ax.legend()
    ax.grid()
    ax.set_xlim([0, len(residuals)])

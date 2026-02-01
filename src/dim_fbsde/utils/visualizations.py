"""
Visualization utilities for FBSDE solutions.

GPU-accelerated plotting functions that handle tensor computations efficiently.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import logging

logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['axes.grid'] = False


def plot_pathwise_comparison(solution, analytical_Y_func, analytical_Z_func,
                             path_indices=None, component_idx=0, device='cpu',
                             analytical_Y_kwargs=None, analytical_Z_kwargs=None):
    """
    Plots a path-by-path comparison of the numerical BSDE solution against an analytical solution.

    GPU-accelerated: Converts numpy arrays to tensors, computes on device, then converts to numpy for plotting.

    Args:
        solution (dict): The dictionary returned by the solver's solve() method (contains numpy arrays).
        analytical_Y_func (function): A function for the analytical Y path that accepts tensors.
                                      Signature: (t_tensor, X_tensor, **kwargs) -> Y_tensor
        analytical_Z_func (function): A function for the analytical Z path that accepts tensors.
                                      Signature: (t_tensor, X_tensor, **kwargs) -> Z_tensor
        path_indices (list or np.ndarray, optional): Specific path indices to plot.
        component_idx (int, optional): The index of the Z component to plot. Defaults to 0.
        device (str or torch.device, optional): Device for tensor computations ('cpu' or 'cuda'). Defaults to 'cpu'.
        analytical_Y_kwargs (dict, optional): Extra keyword arguments for analytical_Y_func.
        analytical_Z_kwargs (dict, optional): Extra keyword arguments for analytical_Z_func.

    Returns:
        matplotlib.figure.Figure: The top-level figure container.
        numpy.ndarray of matplotlib.axes.Axes: An array of the two subplot axes.

    Example:
        >>> solution = solver.solve()
        >>> fig, axes = plot_pathwise_comparison(
        ...     solution=solution,
        ...     analytical_Y_func=lambda t, x, **kw: equation.analytical_y(t, x, **kw),
        ...     analytical_Z_func=lambda t, x, **kw: equation.analytical_z(t, x, **kw),
        ...     device='cuda' if torch.cuda.is_available() else 'cpu',
        ...     analytical_Y_kwargs={'T_terminal': 1.0}
        ... )
        >>> plt.show()
    """
    # --- Handle optional kwargs ---
    if analytical_Y_kwargs is None:
        analytical_Y_kwargs = {}
    if analytical_Z_kwargs is None:
        analytical_Z_kwargs = {}

    device = torch.device(device)

    # --- Extract numpy arrays from solution ---
    time_grid_np = solution['time']
    X_paths_np = solution['X']
    Y_paths_np = solution['Y']
    Z_paths_np = solution['Z']

    # --- Convert to tensors for GPU acceleration ---
    time_grid = torch.tensor(time_grid_np, dtype=torch.float32, device=device)
    X_paths = torch.tensor(X_paths_np, dtype=torch.float32, device=device)

    # --- Select paths to plot ---
    if path_indices is None:
        num_to_plot = min(5, X_paths.shape[0])
        path_indices = np.random.choice(X_paths.shape[0], size=num_to_plot, replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Plotting Loop ---
    for i, path_idx in enumerate(path_indices):
        # Get single path as tensor
        X_path_single = X_paths[path_idx]  # Shape: [N+1, dim_x]

        # -- Compute Analytical Y (on GPU) --
        # Analytical functions expect (t_scalar, x_batch), so we call them for each time step
        Y_analytical_list = []
        with torch.no_grad():
            for t_idx in range(len(time_grid)):
                t_val = time_grid[t_idx]
                x_val = X_path_single[t_idx:t_idx+1, :]  # Shape: [1, dim_x]
                y_val = analytical_Y_func(t_val, x_val, **analytical_Y_kwargs)
                Y_analytical_list.append(y_val.squeeze())
        Y_analytical = torch.stack(Y_analytical_list).cpu().numpy()

        # Get numerical Y from numpy array
        Y_approx = Y_paths_np[path_idx].squeeze()

        # Plot Y
        label_approx = "Deep Picard Approx." if i == 0 else None
        label_analytical = "Analytical" if i == 0 else None

        axes[0].plot(time_grid_np, Y_analytical, 'b', linewidth=1, label=label_analytical)
        axes[0].plot(time_grid_np, Y_approx, 'r--', linewidth=1, label=label_approx)

        # -- Compute Analytical Z (on GPU) --
        # Z is computed at time steps 0 to N-1 (not at terminal time)
        Z_analytical_list = []
        with torch.no_grad():
            for t_idx in range(len(time_grid) - 1):
                t_val = time_grid[t_idx]
                x_val = X_path_single[t_idx:t_idx+1, :]  # Shape: [1, dim_x]
                z_val = analytical_Z_func(t_val, x_val, **analytical_Z_kwargs)
                Z_analytical_list.append(z_val.squeeze())
        Z_analytical_tensor = torch.stack(Z_analytical_list)

        # Get numerical Z from numpy array
        Z_numerical_path = Z_paths_np[path_idx].squeeze()

        # Handle different Z shapes - convert tensor to numpy for plotting
        Z_analytical_np = Z_analytical_tensor.cpu().numpy()
        if Z_analytical_np.ndim == 1:
            Z_analytical_plot = Z_analytical_np
        elif Z_analytical_np.ndim == 2:
            # Shape is [N, dim_w], select component
            Z_analytical_plot = Z_analytical_np[:, component_idx]
        else:
            # Shape is [N, dim_y, dim_w], select Y component 0 and Z component
            Z_analytical_plot = Z_analytical_np[:, 0, component_idx]

        if Z_numerical_path.ndim == 1:
            Z_numerical_plot = Z_numerical_path
        else:
            Z_numerical_plot = Z_numerical_path[:, component_idx]

        # Plot Z
        axes[1].plot(time_grid_np[:-1], Z_analytical_plot, 'b', linewidth=1, label=label_analytical)
        axes[1].plot(time_grid_np[:-1], Z_numerical_plot, 'r--', linewidth=1, label=label_approx)

    # --- Finalize Plots ---
    axes[0].set_ylabel('$Y_t$')
    axes[0].set_xlabel('$t$')
    axes[0].legend()
    axes[1].set_ylabel(f'$Z_t^{{({component_idx+1})}}$')
    axes[1].set_xlabel('$t$')
    axes[1].legend()

    plt.tight_layout()
    return fig, axes

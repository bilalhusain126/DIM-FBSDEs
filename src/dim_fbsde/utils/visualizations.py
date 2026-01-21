import numpy as np
import matplotlib.pyplot as plt
from functools import partial 
import torch

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['axes.grid'] = False



# ==============================================================================================================
#                           SECTION 1: VISUALIZATIONS FOR PICARD ITERATION FBSDE SOLUTIONS
# ==============================================================================================================



def plot_solution_snapshots(solutions, labels, selected_times, x_component_idx=0, y_component_idx=0,
                            exact_solution_func=None, analytical_Y_kwargs=None, plot_type='binned', num_bins=20):
    """
    Plots a component of multiple multi-dimensional BSDE solutions (Y vs. X) at specific time snapshots.
    This function is designed for comparing several numerical approximations against each other and,
    optionally, against a known analytical solution.

    The first solution in the `solutions` list is used as the reference for determining the time grid
    and the x-axis range for all plots to ensure consistent comparisons.

    Args:
        solutions (list[dict]): A list of solution dictionaries. Each dictionary should be the output 
                                from a solver's get_solution() method, containing at least 'X', 'Y', and 'time' as keys.
        labels (list[str]): A list of strings corresponding to each solution in the `solutions` list.
                            These labels are used in the plot legend.
        selected_times (list[float]): A list of time points at which to create plots. The function will find the 
                                      closest available time in the grid for each value.
        x_component_idx (int, optional): The index of the X component (forward process) to plot on the x-axis. Defaults to 0.
        y_component_idx (int, optional): The index of the Y component (backward process) to plot on they-axis. Defaults to 0.
        exact_solution_func (callable, optional): A function for the analytical Y solution. It must accept `(t, X_values, **kwargs)` 
                                                  as arguments, where `t` is a float and `X_values` is a NumPy array. Defaults to None.
        analytical_Y_kwargs (dict, optional): A dictionary of extra keyword arguments to pass to the `exact_solution_func`. Defaults to None.
        plot_type (str, optional): The type of plot for the numerical solutions. Can be:
                                   - 'binned': Groups the data into bins along the x-axis and plots the mean of Y for each bin. 
                                   - 'scatter': Plots all individual (X, Y) sample points.
                                   Defaults to 'binned'.
        num_bins (int, optional): The number of bins to use if `plot_type` is 'binned'. Defaults to 20.
    
    Returns:
        matplotlib.figure.Figure: The figure for the entire grid of plots.
        numpy.ndarray of matplotlib.axes.Axes: An array of the subplot axes.
    """
    if analytical_Y_kwargs is None:
        analytical_Y_kwargs = {}

    # Use the first solution as the reference for time, axes, etc.
    ref_solution = solutions[0]
    X_ref_np, time_np = ref_solution['X'], ref_solution['time']
    
    num_plots = len(selected_times)
    ncols = min(num_plots, 3)
    nrows = (num_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes = axes.flatten()

    # Define a cycle of colors for the different numerical solutions
    colors = ['r', 'g', 'c', 'm', 'y', 'k']

    for i, t_target in enumerate(selected_times):
        ax = axes[i]
        t_idx = np.argmin(np.abs(time_np - t_target))
        actual_t = time_np[t_idx]
        
        # Determine shared axis limits from X-paths, range is the min of all mins to the max of all maxs
        all_mins = [sol['X'][:, t_idx, x_component_idx].min() for sol in solutions]
        all_maxs = [sol['X'][:, t_idx, x_component_idx].max() for sol in solutions]
        x_min = np.min(all_mins)
        x_max = np.max(all_maxs)

        # Plot the exact solution once (if provided)
        if exact_solution_func:
            d_x = X_ref_np.shape[2]
            S_exact_range = np.linspace(x_min, x_max, 100)

            if d_x > 1:
                # For multi-D X, fix other components to their mean value for the 2D slice plot
                other_indices = [j for j in range(d_x) if j != x_component_idx]
                mean_of_other_components = np.mean(X_ref_np[:, t_idx, other_indices], axis=0)
                
                X_for_analytical = np.zeros((100, d_x))
                X_for_analytical[:, other_indices] = mean_of_other_components
                X_for_analytical[:, x_component_idx] = S_exact_range
            else:
                X_for_analytical = S_exact_range[:, np.newaxis]

            Y_exact = exact_solution_func(actual_t, X_for_analytical, **analytical_Y_kwargs)
            ax.plot(S_exact_range, Y_exact, 'b-', zorder=1, label='Analytical')
            #ax.set_xlim(x_min, x_max)

        # Loop through and plot each provided numerical solution
        for sol_idx, sol in enumerate(solutions):
            current_X_comp = sol['X'][:, t_idx, x_component_idx]
            current_Y_comp = sol['Y'][:, t_idx, y_component_idx]
            color = colors[sol_idx % len(colors)]
            label = labels[sol_idx]

            if plot_type == 'binned':
                bins = np.linspace(x_min, x_max, num_bins + 1)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                # Use np.nanmean to safely handle empty bins (returns NaN, which is not plotted)
                bin_means = [np.nanmean(current_Y_comp[(current_X_comp >= lo) & (current_X_comp < hi)])
                             for lo, hi in zip(bins[:-1], bins[1:])]
                ax.plot(bin_centers, bin_means, marker='o', color=color, linestyle='-', ms=4, label=label)
            elif plot_type == 'scatter':
                ax.scatter(current_X_comp, current_Y_comp, c=color, alpha=0.1, s=3, zorder=2, label=label)
                
        ax.set_title(f'$t \\approx {actual_t:.2f}$')
        ax.set_xlabel(f'$X_t^{{{x_component_idx+1}}}$')
        ax.set_ylabel(f'$Y_t^{{{y_component_idx+1}}}$')
        if i == 0:
            leg = ax.legend()
            for lh in leg.legend_handles:
                lh.set_alpha(1) #makes legend markers dark 
    
    # Hide any unused subplots
    for j in range(num_plots, len(axes)): 
        fig.delaxes(axes[j])
        
    plt.tight_layout() 
    
    return fig, axes




def plot_pathwise_comparison(solution, analytical_Y_func, analytical_Z_func, 
                             path_indices=None, component_idx=0,
                             analytical_Y_kwargs=None, analytical_Z_kwargs=None):
    """
    Plots a path-by-path comparison of the numerical BSDE solution against an analytical solution.
    
    This version can handle analytical functions that require additional parameters.

    Args:
        solution (dict): The dictionary returned by the solver's get_solution() method.
        analytical_Y_func (function): A function for the analytical Y path. Its first two arguments
                                      must be `t` and `X_path`.
        analytical_Z_func (function): A function for the analytical Z path. Its first two arguments
                                      must be `t` and `X_path`.
        path_indices (list or np.ndarray, optional): Specific path indices to plot.
        component_idx (int, optional): The index of the Z component to plot.
        analytical_Y_kwargs (dict, optional): A dictionary of extra keyword arguments to pass to
                                              `analytical_Y_func`. Defaults to None.
        analytical_Z_kwargs (dict, optional): A dictionary of extra keyword arguments to pass to
                                              `analytical_Z_func`. Defaults to None.

    Returns:
        matplotlib.figure.Figure: The top-level figure container.
        numpy.ndarray of matplotlib.axes.Axes: An array of the two subplot axes.
    """
    # --- Handle optional kwargs dictionaries ---
    if analytical_Y_kwargs is None:
        analytical_Y_kwargs = {}
    if analytical_Z_kwargs is None:
        analytical_Z_kwargs = {}
        
    # --- Data Extraction and Setup ---
    time_grid, X_paths, Y_paths, Z_paths = solution['time'], solution['X'], solution['Y'], solution['Z']
    
    if path_indices is None:
        num_to_plot = min(5, X_paths.shape[0])
        path_indices = np.random.choice(X_paths.shape[0], size=num_to_plot, replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Plotting Loop ---
    for i, path_idx in enumerate(path_indices):
        X_path_single = X_paths[path_idx]

        # -- Plotting Y -- 
        Y_analytical = analytical_Y_func(time_grid, X_path_single, **analytical_Y_kwargs)
        Y_approx = Y_paths[path_idx]
        
        label_approx = "Deep Picard Approx." if i == 0 else None
        label_analytical = "Analytical" if i == 0 else None

        axes[0].plot(time_grid, Y_analytical, 'b', linewidth=1, label=label_analytical)
        axes[0].plot(time_grid, Y_approx, 'r--', linewidth=1, label=label_approx)

        # -- Plotting Z --
        time_grid_z, X_path_z = time_grid[:-1], X_path_single[:-1, :]
        
        Z_analytical = analytical_Z_func(time_grid_z, X_path_z, **analytical_Z_kwargs)
        Z_numerical_path = Z_paths[path_idx].squeeze(axis=1)
        
        axes[1].plot(time_grid_z, Z_analytical[:, component_idx], 'b', linewidth=1, label=label_analytical)
        axes[1].plot(time_grid_z, Z_numerical_path[:, component_idx], 'r--', linewidth=1, label=label_approx)


    # --- Finalize Plots ---
    axes[0].set_ylabel('$Y_t$'); axes[0].set_xlabel('$t$'); axes[0].legend()
    axes[1].set_ylabel(f'$Z_t^{{({component_idx+1})}}$'); axes[1].set_xlabel('$t$'); axes[1].legend()

    plt.tight_layout()
    return fig, axes




# ==============================================================================================================
#                                SECTION 2: ERROR PLOTS FOR PICARD ITERATION METHODS
# ==============================================================================================================



def plot_loss_by_iteration(history,
                           loss_type='Y',
                           title='Training Loss', 
                           y_log_scale=True,
                           window_size=100):
    """
    Plots moving average of the training loss across multiple Picard 
    iterations, with vertical lines separating each iteration.

    Args:
        history (list of dict): The history object from the solver's solution.
        loss_type (str): Which loss to plot, 'Y' or 'Z'.
        title (str): The title for the plot.
        y_log_scale (bool): If True, sets the y-axis to a logarithmic scale.
        window_size (int): The window size for the moving average calculation.
    """
    # Extract the list of lists from the history object
    if loss_type == 'Y':
        loss_history = [d.get('y_loss_history', []) for d in history]
    elif loss_type == 'Z':
        loss_history = [d.get('z_loss_history', []) for d in history]


    plt.figure(figsize=(9, 3.5))
    
    # Flatten the list of lists into a single continuous list of losses
    all_losses = [loss for iter_losses in loss_history if iter_losses for loss in iter_losses]

    # Calculate the positions for the vertical iteration markers
    iteration_boundaries = np.cumsum([len(l) for l in loss_history if l])[:-1] - 1

    # --- Calculate and plot the moving average ---
    moving_avg = np.convolve(all_losses, np.ones(window_size)/window_size, mode='valid')
    # The x-axis for the moving average starts after the first window
    moving_avg_x = np.arange(window_size - 1, len(all_losses))
    
    plt.plot(moving_avg_x, moving_avg, label=f'Moving Average', linewidth=1)

    # --- Plot vertical lines and text for each iteration ---
    for boundary in iteration_boundaries:
        plt.axvline(x=boundary, color='black', linestyle='--', linewidth=0.8, 
                    label='End of Picard Iteration' if boundary == iteration_boundaries[0] else "")

    # --- Final plot styling ---
    if y_log_scale:
        plt.yscale('log')

    plt.title(f'{title} for' + fr'$\mathcal{{N}}^{{{loss_type}}}$')
    plt.xlabel('Cumulative Training Batches')
    plt.ylabel(f'Loss' + (' (log scale)' if y_log_scale else ''))
    plt.legend()
    plt.xlim(0, len(all_losses))
    plt.tight_layout()
    plt.show()




# ==============================================================================================================
#                      SECTION 3: VISUALIZATIONS FROM TRAINED NN MODELS
# ==============================================================================================================



def _generate_paths_from_model(model_package, X_path_tensor, time_grid_tensor, component_idx, device):
    """
    Internal helper function to generate Y and Z paths from a trained model package.
    """
    model_type = model_package['model_type']
    nn_Y = model_package['nn_Y']
    nn_Y.to(device)
    nn_Y.eval()

    num_time_steps = X_path_tensor.shape[0]
    d_y = model_package.get('d_y', 1)
    d_w = model_package.get('d_w', 1)

    Y_approx_path = torch.zeros(num_time_steps, d_y, device=device)
    Z_approx_path = torch.zeros(num_time_steps - 1, d_y, d_w, device=device)

    # Vectorized Y-path calculation ---
    t_col = time_grid_tensor.unsqueeze(1)
    
    # Check the model type to call its forward pass correctly
    if isinstance(nn_Y, DGM.DGMNet):
        # DGMNet expects separate t and x inputs
        with torch.no_grad():
            Y_approx_path = nn_Y(t_col, X_path_tensor)
    else:
        # Picard-style nets expect a single combined input
        nn_inputs = torch.cat([t_col, X_path_tensor], dim=1)
        with torch.no_grad():
            Y_approx_path = nn_Y(nn_inputs)

    # Time-step loop for Z
    for i in range(num_time_steps - 1):
        t_i_vec = time_grid_tensor[i].unsqueeze(0) # Shape: [1]
        X_i_vec = X_path_tensor[i]                 # Shape: [d_x]

        if model_type == 'gradient':
            sigma_func = model_package['sde_diffusion_func']
            X_i_grad = X_i_vec.clone().detach().requires_grad_(True)
            
            # Apply the same logic to handle different forward pass signatures
            if isinstance(nn_Y, DGM.DGMNet):
                # DGMNet expects inputs with a batch dimension: [1, 1] and [1, d_x]
                Y_pred_grad = nn_Y(t_i_vec.unsqueeze(1), X_i_grad.unsqueeze(0))
            else:
                # Picard-style nets expect a single combined input: [1, 1 + d_x]
                nn_input_grad = torch.cat([t_i_vec, X_i_grad], dim=0).unsqueeze(0)
                Y_pred_grad = nn_Y(nn_input_grad)

            # Jacobian calculation logic remains the same
            jac_Y_X_rows = []
            for j in range(d_y):
                grad_output = torch.zeros_like(Y_pred_grad)
                grad_output[:, j] = 1
                grad_Yj = torch.autograd.grad(
                    outputs=Y_pred_grad,
                    inputs=X_i_grad,
                    grad_outputs=grad_output,
                    retain_graph=True
                )[0]
                jac_Y_X_rows.append(grad_Yj.unsqueeze(0))
            
            jacobian_Y_X = torch.cat(jac_Y_X_rows, dim=0)

            # Get the best available Y and Z values
            Y_val_i = Y_approx_path[i].cpu().numpy()
            if i == 0:
                Z_val_prev = np.zeros((d_y, d_w))
            else:
                Z_val_prev = Z_approx_path[i-1].cpu().numpy()
            
            # Uncoupled functions will simply ignore Y_paths and Z_paths.
            sigma_np = sigma_func(
                X_i_vec.cpu().numpy(), 
                t_i_vec.item(),
                Y_paths=Y_val_i, 
                Z_paths=Z_val_prev
            )
            
            sigma_val = torch.tensor(sigma_np, dtype=torch.float32, device=device)
            Z_pred = torch.matmul(jacobian_Y_X, sigma_val)
            Z_approx_path[i, :, :] = Z_pred.detach()

        elif model_type == 'regression':
            nn_Z = model_package['nn_Z']
            nn_Z.to(device)
            nn_Z.eval()
            
            nn_input = torch.cat([t_i_vec, X_i_vec], dim=0).unsqueeze(0)
            with torch.no_grad():
                Z_pred_flat = nn_Z(nn_input)
                Z_approx_path[i, :, :] = Z_pred_flat.reshape(d_y, d_w)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    return Y_approx_path, Z_approx_path




def plot_pathwise_comparison_from_models(
    models_to_plot,
    time_grid,
    X_paths,
    device,
    analytical_Y_func=None,
    analytical_Z_func=None,
    path_indices=None,
    y_component_idx=0,
    z_component_idx=0,
    analytical_Y_kwargs=None,
    analytical_Z_kwargs=None
):
    """
    Plots a path-by-path comparison using trained NN models.
    Can use pre-computed Y and Z paths for specific models if they are
    provided within the model_package dictionary.
    """
    if analytical_Y_kwargs is None: analytical_Y_kwargs = {}
    if analytical_Z_kwargs is None: analytical_Z_kwargs = {}

    if path_indices is None:
        num_to_plot = min(5, X_paths.shape[0])
        path_indices = np.random.choice(X_paths.shape[0], size=num_to_plot, replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    time_grid_tensor = torch.tensor(time_grid, dtype=torch.float32, device=device)
    X_paths_tensor = torch.tensor(X_paths, dtype=torch.float32, device=device)
    
    colors = ['r', 'g', 'c', 'm', 'y', 'k']

    for i, path_idx in enumerate(path_indices):
        X_path_single_tensor = X_paths_tensor[path_idx]
        X_path_single_np = X_paths[path_idx]
        
        label_analytical = "Analytical" if i == 0 else None
        
        if analytical_Y_func:
            Y_analytical = analytical_Y_func(time_grid, X_path_single_np, **analytical_Y_kwargs)
            if Y_analytical.ndim == 1:
                Y_to_plot = Y_analytical
            else:
                Y_to_plot = Y_analytical[:, y_component_idx]
            axes[0].plot(time_grid, Y_to_plot, 'b', linewidth=1.5, label=label_analytical, zorder=1)

        if analytical_Z_func:
            time_grid_z = time_grid[:-1]
            X_path_z = X_path_single_np[:-1, :]
            Z_analytical = analytical_Z_func(time_grid_z, X_path_z, **analytical_Z_kwargs)
            if Z_analytical.ndim == 1:
                Z_to_plot = Z_analytical
            else:
                Z_to_plot = Z_analytical[:, z_component_idx]
            axes[1].plot(time_grid_z, Z_to_plot, 'b', linewidth=1.5, label=label_analytical, zorder=1)

        for model_idx, model_package in enumerate(models_to_plot):
            label_approx = model_package['label'] if i == 0 else None
            color = model_package.get('color', colors[model_idx % len(colors)])

            # Check if this specific model package contains precomputed paths
            if 'precomputed_paths' in model_package:
                # Use the pre-computed paths directly for this model
                precomputed = model_package['precomputed_paths']
                Y_np = precomputed['Y'][path_idx]
                Z_np = precomputed['Z'][path_idx]
            else:
                # Otherwise, generate the paths on the fly using the helper function
                Y_approx, Z_approx = _generate_paths_from_model(
                    model_package, X_path_single_tensor, time_grid_tensor, z_component_idx, device
                )
                Y_np = Y_approx.detach().cpu().numpy()
                Z_np = Z_approx.detach().cpu().numpy()

            axes[0].plot(time_grid, Y_np[:, y_component_idx], color=color, linestyle='--', linewidth=1, label=label_approx)
            axes[1].plot(time_grid[:-1], Z_np[:, y_component_idx, z_component_idx], color=color, linestyle='--', linewidth=1, label=label_approx)

    axes[0].set_ylabel('$Y_t$')
    axes[0].set_xlabel('$t$')
    axes[0].legend()

    axes[1].set_ylabel(f'$Z_t^{{({y_component_idx+1})}}$')
    axes[1].set_xlabel('$t$')
    axes[1].legend()

    plt.tight_layout()
    return fig, axes



def plot_Y_error_ribbon_subplots(
    models_to_plot,
    time_grid,
    X_paths,
    device,
    analytical_Y_func,
    y_component_idx=0,
    analytical_Y_kwargs=None,
    yscale='linear',
    log_eps=1e-9,
    ribbon_mode='quantile', 
    quantiles=(0.1, 0.9),
    spaghetti_paths=50, 
    spaghetti_alpha=0.05
):
    """
    Plots the error of Y as a grid of ribbon subplots.

    Args:
        ... (previous args) ...
        ribbon_mode (str, optional): 'std', 'quantile', or 'spaghetti'.
        ...
        spaghetti_paths (int, optional): Number of individual path errors to plot
                                         in 'spaghetti' mode.
        spaghetti_alpha (float, optional): Alpha transparency for spaghetti lines.
    """
    if analytical_Y_kwargs is None: analytical_Y_kwargs = {}
    
    num_plots = len(models_to_plot)
    ncols = min(num_plots, 3)
    nrows = (num_plots + ncols - 1) // ncols
    
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), 
        squeeze=False, sharex=True, sharey=True
    )
    axes = axes.flatten()

    time_grid_tensor = torch.tensor(time_grid, dtype=torch.float32, device=device)
    X_paths_tensor = torch.tensor(X_paths, dtype=torch.float32, device=device)
    
    # Calculate analytical solution 
    print("Generating analytical Y paths...")
    y_analytical_list = []
    for i in range(X_paths.shape[0]):
        y_path = analytical_Y_func(time_grid, X_paths[i], **analytical_Y_kwargs)
        y_analytical_list.append(y_path)
    Y_analytical_all_paths = np.stack(y_analytical_list, axis=0)
    
    # Select the correct component for multi-dimensional Y
    if Y_analytical_all_paths.ndim == 3:
        Y_analytical_np = Y_analytical_all_paths[:, :, y_component_idx]
    else:
        Y_analytical_np = Y_analytical_all_paths
        
    for i, (ax, model_package) in enumerate(zip(axes, models_to_plot)):
        label = model_package['label']
        color = model_package.get('color') 
        nn_Y = model_package['nn_Y']
        
        print(f"Generating approximate Y paths for model: {label}...")
        nn_Y.to(device).eval()
        with torch.no_grad():
            if isinstance(nn_Y, DGM.DGMNet):
                num_paths, num_steps = X_paths_tensor.shape[0], X_paths_tensor.shape[1]
                t_expanded = time_grid_tensor.repeat(num_paths, 1).view(num_paths, num_steps, 1)
                dgm_t_input = t_expanded.reshape(-1, 1)
                dgm_x_input = X_paths_tensor.reshape(-1, X_paths_tensor.shape[2])
                Y_approx_flat = nn_Y(dgm_t_input, dgm_x_input)
                Y_approx_tensor = Y_approx_flat.reshape(num_paths, num_steps, -1)
            else:
                num_paths, num_steps = X_paths_tensor.shape[0], X_paths_tensor.shape[1]
                d_y = model_package.get('d_y', 1)
                Y_approx_tensor = torch.zeros(num_paths, num_steps, d_y, device=device)
                t_col = time_grid_tensor.unsqueeze(1)
                for j in range(num_paths):
                    nn_inputs = torch.cat([t_col, X_paths_tensor[j]], dim=1)
                    Y_approx_tensor[j,:,:] = nn_Y(nn_inputs)
        
        Y_approx_np = Y_approx_tensor.cpu().numpy()[:, :, y_component_idx]
        absolute_errors = np.abs(Y_approx_np - Y_analytical_np)

        ax.set_title(label)
        ax.set_yscale(yscale)
        ylabel = r'$\epsilon_{Y_t}$'

        if ribbon_mode == 'spaghetti':
            num_total_paths = absolute_errors.shape[0]
            plot_indices = np.random.choice(
                num_total_paths, size=min(spaghetti_paths, num_total_paths), replace=False
            )
            
            for path_idx in plot_indices:
                path_error = absolute_errors[path_idx, :]
                if yscale == 'log':
                    path_error = np.maximum(path_error, log_eps)
                ax.plot(time_grid, path_error, color=color, alpha=spaghetti_alpha, linewidth=0.7)
            
            median_error = np.median(absolute_errors, axis=0)
            if yscale == 'log':
                median_error = np.maximum(median_error, log_eps)
            ax.plot(time_grid, median_error, color=color, linewidth=2.0)
        
        else: # 'std' or 'quantile' logic
            if ribbon_mode == 'quantile':
                central_line = np.median(absolute_errors, axis=0)
                lower_bound = np.quantile(absolute_errors, quantiles[0], axis=0)
                upper_bound = np.quantile(absolute_errors, quantiles[1], axis=0)
            else: # 'std'
                central_line = np.mean(absolute_errors, axis=0)
                std_abs_error_t = np.std(absolute_errors, axis=0)
                lower_bound = central_line - std_abs_error_t
                upper_bound = central_line + std_abs_error_t

            if yscale == 'log':
                central_line = np.maximum(central_line, log_eps)
                lower_bound = np.maximum(lower_bound, log_eps)
            
            ax.plot(time_grid, central_line, color=color, linewidth=1.5)
            ax.fill_between(time_grid, lower_bound, upper_bound,
                            color=color, alpha=0.3, edgecolor=None)
        
        ax.set_ylim(bottom=0)

    fig.supxlabel('$t$')
    fig.supylabel(ylabel)
    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig, axes




def plot_Z_error_ribbon_subplots(
    models_to_plot,
    time_grid,
    X_paths,
    device,
    analytical_Z_func,
    analytical_Z_kwargs=None,
    yscale='linear',
    log_eps=1e-9,
    ribbon_mode='spaghetti', 
    quantiles=(0.25, 0.75),
    spaghetti_paths=50, 
    spaghetti_alpha=0.05
):
    """
    Plots the error of Z as a grid of subplots, one for each model.
    (Docstring remains the same)
    """
    if analytical_Z_kwargs is None: analytical_Z_kwargs = {}
    
    num_plots = len(models_to_plot)
    ncols = min(num_plots, 3)
    nrows = (num_plots + ncols - 1) // ncols
    
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), 
        squeeze=False, sharex=True, sharey=True
    )
    axes = axes.flatten()

    time_grid_tensor = torch.tensor(time_grid, dtype=torch.float32, device=device)
    time_grid_z = time_grid[:-1]

    # Calculate the analytical solution for all models to compare against.
    print("Generating analytical Z paths...")
    all_Z_analytical_list = []
    for i in range(X_paths.shape[0]):
        # Generate analytical Z for one path
        Z_analytical_path = analytical_Z_func(time_grid, X_paths[i], **analytical_Z_kwargs)
        all_Z_analytical_list.append(Z_analytical_path)
    Z_analytical_all_paths = np.stack(all_Z_analytical_list, axis=0)
    
    # X paths have size N+1, Z paths have size N; this resolves the conflict
    Z_analytical_all_paths = Z_analytical_all_paths[:, :-1, ...] 
    if Z_analytical_all_paths.ndim == 3:
        Z_analytical_all_paths = Z_analytical_all_paths[..., np.newaxis]

    
    for ax, model_package in zip(axes, models_to_plot):
        label = model_package['label']
        color = model_package.get('color')

        # --- Data Generation Loop for APPROXIMATE Z (Path by Path) ---
        all_Z_approx_list = []
        
        print(f"Generating approximate Z paths for model: {label}...")
        for i in range(X_paths.shape[0]):
            X_path_i_tensor = torch.tensor(X_paths[i, :, :], dtype=torch.float32, device=device)
            _, Z_approx_path = _generate_paths_from_model(
                model_package, X_path_i_tensor, time_grid_tensor, 0, device
            )
            all_Z_approx_list.append(Z_approx_path.cpu().numpy())

        Z_approx_all_paths = np.stack(all_Z_approx_list, axis=0)
        
        # --- Error Calculation using Frobenius Norm ---
        error_matrices = Z_approx_all_paths - Z_analytical_all_paths
        absolute_errors = np.linalg.norm(error_matrices, axis=(-2, -1))

        # --- Plotting Logic (This part is unchanged) ---
        ax.set_title(label)
        ax.set_yscale(yscale)

        if ribbon_mode == 'spaghetti':
            num_total_paths = absolute_errors.shape[0]
            plot_indices = np.random.choice(
                num_total_paths, size=min(spaghetti_paths, num_total_paths), replace=False
            )
            for path_idx in plot_indices:
                path_error = absolute_errors[path_idx, :]
                if yscale == 'log': path_error = np.maximum(path_error, log_eps)
                ax.plot(time_grid_z, path_error, color=color, alpha=spaghetti_alpha, linewidth=0.7)
            
            median_error = np.median(absolute_errors, axis=0)
            if yscale == 'log': median_error = np.maximum(median_error, log_eps)
            ax.plot(time_grid_z, median_error, color=color, linewidth=2.0)
        
        else: # 'quantile' or 'std'
            if ribbon_mode == 'quantile':
                central_line = np.median(absolute_errors, axis=0)
                lower_bound = np.quantile(absolute_errors, quantiles[0], axis=0)
                upper_bound = np.quantile(absolute_errors, quantiles[1], axis=0)
            else: # 'std'
                central_line = np.mean(absolute_errors, axis=0)
                std_abs_error_t = np.std(absolute_errors, axis=0)
                lower_bound = central_line - std_abs_error_t
                upper_bound = central_line + std_abs_error_t

            if yscale == 'log':
                central_line = np.maximum(central_line, log_eps)
                lower_bound = np.maximum(lower_bound, log_eps)
            
            ax.plot(time_grid_z, central_line, color=color, linewidth=1.5)
            ax.fill_between(time_grid_z, lower_bound, upper_bound,
                            color=color, alpha=0.3, edgecolor=None)

        ax.set_ylim(bottom=0)


    # Add shared labels and a main title
    ylabel = r'$\epsilon_{Z_t}$'
    fig.supxlabel('$t$')
    fig.supylabel(ylabel)

    # Hide any unused subplots
    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig, axes
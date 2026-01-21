"""
Utility functions for saving and loading trained FBSDE solver models and their results.

This module provides a standardized way to manage experimental artifacts for both
Picard-style solvers (Uncoupled, Coupled, MV) and DGM models.

Functions:
    - save_solver_results: Saves the state of a Picard-style solver.
    - load_and_reconstruct_model: Loads a file from save_solver_results.
    - save_dgm_model: Saves a trained DGM model and its parameters.
    - load_dgm_model: Loads a file from save_dgm_model and reconstructs the DGMNet.
"""

import torch
import torch.nn as nn
import os
import SDE_utils
import DGM



# ==============================================================================================================
#                                SECTION 1: FUNCTIONS FOR DEEP ITERATIVE SOLVERS
# ==============================================================================================================


def save_solver_results(solver, solution, problem_name, save_directory,
                        custom_filename=None, save_solution_paths=True):
    """
    Saves the complete state and results of a trained FBSDE solver to a .pth file.

    This function handles different solver types (Uncoupled, Coupled, MV)
    and Z-methods (gradient, regression). It prunes large tensors from the history
    to keep file sizes manageable.

    Args:
        solver (object): The trained solver instance (e.g., UncoupledFBSDESolver).
        solution (dict): The dictionary returned by the solver's get_solution() method.
        problem_name (str): A descriptive name for the problem (e.g., 'Coupled_Sine_Cosine').
        save_directory (str): The path to the directory where the file will be saved.
        custom_filename (str, optional): A specific filename. If None, a descriptive
                                         filename will be generated automatically.
        save_solution_paths (bool, optional): If False, the large 'X', 'Y', and 'Z'
                                              solution path arrays will be excluded from
                                              the saved file to reduce its size.
                                              Defaults to True.
    """
    print("--- Preparing to save solver results ---")

    # --- 1. Determine Solver Type and Z-Method from the solver instance ---
    solver_type = solver.__class__.__name__
    if solver_type in ['CoupledFBSDESolver', 'McKeanVlasovFBSDESolver']:
        z_method = solver.params['z_method']
        solver_params = solver.params
    else:  # UncoupledFBSDESolver
        z_method = solver.z_method
        solver_params = {
            'T': solver.T, 'N': solver.N, 'y0_X': solver.y0_X,
            'd_x': solver.d_x, 'd_w': solver.d_w, 'd_y': solver.d_y,
            'num_paths': solver.num_paths, 'z_method': solver.z_method,
            'training_config': solver.training_config
        }
    print(f"Solver Type: {solver_type}, Z-Method: {z_method}")

    # --- 2. Prune History ---
    history_to_prune = solution.get('global_history') or solution.get('history', [])
    pruned_history = []
    if history_to_prune:
        is_global = 'inner_solver_history' in history_to_prune[0]
        for record in history_to_prune:
            pruned_record = {k: v for k, v in record.items() if k not in ['Y_iterate', 'Z_iterate']}
            if is_global:
                pruned_inner_history = []
                for inner_record in record.get('inner_solver_history', []):
                    pruned_inner = {k: v for k, v in inner_record.items() if k not in ['Y_iterate', 'Z_iterate']}
                    pruned_inner_history.append(pruned_inner)
                pruned_record['inner_solver_history'] = pruned_inner_history
            pruned_history.append(pruned_record)
        print("Successfully pruned large Y and Z iterates from history.")
    else:
        print("Warning: No history found to prune.")

    solution_pruned = solution.copy()
    if 'global_history' in solution_pruned:
        solution_pruned['global_history'] = pruned_history
    elif 'history' in solution_pruned:
        solution_pruned['history'] = pruned_history

    # --- 3. Optionally remove final solution paths to save space ---
    if not save_solution_paths:
        paths_to_remove = ['X', 'Y', 'Z']
        for key in paths_to_remove:
            if key in solution_pruned:
                del solution_pruned[key]
        print("Final solution paths ('X', 'Y', 'Z') have been excluded from the saved file.")
    else:
        print("Final solution paths will be included in the saved file.")

    solver_params = {
        key: value for key, value in solver_params.items() if not callable(value)
    }
    print("Pruned all function objects from solver parameters.")

    # --- 4. Gather Neural Network Information ---
    def get_nn_architecture(nn_module):
        if not hasattr(nn_module, 'model') or not isinstance(nn_module.model, nn.Sequential): return None
        num_linear_layers = sum(1 for layer in nn_module.model if isinstance(layer, nn.Linear))
        return {
            'input_size': nn_module.model[0].in_features,
            'hidden_size': nn_module.model[0].out_features,
            'output_size': nn_module.model[-1].out_features,
            'num_layers': num_linear_layers - 1,
            'activation': 'SiLU'
        }

    nn_info = {
        'nn_Y_state_dict': solver.nn_Y.state_dict(),
        'nn_Y_architecture': get_nn_architecture(solver.nn_Y)
    }
    if z_method == 'regression' and hasattr(solver, 'nn_Z') and solver.nn_Z:
        nn_info['nn_Z_state_dict'] = solver.nn_Z.state_dict()
        nn_info['nn_Z_architecture'] = get_nn_architecture(solver.nn_Z)

    # --- 5. Assemble the Final Dictionary ---
    saved_data = {
        'problem_name': problem_name,
        'solver_type': solver_type,
        'neural_network_info': nn_info,
        'solver_params': solver_params,
        'final_solution': solution_pruned
    }

    # --- 6. Generate Filename and Save ---
    if custom_filename is None:
        d_x = solver_params.get('d_x', 'Unknown')
        filename = f"{problem_name}_{solver_type}_{z_method.capitalize()}_{d_x}D.pth"
    else:
        filename = custom_filename

    os.makedirs(save_directory, exist_ok=True)
    save_path = os.path.join(save_directory, filename)

    try:
        torch.save(saved_data, save_path)
        print(f"\nSolver state and results successfully saved to: {save_path}")
    except Exception as e:
        print(f"\nError saving file: {e}")


def load_and_reconstruct_model(filepath, device='cpu'):
    """
    Loads a saved solver file and reconstructs the neural network models.

    This function is the counterpart to `save_solver_results`. It reads a .pth
    file and uses the stored architecture information to create new model
    instances, then loads the trained weights into them.

    Args:
        filepath (str): Path to the saved .pth file.
        device (str): The device to load the models onto ('cpu' or 'cuda').

    Returns:
        tuple: A tuple containing:
               - reconstructed_models (dict): A dictionary with 'nn_Y' and optionally 'nn_Z'.
               - saved_data (dict): The full dictionary loaded from the file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No file found at the specified path: {filepath}")

    saved_data = torch.load(filepath, map_location=device)
    nn_info = saved_data['neural_network_info']
    reconstructed_models = {}

    print(f"Reconstructing nn_Y from: {filepath}")
    arch_Y = nn_info['nn_Y_architecture']
    nn_Y = SDE_utils.ConditionalExpectationNN_II(
        input_size=arch_Y['input_size'], hidden_size=arch_Y['hidden_size'],
        output_size=arch_Y['output_size'], num_layers=arch_Y['num_layers'],
        activation=arch_Y.get('activation', 'SiLU')
    )
    nn_Y.load_state_dict(nn_info['nn_Y_state_dict'])
    nn_Y.to(device)
    nn_Y.eval()
    reconstructed_models['nn_Y'] = nn_Y

    if 'nn_Z_state_dict' in nn_info:
        print(f"Reconstructing nn_Z from: {filepath}")
        arch_Z = nn_info['nn_Z_architecture']
        nn_Z = SDE_utils.ConditionalExpectationNN_II(
            input_size=arch_Z['input_size'], hidden_size=arch_Z['hidden_size'],
            output_size=arch_Z['output_size'], num_layers=arch_Z['num_layers'],
            activation=arch_Z.get('activation', 'SiLU')
        )
        nn_Z.load_state_dict(nn_info['nn_Z_state_dict'])
        nn_Z.to(device)
        nn_Z.eval()
        reconstructed_models['nn_Z'] = nn_Z

    print("Successfully reconstructed all models.")
    return reconstructed_models, saved_data




# ==============================================================================================================
#                                    SECTION 2: FUNCTIONS FOR DGM MODELS
# ==============================================================================================================


def save_dgm_model(model, problem_name, save_directory, custom_filename=None, **extra_params):
    """
    Saves a trained DGM model and its associated parameters to a .pth file.

    Args:
        model (DGM.DGMNet): The trained DGMNet instance.
        problem_name (str): A descriptive name for the problem (e.g., 'Hure_Equation').
        save_directory (str): The path to the directory where the file will be saved.
        custom_filename (str, optional): A specific filename. If None, a descriptive
                                         filename will be generated automatically.
        **extra_params: Arbitrary keyword arguments (e.g., learning_rate, dimension_d,
                        terminal_time_T) that will be saved alongside the model.
    """
    print("--- Preparing to save DGM model ---")
    
    model_architecture = {
        'input_dim': model.input_dim, 'hidden_dim': model.hidden_dim,
        'output_dim': model.output_dim, 'n_layers': model.n_layers
    }
    saved_data = {
        'problem_name': problem_name, 'model_architecture': model_architecture,
        'model_state_dict': model.state_dict(), 'extra_saved_params': extra_params
    }

    if custom_filename is None:
        filename = f"{problem_name}_DGM_{model_architecture['input_dim']-1}D.pth"
    else:
        filename = custom_filename

    os.makedirs(save_directory, exist_ok=True)
    save_path = os.path.join(save_directory, filename)

    try:
        torch.save(saved_data, save_path)
        print(f"\nDGM model and parameters successfully saved to: {save_path}")
    except Exception as e:
        print(f"\nError saving DGM model file: {e}")


def load_dgm_model(filepath, device='cpu'):
    """
    Loads a saved DGM model file and reconstructs the DGMNet.

    Args:
        filepath (str): Path to the saved .pth file created by save_dgm_model.
        device (str): The device to load the model onto ('cpu' or 'cuda').

    Returns:
        tuple: A tuple containing:
               - reconstructed_model (DGM.DGMNet): The reconstructed and loaded model.
               - saved_data (dict): The full dictionary loaded from the file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No DGM model file found at the specified path: {filepath}")

    saved_data = torch.load(filepath, map_location=device)
    arch = saved_data['model_architecture']
    print(f"Reconstructing DGM model from: {filepath}")

    reconstructed_model = DGM.DGMNet(
        input_dim=arch['input_dim'], hidden_dim=arch['hidden_dim'],
        output_dim=arch['output_dim'], n_layers=arch['n_layers']
    )
    reconstructed_model.load_state_dict(saved_data['model_state_dict'])
    reconstructed_model.to(device)
    reconstructed_model.eval()

    print("Successfully reconstructed DGM model.")
    return reconstructed_model, saved_data


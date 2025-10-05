import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy




# ==============================================================================================================
#                                   SECTION 1: STOCHASTIC DIFFERENTIAL EQUATIONS
# ==============================================================================================================



class SDE:
    """
    Represents a stochastic differential equation (SDE).

    Attributes:
        name (str): Name of the SDE.
        drift (function): Drift function of the SDE.
        diffusion (function): Diffusion function of the SDE.
    """

    def __init__(self, name, drift, diffusion):
        """
        Initializes an SDE instance.

        Args:
            name (str): Name of the SDE.
            drift (function): Drift function of the SDE.
            diffusion (function): Diffusion function of the SDE.
        """
        self.name = name
        self.drift = drift
        self.diffusion = diffusion





# ==============================================================================================================
#                                      SECTION 2: EULER-MARUYAMA SOLVER
# ==============================================================================================================




class EulerMaruyamaND:
    """
    Simulates paths of a multi-dimensional SDE using the Euler-Maruyama method.

    This version is augmented to generate and store both the state process paths (X)
    and the corresponding Brownian motion increments (dW) used to create them.
    It can also handle drift/diffusion functions that depend on extra, pre-computed 
    time-varying data (for use in coupled FBSDEs).
    """
    def __init__(self, drift_func, diffusion_func, d_x, d_w, s, T, N, x0, **extra_func_kwargs):
        """
        Initializes an EulerMaruyamaND instance.

        Args:
            drift_func (function): Drift function. Signature: drift(x_vec, t, **kwargs).
            diffusion_func (function): Diffusion function. Signature: diffusion(x_vec, t, **kwargs).
            d_x (int): Dimension of the state process X_t.
            d_w (int): Dimension of the Brownian motion W_t.
            s (float): Starting time of the simulation.
            T (float): Ending time of the simulation.
            N (int): Number of time steps in the simulation.
            x0 (np.ndarray): Initial condition for X_s, shape (d_x,).
            **extra_func_kwargs: Optional keyword arguments containing full data paths
                                (e.g., Y_paths=...) passed to drift/diffusion.
        """
        self.drift_func = drift_func
        self.diffusion_func = diffusion_func
        self.d_x = d_x
        self.d_w = d_w
        self.s = s
        self.T = T
        self.N = N
        self.x0 = np.asarray(x0).reshape(d_x) 

        self.dt = (T - s) / N
        self.time_arr = np.linspace(self.s, self.T, self.N + 1)
        
        self.extra_func_kwargs = extra_func_kwargs
        
        # Attributes to store the simulation results
        self.X_paths = None
        self.dW_paths = None


    def simulate_one_path(self, path_index=0):
        """
        Simulates a single path and the corresponding Brownian increments.

        Args:
            path_index (int): The index of the path, used for slicing extra data
                              in coupled problems.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing:
                - The simulated state path of shape (N + 1, d_x).
                - The Brownian increments path of shape (N, d_w).
        """
        X_path = np.zeros((self.N + 1, self.d_x))
        X_path[0, :] = self.x0

        # Generate the Brownian increments for this entire path
        dW = np.random.normal(0, np.sqrt(self.dt), size=(self.N, self.d_w))

        for i in range(self.N):
            t_i = self.time_arr[i]
            X_i = X_path[i, :] 
            dW_i = dW[i, :]    

            # Prepare extra args for this specific path and time step if needed
            kwargs_for_step_i = {}
            for key, full_path_data in self.extra_func_kwargs.items():
                kwargs_for_step_i[key] = full_path_data[path_index, i]

            kwargs_for_step_i['path_index'] = path_index
            kwargs_for_step_i['time_idx'] = i

            drift_val = self.drift_func(X_i, t_i, **kwargs_for_step_i)
            diffusion_val = self.diffusion_func(X_i, t_i, **kwargs_for_step_i)
            
            # Perform the Euler step 
            X_path[i + 1, :] = X_i + drift_val * self.dt + np.dot(diffusion_val, dW_i)

        # Return both the state path and the increments that created it
        return X_path, dW

    
    def generate_simulations(self, num_sims):
        """
        Generates multiple simulations by repeatedly calling simulate_one_path,
        and stores both the X and dW paths.

        Args:
            num_sims (int): Number of simulations to generate.
        """
        x_paths_list = []
        dw_paths_list = []
        
        for i in range(num_sims):
            x_path, dw_path = self.simulate_one_path(path_index=i)
            x_paths_list.append(x_path)
            dw_paths_list.append(dw_path)
        
        # Store results as class attributes
        self.X_paths = np.array(x_paths_list)
        self.dW_paths = np.array(dw_paths_list)

    
    def get_x_paths_as_tensor(self):
        """Converts the final numpy array of X simulations to a PyTorch tensor."""
        return torch.tensor(self.X_paths, dtype=torch.float32)

    def get_dw_paths_as_tensor(self):
        """Converts the final numpy array of dW increments to a PyTorch tensor."""
        return torch.tensor(self.dW_paths, dtype=torch.float32)





# ==============================================================================================================
#                               SECTION 3: CONDITIONAL EXPECTATION APPROXIMATORS
# ==============================================================================================================



class ConditionalExpectationEstimator:
    """
    Estimates conditional expectations using simulated SDE paths.

    Attributes:
        sde (SDE): The SDE to simulate.
        s (float): Starting time of the simulation.
        T (float): Ending time of the simulation.
        N (int): Number of time steps in the simulation.
        Y_grid (np.ndarray): Grid of starting values.
        num_paths (int): Number of paths per starting value.
        F (function): Function applied to paths.
        spline_smoothing (float): Smoothing factor for spline fitting.
    """

    def __init__(self, sde, s, T, N, Y_grid, num_paths, F=lambda y: y, spline_smoothing=1):
        """
        Initializes the ConditionalExpectationEstimator.

        Args:
            sde (SDE): The SDE to simulate.
            s (float): Starting time of the simulation.
            T (float): Ending time of the simulation.
            N (int): Number of time steps in the simulation.
            Y_grid (np.ndarray): Grid of starting values.
            num_paths (int): Number of paths per starting value.
            F (function): Function applied to paths.
            spline_smoothing (float): Smoothing factor for spline fitting.
        """
        self.sde = sde
        self.s = s
        self.T = T
        self.N = N
        self.F = F
        self.Y_grid = Y_grid
        self.num_paths = num_paths
        self.paths = []
        self.terminal_means = []
        self.terminal_SDs = []
        self.spline_smoothing = spline_smoothing
        self.CE_spline = None
        self.solver = EulerMaruyama1D(self.sde, self.s, self.T, self.N, 0)

    
    def simulate_paths(self):
        """
        Simulates paths for each starting value and computes statistics.
        """
        for y_i in self.Y_grid:
            self.solver.edit_sim_params(y0=y_i)
            self.solver.generate_simulations(self.num_paths)
            self.paths.append(self.solver.simulations)

        self.paths = self.F(np.array(self.paths))
        self.compute_path_stats()
        self.CE_spline = UnivariateSpline(self.Y_grid, self.terminal_means, s=self.spline_smoothing)

    
    def compute_path_stats(self):
        """
        Computes mean and standard deviation of terminal values.
        """
        terminal_vals = self.paths[:, :, -1]
        self.terminal_means = np.mean(terminal_vals, axis=1)
        self.terminal_SDs = np.std(terminal_vals, axis=1) / (self.num_paths ** 0.5)

    
    def plot_paths(self):
        """
        Plots all simulated paths with their terminal mean values.
        """
        colors = itertools.cycle(plt.cm.tab10.colors)

        for i in range(len(self.Y_grid)):
            color_curr = next(colors)
            plt.plot(self.solver.time_arr + self.s, self.paths[i].T, linewidth=0.7, color=color_curr)
            plt.scatter(
                self.solver.time_arr[-1] + self.s, self.terminal_means[i],
                color=color_curr, edgecolor='black', linewidth=1, zorder=5
            )
        plt.title(self.sde.name)
        plt.ylabel("$Y$")
        plt.xlabel("$t$")
        plt.show()

    
    def plot_CE_approximation(self):
        """
        Plots the conditional expectation approximation.
        """
        plt.errorbar(
            self.Y_grid, self.terminal_means, yerr=self.terminal_SDs,
            fmt='o', capsize=5, markeredgewidth=1, markeredgecolor='black', color='orange', ecolor='black', label='Mean across sims'
        )
        Ys_pred = np.linspace(min(self.Y_grid), max(self.Y_grid), len(self.Y_grid) * 10)
        Yt_pred = self.CE_spline(Ys_pred)
        plt.plot(Ys_pred, Yt_pred, label='Spline fit', color='orange', linewidth=2)
        plt.title(self.sde.name + r": $\mathbb{E}[F(Y_t)|Y_s]$ Approximation")
        plt.xlabel('$Y_s$')
        plt.ylabel('$F(Y_t)$')
        plt.legend()
        plt.show()





        

class ConditionalExpectationNN_II(nn.Module):
    """
    Neural network for conditional expectation (Version II).
    """
    def __init__(self, input_size=2, hidden_size=10, output_size=1, num_layers=1, activation='SiLU'):
        """
        Initializes the ConditionalExpectationNN_II.

        Args:
            input_size (int): Input dimension (now includes s and Y_s).
            hidden_size (int): Number of neurons in each hidden layer.
            output_size (int): Output dimension.
            num_layers (int): Number of hidden layers.
            activation (str): Activation function to use ('SiLU', 'ReLU', 'Tanh', etc.).
        """
        super(ConditionalExpectationNN_II, self).__init__()

        # Define activation function
        if activation == 'SiLU':
            self.activation = nn.SiLU()
        elif activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'Tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        # Dynamically construct layers
        layers = [nn.Linear(input_size, hidden_size), self.activation]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(self.activation)
            
        layers.append(nn.Linear(hidden_size, output_size))

        # uncomment for indicator function F
        # layers.append(nn.Sigmoid())  

        self.model = nn.Sequential(*layers)

    
    def forward(self, x):
        """
        Forward pass through the neural network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)

    
    def reset_parameters(self):
        """
        Resets the parameters (weights and biases) of all linear layers in the model.
        """
        def _init_weights(m):
            if isinstance(m, nn.Linear): m.reset_parameters()
                
        self.model.apply(_init_weights)

    
    def train_model(self, inputs, targets, device, F=lambda y: y, num_epochs=100, learning_rate=0.001, batch_size=32, verbose=True):
        """
        Trains the neural network to approximate conditional expectations.

        Args:
            inputs (torch.Tensor): Input tensor of shape (N*T, 2).
            targets (torch.Tensor): Target tensor of shape (N*T, 1).
            F (function): Function applied to target values.
            num_epochs (int): Number of epochs for training.
            learning_rate (float): Learning rate for the optimizer.
            batch_size (int): Batch size for training.
            verbose (bool): Whether to print training progress.

        Returns:
            list: List of batch losses.
        """
        self.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # Create dataset and data loader
        dataset = TensorDataset(inputs, targets)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        losses = []

        # Initialize the learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',      # Reduce LR when the metric has stopped decreasing
            factor=0.1,      # New LR = LR * factor
            patience=5,      # Number of epochs with no improvement to wait
        )

        for epoch in range(num_epochs):
            self.train()
            epoch_losses = []

            for batch_idx, (batch_inputs, batch_targets) in enumerate(train_loader):
                batch_inputs = batch_inputs.to(device)
                batch_targets = batch_targets.to(device)
                outputs = self.forward(batch_inputs)
                loss = criterion(outputs, F(batch_targets))
                
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping to prevent explosions (highly recommended)
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                optimizer.step()
                losses.append(loss.item())
                epoch_losses.append(loss.item())

                if verbose and batch_idx % 100 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item()}")

            # The scheduler step is based on the average loss for the entire epoch.
            average_epoch_loss = np.mean(epoch_losses)
            scheduler.step(average_epoch_loss)

            if verbose:
                # Get the current learning rate from the optimizer
                current_lr = optimizer.param_groups[0]['lr']
                print()
                print(f"Epoch Complete - Avg Loss: {average_epoch_loss:.4e} - Current LR: {current_lr:.4e}")
                print()
                
        return losses





# ==============================================================================================================
#                               SECTION 4: DEEP PICARD ITERATION FOR UNCOUPLED FBSDEs 
# ==============================================================================================================


 

class UncoupledFBSDESolver:
    """
    Solves multi-dimensional BSDEs using a choice of methods for Z.
    
    This class implements the Deep Picard Iteration method for uncoupled FBSDEs.
    It offers two distinct approaches for approximating the Z process:
    
    1.  `z_method='gradient'`: A method where Z is approximated from the
        gradient of the neural network for Y with respect to the state X.
        
    2.  `z_method='regression'`: An alternative method where Z is approximated by
        a second, separate neural network. This network is trained to learn the
        conditional expectation Z_t ~ E[(dY_t * dW_t) / dt | F_t]. 
    """

    def __init__(self,
                 sde_forward_drift, sde_forward_diffusion, 
                 d_x, d_w, d_y, 
                 F_terminal_func, driver_func, 
                 T, N, y0_X, 
                 nn_Y, device, nn_Z=None, z_method='gradient',
                 num_paths=10000, training_config=None, 
                 precomputed_X_paths=None, precomputed_delta_W_paths=None):
        """
        Initializes the multi-dimensional BSDE solver.

        Args:
            sde_forward_drift (function): Drift function for the forward SDE of X.
            sde_forward_diffusion (function): Diffusion function for the forward SDE of X.
            d_x (int): Dimension of the forward process X.
            d_w (int): Dimension of the Brownian motion W.
            d_y (int): Dimension of the backward process Y.
            F_terminal_func (function): Terminal condition F(X_T).
            driver_func (function): Driver f(t, X_t, Y_t, Z_t).
            T (float): Terminal time.
            N (int): Number of time steps.
            y0_X (np.ndarray): Initial condition for X.
            nn_Y (nn.Module): Neural network for approximating Y.
            nn_Z (nn.Module, optional): Neural network for Z, required if z_method='regression'.
            z_method (str): Method for Z: 'gradient' or 'regression'.
            num_paths (int): Number of Monte Carlo paths to simulate.
            training_config (dict, optional): Dictionary for NN training parameters.
            precomputed_X_paths (np.ndarray, optional): Pre-simulated forward paths.
            precomputed_delta_W_paths (np.ndarray, optional): Pre-simulated Brownian increments.
        """
        
        self.sde_drift_X = sde_forward_drift
        self.sde_diffusion_X = sde_forward_diffusion
        self.d_x, self.d_w, self.d_y = d_x, d_w, d_y
        self.F, self.driver_func = F_terminal_func, driver_func

        self.T, self.N, self.y0_X = T, N, np.asarray(y0_X)
        self.num_paths = num_paths
        self.dt = T / N
        
        self.z_method = z_method
        self.nn_Y = nn_Y
        self.nn_Z = nn_Z
        self.device = device
        
        # A single list to store a dictionary of results from each iteration.
        self.history = []

        self.training_config = {
            'epochs': 10, 'batch_size': 128, 'learning_rate': 0.001, 'verbose': False
        }
        if training_config:
            self.training_config.update(training_config)

        # Handle path generation or loading
        if precomputed_X_paths is not None:
            self.X_paths = torch.tensor(precomputed_X_paths, dtype=torch.float32).to(self.device)
            self.delta_W_paths = torch.tensor(precomputed_delta_W_paths, dtype=torch.float32).to(self.device)
            self.time_grid = torch.linspace(0, self.T, self.N + 1, dtype=torch.float32).to(self.device)
        else:
            self._simulate_forward_paths()

    
    def _simulate_forward_paths(self):
        """ 
        Generates forward paths for X and stores the driving Brownian increments
        by using the dedicated EulerMaruyamaND simulator class.
        """
        simulator = EulerMaruyamaND(
            drift_func=self.sde_drift_X, 
            diffusion_func=self.sde_diffusion_X,
            d_x=self.d_x, d_w=self.d_w,
            s=0, T=self.T, N=self.N, x0=self.y0_X
        )
        simulator.generate_simulations(self.num_paths)
        
        self.time_grid = torch.tensor(simulator.time_arr, dtype=torch.float32).to(self.device)
        self.X_paths = simulator.get_x_paths_as_tensor().to(self.device)
        self.delta_W_paths = simulator.get_dw_paths_as_tensor().to(self.device)

    
    def _compute_driver_sum(self, Y_tensor, Z_tensor):
        """ Computes the backward sum of the driver: S_t = sum_{j>=i} f(t_j, X_j, Y_j, Z_j)*dt """
        S = torch.zeros_like(Y_tensor, dtype=torch.float32).to(self.device)
        for i in range(self.N - 1, -1, -1):
            t_i, X_i, Y_i, Z_i = self.time_grid[i], self.X_paths[:, i, :], Y_tensor[:, i, :], Z_tensor[:, i, :, :]
            f_val = self.driver_func(t_i, X_i, Y_i, Z_i)
            S[:, i, :] = f_val * self.dt
            if i < self.N:
                S[:, i, :] += S[:, i + 1, :]
        return S

    
    def _train(self, Y_prev, Z_prev): 
        """
        Performs one full training step in the Picard iteration. This involves:
        1. Training the network for Y.
        2. If using the regression method, training the network for Z.
        This function returns the loss histories for the iteration.
        """
        # --- Stage 1: Train the network for Y ---
        S_cumulative = self._compute_driver_sum(Y_prev, Z_prev)
        terminal_Y_vals = self.F(self.X_paths[:, -1, :])

        inputs_list_Y, targets_list_Y = [], []
        for i in range(self.N):
            t_i_repeated = self.time_grid[i].repeat(self.num_paths)
            nn_input_i = torch.cat([t_i_repeated.unsqueeze(1), self.X_paths[:, i, :]], dim=1)
            inputs_list_Y.append(nn_input_i)
            targets_list_Y.append(terminal_Y_vals + S_cumulative[:, i, :])

        all_inputs_Y = torch.cat(inputs_list_Y, dim=0)
        all_targets_Y = torch.cat(targets_list_Y, dim=0)

        y_iteration_losses = self.nn_Y.train_model(
            all_inputs_Y, 
            all_targets_Y,
            self.device,
            num_epochs=self.training_config['epochs'],
            learning_rate=self.training_config['learning_rate'],
            batch_size=self.training_config['batch_size'],
            verbose=self.training_config['verbose']
        )
        
        # --- Stage 2: If using regression method, train the network for Z ---
        
        z_iteration_losses = [] 
        if self.z_method == 'regression':
            # Get an intermediate updated Y from the just-trained Y-network
            Y_new_interim = torch.zeros((self.num_paths, self.N + 1, self.d_y), dtype=torch.float32).to(self.device)
            with torch.no_grad():
                for i in range(self.N):
                    t_i_repeated = self.time_grid[i].repeat(self.num_paths)
                    nn_input_i = torch.cat([t_i_repeated.unsqueeze(1), self.X_paths[:, i, :]], dim=1)
                    Y_new_interim[:, i, :] = self.nn_Y(nn_input_i)
            Y_new_interim[:, -1, :] = terminal_Y_vals
            
            # Now, compute targets for Z using the formula Z_i ~ E[(Y_{i+1} - Y_i) * dW_i / dt | F_i]
            inputs_list_Z, targets_list_Z = [], []
            dY = Y_new_interim[:, 1:, :] - Y_new_interim[:, :-1, :]
            
            for i in range(self.N):
                t_i_repeated = self.time_grid[i].repeat(self.num_paths)
                nn_input_i = torch.cat([t_i_repeated.unsqueeze(1), self.X_paths[:, i, :]], dim=1)
                inputs_list_Z.append(nn_input_i)
                
                # Target Z_i is (Y_{i+1} - Y_i) * delta_W_i / dt
                target_Z_i = torch.bmm(dY[:, i, :].unsqueeze(2), self.delta_W_paths[:, i, :].unsqueeze(1)) / self.dt
                targets_list_Z.append(target_Z_i.reshape(self.num_paths, -1))
                
            all_inputs_Z = torch.cat(inputs_list_Z, dim=0)
            all_targets_Z = torch.cat(targets_list_Z, dim=0)
            
            z_iteration_losses = self.nn_Z.train_model(
                all_inputs_Z, 
                all_targets_Z,
                self.device,
                num_epochs=self.training_config['epochs'],
                learning_rate=self.training_config['learning_rate'],
                batch_size=self.training_config['batch_size'],
                verbose=self.training_config['verbose']
            )
            
        return y_iteration_losses, z_iteration_losses

    
    def _update(self, Y_prev, Z_prev): 
        """ Update Y and Z using the trained neural network(s) based on the chosen z_method. """
        self.nn_Y.to(self.device)
        if self.nn_Z:
            self.nn_Z.to(self.device)
            
        Y_new = torch.zeros((self.num_paths, self.N + 1, self.d_y), dtype=torch.float32).to(self.device)
        Z_new = torch.zeros((self.num_paths, self.N, self.d_y, self.d_w), dtype=torch.float32).to(self.device)

        for i in range(self.N):
            t_i_repeated = self.time_grid[i].repeat(self.num_paths)
            X_i = self.X_paths[:, i, :]
            nn_input_i = torch.cat([t_i_repeated.unsqueeze(1), X_i], dim=1)
            
            # --- Update Y ---
            with torch.no_grad():
                Y_new[:, i, :] = self.nn_Y(nn_input_i)

            # --- Update Z (method dependent) ---
            if self.z_method == 'gradient':
                X_i_grad = X_i.clone().requires_grad_(True)
                nn_input_grad = torch.cat([t_i_repeated.unsqueeze(1), X_i_grad], dim=1)
                Y_pred_grad = self.nn_Y(nn_input_grad)

                # Get the Y and Z values at time t_i from the previous iterates 
                Y_i = Y_prev[:, i, :].cpu().numpy()
                Z_i = Z_prev[:, i, :, :].cpu().numpy()

                # Call the diffusion function with all arguments
                # Required to pass in Y, Z for future class uses for coupled FBSDEs
                sigma_i = torch.tensor(np.array([
                    self.sde_diffusion_X(
                        X_i[p, :].cpu().numpy(), 
                        t_i_repeated[0].item(),
                        Y_paths=Y_i[p],  # Pass Y
                        Z_paths=Z_i[p]   # Pass Z
                    ) for p in range(self.num_paths)
                ]), dtype=torch.float32).to(self.device)
                
                jacobian_Y_X_rows = []
                for j in range(self.d_y):
                    grad_Yj = torch.autograd.grad(outputs=Y_pred_grad[:, j].sum(), inputs=X_i_grad, create_graph=True, retain_graph=True)[0]
                    jacobian_Y_X_rows.append(grad_Yj.unsqueeze(1))
                
                if jacobian_Y_X_rows:
                    jacobian_Y_X = torch.cat(jacobian_Y_X_rows, dim=1)
                    Z_new[:, i, :, :] = torch.bmm(jacobian_Y_X, sigma_i).detach()
            
            elif self.z_method == 'regression':
                with torch.no_grad():
                    Z_pred_i_flat = self.nn_Z(nn_input_i)
                Z_new[:, i, :, :] = Z_pred_i_flat.reshape(self.num_paths, self.d_y, self.d_w)

        Y_new[:, -1, :] = self.F(self.X_paths[:, -1, :]) 
        return Y_new, Z_new

    
    def solve(self, max_iter=10):
        """ Runs the main Picard iteration loop. """

        self.history = []
        
        # Initial guess for Y and Z from randomly initialized network(s) (set Y_prev, Z_prev to 0)
        self.Y, self.Z = self._update(torch.zeros(self.num_paths, self.N + 1, self.d_y), 
                                      torch.zeros(self.num_paths, self.N, self.d_y, self.d_w))
        self.Y[:, -1, :] = self.F(self.X_paths[:, -1, :])

        for k in range(max_iter):
            
            Y_prev = self.Y.clone()
            Z_prev = self.Z.clone()
            
            # Train neural network(s) and update Y and Z iterates
            y_iter_losses, z_iter_losses = self._train(Y_prev, Z_prev)
            self.Y, self.Z = self._update(Y_prev, Z_prev)

            current_error = torch.sqrt(torch.mean((self.Y - Y_prev)**2))
            
            iteration_record = {
                "iteration_number": k + 1,
                "convergence_error": current_error.item(),
                "Y_iterate": self.Y.clone(),
                "Z_iterate": self.Z.clone(),
                "nn_Y_state_dict": copy.deepcopy(self.nn_Y.state_dict()),
                "y_loss_history": y_iter_losses,
            }
            if self.z_method == 'regression' and self.nn_Z:
                iteration_record["nn_Z_state_dict"] = copy.deepcopy(self.nn_Z.state_dict())
                iteration_record["z_loss_history"] = z_iter_losses
            
            self.history.append(iteration_record)
            
            print(f"Iteration {k+1}/{max_iter} - Error: {current_error.item():.4e}")

            #self.nn_Y.reset_parameters()
            #if self.z_method=='regression': 
            #    self.nn_Z.reset_parameters()

        return self.get_solution()

        
    def get_solution(self):
        """ Returns the final solution as a dictionary of numpy arrays. """
        return {
            'time': self.time_grid.cpu().numpy(),
            'X': self.X_paths.cpu().numpy(),    
            'Y': self.Y.cpu().numpy(),          
            'Z': self.Z.cpu().numpy(),          
            'history': self.history
        }





# ==============================================================================================================
#                                SECTION 5: DEEP PICARD ITERATION FOR COUPLED FBSDEs 
# ==============================================================================================================



class CoupledFBSDESolver:
    """
    Solves coupled Forward-Backward SDEs using a global Picard iteration.

    This class approximates the solution by:
    1. Iteratively simulating the forward process X, whose dynamics depend on 
       the backward processes Y and Z from the previous global iteration.
    2. In each iteration, it creates a standard BSDESolverND instance to solve for
       the new Y and Z based on the latest X paths. This inner solver can use
       different methods ('gradient' or 'regression') to compute Z.
    """
    def __init__(self,
                 sde_forward_drift_coupled, sde_forward_diffusion_coupled,
                 d_x, d_w, d_y, F_terminal_func, driver_func,
                 T, N, y0_X,
                 nn_Y, device, nn_Z=None, z_method='gradient',
                 num_paths=1000, training_config=None):
        """
        Initializes the CoupledFBSDESolver.

        Args:
            ... 
            nn_Y (nn.Module): An un-trained instance of the NN for Y.
            nn_Z (nn.Module, optional): A untrained instance of the NN for Z. Required if 
                                        z_method='regression'.
            z_method (str): The method to be used by the inner BSDE solver for Z. 
                            Can be 'gradient' or 'regression'.
            ... 
        """
        
        # Store all parameters needed for the various steps
        self.params = {
            'sde_forward_drift': sde_forward_drift_coupled,
            'sde_forward_diffusion': sde_forward_diffusion_coupled,
            'd_x': d_x, 'd_w': d_w, 'd_y': d_y,
            'F_terminal_func': F_terminal_func, 'driver_func': driver_func,
            'T': T, 'N': N, 'y0_X': y0_X,
            'num_paths': num_paths, 'training_config': training_config,
            'z_method': z_method,  # Store the chosen method for Z
            'device': device
        }

        self.nn_Y = nn_Y
        self.nn_Z = nn_Z        
        self.global_history = []

    
    def solve(self, max_global_iter=10, inner_bsde_max_iter=3):
        """Runs the global Picard iteration to solve the coupled FBSDE."""
        p = self.params
        device = p['device']
        self.global_history=[]
        
        # Initial guess (k=0) for backward processes
        self.Y_k = torch.zeros((p['num_paths'], p['N'] + 1, p['d_y']), dtype=torch.float32).to(device)
        self.Z_k = torch.zeros((p['num_paths'], p['N'], p['d_y'], p['d_w']), dtype=torch.float32).to(device)
        
        print(f"Starting coupled FBSDE solver with Z-method: '{p['z_method']}'...")

        for k in range(max_global_iter):
            
            Y_prev_global = self.Y_k.clone()

            print(f"\n--- Global Iteration {k+1}/{max_global_iter} ---")
            
            # --- Step i: Simulate X_{k+1} and its driving dW using the coupled dynamics ---
            
            em_solver = EulerMaruyamaND(
                drift_func=p['sde_forward_drift'],
                diffusion_func=p['sde_forward_diffusion'],
                d_x=p['d_x'], d_w=p['d_w'], s=0, T=p['T'], N=p['N'], x0=p['y0_X'],
                Y_paths=self.Y_k.cpu().numpy(),
                Z_paths=self.Z_k.cpu().numpy()
            )
            em_solver.generate_simulations(p['num_paths'])
            
            # Retrieve both X paths and the Brownian increments
            self.X_k_plus_1 = em_solver.get_x_paths_as_tensor().to(device)
            self.dW_k_plus_1 = em_solver.get_dw_paths_as_tensor().to(device)

            # --- Step ii: Solve the inner uncoupled FBSDE using the new X_{k+1} and dW_{k+1} ---
            
            # Instantiate the inner solver, passing the pre-computed paths and all necessary configs
            uncoupled_solver = UncoupledFBSDESolver(
                sde_forward_drift=p['sde_forward_drift'],
                sde_forward_diffusion=p['sde_forward_diffusion'],
                d_x=p['d_x'], d_w=p['d_w'], d_y=p['d_y'],
                F_terminal_func=p['F_terminal_func'], 
                driver_func=p['driver_func'],
                T=p['T'], N=p['N'], y0_X=p['y0_X'],
                nn_Y=self.nn_Y,
                device=device,
                nn_Z=self.nn_Z,
                z_method=p['z_method'],
                num_paths=p['num_paths'],
                training_config=p['training_config'],
                precomputed_X_paths=self.X_k_plus_1.cpu().numpy(),
                precomputed_delta_W_paths=self.dW_k_plus_1.cpu().numpy()
            )
            
            solution_inner = uncoupled_solver.solve(max_iter=inner_bsde_max_iter)
            
            # Update the global iterates for Y and Z
            self.Y_k = torch.tensor(solution_inner['Y'], dtype=torch.float32).to(device)
            self.Z_k = torch.tensor(solution_inner['Z'], dtype=torch.float32).to(device)

            # Calculate and store global error for this iteration ---
            current_global_error = torch.sqrt(torch.mean((self.Y_k - Y_prev_global)**2))

            global_iteration_record = {
                "global_iteration_number": k + 1,
                "global_convergence_error": current_global_error.item(),
                "inner_solver_history": solution_inner['history'], # Store the full history from the inner solver
                "Y_iterate": self.Y_k.clone(), # Store final Y for this global step
                "Z_iterate": self.Z_k.clone(), # Store final Z for this global step
            }
            self.global_history.append(global_iteration_record)

            print(f"Global Error: {current_global_error.item():.4e}")
        
        self.X_k = self.X_k_plus_1
        return self.get_solution()

    
    def get_solution(self):
        """Return the final solution as a dictionary of numpy arrays."""
        p = self.params
        return {
            'time': np.linspace(0, p['T'], p['N'] + 1),
            'X': self.X_k.cpu().numpy(),
            'Y': self.Y_k.cpu().numpy(),
            'Z': self.Z_k.cpu().numpy(),
            'global_history': self.global_history,
        }






# ==============================================================================================================
#                               SECTION 6: DEEP PICARD ITERATION FOR MCKEAN-VLASOV FBSDEs 
# ==============================================================================================================



class McKeanVlasovFBSDESolver:
    """
    Solves McKean-Vlasov (MV) Forward-Backward SDEs using a global Picard iteration scheme.
    """

    def __init__(self,
                 sde_forward_drift_mv, sde_forward_diffusion_mv,
                 driver_func_mv, terminal_func_g_mv,
                 d_x, d_w, d_y,
                 T, N, x0,
                 nn_Y, device, nn_Z=None, z_method='gradient',
                 num_paths=1000, training_config=None):
        """
        Initializes the McKeanVlasovFBSDESolver.
        """
        self.params = {
            'sde_forward_drift_mv': sde_forward_drift_mv,
            'sde_forward_diffusion_mv': sde_forward_diffusion_mv,
            'driver_func_mv': driver_func_mv,
            'terminal_func_g_mv': terminal_func_g_mv,
            'd_x': d_x, 'd_w': d_w, 'd_y': d_y,
            'T': T, 'N': N, 'x0': x0,
            'num_paths': num_paths, 'training_config': training_config,
            'z_method': z_method,
            'device': device
        }
        self.nn_Y = nn_Y
        self.nn_Z = nn_Z
        self.global_errors = []


    def solve(self, max_global_iter, inner_bsde_max_iter):
        """
        Runs the global Picard iteration to solve the MV-FBSDE.
        """
        p = self.params
        device = p['device']
        
        # --- Initialization (k=0) ---
        self.Y_k = torch.zeros((p['num_paths'], p['N'] + 1, p['d_y']), dtype=torch.float32).to(device)
        self.Z_k = torch.zeros((p['num_paths'], p['N'], p['d_y'], p['d_w']), dtype=torch.float32).to(device)
    
        print("--- Initializing X_0 paths ---")
        initial_drift = lambda x, t, **kwargs: p['sde_forward_drift_mv'](t, x, 0, 0, 0, 0, 0)
        initial_diffusion = lambda x, t, **kwargs: p['sde_forward_diffusion_mv'](t, x, 0, 0, 0, 0, 0)
        
        em_solver_initial = EulerMaruyamaND(
            drift_func=initial_drift, diffusion_func=initial_diffusion,
            d_x=p['d_x'], d_w=p['d_w'], s=0, T=p['T'], N=p['N'], x0=p['x0']
        )
        em_solver_initial.generate_simulations(p['num_paths'])
        self.X_k = em_solver_initial.get_x_paths_as_tensor().to(device)
    
        print(f"Starting McKean-Vlasov FBSDE solver with Z-method: '{p['z_method']}'...")
    
        for k in range(max_global_iter):
            print(f"\n--- Global Iteration {k+1}/{max_global_iter} ---")
            Y_prev_global = self.Y_k.clone()
    
            # --- Convert to NumPy at the start of the loop ---
            X_k_np = self.X_k.cpu().numpy()
            Y_k_np = self.Y_k.cpu().numpy()
            Z_k_np = self.Z_k.cpu().numpy()
    
            # --- Step 1: Simulate X_{k+1} using coupled dynamics from iteration k ---
            
            print("Step 1: Simulating forward paths X_{k+1}...")
            
            def frozen_forward_drift(x, t, **kwargs):
                path_idx = kwargs.get('path_index', 0)
                time_idx = kwargs.get('time_idx', 0)
                y_t_k = Y_k_np[path_idx, time_idx]
                z_t_k = Z_k_np[path_idx, time_idx]
                return p['sde_forward_drift_mv'](t, x, y_t_k, z_t_k, X_k_np, Y_k_np, Z_k_np)
    
            def frozen_forward_diffusion(x, t, **kwargs):
                path_idx = kwargs.get('path_index', 0)
                time_idx = kwargs.get('time_idx', 0)
                y_t_k = Y_k_np[path_idx, time_idx]
                z_t_k = Z_k_np[path_idx, time_idx]
                return p['sde_forward_diffusion_mv'](t, x, y_t_k, z_t_k, X_k_np, Y_k_np, Z_k_np)
    
            em_solver = EulerMaruyamaND(
                drift_func=frozen_forward_drift,
                diffusion_func=frozen_forward_diffusion,
                d_x=p['d_x'], d_w=p['d_w'], s=0, T=p['T'], N=p['N'], x0=p['x0']
            )
            em_solver.generate_simulations(p['num_paths'])
            X_k_plus_1 = em_solver.get_x_paths_as_tensor().to(device)
            dW_k_plus_1 = em_solver.get_dw_paths_as_tensor().to(device)
    
            # --- Step 2: Solve the inner BSDE for Y_{k+1}, Z_{k+1} ---
            
            print("Step 2: Solving inner BSDE for (Y_{k+1}, Z_{k+1})...")
    
            def terminal_func_for_bsde_solver(X_T):
                return p['terminal_func_g_mv'](X_T, X_k_plus_1[:, -1, :].cpu().numpy())
    
            def driver_func_for_bsde_solver(t, X, Y, Z):
                return p['driver_func_mv'](t, X, Y, Z, X_k_np, Y_k_np, Z_k_np)
            
            bsde_solver_inner = UncoupledFBSDESolver(
                sde_forward_drift=frozen_forward_drift,
                sde_forward_diffusion=frozen_forward_diffusion,
                d_x=p['d_x'], d_w=p['d_w'], d_y=p['d_y'],
                F_terminal_func=terminal_func_for_bsde_solver,
                driver_func=driver_func_for_bsde_solver,
                T=p['T'], N=p['N'], y0_X=p['x0'],
                nn_Y=self.nn_Y, device=device, nn_Z=self.nn_Z, z_method=p['z_method'],
                num_paths=p['num_paths'], training_config=p['training_config'],
                precomputed_X_paths=X_k_plus_1.cpu().numpy(),
                precomputed_delta_W_paths=dW_k_plus_1.cpu().numpy()
            )
            
            solution_inner = bsde_solver_inner.solve(max_iter=inner_bsde_max_iter)
            
            # --- Step 3: Update global iterates for the next loop ---
            self.X_k = X_k_plus_1.clone()
            self.Y_k = torch.tensor(solution_inner['Y'], dtype=torch.float32).to(device)
            self.Z_k = torch.tensor(solution_inner['Z'], dtype=torch.float32).to(device)
    
            current_global_error = torch.sqrt(torch.mean((self.Y_k - Y_prev_global)**2))
            self.global_errors.append(current_global_error.item())
            print(f"Global Iteration {k+1} Complete. Global Error (Y_k vs Y_{k-1}): {current_global_error.item():.4e}")
        
        return self.get_solution()
        

    def get_solution(self):
        """Return the final solution as a dictionary of numpy arrays."""
        p = self.params
        return {
            'time': np.linspace(0, p['T'], p['N'] + 1),
            'X': self.X_k.cpu().numpy(),
            'Y': self.Y_k.cpu().numpy(),
            'Z': self.Z_k.cpu().numpy(),
            'global_errors': np.array(self.global_errors)
        }

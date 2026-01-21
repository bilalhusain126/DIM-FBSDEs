"""
Deep Picard Iteration Solver for Uncoupled FBSDEs.
"""

import torch
import torch.nn as nn
import numpy as np
import copy
from typing import Dict, Any, Tuple, Optional, List

from dim_fbsde.config import SolverConfig, TrainingConfig
from dim_fbsde.equations.base import FBSDE


class UncoupledFBSDESolver:
    """
    Solves uncoupled FBSDEs using the Deep Picard Iteration method.
    """

    def __init__(self,
                 equation: FBSDE,
                 solver_config: SolverConfig,
                 training_config: TrainingConfig,
                 nn_Y: nn.Module,
                 nn_Z: Optional[nn.Module] = None):
        
        self.eqn = equation
        self.cfg = solver_config
        self.train_cfg = training_config
        
        self.nn_Y = nn_Y
        self.nn_Z = nn_Z

        if self.cfg.z_method == 'regression' and self.nn_Z is None:
            raise ValueError("nn_Z must be provided when z_method is 'regression'.")

        self.device = torch.device(self.cfg.device)
        self.nn_Y.to(self.device)
        if self.nn_Z:
            self.nn_Z.to(self.device)

        # Simulation data containers
        self.time_grid: Optional[torch.Tensor] = None
        self.X: Optional[torch.Tensor] = None
        self.dW: Optional[torch.Tensor] = None
        
        # Solution iterates
        self.Y: Optional[torch.Tensor] = None
        self.Z: Optional[torch.Tensor] = None
        
        self.history: List[Dict[str, Any]] = []

    def solve(self) -> Dict[str, Any]:
        """Executes the Deep Picard Iteration loop."""
        if self.train_cfg.verbose:
            print(f"Starting Uncoupled Solver ({self.cfg.z_method})...")
        
        if self.X is None:
            self._simulate_forward_process()

        self.Y = torch.zeros(self.cfg.num_paths, self.cfg.N + 1, self.eqn.dim_y, device=self.device)
        self.Z = torch.zeros(self.cfg.num_paths, self.cfg.N, self.eqn.dim_y, self.eqn.dim_w, device=self.device)
        
        self.Y[:, -1, :] = self.eqn.terminal_condition(self.X[:, -1, :])

        # Initial guess
        self.Y, self.Z = self._update_iterates(self.Y, self.Z) 
        
        for k in range(self.cfg.picard_iterations):
            Y_prev = self.Y.clone()
            Z_prev = self.Z.clone()

            y_loss, z_loss = self._train_networks(Y_prev, Z_prev)
            self.Y, self.Z = self._update_iterates(Y_prev, Z_prev)
            
            error = torch.sqrt(torch.mean((self.Y[:, :-1, :] - Y_prev[:, :-1, :])**2)).item()
            
            if self.train_cfg.verbose:
                print(f"Picard Iter {k+1}/{self.cfg.picard_iterations}: Error={error:.4e} | Mean Y_Loss={np.mean(y_loss):.4e}")

            record = {
                "iteration": k + 1,
                "error": error,
                "y_loss": y_loss,
                "z_loss": z_loss,
                "nn_Y_state": copy.deepcopy(self.nn_Y.state_dict())
            }
            self.history.append(record)

        return self.get_results()

    def _simulate_forward_process(self):
        M = self.cfg.num_paths
        N = self.cfg.N
        dt = self.cfg.dt
        
        t_grid = torch.linspace(0, self.cfg.T, N + 1, device=self.device)
        self.time_grid = t_grid
        
        X = torch.zeros(M, N + 1, self.eqn.dim_x, device=self.device)
        dW = torch.randn(M, N, self.eqn.dim_w, device=self.device) * np.sqrt(dt)
        
        x0_tensor = torch.tensor(self.eqn.x0, dtype=torch.float32, device=self.device)
        X[:, 0, :] = x0_tensor.unsqueeze(0).repeat(M, 1)
        
        dummy_y = torch.zeros(M, self.eqn.dim_y, device=self.device)
        dummy_z = torch.zeros(M, self.eqn.dim_y, self.eqn.dim_w, device=self.device)

        for i in range(N):
            t = t_grid[i]
            x_curr = X[:, i, :]
            
            drift = self.eqn.drift(t, x_curr, dummy_y, dummy_z, T_terminal=self.cfg.T)
            sigma = self.eqn.diffusion(t, x_curr, dummy_y, dummy_z, T_terminal=self.cfg.T)
            
            diffusion_term = torch.bmm(sigma, dW[:, i, :].unsqueeze(-1)).squeeze(-1)
            X[:, i+1, :] = x_curr + drift * dt + diffusion_term
            
        self.X = X
        self.dW = dW

    def _compute_driver_sum(self, Y_tensor, Z_tensor):
        S = torch.zeros_like(Y_tensor, device=self.device)
        for i in range(self.cfg.N - 1, -1, -1):
            t = self.time_grid[i]
            X_i = self.X[:, i, :]
            Y_i = Y_tensor[:, i, :]
            Z_i = Z_tensor[:, i, :, :]
            f_val = self.eqn.driver(t, X_i, Y_i, Z_i, T_terminal=self.cfg.T)
            S[:, i, :] = f_val * self.cfg.dt
            if i < self.cfg.N:
                S[:, i, :] += S[:, i+1, :]
        return S

    def _train_networks(self, Y_prev, Z_prev):
        # --- Train Y ---
        S_cumulative = self._compute_driver_sum(Y_prev, Z_prev)
        terminal_Y = self.eqn.terminal_condition(self.X[:, -1, :], T_terminal=self.cfg.T)
        
        inputs_list, targets_list = [], []
        for i in range(self.cfg.N):
            t_vec = self.time_grid[i].repeat(self.cfg.num_paths).unsqueeze(1)
            x_vec = self.X[:, i, :]
            inputs_list.append(torch.cat([t_vec, x_vec], dim=1))
            targets_list.append(terminal_Y + S_cumulative[:, i, :])
            
        all_inputs = torch.cat(inputs_list, dim=0)
        all_targets = torch.cat(targets_list, dim=0)
        
        y_losses = self._optimize_net(self.nn_Y, all_inputs, all_targets)
        
        # --- Train Z (Regression) ---
        z_losses = []
        if self.cfg.z_method == 'regression' and self.nn_Z:
            with torch.no_grad():
                # --- FIX: Correct Reshaping of Network Output ---
                # all_inputs is stacked by Time (outer) then Path (inner).
                # Output shape is [N*M, dy].
                Y_updated_flat = self.nn_Y(all_inputs)
                
                # Reshape to [N, M, dy] (Time, Path, Dim)
                Y_updated_time_first = Y_updated_flat.reshape(self.cfg.N, self.cfg.num_paths, -1)
                
                # Permute to [M, N, dy] (Path, Time, Dim)
                Y_updated = Y_updated_time_first.permute(1, 0, 2)
                
                # Append terminal Y (which is already [M, dy])
                Y_updated = torch.cat([Y_updated, terminal_Y.unsqueeze(1)], dim=1)
                
            inputs_z, targets_z = [], []
            dY = Y_updated[:, 1:, :] - Y_updated[:, :-1, :]
            
            for i in range(self.cfg.N):
                t_vec = self.time_grid[i].repeat(self.cfg.num_paths).unsqueeze(1)
                x_vec = self.X[:, i, :]
                inputs_z.append(torch.cat([t_vec, x_vec], dim=1))
                
                target_i = torch.bmm(dY[:, i, :].unsqueeze(2), self.dW[:, i, :].unsqueeze(1)) / self.cfg.dt
                targets_z.append(target_i.reshape(self.cfg.num_paths, -1))
                
            all_inputs_z = torch.cat(inputs_z, dim=0)
            all_targets_z = torch.cat(targets_z, dim=0)
            
            z_losses = self._optimize_net(self.nn_Z, all_inputs_z, all_targets_z)
            
        return y_losses, z_losses

    def _optimize_net(self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> List[float]:
        optimizer = torch.optim.Adam(model.parameters(), lr=self.train_cfg.learning_rate)
        
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=self.train_cfg.lr_decay_step, 
            gamma=self.train_cfg.lr_decay_rate
        )
        
        loss_fn = nn.MSELoss()
        
        dataset_size = inputs.shape[0]
        indices = torch.arange(dataset_size)
        model.train()
        losses = []
        
        for _ in range(self.train_cfg.epochs):
            shuffled_idx = indices[torch.randperm(dataset_size)]
            
            for start_idx in range(0, dataset_size, self.train_cfg.batch_size):
                batch_idx = shuffled_idx[start_idx : start_idx + self.train_cfg.batch_size]
                batch_in = inputs[batch_idx]
                batch_target = targets[batch_idx]
                
                optimizer.zero_grad()
                pred = model(batch_in)
                loss = loss_fn(pred, batch_target)
                loss.backward()
                
                if self.train_cfg.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.train_cfg.gradient_clip_val)
                    
                optimizer.step()
                scheduler.step()
                
                losses.append(loss.item())
        return losses

    def _update_iterates(self, Y_prev, Z_prev) -> Tuple[torch.Tensor, torch.Tensor]:
        self.nn_Y.eval()
        if self.nn_Z: self.nn_Z.eval()
        
        Y_new = torch.zeros_like(Y_prev)
        Z_new = torch.zeros_like(Z_prev)
        M = self.cfg.num_paths
        
        for i in range(self.cfg.N):
            t = self.time_grid[i]
            t_vec = t.repeat(M).unsqueeze(1)
            X_i = self.X[:, i, :]
            
            nn_in = torch.cat([t_vec, X_i], dim=1)
            
            with torch.no_grad():
                Y_new[:, i, :] = self.nn_Y(nn_in)
            
            if self.cfg.z_method == 'gradient':
                X_i_grad = X_i.clone().requires_grad_(True)
                nn_input_grad = torch.cat([t_vec, X_i_grad], dim=1)
                Y_pred_grad = self.nn_Y(nn_input_grad)
                
                with torch.no_grad():
                    sigma_i = self.eqn.diffusion(t, X_i, Y_prev[:,i,:], Z_prev[:,i,:,:], T_terminal=self.cfg.T)
                
                jacobian_rows = []
                for j in range(self.eqn.dim_y):
                    grad_Yj = torch.autograd.grad(
                        outputs=Y_pred_grad[:, j].sum(),
                        inputs=X_i_grad,
                        create_graph=True, 
                        retain_graph=True
                    )[0]
                    jacobian_rows.append(grad_Yj.unsqueeze(1))
                
                if jacobian_rows:
                    jacobian_Y_X = torch.cat(jacobian_rows, dim=1)
                    Z_new[:, i, :, :] = torch.bmm(jacobian_Y_X, sigma_i).detach()
                    
            elif self.cfg.z_method == 'regression':
                with torch.no_grad():
                    Z_pred_flat = self.nn_Z(nn_in)
                Z_new[:, i, :, :] = Z_pred_flat.reshape(M, self.eqn.dim_y, self.eqn.dim_w)
                
        Y_new[:, -1, :] = self.eqn.terminal_condition(self.X[:, -1, :], T_terminal=self.cfg.T)
        return Y_new, Z_new

    def get_results(self) -> Dict[str, Any]:
        return {
            'time': self.time_grid.cpu().numpy(),
            'X': self.X.cpu().numpy(),
            'Y': self.Y.cpu().numpy(),
            'Z': self.Z.cpu().numpy(),
            'history': self.history
        }

    def set_external_paths(self, X_paths: np.ndarray, dW_paths: np.ndarray):
        self.X = torch.tensor(X_paths, dtype=torch.float32, device=self.device)
        self.dW = torch.tensor(dW_paths, dtype=torch.float32, device=self.device)
        M, N_plus_1, _ = self.X.shape
        self.cfg.num_paths = M
        self.cfg.N = N_plus_1 - 1
        self.time_grid = torch.linspace(0, self.cfg.T, self.cfg.N + 1, device=self.device)
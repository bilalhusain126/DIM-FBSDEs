"""
Deep Iterative Method for McKean-Vlasov FBSDEs.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from dim_fbsde.config import SolverConfig, TrainingConfig
from dim_fbsde.equations.base import FBSDE
from dim_fbsde.solvers.uncoupled_solver import UncoupledFBSDESolver

class McKeanVlasovSolver:
    """
    Solves McKean-Vlasov (Mean-Field) FBSDEs using a Global Picard Iteration.
    """

    def __init__(self,
                 equation: FBSDE,
                 solver_config: SolverConfig,
                 training_config: TrainingConfig,
                 nn_Y: torch.nn.Module,
                 nn_Z: Optional[torch.nn.Module] = None):
        
        self.eqn = equation
        self.cfg = solver_config
        self.train_cfg = training_config
        
        self.nn_Y = nn_Y
        self.nn_Z = nn_Z
        self.device = torch.device(self.cfg.device)

        self.X: Optional[torch.Tensor] = None
        self.Y_path: Optional[torch.Tensor] = None
        self.Z_path: Optional[torch.Tensor] = None
        self.global_history: List[Dict[str, float]] = []


    def solve(self) -> Dict[str, Any]:
        if self.train_cfg.verbose:
            print(f"Starting McKean-Vlasov Solver (Global Iters: {self.cfg.global_iterations})...")

        # 1. Initialization (k=0)
        M, N = self.cfg.num_paths, self.cfg.N
        self.Y_path = torch.zeros(M, N + 1, self.eqn.dim_y, device=self.device)
        self.Z_path = torch.zeros(M, N, self.eqn.dim_y, self.eqn.dim_w, device=self.device)
        
        # Initial X simulation (no law)
        self.X, _ = self._simulate_mv_forward(is_initial=True)

        # 2. Global Loop
        for k in range(self.cfg.global_iterations):
            Y_prev_global = self.Y_path.clone()

            if self.train_cfg.verbose:
                print(f"\n--- Global Iteration {k+1}/{self.cfg.global_iterations} ---")

            # --- Step A: Forward Pass ---
            X_new, dW_new = self._simulate_mv_forward(is_initial=False)
            
            # --- Step B: Backward Pass (Inner Solver) ---
            inner_solver = UncoupledFBSDESolver(
                equation=self.eqn,
                solver_config=self.cfg,
                training_config=self.train_cfg,
                nn_Y=self.nn_Y,
                nn_Z=self.nn_Z
            )
            
            inner_solver.set_external_paths(X_new.cpu().numpy(), dW_new.cpu().numpy())
            
            # --- Runtime Proxy Setup ---
            # 1. Capture the ORIGINAL bound methods first.
            # This ensures we call the real logic, not the proxy (avoiding recursion).
            original_driver = self.eqn.driver
            original_terminal = self.eqn.terminal_condition

            # 2. Define Proxies using the CAPTURED methods.
            def driver_with_law_proxy(t, x, y, z, **kwargs):
                # Calculate time index from t
                idx = int(round(t.item() / self.cfg.dt))
                idx = min(idx, self.cfg.N)
                
                # Full distribution from previous iteration
                dist_x = self.X[:, idx, :] 
                dist_y = self.Y_path[:, idx, :]
                
                # Call the CAPTURED original method
                return original_driver(t, x, y, z, mean_x=dist_x, mean_y=dist_y, **kwargs)

            def terminal_with_law_proxy(x, **kwargs):
                # Full distribution at terminal time from NEW paths
                dist_x_T = X_new[:, -1, :]
                # Call the CAPTURED original method
                return original_terminal(x, mean_x=dist_x_T, **kwargs)

            # 3. Apply the Monkey-Patch to the equation instance shared with inner_solver
            inner_solver.eqn.driver = driver_with_law_proxy
            inner_solver.eqn.terminal_condition = terminal_with_law_proxy

            try:
                # Solve the uncoupled BSDE with the "frozen" law coefficients
                solution_inner = inner_solver.solve()
            finally:
                # 4. Restore original methods
                inner_solver.eqn.driver = original_driver
                inner_solver.eqn.terminal_condition = original_terminal

            # --- Step C: Update ---
            self.X = torch.tensor(solution_inner['X'], device=self.device)
            self.Y_path = torch.tensor(solution_inner['Y'], device=self.device)
            self.Z_path = torch.tensor(solution_inner['Z'], device=self.device)

            diff = self.Y_path - Y_prev_global
            global_error = torch.sqrt(torch.mean(diff**2)).item()
            
            if self.train_cfg.verbose:
                print(f"Global Error (Y_k vs Y_k-1): {global_error:.4e}")
            
            self.global_history.append({'iteration': k+1, 'global_error': global_error})

        return self.get_results()


    def _simulate_mv_forward(self, is_initial: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        M = self.cfg.num_paths
        N = self.cfg.N
        dt = self.cfg.dt
        
        t_grid = torch.linspace(0, self.cfg.T, N + 1, device=self.device)
        
        X = torch.zeros(M, N + 1, self.eqn.dim_x, device=self.device)
        dW = torch.randn(M, N, self.eqn.dim_w, device=self.device) * np.sqrt(dt)
        
        x0_tensor = torch.tensor(self.eqn.x0, dtype=torch.float32, device=self.device)
        X[:, 0, :] = x0_tensor.unsqueeze(0).repeat(M, 1)

        for i in range(N):
            t = t_grid[i]
            x_curr = X[:, i, :]
            
            mean_x = None
            mean_y = None
            
            y_curr = torch.zeros(M, self.eqn.dim_y, device=self.device)
            z_curr = torch.zeros(M, self.eqn.dim_y, self.eqn.dim_w, device=self.device)

            if not is_initial:
                y_curr = self.Y_path[:, i, :]
                z_curr = self.Z_path[:, i, :, :]
                
                # Pass the FULL DISTRIBUTION (batch) to the equation
                mean_x = self.X[:, i, :] 
                mean_y = self.Y_path[:, i, :]

            # 'mean_x' argument carries the full batch or None
            drift = self.eqn.drift(t, x_curr, y_curr, z_curr, 
                                   mean_x=mean_x, mean_y=mean_y, 
                                   T_terminal=self.cfg.T)
                                   
            sigma = self.eqn.diffusion(t, x_curr, y_curr, z_curr, 
                                       mean_x=mean_x, mean_y=mean_y, 
                                       T_terminal=self.cfg.T)
            
            diffusion_term = torch.bmm(sigma, dW[:, i, :].unsqueeze(-1)).squeeze(-1)
            X[:, i+1, :] = x_curr + drift * dt + diffusion_term
            
        return X, dW

    def get_results(self) -> Dict[str, Any]:
        return {
            'time': np.linspace(0, self.cfg.T, self.cfg.N + 1),
            'X': self.X.cpu().numpy(),
            'Y': self.Y_path.cpu().numpy(),
            'Z': self.Z_path.cpu().numpy(),
            'global_history': self.global_history
        }
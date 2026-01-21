"""
Deep Iterative Method for Coupled FBSDEs.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, List, Tuple  # Added Tuple here

from dim_fbsde.config import SolverConfig, TrainingConfig
from dim_fbsde.equations.base import FBSDE
from dim_fbsde.solvers.uncoupled_solver import UncoupledFBSDESolver


class CoupledFBSDESolver:
    """
    Solves coupled FBSDEs using a Global Picard Iteration.

    In a coupled system, the forward process X depends on the backward processes Y and Z.
    This creates a circular dependency. This solver resolves it by iterating:
    
    1.  Forward Pass: Simulate X^{(k+1)} using approximations Y^{(k)}, Z^{(k)}.
    2.  Backward Pass: Solve for Y^{(k+1)}, Z^{(k+1)} by treating X^{(k+1)} as 
        fixed inputs to an uncoupled BSDE solver.
    """

    def __init__(self,
                 equation: FBSDE,
                 solver_config: SolverConfig,
                 training_config: TrainingConfig,
                 nn_Y: torch.nn.Module,
                 nn_Z: Optional[torch.nn.Module] = None):
        """
        Args:
            equation: The Coupled FBSDE system.
            solver_config: Global solver settings.
            training_config: Training settings for the inner uncoupled solver.
            nn_Y: Neural network for Y.
            nn_Z: Neural network for Z.
        """
        self.eqn = equation
        self.cfg = solver_config
        self.train_cfg = training_config
        
        self.nn_Y = nn_Y
        self.nn_Z = nn_Z
        self.device = torch.device(self.cfg.device)

        # Internal Uncoupled Solver acts as the "Backward Step" engine
        # We share the same networks with it.
        self.inner_solver = UncoupledFBSDESolver(
            equation=equation,
            solver_config=solver_config,
            training_config=training_config,
            nn_Y=nn_Y,
            nn_Z=nn_Z
        )

        # Global storage
        self.X: Optional[torch.Tensor] = None
        self.Y_path: Optional[torch.Tensor] = None
        self.Z_path: Optional[torch.Tensor] = None
        self.global_history: List[Dict[str, float]] = []

    
    def solve(self) -> Dict[str, Any]:
        """
        Executes the Global Picard Iteration.
        """
        if self.train_cfg.verbose:
            print(f"Starting Coupled Solver (Global Iters: {self.cfg.global_iterations}, Inner Iters: {self.cfg.picard_iterations})...")

        # 1. Initialization (k=0)
        # Initialize paths to zero for the first forward pass
        M, N = self.cfg.num_paths, self.cfg.N
        self.Y_path = torch.zeros(M, N + 1, self.eqn.dim_y, device=self.device)
        self.Z_path = torch.zeros(M, N, self.eqn.dim_y, self.eqn.dim_w, device=self.device)

        # 2. Global Loop (Outer Iterations)
        for k in range(self.cfg.global_iterations):
            Y_prev_global = self.Y_path.clone()

            if self.train_cfg.verbose:
                print(f"\n--- Global Iteration {k+1}/{self.cfg.global_iterations} ---")

            # Step A: Simulate Forward Process X^{(k+1)}
            # The drift/diffusion now depend on Y_path and Z_path from previous step
            X_new, dW_new = self._simulate_coupled_forward()
            
            # Step B: Solve Backward Process (Inner Solver)
            # We inject the specific X paths we just generated into the inner solver
            self.inner_solver.set_external_paths(X_new.cpu().numpy(), dW_new.cpu().numpy())
            
            # Run the inner Deep Picard solver
            # This will run for `cfg.picard_iterations` (Inner Iterations)
            solution_inner = self.inner_solver.solve()
            
            # Update global paths
            self.X = torch.tensor(solution_inner['X'], device=self.device)
            self.Y_path = torch.tensor(solution_inner['Y'], device=self.device)
            self.Z_path = torch.tensor(solution_inner['Z'], device=self.device)

            # Step C: Convergence Check
            diff = self.Y_path - Y_prev_global
            global_error = torch.sqrt(torch.mean(diff**2)).item()
            
            if self.train_cfg.verbose:
                print(f"Global Error (Y_k vs Y_k-1): {global_error:.4e}")
            
            self.global_history.append({'iteration': k+1, 'global_error': global_error})

        return self.get_results()

    
    def _simulate_coupled_forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simulates X where drift/diffusion depend on the stored Y_path and Z_path.
        """
        M = self.cfg.num_paths
        N = self.cfg.N
        dt = self.cfg.dt
        
        t_grid = torch.linspace(0, self.cfg.T, N + 1, device=self.device)
        
        X = torch.zeros(M, N + 1, self.eqn.dim_x, device=self.device)
        dW = torch.randn(M, N, self.eqn.dim_w, device=self.device) * np.sqrt(dt)
        
        # Initial condition
        x0_tensor = torch.tensor(self.eqn.x0, dtype=torch.float32, device=self.device)
        X[:, 0, :] = x0_tensor.unsqueeze(0).repeat(M, 1)

        for i in range(N):
            t = t_grid[i]
            x_curr = X[:, i, :]
            
            # Use the stored estimates Y^{(k)}, Z^{(k)} at time i
            y_curr = self.Y_path[:, i, :]
            z_curr = self.Z_path[:, i, :, :]

            # Evaluate coefficients with coupling
            drift = self.eqn.drift(t, x_curr, y_curr, z_curr, T_terminal=self.cfg.T)
            sigma = self.eqn.diffusion(t, x_curr, y_curr, z_curr, T_terminal=self.cfg.T)
            
            # Euler-Maruyama
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
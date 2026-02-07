"""
Standard Benchmark FBSDE problems.

This module implements the analytical definitions for various Forward-Backward 
Stochastic Differential Equations (FBSDEs) used as benchmarks in the thesis. 
Each class defines the coefficients, the driver function, the terminal condition, 
and the known analytical solution for validation. All implementations utilize PyTorch 
operations to ensure hardware acceleration and automatic differentiation support.


References:
- Thesis Chapter 9: Numerical Experiments
"""

import torch
import math
from typing import Optional, Union

from dim_fbsde.equations.base import FBSDE


# ==============================================================================
# 1. Black-Scholes-Barenblatt (BSB) Equation
#    Thesis Section 9.2.2: Uncoupled FBSDE
# ==============================================================================

class BSBEquation(FBSDE):
    """
    High-dimensional Black-Scholes-Barenblatt equation.
    
    Dynamics:
        Forward: Geometric Brownian Motion (dX_t = sigma * X_t * dW_t)
        Backward: Nonlinear driver representing a recursive utility or hedging problem.
    """
    def __init__(self, 
                 dim_x: int = 1, 
                 r: float = 0.05, 
                 sigma: float = 0.4,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float32):
        
        super().__init__(dim_x=dim_x, dim_y=1, dim_w=dim_x, 
                         x0=torch.ones(dim_x), device=device, dtype=dtype)
        self.r = r
        self.sigma_val = sigma

    def drift(self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Zero drift (martingale dynamics)."""
        return torch.zeros_like(x)

    def diffusion(self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Diagonal diffusion matrix: sigma * diag(X_t)."""
        return torch.diag_embed(self.sigma_val * x)

    def driver(self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Nonlinear driver: f = -r * (y - (1/sigma) * sum(z))."""
        # z shape: [Batch, 1, dim_w] -> sum over dim_w -> [Batch, 1]
        sum_z = z.sum(dim=2)
        return -self.r * y + (self.r / self.sigma_val) * sum_z

    def terminal_condition(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Payoff function: ||X_T||^2."""
        return torch.sum(x.pow(2), dim=1, keepdim=True)

    def analytical_y(self, t: torch.Tensor, x: torch.Tensor, **kwargs) -> Optional[torch.Tensor]:
        """Exact solution for Y_t."""
        T = kwargs.get('T_terminal', 1.0)
        norm_x_sq = torch.sum(x.pow(2), dim=1, keepdim=True)
        # Since t is already a tensor, (T - t) is also a tensor on the correct device
        return torch.exp((self.r + self.sigma_val**2) * (T - t)) * norm_x_sq

    def analytical_z(self, t: torch.Tensor, x: torch.Tensor, **kwargs) -> Optional[torch.Tensor]:
        """Exact solution for Z_t."""
        T = kwargs.get('T_terminal', 1.0)
        # Since t is already a tensor, (T - t) is also a tensor on the correct device
        exp_term = torch.exp((self.r + self.sigma_val**2) * (T - t))
        # Z is [Batch, 1, dim_w]
        return (2 * self.sigma_val * exp_term * x.pow(2)).unsqueeze(1)


# ==============================================================================
# 2. Decoupled FBSDE System
#    Thesis Section 9.2.1: Uncoupled FBSDE
# ==============================================================================

class HureEquation(FBSDE):
    """
    Decoupled FBSDE benchmark with trigonometric nonlinearity.
    """
    def __init__(self, 
                 dim_x: int = 1,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float32):
        
        super().__init__(dim_x=dim_x, dim_y=1, dim_w=dim_x, 
                         x0=torch.ones(dim_x), device=device, dtype=dtype)
        
        self.mu_scalar = 0.2 / dim_x
        self.sigma_scalar = 1.0 / math.sqrt(dim_x)
        
        # Pre-allocate identity matrix for diffusion
        self._identity = torch.eye(dim_x, device=self.device, dtype=self.dtype)

    def drift(self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.full_like(x, self.mu_scalar)

    def diffusion(self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, **kwargs) -> torch.Tensor:
        # Expand identity to batch size: [Batch, dim_x, dim_x]
        return self._identity.expand(x.shape[0], -1, -1) * self.sigma_scalar

    def driver(self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, **kwargs) -> torch.Tensor:
        T = kwargs.get('T_terminal', 1.0)
        
        z_sum = z.sum(dim=2) # [Batch, 1]
        x_sum = x.sum(dim=1, keepdim=True)
        
        term_exp = torch.exp((T - t) / 2.0)
        term_trig = torch.cos(x_sum) + 0.2 * torch.sin(x_sum)
        
        term1 = term_trig * term_exp
        term2 = 0.5 * (torch.exp(T - t) * torch.sin(x_sum) * torch.cos(x_sum)).pow(2)
        term3 = 0.5 * (y * z_sum).pow(2) / self.dim_x
        
        return term1 - term2 + term3

    def terminal_condition(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.cos(x.sum(dim=1, keepdim=True))

    def analytical_y(self, t: torch.Tensor, x: torch.Tensor, **kwargs) -> Optional[torch.Tensor]:
        T = kwargs.get('T_terminal', 1.0)
        return torch.exp((T - t) / 2.0) * torch.cos(x.sum(dim=1, keepdim=True))

    def analytical_z(self, t: torch.Tensor, x: torch.Tensor, **kwargs) -> Optional[torch.Tensor]:
        T = kwargs.get('T_terminal', 1.0)
        x_sum = x.sum(dim=1, keepdim=True)
        scalar = -self.sigma_scalar * torch.exp((T - t) / 2.0) * torch.sin(x_sum)
        # Expand scalar to [Batch, 1, dim_w]
        return scalar.unsqueeze(1).repeat(1, 1, self.dim_w)


# ==============================================================================
# 3. Z-Coupled FBSDE
#    Thesis Section 9.3.1: Coupled FBSDE
# ==============================================================================

class ZCoupledEquation(FBSDE):
    """
    Coupled FBSDE where forward drift depends on Z.
    """
    def __init__(self, 
                 dim_x: int = 1,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float32):
        
        super().__init__(dim_x=dim_x, dim_y=1, dim_w=dim_x, 
                         x0=torch.full((dim_x,), 0.5), device=device, dtype=dtype)
        
        self._identity = torch.eye(dim_x, device=self.device, dtype=self.dtype)

    def drift(self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, **kwargs) -> torch.Tensor:
        x_sum = x.sum(dim=1, keepdim=True)
        
        if z is None:
            z_sum = torch.zeros_like(x_sum)
        else:
            z_sum = z.sum(dim=2) # [Batch, 1]

        S = t + x_sum
        coupling = torch.sin(S).pow(2) + (z_sum / self.dim_x)
        scalar = -0.5 * torch.sin(S) * torch.cos(S) * coupling
        return scalar.repeat(1, self.dim_x)

    def diffusion(self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, **kwargs) -> torch.Tensor:
        x_sum = x.sum(dim=1, keepdim=True)
        scalar = torch.cos(t + x_sum).unsqueeze(-1) # [Batch, 1, 1]
        # Return [Batch, dim_x, dim_x]
        return scalar * self._identity.expand(x.shape[0], -1, -1)

    def driver(self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, **kwargs) -> torch.Tensor:
        x_sum = x.sum(dim=1, keepdim=True)
        if z is None:
            z_sum = torch.zeros_like(x_sum)
        else:
            z_sum = z.sum(dim=2)
            
        return y * z_sum - torch.cos(t + x_sum)

    def terminal_condition(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        T = kwargs.get('T_terminal', 1.0)
        return torch.sin(T + x.sum(dim=1, keepdim=True))

    def analytical_y(self, t: torch.Tensor, x: torch.Tensor, **kwargs) -> Optional[torch.Tensor]:
        return torch.sin(t + x.sum(dim=1, keepdim=True))

    def analytical_z(self, t: torch.Tensor, x: torch.Tensor, **kwargs) -> Optional[torch.Tensor]:
        x_sum = x.sum(dim=1, keepdim=True)
        scalar = torch.cos(t + x_sum).pow(2)
        return scalar.unsqueeze(1).repeat(1, 1, self.dim_w)


# ==============================================================================
# 4. Fully-Coupled FBSDE
#    Thesis Section 9.3.2: Coupled FBSDE
# ==============================================================================

class FullyCoupledEquation(FBSDE):
    """
    Fully Coupled FBSDE where both Drift and Diffusion depend on (Y, Z).
    """
    def __init__(self, 
                 dim_x: int = 1,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float32):
        
        super().__init__(dim_x=dim_x, dim_y=1, dim_w=dim_x, 
                         x0=torch.zeros(dim_x), device=device, dtype=dtype)
        
        self._identity = torch.eye(dim_x, device=self.device, dtype=self.dtype)

    def drift(self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, **kwargs) -> torch.Tensor:
        x_sum = x.sum(dim=1, keepdim=True)
        # Handle initialization/null cases
        if y is None: y = torch.zeros_like(x_sum)
        if z is None: z_sum = torch.zeros_like(x_sum)
        else: z_sum = z.sum(dim=2)

        S = t + x_sum
        term = y.pow(2) + (z_sum / self.dim_x)
        scalar = -0.5 * torch.sin(S) * torch.cos(S) * term
        return scalar.repeat(1, self.dim_x)

    def diffusion(self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, **kwargs) -> torch.Tensor:
        x_sum = x.sum(dim=1, keepdim=True)
        if y is None: y = torch.zeros_like(x_sum)
        if z is None: z_sum = torch.zeros_like(x_sum)
        else: z_sum = z.sum(dim=2)
            
        S = t + x_sum
        term = y * torch.sin(S) + (z_sum / self.dim_x) + 1.0
        scalar = 0.5 * torch.cos(S) * term
        
        return scalar.unsqueeze(-1) * self._identity.expand(x.shape[0], -1, -1)

    def driver(self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, **kwargs) -> torch.Tensor:
        x_sum = x.sum(dim=1, keepdim=True)
        if z is None: z_sum = torch.zeros_like(x_sum)
        else: z_sum = z.sum(dim=2)
        return y * z_sum - torch.cos(t + x_sum)

    def terminal_condition(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        T = kwargs.get('T_terminal', 1.0)
        return torch.sin(T + x.sum(dim=1, keepdim=True))
    
    def analytical_y(self, t: torch.Tensor, x: torch.Tensor, **kwargs) -> Optional[torch.Tensor]:
        return torch.sin(t + x.sum(dim=1, keepdim=True))

    def analytical_z(self, t: torch.Tensor, x: torch.Tensor, **kwargs) -> Optional[torch.Tensor]:
        x_sum = x.sum(dim=1, keepdim=True)
        scalar = torch.cos(t + x_sum).pow(2)
        return scalar.unsqueeze(1).repeat(1, 1, self.dim_w)


# ==============================================================================
# 5. McKean-Vlasov FBSDE System
#    Thesis Section 9.4.1: Mean-Field FBSDE
# ==============================================================================

class McKeanVlasovEquation(FBSDE):
    """
    McKean-Vlasov FBSDE with distribution-dependent drift.
    """
    def __init__(self, 
                 dim_x: int = 1,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float32):
        
        super().__init__(dim_x=dim_x, dim_y=1, dim_w=dim_x, 
                         x0=torch.zeros(dim_x), device=device, dtype=dtype)
        
        self._identity = torch.eye(dim_x, device=self.device, dtype=self.dtype)

    def drift(self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Kwargs:
            mean_x (torch.Tensor): Full batch of X paths from previous iteration [M, dim_x].
            mean_y (torch.Tensor): Full batch of Y paths from previous iteration [M, dim_y].
        """
        prev_x_dist = kwargs.get('mean_x') 
        prev_y_dist = kwargs.get('mean_y') 
        
        # --- Term 1: Interaction Kernel ---
        if prev_x_dist is None:
            # Initialization case
            term1 = torch.sin(torch.tensor(1.0, device=self.device, dtype=self.dtype))
        else:
            # Calculate pairwise squared distances: [Batch_curr, Batch_prev]
            # Warning: High memory usage for large batches.
            dists_sq = torch.cdist(x, prev_x_dist, p=2).pow(2)
            
            # Expectation over the distribution (dim=1)
            kernel_expectations = torch.mean(torch.exp(-dists_sq / self.dim_x), dim=1, keepdim=True)
            term1 = torch.sin(kernel_expectations) 

        # --- Term 2: Potential Term ---
        x_norm_sq = torch.sum(x.pow(2), dim=1, keepdim=True)
        denom = max(self.dim_x + 2 * t.item(), 1e-8)
             
        time_factor = (self.dim_x / denom)**(self.dim_x / 2.0)
        term2 = -torch.exp(-x_norm_sq / self.dim_x) * time_factor

        # --- Term 3: Mean-Field Coupling ---
        if prev_y_dist is None:
            exp_y = 0.0
        else:
            exp_y = torch.mean(prev_y_dist) # Scalar mean of Y

        term3 = 0.5 * (exp_y - torch.sin(t) * torch.exp(-t/2.0))

        return term1 + term2 + term3

    def diffusion(self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._identity.expand(x.shape[0], -1, -1)

    def driver(self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, **kwargs) -> torch.Tensor:
        if z is None:
            z_sum = torch.zeros(x.shape[0], 1, device=self.device, dtype=self.dtype)
            z_norm_sq = torch.zeros(x.shape[0], 1, device=self.device, dtype=self.dtype)
        else:
            z_sum = z.sum(dim=2)
            z_norm_sq = z.pow(2).sum(dim=2)

        sqrt_term = torch.clamp(y.pow(2) + z_norm_sq + 1, min=1e-8).sqrt()
        val = (1.0 / math.sqrt(self.dim_x)) * z_sum - y / 2.0 + sqrt_term - math.sqrt(2.0)
        return -val

    def terminal_condition(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        T = kwargs.get('T_terminal', 1.0)
        x_sum = x.sum(dim=1, keepdim=True)
        return torch.sin(T + (1.0 / math.sqrt(self.dim_x)) * x_sum)

    def analytical_y(self, t: torch.Tensor, x: torch.Tensor, **kwargs) -> Optional[torch.Tensor]:
        x_sum = x.sum(dim=1, keepdim=True)
        return torch.sin(t + (1.0 / math.sqrt(self.dim_x)) * x_sum)

    def analytical_z(self, t: torch.Tensor, x: torch.Tensor, **kwargs) -> Optional[torch.Tensor]:
        x_sum = x.sum(dim=1, keepdim=True)
        scalar = (1.0 / math.sqrt(self.dim_x)) * torch.cos(t + (1.0 / math.sqrt(self.dim_x)) * x_sum)
        return scalar.unsqueeze(1).repeat(1, 1, self.dim_w)
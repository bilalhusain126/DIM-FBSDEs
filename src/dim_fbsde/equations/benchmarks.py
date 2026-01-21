"""
Standard Benchmark FBSDE problems.
Refactored to match the exact numerical implementation of the original thesis code.
Includes fixes for broadcasting during visualization (Time-vector vs State-matrix).
"""

import torch
import numpy as np
import math
from typing import Optional

from dim_fbsde.equations.base import FBSDE


# ==============================================================================
# 1. Uncoupled Benchmark: Black-Scholes-Barenblatt
# ==============================================================================

class BSBEquation(FBSDE):
    """
    High-dimensional BSB equation (Thesis Eq 9.2.6).
    """
    def __init__(self, dim_x: int = 1, r: float = 0.05, sigma: float = 0.4):
        # In BSB, dim_x = dim_w, and dim_y = 1
        super().__init__(dim_x=dim_x, dim_y=1, dim_w=dim_x, x0=np.ones(dim_x))
        self.r = r
        self.sigma_val = sigma

    def drift(self, t, x, y, z, **kwargs):
        return torch.zeros_like(x)

    def diffusion(self, t, x, y, z, **kwargs):
        return torch.diag_embed(self.sigma_val * x)

    def driver(self, t, x, y, z, **kwargs):
        # z shape is [Batch, dim_y, dim_w] -> [Batch, 1, d]
        sum_Z_components = z.sum(dim=2) # Result is [Batch, 1]
        driver_val = -self.r * y + (self.r / self.sigma_val) * sum_Z_components
        return driver_val

    def terminal_condition(self, x, **kwargs):
        return torch.sum(x.pow(2), dim=1, keepdim=True)

    def analytical_y(self, t: float, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Analytic solution for Y_t.
        """
        T = kwargs.get('T', 1.0)
        
        # Handle shapes
        if x.ndim == 1:
            norm_S_sq = np.sum(x**2)
        else:
            norm_S_sq = np.sum(x**2, axis=1)
            
        val = np.exp((self.r + self.sigma_val**2) * (T - t)) * norm_S_sq
        
        # Ensure correct broadcast shape for plotting
        if isinstance(val, np.ndarray) and val.ndim == 1:
            return val[:, np.newaxis]
        return val

    def analytical_z(self, t: float, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Analytic solution for Z_t.
        Handles broadcasting for t (vector) vs x (matrix).
        """
        T = kwargs.get('T', 1.0)
        
        # t might be a scalar or a numpy array
        exp_term = np.exp((self.r + self.sigma_val**2) * (T - t))
        
        # Broadcasting Fix:
        # If t is a vector (N,) and x is (N, d), we need exp_term to be (N, 1)
        if isinstance(t, np.ndarray) and t.ndim == 1 and x.ndim == 2:
            exp_term = exp_term[:, np.newaxis]
        
        return 2 * self.sigma_val * exp_term * (x**2)


# ==============================================================================
# 2. Uncoupled Benchmark: Hure et al. (2020)
# ==============================================================================

class HureEquation(FBSDE):
    """
    Decoupled FBSDE benchmark (Thesis Eq 9.2.3).
    """
    def __init__(self, dim_x: int = 1):
        super().__init__(dim_x=dim_x, dim_y=1, dim_w=dim_x, x0=np.ones(dim_x))
        self.mu_scalar = 0.2 / dim_x
        self.sigma_scalar = 1.0 / math.sqrt(dim_x)

    def drift(self, t, x, y, z, **kwargs):
        return torch.full_like(x, self.mu_scalar)

    def diffusion(self, t, x, y, z, **kwargs):
        batch_size = x.shape[0]
        eye = torch.eye(self.dim_x, device=x.device).unsqueeze(0).repeat(batch_size, 1, 1)
        return self.sigma_scalar * eye

    def driver(self, t, x, y, z, **kwargs):
        T = kwargs.get('T_terminal', 1.0)
        
        if z.ndim == 4: 
             z_sum = torch.sum(z, dim=-1).squeeze(-1)
        else:
             z_sum = torch.sum(z, dim=-1)
        
        if z_sum.ndim == 1: z_sum = z_sum.unsqueeze(1)

        x_sum = torch.sum(x, dim=1, keepdim=True)
        
        term_exp = torch.exp((T - t) / 2.0)
        term_trig = torch.cos(x_sum) + 0.2 * torch.sin(x_sum)
        
        term1 = term_trig * term_exp
        term2 = 0.5 * (torch.exp(T - t) * torch.sin(x_sum) * torch.cos(x_sum))**2
        term3 = 0.5 * (y * z_sum)**2
        
        return term1 - term2 + term3

    def terminal_condition(self, x, **kwargs):
        x_sum = torch.sum(x, dim=1, keepdim=True)
        return torch.cos(x_sum)

    def analytical_y(self, t: float, x: np.ndarray, **kwargs) -> np.ndarray:
        T = kwargs.get('T', 1.0)
        x_sum = np.sum(x, axis=-1)
        
        # Expand dims for scalar t vs vector x case if needed
        # But usually numpy handles scalar t fine.
        # If t is vector (N,) and x is (N, d), x_sum is (N,) -> OK.
        
        val = np.exp((T - t) / 2.0) * np.cos(x_sum)
        
        if isinstance(val, np.ndarray) and val.ndim == 1:
            return val[:, np.newaxis]
        return val

    def analytical_z(self, t: float, x: np.ndarray, **kwargs) -> np.ndarray:
        T = kwargs.get('T', 1.0)
        x_sum = np.sum(x, axis=-1)
        
        sigma = 1.0 / np.sqrt(self.dim_x)
        scalar = -sigma * np.exp((T - t) / 2.0) * np.sin(x_sum)
        
        # Broadcasting logic
        if x.ndim == 1:
            # Single state vector
            return np.full((self.dim_w,), scalar)
        else:
            # x is (N, d) or (Batch, d)
            # scalar is (N,) or (Batch,)
            if scalar.ndim == 1:
                scalar = scalar[:, np.newaxis] # (N, 1)
            
            # Broadcast (N, 1) to (N, d)
            return np.tile(scalar, (1, self.dim_w))


# ==============================================================================
# 3. Coupled Benchmark: Z-Coupled (Hwang et al.)
# ==============================================================================

class ZCoupledEquation(FBSDE):
    """
    Z-Coupled FBSDE (Thesis Eq 9.3.1).
    """
    def __init__(self, dim_x: int = 1):
        super().__init__(dim_x=dim_x, dim_y=1, dim_w=dim_x, x0=np.full(dim_x, 0.5))

    def drift(self, t, x, y, z, **kwargs):
        x_sum = torch.sum(x, dim=1, keepdim=True)
        if z is None:
            z_sum = torch.zeros_like(x_sum)
        else:
            z_sum = torch.sum(z, dim=-1)
            if z_sum.ndim == 1: z_sum = z_sum.unsqueeze(1)

        S = t + x_sum
        coupling = torch.sin(S)**2 + (z_sum / self.dim_x)
        scalar = -0.5 * torch.sin(S) * torch.cos(S) * coupling
        return scalar.repeat(1, self.dim_x)

    def diffusion(self, t, x, y, z, **kwargs):
        x_sum = torch.sum(x, dim=1, keepdim=True)
        scalar = torch.cos(t + x_sum).unsqueeze(-1)
        eye = torch.eye(self.dim_x, device=x.device).unsqueeze(0)
        return scalar * eye

    def driver(self, t, x, y, z, **kwargs):
        x_sum = torch.sum(x, dim=1, keepdim=True)
        if z is None:
            z_sum = torch.zeros_like(x_sum)
        else:
            z_sum = torch.sum(z, dim=-1)
            if z_sum.ndim == 1: z_sum = z_sum.unsqueeze(1)
            
        return y * z_sum - torch.cos(t + x_sum)

    def terminal_condition(self, x, **kwargs):
        T = kwargs.get('T_terminal', 1.0)
        x_sum = torch.sum(x, dim=1, keepdim=True)
        return torch.sin(T + x_sum)

    def analytical_y(self, t: float, x: np.ndarray, **kwargs) -> np.ndarray:
        x_sum = np.sum(x, axis=-1)
        if x.ndim > 1 and x_sum.ndim == 1: x_sum = x_sum[:, np.newaxis]
        
        # Broadcasting t if necessary
        if isinstance(t, np.ndarray) and t.ndim == 1 and x_sum.ndim == 2:
             t = t[:, np.newaxis]
             
        return np.sin(t + x_sum)

    def analytical_z(self, t: float, x: np.ndarray, **kwargs) -> np.ndarray:
        x_sum = np.sum(x, axis=-1) # (N,) if path
        
        if isinstance(t, np.ndarray) and t.ndim == 1:
             # t is (N,), x_sum is (N,)
             pass
        
        scalar_val = np.cos(t + x_sum)**2
        
        if x.ndim == 1:
            return np.full((self.dim_w,), scalar_val)
        else:
            if scalar_val.ndim == 1: scalar_val = scalar_val[:, np.newaxis]
            return np.tile(scalar_val, (1, self.dim_w))


# ==============================================================================
# 4. Fully-Coupled Benchmark: Ji et al.
# ==============================================================================

class FullyCoupledEquation(FBSDE):
    """
    Fully Coupled FBSDE (Thesis Eq 9.3.2).
    """
    def __init__(self, dim_x: int = 1):
        super().__init__(dim_x=dim_x, dim_y=1, dim_w=dim_x, x0=np.zeros(dim_x))

    def drift(self, t, x, y, z, **kwargs):
        x_sum = torch.sum(x, dim=1, keepdim=True)
        if y is None: y = torch.zeros_like(x_sum)
        if z is None: 
            z_sum = torch.zeros_like(x_sum)
        else:
            z_sum = torch.sum(z, dim=-1)
            if z_sum.ndim == 1: z_sum = z_sum.unsqueeze(1)

        S = t + x_sum
        term = y**2 + (z_sum / self.dim_x)
        scalar = -0.5 * torch.sin(S) * torch.cos(S) * term
        return scalar.repeat(1, self.dim_x)

    def diffusion(self, t, x, y, z, **kwargs):
        x_sum = torch.sum(x, dim=1, keepdim=True)
        if y is None: y = torch.zeros_like(x_sum)
        if z is None: 
            z_sum = torch.zeros_like(x_sum)
        else:
            z_sum = torch.sum(z, dim=-1)
            if z_sum.ndim == 1: z_sum = z_sum.unsqueeze(1)
            
        S = t + x_sum
        term = y * torch.sin(S) + (z_sum / self.dim_x) + 1.0
        scalar = 0.5 * torch.cos(S) * term
        
        eye = torch.eye(self.dim_x, device=x.device).unsqueeze(0)
        return scalar.unsqueeze(-1) * eye

    def driver(self, t, x, y, z, **kwargs):
        x_sum = torch.sum(x, dim=1, keepdim=True)
        if z is None:
            z_sum = torch.zeros_like(x_sum)
        else:
            z_sum = torch.sum(z, dim=-1)
            if z_sum.ndim == 1: z_sum = z_sum.unsqueeze(1)
        return y * z_sum - torch.cos(t + x_sum)

    def terminal_condition(self, x, **kwargs):
        T = kwargs.get('T_terminal', 1.0)
        x_sum = torch.sum(x, dim=1, keepdim=True)
        return torch.sin(T + x_sum)
    
    def analytical_y(self, t: float, x: np.ndarray, **kwargs) -> np.ndarray:
        x_sum = np.sum(x, axis=-1)
        if x.ndim > 1 and x_sum.ndim == 1: x_sum = x_sum[:, np.newaxis]
        if isinstance(t, np.ndarray) and t.ndim == 1 and x_sum.ndim == 2:
             t = t[:, np.newaxis]
        return np.sin(t + x_sum)

    def analytical_z(self, t: float, x: np.ndarray, **kwargs) -> np.ndarray:
        x_sum = np.sum(x, axis=-1)
        scalar_val = np.cos(t + x_sum)**2
        
        if x.ndim == 1:
            return np.full((self.dim_w,), scalar_val)
        else:
            if scalar_val.ndim == 1: scalar_val = scalar_val[:, np.newaxis]
            return np.tile(scalar_val, (1, self.dim_w))


# ==============================================================================
# 5. McKean-Vlasov Benchmark: Han et al. 
# ==============================================================================

class McKeanVlasovEquation(FBSDE):
    """
    McKean-Vlasov FBSDE (Thesis Eq 9.4.1).
    """
    def __init__(self, dim_x: int = 1):
        super().__init__(dim_x=dim_x, dim_y=1, dim_w=dim_x, x0=np.zeros(dim_x))

    def drift(self, t, x, y, z, **kwargs):
        # We need the full previous distribution X' to compute the interaction kernel
        # kwargs['mean_x'] will now contain the FULL PREVIOUS BATCH (Shape: [M, d])
        prev_x_dist = kwargs.get('mean_x') 
        prev_y_dist = kwargs.get('mean_y') # Actually just need mean for Y term
        
        # --- Term 1: Interaction Kernel sin(E[exp(-|x-x'|^2/d)]) ---
        if prev_x_dist is None:
            # Initialization case (matches screenshot 'else' block)
            # Assume X_t = X'_t = 0 => exp(0) = 1
            term1 = torch.sin(torch.tensor(1.0, device=x.device))
        else:
            # x shape: [M, d]
            # prev_x_dist shape: [M, d]
            # We need pairwise squared distances.
            # Using torch.cdist is efficient: dist matrix [M, M]
            # squared distance: dist^2
            
            # Optimization: If M is large, cdist can be heavy. 
            # But it matches the logic of "diffs = x_np - all_X_at_t" from screenshot.
            
            d = self.dim_x
            
            # Compute pairwise distance squared: |x_i - x'_j|^2
            # Result shape: [M, M]
            dists_sq = torch.cdist(x, prev_x_dist, p=2)**2
            
            # Compute expectation over the second dimension (j)
            # E[exp(-|x-x'|^2/d)]
            kernel_expectations = torch.mean(torch.exp(-dists_sq / d), dim=1, keepdim=True)
            
            term1 = torch.sin(kernel_expectations) # Shape [M, 1]

        # --- Term 2: Standard Term ---
        x_norm_sq = torch.sum(x**2, dim=1, keepdim=True)
        d = self.dim_x
        # Avoid division by zero at t=0 if d=0 (unlikely)
        denom = d + 2*t
        if isinstance(denom, torch.Tensor):
             # Ensure no division by zero at strict t=0 if dynamic
             denom = torch.clamp(denom, min=1e-8)
        else:
             denom = max(denom, 1e-8)
             
        time_factor = (d / denom)**(d/2.0)
        term2 = -torch.exp(-x_norm_sq / d) * time_factor

        # --- Term 3: Law of Y Term ---
        # 0.5 * (E[Y] - sin(t)e^(-t/2))
        if prev_y_dist is None:
            exp_y = 0.0
        else:
            # If prev_y_dist is the full batch, take the mean
            if prev_y_dist.numel() > 1:
                exp_y = torch.mean(prev_y_dist)
            else:
                exp_y = prev_y_dist

        term3 = 0.5 * (exp_y - torch.sin(t) * torch.exp(-t/2.0))

        return term1 + term2 + term3

    def diffusion(self, t, x, y, z, **kwargs):
        batch_size = x.shape[0]
        eye = torch.eye(self.dim_x, device=x.device).unsqueeze(0).repeat(batch_size, 1, 1)
        return eye

    def driver(self, t, x, y, z, **kwargs):
        if z is None:
            z_sum = torch.zeros(x.shape[0], 1, device=x.device)
            z_norm_sq = torch.zeros(x.shape[0], 1, device=x.device)
        else:
            # Sum over d_w dimension
            z_sum = torch.sum(z, dim=-1)
            z_norm_sq = torch.sum(z**2, dim=-1)
            
            if z_sum.ndim == 1: z_sum = z_sum.unsqueeze(1)
            if z_norm_sq.ndim == 1: z_norm_sq = z_norm_sq.unsqueeze(1)

        d = self.dim_x
        sqrt_term = torch.clamp(y**2 + z_norm_sq + 1, min=1e-8).sqrt()
        val = (1.0 / math.sqrt(d)) * z_sum - y / 2.0 + sqrt_term - math.sqrt(2.0)
        return -val

    def terminal_condition(self, x, **kwargs):
        T = kwargs.get('T_terminal', 1.0)
        x_sum = torch.sum(x, dim=1, keepdim=True)
        return torch.sin(T + (1.0 / math.sqrt(self.dim_x)) * x_sum)

    def analytical_y(self, t: float, x: np.ndarray, **kwargs) -> np.ndarray:
        x_sum = np.sum(x, axis=-1)
        if x.ndim > 1 and x_sum.ndim == 1: x_sum = x_sum[:, np.newaxis]
        if isinstance(t, np.ndarray) and t.ndim == 1 and x_sum.ndim == 2:
             t = t[:, np.newaxis]
        return np.sin(t + (1.0 / np.sqrt(self.dim_x)) * x_sum)

    def analytical_z(self, t: float, x: np.ndarray, **kwargs) -> np.ndarray:
        x_sum = np.sum(x, axis=-1)
        scalar = (1.0 / np.sqrt(self.dim_x)) * np.cos(t + (1.0 / np.sqrt(self.dim_x)) * x_sum)
        if x.ndim == 1:
            return np.full((self.dim_w,), scalar)
        else:
            if scalar.ndim == 1: scalar = scalar[:, np.newaxis]
            return np.tile(scalar, (1, self.dim_w))
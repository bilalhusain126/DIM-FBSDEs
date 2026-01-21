"""
Abstract Base Class for FBSDE definitions.
Strictly defines the interface for system dynamics.
"""

from abc import ABC, abstractmethod
import torch
from typing import Optional
import numpy as np

class FBSDE(ABC):
    """
    Abstract Base Class representing a general FBSDE.
    
    System Dynamics:
        Forward:  dX_t = mu(...) dt + sigma(...) dW_t
        Backward: -dY_t = f(...) dt - Z_t dW_t
        Terminal: Y_T  = g(X_T)
    """

    def __init__(self, dim_x: int, dim_y: int, dim_w: int, x0: np.ndarray):
        """
        Args:
            dim_x: Dimension of forward state X.
            dim_y: Dimension of backward value Y.
            dim_w: Dimension of Brownian motion W.
            x0: Initial condition for X (as numpy array).
        """
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_w = dim_w
        self._x0 = x0

    @property
    def x0(self) -> np.ndarray:
        return self._x0

    @abstractmethod
    def drift(self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward drift coefficient mu(t, x, y, z).
        Returns shape: [Batch, dim_x]
        """
        pass

    @abstractmethod
    def diffusion(self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward diffusion coefficient sigma(t, x, y, z).
        Returns shape: [Batch, dim_x, dim_w]
        """
        pass

    @abstractmethod
    def driver(self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Backward driver f(t, x, y, z).
        Returns shape: [Batch, dim_y]
        """
        pass

    @abstractmethod
    def terminal_condition(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Terminal condition g(X_T).
        Returns shape: [Batch, dim_y]
        """
        pass
    
    def analytical_solution(self, t: torch.Tensor, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Optional: Analytical solution for validation.
        Returns Y(t, x).
        """
        return None

    def __repr__(self):
        return f"{self.__class__.__name__}(dx={self.dim_x}, dy={self.dim_y}, dw={self.dim_w})"
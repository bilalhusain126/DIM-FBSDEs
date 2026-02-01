"""
Abstract Base Class for FBSDE definitions.
Strictly defines the interface for system dynamics.
"""

from abc import ABC, abstractmethod
import torch
from typing import Optional, Union, Any

class FBSDE(ABC):
    """
    Abstract Base Class representing a general FBSDE.
    
    System Dynamics:
        Forward:  dX_t = mu(t, X_t, Y_t, Z_t) dt + sigma(t, X_t, Y_t, Z_t) dW_t
        Backward: -dY_t = f(t, X_t, Y_t, Z_t) dt - Z_t dW_t
        Terminal: Y_T  = g(X_T)
    """

    def __init__(self,
                 dim_x: int,
                 dim_y: int,
                 dim_w: int,
                 x0: Union[torch.Tensor, float, list],
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float32):
        """
        Args:
            dim_x: Dimension of forward state X.
            dim_y: Dimension of backward value Y.
            dim_w: Dimension of Brownian motion W.
            x0: Initial condition for X.
            device: Computation device (e.g., 'cpu', 'cuda:0').
            dtype: Data type for tensor computations (default: torch.float32).

        Raises:
            ValueError: If dimensions are not positive integers.
        """
        # Input validation
        if dim_x <= 0 or dim_y <= 0 or dim_w <= 0:
            raise ValueError(f"Dimensions must be positive. Got dim_x={dim_x}, dim_y={dim_y}, dim_w={dim_w}")

        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_w = dim_w
        self.device = torch.device(device)
        self.dtype = dtype

        # Standardize x0 to a generic tensor on the correct device
        if not isinstance(x0, torch.Tensor):
            x0 = torch.tensor(x0, dtype=self.dtype, device=self.device)
        else:
            x0 = x0.to(dtype=self.dtype, device=self.device)
            
        # Ensure shape [1, dim_x] for broadcasting
        self._x0 = x0.view(1, -1) if x0.dim() == 1 else x0

    @property
    def x0(self) -> torch.Tensor:
        """Returns initial state with shape [1, dim_x]."""
        return self._x0

    @abstractmethod
    def drift(self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward drift coefficient mu(t, x, y, z).

        Args:
            t: Time tensor [Batch, 1] or scalar.
            x: State tensor [Batch, dim_x].
            y: Value tensor [Batch, dim_y].
            z: Control tensor [Batch, dim_y, dim_w].
            **kwargs: Additional context (e.g., 'mean_x' for MV-FBSDEs, 'T_terminal').

        Returns:
            Tensor of shape [Batch, dim_x].
        """
        pass

    @abstractmethod
    def diffusion(self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward diffusion coefficient sigma(t, x, y, z).

        Args:
            t: Time tensor [Batch, 1] or scalar.
            x: State tensor [Batch, dim_x].
            y: Value tensor [Batch, dim_y].
            z: Control tensor [Batch, dim_y, dim_w].

        Returns:
            Tensor of shape [Batch, dim_x, dim_w].
        """
        pass

    @abstractmethod
    def driver(self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Backward driver f(t, x, y, z).

        Args:
            t: Time tensor [Batch, 1] or scalar.
            x: State tensor [Batch, dim_x].
            y: Value tensor [Batch, dim_y].
            z: Control tensor [Batch, dim_y, dim_w].

        Returns:
            Tensor of shape [Batch, dim_y].
        """
        pass

    @abstractmethod
    def terminal_condition(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Terminal condition g(X_T).

        Args:
            x: State tensor [Batch, dim_x].

        Returns:
            Tensor of shape [Batch, dim_y].
        """
        pass

    def analytical_y(self, t: torch.Tensor, x: torch.Tensor, **kwargs) -> Optional[torch.Tensor]:
        """
        Computes the analytical solution for Y(t, x).

        Args:
            t: Time tensor [Batch, 1] or scalar.
            x: State tensor [Batch, dim_x].

        Returns:
            Tensor [Batch, dim_y] or None.
        """
        return None

    def analytical_z(self, t: torch.Tensor, x: torch.Tensor, **kwargs) -> Optional[torch.Tensor]:
        """
        Computes the analytical solution for Z(t, x).

        Args:
            t: Time tensor [Batch, 1] or scalar.
            x: State tensor [Batch, dim_x].

        Returns:
            Tensor [Batch, dim_y, dim_w] or None.
        """
        return None

    def __repr__(self):
        return f"<{self.__class__.__name__} dx={self.dim_x} dy={self.dim_y} dw={self.dim_w} on {self.device}>"
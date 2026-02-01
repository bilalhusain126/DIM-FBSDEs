"""
Configuration dataclasses for the DIM FBSDE solvers.
"""

from dataclasses import dataclass
from typing import Literal

@dataclass
class TrainingConfig:
    """
    Hyperparameters for the neural network training loop.
    """
    batch_size: int = 500
    epochs: int = 5          # Gradient descent steps per Picard iteration
    learning_rate: float = 1e-4

    lr_decay_step: int = 1000
    lr_decay_rate: float = 0.95

    verbose: bool = True
    gradient_clip_val: float = 1.0

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.lr_decay_step <= 0:
            raise ValueError(f"lr_decay_step must be positive, got {self.lr_decay_step}")
        if not 0 < self.lr_decay_rate <= 1:
            raise ValueError(f"lr_decay_rate must be in (0, 1], got {self.lr_decay_rate}")
        if self.gradient_clip_val < 0:
            raise ValueError(f"gradient_clip_val must be non-negative, got {self.gradient_clip_val}")


@dataclass
class SolverConfig:
    """
    Hyperparameters for the numerical solver physics and methods.
    """
    T: float = 1.0
    N: int = 120

    num_paths: int = 2000

    # INNER Iterations: How many times the Uncoupled solver refines Y/Z given a fixed X path.
    picard_iterations: int = 10

    # OUTER Iterations: How many times the Coupled solver alternates between simulating X and solving Y/Z.
    global_iterations: int = 10

    z_method: Literal['gradient', 'regression'] = 'gradient'

    device: str = "cpu"
    dtype: str = "float32"

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.T <= 0:
            raise ValueError(f"T (terminal time) must be positive, got {self.T}")
        if self.N <= 0:
            raise ValueError(f"N (time steps) must be positive, got {self.N}")
        if self.num_paths <= 0:
            raise ValueError(f"num_paths must be positive, got {self.num_paths}")
        if self.picard_iterations <= 0:
            raise ValueError(f"picard_iterations must be positive, got {self.picard_iterations}")
        if self.global_iterations <= 0:
            raise ValueError(f"global_iterations must be positive, got {self.global_iterations}")
        if self.z_method not in ['gradient', 'regression']:
            raise ValueError(f"z_method must be 'gradient' or 'regression', got {self.z_method}")

    @property
    def dt(self) -> float:
        return self.T / self.N
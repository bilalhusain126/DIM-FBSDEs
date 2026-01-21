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

    @property
    def dt(self) -> float:
        return self.T / self.N
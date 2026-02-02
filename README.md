# Deep Iterative Method for Forward-Backward SDEs

A PyTorch implementation of the Deep Picard Iteration method for solving high-dimensional Forward-Backward Stochastic Differential Equations (FBSDEs), including uncoupled, coupled, and McKean-Vlasov systems. Developed as part of a Master's thesis at the University of Toronto.

## Overview

This package provides efficient GPU-accelerated solvers for various classes of FBSDEs:

- **Uncoupled FBSDEs**: Standard forward-backward systems
- **Coupled FBSDEs**: Systems where forward dynamics depend on backward processes
- **McKean-Vlasov FBSDEs**: Mean-field systems with distribution-dependent coefficients

The implementation leverages PyTorch for automatic differentiation and GPU acceleration, enabling efficient numerical solutions for high-dimensional problems.

## Features

- **Multiple Solver Types**: Uncoupled, Coupled, and McKean-Vlasov FBSDEs
- **GPU Acceleration**: Full CUDA support via PyTorch
- **Benchmark Problems**: Five standard test equations with analytical solutions
  - Black-Scholes-Barenblatt (BSB)
  - Hure et al. decoupled system
  - Z-Coupled FBSDE
  - Fully-Coupled FBSDE
  - McKean-Vlasov mean-field system
- **Two Solving Methods**: Gradient-based and regression-based Z approximation
- **Neural Network Architecture**: Customizable multi-layer perceptrons
- **Visualization Suite**: Pathwise comparison and error analysis plotting
- **Type-Safe**: Comprehensive type hints throughout codebase
- **Well-Documented**: Detailed docstrings and usage examples

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/bilalhusain126/DIM-FBSDEs.git
cd DIM-FBSDEs

# Install the package
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0.0
- NumPy ≥ 1.24.0
- Matplotlib ≥ 3.7.0

## Quick Start

### Example: Solving Black-Scholes-Barenblatt Equation

```python
import torch
from dim_fbsde.equations import BSBEquation
from dim_fbsde.solvers import UncoupledFBSDESolver
from dim_fbsde.nets import MLP
from dim_fbsde.config import SolverConfig, TrainingConfig

# Configure device
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Define the equation
dim_x = 10
equation = BSBEquation(dim_x=dim_x, r=0.05, sigma=0.4, device=device)

# 2. Configure solver
solver_cfg = SolverConfig(
    T=1.0,                      # Terminal time
    N=120,                      # Time steps
    num_paths=2000,             # Monte Carlo paths
    picard_iterations=10,       # Fixed-point iterations
    z_method='regression',      # Z estimation method
    device=device
)

# 3. Configure training
train_cfg = TrainingConfig(
    batch_size=500,
    epochs=5,
    learning_rate=1e-4,
    verbose=True
)

# 4. Create neural networks
input_dim = 1 + dim_x  # (time, state)
nn_Y = MLP(input_dim=input_dim, output_dim=1, hidden_dims=[40, 40, 40, 40])
nn_Z = MLP(input_dim=input_dim, output_dim=dim_x, hidden_dims=[40, 40, 40, 40])

# 5. Solve
solver = UncoupledFBSDESolver(equation, solver_cfg, train_cfg, nn_Y, nn_Z)
solution = solver.solve()

# 6. Access results
print(f"Solution shape: X={solution['X'].shape}, Y={solution['Y'].shape}")
```

### Visualization

```python
from dim_fbsde.utils import plot_pathwise_comparison

# Compare numerical and analytical solutions
fig, axes = plot_pathwise_comparison(
    solution=solution,
    analytical_Y_func=lambda t, x, **kw: equation.analytical_y(t, x, **kw),
    analytical_Z_func=lambda t, x, **kw: equation.analytical_z(t, x, **kw),
    component_idx=0
)
```

## Package Structure

```
dim_fbsde/
├── equations/          # FBSDE problem definitions
│   ├── base.py        # Abstract base class
│   └── benchmarks.py  # Standard test problems
├── solvers/           # Numerical solvers
│   ├── uncoupled.py   # Deep Picard for uncoupled systems
│   ├── coupled.py     # Global iteration for coupled systems
│   └── mckean_vlasov.py  # Solver for mean-field systems
├── nets/              # Neural network architectures
│   └── mlp.py         # Multi-layer perceptron
├── utils/             # Utilities
│   └── visualizations.py  # Plotting functions
└── config.py          # Configuration dataclasses
```

## Benchmark Problems

### Uncoupled Systems

1. **Black-Scholes-Barenblatt (BSB)**: High-dimensional option pricing
2. **Hure et al.**: Decoupled system with trigonometric nonlinearity

### Coupled Systems

3. **Z-Coupled**: Forward drift depends on control Z
4. **Fully-Coupled**: Both drift and diffusion depend on (Y, Z)

### Mean-Field Systems

5. **McKean-Vlasov**: Distribution-dependent coefficients

## Advanced Usage

### Custom Equations

```python
from dim_fbsde.equations.base import FBSDE
import torch

class MyCustomEquation(FBSDE):
    def __init__(self, dim_x, **kwargs):
        super().__init__(dim_x=dim_x, dim_y=1, dim_w=dim_x,
                         x0=torch.ones(dim_x), **kwargs)

    def drift(self, t, x, y, z, **kwargs):
        # Define your forward drift
        return torch.zeros_like(x)

    def diffusion(self, t, x, y, z, **kwargs):
        # Define your forward diffusion
        return torch.diag_embed(x)

    def driver(self, t, x, y, z, **kwargs):
        # Define your backward driver
        return -y

    def terminal_condition(self, x, **kwargs):
        # Define terminal condition
        return torch.sum(x**2, dim=1, keepdim=True)
```

### Logging Configuration

```python
import logging

# Set custom log level
logging.getLogger('dim_fbsde').setLevel(logging.DEBUG)

# Or disable logging
logging.getLogger('dim_fbsde').setLevel(logging.WARNING)
```

## Algorithm Details

### Deep Picard Iteration

The solver implements the Deep Picard iteration scheme, which iteratively refines neural network approximations of the solution processes Y and Z. The method alternates between:

1. **Path simulation**: Generate Monte Carlo paths of the forward SDE
2. **Network training**: Optimize neural networks to satisfy the BSDE via Picard iteration
3. **Convergence**: Iterate until the change between successive approximations is small

### Z Estimation Methods

Two approaches are available for estimating the control process Z:

- **Gradient Method** (`z_method='gradient'`): Computes Z via automatic differentiation as Z = σ(t,X,Y,Z) · ∇Y. More accurate but slower to train.
- **Regression Method** (`z_method='regression'`): Directly approximates Z with a separate neural network. Faster training but requires more parameters.

## Performance Tips

1. **GPU Acceleration**: CUDA support provides significant speedup for high-dimensional problems
2. **Batch Size**: Larger batches (500-1000) improve GPU utilization
3. **Network Architecture**: Start with 3-4 hidden layers of 40-64 units each
4. **Picard Iterations**: 5-10 iterations typically sufficient for convergence
5. **Method Selection**: Use 'regression' for faster training, 'gradient' for higher accuracy

## Examples

See the `notebooks/` directory for comprehensive demonstrations:
- `benchmark_demonstrations.ipynb`: Complete examples for all five benchmark problems with visualization and error analysis

## Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{husain2025fbsde,
  author  = {Bilal Saleh Husain},
  title   = {Deep Iterative Methods for Forward-Backward Stochastic Differential Equations},
  school  = {University of Toronto},
  year    = {2025}
}
```

## License

MIT License - see LICENSE file for details.

## Author

**Bilal Saleh Husain**
Master of Applied Science, Computational Finance
University of Toronto, 2025

## References

This implementation is based on recent advances in deep learning methods for FBSDEs:

- Han, J., Jentzen, A., & E, W. (2018). Solving high-dimensional partial differential equations using deep learning. *Proceedings of the National Academy of Sciences*, 115(34), 8505-8510.
- Hure, C., Pham, H., & Warin, X. (2020). Deep backward schemes for high-dimensional nonlinear PDEs. *Mathematics of Computation*, 89(324), 1547-1579.
- Raissi, M. (2018). Forward-backward stochastic neural networks: Deep learning of high-dimensional partial differential equations. *arXiv preprint arXiv:1804.07010*.
- Ji, S., Peng, S., & Zhou, C. (2021). Coupled FBSDEs and deep learning. *arXiv preprint arXiv:2102.04242*.

---

**Version**: 1.0.0 | **License**: MIT

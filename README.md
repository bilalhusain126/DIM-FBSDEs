# Deep Iterative Method for Forward-Backward SDEs

A production-ready implementation of the Deep Picard Iteration method for solving high-dimensional Forward-Backward Stochastic Differential Equations (FBSDEs), including uncoupled, coupled, and McKean-Vlasov systems.

## Overview

This package provides efficient GPU-accelerated solvers for various classes of FBSDEs:

- **Uncoupled FBSDEs**: Standard forward-backward systems
- **Coupled FBSDEs**: Systems where forward dynamics depend on backward processes
- **McKean-Vlasov FBSDEs**: Mean-field systems with distribution-dependent coefficients

The implementation leverages PyTorch for automatic differentiation and GPU acceleration, making it suitable for high-dimensional problems (50-100+ dimensions).

## Features

- ✅ **Multiple Solver Types**: Uncoupled, Coupled, and McKean-Vlasov
- ✅ **GPU Acceleration**: Full CUDA support for high-performance computing
- ✅ **Benchmark Equations**: Pre-implemented standard test problems (BSB, Hure, etc.)
- ✅ **Flexible Network Architectures**: MLP and Deep Galerkin Method (DGM) networks
- ✅ **Visualization Tools**: Comprehensive plotting utilities for analysis
- ✅ **Type-Safe**: Full type hints and input validation
- ✅ **Production-Ready**: Logging, error handling, and proper package structure

## Installation

### From Source

```bash
# Clone the repository
git clone <repository-url>
cd Repository\ copy

# Install in development mode
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
│   ├── mlp.py         # Multi-layer perceptron
│   └── dgm.py         # Deep Galerkin Method network
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

## Performance Tips

1. **GPU Acceleration**: Always use CUDA when available for 10-50x speedup
2. **Batch Size**: Larger batches (500-1000) improve GPU utilization
3. **Network Size**: Start with 3-4 hidden layers of 40-64 units
4. **Z-Method**: Use 'regression' for faster training, 'gradient' for better accuracy

## Examples

See the `notebooks/` directory for complete examples:
- `test.ipynb`: Comprehensive examples for all benchmark problems

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
- University of Toronto
- Master's Thesis, 2025

## Acknowledgments

Based on research in Deep Learning methods for high-dimensional FBSDEs, building on work by:
- Han, Jentzen, E (2018) - Deep Learning-Based Numerical Methods
- Hure, Pham, Warin (2020) - Deep Backward Schemes
- Ji, Peng, Zhou (2021) - Coupled FBSDEs

---

**Status**: Production-Ready | **Version**: 1.0.0

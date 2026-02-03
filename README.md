# Deep Iterative Method for Forward-Backward SDEs

This repository contains the PyTorch implementation of the **Deep Iterative Method**, a numerical framework for solving high-dimensional Forward-Backward Stochastic Differential Equations (FBSDEs). This code accompanies the Master's thesis **"A Deep Iterative Method for High-Dimensional Coupled and McKean-Vlasov FBSDEs"** (University of Toronto, 2025).

## Overview

This package solves general FBSDE systems of the form:

$$
\begin{cases}
dX_t = \mu(t, X_t, Y_t, Z_t) dt + \sigma(t, X_t, Y_t, Z_t) dW_t, & X_0 = x_0 \\
-dY_t = f(t, X_t, Y_t, Z_t) dt - Z_t dW_t, & Y_T = g(X_T)
\end{cases}
$$

where $X_t \in \mathbb{R}^d$ is the forward process, $Y_t \in \mathbb{R}^m$ is the backward process, and $Z_t \in \mathbb{R}^{m \times d}$ is the control process.

The solver overcomes the curse of dimensionality by approximating the solution maps $(t, x) \mapsto Y_t$ and $(t, x) \mapsto Z_t$ using deep neural networks. The architecture relies on a **Deep Picard Iteration**, effectively treating the solution as the fixed point of a contraction mapping on the space of stochastic processes.

## Key Capabilities

*   **Hierarchical Solver Support:**
    *   **Uncoupled:** Standard systems where forward dynamics are independent of $(Y, Z)$.
    *   **Coupled:** Systems where $X_t$ depends on $Y_t$ and $Z_t$, resolved via a Global Picard Iteration.
    *   **McKean-Vlasov:** Mean-field systems where coefficients depend on the law $\mathcal{L}(X_t, Y_t, Z_t)$.
*   **Rigorous Benchmarking:** Includes five standard test equations with analytical solutions for the different problem classes. 
*   **Two Z-Approximation Schemes:** Supports both **Gradient-based** and **Regression-based** approximation for the control process.
*   **GPU Accelerated:** Fully vectorized PyTorch implementation supporting CUDA execution for high-dimensional problems.



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

The following example solves the high-dimensional **Black-Scholes-Barenblatt** equation ($d=3$) using the Deep Picard Iteration method.

```python
import torch
from dim_fbsde.equations import BSBEquation
from dim_fbsde.solvers import UncoupledFBSDESolver
from dim_fbsde.nets import MLP
from dim_fbsde.config import SolverConfig, TrainingConfig

# Configure computation device
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Define System Dynamics 
dim_x = 3
equation = BSBEquation(dim_x=dim_x, r=0.05, sigma=0.4, device=device)

# 2. Configure Solver
solver_cfg = SolverConfig(
    T=1.0,                      # Terminal time
    N=120,                      # Time discretization steps
    num_paths=2000,             # Monte Carlo trajectories
    picard_iterations=10,       # Fixed-point iterations
    z_method='regression',      # Control approximation scheme
    device=device
)

# 3. Configure Training
train_cfg = TrainingConfig(
    batch_size=500,
    epochs=5,
    learning_rate=1e-4,
    verbose=True
)

# 4. Initialize Function Approximators
# Input: (t, x) -> dim 1 + dim_x
# Output: Y (dim 1) and Z (dim_x)
input_dim = 1 + dim_x  
nn_Y = MLP(input_dim=input_dim, output_dim=1, hidden_dims=[64, 64, 64])
nn_Z = MLP(input_dim=input_dim, output_dim=dim_x, hidden_dims=[64, 64, 64])

# 5. Solve
# Executes the Deep Picard Iteration
solver = UncoupledFBSDESolver(equation, solver_cfg, train_cfg, nn_Y, nn_Z)
solution = solver.solve()

# 6. Access Results
print(f"Solution shapes: X={solution['X'].shape}, Y={solution['Y'].shape}")
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
├── equations/             # FBSDE problem definitions
│   ├── base.py            # Abstract base class
│   └── benchmarks.py      # Standard test problems
├── solvers/               # Numerical solvers
│   ├── uncoupled.py       # Deep Picard for uncoupled systems
│   ├── coupled.py         # Global iteration for coupled systems
│   └── mckean_vlasov.py   # Solver for mean-field systems
├── nets/                  # Neural network architectures
│   └── mlp.py             # Multi-layer perceptron
├── utils/                 # Utilities
│   └── visualizations.py  # Plotting functions
└── config.py              # Configuration dataclasses
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
2. **Network Architecture**: Start with 3-4 hidden layers of 40-64 units each
3. **Picard Iterations**: 5-10 iterations typically sufficient for convergence

## Examples

See the `notebooks/` directory for comprehensive demonstrations:
- `benchmark_demonstrations.ipynb`: Complete examples for all five benchmark problems with visualization and error analysis

## Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{husain2025fbsde,
  author  = {Bilal Saleh Husain},
  title   = {A Deep Iterative Method for Coupled and McKean-Vlasov FBSDEs},
  school  = {University of Toronto},
  year    = {2025}
}
```

## Author

**Bilal Saleh Husain**
Master of Mathematics,
University of Toronto, 2025

---

**Version**: 1.0.0

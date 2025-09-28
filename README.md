# Deep Iterative Method for High-Dimensional FBSDEs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation for the Master's thesis, *"Deep Learning for High-Dimensional Forward-Backward Stochastic Differential Equations"*. This repository provides a unified framework for solving uncoupled, coupled, and McKean-Vlasov FBSDEs.

## Features

*   Implementation of the **Deep Iterative Method (DIM)** for the full FBSDE hierarchy.
*   Solvers for **Uncoupled**, **Coupled**, and **McKean-Vlasov** systems.
*   Includes both **gradient-based** and **regression-based** methods for the martingale component ($Z_t$).
*   Benchmarking against the **Deep Galerkin Method (DGM)** for uncoupled problems.
*   Modular code for SDE simulation, model I/O, and visualization.

## Reproducing Experiments

The Jupyter notebooks in the `/notebooks` directory contain the complete workflow for reproducing the results and figures from the thesis.

The core logic is implemented in the following modules:
*   `SDE_utils.py`: Contains the primary DIM solver classes and neural network architectures.
*   `DGM.py`: Implements the architecture for the Deep Galerkin Method.

## Citing This Work

If you use this code in your research, please cite the thesis:
```bibtex
@mastersthesis{Husain2025,
  author  = {Bilal Saleh Husain},
  title   = {Deep Learning for High-Dimensional Forward-Backward Stochastic Differential Equations},
  school  = {University of Toronto},
  year    = {2025}
}

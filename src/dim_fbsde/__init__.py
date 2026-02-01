"""
Deep Iterative Method for solving Forward-Backward Stochastic Differential Equations.

This package implements the Deep Picard Iteration method for solving high-dimensional
FBSDEs including uncoupled, coupled, and McKean-Vlasov systems.
"""

import logging

__version__ = "1.0.0"
__author__ = "Bilal Saleh Husain"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Expose main components
from dim_fbsde import equations, solvers, nets, utils, config

__all__ = ['equations', 'solvers', 'nets', 'utils', 'config', '__version__']

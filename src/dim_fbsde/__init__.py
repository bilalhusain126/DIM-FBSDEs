"""
Deep Iterative Method for solving Forward-Backward Stochastic Differential Equations.

This package implements the Deep Picard Iteration method for solving high-dimensional
FBSDEs including uncoupled, coupled, and McKean-Vlasov systems.
"""

import logging
import sys

__version__ = "1.0.0"
__author__ = "Bilal Saleh Husain"

# Auto-configure logging for notebooks and demos
# Users can reconfigure by calling setup_logging() with custom parameters
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
    force=True
)


def setup_logging(level=logging.INFO, format_string=None, use_stdout=True):
    """
    Reconfigure logging with custom parameters.

    Logging is auto-configured on import, so this is only needed for customization.

    Args:
        level: Logging level (default: logging.INFO)
        format_string: Custom format string (default: timestamp - module - message)
        use_stdout: Use stdout instead of stderr for Jupyter compatibility (default: True)

    Example:
        >>> import dim_fbsde
        >>> dim_fbsde.setup_logging(level=logging.DEBUG)  # More verbose
        >>> dim_fbsde.setup_logging(level=logging.WARNING)  # Less verbose
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(message)s'

    # Remove existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    logging.basicConfig(
        level=level,
        format=format_string,
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stdout if use_stdout else sys.stderr,
        force=True
    )

# Expose main components
from dim_fbsde import equations, solvers, nets, utils, config

__all__ = ['equations', 'solvers', 'nets', 'utils', 'config', 'setup_logging', '__version__']

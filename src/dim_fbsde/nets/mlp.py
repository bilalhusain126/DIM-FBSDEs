"""
Multi-Layer Perceptron (MLP) architecture.
"""

import torch
import torch.nn as nn
from typing import List, Union

class MLP(nn.Module):
    """
    A standard Multi-Layer Perceptron (Feed-Forward Neural Network).
    
    The network is constructed dynamically from a list of hidden layer dimensions.
    It expects a concatenated input tensor (e.g., time + state) and produces a 
    single output tensor.
    """

    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 hidden_dims: List[int], 
                 activation: str = 'SiLU'):
        """
        Initializes the MLP.

        Args:
            input_dim (int): Dimension of the input layer.
            output_dim (int): Dimension of the output layer.
            hidden_dims (List[int]): A list containing the number of neurons in each hidden layer.
            activation (str): The name of the activation function to use. 
                              Options: 'SiLU', 'ReLU', 'Tanh', 'Sigmoid'.
                              Defaults to 'SiLU'.
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation_name = activation
        
        # Resolve activation function
        self.activation_fn = self._get_activation(activation)

        layers = []
        
        # Input layer
        if len(hidden_dims) > 0:
            layers.append(nn.Linear(self.input_dim, self.hidden_dims[0]))
            layers.append(self.activation_fn)
            
            # Hidden layers
            for i in range(len(self.hidden_dims) - 1):
                layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]))
                layers.append(self.activation_fn)
            
            # Output layer
            layers.append(nn.Linear(self.hidden_dims[-1], self.output_dim))
        else:
            # Linear model case (no hidden layers)
            layers.append(nn.Linear(self.input_dim, self.output_dim))
        
        self.model = nn.Sequential(*layers)

    def _get_activation(self, name: str) -> nn.Module:
        """Factory method for activation functions."""
        activations = {
            'SiLU': nn.SiLU(),
            'ReLU': nn.ReLU(),
            'Tanh': nn.Tanh(),
            'Sigmoid': nn.Sigmoid(),
            'GELU': nn.GELU()
        }
        if name not in activations:
            raise NotImplementedError(f"Activation '{name}' is not supported. Choose from {list(activations.keys())}")
        return activations[name]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (Tensor): Input tensor of shape [Batch, input_dim].
                        
        Returns:
            Tensor: Output tensor of shape [Batch, output_dim].
        """
        return self.model(x)

    def __repr__(self):
        return (f"MLP(in={self.input_dim}, out={self.output_dim}, "
                f"hidden={self.hidden_dims}, act={self.activation_name})")
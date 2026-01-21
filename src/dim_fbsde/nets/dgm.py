import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMLayer(nn.Module):
    
    def __init__(self, output_dim, input_dim, trans1="tanh", trans2="tanh"):
        """
        Custom LSTM-like layer for DGM.

        Args:
            input_dim (int):  Dimensionality of input data.
            output_dim (int): Number of outputs for LSTM layers.
            trans1, trans2 (str): Activation functions for gates and transformations.
                                  Options: "tanh" (default), "relu", "sigmoid".
        """
        super(LSTMLayer, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim

        # Activation functions
        self.trans1 = self._get_activation(trans1)
        self.trans2 = self._get_activation(trans2)

        # Weight matrices (U and W)
        self.Uz = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.Ug = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.Ur = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.Uh = nn.Parameter(torch.Tensor(input_dim, output_dim))

        self.Wz = nn.Parameter(torch.Tensor(output_dim, output_dim))
        self.Wg = nn.Parameter(torch.Tensor(output_dim, output_dim))
        self.Wr = nn.Parameter(torch.Tensor(output_dim, output_dim))
        self.Wh = nn.Parameter(torch.Tensor(output_dim, output_dim))

        # Biases
        self.bz = nn.Parameter(torch.zeros(output_dim))
        self.bg = nn.Parameter(torch.zeros(output_dim))
        self.br = nn.Parameter(torch.zeros(output_dim))
        self.bh = nn.Parameter(torch.zeros(output_dim))

        # Initialize weights
        self.reset_parameters()

    
    def _get_activation(self, activation_type):
        """Helper function to get activation function."""
        activations = {
            "tanh": torch.tanh,
            "relu": F.relu,
            "sigmoid": torch.sigmoid,
        }
        return activations.get(activation_type, torch.tanh)

    
    def reset_parameters(self):
        """Initialize weights using Xavier initialization."""
        for weight in [self.Uz, self.Ug, self.Ur, self.Uh, self.Wz, self.Wg, self.Wr, self.Wh]:
            nn.init.xavier_uniform_(weight)

    
    def forward(self, S, X):
        """
        Forward pass for the LSTM layer.

        Args:
            S (torch.Tensor): Hidden state from the previous layer.
            X (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Updated hidden state.
        """
        Z = self.trans1(X @ self.Uz + S @ self.Wz + self.bz)
        G = self.trans1(X @ self.Ug + S @ self.Wg + self.bg)
        R = self.trans1(X @ self.Ur + S @ self.Wr + self.br)
        H = self.trans2(X @ self.Uh + (S * R) @ self.Wh + self.bh)

        # Update hidden state
        S_new = (1 - G) * H + Z * S
        return S_new



class DenseLayer(nn.Module):
    
    def __init__(self, output_dim, input_dim, transformation=None):
        """
        Custom dense layer for DGM.

        Args:
            input_dim (int):      Dimensionality of input data.
            output_dim (int):     Number of outputs for the dense layer.
            transformation (str): Activation function. Options: "tanh", "relu", None.
        """
        super(DenseLayer, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim

        # Weight matrix and bias
        self.W = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.b = nn.Parameter(torch.zeros(output_dim))

        # Activation function
        self.transformation = self._get_activation(transformation)

        # Initialize weights
        self.reset_parameters()

    
    def _get_activation(self, activation_type):
        """Helper function to get activation function."""
        activations = {
            "tanh": torch.tanh,
            "relu": F.relu,
        }
        return activations.get(activation_type, None)

    
    def reset_parameters(self):
        """Initialize weights using Xavier initialization."""
        nn.init.xavier_uniform_(self.W)

    
    def forward(self, X):
        """
        Forward pass for the dense layer.

        Args:
            X (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        S = X @ self.W + self.b
        if self.transformation:
            S = self.transformation(S)
        return S




class DGMNet(nn.Module):
    def __init__(self, layer_width, n_layers, input_dim, final_trans=None):
        """
        Deep Galerkin Method Neural Network.

        Args:
            layer_width (int): Number of units per layer.
            n_layers (int):    Number of LSTM layers.
            input_dim (int):   Spatial dimension of input data (excluding time).
            final_trans (str): Transformation used in the final layer.
        """
        super(DGMNet, self).__init__()

        # Initial dense layer
        self.initial_layer = DenseLayer(layer_width, input_dim + 1, transformation="tanh")

        # LSTM layers
        self.LSTMLayerList = nn.ModuleList(
            [LSTMLayer(layer_width, input_dim + 1) for _ in range(n_layers)]
        )

        # Final output layer
        self.final_layer = DenseLayer(1, layer_width, transformation=final_trans)

    
    def forward(self, t, x):
        """
        Forward pass for the DGM model.

        Args:
            t (torch.Tensor): Time input tensor.
            x (torch.Tensor): Spatial input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Combine time and space inputs
        X = torch.cat([t, x], dim=1)

        # Pass through initial layer
        S = self.initial_layer(X)

        # Pass through LSTM layers
        for lstm_layer in self.LSTMLayerList:
            S = lstm_layer(S, X)

        # Final output
        result = self.final_layer(S)
        return result

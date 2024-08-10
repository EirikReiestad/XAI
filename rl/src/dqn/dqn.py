import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DQN(nn.Module):
    """Deep Q-Network (DQN) model."""

    def __init__(
        self, n_observations: tuple, n_actions: int, hidden_layers: list[int] = [128]
    ) -> None:
        """Initialize the DQN model.

        Args:
            n_observations (tuple): Shape of the input observation.
            n_actions (int): Number of possible actions.
            hidden_layers (list[int]): List of hidden layer sizes.
        """
        super(DQN, self).__init__()
        input_size = int(np.prod(n_observations))
        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, hidden_layers[0]))

        # Hidden layers
        for in_size, out_size in zip(hidden_layers[:-1], hidden_layers[1:]):
            layers.append(nn.Linear(in_size, out_size))

        # Output layer
        layers.append(nn.Linear(hidden_layers[-1], n_actions))

        # Register layers as attributes for forward method
        self.input_layer = layers[0]
        self.hidden_layers = nn.ModuleList(layers[1:-1])
        self.output_layer = layers[-1]

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the network.

        Args:
            x (nn.Tensor): Input tensor.

        Returns:
            nn.Tensor: Output tensor representing Q-values for each action.
        """
        x = F.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)
        return x

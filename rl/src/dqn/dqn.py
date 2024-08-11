import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DQN(nn.Module):
    """Deep Q-Network (DQN) model."""

    def __init__(
        self,
        n_observations: tuple,
        n_actions: int,
        hidden_layers: list[int] = [128, 128],
    ) -> None:
        """Initialize the DQN model.

        Args:
            n_observations (tuple): Shape of the input observation.
            n_actions (int): Number of possible actions.
            hidden_layers (list[int]): List of hidden layer sizes.
        """
        super(DQN, self).__init__()
        input_size = int(np.prod(n_observations))

        self.layers = self._build_network(input_size, hidden_layers, n_actions)

    def _build_network(
        self, input_size: int, hidden_layers: list[int], output_size: int
    ) -> nn.ModuleList:
        """Build the neural network.
        Args:
            input_size (int): Size of the input layer.
            hidden_layers (list[int]): List of hidden layer sizes.
            n_actions (int): Number of possible actions.
        Returns:
            nn.ModuleList: List of layers in the network.
        """
        layers = []

        if len(hidden_layers) > 0:
            layers.append(nn.Linear(input_size, hidden_layers[0]))
            for in_size, out_size in zip(hidden_layers[:-1], hidden_layers[1:]):
                layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.Linear(hidden_layers[-1], output_size))
        else:
            layers.append(nn.Linear(input_size, output_size))
        return nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the network.

        Args:
            x (nn.Tensor): Input tensor.

        Returns:
            nn.Tensor: Output tensor representing Q-values for each action.
        """
        x = x.flatten(start_dim=1)
        for layer in self.layers:
            x = F.relu(layer(x))
        return x

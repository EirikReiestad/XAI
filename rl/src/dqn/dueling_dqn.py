import torch
from torch import nn
import numpy as np


class DuelingDQN(nn.Module):
    def __init__(self, input_shape: tuple, n_actions: int, hidden_layers: list):
        """ Initialize the Dueling DQN module.

        Parameters:
            input_shape (tuple): The shape of the input tensor (channels, height, width)
            n_actions (int): The number of actions in the environment
            hidden_layers (list): The sizes of the hidden layers
        """
        super(DuelingDQN, self).__init__()

        input_channels = input_shape[1]

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Calculate the size of the feature map after the conv layers
        conv_output_size = self._get_conv_output(input_shape)

        # Define the common feature layers
        layers = []
        input_dim = conv_output_size
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        self.feature = nn.Sequential(*layers)

        # Define the value stream layers
        value_layers = []
        input_dim = hidden_layers[-1]  # Last hidden layer size
        for hidden_dim in hidden_layers:
            value_layers.append(nn.Linear(input_dim, hidden_dim))
            value_layers.append(nn.ReLU())
            input_dim = hidden_dim
        # Output layer for value stream
        value_layers.append(nn.Linear(hidden_layers[-1], 1))
        self.value_stream = nn.Sequential(*value_layers)

        # Define the advantage stream layers
        advantage_layers = []
        input_dim = hidden_layers[-1]  # Last hidden layer size
        for hidden_dim in hidden_layers:
            advantage_layers.append(nn.Linear(input_dim, hidden_dim))
            advantage_layers.append(nn.ReLU())
            input_dim = hidden_dim
        # Output layer for advantage stream
        advantage_layers.append(nn.Linear(hidden_layers[-1], n_actions))
        self.advantage_stream = nn.Sequential(*advantage_layers)

    def forward(self, x: torch.tensor):
        # Ensure the shape is (batch, channels, height, width)
        if x.dim() != 4:  # Input is already in (batch_size, channels, height, width) format
            raise ValueError(f"Unexpected input shape: {x.shape}")

        x = self.conv(x)  # Apply the convolutional layers
        x = x.reshape(x.size(0), -1)  # Flatten the output from the conv layers
        x = self.feature(x)  # Apply the feature layers
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

    def _get_conv_output(self, shape: tuple[int, int, int]):
        with torch.no_grad():
            dummy = torch.zeros(shape)
            o = self.conv(dummy)
        return int(np.prod(o.size()))

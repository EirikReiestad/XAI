import logging

import torch
from torch import nn

from rl.src.common import ConvLayer
from rl import settings


class DQN(nn.Module):
    """Dueling Deep Q-Network (DQN) with convolutional and fully connected layers."""

    def __init__(
        self,
        input_shape: tuple,
        n_actions: int,
        hidden_layers: list[int],
        conv_layers: list[ConvLayer] | None = None,
        dueling: bool = False,
    ) -> None:
        """Initialize the Dueling DQN module.

        Args:
            input_shape (tuple[int, int, int]): Shape of the input tensor (channels, height, width).
            n_actions (int): Number of actions in the environment.
            hidden_layers (list[int]): Sizes of hidden layers in the fully connected network.
            conv_layers (list[ConvLayer] | None): List of convolutional layer configurations.
        """
        super(DQN, self).__init__()
        if len(input_shape) == 2:
            if conv_layers is not None and len(conv_layers) != 0:
                logging.info(
                    "Then tensor have a shape of 2. Consider dropping the convolutional network."
                )
            input_shape = (1, 1, *input_shape)
        elif len(input_shape) == 3:
            input_shape = (1, *input_shape)
        elif len(input_shape) != 4:
            raise ValueError(
                f"Invalid input shape: {input_shape}. Expected 2, 3, or 4 dimensions."
            )

        self._input_shape = input_shape
        self._n_actions = n_actions

        self.dueling = dueling

        # Convolutional layers
        self.conv = (
            self._build_conv_layers(conv_layers, input_shape[1])
            if conv_layers
            else nn.Sequential()
        )

        # Determine the output size after convolutional layers
        conv_output_size = self._get_conv_output(input_shape)

        # Fully connected feature layers
        if self.dueling:
            self.fc_feature = self._build_fc_layers(hidden_layers, conv_output_size)
            # TODO: Consider having separate hidden_layers for value and advantage streams than the hidden_layers used for the feature layers
            # Value stream layers
            self.value_stream = self._build_dueling_stream(hidden_layers, output_dim=1)
            # Advantage stream layers
            self.advantage_stream = self._build_dueling_stream(
                hidden_layers, output_dim=n_actions
            )
        else:
            self.fc_feature = self._build_fc_layers(
                hidden_layers, conv_output_size, n_actions
            )

    @property
    def input_shape(self) -> tuple:
        """Shape of the input tensor."""
        return self._input_shape

    @property
    def n_actions(self) -> int:
        """Number of actions in the environment."""
        return self._n_actions

    def _build_conv_layers(
        self, conv_layers: list[ConvLayer], input_channels: int
    ) -> nn.Sequential:
        """Build the convolutional layers.

        Args:
            conv_layers (list[ConvLayer]): List of ConvLayer configurations.
            input_channels (int): Number of input channels.

        Returns:
            nn.Sequential: A sequential container of convolutional layers and activations.
        """
        layers = []
        for layer in conv_layers:
            layers.append(layer.build(input_channels))
            layers.append(layer.build_activation())
            input_channels = layer.filters
        return nn.Sequential(*layers)

    def _build_fc_layers(
        self, hidden_layers: list[int], input_dim: int, output_dim: int | None = None
    ) -> nn.Sequential:
        """Build the fully connected feature layers.

        Args:
            hidden_layers (list[int]): Sizes of hidden layers.
            input_dim (int): Dimension of the input to the first hidden layer.

        Returns:
            nn.Sequential: A sequential container of fully connected layers and activations.
        """
        layers = []
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        # layers.append(nn.Dropout(settings.DROPOUT_RATE))
        if output_dim is not None:
            layers.append(nn.Linear(hidden_layers[-1], output_dim))
        return nn.Sequential(*layers)

    def _build_dueling_stream(
        self, hidden_layers: list[int], output_dim: int
    ) -> nn.Sequential:
        """Build a stream for the dueling architecture (value or advantage).

        Args:
            hidden_layers (list[int]): Sizes of hidden layers.
            output_dim (int): Size of the output layer.

        Returns:
            nn.Sequential: A sequential container of fully connected layers and activations.
        """
        layers = []
        input_dim = hidden_layers[-1]
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Dropout(settings.DROPOUT_RATE))
        layers.append(nn.Linear(hidden_layers[-1], output_dim))
        return nn.Sequential(*layers)

    def _get_conv_output(self, shape: tuple) -> int:
        """Calculate the output size after convolutional layers.

        Args:
            shape (tuple[int, int, int]): Shape of the input tensor (channels, height, width).

        Returns:
            int: Size of the flattened output tensor.
        """
        with torch.no_grad():
            dummy_input = torch.zeros(*shape)
            output = self.conv(dummy_input)
        return output.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The output tensor with Q-values.
        """
        match len(x.shape):
            case 2:
                x = x.unsqueeze(1).unsqueeze(1)
            case 3:
                x = x.unsqueeze(1)
            case 4:
                pass
            case _:
                raise ValueError(f"Invalid input tensor shape: {x.shape}")

        x = self.conv(x)  # Apply convolutional layers
        x = x.flatten(start_dim=1)  # Flatten the output from conv layers
        x = self.fc_feature(x)  # Apply feature layers

        if self.dueling:
            value = self.value_stream(x)
            advantage = self.advantage_stream(x)
            return value + (advantage - advantage.mean())
        return x

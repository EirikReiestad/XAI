import logging

import torch
from torch import nn

from rl.src.common import ConvLayer


class QNetwork(nn.Module):
    """Dueling Deep Q-Network (DQN) with convolutional and fully connected layers."""

    def __init__(
        self,
        input_shape: tuple,
        n_actions: int,
        hidden_layers: list[int],
        conv_layers: list[ConvLayer] | None = None,
        dueling: bool = False,
    ) -> None:
        super(QNetwork, self).__init__()
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

        self.conv = (
            self._build_conv_layers(conv_layers, input_shape[1])
            if conv_layers
            else nn.Sequential()
        )
        conv_output_size = self._get_conv_output(input_shape)

        if self.dueling:
            self.fc_feature = self._build_fc_layers(hidden_layers, conv_output_size)
            # TODO: Consider having separate hidden_layers for value and advantage streams than the hidden_layers used for the feature layers
            # Value stream layers
            input_dim = hidden_layers[-1]
            self.value_stream = self._build_fc_layers(hidden_layers, input_dim, 1)
            self.advantage_stream = self._build_fc_layers(
                hidden_layers, input_dim, n_actions
            )
        else:
            self.fc_feature = self._build_fc_layers(
                hidden_layers, conv_output_size, n_actions
            )

    def _build_conv_layers(
        self, conv_layers: list[ConvLayer], input_channels: int
    ) -> nn.Sequential:
        layers = []
        for layer in conv_layers:
            layers.append(layer.build(input_channels))
            layers.append(layer.build_activation())
            input_channels = layer.filters
        return nn.Sequential(*layers)

    def _build_fc_layers(
        self, hidden_layers: list[int], input_dim: int, output_dim: int | None = None
    ) -> nn.Sequential:
        layers = []
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        if output_dim is not None:
            layers.append(nn.Linear(hidden_layers[-1], output_dim))
        return nn.Sequential(*layers)

    def _get_conv_output(self, shape: tuple) -> int:
        with torch.no_grad():
            dummy_input = torch.zeros(*shape)
            output = self.conv(dummy_input)
        return output.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    @property
    def input_shape(self) -> tuple:
        """Shape of the input tensor."""
        return self._input_shape

    @property
    def n_actions(self) -> int:
        """Number of actions in the environment."""
        return self._n_actions

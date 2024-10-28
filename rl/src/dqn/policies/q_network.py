import logging

import numpy as np
import torch
from gymnasium import spaces
from torch import nn

from rl.src.common.policies import BasePolicy


class QNetwork(BasePolicy):
    logged = False

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        hidden_layers: list[int],
        conv_layers: list[int],
        dueling: bool = False,
    ) -> None:
        super(QNetwork, self).__init__(observation_space, action_space)
        self.dueling = dueling
        observation_size = self._observation_size(observation_space)
        number_of_actions = action_space.n

        self.conv_layers = conv_layers

        if self.dueling:
            self.conv_feature = self._build_conv_layers(conv_layers, observation_size)
            conv_out_size = self._conv_layer_output_size(observation_size)
            self.fc_feature = self._build_fc_layers(hidden_layers, conv_out_size)
            # TODO: Consider having separate hidden_layers for value and advantage streams than the hidden_layers used for the feature layers
            # Value stream layers
            input_dim = hidden_layers[-1]
            self.value_stream = self._build_fc_layers(hidden_layers, input_dim, 1)
            self.advantage_stream = self._build_fc_layers(
                hidden_layers, input_dim, number_of_actions
            )
        else:
            self.conv_feature = self._build_conv_layers(conv_layers, observation_size)
            conv_out_size = self._conv_layer_output_size(observation_size)
            self.fc_feature = self._build_fc_layers(
                hidden_layers, conv_out_size, number_of_actions
            )
        self.summary()

    def _build_conv_layers(
        self,
        conv_layers: list[int],
        input_dim: tuple[int, int] | tuple[int, int, int, int],
    ) -> nn.Sequential:
        if len(input_dim) == 2:
            return nn.Sequential()

        input_size = input_dim[1]
        layers = []
        for hidden_dim in conv_layers:
            layers.append(nn.Conv2d(input_size, hidden_dim, kernel_size=3, stride=1))
            layers.append(nn.ReLU())
            input_size = hidden_dim
        return nn.Sequential(*layers)

    def _conv_layer_output_size(
        self, input_dim: tuple[int, int] | tuple[int, int, int, int]
    ) -> int:
        if len(self.conv_layers) == 0:
            return int(np.prod(input_dim))
        device = next(self.parameters()).device
        if len(input_dim) == 2:
            return input_dim[1]
        with torch.no_grad():
            dummy_input = torch.zeros(*input_dim).to(device)
            output = self.conv_feature(dummy_input)
        return output.numel()

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        x = self._alter_input_shape(x)

        x = self.conv_feature(x)

        x = torch.flatten(x, start_dim=1)
        x = x.to(device)
        x = self.fc_feature(x)

        if self.dueling:
            value = self.value_stream(x)
            advantage = self.advantage_stream(x)
            return value + (advantage - advantage.mean())
        return x

    def _alter_input_shape(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        if len(x.shape) == 3:
            if isinstance(x, np.ndarray):
                return (
                    torch.tensor(x, dtype=torch.float32)
                    .to(device)
                    .unsqueeze(0)
                    .permute(1, 0, 2, 3)
                ).to(device)
            return x.unsqueeze(0).permute(1, 0, 2, 3).to(device)
        if isinstance(x, torch.Tensor):
            return x.to(device)
        return torch.tensor(x, dtype=torch.float32).to(device)

    def _observation_size(
        self, observation_space: spaces.Space
    ) -> tuple[int, int] | tuple[int, int, int, int]:
        if observation_space.shape is None or len(observation_space.shape) == 0:
            raise ValueError(
                f"Invalid input shape: {observation_space.shape}. Observation space must be at least 1D."
            )
        elif len(observation_space.shape) == 1:
            return (1, int(observation_space.shape[0]))
        elif len(observation_space.shape) == 2:
            return (1, 1, *observation_space.shape)
        raise ValueError(
            f"Invalid input shape: {observation_space.shape}. Observation space must be at least 1D."
        )

    def _number_of_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def summary(self) -> None:
        if QNetwork.logged:
            return
        QNetwork.logged = True

        logging.info(f"Number of parameters: {self._number_of_parameters()}")
        logging.info(self)

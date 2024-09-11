import logging

import torch
from gymnasium import spaces
from torch import nn

from rl.src.common.policies import BasePolicy


class DQNPolicy(BasePolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        hidden_layers: list[int],
        dueling: bool = False,
    ) -> None:
        super(DQNPolicy, self).__init__(observation_space, action_space)
        self.dueling = dueling

        if self.dueling:
            self.fc_feature = self._build_fc_layers(
                hidden_layers, observation_space.shape
            )
            # TODO: Consider having separate hidden_layers for value and advantage streams than the hidden_layers used for the feature layers
            # Value stream layers
            input_dim = hidden_layers[-1]
            self.value_stream = self._build_fc_layers(hidden_layers, input_dim, 1)
            self.advantage_stream = self._build_fc_layers(
                hidden_layers, input_dim, action_space.shape
            )
        else:
            self.fc_feature = self._build_fc_layers(
                hidden_layers, observation_space.shape, action_space.shape
            )

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
        match len(x.shape):
            case 2:
                x = x.unsqueeze(1).unsqueeze(1)
            case 3:
                x = x.unsqueeze(1)
            case 4:
                pass
            case _:
                raise ValueError(f"Invalid input tensor shape: {x.shape}")

        x = x.flatten(start_dim=1)  # Flatten the output from conv layers
        x = self.fc_feature(x)  # Apply feature layers

        if self.dueling:
            value = self.value_stream(x)
            advantage = self.advantage_stream(x)
            return value + (advantage - advantage.mean())
        return x

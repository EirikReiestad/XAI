import torch
from gymnasium import spaces
from torch import nn

from rl.src.common.policies import BasePolicy


class QNetwork(BasePolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        hidden_layers: list[int],
        dueling: bool = False,
    ) -> None:
        super(QNetwork, self).__init__(observation_space, action_space)
        self.dueling = dueling
        observation_size = self._observation_size(observation_space)
        number_of_actions = action_space.n

        if self.dueling:
            self.fc_feature = self._build_fc_layers(hidden_layers, observation_size)
            # TODO: Consider having separate hidden_layers for value and advantage streams than the hidden_layers used for the feature layers
            # Value stream layers
            input_dim = hidden_layers[-1]
            self.value_stream = self._build_fc_layers(hidden_layers, input_dim, 1)
            self.advantage_stream = self._build_fc_layers(
                hidden_layers, input_dim, number_of_actions
            )
        else:
            self.fc_feature = self._build_fc_layers(
                hidden_layers, observation_size, number_of_actions
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
        x = self.fc_feature(x)

        if self.dueling:
            value = self.value_stream(x)
            advantage = self.advantage_stream(x)
            return value + (advantage - advantage.mean())
        return x

    def _observation_size(self, observation_space: spaces.Space) -> int:
        if observation_space.shape is None:
            raise ValueError("Invalid input shape: None")
        if len(observation_space.shape) > 1:
            raise ValueError(
                f"Invalid input shape: {observation_space.shape}. DQN only supports 1D input shapes"
            )
        if len(observation_space.shape) == 0:
            raise ValueError(
                f"Invalid input shape: {observation_space.shape}. DQN only supports 1D input shapes"
            )
        return observation_space.shape[0]

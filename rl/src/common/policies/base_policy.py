import torch.nn as nn
import torch
from gymnasium import spaces
from abc import abstractmethod


class BasePolicy(nn.Module):
    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space):
        super(BasePolicy, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

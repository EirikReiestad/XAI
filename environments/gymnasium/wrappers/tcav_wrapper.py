import gymnasium as gym
import torch


class TCAVWrapper:
    def __init__(self, env: gym.Env) -> None:
        self.env = env

    def get_concept_inputs(self, concept: str, samples: int) -> torch.Tensor:
        return torch.rand(1, 1, 84, 84)

from abc import ABC, abstractmethod

import torch

from rl.src.managers import WandBConfig, WandBManager


class SingleAgentBase(ABC):
    def __init__(self, wandb: bool = False, wandb_config: WandBConfig = WandBConfig()):
        self.wandb_manager = WandBManager(wandb, wandb_config)

    @abstractmethod
    def learn(self, total_timesteps: int):
        raise NotImplementedError

    @abstractmethod
    def predict(self, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str):
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str):
        raise NotImplementedError

from abc import ABC, abstractmethod
from rl.src.managers import WandBConfig, WandBManager
import torch


class MultiAgentBase(ABC):
    def __init__(self, wandb: bool = False, wandb_config: WandBConfig = WandBConfig()):
        self.wandb_manager = WandBManager(wandb, wandb_config)

    @abstractmethod
    def learn(self, total_timesteps: int):
        raise NotImplementedError

    @abstractmethod
    def predict(self, state: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def save(self):
        raise NotImplementedError

    @abstractmethod
    def load(self):
        raise NotImplementedError

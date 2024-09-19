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
    def save(self, episode: int):
        raise NotImplementedError

    @abstractmethod
    def load(self, run_path: str, model_artifact: str, version_number: str):
        raise NotImplementedError

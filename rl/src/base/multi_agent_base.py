from abc import ABC, abstractmethod

import numpy as np
import torch

from rl.src.managers import WandBConfig, WandBManager


class MultiAgentBase(ABC):
    def __init__(
        self,
        wandb_active: bool = False,
        wandb_config: WandBConfig | None = None,
    ):
        self.wandb_manager = WandBManager(wandb_active, wandb_config)

    @property
    @abstractmethod
    def models(self) -> list:
        raise NotImplementedError

    @abstractmethod
    def learn(self, total_timesteps: int):
        raise NotImplementedError

    @abstractmethod
    def predict(self, state: torch.Tensor) -> list[np.ndarray | list[np.ndarray]]:
        raise NotImplementedError

    @abstractmethod
    def predict_actions(self, state: torch.Tensor) -> list[torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def save(self, episode: int):
        raise NotImplementedError

    @abstractmethod
    def load(
        self,
        run_id: str,
        model_artifact: str,
        version_numbers: list[str],
    ):
        raise NotImplementedError

    def close(self):
        self.wandb_manager.finish()

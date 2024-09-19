from abc import ABC, abstractmethod
from rl.src.managers import WandBConfig, WandBManager
import torch


class MultiAgentBase(ABC):
    def __init__(
        self,
        wandb: bool = False,
        wandb_config: WandBConfig | None = None,
    ):
        self.wandb_manager = WandBManager(wandb, wandb_config)

    @abstractmethod
    def learn(self, total_timesteps: int):
        raise NotImplementedError

    @abstractmethod
    def predict(self, state: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
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

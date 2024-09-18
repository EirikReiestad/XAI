import wandb
from dataclasses import dataclass
import logging


@dataclass
class WandBConfig:
    def __init__(self) -> None:
        self.project: str = ""
        self.run_name: str = ""
        self.tags: list[str] = []
        self.other: dict = {}


class WandBManager:
    def __init__(self, active: bool, config: WandBConfig):
        self.active = active
        if not active:
            return

        self.config = config
        wandb.init(
            project=config.project,
            name=config.run_name,
            config=config.other,
            reinit=True,
            tags=config.tags,
        )

    def log(self, data: dict):
        if not self.active:
            return
        wandb.log(data)

    def finish(self):
        if not self.active:
            return
        wandb.finish()

    def save(self, path: str):
        if not self.active:
            return
        wandb.save(path)

import wandb
from dataclasses import dataclass


@dataclass
class WandBConfig:
    project: str
    run_name: str
    tags: list[str]
    other: dict


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

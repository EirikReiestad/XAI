import wandb
from dataclasses import dataclass


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

    def save_model(self, path: str):
        if not self.active:
            return
        wandb.save(path)

    def save_file(self, path: str):
        if not self.active:
            return
        artifact = wandb.Artifact("model", type="model")
        artifact.add_file(path)
        wandb.log_artifact(artifact)

    def load_model(self, name: str):
        if not self.active:
            return
        artifact = wandb.use_artifact(name, type="model")
        artifact_dir = artifact.download()
        return artifact_dir

    def load_file(self, name: str):
        if not self.active:
            return
        artifact = wandb.use_artifact(name)
        artifact_dir = artifact.download()
        return artifact_dir
